import math

import torch
import torch.nn.functional as F

from ..kernels.block_importance_kernels import (
    flat_group_gemm_fuse_reshape,
    softmax_fuse_block_sum,
)
def pad_to_multiple(x, multiple):
    """
    在序列维度（dim=2）上填充 x，使其长度为 multiple 的倍数。
    x: [B, H, L, D]
    """
    L = x.size(2)
    remainder = L % multiple
    if remainder != 0:
        pad_len = multiple - remainder
        # 对序列维度在后面补 pad_len 个零（注意 F.pad 参数顺序：最后两个数字对应 dim=2 的左右补充）
        x = F.pad(x, (0, 0, 0, pad_len),mode='replicate')
    return x
def calc_k_similarity(k, block_size):
    SIM_2_THRESHOLD = 0.75
    SIM_4_THRESHOLD = 0.7
    SIM_8_THRESHOLD = 0.7
    k = pad_to_multiple(k, block_size)
    k_chunked_num = k.shape[-2] // block_size
    # 重塑为 [B, H, k_chunked_num, block_size, D]
    k_chunked = k[:, :, :k_chunked_num*block_size, :].reshape(
        k.shape[0], k.shape[1], k_chunked_num, block_size, k.shape[-1]
    )
    # 分离偶数和奇数位置的 token
    k_chunked_1 = k_chunked[..., ::2, :]   # [B, H, k_chunked_num, block_size//2, D]
    k_chunked_2 = k_chunked[..., 1::2, :]  # [B, H, k_chunked_num, block_size//2, D]
    # 在特征维度 (dim=-1) 上计算余弦相似度
    # cosine_similarity 内部会自动归一化，无需手动 norm
    similarity_2 = F.cosine_similarity(k_chunked_1, k_chunked_2, dim=-1).mean(dim=-1)  # [B, H, k_chunked_num]
    similarity_4 = F.cosine_similarity(
        k_chunked[...,0::4, :],
        k_chunked[...,3::4, :],
        dim=-1
    ).mean(dim=-1)  # [B, H, k_chunked_num]
    similarity_8 = F.cosine_similarity(
        k_chunked[...,0::8, :],
        k_chunked[...,7::8, :],
        dim=-1
    ).mean(dim=-1)  # [B, H, k_chunked_num]
    sim_2_mask = 2*(similarity_2> SIM_2_THRESHOLD)
    sim_4_mask = 4*(similarity_4> SIM_4_THRESHOLD)
    sim_8_mask = 8*(similarity_8> SIM_8_THRESHOLD)
    one_tensor = torch.ones_like(sim_2_mask)
    sim_mask = torch.maximum(one_tensor, torch.maximum(sim_2_mask, torch.maximum(sim_4_mask, sim_8_mask)))
    return sim_mask
import torch
import triton
import triton.language as tl


@triton.jit
def _calc_k_similarity_split_kernel(
    K_ptr,           # [B, H, N, D]
    Out_ptr,         # [B, H, N_blocks]
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_ob, stride_oh, stride_on,
    t2, t4, t8,
    N,               # total sequence length
    USER_BLOCK_SIZE: tl.constexpr,   # e.g. 32
    HEAD_DIM: tl.constexpr,          # e.g. 128
    TRITON_BLOCK: tl.constexpr,      # e.g. 128 (number of K vectors per triton program)
):
    """
    Fused kernel using split optimization pattern.

    Grid: (num_chunks, B * H)
    - Each program processes TRITON_BLOCK K vectors (e.g. 128)
    - This corresponds to NUM_SUB = TRITON_BLOCK // USER_BLOCK_SIZE user blocks (e.g. 4)

    Algorithm:
    1. Load even (0,2,4,...) and odd (1,3,5,...) K vectors - only 2 loads
    2. Precompute norms for even and odd
    3. Compute 2x similarity directly
    4. Use split to get 4x positions, compute 4x similarity
    5. Split again for 8x positions, compute 8x similarity
    6. Apply thresholds and store results
    """
    pid_chunk = tl.program_id(0)  # which chunk of TRITON_BLOCK vectors
    pid_bh = tl.program_id(1)     # which batch*head

    # Number of user blocks per triton block
    NUM_SUB: tl.constexpr = TRITON_BLOCK // USER_BLOCK_SIZE  # e.g. 4
    HALF_BS: tl.constexpr = USER_BLOCK_SIZE // 2             # e.g. 16

    # Base K pointer for this batch*head
    k_base = K_ptr + pid_bh * stride_kh

    # ========== 1. Compute load offsets ==========
    # offs_base: [0, USER_BLOCK_SIZE, 2*USER_BLOCK_SIZE, ...]
    offs_base = (pid_chunk * TRITON_BLOCK) + tl.arange(0, NUM_SUB) * USER_BLOCK_SIZE
    offs_base = offs_base[:, None]  # [NUM_SUB, 1]

    # r_half: [0, 1, 2, ..., HALF_BS-1]
    r_half = tl.arange(0, HALF_BS)[None, :]  # [1, HALF_BS]

    # even: 0, 2, 4, ...  odd: 1, 3, 5, ...
    offs_even = offs_base + r_half * 2       # [NUM_SUB, HALF_BS]
    offs_odd = offs_base + r_half * 2 + 1    # [NUM_SUB, HALF_BS]

    offs_d = tl.arange(0, HEAD_DIM)[None, None, :]  # [1, 1, HEAD_DIM]

    # ========== 2. Load (only two loads!) ==========
    # Clip offsets to valid range [0, N-1] to avoid mask load
    offs_even_clipped = tl.minimum(offs_even, N - 1)
    offs_odd_clipped = tl.minimum(offs_odd, N - 1)

    # ptr shape: [NUM_SUB, HALF_BS, HEAD_DIM]
    ptr_even = k_base + offs_even_clipped[:, :, None] * stride_kn + offs_d * stride_kd
    ptr_odd = k_base + offs_odd_clipped[:, :, None] * stride_kn + offs_d * stride_kd

    val_even = tl.load(ptr_even)  # [NUM_SUB, HALF_BS, HEAD_DIM]
    val_odd = tl.load(ptr_odd)    # [NUM_SUB, HALF_BS, HEAD_DIM]

    # ========== 3. Precompute norms ==========
    # [NUM_SUB, HALF_BS]
    norm_even = tl.sqrt(tl.sum(val_even * val_even, axis=2) + 1e-12)
    norm_odd = tl.sqrt(tl.sum(val_odd * val_odd, axis=2) + 1e-12)

    # ========== 4. Sim 2x: directly compute ==========
    dot_2 = tl.sum(val_even * val_odd, axis=2)  # [NUM_SUB, HALF_BS]
    cos_2 = dot_2 / (norm_even * norm_odd + 1e-6)
    sim_2 = tl.sum(cos_2, axis=1) / HALF_BS  # [NUM_SUB]

    # ========== 5. Sim 4x: use split ==========
    # val_even contains indices 0,2,4,6,... We need to split into:
    #   - k_0: indices 0,4,8,12,... (even positions in val_even)
    #   - k_2: indices 2,6,10,14,... (odd positions in val_even)
    # val_odd contains indices 1,3,5,7,... We need to split into:
    #   - k_1: indices 1,5,9,13,...
    #   - k_3: indices 3,7,11,15,...

    QUARTER_BS: tl.constexpr = USER_BLOCK_SIZE // 4  # e.g. 8

    # Reshape val_even: [NUM_SUB, HALF_BS, HEAD_DIM] -> [NUM_SUB, QUARTER_BS, 2, HEAD_DIM]
    val_even_re = tl.reshape(val_even, (NUM_SUB, QUARTER_BS, 2, HEAD_DIM))
    # Permute to: [NUM_SUB, QUARTER_BS, HEAD_DIM, 2]
    val_even_perm = tl.permute(val_even_re, (0, 1, 3, 2))
    # Split along last dim (size 2)
    k_0, k_2 = tl.split(val_even_perm)  # each: [NUM_SUB, QUARTER_BS, HEAD_DIM]

    # Same for val_odd
    val_odd_re = tl.reshape(val_odd, (NUM_SUB, QUARTER_BS, 2, HEAD_DIM))
    val_odd_perm = tl.permute(val_odd_re, (0, 1, 3, 2))
    k_1, k_3 = tl.split(val_odd_perm)  # each: [NUM_SUB, QUARTER_BS, HEAD_DIM]

    # Reshape norms: [NUM_SUB, HALF_BS] -> [NUM_SUB, QUARTER_BS, 2]
    norm_even_re = tl.reshape(norm_even, (NUM_SUB, QUARTER_BS, 2))
    n_0, n_2 = tl.split(norm_even_re)  # each: [NUM_SUB, QUARTER_BS]

    norm_odd_re = tl.reshape(norm_odd, (NUM_SUB, QUARTER_BS, 2))
    n_1, n_3 = tl.split(norm_odd_re)  # each: [NUM_SUB, QUARTER_BS]

    # Compute Sim 4x: k_0 (0, 4, ...) vs k_3 (3, 7, ...)
    dot_4 = tl.sum(k_0 * k_3, axis=2)  # [NUM_SUB, QUARTER_BS]
    cos_4 = dot_4 / (n_0 * n_3 + 1e-6)
    sim_4 = tl.sum(cos_4, axis=1) / QUARTER_BS  # [NUM_SUB]

    # ========== 6. Sim 8x: split again ==========
    # From k_0 (0,4,8,12,...), split into:
    #   - k_00: 0,8,16,...
    #   - k_04: 4,12,20,...
    # From k_3 (3,7,11,15,...), split into:
    #   - k_31: 3,11,19,...
    #   - k_33: 7,15,23,...

    EIGHTH_BS: tl.constexpr = USER_BLOCK_SIZE // 8  # e.g. 4

    # k_0: [NUM_SUB, QUARTER_BS, HEAD_DIM] -> [NUM_SUB, EIGHTH_BS, 2, HEAD_DIM]
    k_0_re = tl.reshape(k_0, (NUM_SUB, EIGHTH_BS, 2, HEAD_DIM))
    k_0_perm = tl.permute(k_0_re, (0, 1, 3, 2))
    k_00, k_04 = tl.split(k_0_perm)  # each: [NUM_SUB, EIGHTH_BS, HEAD_DIM]

    # k_3: [NUM_SUB, QUARTER_BS, HEAD_DIM] -> [NUM_SUB, EIGHTH_BS, 2, HEAD_DIM]
    k_3_re = tl.reshape(k_3, (NUM_SUB, EIGHTH_BS, 2, HEAD_DIM))
    k_3_perm = tl.permute(k_3_re, (0, 1, 3, 2))
    k_31, k_33 = tl.split(k_3_perm)  # each: [NUM_SUB, EIGHTH_BS, HEAD_DIM]

    # Norms
    n_0_re = tl.reshape(n_0, (NUM_SUB, EIGHTH_BS, 2))
    n_00, n_04 = tl.split(n_0_re)  # each: [NUM_SUB, EIGHTH_BS]

    n_3_re = tl.reshape(n_3, (NUM_SUB, EIGHTH_BS, 2))
    n_31, n_33 = tl.split(n_3_re)  # each: [NUM_SUB, EIGHTH_BS]

    # Compute Sim 8x: k_00 (0, 8, ...) vs k_33 (7, 15, ...)
    dot_8 = tl.sum(k_00 * k_33, axis=2)  # [NUM_SUB, EIGHTH_BS]
    cos_8 = dot_8 / (n_00 * n_33 + 1e-6)
    sim_8 = tl.sum(cos_8, axis=1) / EIGHTH_BS  # [NUM_SUB]

    # ========== 7. Apply thresholds and compute final score ==========
    final_score = tl.full((NUM_SUB,), 1.0, dtype=tl.float32)
    final_score = tl.where(sim_2 > t2, 2.0, final_score)
    final_score = tl.where(sim_4 > t4, 4.0, final_score)
    final_score = tl.where(sim_8 > t8, 8.0, final_score)

    # ========== 8. Store results ==========
    out_idx = pid_chunk * NUM_SUB + tl.arange(0, NUM_SUB)
    num_blocks = N // USER_BLOCK_SIZE
    out_mask = out_idx < num_blocks

    out_ptr = Out_ptr + pid_bh * stride_oh + out_idx * stride_on
    tl.store(out_ptr, final_score.to(tl.int32), mask=out_mask)


def calc_k_similarity_triton(k: torch.Tensor, blocksize: int, config) -> torch.Tensor:
    """
    Triton implementation of K similarity calculation using split optimization.

    This kernel computes cosine similarity between K vectors at different strides
    to determine the optimal pooling level for each block:
    - 2x: similarity between adjacent pairs (0,1), (2,3), ...
    - 4x: similarity between (0,3), (4,7), ...
    - 8x: similarity between (0,7), (8,15), ...

    Args:
        k: Key tensor of shape [B, H, L, D]
        blocksize: Block size (must be divisible by 8)
        config: Configuration object with sim_2x_threshold, sim_4x_threshold, sim_8x_threshold

    Returns:
        Similarity mask of shape [B, H, num_blocks] with values in {1, 2, 4, 8}
    """
    assert blocksize % 8 == 0, "blocksize must be divisible by 8"
    assert blocksize >= 8, "blocksize must be at least 8"

    B, H, L, D = k.shape
    num_blocks = (L + blocksize - 1) // blocksize

    # Ensure contiguous float32
    k = k.contiguous()
    if k.dtype != torch.float32:
        k = k.to(torch.float32)

    # Output tensor
    out = torch.empty((B, H, num_blocks), device=k.device, dtype=torch.int32)

    # Thresholds
    t2 = float(getattr(config, 'sim_2x_threshold', 0.0))
    t4 = float(getattr(config, 'sim_4x_threshold', 0.0))
    t8 = float(getattr(config, 'sim_8x_threshold', -1.0))

    # Triton block size (how many K vectors per program)
    # Must be multiple of blocksize
    TRITON_BLOCK = max(128, blocksize)
    if TRITON_BLOCK % blocksize != 0:
        TRITON_BLOCK = ((TRITON_BLOCK + blocksize - 1) // blocksize) * blocksize

    # Number of chunks needed
    N_effective = num_blocks * blocksize  # Only process complete blocks
    num_chunks = (N_effective + TRITON_BLOCK - 1) // TRITON_BLOCK

    # Grid: (num_chunks, B * H)
    grid = (num_chunks, B * H)

    # Strides
    stride_kb, stride_kh, stride_kn, stride_kd = k.stride()
    stride_ob, stride_oh, stride_on = out.stride()

    _calc_k_similarity_split_kernel[grid](
        k, out,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_ob, stride_oh, stride_on,
        t2, t4, t8,
        N_effective,
        USER_BLOCK_SIZE=blocksize,
        HEAD_DIM=D,
        TRITON_BLOCK=TRITON_BLOCK,
    )

    return out



def estimate_block_importance(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size: int,
    stride: int,
    *,
    norm: float = 1.0,
    softmax: bool = True,
    chunk_size: int = 16384,
    select_mode: str = "inverse",
    use_triton: bool = True,
    causal: bool = True,
    kdb: int = 1,
) -> torch.Tensor:
    """Estimate block-wise attention importance via antidiagonal sampling."""

    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    _, num_q_head, q_len, _ = query_states.shape
    assert num_q_head == num_kv_head

    target_device = key_states.device

    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size
    assert k_chunk_num >= q_chunk_num
    offset_token_chunk_num = k_chunk_num - q_chunk_num

    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0)
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0)
    else:
        pad_query_states = query_states

    pad_key_states = pad_key_states.to(target_device)
    pad_query_states = pad_query_states.to(target_device)

    attn_sum_list = []

    if use_triton:
        if not torch.cuda.is_available():
            use_triton = False
        else:
            device_name = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).name
            if "100" not in device_name and  "200" not in device_name and "6000" not in device_name:
                use_triton = False
                print(
                    "setting use triton to false. Triton kernel not surpported on this device"
                )
            else:
                pad_key_states = pad_key_states.to("cuda")
                pad_query_states = pad_query_states.to("cuda")
                target_device = pad_key_states.device
    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size

    if not use_triton:
        if select_mode == "random":
            perm_idx = torch.randperm(stride)
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[:, :, perm_idx[i] :: stride, :]
                    for i in range(stride)
                ],
                dim=-1,
            )
        elif select_mode in {"inverse", ""}:
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[:, :, (stride - 1 - q) :: (stride * kdb), :]
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "slash":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [(pad_query_states[:, :, q::stride, :]) for q in range(stride)], dim=-1
            )
        elif select_mode == "double":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [
                    reshaped_key[:, :, :, head_dim:],
                    reshaped_key[:, :, :, 0:head_dim],
                ],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[:, :, (stride - 1 - q) :: stride, :]
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "triple":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [
                    reshaped_key[:, :, :, head_dim:],
                    reshaped_key[:, :, :, 0:head_dim],
                ],
                dim=-1,
            )
            reshaped_key = reshaped_key + torch.cat(
                [
                    reshaped_key[:, :, :, -head_dim:],
                    reshaped_key[:, :, :, 0:-head_dim],
                ],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[:, :, (stride - 1 - q) :: stride, :]
                    for q in range(stride)
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Unsupported select_mode: {select_mode}")
        assert reshaped_key.shape[-2] == k_reshaped_seq_len

    for chunk_idx in range(q_chunk_num):
        if use_triton:
            if kdb != 1:
                raise ValueError("use_triton and kdb cannot be used together")
            attn_weights_slice = flat_group_gemm_fuse_reshape(
                pad_query_states[
                    :,
                    :,
                    (chunk_idx * reshaped_chunk_size)
                    * stride : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
                    * stride,
                    :,
                ],
                pad_key_states,
                stride,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size
                + reshaped_chunk_size,
                is_causal=causal,
            )
            attn_sum = softmax_fuse_block_sum(
                attn_weights_slice,
                reshaped_block_size,
                min(4096, reshaped_block_size),
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size
                + reshaped_chunk_size,
                k_reshaped_seq_len - k_reshaped_num_to_pad,
                1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
                is_causal=causal,
            )
        else:
            chunked_query = reshaped_query[
                :,
                :,
                (chunk_idx * reshaped_chunk_size)
                // kdb : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
                // kdb,
                :,
            ]
            attn_weights_slice = torch.matmul(
                chunked_query,
                reshaped_key.transpose(2, 3),
            ).to(pad_query_states.device)

            attn_weights_slice = (
                attn_weights_slice / math.sqrt(head_dim) / stride / norm
            )

            if causal:
                causal_mask = torch.zeros(
                    (
                        batch_size,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size * k_chunk_num,
                    ),
                    device=target_device,
                )
                causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
                chunk_start = (
                    (chunk_idx + offset_token_chunk_num) * reshaped_chunk_size
                )
                chunk_end = chunk_start + reshaped_chunk_size
                causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
                    torch.ones(
                        1,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size,
                        device=target_device,
                    )
                    * float("-inf"),
                    diagonal=1,
                )

                if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                    causal_mask[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = float(
                        "-inf"
                    )

                causal_mask[:, :, :, chunk_end:] = float("-inf")
                causal_mask = causal_mask[:, :, kdb - 1 :: kdb, :]
                attn_weights_slice = attn_weights_slice + causal_mask.to(
                    attn_weights_slice.device
                )

            if softmax:
                attn_weights_slice = F.softmax(
                    attn_weights_slice, dim=-1, dtype=torch.float32
                ).to(pad_query_states.dtype)
            else:
                attn_weights_slice = torch.exp(attn_weights_slice).to(
                    pad_query_states.dtype
                )
            attn_weights_slice = F.dropout(attn_weights_slice, p=0, training=False)

            if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                attn_weights_slice[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = 0

            attn_sum = (
                attn_weights_slice.view(
                    batch_size,
                    num_kv_head,
                    num_blocks_per_chunk,
                    reshaped_block_size // kdb,
                    -1,
                    reshaped_block_size,
                )
                .sum(dim=-1)
                .sum(dim=-2)
                .to(pad_query_states.device)
            )

        attn_sum_list.append(attn_sum)

        del attn_weights_slice
        if not use_triton:
            del chunked_query

    if not use_triton:
        del reshaped_query, reshaped_key

    attn_sums = torch.cat(attn_sum_list, dim=-2)

    q_blocks_valid = math.ceil(q_len / block_size)
    k_blocks_valid = math.ceil(k_len / block_size)

    attn_sums = attn_sums[:, :, :q_blocks_valid, :k_blocks_valid]

    return attn_sums.to(target_device)


