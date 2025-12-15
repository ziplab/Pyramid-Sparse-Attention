import torch
import triton
import triton.language as tl
import torch.nn.functional as F

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

def calc_k_similarity_pytorch(k, blocksize, config):
    """
    PyTorch fallback implementation of calc_k_similarity.

    Args:
        k: [B, H, L, D] - Key tensor
        blocksize: int - Block size
        config: AttentionConfig - Configuration containing similarity thresholds

    Returns:
        similarity: [B, H, k_chunked_num] - Similarity mask with values in {1, 2, 4, 8}
    """
    k_chunked_num = k.shape[-2] // blocksize
    # Reshape to [B, H, k_chunked_num, blocksize, D]
    k_chunked = k[:, :, :k_chunked_num*blocksize, :].reshape(
        k.shape[0], k.shape[1], k_chunked_num, blocksize, k.shape[-1]
    )
    # Separate tokens at even and odd positions
    k_chunked_1 = k_chunked[..., ::2, :]   # [B, H, k_chunked_num, blocksize//2, D]
    k_chunked_2 = k_chunked[..., 1::2, :]  # [B, H, k_chunked_num, blocksize//2, D]
    # Compute cosine similarity on feature dimension (dim=-1)
    # cosine_similarity normalizes internally, no manual norm needed
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
    sim_2_mask = 2*(similarity_2 > config.sim_2x_threshold)
    sim_4_mask = 4*(similarity_4 > config.sim_4x_threshold)
    sim_8_mask = 8*(similarity_8 > config.sim_8x_threshold)
    one_tensor = torch.ones_like(sim_2_mask)
    sim_mask = torch.maximum(one_tensor, torch.maximum(sim_2_mask, torch.maximum(sim_4_mask, sim_8_mask)))
    return sim_mask
