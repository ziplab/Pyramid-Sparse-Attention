"""Attention Pooling Kernel"""

import torch
import triton
import triton.language as tl


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner_optimized(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    R_block_ptr,  #
    A_block_ptr,  #
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * TILE_M
    elif STAGE == 2:
        lo, hi = start_m * TILE_M, (start_m + 1) * TILE_M
        lo = tl.multiple_of(lo, TILE_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    # V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, TILE_N):
        start_n = tl.multiple_of(start_n, TILE_N)
        is_last_block = start_n + TILE_N >= hi  # Check if this is the last block
        remaining = hi - start_n
        mask = tl.arange(0, TILE_N) < remaining
        k = tl.load(K_block_ptr)

        # Apply the same logic when computing qk
        qk = tl.dot(q, k)
        if is_last_block:
            qk = tl.where(mask, qk, -float("inf"))  # Mask invalid positions
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk += tl.where(mask, 0, -1.0e6)
        # Perform blocking, compute block_row_max
        blocked_qk = tl.reshape(qk, (TILE_M, TILE_N // BLOCK_N, BLOCK_N))
        block_row_max = (
            tl.max(blocked_qk, axis=2) * qk_scale
        )  # (TILE_M, TILE_N // BLOCK_N)
        max = tl.max(block_row_max, axis=1)  # (TILE_M,)
        m_ij = tl.maximum(m_i, max)
        # qk = qk * qk_scale - m_ij[:, None]
        tl.store(
            tl.advance(R_block_ptr, (0, start_n // BLOCK_N)), block_row_max.to(q.dtype)
        )
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, TILE_N))

    # -- update Po --
    if STAGE == 2:
        for start_n in range(0, (start_m + 1) * BLOCK_N, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            row_max = tl.load(R_block_ptr)
            xi = row_max - m_i[:, None]
            row_max = tl.exp2(xi) / l_i[:, None]
            blocked_row_max = tl.reshape(
                row_max, (TILE_M // BLOCK_M, BLOCK_M, TILE_N // BLOCK_N)
            )
            col_max = tl.max(
                blocked_row_max, axis=1
            )  # (TILE_M // BLOCK_M, TILE_N // BLOCK_N)
            col_max = col_max.to(q.dtype)
            tl.store(A_block_ptr, col_max)
            A_block_ptr = tl.advance(A_block_ptr, (0, TILE_N // BLOCK_N))
            R_block_ptr = tl.advance(R_block_ptr, (0, TILE_N // BLOCK_N))

    elif STAGE == 3:
        for start_n in range(lo, hi, TILE_N):
            start_n = tl.multiple_of(start_n, TILE_N)
            row_max = tl.load(R_block_ptr)
            xi = row_max - m_i[:, None]
            row_max = tl.exp2(xi) / l_i[:, None]
            blocked_row_max = tl.reshape(
                row_max, (TILE_M // BLOCK_M, BLOCK_M, TILE_N // BLOCK_N)
            )
            col_max = tl.max(
                blocked_row_max, axis=1
            )  # (TILE_M // BLOCK_M, TILE_N // BLOCK_N)
            col_max = col_max.to(q.dtype)
            # if tl.program_id(0) == 0 and tl.program_id(1) == 0:
            #     tl.device_print("start_n", start_n)
            #     # tl.device_print("max_opt", m_i)
            #     tl.device_print("col_max_opt", col_max)
            # if tl.program_id(0) == 0 and tl.program_id(1) == 0:
            #     tl.device_print("start_m", start_m)
            tl.store(A_block_ptr, col_max)
            A_block_ptr = tl.advance(A_block_ptr, (0, TILE_N // BLOCK_N))
            R_block_ptr = tl.advance(R_block_ptr, (0, TILE_N // BLOCK_N))

    return acc, l_i, m_i


@triton.jit
def _attn_fwd_optimized(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
    R,
    Po,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    stride_rz,
    stride_rh,
    stride_rm,
    stride_rn,  #
    stride_poz,
    stride_poh,
    stride_pom,
    stride_pon,  #
    Z,
    H,
    N_CTX_Q,  # Q sequence length
    N_CTX_K,  # K sequence length
    n_rep,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    TILE_M: tl.constexpr,  #
    TILE_N: tl.constexpr,  #
    N_DOWNSAMPLE: tl.constexpr,  #
    STAGE: tl.constexpr,  #
):
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_kvh = off_h // n_rep
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_kvh.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_kvh.to(tl.int64) * stride_vh
    r_offset = off_z.to(tl.int64) * stride_rz + off_h.to(tl.int64) * stride_rh
    po_offset = off_z.to(tl.int64) * stride_poz + off_h.to(tl.int64) * stride_poh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * TILE_M, 0),
        block_shape=(TILE_M, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX_K),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_N),
        order=(0, 1),
    )

    R_block_ptr = tl.make_block_ptr(
        base=R + r_offset,
        shape=(N_CTX_Q, N_DOWNSAMPLE),
        strides=(stride_rm, stride_rn),
        offsets=(start_m * TILE_M, 0),
        block_shape=(TILE_M, TILE_N // BLOCK_N),
        order=(0, 1),
    )
    A_block_ptr = tl.make_block_ptr(
        base=Po + po_offset,
        shape=(N_DOWNSAMPLE, N_DOWNSAMPLE),
        strides=(stride_pom, stride_pon),
        offsets=(start_m * (TILE_M // BLOCK_M), 0),
        block_shape=(TILE_M // BLOCK_M, TILE_N // BLOCK_N),
        order=(0, 1),
    )
    # initialize offsets
    offs_m = start_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = tl.arange(0, TILE_N)
    # initialize pointer to m and l
    m_i = tl.zeros([TILE_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([TILE_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([TILE_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_optimized(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            None,  #
            R_block_ptr,  #
            A_block_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            TILE_M,
            TILE_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX_K,
            V.dtype.element_ty == tl.float8e5,  #
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner_optimized(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            None,  #
            R_block_ptr,  #
            A_block_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            TILE_M,
            TILE_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX_K,
            V.dtype.element_ty == tl.float8e5,  #
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX_Q + offs_m
    tl.store(m_ptrs, m_i)
    # tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class _attention_pooling_optimized(torch.autograd.Function):

    _preallocations = {}
    _disable_cache = True  # Set to True to disable buffer caching for debugging

    @staticmethod
    def _get_buffers(q, m_d, n_d):
        # Option to disable caching for debugging
        if _attention_pooling_optimized._disable_cache:
            R = torch.empty(
                (q.shape[0], q.shape[1], q.shape[2], n_d),
                device=q.device,
                dtype=q.dtype,
            )
            Po = torch.empty(
                (q.shape[0], q.shape[1], m_d, n_d),
                device=q.device,
                dtype=q.dtype,
            )
            R.fill_(-65504.0)
            Po.zero_()
            return R, Po

        key = (
            q.device,
            q.dtype,
            q.shape[0],
            q.shape[1],
            q.shape[2],
            m_d,  # Include m_d in cache key
            n_d,
        )
        cached = _attention_pooling_optimized._preallocations.get(key)
        # Check both R and Po shapes to ensure cache validity
        if (cached is None or
            cached[0].shape != (q.shape[0], q.shape[1], q.shape[2], n_d) or
            cached[1].shape != (q.shape[0], q.shape[1], m_d, n_d)):
            R = torch.empty(
                (q.shape[0], q.shape[1], q.shape[2], n_d),
                device=q.device,
                dtype=q.dtype,
            )
            Po = torch.empty(
                (q.shape[0], q.shape[1], m_d, n_d),
                device=q.device,
                dtype=q.dtype,
            )
            _attention_pooling_optimized._preallocations[key] = (R, Po)
        else:
            R, Po = cached

        R.fill_(-65504.0)
        Po.zero_()
        return R, Po

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, block_size_m, block_size_n, tile_m=64, tile_n=64):
        assert block_size_m in {4, 8, 16, 32, 64, 128}
        assert block_size_n in {4, 8, 16, 32, 64, 128}
        assert tile_n % block_size_n == 0 and tile_m % block_size_m == 0
        assert tile_m > 0 and tile_n > 0
        assert tile_m <= 256 and tile_n <= 256  # Reasonable upper bounds
        orig_dtype = q.dtype
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        NUM_HEADS_Q, NUM_HEADS_K, NUM_HEADS_V = q.shape[1], k.shape[1], v.shape[1]
        assert NUM_HEADS_K == NUM_HEADS_V
        n_rep = NUM_HEADS_Q // NUM_HEADS_K
        o = torch.empty_like(q)
        # BLOCK_N = block_size_n
        # m_d: number of downsampled blocks along Q dimension
        # n_d: number of downsampled blocks along K dimension
        m_d = triton.cdiv(q.shape[2], block_size_m)
        n_d = triton.cdiv(k.shape[2], block_size_n)  # Use k.shape[2] for K sequence length
        R, Po = _attention_pooling_optimized._get_buffers(q, m_d, n_d)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        grid = lambda args: (
            triton.cdiv(q.shape[2], tile_m),
            q.shape[0] * q.shape[1],
            1,
        )
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        _attn_fwd_optimized[grid](
            q,
            k,
            v,
            sm_scale,
            M,
            o,  #
            R,
            Po,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),  #
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            R.stride(0),
            R.stride(1),
            R.stride(2),
            R.stride(3),  #
            Po.stride(0),
            Po.stride(1),
            Po.stride(2),
            Po.stride(3),  #
            q.shape[0],
            q.shape[1],  #
            N_CTX_Q=q.shape[2],  # Q sequence length
            N_CTX_K=k.shape[2],  # K sequence length
            n_rep=n_rep,  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            BLOCK_M=block_size_m,
            BLOCK_N=block_size_n,
            TILE_M=tile_m,
            TILE_N=tile_n,
            N_DOWNSAMPLE=n_d,
            num_stages=4,
            num_warps=4,
            **extra_kern_args
        )
        Sum = torch.sum(Po, dim=-1, keepdim=True)
        Po.div_(Sum)
        # o = o.to(orig_dtype)
        return o, Po


def attn_with_pooling_optimized(q, k, v, causal, sm_scale, block_size_m, block_size_n, tile_m=64, tile_n=64):
    """
    Optimized attention pooling with configurable tile sizes.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        causal: Whether to use causal attention
        sm_scale: Scale factor for attention
        block_size: Block size for computation
        tile_m: Tile size for M dimension (default: 64)
        tile_n: Tile size for N dimension (default: 64)

    Returns:
        Tuple of (output, pooling_map)
    """
    return _attention_pooling_optimized.apply(q, k, v, causal, sm_scale, block_size_m, block_size_n, tile_m, tile_n)
