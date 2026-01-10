"""
PSA Triton Kernel with Optional Causal Mask Support (Old Mask Format)

Original Author: Eric Lin (xihlin)
Modified by Yizhao Gao - Added causal mask support for pyramid sparse attention.

This kernel supports:
- Multi-level pooled KV representations (1x, 2x, 4x, 8x pooling)
- Optional causal masking with proper handling of pooled tokens (causal=True/False)
- Old mask format where mask values indicate pooling level (0, 1, 2, 4, 8)

Usage:
    # Without causal masking (default)
    fn = sparse_attention_factory(causal=False)

    # With causal masking
    fn = sparse_attention_factory(causal=True)
"""


import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl




def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@triton.jit
def _fwd_kernel_inner(
    acc,
    l_i,
    m_i,
    q,
    pooling_block_idx,
    k_ptrs,
    v_ptrs,
    offs_m_idx,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    pooling_bias: tl.constexpr,  # log(pooling_level)
    POOLING_RATIO: tl.constexpr,
    LAST_K_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    POOLING_BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    for inner_idx in range(POOLING_BLOCK_N // BLOCK_N):
        start_n = pooling_block_idx * POOLING_BLOCK_N + inner_idx * BLOCK_N
        # -- compute qk ----

        valid_k = offs_n[None, :] + start_n < seqlen_k
        k = tl.load(
            k_ptrs + start_n * stride_kt,
            mask=valid_k,
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
        qk += pooling_bias

        if LAST_K_BLOCK:
            qk += tl.where(valid_k, 0.0, -float("inf"))
        if CAUSAL:
            key_upper = (start_n + offs_n + 1) * POOLING_RATIO - 1
            causal_mask = key_upper[None, :] <= offs_m_idx[:, None]
            qk = tl.where(causal_mask, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk * RCP_LN2)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # update acc
        v = tl.load(
            v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k
        )

        p = p.to(v.type.element_ty)

        acc += tl.dot(p, v).to(tl.float32)
        # update m_i and l_i
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel_inner_1(
    acc,
    l_i,
    m_i,
    q,
    pooling_block_idx,
    k_ptrs,
    v_ptrs,
    offs_m_idx,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # POOLING_BLOCK_N
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = pooling_block_idx * BLOCK_N
    # -- compute qk ----

    valid_k = offs_n[None, :] + start_n < seqlen_k
    k = tl.load(
        k_ptrs + start_n * stride_kt,
        mask=valid_k,
        other=0.0,
    )

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    if LAST_K_BLOCK:
        qk += tl.where(valid_k, 0.0, -float("inf"))
    if CAUSAL:
        key_positions = start_n + offs_n
        causal_mask = key_positions[None, :] <= offs_m_idx[:, None]
        qk = tl.where(causal_mask, qk, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk * RCP_LN2)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # update acc
    v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)

    p = p.to(v.type.element_ty)

    acc += tl.dot(p, v).to(tl.float32)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i

@triton.jit
def _fwd_kernel_inner_2(
    acc,
    l_i,
    m_i,
    q,
    pooling_block_idx,
    k_ptrs,
    v_ptrs,
    offs_m_idx,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = pooling_block_idx * BLOCK_N
    # -- compute qk ----

    valid_k = offs_n[None, :] + start_n < seqlen_k
    k = tl.load(
        k_ptrs + start_n * stride_kt,
        mask=valid_k,
        other=0.0,
    )

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    qk += 0.6931471805599453094  # log(2)
    if LAST_K_BLOCK:
        qk += tl.where(valid_k, 0.0, -float("inf"))
    if CAUSAL:
        key_upper = (start_n + offs_n + 1) * 2 - 1
        causal_mask = key_upper[None, :] <= offs_m_idx[:, None]
        qk = tl.where(causal_mask, qk, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk * RCP_LN2)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # update acc
    v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)

    p = p.to(v.type.element_ty)

    acc += tl.dot(p, v).to(tl.float32)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i

@triton.jit
def _fwd_kernel_inner_4(
    acc,
    l_i,
    m_i,
    q,
    k_block_col_idx,
    k_ptrs,
    v_ptrs,
    offs_m_idx,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = k_block_col_idx * BLOCK_N
    # -- compute qk ----

    valid_k = offs_n[None, :] + start_n < seqlen_k
    k = tl.load(
        k_ptrs + start_n * stride_kt,
        mask=valid_k,
        other=0.0,
    )

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    qk += 1.3862943611198906188  # log(4)
    if LAST_K_BLOCK:
        qk += tl.where(valid_k, 0.0, -float("inf"))
    if CAUSAL:
        key_upper = (start_n + offs_n + 1) * 4 - 1
        causal_mask = key_upper[None, :] <= offs_m_idx[:, None]
        qk = tl.where(causal_mask, qk, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk * RCP_LN2)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # update acc
    v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)

    p = p.to(v.type.element_ty)

    acc += tl.dot(p, v).to(tl.float32)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i
@triton.jit
def _fwd_kernel_inner_8(
    acc,
    l_i,
    m_i,
    q,
    k_block_col_idx,
    k_ptrs,
    v_ptrs,
    offs_m_idx,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = k_block_col_idx * BLOCK_N
    # -- compute qk ----

    valid_k = offs_n[None, :] + start_n < seqlen_k
    k = tl.load(
        k_ptrs + start_n * stride_kt,
        mask=valid_k,
        other=0.0,
    )

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    qk += 2.0794415416798359283  # log(8)
    if LAST_K_BLOCK:
        qk += tl.where(valid_k, 0.0, -float("inf"))
    if CAUSAL:
        key_upper = (start_n + offs_n + 1) * 8 - 1
        causal_mask = key_upper[None, :] <= offs_m_idx[:, None]
        qk = tl.where(causal_mask, qk, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk * RCP_LN2)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # update acc
    v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)

    p = p.to(v.type.element_ty)

    acc += tl.dot(p, v).to(tl.float32)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i
@triton.jit
def _fwd_kernel_inner_16(
    acc,
    l_i,
    m_i,
    q,
    k_block_col_idx,
    k_ptrs,
    v_ptrs,
    offs_m_idx,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = k_block_col_idx * BLOCK_N
    # -- compute qk ----

    valid_k = offs_n[None, :] + start_n < seqlen_k
    k = tl.load(
        k_ptrs + start_n * stride_kt,
        mask=valid_k,
        other=0.0,
    )

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    if LAST_K_BLOCK:
        qk += tl.where(valid_k, 0.0, -float("inf"))
    qk += 2.7725887222397812377  # log(16)

    if CAUSAL:
        key_upper = (start_n + offs_n + 1) * 16 - 1
        causal_mask = key_upper[None, :] <= offs_m_idx[:, None]
        qk = tl.where(causal_mask, qk, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.exp(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # update acc
    v = tl.load(
        v_ptrs + start_n * stride_vt,
        mask=offs_n[:, None] + start_n < seqlen_k,
        other=0.0,
    )

    p = p.to(v.type.element_ty)

    acc += tl.dot(p, v).to(tl.float32)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i

@triton.jit
def _fwd_kernel(
    Q,
    K,
    K_2,
    K_4,
    K_8,
    K_16,
    V,
    V_2,
    V_4,
    V_8,
    V_16,
    sm_scale,
    block_mask_ptr,
    Out,
    L, M,  # For backward pass: L is row-wise sum, M is row-wise max
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_kz_2,
    stride_kh_2,
    stride_kn_2,
    stride_kd_2,
    stride_kz_4,
    stride_kh_4,
    stride_kn_4,
    stride_kd_4,
    stride_kz_8,
    stride_kh_8,
    stride_kn_8,
    stride_kd_8,
    stride_kz_16,
    stride_kh_16,
    stride_kn_16,
    stride_kd_16,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_vz_2,
    stride_vh_2,
    stride_vn_2,
    stride_vd_2,
    stride_vz_4,
    stride_vh_4,
    stride_vn_4,
    stride_vd_4,
    stride_vz_8,
    stride_vh_8,
    stride_vn_8,
    stride_vd_8,
    stride_vz_16,
    stride_vh_16,
    stride_vn_16,
    stride_vd_16,
    stride_bmz,
    stride_bmh,
    stride_bmm,
    stride_bmn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    H,
    N_CTX,
    N_CTX_2,
    N_CTX_4,
    N_CTX_8,
    N_CTX_16,
    INFERENCE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    POOLING_BLOCK_N: tl.constexpr,
    POOLING_BLOCK_N_2: tl.constexpr,
    POOLING_BLOCK_N_4: tl.constexpr,
    POOLING_BLOCK_N_8: tl.constexpr,
    POOLING_BLOCK_N_16: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    LOG_1 = 0.0
    LOG_2 = 0.6931471805599453094  # log(2)
    LOG_4 = 1.3862943611198906188  # log(4)
    LOG_8 = 2.0794415416798359283  # log(8)

    Q_LEN = N_CTX
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    K_2 += off_z * stride_kz_2 + off_h * stride_kh_2
    K_4 += off_z * stride_kz_4 + off_h * stride_kh_4
    K_8 += off_z * stride_kz_8 + off_h * stride_kh_8
    K_16 += off_z * stride_kz_16 + off_h * stride_kh_16
    V += off_z * stride_vz + off_h * stride_vh
    V_2 += off_z * stride_vz_2 + off_h * stride_vh_2
    V_4 += off_z * stride_vz_4 + off_h * stride_vh_4
    V_8 += off_z * stride_vz_8 + off_h * stride_vh_8
    V_16 += off_z * stride_vz_16 + off_h * stride_vh_16
    block_mask_ptr += off_z * stride_bmz + off_h * stride_bmh

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_n_1 = tl.arange(0, POOLING_BLOCK_N)
    offs_n_2 = tl.arange(0, POOLING_BLOCK_N_2)
    offs_n_4 = tl.arange(0, POOLING_BLOCK_N_4)
    offs_n_8 = tl.arange(0, POOLING_BLOCK_N_8)
    offs_n_16 = tl.arange(0, POOLING_BLOCK_N_16)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd

    if POOLING_BLOCK_N < BLOCK_N:
        off_k_1 = offs_n_1[None, :] * stride_kn + offs_d[:, None] * stride_kd
        off_v_1 = offs_n_1[:, None] * stride_vn + offs_d[None, :] * stride_vd
    else:
        off_k_1 = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        off_v_1 = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    if POOLING_BLOCK_N_2 < BLOCK_N:
        off_k_2 = offs_n_2[None, :] * stride_kn_2 + offs_d[:, None] * stride_kd_2
        off_v_2 = offs_n_2[:, None] * stride_vn_2 + offs_d[None, :] * stride_vd_2
    else:
        off_k_2 = offs_n[None, :] * stride_kn_2 + offs_d[:, None] * stride_kd_2
        off_v_2 = offs_n[:, None] * stride_vn_2 + offs_d[None, :] * stride_vd_2
    if POOLING_BLOCK_N_4 < BLOCK_N:
        off_k_4 = offs_n_4[None, :] * stride_kn_4 + offs_d[:, None] * stride_kd_4
        off_v_4 = offs_n_4[:, None] * stride_vn_4 + offs_d[None, :] * stride_vd_4
    else:
        off_k_4 = offs_n[None, :] * stride_kn_4 + offs_d[:, None] * stride_kd_4
        off_v_4 = offs_n[:, None] * stride_vn_4 + offs_d[None, :] * stride_vd_4
    if POOLING_BLOCK_N_8 < BLOCK_N:
        off_k_8 = offs_n_8[None, :] * stride_kn_8 + offs_d[:, None] * stride_kd_8
        off_v_8 = offs_n_8[:, None] * stride_vn_8 + offs_d[None, :] * stride_vd_8
    else:
        off_k_8 = offs_n[None, :] * stride_kn_8 + offs_d[:, None] * stride_kd_8
        off_v_8 = offs_n[:, None] * stride_vn_8 + offs_d[None, :] * stride_vd_8

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs_1 = K + off_k_1
    k_ptrs_2 = K_2 + off_k_2
    k_ptrs_4 = K_4 + off_k_4
    k_ptrs_8 = K_8 + off_k_8
    # k_ptrs_16 = K_16 + off_k_16
    v_ptrs_1 = V + off_v_1
    v_ptrs_2 = V_2 + off_v_2
    v_ptrs_4 = V_4 + off_v_4
    v_ptrs_8 = V_8 + off_v_8
    # v_ptrs_16 = V_16 + off_v_16
    mask_ptrs = block_mask_ptr + start_m * stride_bmm

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32)-float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)

    pooling_block_start = 0
    pooling_block_end = tl.cdiv(N_CTX, POOLING_BLOCK_N)

    # For causal attention, only process K blocks up to current Q position
    if CAUSAL:
        causal_block_end = tl.cdiv((start_m + 1) * BLOCK_M, POOLING_BLOCK_N)
        pooling_block_end = tl.minimum(pooling_block_end, causal_block_end)

    # loop over k, v and update accumulator
    for pb_idx in range(pooling_block_start, pooling_block_end):
        mask = tl.load(mask_ptrs + pb_idx * stride_bmn)

        # Check if this is the last block
        is_last_block = pb_idx == pooling_block_end - 1

        if mask > 0:
            if mask < 4:
                if mask == 1:
                    if POOLING_BLOCK_N <= BLOCK_N:
                        acc, l_i, m_i = _fwd_kernel_inner_1(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_1,
                            v_ptrs_1,
                            offs_m,
                            offs_n_1,
                            stride_kn,
                            stride_vn,
                            sm_scale,
                            N_CTX,
                            is_last_block,
                            CAUSAL,
                            BLOCK_M,
                            POOLING_BLOCK_N,
                        )
                    else:
                        acc, l_i, m_i = _fwd_kernel_inner(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_1,
                            v_ptrs_1,
                            offs_m,
                            offs_n,
                            stride_kn,
                            stride_vn,
                            sm_scale,
                            N_CTX,
                            LOG_1,
                            1,
                            is_last_block,
                            CAUSAL,
                            BLOCK_M,
                            BLOCK_N,
                            POOLING_BLOCK_N,
                        )
                if mask == 2:
                    if POOLING_BLOCK_N_2 <= BLOCK_N:
                        acc, l_i, m_i = _fwd_kernel_inner_2(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_2,
                            v_ptrs_2,
                            offs_m,
                            offs_n_2,
                            stride_kn_2,
                            stride_vn_2,
                            sm_scale,
                            N_CTX_2,
                            is_last_block,
                            CAUSAL,
                            BLOCK_M,
                            POOLING_BLOCK_N_2,
                        )
                    else:
                        acc, l_i, m_i = _fwd_kernel_inner(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_2,
                            v_ptrs_2,
                            offs_m,
                            offs_n,
                            stride_kn_2,
                            stride_vn_2,
                            sm_scale,
                            N_CTX_2,
                            LOG_2,
                            2,
                            is_last_block,
                            CAUSAL,
                            BLOCK_M,
                            BLOCK_N,
                            POOLING_BLOCK_N_2,
                        )
            else:
                if mask == 8:
                    if POOLING_BLOCK_N_8 <= BLOCK_N:
                        acc, l_i, m_i = _fwd_kernel_inner_8(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_8,
                            v_ptrs_8,
                            offs_m,
                            offs_n_8,
                            stride_kn_8,
                            stride_vn_8,
                            sm_scale,
                            N_CTX_8,
                            is_last_block,
                            CAUSAL,
                            BLOCK_M,
                            POOLING_BLOCK_N_8,
                        )
                    else:
                        acc, l_i, m_i = _fwd_kernel_inner(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_8,
                            v_ptrs_8,
                            offs_m,
                            offs_n,
                            stride_kn_8,
                            stride_vn_8,
                            sm_scale,
                            N_CTX_8,
                            LOG_8,
                            8,
                            is_last_block,
                            CAUSAL,
                            BLOCK_M,
                            BLOCK_N,
                            POOLING_BLOCK_N_8,
                        )
                if mask == 4:
                    if POOLING_BLOCK_N_4 <= BLOCK_N:
                        acc, l_i, m_i = _fwd_kernel_inner_4(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_4,
                            v_ptrs_4,
                            offs_m,
                            offs_n_4,
                            stride_kn_4,
                            stride_vn_4,
                            sm_scale,
                            N_CTX_4,
                            is_last_block,
                            CAUSAL,
                            BLOCK_M,
                            POOLING_BLOCK_N_4,
                        )
                    else:
                        acc, l_i, m_i = _fwd_kernel_inner(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_4,
                            v_ptrs_4,
                            offs_m,
                            offs_n,
                            stride_kn_4,
                            stride_vn_4,
                            sm_scale,
                            N_CTX_4,
                            LOG_4,
                            4,
                            is_last_block,
                            CAUSAL,
                            BLOCK_M,
                            BLOCK_N,
                            POOLING_BLOCK_N_4,
                        )

    # Store L and M for backward pass
    if not INFERENCE:
        l_ptrs = L + off_hz * N_CTX + offs_m
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(l_ptrs, l_i, mask=offs_m < Q_LEN)
        tl.store(m_ptrs, m_i, mask=offs_m < Q_LEN)

    # Add epsilon for numerical stability to prevent log(0) and division by zero
    EPS: tl.constexpr = 1e-8
    l_i_safe = tl.maximum(l_i, EPS)

    m_i += tl.math.log(l_i_safe)
    l_recip = 1 / l_i_safe[:, None]
    acc = acc * l_recip

    acc = acc.to(Out.dtype.element_ty)

    off_o = (
        off_z * stride_oz
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < Q_LEN)


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def pad_to_multiple(x, multiple):
    """
    Pad sequence dimension (dim=2) to make length a multiple of `multiple`.
    x: [B, H, L, D]
    """
    L = x.size(2)
    remainder = L % multiple
    if remainder != 0:
        pad_len = multiple - remainder
        x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
    return x


def pooling(x, zoom_ratio):
    """
    Apply average pooling to input tensor.
    x: [B, H, L, D]
    zoom_ratio: pooling ratio
    """
    B, H, L, D = x.shape

    remainder = L % zoom_ratio
    if remainder != 0:
        pad_len = zoom_ratio - remainder
        x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
        L = x.shape[2]

    x = torch.mean(x.view(B, H, -1, zoom_ratio, D), dim=3)
    return x


def _forward(
    ctx,
    q,
    k,
    v,
    block_sparse_mask,
    sm_scale,
    BLOCK_M=64,
    BLOCK_N=64,
    POOLING_BLOCK_N=128,
    num_warps=None,
    num_stages=1,
    out=None,
    causal=False,
):
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[2] == v.shape[2]
    o = out if out is not None else torch.empty_like(q).contiguous()

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])

    assert q.shape[-1] in [64, 128]
    BLOCK_DMODEL = q.shape[-1]

    if is_hip():
        num_warps, num_stages = 8, 1
    else:
        num_warps, num_stages = 4, 2

    H = q.shape[1]

    k_padding = pad_to_multiple(k, POOLING_BLOCK_N)
    v_padding = pad_to_multiple(v, POOLING_BLOCK_N)
    k_2 = pooling(k_padding, 2)
    v_2 = pooling(v_padding, 2)
    k_4 = pooling(k_2, 2)
    v_4 = pooling(v_2, 2)
    k_8 = pooling(k_4, 2)
    v_8 = pooling(v_4, 2)
    k_16 = pooling(k_8, 2)
    v_16 = pooling(v_8, 2)
    N_CTX = k.shape[2]
    N_CTX_2 = k_2.shape[2]
    N_CTX_4 = k_4.shape[2]
    N_CTX_8 = k_8.shape[2]
    N_CTX_16 = k_16.shape[2]

    # Note: torch.autograd.Function.forward runs under no_grad by default, so
    # torch.is_grad_enabled() is not a reliable signal here.
    need_backward = ctx is not None and (q.requires_grad or k.requires_grad or v.requires_grad)
    L_buf = M_buf = None
    if need_backward:
        # Row-wise softmax intermediates for backward:
        # L = sum(exp(scores - m)), M = max(scores) (both in natural-log domain).
        L_buf = torch.empty((q.shape[0] * q.shape[1], N_CTX), device=q.device, dtype=torch.float32)
        M_buf = torch.empty((q.shape[0] * q.shape[1], N_CTX), device=q.device, dtype=torch.float32)

    with torch.cuda.device(q.device.index):
        _fwd_kernel[grid](
            q, k, k_2, k_4, k_8, k_16, v, v_2, v_4, v_8, v_16, sm_scale,
            block_sparse_mask,
            o,
            L_buf, M_buf,
            *q.stride(),
            *k.stride(),
            *k_2.stride(),
            *k_4.stride(),
            *k_8.stride(),
            *k_16.stride(),
            *v.stride(),
            *v_2.stride(),
            *v_4.stride(),
            *v_8.stride(),
            *v_16.stride(),
            *block_sparse_mask.stride(),
            *o.stride(),
            H, N_CTX, N_CTX_2, N_CTX_4, N_CTX_8, N_CTX_16,
            not need_backward,
            causal,
            BLOCK_M,
            BLOCK_N,
            POOLING_BLOCK_N,
            POOLING_BLOCK_N // 2,
            POOLING_BLOCK_N // 4,
            POOLING_BLOCK_N // 8,
            POOLING_BLOCK_N // 16,
            BLOCK_DMODEL,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if need_backward:
        # Reconstruct per-row LSE (logsumexp) in natural-log domain.
        eps = 1e-8
        lse = M_buf + torch.log(torch.clamp(L_buf, min=eps))
        ctx.save_for_backward(q, k, v, block_sparse_mask, o, lse)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.BLOCK_M = BLOCK_M
        ctx.POOLING_BLOCK_N = POOLING_BLOCK_N

    return o


class _sparse_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, block_sparse_dense, sm_scale):
        return _forward(ctx, q, k, v, block_sparse_dense, sm_scale)

    @staticmethod
    def backward(ctx, do):
        q, k, v, block_sparse_mask, out, lse = ctx.saved_tensors
        do = do.contiguous()
        B, H, L, D = q.shape
        sm_scale = ctx.sm_scale
        causal = bool(ctx.causal)
        POOLING_BLOCK_N = int(ctx.POOLING_BLOCK_N)

        # Backward tiling: use 64x64 tiles for better parallelism.
        # The legacy mask is defined at 128-token granularity; expand it to 64 (one -> 2x2 blocks).
        BLOCK_M_BWD = 64
        BLOCK_N_BWD = 64

        # We need a consistent K length for pooled levels: legacy forward pads K/V to POOLING_BLOCK_N.
        L_k_pad = int(math.ceil(L / POOLING_BLOCK_N) * POOLING_BLOCK_N)
        M_BLOCKS = triton.cdiv(L, BLOCK_M_BWD)
        N_BLOCKS = triton.cdiv(L_k_pad, BLOCK_N_BWD)

        M_BLOCKS_128 = block_sparse_mask.shape[2]
        N_BLOCKS_128 = block_sparse_mask.shape[3]
        assert M_BLOCKS_128 == triton.cdiv(L, POOLING_BLOCK_N)
        assert N_BLOCKS_128 == triton.cdiv(L_k_pad, POOLING_BLOCK_N)

        # Convert stored LSE to log2 domain for exp2-based backward kernels.
        rcp_ln2 = 1.4426950408889634
        lse_log2 = (lse * rcp_ln2).contiguous()  # [B*H, L]

        # Compute delta = rowsum(do * out) in fp32.
        delta = torch.empty((B * H, L), device=q.device, dtype=torch.float32)
        grid = (M_BLOCKS, B * H)
        _attn_bwd_preprocess[grid](
            out, do, delta,
            L=L,
            D=D,
            BLOCK_M=BLOCK_M_BWD,
            num_warps=4 if D == 64 else 8,
        )

        # Build per-level 0/1 block masks at 64-token resolution (expanded from 128-token mask).
        level_masks = _oldmask_to_level_01masks(
            block_sparse_mask=block_sparse_mask,
            out_m_blocks=M_BLOCKS,
            out_n_blocks=N_BLOCKS,
            device=q.device,
        )

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        # Process each level independently using the merged LSE from forward.
        for level, mask_flat in level_masks:
            if level == 1:
                k_level = k
                v_level = v
                L_k = L
            else:
                # Match legacy forward: pad to POOLING_BLOCK_N first, then average pool by 'level'.
                k_pad = pad_to_multiple(k, POOLING_BLOCK_N)
                v_pad = pad_to_multiple(v, POOLING_BLOCK_N)
                k_pooled = _avg_pool_kv_noextra_pad(k_pad, level)
                v_pooled = _avg_pool_kv_noextra_pad(v_pad, level)
                k_level = _repeat_kv(k_pooled, level, L_k_pad)
                v_level = _repeat_kv(v_pooled, level, L_k_pad)
                L_k = L_k_pad

            # dQ
            dq_level = torch.empty_like(q, dtype=torch.float32)
            _attn_bwd_dq_01mask_oldmask[grid](
                q, k_level, v_level, lse_log2, delta,
                do, dq_level, mask_flat,
                sm_scale,
                L_Q=L,
                L_K=L_k,
                M_BLOCKS=M_BLOCKS,
                N_BLOCKS=N_BLOCKS,
                D=D,
                BLOCK_M=BLOCK_M_BWD,
                BLOCK_N=BLOCK_N_BWD,
                POOLING_RATIO=level,
                IS_CAUSAL=causal,
                num_warps=4 if D == 64 else 8,
                num_stages=3,
            )
            dq += dq_level

            # dK, dV (computed w.r.t the expanded/repeated K/V at this level)
            dk_level = torch.empty((B, H, L_k, D), device=q.device, dtype=torch.float32)
            dv_level = torch.empty((B, H, L_k, D), device=q.device, dtype=torch.float32)
            grid_k = (N_BLOCKS, B * H)
            _attn_bwd_dkdv_01mask_oldmask[grid_k](
                q, k_level, v_level, do, dk_level, dv_level,
                sm_scale, mask_flat, lse_log2, delta,
                L_Q=L,
                L_K=L_k,
                M_BLOCKS=M_BLOCKS,
                N_BLOCKS=N_BLOCKS,
                D=D,
                BLOCK_M=BLOCK_M_BWD,
                BLOCK_N=BLOCK_N_BWD,
                POOLING_RATIO=level,
                IS_CAUSAL=causal,
                num_warps=4 if D == 64 else 8,
                num_stages=3,
            )

            if level == 1:
                dk += dk_level[:, :, :L, :]
                dv += dv_level[:, :, :L, :]
            else:
                dk += _avg_pool_repeat_backward_with_pad(
                    dk_level, ratio=level, original_len=L, padded_len=L_k_pad
                )
                dv += _avg_pool_repeat_backward_with_pad(
                    dv_level, ratio=level, original_len=L, padded_len=L_k_pad
                )

        # Match input dtypes
        dq = dq.to(q.dtype)
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)
        return dq, dk, dv, None, None


@triton.jit
def _attn_bwd_preprocess(OUT, DO, DELTA, L: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    base = pid_bh * L * D
    o = tl.load(OUT + base + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L, other=0.0).to(tl.float32)
    do = tl.load(DO + base + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(DELTA + pid_bh * L + offs_m, delta, mask=offs_m < L)


@triton.jit
def _attn_bwd_dq_01mask_oldmask(
    Q, K, V, LSE_LOG2, DELTA,
    DO, DQ, MASK,
    sm_scale,
    L_Q: tl.constexpr,
    L_K: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    POOLING_RATIO: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634

    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    qkv_offset_q = idx_bh * L_Q * D
    qkv_offset_kv = idx_bh * L_K * D
    lse_offset = idx_bh * L_Q
    mask_offset = (idx_bh * M_BLOCKS + idx_m) * N_BLOCKS

    Q_ptrs = Q + qkv_offset_q + offs_m[:, None] * D + offs_d[None, :]
    DO_ptrs = DO + qkv_offset_q + offs_m[:, None] * D + offs_d[None, :]
    DQ_ptrs = DQ + qkv_offset_q + offs_m[:, None] * D + offs_d[None, :]
    LSE_ptrs = LSE_LOG2 + lse_offset + offs_m
    DELTA_ptrs = DELTA + lse_offset + offs_m
    MASK_ptr = MASK + mask_offset

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L_Q, other=0.0)
    do = tl.load(DO_ptrs, mask=offs_m[:, None] < L_Q, other=0.0)
    delta = tl.load(DELTA_ptrs, mask=offs_m < L_Q, other=0.0)
    lse_log2 = tl.load(LSE_ptrs, mask=offs_m < L_Q, other=-float("inf"))

    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for idx_n in range(N_BLOCKS):
        mask_val = tl.load(MASK_ptr + idx_n)
        if mask_val == 1:
            start_n = idx_n * BLOCK_N
            n_mask = offs_n < (L_K - start_n)

            k = tl.load(K + qkv_offset_kv + (start_n + offs_n)[:, None] * D + offs_d[None, :], mask=n_mask[:, None], other=0.0)
            v = tl.load(V + qkv_offset_kv + (start_n + offs_n)[:, None] * D + offs_d[None, :], mask=n_mask[:, None], other=0.0)

            qk = tl.dot(q, k.T).to(tl.float32) * sm_scale

            if IS_CAUSAL:
                q_pos = offs_m
                k_pos = start_n + offs_n
                # K/V are repeated to original positions, so causal uses plain q_pos >= k_pos.
                causal_mask = q_pos[:, None] >= k_pos[None, :]
                qk = tl.where(causal_mask, qk, -float("inf"))

            # p = exp(qk - lse) computed via exp2 in log2 domain
            p = tl.math.exp2(qk * RCP_LN2 - lse_log2[:, None])
            p = tl.where(n_mask[None, :], p, 0.0)
            if IS_CAUSAL:
                p = tl.where(causal_mask, p, 0.0)

            dp = tl.dot(do, v.T).to(tl.float32)
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k)

    tl.store(DQ_ptrs, dq * sm_scale, mask=offs_m[:, None] < L_Q)


@triton.jit
def _attn_bwd_dkdv_01mask_oldmask(
    Q, K, V, DO, DK, DV,
    sm_scale, MASK, LSE_LOG2, DELTA,
    L_Q: tl.constexpr,
    L_K: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    POOLING_RATIO: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634

    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    qkv_offset_q = idx_bh * L_Q * D
    qkv_offset_kv = idx_bh * L_K * D
    lse_offset = idx_bh * L_Q
    mask_offset = idx_bh * M_BLOCKS * N_BLOCKS

    # K/V block pointers
    K_ptrs = K + qkv_offset_kv + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_offset_kv + offs_n[:, None] * D + offs_d[None, :]
    DK_ptrs = DK + qkv_offset_kv + offs_n[:, None] * D + offs_d[None, :]
    DV_ptrs = DV + qkv_offset_kv + offs_n[:, None] * D + offs_d[None, :]
    MASK_ptr = MASK + mask_offset + idx_n

    k = tl.load(K_ptrs, mask=offs_n[:, None] < L_K, other=0.0)
    v = tl.load(V_ptrs, mask=offs_n[:, None] < L_K, other=0.0)

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    idx_m_block = 0
    for idx_m in tl.range(0, L_Q, BLOCK_M):
        mask_val = tl.load(MASK_ptr + idx_m_block * N_BLOCKS)
        if mask_val == 1:
            q_pos = idx_m + offs_m
            m_mask = q_pos < L_Q

            q = tl.load(Q + qkv_offset_q + q_pos[:, None] * D + offs_d[None, :], mask=m_mask[:, None], other=0.0)
            do = tl.load(DO + qkv_offset_q + q_pos[:, None] * D + offs_d[None, :], mask=m_mask[:, None], other=0.0)
            lse_log2 = tl.load(LSE_LOG2 + lse_offset + q_pos, mask=m_mask, other=-float("inf"))
            delta = tl.load(DELTA + lse_offset + q_pos, mask=m_mask, other=0.0)

            qkT = tl.dot(k, q.T).to(tl.float32) * sm_scale

            if IS_CAUSAL:
                k_pos = offs_n
                # K/V are repeated to original positions, so causal uses plain q_pos >= k_pos.
                causal_mask = q_pos[None, :] >= k_pos[:, None]
                qkT = tl.where(causal_mask, qkT, -float("inf"))

            pT = tl.math.exp2(qkT * RCP_LN2 - lse_log2[None, :])
            pT = tl.where(offs_n[:, None] < L_K, pT, 0.0)
            pT = tl.where(m_mask[None, :], pT, 0.0)
            if IS_CAUSAL:
                pT = tl.where(causal_mask, pT, 0.0)

            dv += tl.dot(pT.to(do.dtype), do)
            dpT = tl.dot(v, tl.trans(do))
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q)

        idx_m_block += 1

    tl.store(DK_ptrs, dk * sm_scale, mask=offs_n[:, None] < L_K)
    tl.store(DV_ptrs, dv, mask=offs_n[:, None] < L_K)


def _oldmask_to_level_01masks(
    *,
    block_sparse_mask: torch.Tensor,
    out_m_blocks: int,
    out_n_blocks: int,
    device: torch.device,
):
    """
    Convert old dense mask ([B,H,M128,N128] with values in {0,1,2,4,8}) into
    per-level dense 0/1 masks at the requested output resolution.

    The legacy forward uses 128-token block granularity; when output blocks are smaller
    (e.g. 64), each (m128, n128) entry expands to multiple blocks.

    Returns: list of (level, mask_flat) where mask_flat is [B*H, out_m_blocks, out_n_blocks] int32 contiguous.
    """
    assert block_sparse_mask.is_cuda
    assert block_sparse_mask.dtype in (torch.int32, torch.int64)

    B, H, m128, n128 = block_sparse_mask.shape
    # The requested output resolution may not be an exact multiple when L is not aligned;
    # we allow trimming after expansion.
    scale = 2 if (out_m_blocks > m128 or out_n_blocks > n128) else 1
    if scale not in (1, 2):
        raise ValueError(f"Unsupported mask resize scale={scale}.")
    if out_m_blocks > m128 * scale or out_n_blocks > n128 * scale:
        raise ValueError(
            f"Unsupported mask resize: mask is [{m128},{n128}] blocks but requested [{out_m_blocks},{out_n_blocks}] (scale={scale})."
        )

    levels = [1, 2, 4, 8]
    out = []
    for level in levels:
        dense = (block_sparse_mask == level).to(torch.int32).view(B * H, m128, n128)
        if scale == 2:
            # Expand 128-block mask to 64-block resolution (2x in each dimension), then trim.
            dense = dense.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        dense = dense[:, :out_m_blocks, :out_n_blocks].contiguous()
        if dense.any():
            out.append((level, dense))
    return out


def _avg_pool_kv_noextra_pad(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """
    Average pool along sequence dimension by 'ratio' assuming x is already padded
    to a multiple of POOLING_BLOCK_N (so divisible by ratio).
    """
    B, H, L, D = x.shape
    assert L % ratio == 0
    return x.view(B, H, L // ratio, ratio, D).mean(dim=3)


def _repeat_kv(x: torch.Tensor, ratio: int, target_len: int) -> torch.Tensor:
    B, H, L_pooled, D = x.shape
    x = x.unsqueeze(3).repeat(1, 1, 1, ratio, 1).reshape(B, H, L_pooled * ratio, D)
    if x.shape[2] > target_len:
        x = x[:, :, :target_len, :]
    return x.contiguous()


def _avg_pool_repeat_backward_with_pad(grad_rep: torch.Tensor, *, ratio: int, original_len: int, padded_len: int) -> torch.Tensor:
    """
    Backward for: x_pad=pad_to_multiple(x, POOLING_BLOCK_N); pooled=avg_pool(x_pad, ratio); rep=repeat(pooled, ratio)

    grad_rep: [B,H,padded_len,D] gradient wrt repeated output at padded length.
    Returns: [B,H,original_len,D] gradient wrt original x (replicate pad folded into last token).
    """
    B, H, Lr, D = grad_rep.shape
    assert Lr == padded_len
    assert padded_len % ratio == 0

    # Sum gradients across repeats -> gradient w.r.t pooled tokens.
    grad_pooled = grad_rep.view(B, H, padded_len // ratio, ratio, D).sum(dim=3)

    # Average pooling: pooled = mean(x), so distribute grad / ratio to each token.
    grad_pad = (grad_pooled / ratio).unsqueeze(3).repeat(1, 1, 1, ratio, 1).reshape(B, H, padded_len, D)

    if padded_len > original_len:
        tail = grad_pad[:, :, original_len:, :].sum(dim=2, keepdim=True)
        grad_pad = grad_pad[:, :, :original_len, :]
        grad_pad[:, :, original_len - 1 : original_len, :] += tail

    return grad_pad.contiguous()


def sparse_attention_factory(BLOCK_M=128, BLOCK_N=32, POOLING_BLOCK_N=128, causal=False, **kwargs):
    """
    Factory function to create sparse attention with specific configuration.

    Args:
        BLOCK_M: Query block size (must be 128)
        BLOCK_N: Key block size for computation tiles (must be 32)
        POOLING_BLOCK_N: Logical pooling block size (must be 128)
        causal: Whether to use causal masking
    """
    # Validate block size parameters - current version only supports fixed values
    if BLOCK_M != 128:
        raise ValueError(
            f"BLOCK_M must be 128, got {BLOCK_M}. "
            "Current version only supports BLOCK_M=128, BLOCK_N=32, POOLING_BLOCK_N=128. "
            "For more flexible configurations, please use psa_triton/kernels/psa_kernel.py"
        )
    if BLOCK_N != 32:
        raise ValueError(
            f"BLOCK_N must be 32, got {BLOCK_N}. "
            "Current version only supports BLOCK_M=128, BLOCK_N=32, POOLING_BLOCK_N=128. "
            "For more flexible configurations, please use psa_triton/kernels/psa_kernel.py"
        )
    if POOLING_BLOCK_N != 128:
        raise ValueError(
            f"POOLING_BLOCK_N must be 128, got {POOLING_BLOCK_N}. "
            "Current version only supports BLOCK_M=128, BLOCK_N=32, POOLING_BLOCK_N=128. "
            "For more flexible configurations, please use psa_triton/kernels/psa_kernel.py"
        )

    class _sparse_attention_config(_sparse_attention):
        @staticmethod
        def forward(ctx, q, k, v, block_sparse_dense, sm_scale=None):
            sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
            return _forward(
                ctx,
                q,
                k,
                v,
                block_sparse_dense,
                sm_scale,
                BLOCK_M,
                BLOCK_N,
                POOLING_BLOCK_N,
                causal=causal,
                **kwargs,
            )
    return _sparse_attention_config.apply


block_sparse_triton_fn = _sparse_attention.apply
