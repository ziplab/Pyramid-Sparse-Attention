"""PSA Triton Kernel"""

from typing import TypeVar
from functools import lru_cache
import math
import torch
import numpy as np

import triton
import triton.language as tl
import torch.nn.functional as F
import torch
import os


import dataclasses




def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@triton.jit
def _fwd_kernel_inner(
    acc,
    l_i,
    m_i,
    q,
    k_ptrs,
    v_ptrs,
    offs,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    pooling_bias: tl.constexpr,  # log(pooling_level)
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    POOLING_BLOCK_N: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634  # 1/ln(2)
    # -- compute qk ----

    k = tl.load(k_ptrs + (offs * stride_kt)[None, :], mask=offs[None, :] < seqlen_k)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    qk += tl.where(offs[None, :] < seqlen_k, 0, -float("inf"))
    qk += pooling_bias
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    if USE_EXP2:
        p = tl.math.exp2(qk * RCP_LN2)
    else:
        p = tl.math.exp(qk)
    l_ij = tl.sum(p, 1)
    if USE_EXP2:
        alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
    else:
        alpha = tl.math.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # update acc
    v = tl.load(v_ptrs + (offs * stride_vt)[:, None], mask=offs[:, None] < seqlen_k)

    p = p.to(v.type.element_ty)

    acc += tl.dot(p, v)
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
    BLOCK_NUM: tl.constexpr,
    N_CTX,
    N_CTX_2,
    N_CTX_4,
    N_CTX_8,
    N_CTX_16,
    INFERENCE: tl.constexpr,
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
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    mask_ptrs = block_mask_ptr + start_m * stride_bmm

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)

    # global status
    MASK_MODE = 1

    # loop over k, v and update accumulator
    block_idx = 0
    while block_idx < BLOCK_NUM and MASK_MODE <= 8:

        block_addr = tl.load(mask_ptrs + block_idx * stride_bmn)
        # tl.device_print("block_addr", block_addr)
        if block_addr == -1:
            MASK_MODE *= 2
            block_idx += 1
        else:
            # set parameters according to mask
            exec_k_ptrs = K_2 + offs_d[:, None] * stride_kd_2
            exec_v_ptrs = V_2 + offs_d[None, :] * stride_vd_2
            POOLING_BLOCK_N_CUR = POOLING_BLOCK_N_2
            POOLING_BIAS = LOG_2
            stride_kt = stride_kn_2
            stride_vt = stride_vn_2
            seqlen_k = N_CTX_2
            if MASK_MODE == 1:
                exec_k_ptrs = K + offs_d[:, None] * stride_kd
                exec_v_ptrs = V + offs_d[None, :] * stride_vd
                POOLING_BLOCK_N_CUR = POOLING_BLOCK_N
                POOLING_BIAS = LOG_1
                stride_kt = stride_kn
                stride_vt = stride_vn
                seqlen_k = N_CTX
            if MASK_MODE == 2:
                exec_k_ptrs = K_2 + offs_d[:, None] * stride_kd_2
                exec_v_ptrs = V_2 + offs_d[None, :] * stride_vd_2
                POOLING_BLOCK_N_CUR = POOLING_BLOCK_N_2
                POOLING_BIAS = LOG_2
                stride_kt = stride_kn_2
                stride_vt = stride_vn_2
                seqlen_k = N_CTX_2
            if MASK_MODE == 4:
                exec_k_ptrs = K_4 + offs_d[:, None] * stride_kd_4
                exec_v_ptrs = V_4 + offs_d[None, :] * stride_vd_4
                POOLING_BLOCK_N_CUR = POOLING_BLOCK_N_4
                POOLING_BIAS = LOG_4
                stride_kt = stride_kn_4
                stride_vt = stride_vn_4
                seqlen_k = N_CTX_4
            if MASK_MODE == 8:
                exec_k_ptrs = K_8 + offs_d[:, None] * stride_kd_8
                exec_v_ptrs = V_8 + offs_d[None, :] * stride_vd_8
                POOLING_BLOCK_N_CUR = POOLING_BLOCK_N_8
                POOLING_BIAS = LOG_8
                stride_kt = stride_kn_8
                stride_vt = stride_vn_8
                seqlen_k = N_CTX_8

            # # compute attention
            if POOLING_BLOCK_N_CUR >= BLOCK_N:
                # break down the large pooling block into small pieces
                offs = offs_n + block_addr * POOLING_BLOCK_N_CUR
                block_idx += 1
            else:
                # merge all small pooling blocks into one large piece
                offs = offs_n
                for i in range(2):
                    delta_mask_1 = offs_n >= (i * POOLING_BLOCK_N_CUR)
                    delta_mask_2 = offs_n < ((i + 1) * POOLING_BLOCK_N_CUR)
                    delta_mask = delta_mask_1 & delta_mask_2
                    offs = offs + tl.where(
                        delta_mask,
                        block_addr * POOLING_BLOCK_N_CUR - i * POOLING_BLOCK_N_CUR,
                        0,
                    )
                    block_idx += 1
                    block_addr = tl.load(mask_ptrs + block_idx * stride_bmn)
                    
            for _ in range(
                POOLING_BLOCK_N_CUR // BLOCK_N if POOLING_BLOCK_N_CUR >= BLOCK_N else 1
            ):
                acc, l_i, m_i = _fwd_kernel_inner(
                    acc,
                    l_i,
                    m_i,
                    q,
                    exec_k_ptrs,
                    exec_v_ptrs,
                    offs,
                    stride_kt,
                    stride_vt,
                    sm_scale,
                    seqlen_k,
                    POOLING_BIAS,
                    1 if (block_addr + 1) * POOLING_BLOCK_N_CUR >= seqlen_k else 0,
                    BLOCK_M,
                    BLOCK_N,
                    POOLING_BLOCK_N_CUR,
                    False
                )
                offs = offs + BLOCK_N

    m_i += tl.math.log(l_i)
    l_recip = 1 / l_i[:, None]
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

def pooling(x, zoom_ratio):
    """
    对输入张量 x 进行池化操作，使用平均池化。
    x: [B, H, L, D]
    zoom_ratio: 池化的缩放比例
    """
    B, H, L, D = x.shape
    
    # 确保序列长度能被zoom_ratio整除，否则pad
    remainder = L % zoom_ratio
    if remainder != 0:
        pad_len = zoom_ratio - remainder
        # 在序列维度后面补充，使用replicate模式
        x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
        L = x.shape[2]  # 更新长度
    
    # 使用平均池化
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

    block_num = block_sparse_mask.shape[-1]

    with torch.cuda.device(q.device.index): 
        _fwd_kernel[grid](
            q, k, k_2, k_4, k_8, k_16, v, v_2, v_4, v_8, v_16, sm_scale,
            block_sparse_mask,
            o,
            None, None,  # L and M not needed for inference-only
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
            H, block_num, N_CTX, N_CTX_2, N_CTX_4, N_CTX_8, N_CTX_16,
            True,  # Always inference mode
            BLOCK_M,
            BLOCK_N,
            POOLING_BLOCK_N,
            POOLING_BLOCK_N // 2,  # POOLING_BLOCK_N_2
            POOLING_BLOCK_N // 4,  # POOLING_BLOCK_N_4
            POOLING_BLOCK_N // 8,  # POOLING_BLOCK_N_8
            POOLING_BLOCK_N // 16,  # POOLING_BLOCK_N_16
            BLOCK_DMODEL,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return o


def sparse_attention(q, k, v, block_sparse_dense, sm_scale=None):
    """Sparse attention function for inference."""
    sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
    return _forward(None, q, k, v, block_sparse_dense, sm_scale)


def sparse_attention_factory(BLOCK_M=128, BLOCK_N=128, POOLING_BLOCK_N=128, **kwargs):
    """Factory function to create sparse attention with custom block sizes."""
    def _sparse_attention_fn(q, k, v, block_sparse_dense, sm_scale=None):
        sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        return _forward(
            None,
            q,
            k,
            v,
            block_sparse_dense,
            sm_scale,
            BLOCK_M,
            BLOCK_N,
            POOLING_BLOCK_N,
            **kwargs,
        )
    return _sparse_attention_fn


block_sparse_triton_fn = sparse_attention

    
