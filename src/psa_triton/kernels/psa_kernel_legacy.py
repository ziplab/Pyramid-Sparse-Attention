"""PSA Triton Kernel (Legacy Mask Format)"""

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
    pooling_block_idx,
    k_ptrs,
    v_ptrs,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    pooling_bias: tl.constexpr,  # log(pooling_level)
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    POOLING_BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    for inner_idx in range(
        POOLING_BLOCK_N // BLOCK_N if POOLING_BLOCK_N > BLOCK_N else 1
    ):
        start_n = pooling_block_idx * POOLING_BLOCK_N + inner_idx * BLOCK_N
        # -- compute qk ----

        k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
        qk += tl.where(offs_n[None, :] + start_n < seqlen_k, 0, -float("inf"))
        qk += pooling_bias

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

        acc += tl.dot(p, v)
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
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # POOLING_BLOCK_N
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = pooling_block_idx * BLOCK_N
    # -- compute qk ----

    k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    qk += tl.where(offs_n[None, :] + start_n< seqlen_k, 0, -float("inf"))
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

    acc += tl.dot(p, v)
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
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = pooling_block_idx * BLOCK_N
    # -- compute qk ----

    k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    qk += tl.where(offs_n[None, :] + start_n < seqlen_k, 0, -float("inf"))
    qk += 0.6931471805599453094  # log(2)
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

    acc += tl.dot(p, v)
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
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = k_block_col_idx * BLOCK_N
    # -- compute qk ----

    k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    qk += tl.where(offs_n[None, :] + start_n < seqlen_k, 0, -float("inf"))
    qk += 1.3862943611198906188  # log(4)
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

    acc += tl.dot(p, v)
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
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = k_block_col_idx * BLOCK_N
    # -- compute qk ----

    k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    qk += tl.where(offs_n[None, :] + start_n < seqlen_k, 0, -float("inf"))
    qk += 2.0794415416798359283  # log(8)
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

    acc += tl.dot(p, v)
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
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = k_block_col_idx * BLOCK_N
    # -- compute qk ----

    if LAST_K_BLOCK:
        k = tl.load(
            k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k
        )
    else:
        k = tl.load(k_ptrs + start_n * stride_kt)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    if LAST_K_BLOCK:
        qk += tl.where(offs_n[None, :] + start_n < seqlen_k, 0, -float("inf"))
    qk += 2.7725887222397812377  # log(16)

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.exp(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # update acc
    if LAST_K_BLOCK:
        v = tl.load(
            v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k
        )
    else:
        v = tl.load(v_ptrs + start_n * stride_vt)

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
    # if POOLING_BLOCK_N_8 < BLOCK_N:
    off_k_8 = offs_n_8[None, :] * stride_kn_8 + offs_d[:, None] * stride_kd_8
    off_v_8 = offs_n_8[:, None] * stride_vn_8 + offs_d[None, :] * stride_vd_8
    # else:
    off_k_8_merged = offs_n[None, :] * stride_kn_8 + offs_d[:, None] * stride_kd_8
    off_v_8_merged = offs_n[:, None] * stride_vn_8 + offs_d[None, :] * stride_vd_8

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs_1 = K + off_k_1
    k_ptrs_2 = K_2 + off_k_2
    k_ptrs_4 = K_4 + off_k_4
    k_ptrs_8 = K_8 + off_k_8
    k_ptrs_8_merged = K_8 + off_k_8_merged
    # k_ptrs_16 = K_16 + off_k_16
    v_ptrs_1 = V + off_v_1
    v_ptrs_2 = V_2 + off_v_2
    v_ptrs_4 = V_4 + off_v_4
    v_ptrs_8 = V_8 + off_v_8
    v_ptrs_8_merged = V_8 + off_v_8_merged
    # v_ptrs_16 = V_16 + off_v_16
    mask_ptrs = block_mask_ptr + start_m * stride_bmm

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)

    pooling_block_start = 0
    # k_block_end = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)
    pooling_block_end = tl.cdiv(N_CTX, POOLING_BLOCK_N)  # seq_len_k 是 K 的序列长度

    # block merging
    MERGE_STAGE = 0  # 0: no merge, 1: waiting for 1 block
    MERGE_INDEX = -1  # first block index

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
                            offs_n_1,
                            stride_kn,
                            stride_vn,
                            sm_scale,
                            N_CTX,
                            is_last_block,
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
                            offs_n,
                            stride_kn,
                            stride_vn,
                            sm_scale,
                            N_CTX,
                            LOG_1,
                            is_last_block,
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
                            offs_n_2,
                            stride_kn_2,
                            stride_vn_2,
                            sm_scale,
                            N_CTX_2,
                            is_last_block,
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
                            offs_n,
                            stride_kn_2,
                            stride_vn_2,
                            sm_scale,
                            N_CTX_2,
                            LOG_2,
                            is_last_block,
                            BLOCK_M,
                            BLOCK_N,
                            POOLING_BLOCK_N_2,
                        )
            else:
                if mask == 8:
                    if POOLING_BLOCK_N_8 <= BLOCK_N:
                        if MERGE_STAGE == 0:
                            # awaiting for 1 block
                            MERGE_STAGE = 1
                            MERGE_INDEX = pb_idx
                        else:  # MERGE_STAGE == 1
                            # merge 2 blocks
                            MERGE_STAGE = 0
                            merged_mask = offs_n < POOLING_BLOCK_N_8
                            offs_merged_k = (
                                tl.where(merged_mask, 0, pb_idx - MERGE_INDEX - 1)
                                * stride_kn_8
                                * POOLING_BLOCK_N_8
                            )[None, :]
                            offs_merged_v = (
                                tl.where(merged_mask, 0, pb_idx - MERGE_INDEX - 1)
                                * stride_vn_8
                                * POOLING_BLOCK_N_8
                            )[:, None]
                            # debug
                            # tl.device_print("pb_idx", pb_idx)
                            # tl.device_print("stride_vn_8", stride_vn_8)
                            # tl.device_print("offs_merged_v", offs_merged_v)
                            # tl.static_print("k_ptrs_8_merged", k_ptrs_8_merged.shape)
                            # tl.static_print("v_ptrs_8_merged", v_ptrs_8_merged.shape)

                            acc, l_i, m_i = _fwd_kernel_inner(
                                acc,
                                l_i,
                                m_i,
                                q,
                                MERGE_INDEX,
                                k_ptrs_8_merged + offs_merged_k,
                                v_ptrs_8_merged + offs_merged_v,
                                offs_n,
                                stride_kn_8,
                                stride_vn_8,
                                sm_scale,
                                N_CTX_8,
                                LOG_8,
                                is_last_block,
                                BLOCK_M,
                                BLOCK_N,
                                POOLING_BLOCK_N_8,
                            )
                            MERGE_INDEX = -1
                    else:
                        acc, l_i, m_i = _fwd_kernel_inner(
                            acc,
                            l_i,
                            m_i,
                            q,
                            pb_idx,
                            k_ptrs_8,
                            v_ptrs_8,
                            offs_n,
                            stride_kn_8,
                            stride_vn_8,
                            sm_scale,
                            N_CTX_8,
                            LOG_8,
                            is_last_block,
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
                            offs_n_4,
                            stride_kn_4,
                            stride_vn_4,
                            sm_scale,
                            N_CTX_4,
                            is_last_block,
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
                            offs_n,
                            stride_kn_4,
                            stride_vn_4,
                            sm_scale,
                            N_CTX_4,
                            LOG_4,
                            is_last_block,
                            BLOCK_M,
                            BLOCK_N,
                            POOLING_BLOCK_N_4,
                        )
    if MERGE_STAGE == 1:
        acc, l_i, m_i = _fwd_kernel_inner_8(
            acc,
            l_i,
            m_i,
            q,
            MERGE_INDEX,
            k_ptrs_8,
            v_ptrs_8,
            offs_n_8,
            stride_kn_8,
            stride_vn_8,
            sm_scale,
            N_CTX_8,
            True,
            BLOCK_M,
            POOLING_BLOCK_N_8,
        )
    if not INFERENCE:
        l_ptrs = L + off_hz * N_CTX + offs_m
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(l_ptrs, l_i, mask=offs_m < Q_LEN)
        tl.store(m_ptrs, m_i, mask=offs_m < Q_LEN)
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

    # Always set inference mode - no backward pass support
    L = m = None

    with torch.cuda.device(q.device.index): 
        _fwd_kernel[grid](
            q, k, k_2, k_4, k_8, k_16, v, v_2, v_4, v_8, v_16, sm_scale,
            block_sparse_mask,
            o,
            L, m,  # None for inference
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
            True,  # INFERENCE = True
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


class _sparse_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, block_sparse_dense, sm_scale):
        # shape constraints
        return _forward(ctx, q, k, v, block_sparse_dense, sm_scale)


def sparse_attention_factory(BLOCK_M=128, BLOCK_N=128, POOLING_BLOCK_N=128, **kwargs):
    class _sparse_attention_config(_sparse_attention):
        @staticmethod
        def forward(ctx, q, k, v, block_sparse_dense, sm_scale=None):
            sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
            # shape constraints
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
                **kwargs,
            )
    return _sparse_attention_config.apply

block_sparse_triton_fn = _sparse_attention.apply
