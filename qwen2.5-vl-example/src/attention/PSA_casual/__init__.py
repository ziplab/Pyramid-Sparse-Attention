"""
Pyramid Adaptive Block Sparse Attention (PSA) implementation.

This module provides efficient sparse attention mechanisms for vision-language models,
particularly optimized for Qwen2.5-VL.
"""

from .PyramidAdaptiveBlockSparseAttn import (
    AttentionConfig,
    PyramidAdaptiveBlockSparseAttnTrain,
    adaptive_block_sparse_attn,
)

from .utils.block_importance import estimate_block_importance

__all__ = [
    "AttentionConfig",
    "PyramidAdaptiveBlockSparseAttnTrain",
    "adaptive_block_sparse_attn",
    "estimate_block_importance",
]
