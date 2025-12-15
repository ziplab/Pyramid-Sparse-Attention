"""
PSA Attention - Plug-and-play Pyramid Sparse Attention

A simplified, drop-in replacement for standard attention mechanisms.
Just pass q, k, v and get sparse attention output.

Usage:
    from psa_triton import PSAAttention, PSAConfig
    
    # Simplest usage - default config
    psa = PSAAttention()
    out = psa(q, k, v)
    
    # Custom config
    config = PSAConfig(
        mask_ratios={1: (0, 0.3), 2: (0.3, 0.5), 4: (0.5, 0.7), 0: (0.7, 1.0)},
        mask_mode='thresholdbound',
    )
    psa = PSAAttention(config)
    out = psa(q, k, v)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from .kernels.attn_pooling_kernel import attn_with_pooling_optimized
from .kernels.psa_kernel import sparse_attention_factory as psa_sparse_attention
from .kernels.psa_kernel_legacy import sparse_attention_factory as psa_sparse_attention_legacy
from .kernels import calc_k_similarity_triton as _calc_k_similarity_triton, _USE_TRITON_KSIM
from .utils.transfer_attn_to_mask import transfer_attn_to_mask


@dataclass
class PSAConfig:
    """
    Configuration for PSA Attention.

    All parameters have sensible defaults for ~85% sparsity with good quality.
    """
    # Block size configuration
    block_m: int = 128          # Query block size
    block_n: int = 64           # Key/Value block size (new_mask_type: 128/64/32, old_mask_type: fixed 128)
    tile_n: int = 32            # Tile size for K/V processing

    # Mask ratio configuration - controls sparsity distribution
    # Key: pyramid level (1=full, 2/4/8=pooled, 0=skip)
    # Value: (lower_bound, upper_bound) cumulative ratio
    mask_ratios: Dict[int, Tuple[float, float]] = field(default_factory=lambda: {
        1: (0.0, 0.1),    # 10% - Full resolution attention
        2: (0.1, 0.15),   # 5% - 2x pooled KV
        4: (0.15, 0.15),  # 0% - 4x pooled KV (skip)
        8: (0.15, 0.35),  # 20% - 8x pooled KV
        0: (0.35, 1.0),   # 65% - Skip attention
    })

    # Mask generation mode: 'topk' (fixed quota) or 'thresholdbound' (dynamic)
    mask_mode: str = 'topk'

    # Similarity-based pooling constraint (adaptive pooling decisions)
    # Note: only works with attn_impl='old_mask_type'
    use_sim_mask: bool = False
    sim_2x_threshold: float = 0.0   # 2x pooling threshold
    sim_4x_threshold: float = 0.0   # 4x pooling threshold
    sim_8x_threshold: float = -1.0  # 8x pooling threshold (disabled by default)

    # Rearrangement method for token ordering
    rearrange_method: str = None  # 'Gilbert', 'STA', 'SemanticAware', 'Hybrid'

    # Kernel implementation - selects mask format
    # "new_mask_type": uses separator-based mask format (default, more efficient, block_n: 128/64/32)
    # "old_mask_type": uses legacy mask format (required for use_sim_mask=True)
    attn_impl: str = "new_mask_type"

    def __post_init__(self):
        """Validate configuration and check compatibility."""
        # Compatibility check: new_mask_type does not support use_sim_mask
        if self.attn_impl == "new_mask_type" and self.use_sim_mask:
            raise ValueError(
                "Incompatible configuration: 'new_mask_type' does not support 'use_sim_mask=True'.\n"
                "Please choose one of the following options:\n"
                "  1. Set 'use_sim_mask=False' to disable similarity mask\n"
                "  2. Use 'attn_impl=\"old_mask_type\"' with block_size config (m=128, n=128, tile_n=32)"
            )

        # Validate block sizes for old_mask_type
        if self.attn_impl == "old_mask_type":
            if self.block_m != 128 or self.block_n != 128:
                raise ValueError(
                    f"attn_impl='old_mask_type' only supports block_m=128 and block_n=128.\n"
                    f"Got block_m={self.block_m}, block_n={self.block_n}.\n"
                    f"Use attn_impl='new_mask_type' for flexible block sizes (128/64/32)."
                )

        # Validate block_n for new_mask_type
        if self.attn_impl == "new_mask_type":
            if self.block_n not in [128, 64, 32]:
                raise ValueError(
                    f"attn_impl='new_mask_type' only supports block_n in [128, 64, 32].\n"
                    f"Got block_n={self.block_n}."
                )


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad sequence dimension to be a multiple of given value."""
    L = x.size(2)
    remainder = L % multiple
    if remainder != 0:
        pad_len = multiple - remainder
        x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
    return x


def _random_sample_tokens(x: torch.Tensor, block_size: int, sample_num: int) -> torch.Tensor:
    """
    Sample tokens from each block for efficient attention pooling estimation.
    
    Args:
        x: [B, H, L, D] input tensor
        block_size: Size of each block
        sample_num: Number of tokens to sample per block
    
    Returns:
        Sampled tensor [B, H, num_blocks * sample_num, D]
    """
    B, H, L, D = x.size()
    num_blocks = L // block_size
    x_blocks = x.view(B, H, num_blocks, block_size, D)
    
    rand_vals = torch.rand(B, H, 1, block_size, device=x.device)
    _, indices = torch.topk(rand_vals, sample_num, dim=3)
    
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, num_blocks, -1, D)
    sampled = torch.gather(x_blocks, 3, indices_expanded)
    return sampled.view(B, H, num_blocks * sample_num, D)


def _calc_k_similarity_pytorch(k: torch.Tensor, blocksize: int, config: PSAConfig) -> torch.Tensor:
    """Calculate similarity-based pooling constraints for K blocks (PyTorch fallback)."""
    k_chunked_num = k.shape[-2] // blocksize
    k_chunked = k[:, :, :k_chunked_num * blocksize, :].reshape(
        k.shape[0], k.shape[1], k_chunked_num, blocksize, k.shape[-1]
    )

    # Cosine similarity between adjacent tokens
    similarity_2 = F.cosine_similarity(
        k_chunked[..., ::2, :], k_chunked[..., 1::2, :], dim=-1
    ).mean(dim=-1)
    similarity_4 = F.cosine_similarity(
        k_chunked[..., 0::4, :], k_chunked[..., 3::4, :], dim=-1
    ).mean(dim=-1)
    similarity_8 = F.cosine_similarity(
        k_chunked[..., 0::8, :], k_chunked[..., 7::8, :], dim=-1
    ).mean(dim=-1)

    # Build mask based on thresholds
    sim_2_mask = 2 * (similarity_2 > config.sim_2x_threshold)
    sim_4_mask = 4 * (similarity_4 > config.sim_4x_threshold)
    sim_8_mask = 8 * (similarity_8 > config.sim_8x_threshold)
    one_tensor = torch.ones_like(sim_2_mask)

    return torch.maximum(one_tensor, torch.maximum(sim_2_mask, torch.maximum(sim_4_mask, sim_8_mask)))


def _calc_k_similarity(k: torch.Tensor, blocksize: int, config: PSAConfig) -> torch.Tensor:
    """
    Calculate similarity-based pooling constraints for K blocks.
    Uses Triton kernel if available, otherwise falls back to PyTorch.
    """
    if _USE_TRITON_KSIM and k.is_cuda:
        return _calc_k_similarity_triton(k, blocksize, config)
    else:
        return _calc_k_similarity_pytorch(k, blocksize, config)


def _compute_attention_pooling(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    config: PSAConfig
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute efficient attention pooling for mask generation.
    
    Returns:
        pooling: Block-level attention scores
        sim_mask: Similarity-based pooling constraint (or None)
    """
    num_keep_m = config.block_m // 4
    num_keep_n = config.block_n // 4
    
    # Pad to block size
    q_ = _pad_to_multiple(q, config.block_m)
    k_ = _pad_to_multiple(k, config.block_n)
    
    q_block_num = q_.shape[-2] // config.block_m
    
    # Compute similarity mask if enabled
    sim_mask = None
    if config.use_sim_mask:
        sim_mask = _calc_k_similarity(k_, config.block_n, config)
        sim_mask = sim_mask.unsqueeze(-2).repeat(1, 1, q_block_num, 1)
    
    # Sample tokens for efficient pooling
    sampled_q = _random_sample_tokens(q_, config.block_m, num_keep_m)
    sampled_k = _random_sample_tokens(k_, config.block_n, num_keep_n)
    
    seqlen_pooling_m = sampled_q.size(2) // num_keep_m
    seqlen_pooling_n = sampled_k.size(2) // num_keep_n
    
    # Pad to 64 alignment for pooling kernel
    sampled_q_pad = _pad_to_multiple(sampled_q, 64)
    sampled_k_pad = _pad_to_multiple(sampled_k, 64)
    
    _, pooling = attn_with_pooling_optimized(
        sampled_q_pad, sampled_k_pad, v, False, 
        1.0 / (sampled_q.size(-1) ** 0.5),
        num_keep_m, num_keep_n
    )
    
    return pooling[:, :, :seqlen_pooling_m, :seqlen_pooling_n], sim_mask


class PSAAttention(nn.Module):
    """
    Plug-and-play Pyramid Sparse Attention.

    Drop-in replacement for standard attention - just pass q, k, v.

    Example:
        psa = PSAAttention()
        out = psa(q, k, v)  # q, k, v: [B, H, L, D]
    """

    def __init__(self, config: PSAConfig = None):
        """
        Initialize PSA Attention.

        Args:
            config: PSAConfig object. If None, uses default config (~85% sparsity).
        """
        super().__init__()
        self.config = config if config is not None else PSAConfig()

        # Create sparse attention kernel based on attn_impl
        if self.config.attn_impl == "old_mask_type":
            self.sparse_attention_fn = psa_sparse_attention_legacy(
                self.config.block_m,
                self.config.tile_n,
                self.config.block_n
            )
        else:  # new_mask_type
            self.sparse_attention_fn = psa_sparse_attention(
                self.config.block_m,
                self.config.tile_n,
                self.config.block_n
            )
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute pyramid sparse attention.

        Args:
            q: Query tensor [B, H, L, D]
            k: Key tensor [B, H, L, D]
            v: Value tensor [B, H, L, D]

        Returns:
            Output tensor [B, H, L, D]
        """
        config = self.config

        # Compute attention pooling and mask (no gradient tracking)
        with torch.no_grad():
            pooling, sim_mask = _compute_attention_pooling(q, k, v, config)

            # Determine mask mode based on attn_impl setting
            if config.attn_impl == "old_mask_type":
                mode = config.mask_mode  # 'topk' or 'thresholdbound'
            else:
                mode = f"{config.mask_mode}_newtype"  # 'topk_newtype' or 'thresholdbound_newtype'

            # Generate sparse mask
            mask = transfer_attn_to_mask(
                pooling,
                config.mask_ratios,
                text_length=0,  # No text-specific handling
                mode=mode,
                blocksize=config.block_n,
                compute_tile=config.tile_n
            )

            # Apply similarity constraint if enabled (only for old_mask_type)
            if config.use_sim_mask and sim_mask is not None:
                if sim_mask.dtype != mask.dtype:
                    sim_mask = sim_mask.to(mask.dtype)
                mask = torch.min(sim_mask, mask)

        # Compute sparse attention
        out = self.sparse_attention_fn(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            mask,
            None
        )

        return out


# Convenience function for functional-style usage
def psa_attention(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    *,
    config: PSAConfig = None,
    mask_ratios: Dict[int, Tuple[float, float]] = None,
    mask_mode: str = None,
    block_m: int = None,
    block_n: int = None,
) -> torch.Tensor:
    """
    Functional interface for PSA attention.
    
    Args:
        q: Query tensor [B, H, L, D]
        k: Key tensor [B, H, L, D]
        v: Value tensor [B, H, L, D]
        config: PSAConfig object (optional)
        mask_ratios: Override mask ratios (optional)
        mask_mode: Override mask mode (optional)
        block_m: Override query block size (optional)
        block_n: Override key/value block size (optional)
    
    Returns:
        Output tensor [B, H, L, D]
    
    Example:
        # Simplest usage
        out = psa_attention(q, k, v)
        
        # With custom sparsity
        out = psa_attention(q, k, v, mask_ratios={
            1: (0, 0.2), 2: (0.2, 0.4), 4: (0.4, 0.6), 0: (0.6, 1.0)
        })
    """
    # Build config from parameters
    if config is None:
        config = PSAConfig()
    
    # Override with explicit parameters
    if mask_ratios is not None:
        config.mask_ratios = mask_ratios
    if mask_mode is not None:
        config.mask_mode = mask_mode
    if block_m is not None:
        config.block_m = block_m
    if block_n is not None:
        config.block_n = block_n
        config.tile_n = min(32, block_n)
    
    # Create and run attention
    psa = PSAAttention(config)
    return psa(q, k, v)
