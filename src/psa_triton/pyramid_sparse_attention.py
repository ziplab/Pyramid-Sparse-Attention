import torch
from .kernels.attn_pooling_kernel import attn_with_pooling_optimized
from .utils.gilbert3d import gilbert3d
from torch.nn import functional as F
from .utils.tools import timeit
import torch.nn as nn
from .kernels.psa_kernel_legacy import sparse_attention_factory as psa_sparse_attention_legacy
from .kernels.psa_kernel import sparse_attention_factory as psa_sparse_attention
from .utils.transfer_attn_to_mask import transfer_attn_to_mask, calc_density, calc_density_newtype
from .utils.psa_logger import PSALogger
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from einops import rearrange
from .utils.rearranger import GilbertRearranger, SemanticAwareRearranger, STARearranger, HybridRearranger
import math
import os

# Import triton version of calc_k_similarity if available
from .kernels import calc_k_similarity_triton as _calc_k_similarity_triton, _USE_TRITON_KSIM
from .kernels import calc_k_similarity_pytorch as _calc_k_similarity_pytorch
if _USE_TRITON_KSIM:
    print("[PSA] Using Triton kernel for calc_k_similarity")
else:
    print("[PSA] Triton kernel for calc_k_similarity not available, using PyTorch fallback")


@dataclass
class AttentionConfig:
    """Attention configuration class for storing all configurable parameters"""
    # Video dimension parameters
    width: int = 45
    height: int = 30
    depth: int = 13
    text_length: int = 226

    # Attention-related parameters
    use_rearrange: bool = True
    block_m: int = 32
    block_n: int = 32
    tile_n: int = 32
    use_sim_mask: bool = True

    # Mask ratio configuration
    mask_ratios: Dict[int, Tuple[float, float]] = None
    mask_mode: str = 'topk'  # 'topk' or 'thresholdbound'
    warmup_steps: int = 12

    # STA-related parameters (for Sliding Tile Attention)
    tile_size: Optional[Tuple[int, int, int]] = None  # (T, H, W)
    rearrange_method: str = None  # 'Gilbert', 'STA', 'SemanticAware', 'Hybrid'

    # attn_impl - Reserved for selecting different kernel versions
    attn_impl: str = "new_mask_type"  # Optional values: "old_mask_type", "new_mask_type"

    # Logging control
    verbose: bool = False  # Whether to output log information
    enable_logging: bool = True  # Whether to enable PSA logger
    log_dir: str = "./psa_logs/"  # Logger save directory (relative path)

    # Sim Mask similarity thresholds (for adaptive pooling decisions)
    sim_2x_threshold: float = 0  # 2x pooling threshold
    sim_4x_threshold: float = 0   # 4x pooling threshold
    sim_8x_threshold: float = -1  # 8x pooling threshold

    def __post_init__(self):
        if self.mask_ratios is None:
            self.mask_ratios = {
                1: (0.0, 0.05),    # Top 5% - Full attention
                2: (0.05, 0.15),   # 5%-15% - 2x pooling
                4: (0.15, 0.25),   # 15%-25% - 4x pooling
                8: (0.25, 0.5),    # 25%-50% - 8x pooling
                0: (0.5, 1.0)      # 50%-100% - Skip
            }

        # Compatibility check: new_mask_type does not support use_sim_mask yet
        if self.attn_impl == "new_mask_type" and self.use_sim_mask:
            raise ValueError(
                "Incompatible configuration: 'new_mask_type' does not support 'use_sim_mask=True'.\n"
                "Please choose one of the following options:\n"
                "  1. Set 'use_sim_mask: false' to disable similarity mask\n"
                "  2. Use 'attn_impl: old_mask_type' with block_size config (m=128, n=128, tile_n=32)"
            )


def calc_k_similarity(k, blocksize, config):
    """
    Calculate the similarity of k within blocks.
    Uses Triton kernel if available, otherwise falls back to PyTorch.

    Args:
        k: [B, H, L, D] - Key tensor
        blocksize: int - Block size(k)
        config: AttentionConfig - Configuration containing similarity thresholds

    Returns:
        similarity: [B, H, k_chunked_num] - Similarity mask with values in {1, 2, 4, 8}
    """
    if _USE_TRITON_KSIM and k.is_cuda:
        return _calc_k_similarity_triton(k, blocksize, config)
    else:
        return _calc_k_similarity_pytorch(k, blocksize, config)
def pad_to_multiple(x, multiple):
    """
    Pad x on the sequence dimension (dim=2) to make its length a multiple of multiple.
    x: [B, H, L, D]
    """
    L = x.size(2)
    remainder = L % multiple
    if remainder != 0:
        pad_len = multiple - remainder
        # Pad the sequence dimension at the end (Note: F.pad parameter order - last two numbers correspond to left and right padding of dim=2)
        x = F.pad(x, (0, 0, 0, pad_len),mode='replicate')
    return x
def random_sample_tokens(x, block_size=64, sample_num=8):
    """
    Divide input x (shape: [B, H, L, D]) into blocks of block_size tokens,
    randomly sample sample_num tokens from each block.
    Requires L to be a multiple of block_size.
    Returns sampled result with shape [B, H, num_blocks * sample_num, D]
    """
    B, H, L, D = x.size()
    num_blocks = L // block_size
    # Reshape to [B, H, num_blocks, block_size, D]
    x_blocks = x.view(B, H, num_blocks, block_size, D)

    # Generate random numbers for each block and use topk to select sample_num random indices
    rand_vals = torch.rand(B, H, 1, block_size, device=x.device)
    _, indices = torch.topk(rand_vals, sample_num, dim=3)
    # indices shape: [B, H, num_blocks, sample_num]

    # Expand indices to align with the last dimension D of x_blocks
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, num_blocks, -1, D)
    # Use torch.gather to sample tokens from each block
    sampled = torch.gather(x_blocks, 3, indices_expanded)
    # Reshape back to [B, H, num_blocks * sample_num, D]
    sampled = sampled.view(B, H, num_blocks * sample_num, D)
    return sampled
def efficient_attn_with_pooling(q, k, v, config, num_keep_m=32, num_keep_n=32):
    """
    Compute downsampled attention pooling:
      - From q, k (shape: [B, H, seq, D]), randomly sample num_keep tokens every 64 tokens starting from token 224.
      - Compute attention on sampled q, k, using convolution for downsampling, achieving effect equivalent to 64×64 sum pooling on original attn.
      - Pad q and k on sequence dimension to ensure no loss of tail data during sampling.

    Parameters:
      config: AttentionConfig - Configuration object
      block_size: Number of tokens per block (fixed at 64)
      num_keep: Number of tokens to keep per block (default 8, customizable)
    """
    # Take the part starting from token 224
    q_ = q[:, :, :, :]
    k_ = k[:, :, :, :]

    # Pad on sequence dimension to multiple of block_size
    q_ = pad_to_multiple(q_, config.block_m)
    k_ = pad_to_multiple(k_, config.block_n)
    use_sim_mask = getattr(config, "use_sim_mask", True)
    sim_mask = None
    q_block_num = int(q_.shape[-2] // config.block_m)

    if use_sim_mask:
        sim_mask = calc_k_similarity(k_, config.block_n, config)
        # Expand sim_mask to [B, H, q_block_num, k_block_num]
        sim_mask = sim_mask.unsqueeze(-2).repeat(1, 1, q_block_num, 1)

    # Block and randomly sample q and k
    sampled_q = random_sample_tokens(q_, config.block_m, num_keep_m)  # [B, H, num_blocks*num_keep, D]
    sampled_k = random_sample_tokens(k_, config.block_n, num_keep_n)  # [B, H, num_blocks*num_keep, D]

    # Calculate effective pooling sequence length
    seqlen_pooling_m = sampled_q.size(2) // num_keep_m
    seqlen_pooling_n = sampled_k.size(2) // num_keep_n

    # Pad to 64 alignment (because pooling kernel tile size is 64)
    sampled_q_padding = pad_to_multiple(sampled_q, 64)
    sampled_k_padding = pad_to_multiple(sampled_k, 64)

    _, pooling = attn_with_pooling_optimized(
        sampled_q_padding, sampled_k_padding, v, False, 1.0 / (sampled_q.size(-1) ** 0.5),
        num_keep_m, num_keep_n
    )

    # Extract valid part
    pooling = pooling[:, :, :seqlen_pooling_m, :seqlen_pooling_n]

    return pooling, sim_mask  # sim_mask could be None when disabled
def adaptive_block_sparse_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               config: AttentionConfig, sparse_attention_fn,
                               compute_stats: bool = False) -> Tuple[torch.Tensor, float, list, torch.Tensor]:
    """
    Adaptive block sparse attention mechanism.
    Automatically creates block mask based on q, k, mask step does not track gradients.

    Args:
        q: Query tensor (batch_size, nheads, seqlen, d)
        k: Key tensor (batch_size, nheads, seqlen, d)
        v: Value tensor (batch_size, nheads, seqlen, d)
        config: Attention configuration parameters
        sparse_attention_fn: Sparse attention function
        compute_stats: Whether to compute density statistics (only when verbose=True or logging enabled)

    Returns:
        out: Output tensor (batch_size, nheads, seqlen, d)
        sparsity: Sparsity (0-1), 0.0 if compute_stats=False
        per_head_density: List of density values per head, empty list if compute_stats=False
        sim_mask: Similarity mask (for logging)
    """
    sm_scale = 1.0 / (q.size(-1) ** 0.5)
    block_size_m = config.block_m
    block_size_n = config.block_n

    # Disable gradient tracking for pooling and mask operations
    with torch.no_grad():
        pooling, sim_mask = efficient_attn_with_pooling(q, k, v, config, num_keep_m=block_size_m//4, num_keep_n=block_size_n//4)
        # Support both attn_impl and mask_mode methods
        if config.attn_impl == "old_mask_type":
            mask = transfer_attn_to_mask(pooling, config.mask_ratios, config.text_length, mode=config.mask_mode, blocksize=config.block_n, compute_tile=config.tile_n)
        elif config.attn_impl == "new_mask_type":
            # Use mask_mode parameter, supports topk and thresholdbound
            mode_map = {
                'topk': 'topk_newtype',
                'thresholdbound': 'thresholdbound_newtype'
            }
            mode = mode_map.get(config.mask_mode, 'topk_newtype')
            mask = transfer_attn_to_mask(pooling, config.mask_ratios, config.text_length, mode=mode, blocksize=config.block_n, compute_tile=config.tile_n)
        else:
            raise ValueError(f"Unknown attn_impl: {config.attn_impl}")
    use_sim_mask = getattr(config, "use_sim_mask", True)
    if use_sim_mask and sim_mask is not None:
        if sim_mask.dtype != mask.dtype:
            sim_mask = sim_mask.to(mask.dtype)
        fixed_mask = torch.min(sim_mask, mask)
    else:
        fixed_mask = mask
    out = sparse_attention_fn(q.contiguous(), k.contiguous(), v.contiguous(), fixed_mask, None)

    # Only compute density statistics when needed (verbose or logging enabled)
    if compute_stats:
        if config.attn_impl == "new_mask_type":
            avg_density, per_head_density = calc_density_newtype(fixed_mask)
        else:
            avg_density, per_head_density = calc_density(fixed_mask)
        sparsity = 1 - avg_density
    else:
        sparsity = 0.0
        per_head_density = []

    return out, sparsity, per_head_density, sim_mask


class PyramidSparseAttention(nn.Module):
    def __init__(self, config: AttentionConfig = None, inference_num=50, layer_num=42, model_type="wan"):
        super().__init__()
        # If config not provided, use default config
        if config is None:
            config = AttentionConfig()
        self.config = config
        self.model_type = model_type.lower()  # 'cogvideo' or 'wan'

        # Only initialize required rearranger based on config.rearrange_method
        if config.use_rearrange and config.rearrange_method == 'Gilbert':
            self.gilbert_rearranger = GilbertRearranger(
                config.width, config.height, config.depth, config.text_length
            )
        else:
            self.gilbert_rearranger = None

        if config.use_rearrange and config.rearrange_method == 'SemanticAware':
            self.semantic_aware_rearranger_list = [SemanticAwareRearranger(
                num_q_centroids=200, num_k_centroids=1000,
                kmeans_iter_init=50, kmeans_iter_step=2,
                layer_idx=i
            ) for i in range(layer_num)]
        else:
            self.semantic_aware_rearranger_list = None

        if config.use_rearrange and config.rearrange_method == 'STA':
            self.STARearranger = STARearranger(width=self.config.width, height=self.config.height, depth=self.config.depth,
                                             text_length=self.config.text_length, tile_size=self.config.tile_size)
        else:
            self.STARearranger = None

        if config.use_rearrange and config.rearrange_method == 'Hybrid':
            self.hybrid_rearranger_list = [HybridRearranger(
                width=self.config.width, height=self.config.height, depth=self.config.depth,
                text_length=self.config.text_length,
                num_k_centroids=1000, kmeans_iter_init=50, kmeans_iter_step=2,
                layer_idx=i
            ) for i in range(layer_num)]
        else:
            self.hybrid_rearranger_list = None

        self.sparsity_acc = 0.0  # Accumulator for sparsity sum
        self.sparsity_counter = 0  # Counter for number of updates
        self.use_rearrange = config.use_rearrange
        self.inference_num = inference_num
        self.layer_num = layer_num

        # Create sparse attention function (preserve multi-version kernel support)
        if config.attn_impl == "old_mask_type":
            self.sparse_attention_fn = psa_sparse_attention_legacy(config.block_m, config.tile_n, config.block_n)
        elif config.attn_impl == "new_mask_type":
            self.sparse_attention_fn = psa_sparse_attention(config.block_m, config.tile_n, config.block_n)

        # Initialize logger (optional)
        self.logger = None
        if config.enable_logging:
            self.logger = PSALogger(
                log_dir=config.log_dir,
                config=config,
                model_type=model_type,
                layer_num=layer_num,
                inference_num=inference_num,
            )
            print(f"[PSA] Logger enabled, saving to {config.log_dir}")
    def get_current_time(self):
        if self.model_type == "cogvideo" or self.model_type == "wan2.1_1.3b_4steps":
            time = (self.sparsity_counter // self.layer_num) % self.inference_num
        else:  # wan series
            time = (self.sparsity_counter // (2 * self.layer_num)) % self.inference_num
        return time
    def forward(self, q, k, v, layer_idx):
        # print(f"layer_idx: {layer_idx}")

        # Rearrange based on the selected method
        if self.use_rearrange:
            if self.config.rearrange_method == 'Gilbert':
                q_r, k_r, v_r = self.gilbert_rearranger.rearrange(q, k, v)
                q_sorted_indices = None
            elif self.config.rearrange_method == 'SemanticAware':
                q_r, k_r, v_r, q_sorted_indices = self.semantic_aware_rearranger_list[layer_idx].semantic_aware_permutation(q, k, v)
            elif self.config.rearrange_method == 'STA':
                q_r, k_r, v_r = self.STARearranger.rearrange(q, k, v)
                q_sorted_indices = None
            elif self.config.rearrange_method == 'Hybrid':
                q_r, k_r, v_r, q_sorted_indices = self.hybrid_rearranger_list[layer_idx].rearrange(q, k, v)
            else:
                raise ValueError(f"Unknown rearrange_method: {self.config.rearrange_method}")
        else:
            q_r = q
            k_r = k
            v_r = v
            q_sorted_indices = None

        # Compute block-sparse attention and get sparsity
        time = self.get_current_time()
        is_warmup = time < self.config.warmup_steps

        # Only compute stats if verbose or logging is enabled
        compute_stats = self.config.verbose or self.logger is not None

        if is_warmup:
            out_r = torch.nn.functional.scaled_dot_product_attention(q_r, k_r, v_r)
            sparsity = 0.0
            per_head_density = [1.0] * q_r.shape[1] if compute_stats else []
            sim_mask = None
        else:
            out_r, sparsity, per_head_density, sim_mask = adaptive_block_sparse_attn(
                q_r, k_r, v_r, self.config, self.sparse_attention_fn, compute_stats=compute_stats
            )
            # Update sparsity statistics only if computing stats
            if compute_stats:
                self.sparsity_acc += sparsity

        self.sparsity_counter += 1

        # Log sparsity if logger is enabled
        if self.logger is not None:
            self.logger.log_sparsity(
                layer_idx=layer_idx,
                sparsity=sparsity,
                per_head_density=per_head_density,
                sequence_length=q.shape[2],
                batch_size=q.shape[0],
                num_heads=q.shape[1],
                sim_mask=sim_mask,
                is_warmup=is_warmup,
            )

        # # Print average sparsity every 30 calls (only if verbose is enabled)
        # if self.config.verbose and self.sparsity_counter % 30 == 0:
        #     avg_sparsity = self.sparsity_acc / self.sparsity_counter
        #     print(f"avg_sparsity: {avg_sparsity},current_sparsity: {sparsity},layer_idx: {layer_idx}")

        # Reverse the arrangement based on the selected method
        if self.use_rearrange:
            if self.config.rearrange_method == 'Gilbert':
                out = self.gilbert_rearranger.reversed_rearrange(out_r)
            elif self.config.rearrange_method == 'SemanticAware':
                out = self.semantic_aware_rearranger_list[layer_idx].reverse_permutation(out_r, q_sorted_indices)
            elif self.config.rearrange_method == 'STA':
                out = self.STARearranger.reversed_rearrange(out_r)
            elif self.config.rearrange_method == 'Hybrid':
                out = self.hybrid_rearranger_list[layer_idx].reversed_rearrange(out_r, q_sorted_indices)
            else:
                raise ValueError(f"Unknown rearrange_method: {self.config.rearrange_method}")
        else:
            out = out_r

        # print(f"out shape: {out.shape}, out.dtype: {out.dtype}")
        return out

    def close_logger(self):
        """Close logger and write summary"""
        if self.logger is not None:
            self.logger.close()
            print("[PSA] Logger closed and summary written")

    def __del__(self):
        """Automatically close logger on destruction"""
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.close()
