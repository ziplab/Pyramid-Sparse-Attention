import math
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from .kernels.psa_kernel_causal import (
    sparse_attention_factory,
)
from .utils.block_importance import estimate_block_importance,calc_k_similarity,calc_k_similarity_triton
from .utils.timer import time_logging_decorator
from .utils.psa_logger import PSALogger



@dataclass
class AttentionConfig:
    """Runtime configuration for pyramid adaptive sparse attention."""

    text_length: int = 512
    query_block: int = 128
    warmup_steps: int = 0

    mask_ratios: Dict[int, Tuple[float, float]] = field(
        default_factory=lambda: {
            1: (0.0, 0.05),
            2: (0.05, 0.15),
            4: (0.15, 0.25),
            8: (0.25, 0.5),
            0: (0.5, 1.0),
        }
    )
    mask_mode: str = "topk"
    importance_method: str = "xattn"  # pooling | xattn
    xattn_stride: int = 16
    xattn_chunk_size: int = 4096
    causal_main: bool = True

    # K similarity thresholds for pooling level selection
    sim_2x_threshold: float = 0.75
    sim_4x_threshold: float = 0.7
    sim_8x_threshold: float = 0.7

    def __post_init__(self) -> None:
        self.importance_method = self.importance_method.lower()
        if self.importance_method not in {"xattn"}:
            raise ValueError("importance_method must be 'xattn'")

        self.mask_mode = self.mask_mode.lower()
        if self.mask_mode not in {"topk", "energybound"}:
            raise ValueError("mask_mode must be 'topk' or 'energybound'")


def x_block_imp_estimate(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    block_size: int,
    stride: int,
    chunk_size: int,
    causal: bool,
) -> torch.Tensor:
    return estimate_block_importance(
        query,
        key,
        block_size=block_size,
        stride=stride,
        norm=1.0,
        softmax=True,
        chunk_size=chunk_size,
        select_mode="inverse",
        use_triton=True,
        causal=causal,
        kdb=1,
    )


@time_logging_decorator("transfer_attn_to_mask")
def transfer_attn_to_mask(
    attn: torch.Tensor,
    mask_ratios: Dict[int, Tuple[float, float]],
    text_length: int,
    mode: str,
    *,
    block_size: int,
    causal: bool,
) -> torch.Tensor:
    """Convert block-level attention scores to multi-scale pooling masks."""

    if not mask_ratios:
        raise ValueError("mask_ratios must not be empty")

    batch, heads, seq, _ = attn.shape
    device = attn.device
    mask = torch.zeros_like(attn, dtype=torch.int32)

    if mode not in {"topk", "energybound"}:
        raise ValueError("mode must be 'topk' or 'energybound'")

    sorted_weights, indices = torch.sort(attn, dim=-1, descending=True)

    row_indices = torch.arange(seq, device=device)
    if causal:
        valid_lengths = row_indices + 1
    else:
        valid_lengths = torch.full((seq,), seq, device=device, dtype=torch.long)

    if mode == "topk":
        position_range = torch.arange(seq, device=device)
        for value, (start_ratio, end_ratio) in mask_ratios.items():
            start_idx = (valid_lengths.float() * start_ratio).long().clamp(max=seq)
            end_idx = (valid_lengths.float() * end_ratio).long().clamp(max=seq)
            range_mask = (position_range.unsqueeze(0) >= start_idx.unsqueeze(1)) & (
                position_range.unsqueeze(0) < end_idx.unsqueeze(1)
            )
            update = torch.full_like(mask, value)
            mask.scatter_(
                -1,
                indices,
                torch.where(
                    range_mask.unsqueeze(0).unsqueeze(0),
                    update.gather(-1, indices),
                    mask.gather(-1, indices),
                ),
            )
    else:
        row_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        energy_ratio = torch.cumsum(sorted_weights, dim=-1) / row_sum
        prev_upper = 0.0
        for value, (start_ratio, end_ratio) in mask_ratios.items():
            lower = max(start_ratio, prev_upper)
            prev_upper = end_ratio
            range_mask = (energy_ratio > lower) & (energy_ratio <= end_ratio)
            update = torch.full_like(mask, value)
            mask.scatter_(
                -1,
                indices,
                torch.where(
                    range_mask,
                    update.gather(-1, indices),
                    mask.gather(-1, indices),
                ),
            )


    num_special_blocks = min((
        math.ceil(text_length / block_size) if text_length > 0 else 0
    ),mask.size(-1))
    if num_special_blocks:
        mask[:, :, :, -num_special_blocks:] = 1
        mask[:, :, -num_special_blocks:, :] = 1

    if causal:
        upper = torch.triu(
            torch.ones(seq, seq, device=device, dtype=torch.bool), diagonal=1
        )
        mask[:, :, upper] = 0

    diag = torch.arange(seq, device=device)
    mask[:, :, diag, diag] = 1
    mask[:,:,:,0]=1
    return mask


def calc_density(mask: torch.Tensor, causal: bool) -> Tuple[float, list]:
    density = torch.zeros_like(mask, dtype=torch.float32)
    non_zero = mask > 0
    density[non_zero] = 1.0 / mask[non_zero].float()

    seq = mask.size(-1)
    if causal:
        valid = torch.tril(
            torch.ones(seq, seq, device=mask.device, dtype=torch.float32), diagonal=0
        )
    else:
        valid = torch.ones(seq, seq, device=mask.device, dtype=torch.float32)

    valid = valid.view(1, 1, seq, seq).expand(mask.size(0), mask.size(1), -1, -1)
    density_valid = density * valid

    valid_counts = valid.sum().clamp(min=1.0)
    avg_density = density_valid.sum() / valid_counts

    per_head_counts = valid.sum(dim=(0, 2, 3)).clamp(min=1.0)
    per_head_density = (
        density_valid.sum(dim=(0, 2, 3)) / per_head_counts
    ).tolist()

    return avg_density.item(), per_head_density


@time_logging_decorator("adaptive_block_sparse_attn")
def adaptive_block_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: AttentionConfig,
    sparse_attention_fn,
) -> Tuple[torch.Tensor, float, list]:
    block_size = config.query_block

    with torch.no_grad():
        if config.importance_method == "xattn":
            attn_est = x_block_imp_estimate(
                q,
                k,
                block_size=block_size,
                stride=config.xattn_stride,
                chunk_size=config.xattn_chunk_size,
                causal=config.causal_main,
            )
        else:
            raise ValueError(
                f"Unknown importance_method: {config.importance_method}"
            )
        mask = transfer_attn_to_mask(
            attn_est,
            config.mask_ratios,
            config.text_length,
            config.mask_mode,
            block_size=block_size,
            causal=config.causal_main,
        )
        sim_mask = calc_k_similarity_triton(
            k,
            block_size,
            config,
        ).unsqueeze(-2).repeat(1,1,mask.size(-2),1) 
        #print("sim_mask shape:",sim_mask.shape,"sim_mask.min:",sim_mask.min().item(),"sim_mask.max:",sim_mask.max().item())
        mask = torch.min(sim_mask, mask)

    out = sparse_attention_fn(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        mask.contiguous(),
        None,
    )

    avg_density, per_head_density = calc_density(mask, config.causal_main)
    return out, 1 - avg_density, per_head_density


class PyramidAdaptiveBlockSparseAttnTrain(nn.Module):
    """Adaptive sparse attention with optional rearrangement and logging."""

    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        *,
        layer_idx: int = -1,
        log_dir: Optional[str] = None,
        log_prefix: str = "psa_sparsity",
        session_name: Optional[str] = None,
        **extra: object,
    ) -> None:
        super().__init__()
        self.config = config or AttentionConfig()
        self.layer_idx = layer_idx

        self.sparse_attention_fn = sparse_attention_factory(
            causal=self.config.causal_main,
        )

        self.sparsity_acc = 0.0
        self.sparsity_counter = 0

        if log_dir is None and "save_dir" in extra:
            log_dir = extra["save_dir"]

        # Initialize PSA logger
        self.logger: Optional[PSALogger] = None
        if log_dir is not None:
            self.logger = PSALogger(
                log_dir=log_dir,
                config=self.config,
                layer_idx=layer_idx,
                session_name=session_name or log_prefix,
            )

    def __del__(self) -> None:
        if self.logger is not None:
            self.logger.close()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        frame_num: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        *,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if layer_idx is not None:
            self.layer_idx = layer_idx
        self.sparsity_counter += 1

        in_warmup = self.sparsity_counter <= self.config.warmup_steps
        if in_warmup:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        out, sparsity, per_head_density = adaptive_block_sparse_attn(
            q,
            k,
            v,
            self.config,
            self.sparse_attention_fn,
        )

        self.sparsity_acc += sparsity

        # Log using PSA logger
        if self.logger is not None:
            self.logger.log_sparsity(
                layer_idx=self.layer_idx,
                sparsity=sparsity,
                per_head_density=per_head_density,
                sequence_length=q.size(2),
                batch_size=q.size(0),
                num_heads=q.size(1),
            )
            # Print progress at intervals
            self.logger.print_progress(self.layer_idx, interval=100)

        return out
