"""
Qwen2.5-VL Attention with PSA (Pyramid Adaptive Block Sparse Attention)
"""
from typing import Optional
import types
import os
from datetime import datetime, timezone, timedelta

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging

from .PSA_casual import PyramidAdaptiveBlockSparseAttnTrain, AttentionConfig

logger = logging.get_logger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for GQA."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    Applies Multimodal Rotary Position Embedding to Q and K.

    Qwen2.5-VL uses 3D RoPE for vision (temporal, height, width) and 1D for text.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def replace_psa_attention_qwen2vl(
    target,
    attention_config: Optional[AttentionConfig] = None,
    *,
    log_dir: Optional[str] = None,
):
    """
    Replace Qwen2.5-VL attention layers with PSA sparse attention.

    Args:
        target: transformers Pipeline or model instance
        attention_config: PSA AttentionConfig (optional)
        log_dir: Directory for PSA logs

    Returns:
        The same object with replaced attention
    """
    default_config = AttentionConfig(
        mask_mode="energybound",
        mask_ratios={
            1: (0.0, 0.7),
            2: (0.7, 0.8),
            4: (0.8, 0.9),
            8: (0.9, 0.9),
            0: (0.9, 1.0),
        },
        importance_method="xattn",
        causal_main=True,
        xattn_stride=8,
        warmup_steps=0,
    )
    attention_config = attention_config or default_config

    # Create log directory
    tz = timezone(timedelta(hours=8))
    timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_dir or "./logs/PSA_Log"
    log_dir = os.path.join(log_dir, f"log_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 60)
    print("PSA Attention Configuration for Qwen2.5-VL")
    print("=" * 60)
    print(f"mask_mode: {attention_config.mask_mode}")
    print(f"mask_ratios: {attention_config.mask_ratios}")
    print(f"importance_method: {attention_config.importance_method}")
    print(f"xattn_stride: {attention_config.xattn_stride}")
    print(f"causal_main: {attention_config.causal_main}")
    print(f"log_dir: {log_dir}")
    print("=" * 60 + "\n")

    # Extract model from pipeline if needed
    from transformers import Pipeline
    if isinstance(target, Pipeline):
        model = target.model
        is_pipeline = True
    else:
        model = target
        is_pipeline = False

    # Find decoder layers
    def _resolve_layers(module):
        # Qwen2_5_VLForConditionalGeneration -> model.model.language_model.layers
        if hasattr(module, "model"):
            inner = module.model
            if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
                return inner.language_model.layers
            if hasattr(inner, "model") and hasattr(inner.model, "layers"):
                return inner.model.layers
        # Qwen2_5_VLModel -> language_model.layers
        if hasattr(module, "language_model") and hasattr(module.language_model, "layers"):
            return module.language_model.layers
        if hasattr(module, "layers"):
            return module.layers
        raise AttributeError("Unsupported model structure for PSA replacement.")

    layers = _resolve_layers(model)
    num_layers = len(layers)

    print(f"Model: {type(model).__name__}")
    print(f"Detected {num_layers} decoder layers")
    print("Replacing attention layers...")

    # Create shared PSA module
    shared_sparse = PyramidAdaptiveBlockSparseAttnTrain(
        config=attention_config,
        layer_idx=-1,
        log_dir=log_dir,
    )

    def _make_psa_forward(original_forward, layer_index):
        def psa_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
        ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
            # Decode stage (seq_len=1) or PSA disabled: use original forward
            if hidden_states.size(1) == 1 or not getattr(self, "use_psa_attention", True):
                return original_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            # Get position embeddings
            if position_embeddings is None:
                if position_ids is None:
                    return original_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )
                cos, sin = self.rotary_emb(hidden_states, position_ids)
            else:
                cos, sin = position_embeddings

            bsz, q_len, _ = hidden_states.size()

            # Project Q, K, V
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            # Apply multimodal RoPE
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
            )

            # Update KV cache if present
            if past_key_values is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_values.update(
                    key_states, value_states, layer_index, cache_kwargs
                )

            # Expand KV for GQA
            if getattr(self, "num_key_value_groups", 1) > 1:
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

            # Call PSA sparse attention
            sparse_module = getattr(self, "_psa_sparse_module", None)
            if sparse_module is None:
                raise RuntimeError("PSA sparse attention module not initialized.")

            sparse_out = sparse_module(
                query_states,
                key_states,
                value_states,
                layer_idx=layer_index,
            )

            attn_output = sparse_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)

            return attn_output, None

        return psa_forward

    # Replace attention in each layer
    for layer_idx, layer in enumerate(layers):
        attn_module = layer.self_attn

        # Save original forward
        if not hasattr(attn_module, "_original_forward_psa"):
            attn_module._original_forward_psa = attn_module.forward

        # Set up PSA
        attn_module.layer_idx = getattr(attn_module, "layer_idx", layer_idx)
        attn_module._psa_sparse_module = shared_sparse
        attn_module.use_psa_attention = True

        # Replace forward
        psa_forward = _make_psa_forward(attn_module._original_forward_psa, layer_idx)
        attn_module.forward = types.MethodType(psa_forward, attn_module)
        attn_module._psa_forward_installed = True

    print(f"Replaced {num_layers} layers with PSA attention\n")

    return target if is_pipeline else model


def verify_attention_replacement(target):
    """
    Verify that attention has been successfully replaced.

    Args:
        target: Pipeline or model object

    Returns:
        bool: True if all checked layers are replaced
    """
    print("=" * 60)
    print("Verifying Attention Replacement")
    print("=" * 60)

    if hasattr(target, "model"):
        model = target.model
    else:
        model = target

    # Find layers
    try:
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            layers = model.model.language_model.layers
        elif hasattr(model, "language_model"):
            layers = model.language_model.layers
        else:
            layers = model.layers
    except AttributeError:
        print("Could not find decoder layers")
        return False

    # Check first and last few layers
    check_indices = [0, 1, 2, len(layers) - 3, len(layers) - 2, len(layers) - 1]

    all_replaced = True
    for idx in check_indices:
        if idx < 0 or idx >= len(layers):
            continue

        attn = layers[idx].self_attn
        has_psa = getattr(attn, "_psa_forward_installed", False) and getattr(
            attn, "use_psa_attention", False
        )

        status = "PSA" if has_psa else "Original"
        print(f"Layer {idx:2d}: {status}")

        if not has_psa:
            all_replaced = False

    print("=" * 60)
    if all_replaced:
        print("All checked layers replaced successfully")
    else:
        print("Some layers still use original attention")
    print("=" * 60 + "\n")

    return all_replaced
