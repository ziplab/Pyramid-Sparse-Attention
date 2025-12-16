"""Qwen2.5-VL 推理模块"""
from .inference import QwenVLInference

__all__ = ["QwenVLInference"]

# PSA attention is available via:
# from src.attention import replace_psa_attention_qwen2vl, AttentionConfig
