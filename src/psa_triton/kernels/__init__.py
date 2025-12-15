"""PSA Triton Kernels"""

from .psa_kernel import sparse_attention_factory as psa_sparse_attention
from .psa_kernel_legacy import sparse_attention_factory as psa_sparse_attention_legacy
from .attn_pooling_kernel import attn_with_pooling_optimized

# Try to import triton version of calc_k_similarity
_USE_TRITON_KSIM = False
calc_k_similarity_triton = None

try:
    from .calc_ksim_kernel import calc_k_similarity_triton, calc_k_similarity_pytorch
    _USE_TRITON_KSIM = True
except ImportError:
    pass

__all__ = [
    "psa_sparse_attention",
    "psa_sparse_attention_legacy",
    "attn_with_pooling_optimized",
    "calc_k_similarity_triton",
    "calc_k_similarity_pytorch",
    "_USE_TRITON_KSIM",
]
