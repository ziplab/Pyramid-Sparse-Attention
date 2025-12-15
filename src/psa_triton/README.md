# PSA Attention

Plug-and-play Pyramid Sparse Attention module.

## Usage

### Default Config

```python
from psa_triton import PSAAttention

psa = PSAAttention()
out = psa(q, k, v)  # q, k, v: [B, H, L, D]
```

### Custom Config

```python
from psa_triton import PSAAttention, PSAConfig

# Only override what you need, rest uses defaults
config = PSAConfig(mask_mode='thresholdbound')
psa = PSAAttention(config)
```

### Full Config Reference

```python
config = PSAConfig(
    # Block size configuration
    block_m=128,          # Query block size
    block_n=64,           # Key/Value block size (new_mask_type: 128/64/32)
    tile_n=32,            # Tile size for K/V processing

    # Mask ratio configuration
    mask_ratios={
        1: (0.0, 0.1),    # 10% full attention
        2: (0.1, 0.15),   # 5% with 2x pooling
        4: (0.15, 0.15),  # 0% with 4x pooling
        8: (0.15, 0.35),  # 20% with 8x pooling
        0: (0.35, 1.0),   # 65% skipped
    },
    mask_mode='topk',     # 'topk' or 'thresholdbound'

    # Kernel implementation
    attn_impl='new_mask_type',  # 'new_mask_type' or 'old_mask_type'

    # Similarity-based pooling constraint (only works with old_mask_type)
    use_sim_mask=False,
    sim_2x_threshold=0.0,
    sim_4x_threshold=0.0,
    sim_8x_threshold=-1.0,
)
psa = PSAAttention(config)
out = psa(q, k, v)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_m` | int | 128 | Query block size |
| `block_n` | int | 64 | Key/Value block size (new_mask_type: 128/64/32, old_mask_type: fixed 128) |
| `tile_n` | int | 32 | Tile size for K/V processing |
| `mask_ratios` | dict | See above | Sparsity distribution per pyramid level |
| `mask_mode` | str | `'topk'` | `'topk'` (fixed quota) or `'thresholdbound'` (dynamic) |
| `attn_impl` | str | `'new_mask_type'` | Kernel implementation: `'new_mask_type'` (default) or `'old_mask_type'` |
| `use_sim_mask` | bool | False | Enable similarity-based pooling constraint (requires `attn_impl='old_mask_type'`) |
| `sim_2x_threshold` | float | 0.0 | Similarity threshold for 2x pooling |
| `sim_4x_threshold` | float | 0.0 | Similarity threshold for 4x pooling |
| `sim_8x_threshold` | float | -1.0 | Similarity threshold for 8x pooling |
| `rearrange_method` | str | None | Token rearrangement method: `'Gilbert'`, `'STA'`, `'SemanticAware'`, `'Hybrid'` |

## attn_impl Selection

PSA provides two attention kernel implementations:

| Feature | `new_mask_type` | `old_mask_type` |
|---------|-----------------|-----------------|
| K block size (n) | Configurable: 128 / 64 / 32 | Fixed: 128 |
| use_sim_mask | Not supported | Supported |
| Causal mask | Not supported | Supported |
| Performance | Better | Slightly lower |

**Recommendation**: Use `new_mask_type` (default) for most scenarios. Use `old_mask_type` when you need sim_mask.

## Compatibility Notes

> **Important**: `attn_impl='new_mask_type'` is incompatible with `use_sim_mask=True`.

If both options are enabled, an error will be raised. Choose one of the following solutions:

1. **Use `new_mask_type`** (recommended, better performance):
   ```python
   config = PSAConfig(
       attn_impl='new_mask_type',
       use_sim_mask=False,
       block_n=64,  # Options: 128 / 64 / 32
   )
   ```

2. **Use `old_mask_type` + `use_sim_mask`**:
   ```python
   config = PSAConfig(
       attn_impl='old_mask_type',
       use_sim_mask=True,
       block_m=128,
       block_n=128,  # Must be 128
       tile_n=32,
   )
   ```

## Causal Attention Support

The legacy kernel (`psa_kernel_legacy.py`) supports causal masking. For causal attention use cases (e.g., vision-language models like Qwen2.5-VL), please refer to:

- Kernel: [`kernels/psa_kernel_legacy.py`](kernels/psa_kernel_legacy.py)
- Example: [`qwenvl2.5-example/`](../../qwenvl2.5-example/) for Qwen2.5-VL integration

> **Note**: Causal mask should NOT be used with token rearrangement (`rearrange_method`). After rearrangement, the lower triangular mask no longer represents true causal relationships.
