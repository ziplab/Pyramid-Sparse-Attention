# Attention Configuration Guide

[**English**](README.md) | [**中文**](README_zh.md)

This directory contains the PSA attention configuration file (`attention_config.yaml`). This guide explains the configuration structure and how to create or modify presets.

## Configuration Structure

Each model has its own configuration section with the following structure:

```yaml
ModelName:
  default_attention: preset_name     # Default preset to use
  video_scale:                       # Video dimension divisors (model-specific)
    width_divisor: 16
    height_divisor: 16
    depth_divisor: 4
  text_length: 226                   # Text token length (0 for text-free models)
  attention_configs:
    preset_name:                     # Your preset name
      type: psa                      # "psa" or "dense"
      description: "..."
      # PSA-specific parameters...
```

## Parameter Reference

### Model-Level Parameters

| Parameter | Description |
|-----------|-------------|
| `default_attention` | Default preset name to use when `--attention_preset` is not specified |
| `video_scale` | Divisors for converting video dimensions to token grid dimensions |
| `text_length` | Number of text tokens prepended to visual tokens (model-specific) |

### Preset-Level Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | string | `psa` for sparse attention, `dense` for full attention baseline |
| `description` | string | Human-readable description of the preset |
| `use_rearrange` | bool | Enable spatial token rearrangement (improves locality) |
| `use_sim_mask` | bool | Enable cosine similarity-based pooling constraint |
| `block_size.m` | int | Query block size |
| `block_size.n` | int | Key/Value block size |
| `block_size.tile_n` | int | Hardware tile size for K/V processing |
| `mask_ratios` | dict | Cumulative importance thresholds per pyramid level |
| `mask_mode` | string | `thresholdbound` or `topk` (see below) |
| `attn_impl` | string | Attention implementation: `new_mask_type` or `old_mask_type` |
| `tile_size` | list | 3D tile dimensions `[depth, height, width]` for rearrangement |
| `warmup_steps` | int | Initial steps using dense attention before switching to sparse |
| `rearrange_method` | string | Token rearrangement algorithm (e.g., `Gilbert`) |
| `verbose` | bool | Enable verbose logging during inference |
| `sim_thresholds` | dict | Cosine similarity thresholds for pooling levels |

## Understanding mask_ratios

The `mask_ratios` parameter controls how query-key block pairs are assigned to different pyramid levels based on their importance scores.

### thresholdbound Mode

In `thresholdbound` mode, `mask_ratios` defines cumulative importance score thresholds:

```yaml
mask_ratios:
  1: [0.0, 0.4]    # Level 1: top 0-40% importance → full resolution
  2: [0.4, 0.5]    # Level 2: 40-50% → 2x pooled
  4: [0.5, 0.6]    # Level 4: 50-60% → 4x pooled
  8: [0.6, 0.8]    # Level 8: 60-80% → 8x pooled
  0: [0.8, 1.0]    # Level 0: 80-100% → skip attention
```

The key is the pyramid level:
- **Level 1**: Full resolution KV (highest quality)
- **Level 2/4/8**: Progressively pooled KV (2x/4x/8x average pooling)
- **Level 0**: Skip attention entirely

### topk Mode

In `topk` mode, `mask_ratios` defines fixed quotas (percentages) for each level:

```yaml
mask_ratios:
  1: [0.0, 0.1]    # 10% of blocks at full resolution
  2: [0.1, 0.15]   # 5% of blocks at 2x pooled
  4: [0.15, 0.15]  # 0% at 4x pooled (skipped)
  8: [0.15, 0.35]  # 20% of blocks at 8x pooled
  0: [0.35, 1.0]   # 65% of blocks skipped
```

### Choosing Between Modes

- **`thresholdbound`**: Dynamically adjusts allocation based on importance distribution. Generally achieves better similarity metrics (PSNR/SSIM/LPIPS).
- **`topk`**: Fixed per-level quotas. Produces more stable visual results under extremely high sparsity.

## Creating a New Preset

1. Open `configs/attention_config.yaml`
2. Find your target model section (e.g., `CogVideo_5b`)
3. Add a new preset under `attention_configs`:

```yaml
CogVideo_5b:
  attention_configs:
    # ... existing presets ...
    
    my_custom_preset:
      type: psa
      description: "My custom PSA configuration"
      use_rearrange: true
      use_sim_mask: false
      block_size:
        m: 128
        n: 64
        tile_n: 32
      mask_ratios:
        1: [0.0, 0.5]
        2: [0.5, 0.7]
        4: [0.7, 0.9]
        0: [0.9, 1.0]
      mask_mode: thresholdbound
      attn_impl: new_mask_type
      tile_size: [2, 8, 8]
      warmup_steps: 10
      rearrange_method: Gilbert
      verbose: false
      sim_thresholds:
        x2: 0.7
        x4: 0.65
        x8: 0.6
```

4. Use your preset via command line:

```bash
python examples/inference/cogvideo/cogvideo_5b.py \
    --prompt "your prompt" \
    --use_psa \
    --attention_preset my_custom_preset
```
