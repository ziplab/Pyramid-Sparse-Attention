# Pyramid Sparse Attention (PSA)

[**English**](README.md) | [**中文**](README_zh.md)

**Website:** [http://ziplab.co/PSA](http://ziplab.co/PSA) | **Paper:** [arXiv](https://arxiv.org/abs/2512.04025)

Official PyTorch implementation of [PSA: Pyramid Sparse Attention for Efficient Video Understanding and Generation](https://arxiv.org/abs/2512.04025).

<p align="center">
  <img src="figures/prompt007comparison.jpg" width="100%">
</p>

<p align="center"><em>Visual comparison of sparse attention methods at similar sparsity levels (~90%). PSA maintains visual fidelity close to full attention while other methods show noticeable artifacts.</em></p>

> **Note:** This release focuses on **inference-only** with **bidirectional attention**. Support for causal attention masks and backward propagation (training) is still under optimization and will be released in a future update.

## Installation

### Using uv (Recommended)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> For best performance, we recommend using PyTorch nightly version.

## Download Weights

### CogVideoX-5B LoRA (4-step)

```bash
huggingface-cli download GYP666/BLADE cogvideox-5b-psa-lora/pytorch_lora_weights.safetensors --local-dir ./weights
```

**Note:** After downloading, update the `lora_path` in `examples/configs/model_configs.py` to point to your weights directory.

## Quick Start

### CogVideoX1.5-5B

```bash
python examples/inference/cogvideo/cogvideo_5b.py \
    --model cogvideo1.5_5b \
    --prompt "your prompt here" \
    --use_psa
```

### Wan2.1-1.3B

```bash
python examples/inference/wan21/wan21_1.3b.py \
    --prompt "your prompt here" \
    --use_psa --no_warmup
```

For more inference examples, see [examples/README.md](examples/README.md).

## Attention Configuration

The PSA behavior is configured via `configs/attention_config.yaml`. Each model has its own configuration section.

### Configuration Structure

```yaml
ModelName:
  default_attention: psa_balanced    # Default preset to use
  video_scale:                       # Video dimension divisors
    width_divisor: 16
    height_divisor: 16
    depth_divisor: 4
  text_length: 226                   # Text token length (model-specific)
  attention_configs:
    preset_name:                     # e.g., psa_balanced, psa_4steps, baseline
      type: psa                      # "psa" for sparse attention, "dense" for baseline
      description: "..."
      # PSA-specific parameters below
```

### Key Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `type` | Attention type | `psa` (sparse) or `dense` (baseline) |
| `use_rearrange` | Enable spatial rearrangement | `true` / `false` |
| `block_size.m` | Query block size | `128` |
| `block_size.n` | Key/Value block size | `32`, `128` |
| `block_size.tile_n` | Tile size for K/V | `32` |
| `mask_ratios` | Cumulative importance thresholds per pyramid level | See below |
| `mask_mode` | Mask generation strategy | `thresholdbound`, `topk` (see below) |
| `warmup_steps` | Initial steps using dense attention before switching to sparse | `0`, `12`, `15` |
| `rearrange_method` | Token rearrangement algorithm | `Gilbert` |

### Mask Ratios Explained

The `mask_ratios` parameter defines cumulative importance score thresholds for assigning query-key block pairs to different pyramid levels. The following example uses the `thresholdbound` mask mode:

```yaml
mask_ratios:
  1: [0.0, 0.4]    # Level 1: cumulative score in range [0%, 40%] → full resolution KV
  2: [0.4, 0.5]    # Level 2: cumulative score in range [40%, 50%] → 2x pooled KV
  4: [0.5, 0.6]    # Level 4: cumulative score in range [50%, 60%] → 4x pooled KV
  8: [0.6, 0.8]    # Level 8: cumulative score in range [60%, 80%] → 8x pooled KV
  0: [0.8, 1.0]    # Level 0: cumulative score in range [80%, 100%] → skip attention
```

- **Level 1**: Full resolution attention (highest quality, for most important KV blocks)
- **Level 2/4/8**: Progressively pooled KV representations (coarser levels for less important blocks)
- **Level 0**: Attention skipped entirely (for least important blocks)

### Mask Mode

- **`thresholdbound`**: Threshold-based assignment using cumulative importance scores. Generally achieves better similarity metrics (PSNR/SSIM/LPIPS).
- **`topk`**: Quantile-based assignment with fixed per-level quotas. Produces more stable visual results under extremely high sparsity.

### Customizing Configuration

1. Edit `configs/attention_config.yaml`
2. Add a new preset under the target model's `attention_configs`
3. Use it via `--attention_preset your_preset_name`

Example custom preset:
```yaml
CogVideo_5b:
  attention_configs:
    my_custom_preset:
      type: psa
      description: "My custom PSA configuration"
      use_rearrange: true
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
      warmup_steps: 10
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{li2025psapyramidsparseattention,
      title={PSA: Pyramid Sparse Attention for Efficient Video Understanding and Generation}, 
      author={Xiaolong Li and Youping Gu and Xi Lin and Weijie Wang and Bohan Zhuang},
      year={2025},
      eprint={2512.04025},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.04025}, 
}
```
