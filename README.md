# Pyramid Sparse Attention (PSA)

[**English**](README.md) | [**中文**](README_zh.md)

**Website:** [http://ziplab.co/PSA](http://ziplab.co/PSA) | **Paper:** [arXiv](https://arxiv.org/abs/2512.04025)

Official PyTorch implementation of [PSA: Pyramid Sparse Attention for Efficient Video Understanding and Generation](https://arxiv.org/abs/2512.04025).

<p align="center">
  <img src="figures/prompt007comparison.jpg" width="100%">
</p>

<p align="center"><em>Visual comparison of sparse attention methods at similar sparsity levels (~90%). PSA maintains visual fidelity close to full attention while other methods show noticeable artifacts.</em></p>

> **Note:** The legacy kernel (`psa_kernel_legacy.py`) now supports **backward propagation** for training. Causal attention is also supported via this kernel, see [qwen2.5-vl-example/](qwen2.5-vl-example/) for details.

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

**Plug-and-play module:** For a simple drop-in replacement, see [`src/psa_triton/README.md`](src/psa_triton/README.md).

## Quick Start

### CogVideoX-5B LoRA (4-step Fast Inference with PSA + Distillation)

This configuration combines **PSA sparse attention** with **step distillation (TDM)** to achieve maximum inference speedup. By integrating PSA into the student model during the distillation training phase, we achieve:

| Metric | Value |
|--------|-------|
| **VBench Score** | 0.826 (vs. 50-step full-attention: 0.819, 4-step distillation-only: 0.818) |
| **Sparsity** | 85% without quality loss |

This demonstrates that PSA is a highly compatible plug-and-play module that compounds effectively with distillation techniques.

**1. Download LoRA Weights**

```bash
huggingface-cli download GYP666/BLADE cogvideox-5b-psa-lora/pytorch_lora_weights.safetensors --local-dir ./weights
```

> After downloading, update `lora_path` in [`examples/configs/model_configs.py`](examples/configs/model_configs.py) to point to your weights directory.

**2. Run Inference**

```bash
python examples/inference/cogvideo/cogvideo_5b.py \
    --model cogvideo_5b_lora \
    --prompt "A garden comes to life as a kaleidoscope of butterflies flutters amidst the blossoms, their delicate wings casting shadows on the petals below. In the background, a grand fountain cascades water with a gentle splendor, its rhythmic sound providing a soothing backdrop. Beneath the cool shade of a mature tree, a solitary wooden chair invites solitude and reflection, its smooth surface worn by the touch of countless visitors seeking a moment of tranquility in nature's embrace." \
    --use_psa --attention_preset psa_4steps
```

### Wan2.1-1.3B

```bash
python examples/inference/wan21/wan21_1.3b.py \
    --prompt "your prompt here" \
    --use_psa --no_warmup
```

For more inference examples and model configurations, see **[examples/README.md](examples/README.md)**.

### Qwen2.5-VL Vision Understanding

PSA also supports vision understanding models. For complete Qwen2.5-VL usage guide, see **[qwen2.5-vl-example/README.md](qwen2.5-vl-example/README.md)**.

## Attention Configuration

PSA is configured via `configs/attention_config.yaml`. For detailed parameter documentation and custom preset creation, see **[configs/README.md](configs/README.md)**.

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
