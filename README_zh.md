# Pyramid Sparse Attention (PSA)

[**English**](README.md) | [**中文**](README_zh.md)

**Website:** [http://ziplab.co/PSA](http://ziplab.co/PSA) | **Paper:** [arXiv](https://arxiv.org/abs/2512.04025)

[PSA: Pyramid Sparse Attention for Efficient Video Understanding and Generation](https://arxiv.org/abs/2512.04025) 的官方 PyTorch 实现。

<p align="center">
  <img src="figures/prompt007comparison.jpg" width="100%">
</p>

<p align="center"><em>相近稀疏度（~90%）下各稀疏注意力方法的视觉对比。PSA 保持了接近全注意力的视觉质量，而其他方法则呈现明显的伪影。</em></p>

> **注意：** 当前版本仅支持**推理**。反向传播（训练）的算子仍在优化中，将在后续版本中发布。因果注意力通过 legacy kernel 支持，详见 [qwen2.5-vl-example/](qwen2.5-vl-example/)。

## 安装

### 使用 uv (推荐)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### 使用 pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**即插即用模块：** 如需简单的 drop-in 替换，请参阅 [`src/psa_triton/README_zh.md`](src/psa_triton/README_zh.md)。

## 快速开始

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

更多推理示例和模型配置请参考 **[examples/README_zh.md](examples/README_zh.md)**。

### Qwen2.5-VL 视觉理解

PSA 同样支持视觉理解模型。Qwen2.5-VL 的完整使用指南请参考 **[qwen2.5-vl-example/README.md](qwen2.5-vl-example/README.md)**。

## 下载权重

### CogVideoX-5B LoRA (4步推理)

```bash
huggingface-cli download GYP666/BLADE cogvideox-5b-psa-lora/pytorch_lora_weights.safetensors --local-dir ./weights
```

**注意：** 下载后需要修改 `examples/configs/model_configs.py` 中的 `lora_path` 指向你的权重目录。

## 注意力配置

PSA 通过 `configs/attention_config.yaml` 配置。详细的参数说明和自定义预设方法请参考 **[configs/README_zh.md](configs/README_zh.md)**。

## 引用

如果本项目对你有帮助，请引用我们的论文：

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
