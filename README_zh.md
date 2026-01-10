# Pyramid Sparse Attention (PSA)

[**English**](README.md) | [**中文**](README_zh.md)

**Website:** [http://ziplab.co/PSA](http://ziplab.co/PSA) | **Paper:** [arXiv](https://arxiv.org/abs/2512.04025)

[PSA: Pyramid Sparse Attention for Efficient Video Understanding and Generation](https://arxiv.org/abs/2512.04025) 的官方 PyTorch 实现。

<p align="center">
  <img src="figures/prompt007comparison.jpg" width="100%">
</p>

<p align="center"><em>相近稀疏度（~90%）下各稀疏注意力方法的视觉对比。PSA 保持了接近全注意力的视觉质量，而其他方法则呈现明显的伪影。</em></p>

> **注意：** Legacy kernel (`psa_kernel_legacy.py`) 现已支持**反向传播**，可用于训练。因果注意力也通过此 kernel 支持，详见 [qwen2.5-vl-example/](qwen2.5-vl-example/)。

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

### CogVideoX-5B LoRA (4步快速推理：PSA + 蒸馏)

此配置将 **PSA 稀疏注意力** 与 **步数蒸馏 (TDM)** 相结合，实现最大推理加速。通过在蒸馏训练阶段将 PSA 集成到学生模型中，我们实现了：

| 指标 | 数值 |
|------|------|
| **VBench 得分** | 0.826（对比 50 步全注意力: 0.819，4 步纯蒸馏: 0.818） |
| **稀疏度** | 85%，无质量损失 |

这证明了 PSA 是一个高度兼容的即插即用模块，能够与蒸馏技术有效叠加。

**1. 下载 LoRA 权重**

```bash
huggingface-cli download GYP666/BLADE cogvideox-5b-psa-lora/pytorch_lora_weights.safetensors --local-dir ./weights
```

> 下载后，请修改 [`examples/configs/model_configs.py`](examples/configs/model_configs.py) 中的 `lora_path` 指向你的权重目录。

**2. 运行推理**

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

更多推理示例和模型配置请参考 **[examples/README_zh.md](examples/README_zh.md)**。

### Qwen2.5-VL 视觉理解

PSA 同样支持视觉理解模型。Qwen2.5-VL 的完整使用指南请参考 **[qwen2.5-vl-example/README.md](qwen2.5-vl-example/README.md)**。

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
