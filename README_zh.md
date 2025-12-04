# Pyramid Sparse Attention (PSA)

[**English**](README.md) | [**中文**](README_zh.md)

**Website:** [http://ziplab.co/PSA](http://ziplab.co/PSA) | **Paper:** [arXiv](https://arxiv.org/abs/2512.04025)

[PSA: Pyramid Sparse Attention for Efficient Video Understanding and Generation](https://arxiv.org/abs/2512.04025) 的官方 PyTorch 实现。

<p align="center">
  <img src="figures/prompt007comparison.jpg" width="100%">
</p>

<p align="center"><em>相近稀疏度（~90%）下各稀疏注意力方法的视觉对比。PSA 保持了接近全注意力的视觉质量，而其他方法则呈现明显的伪影。</em></p>

> **注意：** 当前版本仅支持**推理**和**双向注意力**。因果掩码和反向传播（训练）的算子仍在优化中，将在后续版本中发布。

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

> 为获得最佳性能，建议使用 PyTorch nightly 版本。

## 下载权重

### CogVideoX-5B LoRA (4步推理)

```bash
huggingface-cli download GYP666/BLADE cogvideox-5b-psa-lora/pytorch_lora_weights.safetensors --local-dir ./weights
```

**注意：** 下载后需要修改 `examples/configs/model_configs.py` 中的 `lora_path` 指向你的权重目录。

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

更多推理示例请参考 [examples/README_zh.md](examples/README_zh.md)。

## 注意力配置说明

PSA 的行为通过 `configs/attention_config.yaml` 文件配置。每个模型都有独立的配置节。

### 配置文件结构

```yaml
ModelName:
  default_attention: psa_balanced    # 默认预设
  video_scale:
    width_divisor: 16
    height_divisor: 16
    depth_divisor: 4
  text_length: 226                   # Text token 长度
  attention_configs:
    preset_name:                     # 如 psa_balanced, psa_4steps, baseline
      type: psa                      # "psa" 或 "dense"
      description: "..."
      # PSA 参数见下文
```

### 核心参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `type` | Attention 类型 | `psa` 或 `dense` |
| `use_rearrange` | 启用空间重排 | `true` / `false` |
| `block_size.m` | Query block 大小 | `128` |
| `block_size.n` | Key/Value block 大小 | `32`, `128` |
| `block_size.tile_n` | K/V tile 大小 | `32` |
| `mask_ratios` | 各 pyramid level 的累积阈值 | 见下文 |
| `mask_mode` | Mask 生成模式 | `thresholdbound`, `topk` |
| `warmup_steps` | Dense attention warmup 步数 | `0`, `12`, `15` |
| `rearrange_method` | Token 重排算法 | `Gilbert` |

### mask_ratios 参数详解

`mask_ratios` 定义累积重要性分数的阈值，用于将 query-key block pairs 分配到不同 pyramid level。以下示例使用 `thresholdbound` 模式：

```yaml
mask_ratios:
  1: [0.0, 0.4]    # Level 1: [0%, 40%] → 全分辨率 KV
  2: [0.4, 0.5]    # Level 2: [40%, 50%] → 2x pooling KV
  4: [0.5, 0.6]    # Level 4: [50%, 60%] → 4x pooling KV
  8: [0.6, 0.8]    # Level 8: [60%, 80%] → 8x pooling KV
  0: [0.8, 1.0]    # Level 0: [80%, 100%] → 跳过 attention
```

- **Level 1**：全分辨率 attention（最高质量，用于最重要的 KV blocks）
- **Level 2/4/8**：Pooled KV（较粗 level 用于次要 blocks）
- **Level 0**：跳过 attention（用于最不重要的 blocks）

### Mask Mode 说明

- **`thresholdbound`**：基于阈值分配，使用累积重要性分数。通常能获得更好的 PSNR/SSIM/LPIPS。
- **`topk`**：基于分位数分配，每个 level 固定配额。在极高稀疏度下视觉效果更稳定。

### 自定义配置

1. 编辑 `configs/attention_config.yaml`
2. 在目标模型的 `attention_configs` 下添加新预设
3. 通过 `--attention_preset your_preset_name` 使用

自定义预设示例：
```yaml
CogVideo_5b:
  attention_configs:
    my_custom_preset:
      type: psa
      description: "Custom PSA config"
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
