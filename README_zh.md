# 金字塔稀疏注意力 (PSA)

[**English**](README.md) | [**中文**](README_zh.md)

**项目主页:** [http://ziplab.co/PSA](http://ziplab.co/PSA) | **论文:** [arXiv](https://arxiv.org/abs/2512.04025)

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
ModelName:                           # 模型名称
  default_attention: psa_balanced    # 默认使用的预设
  video_scale:                       # 视频维度除数
    width_divisor: 16
    height_divisor: 16
    depth_divisor: 4
  text_length: 226                   # 文本token长度（模型相关）
  attention_configs:
    preset_name:                     # 预设名称：psa_balanced, psa_4steps, baseline
      type: psa                      # "psa" 稀疏注意力，"dense" 基线
      description: "..."
      # PSA 特定参数见下文
```

### 核心参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `type` | 注意力类型 | `psa`（稀疏）或 `dense`（密集基线） |
| `use_rearrange` | 启用空间重排 | `true` / `false` |
| `block_size.m` | Query 块大小 | `128` |
| `block_size.n` | Key/Value 块大小 | `32`, `128` |
| `block_size.tile_n` | K/V 的 Tile 大小 | `32` |
| `mask_ratios` | 每个金字塔层级的累积重要性阈值 | 见下文 |
| `mask_mode` | 掩码生成策略 | `thresholdbound`, `topk`（见下文） |
| `warmup_steps` | 密集注意力预热步数 | `0`, `12`, `15` |
| `rearrange_method` | Token 重排算法 | `Gilbert` |

### mask_ratios 参数详解

`mask_ratios` 定义了累积重要性分数的阈值，用于将 query-key block pairs 分配到不同的金字塔层级。以下示例使用 `thresholdbound` 掩码模式：

```yaml
mask_ratios:
  1: [0.0, 0.4]    # 层级1：累积分数在 [0%, 40%] 范围 → 全分辨率 KV
  2: [0.4, 0.5]    # 层级2：累积分数在 [40%, 50%] 范围 → 2倍池化 KV
  4: [0.5, 0.6]    # 层级4：累积分数在 [50%, 60%] 范围 → 4倍池化 KV
  8: [0.6, 0.8]    # 层级8：累积分数在 [60%, 80%] 范围 → 8倍池化 KV
  0: [0.8, 1.0]    # 层级0：累积分数在 [80%, 100%] 范围 → 跳过注意力
```

- **层级 1**：全分辨率注意力（最高质量，用于最重要的 KV blocks）
- **层级 2/4/8**：逐级池化的 KV 表示（较粗层级用于次要 blocks）
- **层级 0**：完全跳过注意力（用于最不重要的 blocks）

### 掩码模式说明

- **`thresholdbound`**：基于阈值的分配策略，使用累积重要性分数。通常能获得更好的相似度评测结果（PSNR/SSIM/LPIPS）。
- **`topk`**：基于分位数的分配策略，每个层级固定配额。在极高稀疏度下能产生更稳定的视觉效果。

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
      description: "我的自定义PSA配置"
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
