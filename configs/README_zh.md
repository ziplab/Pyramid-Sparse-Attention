# 注意力配置指南

[**English**](README.md) | [**中文**](README_zh.md)

本目录包含 PSA 注意力配置文件（`attention_config.yaml`）。本指南解释配置结构以及如何创建或修改预设。

## 配置结构

每个模型都有独立的配置节，结构如下：

```yaml
ModelName:
  default_attention: preset_name     # 默认使用的预设
  video_scale:                       # 视频维度除数（模型相关）
    width_divisor: 16
    height_divisor: 16
    depth_divisor: 4
  text_length: 226                   # 文本 token 长度（无文本模型为 0）
  attention_configs:
    preset_name:                     # 预设名称
      type: psa                      # "psa" 或 "dense"
      description: "..."
      # PSA 特定参数...
```

## 参数说明

### 模型级参数

| 参数 | 说明 |
|------|------|
| `default_attention` | 未指定 `--attention_preset` 时使用的默认预设名称 |
| `video_scale` | 将视频维度转换为 token 网格维度的除数 |
| `text_length` | 视觉 token 前的文本 token 数量（模型相关） |

### 预设级参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `type` | string | `psa` 稀疏注意力，`dense` 全注意力基线 |
| `description` | string | 预设的描述 |
| `use_rearrange` | bool | 启用空间 token 重排（提升局部性） |
| `use_sim_mask` | bool | 启用基于余弦相似度的池化约束 |
| `block_size.m` | int | Query 块大小 |
| `block_size.n` | int | Key/Value 块大小 |
| `block_size.tile_n` | int | K/V 处理的硬件 tile 大小 |
| `mask_ratios` | dict | 每个金字塔层级的累积重要性阈值 |
| `mask_mode` | string | `thresholdbound` 或 `topk`（见下文） |
| `attn_impl` | string | 注意力实现：`new_mask_type` 或 `old_mask_type` |
| `tile_size` | list | 重排的 3D tile 维度 `[depth, height, width]` |
| `warmup_steps` | int | 切换到稀疏注意力前使用全注意力的初始步数 |
| `rearrange_method` | string | Token 重排算法（如 `Gilbert`） |
| `verbose` | bool | 推理时启用详细日志 |
| `sim_thresholds` | dict | 各池化层级的余弦相似度阈值 |

## 理解 mask_ratios

`mask_ratios` 参数控制如何根据重要性分数将 query-key block pairs 分配到不同的金字塔层级。

### thresholdbound 模式

在 `thresholdbound` 模式下，`mask_ratios` 定义累积重要性分数阈值：

```yaml
mask_ratios:
  1: [0.0, 0.4]    # 层级 1：重要性前 0-40% → 全分辨率
  2: [0.4, 0.5]    # 层级 2：40-50% → 2 倍池化
  4: [0.5, 0.6]    # 层级 4：50-60% → 4 倍池化
  8: [0.6, 0.8]    # 层级 8：60-80% → 8 倍池化
  0: [0.8, 1.0]    # 层级 0：80-100% → 跳过注意力
```

键是金字塔层级：
- **层级 1**：全分辨率 KV（最高质量）
- **层级 2/4/8**：逐级池化的 KV（2x/4x/8x 平均池化）
- **层级 0**：完全跳过注意力

### topk 模式

在 `topk` 模式下，`mask_ratios` 定义各层级的固定配额（百分比）：

```yaml
mask_ratios:
  1: [0.0, 0.1]    # 10% 的 blocks 使用全分辨率
  2: [0.1, 0.15]   # 5% 的 blocks 使用 2 倍池化
  4: [0.15, 0.15]  # 0% 使用 4 倍池化（跳过）
  8: [0.15, 0.35]  # 20% 的 blocks 使用 8 倍池化
  0: [0.35, 1.0]   # 65% 的 blocks 被跳过
```

### 模式选择

- **`thresholdbound`**：根据重要性分布动态调整分配。通常能获得更好的相似度指标（PSNR/SSIM/LPIPS）。
- **`topk`**：各层级固定配额。在极高稀疏度下能产生更稳定的视觉效果。

## 创建新预设

1. 打开 `configs/attention_config.yaml`
2. 找到目标模型节（如 `CogVideo_5b`）
3. 在 `attention_configs` 下添加新预设：

```yaml
CogVideo_5b:
  attention_configs:
    # ... 现有预设 ...
    
    my_custom_preset:
      type: psa
      description: "我的自定义 PSA 配置"
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

4. 通过命令行使用你的预设：

```bash
python examples/inference/cogvideo/cogvideo_5b.py \
    --prompt "your prompt" \
    --use_psa \
    --attention_preset my_custom_preset
```
