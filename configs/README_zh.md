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
| `type` | string | `psa` sparse attention，`dense` full attention |
| `description` | string | 预设描述 |
| `use_rearrange` | bool | 启用空间 token 重排 |
| `use_sim_mask` | bool | 启用基于 cosine similarity 的 pooling 约束 |
| `block_size.m` | int | Query block 大小 |
| `block_size.n` | int | Key/Value block 大小 |
| `block_size.tile_n` | int | K/V tile 大小 |
| `mask_ratios` | dict | 各 pyramid level 的累积重要性阈值 |
| `mask_mode` | string | `thresholdbound` 或 `topk` |
| `attn_impl` | string | Attention 实现：`new_mask_type` 或 `old_mask_type` |
| `tile_size` | list | 3D tile 维度 `[depth, height, width]` |
| `warmup_steps` | int | 切换到 sparse attention 前使用 full attention 的步数 |
| `rearrange_method` | string | Token 重排算法（如 `Gilbert`） |
| `verbose` | bool | 启用详细日志 |
| `sim_thresholds` | dict | 各 pooling level 的 cosine similarity 阈值 |

## mask_ratios 详解

`mask_ratios` 控制如何根据重要性分数将 query-key block pairs 分配到不同 pyramid level。

### thresholdbound 模式

定义累积重要性分数阈值：

```yaml
mask_ratios:
  1: [0.0, 0.4]    # Level 1: 0-40% → 全分辨率
  2: [0.4, 0.5]    # Level 2: 40-50% → 2x pooling
  4: [0.5, 0.6]    # Level 4: 50-60% → 4x pooling
  8: [0.6, 0.8]    # Level 8: 60-80% → 8x pooling
  0: [0.8, 1.0]    # Level 0: 80-100% → skip
```

Pyramid level 含义：
- **Level 1**：全分辨率 KV（最高质量）
- **Level 2/4/8**：Pooled KV（2x/4x/8x average pooling）
- **Level 0**：跳过 attention

### topk 模式

定义各 level 的固定配额（百分比）：

```yaml
mask_ratios:
  1: [0.0, 0.1]    # 10% blocks 用全分辨率
  2: [0.1, 0.15]   # 5% blocks 用 2x pooling
  4: [0.15, 0.15]  # 0% 用 4x pooling（skip）
  8: [0.15, 0.35]  # 20% blocks 用 8x pooling
  0: [0.35, 1.0]   # 65% blocks skip
```

### 模式选择

- **`thresholdbound`**：根据重要性分布动态分配。通常 PSNR/SSIM/LPIPS 更好。
- **`topk`**：各 level 固定配额。极高稀疏度下视觉效果更稳定。

## attn_impl 实现选择

PSA 提供两种 attention kernel 实现，各有特点：

| 特性 | `new_mask_type` | `old_mask_type` |
|------|-----------------|-----------------|
| K block size (n) | 可选 128 / 64 / 32 | 固定 128 |
| use_sim_mask | 不支持 | 支持 |
| 因果掩码 (causal) | 不支持 | 支持 |
| 性能 | 更优 | 略低 |

**推荐**：大多数场景使用 `new_mask_type`（默认），需要 sim_mask 或因果掩码时使用 `old_mask_type`。

## 兼容性注意事项

> **重要**：`attn_impl: new_mask_type` 与 `use_sim_mask: true` 目前不兼容。

如果同时启用这两个选项，程序会抛出错误。请选择以下方案之一：

1. **使用 `new_mask_type`**（推荐）：设置 `use_sim_mask: false`
   ```yaml
   attn_impl: new_mask_type
   use_sim_mask: false
   block_size:
     m: 128
     n: 64      # 可选 128 / 64 / 32
     tile_n: 32
   ```

2. **使用 `old_mask_type` + `use_sim_mask`**：需配合固定 block_size
   ```yaml
   attn_impl: old_mask_type
   use_sim_mask: true
   block_size:
     m: 128
     n: 128    # 必须为 128
     tile_n: 32
   ```

## 创建新预设

1. 打开 `configs/attention_config.yaml`
2. 找到目标模型节（如 `CogVideo_5b`）
3. 在 `attention_configs` 下添加新预设：

```yaml
CogVideo_5b:
  attention_configs:
    # ... existing presets ...
    
    my_custom_preset:
      type: psa
      description: "Custom PSA config"
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
