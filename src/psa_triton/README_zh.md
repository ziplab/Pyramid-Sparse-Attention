# PSA Attention

即插即用的 Pyramid Sparse Attention 模块。

## 使用方法

### 默认配置

```python
from psa_triton import PSAAttention

psa = PSAAttention()
out = psa(q, k, v)  # q, k, v: [B, H, L, D]
```

### 自定义配置

```python
from psa_triton import PSAAttention, PSAConfig

# 只修改需要的参数，其余使用默认值
config = PSAConfig(mask_mode='thresholdbound')
psa = PSAAttention(config)
```

### 完整配置参考

```python
config = PSAConfig(
    # Block 大小配置
    block_m=128,          # Query block 大小
    block_n=64,           # Key/Value block 大小 (new_mask_type: 128/64/32)
    tile_n=32,            # K/V 处理的 tile 大小

    # Mask ratio 配置
    mask_ratios={
        1: (0.0, 0.1),    # 10% full attention
        2: (0.1, 0.15),   # 5% with 2x pooling
        4: (0.15, 0.15),  # 0% with 4x pooling
        8: (0.15, 0.35),  # 20% with 8x pooling
        0: (0.35, 1.0),   # 65% skipped
    },
    mask_mode='topk',     # 'topk' 或 'thresholdbound'

    # 内核实现
    attn_impl='new_mask_type',  # 'new_mask_type' 或 'old_mask_type'

    # 相似度约束的 pooling 配置（仅 old_mask_type 支持）
    use_sim_mask=False,
    sim_2x_threshold=0.0,
    sim_4x_threshold=0.0,
    sim_8x_threshold=-1.0,
)
psa = PSAAttention(config)
out = psa(q, k, v)
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `block_m` | int | 128 | Query block 大小 |
| `block_n` | int | 64 | Key/Value block 大小 (new_mask_type: 128/64/32, old_mask_type: 固定 128) |
| `tile_n` | int | 32 | K/V 处理的 tile 大小 |
| `mask_ratios` | dict | 见上 | 各 pyramid level 的稀疏度分布 |
| `mask_mode` | str | `'topk'` | `'topk'` (固定配额) 或 `'thresholdbound'` (动态分配) |
| `attn_impl` | str | `'new_mask_type'` | 内核实现: `'new_mask_type'`（默认）或 `'old_mask_type'` |
| `use_sim_mask` | bool | False | 启用相似度约束的 pooling（需要 `attn_impl='old_mask_type'`） |
| `sim_2x_threshold` | float | 0.0 | 2x pooling 的相似度阈值 |
| `sim_4x_threshold` | float | 0.0 | 4x pooling 的相似度阈值 |
| `sim_8x_threshold` | float | -1.0 | 8x pooling 的相似度阈值 |
| `rearrange_method` | str | None | Token 重排方法: `'Gilbert'`, `'STA'`, `'SemanticAware'`, `'Hybrid'` |

## attn_impl 实现选择

PSA 提供两种 attention kernel 实现：

| 特性 | `new_mask_type` | `old_mask_type` |
|------|-----------------|-----------------|
| K block size (n) | 可选 128 / 64 / 32 | 固定 128 |
| use_sim_mask | 不支持 | 支持 |
| 因果掩码 (causal) | 不支持 | 支持 |
| 性能 | 更优 | 略低 |

**推荐**：大多数场景使用 `new_mask_type`（默认），需要 sim_mask 时使用 `old_mask_type`。

## 兼容性注意事项

> **重要**：`attn_impl='new_mask_type'` 与 `use_sim_mask=True` 不兼容。

如果同时启用这两个选项，程序会抛出错误。请选择以下方案之一：

1. **使用 `new_mask_type`**（推荐，性能更优）：
   ```python
   config = PSAConfig(
       attn_impl='new_mask_type',
       use_sim_mask=False,
       block_n=64,  # 可选 128 / 64 / 32
   )
   ```

2. **使用 `old_mask_type` + `use_sim_mask`**：
   ```python
   config = PSAConfig(
       attn_impl='old_mask_type',
       use_sim_mask=True,
       block_m=128,
       block_n=128,  # 必须为 128
       tile_n=32,
   )
   ```

## 因果注意力支持

Legacy kernel (`psa_kernel_legacy.py`) 支持因果掩码。对于需要因果注意力的场景（如视觉语言模型 Qwen2.5-VL），请参考：

- Kernel: [`kernels/psa_kernel_legacy.py`](kernels/psa_kernel_legacy.py)
- 示例: [`qwenvl2.5-example/`](../../qwenvl2.5-example/) Qwen2.5-VL 集成

> **注意**：因果掩码不能与 token 重排序（`rearrange_method`）同时使用。重排序后的 token 顺序已改变，下三角掩码不再代表真正的因果关系，会导致结果错误。
