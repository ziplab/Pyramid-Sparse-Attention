# Qwen2.5-VL 推理

简洁的 Qwen2.5-VL 多模态模型推理代码，支持图像和视频输入，可选 PSA 稀疏注意力加速。

## 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e .

# PSA 需要额外安装 triton
pip install triton
```

## 快速开始

### 命令行推理

```bash
# 图像推理
python src/inference.py --type image --input "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" --prompt "请描述这张图片" --use-psa

# 视频推理
python src/inference.py --type video --input "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4" --prompt "请描述这个视频"

# 使用 PSA 稀疏注意力加速 (推荐长视频)
python src/inference.py --type video --input "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" --prompt "请描述这个视频" --use-psa

# 长视频推理 (预处理 + PSA)
python src/inference.py --type video --input "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4" --prompt "请描述这个视频" \
    --preprocess-frames 64 --use-psa
```

### Python 代码调用

#### 基础推理

```python
from src.inference import QwenVLInference

# 初始化模型
model = QwenVLInference(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    device="auto",
    dtype="auto",
)

# 图像推理
result = model.inference_image(
    image_path="./test.jpg",
    prompt="请描述这张图片",
)
print(result)

# 视频推理
result = model.inference_video(
    video_path="./video.mp4",
    prompt="请描述这个视频",
)
print(result)
```

#### 使用 PSA 稀疏注意力

```python
from src.inference import QwenVLInference

# 初始化模型 (启用 PSA)
model = QwenVLInference(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    device="auto",
    dtype="auto",
    use_psa=True,  # 启用 PSA
)

# 视频推理 (PSA 对长序列效果更明显)
result = model.inference_video(
    video_path="./video.mp4",
    prompt="请描述这个视频",
    preprocess_frames=64,
)
print(result)
```

#### 自定义 PSA 配置

```python
from src.attention import replace_psa_attention_qwen2vl, AttentionConfig
from transformers import Qwen2_5_VLForConditionalGeneration

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

# 自定义 PSA 配置
psa_config = AttentionConfig(
    mask_mode="energybound",  # topk 或 energybound
    mask_ratios={
        1: (0.0, 0.7),   # 1x: 0-70% 注意力块
        2: (0.7, 0.8),   # 2x 池化: 70-80%
        4: (0.8, 0.9),   # 4x 池化: 80-90%
        8: (0.9, 0.9),   # 8x 池化: 90%
        0: (0.9, 1.0),   # 边界: 90-100%
    },
    importance_method="xattn",  # xattn 或 pooling
    xattn_stride=8,
    causal_main=True,
)

# 应用 PSA
model = replace_psa_attention_qwen2vl(model, attention_config=psa_config)
```

## PSA 稀疏注意力

PSA (Pyramid Adaptive Block Sparse Attention) 是一种自适应稀疏注意力机制：

- **原理**: 根据注意力分数自动选择不同的池化级别 (1x/2x/4x/8x)
- **优势**: 在保持精度的前提下显著降低注意力计算量
- **适用场景**: 长视频、长序列推理

### PSA 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| mask_mode | 掩码模式 (topk/energybound) | energybound |
| mask_ratios | 各池化级别的比例 | 见上方代码 |
| importance_method | 重要性估计方法 (xattn/pooling) | xattn |
| xattn_stride | 交叉注意力采样步长 | 8 |
| causal_main | 是否使用因果注意力 | True |

## 可用模型

| 模型 | 参数量 | 推荐显存 |
|------|--------|----------|
| Qwen/Qwen2.5-VL-3B-Instruct | 3B | 8GB |
| Qwen/Qwen2.5-VL-7B-Instruct | 7B | 16GB |
| Qwen/Qwen2.5-VL-72B-Instruct | 72B | 多卡 |

## 项目结构

```
qwenvl2.5-clean/
├── src/
│   ├── inference.py              # 核心推理代码
│   └── attention/
│       ├── qwen2vl_attention.py  # PSA 适配层
│       └── PSA_casual/           # PSA 核心实现
│           ├── PyramidAdaptiveBlockSparseAttn.py
│           ├── kernels/          # Triton 内核
│           └── utils/            # 工具函数
├── examples/
│   ├── demo_image.py             # 图像推理示例
│   ├── demo_video.py             # 视频推理示例
│   ├── demo_quick.py             # Pipeline 示例
│   └── demo_psa.py               # PSA 示例
├── pyproject.toml
└── README.md
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 模型名称或路径 | Qwen/Qwen2.5-VL-7B-Instruct |
| --type | 输入类型 (image/video) | image |
| --input | 输入文件路径或URL | 必填 |
| --prompt | 提示词 | 请描述这张图片 |
| --max-tokens | 最大生成token数 | 512 |
| --device | 设备 (auto/cuda/cpu) | auto |
| --dtype | 数据类型 | auto |
| --preprocess-frames | 视频预处理帧数 | None |
| --use-psa | 启用 PSA 稀疏注意力 | False |
| --psa-log-dir | PSA 日志目录 | None |

## 依赖

- Python >= 3.10
- PyTorch >= 2.0
- transformers >= 4.45
- qwen-vl-utils
- triton (PSA 需要)
- ffmpeg (视频处理)
