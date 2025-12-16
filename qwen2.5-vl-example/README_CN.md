# Qwen2.5-VL + PSA 示例

[English](README.md)

Qwen2.5-VL 推理示例，支持 PSA（多级稀疏注意力）加速。

## 安装

```bash
pip install -e .
pip install triton  # PSA 需要
```

## 快速开始

```bash
# 图像推理
python src/inference.py --type image --input "image.jpg" --prompt "描述这张图片"

# 视频推理 + PSA 加速
python src/inference.py --type video --input "video.mp4" --prompt "描述这个视频" --use-psa
```

## Python API

```python
from src.inference import QwenVLInference

model = QwenVLInference(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    use_psa=True,  # 启用 PSA 加速
)

result = model.inference_video("video.mp4", "描述这个视频")
```

## 自定义 PSA 配置

```python
from src.attention import replace_psa_attention_qwen2vl, AttentionConfig

psa_config = AttentionConfig(
    mask_mode="energybound",
    mask_ratios={
        1: (0.0, 0.7),   # 全注意力: 0-70%
        2: (0.7, 0.8),   # 2x 池化: 70-80%
        4: (0.8, 0.9),   # 4x 池化: 80-90%
        8: (0.9, 0.9),   # 8x 池化: 90%
        0: (0.9, 1.0),   # 边界: 90-100%
    },
)
model = replace_psa_attention_qwen2vl(model, attention_config=psa_config)
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称或路径 | Qwen/Qwen2.5-VL-7B-Instruct |
| `--type` | 输入类型 (image/video) | image |
| `--input` | 输入文件路径或 URL | 必填 |
| `--prompt` | 提示词 | 必填 |
| `--use-psa` | 启用 PSA 稀疏注意力 | False |
| `--preprocess-frames` | 视频预处理帧数 | None |
