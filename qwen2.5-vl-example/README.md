# Qwen2.5-VL + PSA Example

[中文](README_CN.md)

Qwen2.5-VL inference with optional PSA (Pyramid Sparse Attention) acceleration.

## Setup

```bash
pip install -e .
pip install triton  # Required for PSA
```

## Quick Start

```bash
# Image inference
python src/inference.py --type image --input "image.jpg" --prompt "Describe this image"

# Video inference with PSA
python src/inference.py --type video --input "video.mp4" --prompt "Describe this video" --use-psa
```

## Python API

```python
from src.inference import QwenVLInference

model = QwenVLInference(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    use_psa=True,  # Enable PSA acceleration
)

result = model.inference_video("video.mp4", "Describe this video")
```

## Custom PSA Config

```python
from src.attention import replace_psa_attention_qwen2vl, AttentionConfig

psa_config = AttentionConfig(
    mask_mode="energybound",
    mask_ratios={
        1: (0.0, 0.7),   # Full attention: 0-70%
        2: (0.7, 0.8),   # 2x pooling: 70-80%
        4: (0.8, 0.9),   # 4x pooling: 80-90%
        8: (0.9, 0.9),   # 8x pooling: 90%
        0: (0.9, 1.0),   # Boundary: 90-100%
    },
)
model = replace_psa_attention_qwen2vl(model, attention_config=psa_config)
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model name/path | Qwen/Qwen2.5-VL-7B-Instruct |
| `--type` | Input type (image/video) | image |
| `--input` | Input file path or URL | Required |
| `--prompt` | Prompt text | Required |
| `--use-psa` | Enable PSA sparse attention | False |
| `--preprocess-frames` | Video frame count | None |
