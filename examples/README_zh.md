# 推理示例

[**English**](README.md) | [**中文**](README_zh.md)

本文档提供所有支持模型的详细推理命令。

## CogVideoX 系列

### CogVideoX1.5-5B (50步)

```bash
python examples/inference/cogvideo/cogvideo_5b.py \
    --model cogvideo1.5_5b \
    --prompt "An elderly gentleman, with a serene expression, sits at the water's edge, a steaming cup of tea by his side. He is engrossed in his artwork, brush in hand, as he renders an oil painting on a canvas that's propped up against a small, weathered table. The sea breeze whispers through his silver hair, gently billowing his loose-fitting white shirt, while the salty air adds an intangible element to his masterpiece in progress. The scene is one of tranquility and inspiration, with the artist's canvas capturing the vibrant hues of the setting sun reflecting off the tranquil sea." \
    --use_psa
```

### CogVideoX-5B LoRA (PSA + 蒸馏 4步快速推理)

此配置将 **PSA 稀疏注意力** 与 **步数蒸馏 (TDM)** 相结合，实现最大化推理加速。通过在蒸馏训练阶段将 PSA 集成到学生模型中，我们实现了：

- **VBench 得分 0.826**，超越 50 步全注意力基线（0.819）和 4 步纯蒸馏基线（0.818）
- 在 **85% 稀疏度** 下运行，且无质量损失

这证明了 PSA 是一个高度兼容的即插即用模块，可以与蒸馏技术有效结合以最大化推理效率。

```bash
python examples/inference/cogvideo/cogvideo_5b.py \
    --model cogvideo_5b_lora \
    --prompt "A garden comes to life as a kaleidoscope of butterflies flutters amidst the blossoms, their delicate wings casting shadows on the petals below. In the background, a grand fountain cascades water with a gentle splendor, its rhythmic sound providing a soothing backdrop. Beneath the cool shade of a mature tree, a solitary wooden chair invites solitude and reflection, its smooth surface worn by the touch of countless visitors seeking a moment of tranquility in nature's embrace." \
    --use_psa --attention_preset psa_4steps
```

## Wan2.1 系列

### Wan2.1-1.3B

```bash
python examples/inference/wan21/wan21_1.3b.py \
    --prompt "A vintage school bus slowly turning a corner on a dusty rural road at sunset. The bus is adorned with colorful retro decals and has a worn wooden dashboard. Children with backpacks and school bags are inside, some playing and others reading books. The bus driver, a stern but kind-looking man, is behind the wheel, gesturing confidently as he navigates the curve. The sun sets behind a cluster of old oak trees, casting warm golden hues over the landscape. The corner is tight, with the bus making a slight sway as it turns. The children's expressions range from curious to excited, and the bus driver's face shows determination and satisfaction. The blurred background features the rolling hills and scattered farmhouses. Warm, nostalgic cinematography style. Low-angle shot from the side, focusing on the driver and the turning bus." \
    --use_psa --no_warmup
```

### Wan2.1-14B

```bash
python examples/inference/wan21/wan21_14b.py \
    --prompt "A couple in elegant formal evening wear, the man in a tuxedo and the woman in a ball gown, holding matching bright red umbrellas. They walk hand in hand through the rain-soaked streets, their reflections shimmering in the water droplets. The lighting is soft and dramatic, highlighting the intricate details of their attire. The man has slicked-back hair and a stern expression, while the woman has flowing blonde hair and a serene smile. Their umbrellas create gentle arcs overhead, casting dappled shadows on the wet cobblestones. The background is a bustling cityscape with tall buildings and flickering streetlights. The couple pauses at a flooded intersection, the rain creating a mist around them. Aerial shot focusing on the couple's faces as they exchange meaningful glances, capturing the intensity of their moment." \
    --use_psa --no_warmup
```

## Wan2.2 系列

### Wan2.2-5B

```bash
python examples/inference/wan22/wan22_5b.py \
    --prompt "An aerial perspective video showcasing an airplane taking off from an airport runway, its sleek wings slicing through the sky with dramatic lighting effects. In the foreground, a busy cityscape with towering skyscrapers and bustling streets below. The train, a vintage steam locomotive, slowly pulling into a small town station, emitting billowing smoke and steam. The train is adorned with colorful painted designs and graffiti along its sides. Both the airplane and train are prominently featured, capturing their distinct appearances and unique moments. The city skyline provides a dynamic backdrop, highlighting the contrast between modern aviation and classic rail travel. Vibrant color grading and cinematic camera movement. Wide shot of the airport and cityscape, then medium shot of the airplane, followed by a close-up of the train entering the station." \
    --use_psa --no_warmup
```

### Wan2.2-A14B

```bash
python examples/inference/wan22/wan22_a14b.py \
    --prompt "A dramatic lightning strike illuminates the iconic Eiffel Tower against a backdrop of dark, ominous clouds in the sky. The lightning crackles and illuminates the structure, casting stark shadows and highlighting every intricate detail. Dark storm clouds swirl menacingly overhead, adding to the intense atmosphere. The Eiffel Tower stands tall and proud amidst the storm, its metal lattice frame gleaming in the electric flash. The scene captures the raw power and beauty of nature's fury, with the lightning slicing through the sky. The lightning bolt strikes the tower, sending a shockwave through the air as it leaves a trail of sparks. The image is captured from a low-angle perspective, emphasizing the grandeur and vulnerability of the tower during the storm." \
    --use_psa --no_warmup
```

## 批量推理

创建 `prompts.txt` 文件，每行一个提示词，然后运行：

```bash
python examples/inference/cogvideo/cogvideo_5b.py \
    --prompt_file prompts.txt \
    --use_psa
```

## 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--prompt` | 文本提示词 | - |
| `--prompt_file` | 提示词文件（每行一个） | - |
| `--use_psa` | 启用 PSA 加速 | `False` |
| `--attention_preset` | PSA 预设名称 | `psa_balanced` |
| `--num_inference_steps` | 覆盖推理步数 | 模型默认值 |
| `--output_dir` | 输出目录 | `outputs` |
| `--seed` | 随机种子 | `42` |
| `--no_warmup` | 跳过 GPU 预热推理（首次运行预热 GPU） | `False` |
| `--verbose` | 启用详细 PSA 日志 | `False` |

### CogVideoX 专用参数

| 参数 | 说明 |
|------|------|
| `--model` | 模型配置：`cogvideo_5b`, `cogvideo1.5_5b`, `cogvideo_5b_lora` |
| `--lora_path` | 覆盖配置中的 LoRA 路径 |

### Wan 系列专用参数

| 参数 | 说明 |
|------|------|
| `--width` | 视频宽度 |
| `--height` | 视频高度 |
| `--num_frames` | 帧数 |
| `--negative_prompt` | 负向提示词 |

## 模型配置

| 模型 | 分辨率 | 帧数 | 步数 | Guidance |
|------|--------|------|------|----------|
| cogvideo_5b | 720x480 | 49 | 50 | 6.0 |
| cogvideo1.5_5b | 1360x768 | 81 | 50 | 6.0 |
| cogvideo_5b_lora | 720x480 | 49 | 4 | 1.0 |
| wan21_1.3b | 1280x720 | 69 | 50 | 5.0 |
| wan21_14b | 1280x720 | 69 | 50 | 5.0 |
| wan22_5b | 1280x704 | 121 | 50 | 5.0 |
| wan22_a14b | 1280x768 | 69 | 40 | 4.0 |

## 输出

视频保存至：`outputs/{model}/{timestamp}/video_0000.mp4`
