# Inference Examples

[**English**](README.md) | [**中文**](README_zh.md)

This document provides detailed inference commands for all supported models.

## CogVideoX Series

### CogVideoX1.5-5B (50 steps)

```bash
python examples/inference/cogvideo/cogvideo_5b.py \
    --model cogvideo1.5_5b \
    --prompt "An elderly gentleman, with a serene expression, sits at the water's edge, a steaming cup of tea by his side. He is engrossed in his artwork, brush in hand, as he renders an oil painting on a canvas that's propped up against a small, weathered table. The sea breeze whispers through his silver hair, gently billowing his loose-fitting white shirt, while the salty air adds an intangible element to his masterpiece in progress. The scene is one of tranquility and inspiration, with the artist's canvas capturing the vibrant hues of the setting sun reflecting off the tranquil sea." \
    --use_psa
```

### CogVideoX-5B LoRA (4-step Fast Inference with PSA + Distillation)

This configuration combines **PSA sparse attention** with **step distillation (TDM)** to achieve maximum inference speedup. By integrating PSA into the student model during the distillation training phase, we achieve:

- **VBench score of 0.826**, surpassing both the 50-step full-attention baseline (0.819) and the 4-step distillation-only baseline (0.818)
- Operating at **85% sparsity** without any quality loss

This demonstrates that PSA is a highly compatible plug-and-play module that compounds effectively with distillation techniques.

```bash
python examples/inference/cogvideo/cogvideo_5b.py \
    --model cogvideo_5b_lora \
    --prompt "A garden comes to life as a kaleidoscope of butterflies flutters amidst the blossoms, their delicate wings casting shadows on the petals below. In the background, a grand fountain cascades water with a gentle splendor, its rhythmic sound providing a soothing backdrop. Beneath the cool shade of a mature tree, a solitary wooden chair invites solitude and reflection, its smooth surface worn by the touch of countless visitors seeking a moment of tranquility in nature's embrace." \
    --use_psa --attention_preset psa_4steps
```

## Wan2.1 Series

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

## Wan2.2 Series

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

## Batch Inference

Create a `prompts.txt` file with one prompt per line, then run:

```bash
python examples/inference/cogvideo/cogvideo_5b.py \
    --prompt_file prompts.txt \
    --use_psa
```

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--prompt` | Text prompt for generation | - |
| `--prompt_file` | File with prompts (one per line) | - |
| `--use_psa` | Enable PSA acceleration | `False` |
| `--attention_preset` | PSA preset name | `psa_balanced` |
| `--num_inference_steps` | Override inference steps | Model default |
| `--output_dir` | Output directory | `outputs` |
| `--seed` | Random seed | `42` |
| `--no_warmup` | Skip GPU warmup inference (first run to warm up GPU) | `False` |
| `--verbose` | Enable verbose PSA logging | `False` |

### CogVideoX-specific Options

| Option | Description |
|--------|-------------|
| `--model` | Model config: `cogvideo_5b`, `cogvideo1.5_5b`, `cogvideo_5b_lora` |
| `--lora_path` | Override LoRA path from config |

### Wan-specific Options

| Option | Description |
|--------|-------------|
| `--width` | Video width |
| `--height` | Video height |
| `--num_frames` | Number of frames |
| `--negative_prompt` | Negative prompt |

## Model Configurations

| Model | Resolution | Frames | Steps | Guidance |
|-------|------------|--------|-------|----------|
| cogvideo_5b | 720x480 | 49 | 50 | 6.0 |
| cogvideo1.5_5b | 1360x768 | 81 | 50 | 6.0 |
| cogvideo_5b_lora | 720x480 | 49 | 4 | 1.0 |
| wan21_1.3b | 1280x720 | 69 | 50 | 5.0 |
| wan21_14b | 1280x720 | 69 | 50 | 5.0 |
| wan22_5b | 1280x704 | 121 | 50 | 5.0 |
| wan22_a14b | 1280x768 | 69 | 40 | 4.0 |

## Output

Videos are saved to: `outputs/{model}/{timestamp}/video_0000.mp4`
