"""Unified model configurations for all supported models.

These configurations follow the official model settings from:
- THUDM/CogVideoX documentation
- Wan-AI model repositories
- SSIM_PSNR_EXP/evaluation_system/scripts/multi_gpu_evaluation.py
"""

from pathlib import Path
from typing import Dict, Any

# Project root directory (relative path support)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ========== CogVideoX-5B Models ==========
    "cogvideo_5b": {
        "model_id": "THUDM/CogVideoX-5b",
        "model_name": "CogVideo_5b",
        "dtype": "bfloat16",
        "video_shape": [720, 480, 49],  # [width, height, num_frames]
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "fps": 8,
        "scheduler": "cogvideox_dpm",
        "enable_cpu_offload": True,
        "enable_vae_tiling": True,
        "enable_vae_slicing": True,
        "default_prompt": (
            "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool "
            "in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, "
            "producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously "
            "and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle "
            "glow on the scene. The panda's face is expressive, showing concentration and joy as it plays."
        ),
    },

    # ========== CogVideoX1.5-5B Models ==========
    "cogvideo1.5_5b": {
        "model_id": "THUDM/CogVideoX1.5-5B",
        "model_name": "CogVideo1.5_5b",
        "dtype": "bfloat16",
        "video_shape": [1360, 768, 81],  # [width, height, num_frames]
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "fps": 8,
        "scheduler": "cogvideox_dpm",
        "enable_cpu_offload": True,
        "enable_vae_tiling": True,
        "enable_vae_slicing": True,
        "default_prompt": (
            "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool "
            "in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, "
            "producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously "
            "and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle "
            "glow on the scene. The panda's face is expressive, showing concentration and joy as it plays."
        ),
    },

    # ========== CogVideoX-5B LoRA Models (4-step distillation) ==========
    "cogvideo_5b_lora": {
        "model_id": "THUDM/CogVideoX-5b",
        "model_name": "CogVideo_5b",
        "dtype": "bfloat16",  # LoRA uses bfloat16
        "video_shape": [720, 480, 49],  # [width, height, num_frames]
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "fps": 8,
        "scheduler": "cogvideox_dpm",
        "enable_cpu_offload": False,
        "enable_vae_tiling": True,
        # LoRA specific configurations
        "use_lora": True,
        "lora_path": "/workspace/VIDEO-BLADE/cogvideox/outputs/cogvideox/8.29/bs80-new_asa-4_steps_lambda-reg_0.5_cfg_3.5_eta_0.9_K_4/checkpoint-135",
        "lora_weight": "pytorch_lora_weights.safetensors",
        "default_prompt": (
            "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool "
            "in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, "
            "producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously "
            "and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle "
            "glow on the scene. The panda's face is expressive, showing concentration and joy as it plays."
        ),
    },
    
    # ========== Wan2.1 Models ==========
    "wan21_1.3b": {
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "model_name": "Wan2.1_1.3b",
        "dtype": "bfloat16",
        "vae_dtype": "float32",  # VAE uses float32
        "video_shape": [1280, 720, 69],  # [width, height, num_frames]
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "fps": 16,
        "flow_shift": 5.0,  # 5.0 for 720P, 3.0 for 480P
        "default_prompt": (
            "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, "
            "while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight "
            "streaming through the window."
        ),
        "default_negative_prompt": (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
            "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
            "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
            "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
            "in the background, walking backwards"
        ),
    },
    
    "wan21_14b": {
        "model_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "model_name": "Wan2.1_14b",
        "dtype": "bfloat16",
        "vae_dtype": "float32",
        "video_shape": [1280, 720, 69],  # [width, height, num_frames]
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "fps": 16,
        "flow_shift": 5.0,  # 5.0 for 720P
        "default_prompt": (
            "Warm colors dominate the room, with a focus on the tabby cat sitting contently in the center. "
            "The scene captures the fluffy orange tabby cat wearing a tiny virtual reality headset. "
            "The setting is a cozy living room, adorned with soft, warm lighting and a modern aesthetic. "
            "A plush sofa is visible in the background, along with a few lush potted plants, adding a touch "
            "of greenery. The cat's tail flicks curiously, as if engaging with an unseen virtual environment. "
            "Its paws swipe at the air, indicating a playful and inquisitive nature, as it delves into the "
            "digital realm. The atmosphere is both whimsical and futuristic, highlighting the blend of analog "
            "and digital experiences."
        ),
        "default_negative_prompt": (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
            "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
            "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
            "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
            "in the background, walking backwards"
        ),
    },
    
    # ========== Wan2.2 Models (MoE Architecture) ==========
    "wan22_5b": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "model_name": "Wan2.2_5B",
        "dtype": "bfloat16",
        "vae_dtype": "float32",
        "video_shape": [1280, 704, 121],  # [width, height, num_frames]
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "fps": 24,
        "default_prompt": (
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely "
            "on a spotlighted stage."
        ),
        "default_negative_prompt": (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
            "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
            "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
            "杂乱的背景，三条腿，背景人很多，倒着走"
        ),
    },
    "wan22_a14b": {
        "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "model_name": "Wan2.2_A14B",
        "dtype": "bfloat16",
        "vae_dtype": "float32",
        "video_shape": [1280, 768, 69],  # [width, height, num_frames]
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,  # Secondary guidance for MoE
        "fps": 16,
        "default_prompt": (
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely "
            "on a spotlighted stage."
        ),
        "default_negative_prompt": (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
            "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
            "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
            "杂乱的背景，三条腿，背景人很多，倒着走"
        ),
    },
}


def get_models_dir() -> Path:
    """Get the models directory path, creating it if needed."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def get_model_config(model_key: str) -> Dict[str, Any]:
    """Get model configuration by key.

    Args:
        model_key: Model identifier (e.g., "cogvideo_5b", "wan21_14b")

    Returns:
        Model configuration dictionary with cache_dir included

    Raises:
        ValueError: If model_key is not found
    """
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Unknown model: {model_key}. Available models: {available}"
        )
    config = MODEL_CONFIGS[model_key].copy()
    config["cache_dir"] = str(get_models_dir())
    return config


def list_available_models() -> list:
    """List all available model keys."""
    return list(MODEL_CONFIGS.keys())
