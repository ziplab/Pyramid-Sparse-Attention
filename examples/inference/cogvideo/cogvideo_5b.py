"""
CogVideoX-5B Inference Script with PSA Support

This script generates videos using CogVideoX-5B model with optional Pyramid Sparse Attention (PSA).

Usage:
    # Single prompt with PSA
    python cogvideo_5b.py --prompt "A panda playing guitar" --use_psa

    # Batch inference from file
    python cogvideo_5b.py --prompt_file prompts.txt --batch_size 2

    # Custom output directory
    python cogvideo_5b.py --prompt "..." --output_dir ./my_outputs

    # Using LoRA model (4-step distillation)
    python cogvideo_5b.py --prompt "..." --model cogvideo_5b_lora --use_psa

Example prompts.txt:
    A cat chasing a butterfly in a garden
    A robot dancing in the rain
    A sunset over mountains
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

import torch
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modify_diffusers import set_adaptive_sparse_attention
from configs.model_configs import get_model_config
from utils import Timer, save_video, create_output_dir, seed_everything

# Optional: Legacy sparse attention from cogvideo_batch_sampler
LEGACY_SPARSE_ATTN_PATH = "/workspace/Vbench_EVA/cogvideo_batch_sampler/simple"
try:
    sys.path.insert(0, LEGACY_SPARSE_ATTN_PATH)
    from modify_cogvideo_legacy import set_block_sparse_attn_cogvideox as set_legacy_sparse_attn
    LEGACY_SPARSE_ATTN_AVAILABLE = True
except ImportError:
    set_legacy_sparse_attn = None
    LEGACY_SPARSE_ATTN_AVAILABLE = False


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from text file (one prompt per line)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def setup_pipeline(config: dict, device: str = "cuda"):
    """Setup CogVideoX pipeline with configuration."""
    print(f"\n{'='*70}")
    print(f"Loading CogVideoX Pipeline")
    print(f"{'='*70}")

    # Load pipeline
    dtype = torch.float16 if config["dtype"] == "float16" else torch.bfloat16
    cache_dir = config.get("cache_dir", None)

    pipe = CogVideoXPipeline.from_pretrained(
        config["model_id"],
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )

    # Setup scheduler
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing"
    )

    # Load LoRA weights if configured
    if config.get("use_lora", False):
        lora_path = Path(config["lora_path"])
        lora_weight = config.get("lora_weight", "pytorch_lora_weights.safetensors")
        weight_file = lora_path / lora_weight

        if not weight_file.exists():
            raise FileNotFoundError(f"LoRA weight file not found: {weight_file}")

        pipe.load_lora_weights(str(lora_path), weight_name=lora_weight)
        print(f"✅ Loaded LoRA weights from {weight_file}")

    # Memory optimizations
    if config.get("enable_cpu_offload", False):
        pipe.enable_model_cpu_offload()
        print("✅ Enabled CPU offload")
    else:
        pipe.to(device)

    if config.get("enable_vae_tiling", False):
        pipe.vae.enable_tiling()
        print("✅ Enabled VAE tiling")

    if config.get("enable_vae_slicing", False):
        pipe.vae.enable_slicing()
        print("✅ Enabled VAE slicing")

    return pipe


def generate_video(
    pipe,
    prompt: str,
    config: dict,
    seed: int,
    output_path: str,
) -> float:
    """Generate a single video.

    Returns:
        Inference time in seconds
    """
    # Setup generator
    execution_device = pipe._execution_device if hasattr(pipe, "_execution_device") else pipe.device
    if isinstance(execution_device, str):
        execution_device = torch.device(execution_device)
    if execution_device.type not in {"cpu", "cuda"}:
        execution_device = torch.device("cpu")

    gen = torch.Generator(device=execution_device).manual_seed(seed)

    # Generate video with timing
    video_shape = config["video_shape"]
    num_frames = video_shape[2]

    print(f"\n📹 Generating video...")
    print(f"   Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"   Video shape: {video_shape[0]}×{video_shape[1]}×{num_frames}")
    print(f"   Steps: {config['num_inference_steps']}, Seed: {seed}")

    with Timer("Video Generation", use_cuda=True) as timer:
        result = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=config["num_inference_steps"],
            num_frames=num_frames,
            guidance_scale=config["guidance_scale"],
            generator=gen,
        )

    frames = result.frames[0]

    # Save video
    save_video(frames, output_path, fps=config["fps"], verbose=True)

    timer.print_summary()
    return timer.elapsed_seconds


def main():
    parser = argparse.ArgumentParser(description="CogVideoX-5B Video Generation")

    # Input options
    parser.add_argument("--prompt", type=str, help="Single prompt for generation")
    parser.add_argument("--prompt_file", type=str, help="File containing prompts (one per line)")

    # Model options
    parser.add_argument("--model", type=str, default="cogvideo1.5_5b",
                       choices=["cogvideo_5b", "cogvideo1.5_5b", "cogvideo_5b_lora"],
                       help="Model configuration to use (default: cogvideo1.5_5b)")
    parser.add_argument("--use_psa", action="store_true", help="Enable Pyramid Sparse Attention")
    parser.add_argument("--use_legacy_sparse_attn", action="store_true",
                       help="Use legacy sparse attention from cogvideo_batch_sampler (AdaptiveBlockSparseAttnTrain)")
    parser.add_argument("--attention_preset", type=str, default=None,
                       help="PSA preset name (default: from attention_config.yaml)")
    parser.add_argument("--num_inference_steps", type=int, help="Override default inference steps")
    parser.add_argument("--lora_path", type=str, help="Override LoRA path from config")

    # Output options
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Base output directory (default: outputs)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Number of videos to generate in parallel (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")

    # Advanced options
    parser.add_argument("--no_timestamp", action="store_true",
                       help="Don't add timestamp to output directory")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose PSA logging")

    args = parser.parse_args()

    # Validate input
    if not args.prompt and not args.prompt_file:
        parser.error("Either --prompt or --prompt_file must be specified")

    # Load configuration
    config = get_model_config(args.model)
    if args.num_inference_steps:
        config["num_inference_steps"] = args.num_inference_steps
    if args.lora_path:
        config["lora_path"] = args.lora_path
        config["use_lora"] = True

    print(f"📦 Using model configuration: {args.model}")
    if config.get("use_lora"):
        print(f"   LoRA path: {config['lora_path']}")
    print(f"   Inference steps: {config['num_inference_steps']}")
    print(f"   Guidance scale: {config['guidance_scale']}")

    # Get prompts
    if args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
        print(f"📝 Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        prompts = [args.prompt]

    # Create output directory
    output_dir = create_output_dir(
        args.output_dir,
        args.model,
        use_timestamp=not args.no_timestamp
    )
    print(f"📁 Output directory: {output_dir}")

    # Set random seed
    seed_everything(args.seed)

    # Setup pipeline
    pipe = setup_pipeline(config)

    # Apply PSA if requested
    if args.use_psa:
        preset_info = args.attention_preset if args.attention_preset else "default from config"
        print(f"\n⚡ Applying Pyramid Sparse Attention (preset: {preset_info})")
        set_adaptive_sparse_attention(
            pipe,
            model_name=config["model_name"],
            inference_num=config["num_inference_steps"],
            video_shape=config["video_shape"],
            attention_preset=args.attention_preset,
            verbose=args.verbose
        )

    # Generate videos
    total_time = 0.0
    for idx, prompt in enumerate(prompts):
        output_path = output_dir / f"video_{idx:04d}.mp4"

        # Use different seed for each video
        video_seed = args.seed + idx

        inference_time = generate_video(
            pipe,
            prompt,
            config,
            video_seed,
            str(output_path)
        )
        total_time += inference_time

    # Print summary
    print(f"\n{'='*70}")
    print(f"✅ Generation Complete!")
    print(f"{'='*70}")
    print(f"Total videos: {len(prompts)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per video: {total_time/len(prompts):.2f}s")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
