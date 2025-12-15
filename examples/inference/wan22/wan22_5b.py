"""
Wan2.2-5B Inference Script with PSA Support

This script generates videos using Wan2.2-5B model with optional Pyramid Sparse Attention (PSA).

Usage:
    # Single prompt with PSA
    python wan22_5b.py --prompt "A cat baking a cake" --use_psa

    # Batch inference from file
    python wan22_5b.py --prompt_file prompts.txt --batch_size 2

    # Custom video resolution
    python wan22_5b.py --prompt "..." --width 1280 --height 768 --num_frames 69
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modify_diffusers import set_adaptive_sparse_attention
from configs.model_configs import get_model_config
from utils import Timer, save_video, create_output_dir, seed_everything


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from text file (one prompt per line)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def setup_pipeline(config: dict, device: str = "cuda"):
    """Setup Wan2.2-5B pipeline with configuration."""
    print(f"\n{'='*70}")
    print(f"Loading Wan2.2-5B Pipeline")
    print(f"{'='*70}")

    cache_dir = config.get("cache_dir")
    if cache_dir:
        print(f"📂 Using cache directory: {cache_dir}")

    # Setup VAE (float32 for better quality)
    vae = AutoencoderKLWan.from_pretrained(
        config["model_id"],
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=cache_dir
    )
    print("✅ VAE loaded (float32)")

    # Setup scheduler
    flow_shift = config.get("flow_shift", 3.0)
    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=flow_shift
    )
    print(f"✅ Scheduler configured (flow_shift={flow_shift})")

    # Load main pipeline (bfloat16 for efficiency)
    dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float16
    pipe = WanPipeline.from_pretrained(
        config["model_id"],
        vae=vae,
        torch_dtype=dtype,
        cache_dir=cache_dir
    )
    pipe.scheduler = scheduler
    pipe.to(device)

    print(f"✅ Pipeline loaded on {device} (dtype={config['dtype']})")

    return pipe


def generate_video(
    pipe,
    prompt: str,
    negative_prompt: str,
    config: dict,
    video_shape: List[int],
    seed: int,
    output_path: str,
) -> float:
    """Generate a single video.

    Returns:
        Inference time in seconds
    """
    # Setup generator
    gen = torch.Generator(device="cuda").manual_seed(seed)

    # Extract video dimensions
    width, height, num_frames = video_shape

    print(f"\n📹 Generating video...")
    print(f"   Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"   Video shape: {width}×{height}×{num_frames}")
    print(f"   Steps: {config['num_inference_steps']}, Seed: {seed}")

    # Warmup pass (optional, helps with timing accuracy)
    if hasattr(generate_video, '_warmup_done') and not generate_video._warmup_done:
        print("🔥 Performing warmup inference...")
        with torch.inference_mode():
            _ = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=config["guidance_scale"],
                generator=gen,
                num_inference_steps=config["num_inference_steps"],
            )
        generate_video._warmup_done = True
        torch.cuda.empty_cache()
        print("✅ Warmup complete")
        # Reset generator for actual inference
        gen = torch.Generator(device="cuda").manual_seed(seed)

    # Actual inference with timing
    with Timer("Video Generation", use_cuda=True) as timer:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=config["guidance_scale"],
                generator=gen,
                num_inference_steps=config["num_inference_steps"],
            )

    frames = result.frames[0]

    # Save video
    save_video(frames, output_path, fps=config["fps"], verbose=True)

    timer.print_summary()
    return timer.elapsed_seconds


# Class variable for warmup tracking
generate_video._warmup_done = False


def main():
    parser = argparse.ArgumentParser(description="Wan2.2-5B Video Generation")

    # Input options
    parser.add_argument("--prompt", type=str, help="Single prompt for generation")
    parser.add_argument("--prompt_file", type=str, help="File containing prompts (one per line)")
    parser.add_argument("--negative_prompt", type=str, help="Negative prompt (optional)")

    # Model options
    parser.add_argument("--use_psa", action="store_true", help="Enable Pyramid Sparse Attention")
    parser.add_argument("--attention_preset", type=str, default=None,
                       help="PSA preset name (default: from attention_config.yaml)")
    parser.add_argument("--num_inference_steps", type=int, help="Override default inference steps")

    # Video options
    parser.add_argument("--width", type=int, help="Video width (default: 1280)")
    parser.add_argument("--height", type=int, help="Video height (default: 704)")
    parser.add_argument("--num_frames", type=int, help="Number of frames (default: 121)")

    # Output options
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Base output directory (default: outputs)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")

    # Advanced options
    parser.add_argument("--no_timestamp", action="store_true",
                       help="Don't add timestamp to output directory")
    parser.add_argument("--no_warmup", action="store_true",
                       help="Skip warmup inference")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose PSA logging")

    args = parser.parse_args()

    # Validate input
    if not args.prompt and not args.prompt_file:
        parser.error("Either --prompt or --prompt_file must be specified")

    # Load configuration
    config = get_model_config("wan22_5b")
    if args.num_inference_steps:
        config["num_inference_steps"] = args.num_inference_steps

    # Override video shape if specified
    video_shape = config["video_shape"].copy()
    if args.width:
        video_shape[0] = args.width
    if args.height:
        video_shape[1] = args.height
    if args.num_frames:
        video_shape[2] = args.num_frames

    # Get prompts
    if args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
        print(f"📝 Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        prompts = [args.prompt]

    # Get negative prompt
    negative_prompt = args.negative_prompt or config.get("default_negative_prompt", "")

    # Create output directory
    output_dir = create_output_dir(
        args.output_dir,
        "wan22_5b",
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
            video_shape=video_shape,
            attention_preset=args.attention_preset,
            verbose=args.verbose
        )

    # Disable warmup if requested
    if args.no_warmup:
        generate_video._warmup_done = True

    # Generate videos
    total_time = 0.0
    for idx, prompt in enumerate(prompts):
        output_path = output_dir / f"video_{idx:04d}.mp4"

        # Use different seed for each video
        video_seed = args.seed + idx

        inference_time = generate_video(
            pipe,
            prompt,
            negative_prompt,
            config,
            video_shape,
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
