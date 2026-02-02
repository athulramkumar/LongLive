#!/usr/bin/env python3
"""
Profiled LongLive Inference Script
Tracks: model load time, inference time, FPS, memory usage
"""

import argparse
import torch
import os
import time
import json
from datetime import datetime
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller


class ProfilerMetrics:
    """Collects and reports profiling metrics"""
    
    def __init__(self, device):
        self.device = device
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "N/A",
            "total_gpu_memory_gb": torch.cuda.get_device_properties(device).total_memory / (1024**3) if torch.cuda.is_available() else 0,
        }
        self.timers = {}
        
    def start_timer(self, name):
        torch.cuda.synchronize(self.device)
        self.timers[name] = time.perf_counter()
        
    def stop_timer(self, name):
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self.timers[name]
        self.metrics[f"{name}_seconds"] = round(elapsed, 3)
        return elapsed
        
    def record_memory(self, name):
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        free = get_cuda_free_memory_gb(self.device)
        self.metrics[f"{name}_allocated_gb"] = round(allocated, 3)
        self.metrics[f"{name}_reserved_gb"] = round(reserved, 3)
        self.metrics[f"{name}_free_gb"] = round(free, 3)
        
    def record(self, key, value):
        self.metrics[key] = value
        
    def report(self):
        """Print formatted profiling report"""
        print("\n" + "="*70)
        print("                    PROFILING REPORT")
        print("="*70)
        
        # System info
        print(f"\n{'GPU:':<30} {self.metrics.get('gpu_name', 'N/A')}")
        print(f"{'Total GPU Memory:':<30} {self.metrics.get('total_gpu_memory_gb', 0):.2f} GB")
        
        # Timing breakdown
        print(f"\n{'--- Timing Breakdown ---'}")
        print(f"{'Model Load Time:':<30} {self.metrics.get('model_load_seconds', 0):.3f} s")
        print(f"{'Text Encoding Time:':<30} {self.metrics.get('text_encoding_seconds', 0):.3f} s")
        print(f"{'Inference Time:':<30} {self.metrics.get('inference_seconds', 0):.3f} s")
        print(f"{'Video Decoding Time:':<30} {self.metrics.get('video_decode_seconds', 0):.3f} s")
        print(f"{'Video Save Time:':<30} {self.metrics.get('video_save_seconds', 0):.3f} s")
        print(f"{'Total Pipeline Time:':<30} {self.metrics.get('total_pipeline_seconds', 0):.3f} s")
        
        # Performance metrics
        print(f"\n{'--- Performance Metrics ---'}")
        print(f"{'Frames Generated:':<30} {self.metrics.get('num_frames', 0)}")
        print(f"{'Video Duration:':<30} {self.metrics.get('video_duration_seconds', 0):.2f} s")
        print(f"{'Generation FPS:':<30} {self.metrics.get('generation_fps', 0):.2f} frames/sec")
        print(f"{'Realtime Factor:':<30} {self.metrics.get('realtime_factor', 0):.2f}x")
        
        # Memory usage
        print(f"\n{'--- Peak Memory Usage ---'}")
        print(f"{'After Model Load:':<30} {self.metrics.get('after_model_load_allocated_gb', 0):.2f} GB allocated")
        print(f"{'During Inference:':<30} {self.metrics.get('during_inference_allocated_gb', 0):.2f} GB allocated")
        print(f"{'Peak Reserved:':<30} {self.metrics.get('during_inference_reserved_gb', 0):.2f} GB reserved")
        
        print("\n" + "="*70)
        
        return self.metrics
    
    def save_json(self, filepath):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="LongLive Profiled Video Generation")
    parser.add_argument("--config_path", type=str, default="configs/longlive_inference.yaml")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--duration", type=float, default=5.0, help="Target video duration in seconds (will be rounded to nearest valid frame count)")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames (overrides --duration). Must be divisible by 3.")
    parser.add_argument("--output_dir", type=str, default="videos/profiled")
    parser.add_argument("--output_name", type=str, default="output.mp4")
    parser.add_argument("--save_metrics", type=str, default=None, help="Path to save metrics JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", action="store_true", help="Run a warmup inference first (for more accurate timing)")
    args = parser.parse_args()
    
    # Load base config
    config = OmegaConf.load(args.config_path)
    
    # Calculate frame count - must be divisible by num_frame_per_block (default 3)
    frame_block_size = config.get("num_frame_per_block", 3)
    fps = 16
    
    if args.num_frames is not None:
        # Round to nearest valid frame count
        num_frames = (args.num_frames // frame_block_size) * frame_block_size
        if num_frames != args.num_frames:
            print(f"Note: Adjusted frames from {args.num_frames} to {num_frames} (must be divisible by {frame_block_size})")
    else:
        # Calculate from duration
        target_frames = int(args.duration * fps)
        num_frames = ((target_frames + frame_block_size - 1) // frame_block_size) * frame_block_size
    
    config.num_output_frames = num_frames
    config.seed = args.seed
    
    # Setup device
    device = torch.device("cuda:0")
    set_seed(config.seed)
    
    # Initialize profiler
    profiler = ProfilerMetrics(device)
    profiler.record("prompt", args.prompt[:200] + "..." if len(args.prompt) > 200 else args.prompt)
    profiler.record("num_frames", num_frames)
    profiler.record("fps", fps)
    profiler.record("video_duration_seconds", num_frames / fps)
    
    print(f"\n{'='*70}")
    print("LongLive Profiled Inference")
    print(f"{'='*70}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Free VRAM: {get_cuda_free_memory_gb(device):.2f} GB")
    print(f"Frames to generate: {num_frames} ({num_frames/fps:.1f}s at {fps}fps)")
    print(f"{'='*70}\n")
    
    torch.set_grad_enabled(False)
    
    # ==================== MODEL LOADING ====================
    print("[1/5] Loading model...")
    profiler.start_timer("model_load")
    
    # Initialize pipeline
    config.distributed = False
    pipeline = CausalInferencePipeline(config, device=device)
    
    # Load generator checkpoint
    if config.generator_ckpt:
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        if "generator" in state_dict or "generator_ema" in state_dict:
            raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
        elif "model" in state_dict:
            raw_gen_state_dict = state_dict["model"]
        else:
            raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
        pipeline.generator.load_state_dict(raw_gen_state_dict)
    
    # LoRA setup
    from utils.lora_utils import configure_lora_for_model
    import peft
    
    pipeline.is_lora_enabled = False
    if getattr(config, "adapter", None):
        print("  Applying LoRA...")
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=True,
        )
        
        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path:
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
        pipeline.is_lora_enabled = True
    
    # Move to device
    pipeline = pipeline.to(dtype=torch.bfloat16)
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    
    model_load_time = profiler.stop_timer("model_load")
    profiler.record_memory("after_model_load")
    print(f"  Model loaded in {model_load_time:.2f}s")
    print(f"  Memory: {profiler.metrics['after_model_load_allocated_gb']:.2f} GB allocated")
    
    # ==================== TEXT ENCODING ====================
    print("\n[2/5] Encoding text prompt...")
    profiler.start_timer("text_encoding")
    
    # Pre-encode text (this happens inside inference, but we track separately)
    prompts = [args.prompt]
    
    profiler.stop_timer("text_encoding")
    
    # ==================== NOISE PREPARATION ====================
    print("\n[3/5] Preparing noise...")
    sampled_noise = torch.randn(
        [1, num_frames, 16, 60, 104], 
        device=device, 
        dtype=torch.bfloat16
    )
    
    # ==================== INFERENCE ====================
    print("\n[4/5] Running inference...")
    profiler.start_timer("inference")
    profiler.record_memory("before_inference")
    
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=True,
        profile=False,
    )
    
    inference_time = profiler.stop_timer("inference")
    profiler.record_memory("during_inference")
    
    generation_fps = num_frames / inference_time
    realtime_factor = (num_frames / fps) / inference_time  # video_duration / inference_time
    profiler.record("generation_fps", round(generation_fps, 2))
    profiler.record("realtime_factor", round(realtime_factor, 2))
    
    print(f"  Inference completed in {inference_time:.2f}s")
    print(f"  Generation speed: {generation_fps:.2f} frames/sec")
    print(f"  Realtime factor: {realtime_factor:.2f}x {'(faster than realtime!)' if realtime_factor > 1 else ''}")
    
    # ==================== VIDEO DECODE ====================
    print("\n[5/5] Decoding and saving video...")
    profiler.start_timer("video_decode")
    
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    video_output = 255.0 * current_video
    
    profiler.stop_timer("video_decode")
    
    # Clear VAE cache
    pipeline.vae.model.clear_cache()
    
    # Save video
    profiler.start_timer("video_save")
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    write_video(output_path, video_output[0], fps=fps)
    profiler.stop_timer("video_save")
    
    # Calculate total time
    total_time = (
        profiler.metrics.get("model_load_seconds", 0) +
        profiler.metrics.get("text_encoding_seconds", 0) +
        profiler.metrics.get("inference_seconds", 0) +
        profiler.metrics.get("video_decode_seconds", 0) +
        profiler.metrics.get("video_save_seconds", 0)
    )
    profiler.record("total_pipeline_seconds", round(total_time, 3))
    profiler.record("output_path", output_path)
    
    print(f"  Video saved to: {output_path}")
    
    # Print final report
    metrics = profiler.report()
    
    # Save metrics if requested
    if args.save_metrics:
        profiler.save_json(args.save_metrics)
    
    return metrics


if __name__ == "__main__":
    main()
