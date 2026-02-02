#!/usr/bin/env python3
"""
Batch Profiled LongLive Inference Script
Runs multiple prompts from a file and collects metrics for each
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

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller


class BatchProfiler:
    """Collects profiling metrics for batch inference"""
    
    def __init__(self, device):
        self.device = device
        self.results = []
        self.model_load_time = None
        self.gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "N/A"
        self.total_gpu_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3) if torch.cuda.is_available() else 0
        
    def start_timer(self):
        torch.cuda.synchronize(self.device)
        return time.perf_counter()
    
    def stop_timer(self, start_time):
        torch.cuda.synchronize(self.device)
        return time.perf_counter() - start_time
    
    def get_memory(self):
        torch.cuda.synchronize(self.device)
        return {
            "allocated_gb": round(torch.cuda.memory_allocated(self.device) / (1024**3), 3),
            "reserved_gb": round(torch.cuda.memory_reserved(self.device) / (1024**3), 3),
            "free_gb": round(get_cuda_free_memory_gb(self.device), 3)
        }
    
    def record_inference(self, prompt_idx, prompt, num_frames, fps, inference_time, decode_time, save_time, memory, output_path):
        video_duration = num_frames / fps
        generation_fps = num_frames / inference_time
        realtime_factor = video_duration / inference_time
        
        result = {
            "prompt_idx": prompt_idx,
            "prompt": prompt[:150] + "..." if len(prompt) > 150 else prompt,
            "num_frames": num_frames,
            "fps": fps,
            "video_duration_seconds": round(video_duration, 2),
            "inference_seconds": round(inference_time, 3),
            "decode_seconds": round(decode_time, 3),
            "save_seconds": round(save_time, 3),
            "total_generation_seconds": round(inference_time + decode_time + save_time, 3),
            "generation_fps": round(generation_fps, 2),
            "realtime_factor": round(realtime_factor, 3),
            "memory": memory,
            "output_path": output_path
        }
        self.results.append(result)
        return result
    
    def generate_report(self):
        """Generate summary report"""
        if not self.results:
            return {}
        
        total_frames = sum(r["num_frames"] for r in self.results)
        total_inference_time = sum(r["inference_seconds"] for r in self.results)
        total_video_duration = sum(r["video_duration_seconds"] for r in self.results)
        avg_fps = total_frames / total_inference_time if total_inference_time > 0 else 0
        avg_realtime = total_video_duration / total_inference_time if total_inference_time > 0 else 0
        
        # Memory stats
        peak_memory = max(r["memory"]["allocated_gb"] for r in self.results)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "gpu": self.gpu_name,
            "total_gpu_memory_gb": round(self.total_gpu_memory, 2),
            "model_load_seconds": round(self.model_load_time, 3) if self.model_load_time else None,
            "num_prompts": len(self.results),
            "total_frames_generated": total_frames,
            "total_video_duration_seconds": round(total_video_duration, 2),
            "total_inference_seconds": round(total_inference_time, 3),
            "average_generation_fps": round(avg_fps, 2),
            "average_realtime_factor": round(avg_realtime, 3),
            "peak_memory_allocated_gb": round(peak_memory, 2),
            "per_prompt_results": self.results
        }
        return report
    
    def print_report(self):
        report = self.generate_report()
        
        print("\n" + "="*80)
        print("                         BATCH PROFILING REPORT")
        print("="*80)
        
        print(f"\n{'GPU:':<35} {report['gpu']}")
        print(f"{'Total GPU Memory:':<35} {report['total_gpu_memory_gb']:.2f} GB")
        print(f"{'Model Load Time:':<35} {report['model_load_seconds']:.2f} s")
        
        print(f"\n{'--- Batch Summary ---'}")
        print(f"{'Number of Prompts:':<35} {report['num_prompts']}")
        print(f"{'Total Frames Generated:':<35} {report['total_frames_generated']}")
        print(f"{'Total Video Duration:':<35} {report['total_video_duration_seconds']:.2f} s")
        print(f"{'Total Inference Time:':<35} {report['total_inference_seconds']:.2f} s")
        print(f"{'Average Generation FPS:':<35} {report['average_generation_fps']:.2f} frames/sec")
        print(f"{'Average Realtime Factor:':<35} {report['average_realtime_factor']:.3f}x")
        print(f"{'Peak Memory Usage:':<35} {report['peak_memory_allocated_gb']:.2f} GB")
        
        print(f"\n{'--- Per-Prompt Results ---'}")
        print(f"{'#':<4} {'Frames':<8} {'Duration':<10} {'Inference':<12} {'FPS':<10} {'Realtime':<10}")
        print("-"*60)
        for r in self.results:
            print(f"{r['prompt_idx']:<4} {r['num_frames']:<8} {r['video_duration_seconds']:<10.2f} {r['inference_seconds']:<12.2f} {r['generation_fps']:<10.2f} {r['realtime_factor']:<10.3f}x")
        
        print("\n" + "="*80)
        return report


def load_prompts(filepath):
    """Load prompts from a text file (one per line)"""
    with open(filepath, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Batch LongLive Profiled Video Generation")
    parser.add_argument("--config_path", type=str, default="configs/longlive_inference.yaml")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to file with prompts (one per line)")
    parser.add_argument("--duration", type=float, default=5.0, help="Target video duration in seconds")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames (overrides --duration)")
    parser.add_argument("--output_dir", type=str, default="videos/batch_profiled")
    parser.add_argument("--save_metrics", type=str, default=None, help="Path to save metrics JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompts", type=int, default=None, help="Limit number of prompts to process")
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config_path)
    
    # Calculate frame count
    frame_block_size = config.get("num_frame_per_block", 3)
    fps = 16
    
    if args.num_frames is not None:
        num_frames = (args.num_frames // frame_block_size) * frame_block_size
    else:
        target_frames = int(args.duration * fps)
        num_frames = ((target_frames + frame_block_size - 1) // frame_block_size) * frame_block_size
    
    config.num_output_frames = num_frames
    config.seed = args.seed
    
    # Load prompts
    prompts = load_prompts(args.prompts_file)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    print(f"\n{'='*80}")
    print("LongLive Batch Profiled Inference")
    print(f"{'='*80}")
    print(f"Prompts file: {args.prompts_file}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Frames per video: {num_frames} ({num_frames/fps:.1f}s at {fps}fps)")
    print(f"{'='*80}\n")
    
    # Setup device
    device = torch.device("cuda:0")
    set_seed(config.seed)
    torch.set_grad_enabled(False)
    
    # Initialize profiler
    profiler = BatchProfiler(device)
    
    # ==================== MODEL LOADING ====================
    print("[1/2] Loading model...")
    model_start = profiler.start_timer()
    
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
    
    profiler.model_load_time = profiler.stop_timer(model_start)
    print(f"  Model loaded in {profiler.model_load_time:.2f}s")
    print(f"  Memory: {profiler.get_memory()['allocated_gb']:.2f} GB allocated")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ==================== BATCH INFERENCE ====================
    print(f"\n[2/2] Running batch inference ({len(prompts)} prompts)...\n")
    
    for idx, prompt in enumerate(prompts):
        print(f"  [{idx+1}/{len(prompts)}] Processing prompt {idx}...")
        print(f"      Prompt: {prompt[:80]}...")
        
        # Prepare noise
        sampled_noise = torch.randn(
            [1, num_frames, 16, 60, 104], 
            device=device, 
            dtype=torch.bfloat16
        )
        
        # Inference
        inference_start = profiler.start_timer()
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=[prompt],
            return_latents=True,
            low_memory=True,
            profile=False,
        )
        inference_time = profiler.stop_timer(inference_start)
        
        # Decode
        decode_start = profiler.start_timer()
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video_output = 255.0 * current_video
        decode_time = profiler.stop_timer(decode_start)
        
        # Clear VAE cache
        pipeline.vae.model.clear_cache()
        
        # Save
        save_start = profiler.start_timer()
        output_path = os.path.join(args.output_dir, f"prompt_{idx:03d}.mp4")
        write_video(output_path, video_output[0], fps=fps)
        save_time = profiler.stop_timer(save_start)
        
        # Record metrics
        memory = profiler.get_memory()
        result = profiler.record_inference(
            prompt_idx=idx,
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
            inference_time=inference_time,
            decode_time=decode_time,
            save_time=save_time,
            memory=memory,
            output_path=output_path
        )
        
        print(f"      Inference: {inference_time:.2f}s | FPS: {result['generation_fps']:.2f} | Memory: {memory['allocated_gb']:.2f} GB")
    
    # Print final report
    report = profiler.print_report()
    
    # Save metrics
    if args.save_metrics:
        with open(args.save_metrics, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nMetrics saved to: {args.save_metrics}")
    
    return report


if __name__ == "__main__":
    main()
