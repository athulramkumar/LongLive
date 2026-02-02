#!/usr/bin/env python3
"""
Run 5 interactive 60-second video examples with profiling
"""

import os
import time
import json
import torch
from datetime import datetime
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange

from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller
from utils.dataset import MultiTextDataset


def main():
    # Configuration - 30 second videos (6 segments of 5 seconds each)
    examples = [
        ("example/interactive_30s_1.jsonl", "Surfer at Sunrise"),
        ("example/interactive_30s_2.jsonl", "Jaguar in Rainforest"),
        ("example/interactive_30s_3.jsonl", "Tokyo Night Ramen"),
        ("example/interactive_30s_4.jsonl", "Space Exploration"),
        ("example/interactive_30s_5.jsonl", "Medieval Feast"),
    ]
    
    config_path = "configs/longlive_interactive_60s.yaml"
    output_dir = "videos/interactive_30s"
    
    # Load base config
    config = OmegaConf.load(config_path)
    device = torch.device("cuda:0")
    set_seed(config.seed)
    torch.set_grad_enabled(False)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("Interactive 30-Second Video Generation - 5 Examples")
    print("="*80)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Free VRAM: {get_cuda_free_memory_gb(device):.2f} GB")
    print(f"Frames per video: {config.num_output_frames} ({config.num_output_frames/16:.1f}s at 16fps)")
    print(f"Segments per video: 6 (5 seconds each)")
    print("="*80 + "\n")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(device),
        "config": {
            "num_output_frames": config.num_output_frames,
            "fps": 16,
            "video_duration_seconds": config.num_output_frames / 16,
            "num_segments": 6,
            "segment_duration_seconds": 5,
        },
        "examples": []
    }
    
    # Load model once
    print("[1/6] Loading model (one-time cost)...")
    model_start = time.perf_counter()
    
    config.distributed = False
    pipeline = InteractiveCausalInferencePipeline(config, device=device)
    
    if config.generator_ckpt:
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
        pipeline.generator.load_state_dict(raw_gen_state_dict)
    
    # LoRA
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
    
    pipeline = pipeline.to(dtype=torch.bfloat16)
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    
    torch.cuda.synchronize(device)
    model_load_time = time.perf_counter() - model_start
    results["model_load_seconds"] = round(model_load_time, 2)
    print(f"  Model loaded in {model_load_time:.2f}s")
    print(f"  Memory: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB allocated\n")
    
    # Parse switch frame indices
    switch_frame_indices = [int(x) for x in str(config.switch_frame_indices).split(",")]
    
    # Run each example
    total_inference_time = 0
    for i, (jsonl_path, name) in enumerate(examples):
        print(f"[{i+2}/6] Generating: {name}")
        print(f"     Source: {jsonl_path}")
        
        # Load prompts
        dataset = MultiTextDataset(jsonl_path)
        batch = dataset[0]
        prompts_list = [[p] for p in batch["prompts_list"]]
        
        print(f"     Prompts: {len(prompts_list)} segments")
        
        # Prepare noise
        sampled_noise = torch.randn(
            [1, config.num_output_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16
        )
        
        # Run inference
        torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        
        video = pipeline.inference(
            noise=sampled_noise,
            text_prompts_list=prompts_list,
            switch_frame_indices=switch_frame_indices,
            return_latents=False,
        )
        
        torch.cuda.synchronize(device)
        inference_time = time.perf_counter() - inference_start
        total_inference_time += inference_time
        
        # Decode and save
        decode_start = time.perf_counter()
        current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0
        decode_time = time.perf_counter() - decode_start
        
        save_start = time.perf_counter()
        output_path = os.path.join(output_dir, f"{i+1}_{name.lower().replace(' ', '_')}.mp4")
        write_video(output_path, current_video[0].to(torch.uint8), fps=16)
        save_time = time.perf_counter() - save_start
        
        # Calculate metrics
        fps = config.num_output_frames / inference_time
        realtime_factor = (config.num_output_frames / 16) / inference_time
        memory_gb = torch.cuda.memory_allocated(device) / (1024**3)
        
        example_result = {
            "name": name,
            "jsonl_path": jsonl_path,
            "output_path": output_path,
            "inference_seconds": round(inference_time, 2),
            "decode_seconds": round(decode_time, 2),
            "save_seconds": round(save_time, 2),
            "generation_fps": round(fps, 2),
            "realtime_factor": round(realtime_factor, 3),
            "memory_allocated_gb": round(memory_gb, 2),
        }
        results["examples"].append(example_result)
        
        print(f"     Inference: {inference_time:.2f}s | FPS: {fps:.2f} | Realtime: {realtime_factor:.3f}x")
        print(f"     Output: {output_path}")
        
        # Clear memory between videos
        del video, current_video, sampled_noise
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(f"     Memory cleared, now at {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB\n")
    
    # Summary
    results["total_inference_seconds"] = round(total_inference_time, 2)
    results["average_inference_seconds"] = round(total_inference_time / len(examples), 2)
    results["average_generation_fps"] = round(
        sum(e["generation_fps"] for e in results["examples"]) / len(examples), 2
    )
    results["average_realtime_factor"] = round(
        sum(e["realtime_factor"] for e in results["examples"]) / len(examples), 3
    )
    
    # Print summary
    print("="*80)
    print("                         SUMMARY")
    print("="*80)
    print(f"{'Model Load Time:':<30} {results['model_load_seconds']:.2f}s")
    print(f"{'Total Inference Time:':<30} {results['total_inference_seconds']:.2f}s")
    print(f"{'Average per Video:':<30} {results['average_inference_seconds']:.2f}s")
    print(f"{'Average Generation FPS:':<30} {results['average_generation_fps']:.2f}")
    print(f"{'Average Realtime Factor:':<30} {results['average_realtime_factor']:.3f}x")
    print("\n--- Per-Video Results ---")
    print(f"{'#':<4} {'Name':<25} {'Inference':<12} {'FPS':<10} {'Realtime'}")
    print("-"*65)
    for i, e in enumerate(results["examples"]):
        print(f"{i+1:<4} {e['name']:<25} {e['inference_seconds']:<12.2f} {e['generation_fps']:<10.2f} {e['realtime_factor']:.3f}x")
    print("="*80)
    
    # Save results
    metrics_path = os.path.join(output_dir, "batch_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    return results


if __name__ == "__main__":
    main()
