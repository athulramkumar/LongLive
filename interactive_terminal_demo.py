#!/usr/bin/env python3
"""
Interactive Terminal Video Builder

Generate videos in 10-second chunks with real-time user prompts.
Supports going back to edit previous chunks.

Commands:
- Type a prompt to generate the next chunk
- 'back' or 'b' - Go back to regenerate previous chunk
- 'goto N' - Jump to chunk N
- 'list' or 'l' - Show all chunks and prompts
- 'preview' - Show current progress
- 'done' or 'd' - Finalize and save complete video
- 'quit' or 'q' - Exit without saving final video
"""

import os
import sys
import json
import time
import shutil
from datetime import datetime
from typing import Optional, Dict, List, Any

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video, read_video
from einops import rearrange

from pipeline.causal_inference import CausalInferencePipeline
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller


class ChunkState:
    """Represents the state after generating a chunk - stores only endpoint latent"""
    def __init__(self, chunk_idx: int, prompt: str, end_latent: torch.Tensor, 
                 kv_cache: List[Dict], crossattn_cache: List[Dict], 
                 current_frame: int):
        self.chunk_idx = chunk_idx
        self.prompt = prompt
        self.end_latent = end_latent  # Only the LAST frame's latent (for continuity)
        self.kv_cache = kv_cache
        self.crossattn_cache = crossattn_cache
        self.current_frame = current_frame  # Frame index at end of this chunk
        self.timestamp = datetime.now().isoformat()


class InteractiveVideoBuilder:
    """Interactive video builder with chunk-by-chunk generation"""
    
    def __init__(self, config_path: str, output_dir: str, 
                 chunk_duration: float = 10.0, total_duration: float = 60.0,
                 seed: int = 42):
        self.config_path = config_path
        self.output_dir = output_dir
        self.chunk_duration = chunk_duration
        self.total_duration = total_duration
        self.seed = seed
        
        # Calculate frame counts
        # Note: VAE does ~4x temporal upsampling, so latent_frames * 4 ≈ video_frames
        self.fps = 16
        self.temporal_upsample = 4  # VAE temporal upsampling factor
        
        # Calculate latent frames needed for desired video duration
        video_frames_per_chunk = int(chunk_duration * self.fps)  # 160 video frames for 10s
        self.latent_frames_per_chunk = video_frames_per_chunk // self.temporal_upsample  # 40 latent frames
        
        # Ensure divisible by 3 (num_frame_per_block)
        self.latent_frames_per_chunk = (self.latent_frames_per_chunk // 3) * 3  # 39 latent frames
        
        self.num_chunks = int(total_duration / chunk_duration)  # 6 chunks
        self.total_latent_frames = self.latent_frames_per_chunk * self.num_chunks  # 234 latent frames
        
        # For compatibility with existing code
        self.frames_per_chunk = self.latent_frames_per_chunk
        self.total_frames = self.total_latent_frames
        
        # State management
        self.states: Dict[int, ChunkState] = {}  # chunk_idx -> state
        self.current_chunk = 0
        self.pipeline = None
        self.device = None
        self.config = None
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"session_{self.session_id}")
        
    def setup(self):
        """Initialize the model and pipeline"""
        print("\n" + "="*70)
        print("       INTERACTIVE VIDEO BUILDER")
        print("="*70)
        
        # Create session directory
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Load config
        self.config = OmegaConf.load(self.config_path)
        self.config.num_output_frames = self.total_frames
        self.config.seed = self.seed
        self.config.distributed = False
        
        # Setup device
        self.device = torch.device("cuda:0")
        set_seed(self.seed)
        torch.set_grad_enabled(False)
        
        print(f"\nSession: {self.session_id}")
        print(f"Output directory: {self.session_dir}")
        print(f"GPU: {torch.cuda.get_device_name(self.device)}")
        print(f"Free VRAM: {get_cuda_free_memory_gb(self.device):.2f} GB")
        print(f"\nVideo settings:")
        print(f"  - Chunk duration: {self.chunk_duration}s ({self.latent_frames_per_chunk} latent frames -> ~{self.latent_frames_per_chunk * self.temporal_upsample} video frames)")
        print(f"  - Total duration: {self.total_duration}s ({self.total_latent_frames} latent frames)")
        print(f"  - Number of chunks: {self.num_chunks}")
        print(f"  - Output FPS: {self.fps}")
        
        # Load model
        print("\n[Loading model...]")
        load_start = time.perf_counter()
        
        self.pipeline = CausalInferencePipeline(self.config, device=self.device)
        
        # Load generator checkpoint
        if self.config.generator_ckpt:
            state_dict = torch.load(self.config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict or "generator_ema" in state_dict:
                raw_gen_state_dict = state_dict["generator_ema" if self.config.use_ema else "generator"]
            elif "model" in state_dict:
                raw_gen_state_dict = state_dict["model"]
            else:
                raise ValueError(f"Generator state dict not found")
            self.pipeline.generator.load_state_dict(raw_gen_state_dict)
        
        # LoRA setup
        from utils.lora_utils import configure_lora_for_model
        import peft
        
        self.pipeline.is_lora_enabled = False
        if getattr(self.config, "adapter", None):
            print("  Applying LoRA...")
            self.pipeline.generator.model = configure_lora_for_model(
                self.pipeline.generator.model,
                model_name="generator",
                lora_config=self.config.adapter,
                is_main_process=True,
            )
            lora_ckpt_path = getattr(self.config, "lora_ckpt", None)
            if lora_ckpt_path:
                lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
                if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                    peft.set_peft_model_state_dict(self.pipeline.generator.model, lora_checkpoint["generator_lora"])
                else:
                    peft.set_peft_model_state_dict(self.pipeline.generator.model, lora_checkpoint)
            self.pipeline.is_lora_enabled = True
        
        # Move to device
        self.pipeline = self.pipeline.to(dtype=torch.bfloat16)
        DynamicSwapInstaller.install_model(self.pipeline.text_encoder, device=self.device)
        self.pipeline.generator.to(device=self.device)
        self.pipeline.vae.to(device=self.device)
        
        load_time = time.perf_counter() - load_start
        print(f"  Model loaded in {load_time:.1f}s")
        print(f"  Memory: {torch.cuda.memory_allocated(self.device) / (1024**3):.2f} GB")
        
        # Initialize output tensor
        self.full_latents = torch.zeros(
            [1, self.total_frames, 16, 60, 104],
            device='cpu',
            dtype=torch.bfloat16
        )
        
        # Initialize base noise (shared across regenerations for consistency)
        self.base_noise = torch.randn(
            [1, self.total_frames, 16, 60, 104],
            device=self.device,
            dtype=torch.bfloat16
        )
        
        print("\n" + "="*70)
        print("Ready! Enter prompts for each 10-second chunk.")
        print("Commands: back, goto N, list, preview, done, quit")
        print("="*70 + "\n")
    
    def _initialize_caches(self):
        """Initialize KV and cross-attention caches"""
        local_attn_cfg = getattr(self.config.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.pipeline.frame_seq_length
        else:
            kv_cache_size = self.total_frames * self.pipeline.frame_seq_length
        
        self.pipeline._initialize_kv_cache(
            batch_size=1,
            dtype=torch.bfloat16,
            device=self.device,
            kv_cache_size_override=kv_cache_size
        )
        self.pipeline._initialize_crossattn_cache(
            batch_size=1,
            dtype=torch.bfloat16,
            device=self.device
        )
        
        self.pipeline.generator.model.local_attn_size = self.pipeline.local_attn_size
        self.pipeline._set_all_modules_max_attention_size(self.pipeline.local_attn_size)
    
    def _copy_cache(self, cache_list):
        """Deep copy a cache list"""
        return [
            {k: v.clone() if isinstance(v, torch.Tensor) else v 
             for k, v in cache.items()}
            for cache in cache_list
        ]
    
    def _restore_cache(self, cache_list, saved_cache):
        """Restore cache from saved state"""
        for i, saved in enumerate(saved_cache):
            for k, v in saved.items():
                if isinstance(v, torch.Tensor):
                    cache_list[i][k].copy_(v)
                else:
                    cache_list[i][k] = v
    
    def _find_nearest_checkpoint(self, target_chunk: int) -> int:
        """Find the nearest available checkpoint at or before target_chunk"""
        for i in range(target_chunk, -1, -1):
            if i in self.states:
                return i
        return -1
    
    def _recache_after_switch(self, current_start_frame: int, new_conditional_dict: dict):
        """
        Recache KV with NEW prompt for dramatic prompt switching.
        This is KEY to making big visual changes between chunks.
        
        Unlike _recache_context, this:
        1. Zeros out the KV cache completely first
        2. Zeros out cross-attention cache first  
        3. Passes sink_recache_after_switch=True to allow overwriting sink tokens
        """
        # Step 1: Zero out the KV cache completely AND reset indices
        for cache in self.pipeline.kv_cache1:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["global_end_index"].zero_()
            cache["local_end_index"].zero_()
        
        # Step 2: Zero out cross-attention cache
        for blk in self.pipeline.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        
        if current_start_frame == 0:
            return
        
        # Step 3: Determine how many frames to recache
        local_attn_size = self.pipeline.local_attn_size
        if local_attn_size == -1:
            num_recache_frames = current_start_frame
        else:
            num_recache_frames = min(local_attn_size, current_start_frame)
        
        recache_start_frame = current_start_frame - num_recache_frames
        
        # Get the latents to recache
        frames_to_recache = self.full_latents[:, recache_start_frame:current_start_frame].to(self.device)
        
        print(f"  Recaching {num_recache_frames} frames with NEW prompt (frames {recache_start_frame}-{current_start_frame})...")
        
        # Step 4: Prepare block mask for recaching
        block_mask = self.pipeline.generator.model._prepare_blockwise_causal_attn_mask(
            device=self.device,
            num_frames=num_recache_frames,
            frame_seqlen=self.pipeline.frame_seq_length,
            num_frame_per_block=self.pipeline.num_frame_per_block,
            local_attn_size=local_attn_size
        )
        
        context_timestep = torch.ones(
            [1, num_recache_frames], 
            device=self.device, 
            dtype=torch.int64
        ) * self.config.context_noise
        
        # Temporarily set the block mask
        old_block_mask = self.pipeline.generator.model.block_mask
        self.pipeline.generator.model.block_mask = block_mask
        
        # Step 5: Recache with NEW prompt - this is key for dramatic changes!
        # sink_recache_after_switch=True allows overwriting sink tokens
        global_sink = getattr(self.config.model_kwargs, 'global_sink', False)
        with torch.no_grad():
            self.pipeline.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.pipeline.kv_cache1,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=recache_start_frame * self.pipeline.frame_seq_length,
                sink_recache_after_switch=not global_sink,
            )
        
        # Restore block mask
        self.pipeline.generator.model.block_mask = old_block_mask
        
        # Step 6: Reset cross-attention cache (will be recomputed during generation)
        for blk in self.pipeline.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
    
    def _recache_context(self, output_latents, current_start_frame, conditional_dict):
        """Recache context from previous frames for continuity"""
        if current_start_frame == 0:
            return
        
        local_attn_size = self.pipeline.local_attn_size
        num_recache_frames = min(local_attn_size, current_start_frame) if local_attn_size != -1 else current_start_frame
        recache_start_frame = current_start_frame - num_recache_frames
        
        frames_to_recache = output_latents[:, recache_start_frame:current_start_frame].to(self.device)
        
        # Prepare block mask
        block_mask = self.pipeline.generator.model._prepare_blockwise_causal_attn_mask(
            device=self.device,
            num_frames=num_recache_frames,
            frame_seqlen=self.pipeline.frame_seq_length,
            num_frame_per_block=self.pipeline.num_frame_per_block,
            local_attn_size=local_attn_size
        )
        
        context_timestep = torch.ones(
            [1, num_recache_frames],
            device=self.device,
            dtype=torch.int64
        ) * self.config.context_noise
        
        self.pipeline.generator.model.block_mask = block_mask
        
        # Recache
        with torch.no_grad():
            self.pipeline.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.pipeline.kv_cache1,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=recache_start_frame * self.pipeline.frame_seq_length,
            )
        
        # Reset cross-attention cache
        for blk in self.pipeline.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
    
    def generate_chunk(self, chunk_idx: int, prompt: str) -> float:
        """Generate a single chunk and return inference time"""
        start_frame = chunk_idx * self.frames_per_chunk
        end_frame = start_frame + self.frames_per_chunk
        num_blocks = self.frames_per_chunk // self.pipeline.num_frame_per_block
        
        print(f"  Generating frames {start_frame}-{end_frame}...")
        
        # Encode text with NEW prompt
        conditional_dict = self.pipeline.text_encoder(text_prompts=[prompt])
        
        # If not first chunk, need to recache with the NEW prompt for dramatic changes
        if chunk_idx > 0:
            if chunk_idx - 1 in self.states:
                # Restore latents from previous chunk (for recaching)
                prev_state = self.states[chunk_idx - 1]
                context_frames = prev_state.end_latent.shape[1]
                self.full_latents[:, prev_state.current_frame - context_frames:prev_state.current_frame] = prev_state.end_latent
            
            # KEY: Recache with NEW prompt for dramatic prompt switching
            # This re-runs previous frames through model with new prompt embeddings
            self._recache_after_switch(start_frame, conditional_dict)
        else:
            self._initialize_caches()
        
        # Generate chunk
        inference_start = time.perf_counter()
        current_start_frame = start_frame
        
        for block_idx in range(num_blocks):
            current_num_frames = self.pipeline.num_frame_per_block
            noisy_input = self.base_noise[:, current_start_frame:current_start_frame + current_num_frames]
            
            # Denoising loop
            for step_idx, current_timestep in enumerate(self.pipeline.denoising_step_list):
                timestep = torch.ones(
                    [1, current_num_frames],
                    device=self.device,
                    dtype=torch.int64
                ) * current_timestep
                
                if step_idx < len(self.pipeline.denoising_step_list) - 1:
                    _, denoised_pred = self.pipeline.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.pipeline.kv_cache1,
                        crossattn_cache=self.pipeline.crossattn_cache,
                        current_start=current_start_frame * self.pipeline.frame_seq_length
                    )
                    next_timestep = self.pipeline.denoising_step_list[step_idx + 1]
                    noisy_input = self.pipeline.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([current_num_frames], device=self.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.pipeline.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.pipeline.kv_cache1,
                        crossattn_cache=self.pipeline.crossattn_cache,
                        current_start=current_start_frame * self.pipeline.frame_seq_length
                    )
            
            # Store output
            self.full_latents[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.cpu()
            
            # Update cache with clean context
            context_timestep = torch.ones_like(timestep) * self.config.context_noise
            self.pipeline.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.pipeline.kv_cache1,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=current_start_frame * self.pipeline.frame_seq_length,
            )
            
            current_start_frame += current_num_frames
        
        inference_time = time.perf_counter() - inference_start
        
        # Save state - only store the END latent (last few frames for context), not all frames
        # This is much more memory efficient
        context_frames = min(12, self.frames_per_chunk)  # Keep last 12 frames for recaching context
        self.states[chunk_idx] = ChunkState(
            chunk_idx=chunk_idx,
            prompt=prompt,
            end_latent=self.full_latents[:, end_frame - context_frames:end_frame].clone(),
            kv_cache=self._copy_cache(self.pipeline.kv_cache1),
            crossattn_cache=self._copy_cache(self.pipeline.crossattn_cache),
            current_frame=end_frame
        )
        
        # Save chunk video and running video
        self._save_chunk_outputs(chunk_idx, prompt)
        
        return inference_time
    
    def _decode_latents_chunked(self, latents, chunk_size=80):
        """Decode latents in chunks to avoid OOM"""
        num_frames = latents.shape[1]
        all_videos = []
        
        for start in range(0, num_frames, chunk_size):
            end = min(start + chunk_size, num_frames)
            chunk_latents = latents[:, start:end].to(self.device)
            chunk_video = self.pipeline.vae.decode_to_pixel(chunk_latents, use_cache=False)
            chunk_video = (chunk_video * 0.5 + 0.5).clamp(0, 1)
            chunk_video = rearrange(chunk_video, 'b t c h w -> b t h w c').cpu() * 255.0
            all_videos.append(chunk_video)
            self.pipeline.vae.model.clear_cache()
            torch.cuda.empty_cache()
        
        return torch.cat(all_videos, dim=1)
    
    def _concatenate_running_video(self, chunk_idx: int):
        """Concatenate chunk videos to create running video (avoids re-decoding)"""
        import subprocess
        
        # Collect all chunk video paths up to current chunk
        chunk_paths = []
        for i in range(chunk_idx + 1):
            chunk_path = os.path.join(self.session_dir, f"chunk_{i + 1}.mp4")
            if os.path.exists(chunk_path):
                chunk_paths.append(chunk_path)
        
        if len(chunk_paths) == 1:
            # Only one chunk, just copy it
            import shutil
            running_path = os.path.join(self.session_dir, f"running_{chunk_idx + 1}.mp4")
            shutil.copy(chunk_paths[0], running_path)
        else:
            # Use ffmpeg to concatenate videos
            running_path = os.path.join(self.session_dir, f"running_{chunk_idx + 1}.mp4")
            
            # Create concat file
            concat_file = os.path.join(self.session_dir, "concat_temp.txt")
            with open(concat_file, 'w') as f:
                for path in chunk_paths:
                    f.write(f"file '{os.path.abspath(path)}'\n")
            
            # Run ffmpeg concat
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_file, "-c", "copy", running_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Clean up
            os.remove(concat_file)
    
    def _save_chunk_outputs(self, chunk_idx: int, prompt: str):
        """Save chunk video, running video, and metadata"""
        start_frame = chunk_idx * self.frames_per_chunk
        end_frame = (chunk_idx + 1) * self.frames_per_chunk
        
        # Decode and save chunk video
        chunk_latents = self.full_latents[:, start_frame:end_frame].clone()
        chunk_video = self._decode_latents_chunked(chunk_latents)
        
        chunk_path = os.path.join(self.session_dir, f"chunk_{chunk_idx + 1}.mp4")
        write_video(chunk_path, chunk_video[0].to(torch.uint8), fps=self.fps)
        
        # Save chunk prompt
        prompt_path = os.path.join(self.session_dir, f"chunk_{chunk_idx + 1}_prompt.txt")
        with open(prompt_path, 'w') as f:
            f.write(prompt)
        
        del chunk_video
        torch.cuda.empty_cache()
        
        # Create running video by concatenating chunk videos (not re-decoding latents)
        self._concatenate_running_video(chunk_idx)
        
        # Save lightweight checkpoint (only end context latents + caches, not full video)
        state = self.states[chunk_idx]
        state_path = os.path.join(self.session_dir, f"checkpoint_{chunk_idx + 1}.pt")
        torch.save({
            'end_latent': state.end_latent,
            'prompt': prompt,
            'chunk_idx': chunk_idx,
            'current_frame': end_frame,
        }, state_path)
        
        # Update history
        self._save_history()
        
        print(f"  Saved: chunk_{chunk_idx + 1}.mp4, running_{chunk_idx + 1}.mp4")
    
    def _save_history(self):
        """Save session history to JSON"""
        history = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "settings": {
                "chunk_duration": self.chunk_duration,
                "total_duration": self.total_duration,
                "latent_frames_per_chunk": self.latent_frames_per_chunk,
                "video_frames_per_chunk": self.latent_frames_per_chunk * self.temporal_upsample,
                "total_latent_frames": self.total_latent_frames,
                "num_chunks": self.num_chunks,
                "fps": self.fps,
                "temporal_upsample": self.temporal_upsample,
            },
            "chunks": [
                {
                    "idx": idx,
                    "prompt": state.prompt,
                    "timestamp": state.timestamp,
                    "start_time": f"{idx * self.chunk_duration:.1f}s",
                    "end_time": f"{(idx + 1) * self.chunk_duration:.1f}s",
                }
                for idx, state in sorted(self.states.items())
            ]
        }
        
        history_path = os.path.join(self.session_dir, "history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def list_chunks(self):
        """Print all chunks and their prompts"""
        print("\n--- Current Chunks ---")
        for idx in range(self.num_chunks):
            status = "✓" if idx in self.states else " "
            start_time = idx * self.chunk_duration
            end_time = (idx + 1) * self.chunk_duration
            if idx in self.states:
                prompt_preview = self.states[idx].prompt[:60] + "..." if len(self.states[idx].prompt) > 60 else self.states[idx].prompt
                print(f"  [{status}] Chunk {idx + 1} ({start_time:.0f}-{end_time:.0f}s): {prompt_preview}")
            else:
                print(f"  [{status}] Chunk {idx + 1} ({start_time:.0f}-{end_time:.0f}s): <not generated>")
        print()
    
    def goto_chunk(self, target_chunk: int):
        """Go to a specific chunk for regeneration"""
        if target_chunk < 1 or target_chunk > self.num_chunks:
            print(f"  Invalid chunk number. Must be 1-{self.num_chunks}")
            return False
        
        # Clear states after target
        for idx in list(self.states.keys()):
            if idx >= target_chunk - 1:
                del self.states[idx]
        
        self.current_chunk = target_chunk - 1
        print(f"  Moved to chunk {target_chunk}. Previous progress up to chunk {target_chunk - 1} preserved.")
        return True
    
    def finalize(self):
        """Save the final complete video"""
        if not self.states:
            print("  No chunks generated yet!")
            return
        
        max_chunk = max(self.states.keys())
        final_latent_frames = (max_chunk + 1) * self.frames_per_chunk
        final_video_frames = final_latent_frames * self.temporal_upsample
        
        print(f"\n  Finalizing video (~{final_video_frames} video frames, ~{final_video_frames/self.fps:.1f}s)...")
        
        # The running video for the last chunk is already the final video
        running_path = os.path.join(self.session_dir, f"running_{max_chunk + 1}.mp4")
        final_path = os.path.join(self.session_dir, "final_video.mp4")
        
        if os.path.exists(running_path):
            shutil.copy(running_path, final_path)
            print(f"  Final video saved: {final_path}")
        else:
            print("  Error: Running video not found")
        
        self._save_history()
        print(f"  Session saved to: {self.session_dir}")
    
    def run(self):
        """Main interactive loop"""
        self.setup()
        
        while self.current_chunk < self.num_chunks:
            chunk_num = self.current_chunk + 1
            start_time = self.current_chunk * self.chunk_duration
            end_time = (self.current_chunk + 1) * self.chunk_duration
            
            # Prompt user
            prompt_text = f"[{chunk_num}/{self.num_chunks}] Enter prompt ({start_time:.0f}-{end_time:.0f}s): "
            try:
                user_input = input(prompt_text).strip()
            except EOFError:
                print("\nEOF received, exiting...")
                break
            except KeyboardInterrupt:
                print("\n\nInterrupted. Use 'done' to save progress or 'quit' to exit.")
                continue
            
            if not user_input:
                continue
            
            # Handle commands
            cmd = user_input.lower()
            
            if cmd in ('quit', 'q'):
                print("\nExiting without saving final video.")
                print(f"Progress saved in: {self.session_dir}")
                break
            
            elif cmd in ('done', 'd'):
                self.finalize()
                break
            
            elif cmd in ('list', 'l'):
                self.list_chunks()
                continue
            
            elif cmd in ('back', 'b'):
                if self.current_chunk > 0:
                    self.goto_chunk(self.current_chunk)  # Go back one
                else:
                    print("  Already at first chunk!")
                continue
            
            elif cmd.startswith('goto '):
                try:
                    target = int(cmd.split()[1])
                    self.goto_chunk(target)
                except (IndexError, ValueError):
                    print("  Usage: goto N (where N is chunk number)")
                continue
            
            elif cmd == 'preview':
                if self.current_chunk > 0 and (self.current_chunk - 1) in self.states:
                    print(f"  Current progress: running_{self.current_chunk}.mp4")
                else:
                    print("  No chunks generated yet.")
                continue
            
            # Generate chunk with the prompt
            print()
            inference_time = self.generate_chunk(self.current_chunk, user_input)
            fps = self.frames_per_chunk / inference_time
            print(f"  Done in {inference_time:.1f}s ({fps:.2f} frames/sec)")
            print()
            
            self.current_chunk += 1
        
        # If we completed all chunks, finalize
        if self.current_chunk >= self.num_chunks:
            self.finalize()
        
        print("\nSession complete!")
        print(f"Files saved in: {self.session_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Terminal Video Builder")
    parser.add_argument("--config", type=str, default="configs/longlive_inference.yaml")
    parser.add_argument("--output_dir", type=str, default="videos/interactive_sessions")
    parser.add_argument("--chunk_duration", type=float, default=10.0, help="Duration per chunk in seconds")
    parser.add_argument("--total_duration", type=float, default=60.0, help="Total video duration in seconds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    builder = InteractiveVideoBuilder(
        config_path=args.config,
        output_dir=args.output_dir,
        chunk_duration=args.chunk_duration,
        total_duration=args.total_duration,
        seed=args.seed,
    )
    builder.run()


if __name__ == "__main__":
    main()
