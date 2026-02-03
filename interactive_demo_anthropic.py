#!/usr/bin/env python3
"""
Interactive Video Builder with Anthropic-powered Prompt Enhancement

Features:
- Takes initial grounding prompt (subject + setting)
- Uses Claude to expand short user prompts into detailed video descriptions
- Maintains grounding across all prompts for coherence
- Stores both user prompts and AI-processed prompts
- Interactive terminal with back/goto/done commands
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
from omegaconf import OmegaConf
from einops import rearrange
from torchvision.io import write_video

# Anthropic
import anthropic

# LongLive imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.causal_inference import CausalInferencePipeline
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb


@dataclass
class ChunkState:
    """Represents the state after generating a chunk"""
    chunk_idx: int
    user_prompt: str  # Original user input
    processed_prompt: str  # AI-enhanced prompt
    end_latent: torch.Tensor
    kv_cache: List[Dict]
    crossattn_cache: List[Dict]
    current_frame: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class PromptEnhancer:
    """Uses Anthropic Claude to enhance prompts while maintaining grounding"""
    
    SYSTEM_PROMPT = """You are a video prompt engineer. Your job is to create CONCISE, EFFECTIVE prompts for AI video generation.

STRUCTURE (follow this exactly):
1. GROUNDING: State the subject and current setting (1 short phrase)
2. CHANGE: What is visually different/happening (specific details)
3. MOTION: How things are moving in the scene (temporal description)

RULES:
- BE CONCISE: Maximum 2 sentences total. Every word must serve a purpose.
- NO FLUFF: Cut adjectives like "dramatic", "majestic", "beautiful" - they waste tokens
- EXECUTE THE ACTION: User says "transforms" = describe transformation happening
- VISUAL SPECIFICS: Colors, shapes, positions, movements - not emotions or atmosphere
- PRESENT TENSE: Describe what IS happening, not what "begins to" happen

FORMAT: "[Subject] [action verb] [specific visual details]. [Motion/temporal description]."

EXAMPLES:

Input: "A red sports car" + "it transforms into a robot"
OUTPUT: Red sports car's panels split and unfold, hood becoming arms, wheels rotating into legs, chassis rising vertical. Metal parts click into humanoid form, crimson paint visible on robotic torso.

Input: "A young woman with red hair in a forest" + "she ages 50 years"
OUTPUT: Woman's face wrinkles deeply, red hair turns gray then white, posture hunches forward, hands become weathered. Aging progresses visibly across her features over seconds.

Input: "A cat on a windowsill" + "it jumps down"
OUTPUT: Orange cat leaps from windowsill, body arcing downward, legs extending for landing. Fur ripples with motion, tail trails behind for balance.

BAD (too verbose): "The majestic red sports car begins its dramatic transformation as gleaming metal panels elegantly unfold..."
GOOD (concise): "Red car's panels split open, revealing mechanical joints. Hood rises as arms, wheels become legs."

Output ONLY the enhanced prompt. No explanations."""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.grounding: Optional[str] = None
        self.previous_prompts: List[str] = []
    
    def set_grounding(self, grounding: str) -> str:
        """Set the initial grounding and return an enhanced version"""
        self.grounding = grounding
        self.previous_prompts = []
        
        # Enhance the initial grounding - keep it concise
        enhanced = self._call_claude(
            f"""Create a concise visual grounding for a video. This is the STARTING STATE.

User's description: {grounding}

Output format: "[Subject with key visual details] [position/setting] [one distinguishing feature]."
Keep it to 1-2 sentences max. Focus on: what it looks like, where it is, one unique detail."""
        )
        self.previous_prompts.append(enhanced)
        return enhanced
    
    def enhance_prompt(self, user_input: str, add_to_history: bool = True) -> str:
        """Enhance a user's short prompt while maintaining grounding"""
        if not self.grounding:
            raise ValueError("Grounding must be set first")
        
        # Determine current state - either from last prompt or initial grounding
        if len(self.previous_prompts) > 1:
            current_state = self.previous_prompts[-1]
            state_label = "CURRENT STATE (scene has evolved)"
        else:
            current_state = self.grounding
            state_label = "INITIAL STATE"
        
        message = f"""{state_label}:
{current_state}

CHANGE REQUESTED:
{user_input}

Create a concise 2-sentence prompt. Sentence 1: visual changes. Sentence 2: motion/action."""

        enhanced = self._call_claude(message)
        if add_to_history:
            self.previous_prompts.append(enhanced)
        return enhanced
    
    def add_to_history(self, prompt: str):
        """Add a prompt to history (for manually edited prompts)"""
        self.previous_prompts.append(prompt)
    
    def _call_claude(self, message: str) -> str:
        """Call Claude API to enhance prompt"""
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=150,  # Force brevity
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": message}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"  [Warning] Anthropic API error: {e}")
            # Fallback: just combine grounding with user input
            if self.grounding:
                return f"{self.grounding}, {message}"
            return message
    
    def revert_to(self, chunk_idx: int):
        """Revert prompt history when going back
        
        chunk_idx is 0-based chunk index we want to go back to.
        previous_prompts[0] = enhanced grounding
        previous_prompts[1] = chunk 0's enhanced prompt
        previous_prompts[2] = chunk 1's enhanced prompt
        etc.
        
        To regenerate chunk N (chunk_idx = N), we keep prompts [0..N]
        """
        keep_count = chunk_idx + 1  # Keep grounding + all chunks before chunk_idx
        if keep_count < 1:
            keep_count = 1  # Always keep at least the grounding
        if keep_count < len(self.previous_prompts):
            self.previous_prompts = self.previous_prompts[:keep_count]


class InteractiveVideoBuilderWithAI:
    """Interactive video builder with AI-powered prompt enhancement"""
    
    def __init__(self, config_path: str, output_dir: str, 
                 anthropic_key: str,
                 chunk_duration: float = 10.0, 
                 max_chunks: int = 12,  # Max 2 minutes
                 seed: int = 42):
        self.config_path = config_path
        self.output_dir = output_dir
        self.chunk_duration = chunk_duration
        self.max_chunks = max_chunks
        self.seed = seed
        
        # Prompt enhancer
        self.enhancer = PromptEnhancer(anthropic_key)
        
        # Calculate frame counts
        self.fps = 16
        self.temporal_upsample = 4
        video_frames_per_chunk = int(chunk_duration * self.fps)
        self.latent_frames_per_chunk = video_frames_per_chunk // self.temporal_upsample
        self.latent_frames_per_chunk = (self.latent_frames_per_chunk // 3) * 3
        
        # Dynamic total frames (grows as we add chunks)
        self.frames_per_chunk = self.latent_frames_per_chunk
        
        # State management
        self.states: Dict[int, ChunkState] = {}
        self.current_chunk = 0
        self.pipeline = None
        self.device = None
        self.config = None
        self.grounding_set = False
        
        # Warm-up tracking for back/goto operations
        self.needs_warmup = False
        self.warmup_prompt = None
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"session_{self.session_id}")
    
    def setup(self):
        """Initialize the model and pipeline"""
        print("\n" + "="*70)
        print("   INTERACTIVE VIDEO BUILDER WITH AI PROMPT ENHANCEMENT")
        print("="*70)
        
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Load config
        self.config = OmegaConf.load(self.config_path)
        self.config.num_output_frames = self.latent_frames_per_chunk * self.max_chunks
        self.config.seed = self.seed
        self.config.distributed = False
        
        # Setup device
        self.device = torch.device("cuda:0")
        set_seed(self.seed)
        torch.set_grad_enabled(False)
        
        print(f"\nSession: {self.session_id}")
        print(f"Output: {self.session_dir}")
        print(f"GPU: {torch.cuda.get_device_name(self.device)}")
        print(f"Free VRAM: {get_cuda_free_memory_gb(self.device):.2f} GB")
        print(f"\nChunk duration: {self.chunk_duration}s (~{self.latent_frames_per_chunk * self.temporal_upsample} video frames)")
        print(f"Max chunks: {self.max_chunks} (~{self.max_chunks * self.chunk_duration}s max video)")
        
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
        from utils.memory import DynamicSwapInstaller
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
        print(f"  Memory: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB")
        
        # Initialize noise and latent buffers
        self.max_latent_frames = self.latent_frames_per_chunk * self.max_chunks
        self.base_noise = torch.randn(
            [1, self.max_latent_frames, 16, 60, 104],
            device=self.device,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=self.device).manual_seed(self.seed)
        )
        self.full_latents = torch.zeros(
            [1, self.max_latent_frames, 16, 60, 104],
            device='cpu',
            dtype=torch.bfloat16
        )
    
    def _initialize_caches(self):
        """Initialize KV caches"""
        local_attn_size = self.pipeline.local_attn_size
        if local_attn_size != -1:
            kv_cache_size = local_attn_size * self.pipeline.frame_seq_length
        else:
            kv_cache_size = self.latent_frames_per_chunk * self.max_chunks * self.pipeline.frame_seq_length
        
        self.pipeline._initialize_kv_cache(1, torch.bfloat16, self.device, kv_cache_size)
        self.pipeline._initialize_crossattn_cache(1, torch.bfloat16, self.device)
    
    def _copy_cache(self, cache_list):
        """Deep copy a cache list"""
        return [{k: v.clone() if isinstance(v, torch.Tensor) else v 
                 for k, v in cache.items()} for cache in cache_list]
    
    def _recache_after_switch(self, current_start_frame: int, new_conditional_dict: dict):
        """Recache KV with NEW prompt for dramatic changes"""
        # Reset KV cache completely including position indices
        for cache in self.pipeline.kv_cache1:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["global_end_index"].zero_()
            cache["local_end_index"].zero_()
        
        for blk in self.pipeline.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        
        if current_start_frame == 0:
            return
        
        local_attn_size = self.pipeline.local_attn_size
        if local_attn_size == -1:
            num_recache_frames = current_start_frame
        else:
            num_recache_frames = min(local_attn_size, current_start_frame)
        
        recache_start_frame = current_start_frame - num_recache_frames
        frames_to_recache = self.full_latents[:, recache_start_frame:current_start_frame].to(self.device)
        
        print(f"  Recaching {num_recache_frames} frames with new prompt...")
        
        block_mask = self.pipeline.generator.model._prepare_blockwise_causal_attn_mask(
            device=self.device,
            num_frames=num_recache_frames,
            frame_seqlen=self.pipeline.frame_seq_length,
            num_frame_per_block=self.pipeline.num_frame_per_block,
            local_attn_size=local_attn_size
        )
        
        context_timestep = torch.ones([1, num_recache_frames], device=self.device, dtype=torch.int64) * self.config.context_noise
        
        old_block_mask = self.pipeline.generator.model.block_mask
        self.pipeline.generator.model.block_mask = block_mask
        
        global_sink = getattr(self.config.model_kwargs, 'global_sink', False)
        with torch.no_grad():
            self.pipeline.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.pipeline.kv_cache1,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=0,  # Start from 0 since cache was just reset
                sink_recache_after_switch=not global_sink,
            )
        
        self.pipeline.generator.model.block_mask = old_block_mask
        
        for blk in self.pipeline.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
    
    def _run_dummy_generation(self, chunk_idx: int, prompt: str):
        """Run a dummy generation pass to reset model context (output discarded)"""
        start_frame = chunk_idx * self.frames_per_chunk
        end_frame = start_frame + self.frames_per_chunk
        num_blocks = self.frames_per_chunk // self.pipeline.num_frame_per_block
        
        print(f"  [Warm-up pass] Resetting model context...")
        
        conditional_dict = self.pipeline.text_encoder(text_prompts=[prompt])
        
        # Initialize fresh cache
        self._initialize_caches()
        
        # Recache with the previous prompt
        if chunk_idx > 0:
            self._recache_after_switch(start_frame, conditional_dict)
        
        local_attn_size = self.pipeline.local_attn_size if self.pipeline.local_attn_size != -1 else 12
        recache_frames = min(local_attn_size, start_frame) if chunk_idx > 0 else 0
        recache_offset = start_frame - recache_frames if chunk_idx > 0 else 0
        
        # Generate (but don't save) - this fills the cache with proper context
        current_start_frame = start_frame
        dummy_noise = torch.randn_like(self.base_noise[:, start_frame:end_frame])
        
        for block_idx in range(num_blocks):
            current_num_frames = self.pipeline.num_frame_per_block
            noisy_input = dummy_noise[:, block_idx * current_num_frames:(block_idx + 1) * current_num_frames]
            
            cache_frame = current_start_frame - recache_offset
            cache_pos = cache_frame * self.pipeline.frame_seq_length
            
            for step_idx, current_timestep in enumerate(self.pipeline.denoising_step_list):
                timestep = torch.ones([1, current_num_frames], device=self.device, dtype=torch.int64) * current_timestep
                
                _, denoised_pred = self.pipeline.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.pipeline.kv_cache1,
                    crossattn_cache=self.pipeline.crossattn_cache,
                    current_start=cache_pos
                )
                
                if step_idx < len(self.pipeline.denoising_step_list) - 1:
                    next_timestep = self.pipeline.denoising_step_list[step_idx + 1]
                    noisy_input = self.pipeline.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([current_num_frames], device=self.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
            
            # Update cache with context (but don't save latents)
            context_timestep = torch.ones([1, current_num_frames], device=self.device, dtype=torch.int64) * self.config.context_noise
            self.pipeline.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.pipeline.kv_cache1,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=cache_pos,
            )
            
            current_start_frame += current_num_frames
        
        print(f"  [Warm-up complete] Cache reset with previous context")
    
    def generate_chunk(self, chunk_idx: int, user_prompt: str, processed_prompt: str, 
                       needs_warmup: bool = False, warmup_prompt: str = None) -> float:
        """Generate a single chunk"""
        start_frame = chunk_idx * self.frames_per_chunk
        end_frame = start_frame + self.frames_per_chunk
        num_blocks = self.frames_per_chunk // self.pipeline.num_frame_per_block
        
        # Track whether we did warm-up (affects cache handling)
        did_warmup = False
        
        # If coming back from a later chunk, do a warm-up pass first
        if needs_warmup and warmup_prompt and chunk_idx > 0:
            # Restore context for warm-up
            if chunk_idx - 1 in self.states:
                prev_state = self.states[chunk_idx - 1]
                context_frames = prev_state.end_latent.shape[1]
                self.full_latents[:, prev_state.current_frame - context_frames:prev_state.current_frame] = prev_state.end_latent
            
            self._run_dummy_generation(chunk_idx, warmup_prompt)
            did_warmup = True
            
            # Re-restore context for real generation
            if chunk_idx - 1 in self.states:
                prev_state = self.states[chunk_idx - 1]
                context_frames = prev_state.end_latent.shape[1]
                self.full_latents[:, prev_state.current_frame - context_frames:prev_state.current_frame] = prev_state.end_latent
        
        print(f"  Generating frames {start_frame}-{end_frame}...")
        
        # IMPORTANT: Create fresh text embeddings for this prompt
        conditional_dict = self.pipeline.text_encoder(text_prompts=[processed_prompt])
        
        # Track whether we recached (positions will be relative to recache start)
        recached = False
        recache_offset = 0
        
        if chunk_idx > 0:
            # Restore context latents from previous chunk for continuity
            if chunk_idx - 1 in self.states:
                prev_state = self.states[chunk_idx - 1]
                context_frames = prev_state.end_latent.shape[1]
                self.full_latents[:, prev_state.current_frame - context_frames:prev_state.current_frame] = prev_state.end_latent
            
            if did_warmup:
                # After warm-up, the cache has context from chunk 1's continuation
                # DON'T zero the self-attention KV cache - keep that context
                # Only reset cross-attention cache to switch text conditioning
                for blk in self.pipeline.crossattn_cache:
                    blk["k"].zero_()
                    blk["v"].zero_()
                    blk["is_init"] = False
                
                # Reset cache position indices to generate from start of this chunk
                # The warm-up filled cache up to end of chunk 2, but we want to 
                # regenerate from start of chunk 2
                local_attn_size = self.pipeline.local_attn_size if self.pipeline.local_attn_size != -1 else 12
                recache_frames = min(local_attn_size, start_frame)
                recache_offset = start_frame - recache_frames
                
                # Set cache indices to recache start position
                recache_cache_pos = recache_frames * self.pipeline.frame_seq_length
                for cache in self.pipeline.kv_cache1:
                    cache["global_end_index"].fill_(recache_cache_pos)
                    cache["local_end_index"].fill_(recache_cache_pos % (local_attn_size * self.pipeline.frame_seq_length))
                
                recached = True
                # No recache needed - warm-up already established context
            else:
                # Normal case (no warmup): reinitialize caches and recache
                self._initialize_caches()
                self._recache_after_switch(start_frame, conditional_dict)
                recached = True
                local_attn_size = self.pipeline.local_attn_size if self.pipeline.local_attn_size != -1 else 12
                recache_frames = min(local_attn_size, start_frame)
                recache_offset = start_frame - recache_frames
        else:
            self._initialize_caches()
        
        inference_start = time.perf_counter()
        current_start_frame = start_frame
        
        for block_idx in range(num_blocks):
            current_num_frames = self.pipeline.num_frame_per_block
            noisy_input = self.base_noise[:, current_start_frame:current_start_frame + current_num_frames]
            
            # Compute cache position - relative to recache_offset if we recached
            cache_frame = current_start_frame - recache_offset if recached else current_start_frame
            cache_pos = cache_frame * self.pipeline.frame_seq_length
            
            for step_idx, current_timestep in enumerate(self.pipeline.denoising_step_list):
                timestep = torch.ones([1, current_num_frames], device=self.device, dtype=torch.int64) * current_timestep
                
                if step_idx < len(self.pipeline.denoising_step_list) - 1:
                    _, denoised_pred = self.pipeline.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.pipeline.kv_cache1,
                        crossattn_cache=self.pipeline.crossattn_cache,
                        current_start=cache_pos
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
                        current_start=cache_pos
                    )
            
            self.full_latents[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.cpu()
            
            context_timestep = torch.ones([1, current_num_frames], device=self.device, dtype=torch.int64) * self.config.context_noise
            self.pipeline.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.pipeline.kv_cache1,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=cache_pos,
            )
            
            current_start_frame += current_num_frames
        
        inference_time = time.perf_counter() - inference_start
        
        # Save state
        context_frames = min(12, end_frame)
        end_latent = self.full_latents[:, end_frame - context_frames:end_frame].clone()
        
        self.states[chunk_idx] = ChunkState(
            chunk_idx=chunk_idx,
            user_prompt=user_prompt,
            processed_prompt=processed_prompt,
            end_latent=end_latent,
            kv_cache=self._copy_cache(self.pipeline.kv_cache1),
            crossattn_cache=self._copy_cache(self.pipeline.crossattn_cache),
            current_frame=end_frame
        )
        
        # Save outputs
        self._save_chunk_outputs(chunk_idx, user_prompt, processed_prompt)
        
        fps = self.frames_per_chunk / inference_time
        print(f"  Done in {inference_time:.1f}s ({fps:.2f} frames/sec)")
        
        return inference_time
    
    def _decode_latents_chunked(self, latents, chunk_size=80):
        """Decode latents in chunks to avoid OOM"""
        num_frames = latents.shape[1]
        all_video_chunks = []
        
        for start in range(0, num_frames, chunk_size):
            end = min(start + chunk_size, num_frames)
            chunk_latents = latents[:, start:end].to(self.device)
            chunk_video = self.pipeline.vae.decode_to_pixel(chunk_latents, use_cache=False)
            chunk_video = (chunk_video * 0.5 + 0.5).clamp(0, 1)
            # VAE output is (b, t, c, h, w), convert to (b, t, h, w, c) for video writing
            chunk_video = rearrange(chunk_video, "b t c h w -> b t h w c")
            chunk_video = (chunk_video * 255).cpu()
            all_video_chunks.append(chunk_video)
            self.pipeline.vae.model.clear_cache()
            torch.cuda.empty_cache()
        
        return torch.cat(all_video_chunks, dim=1)
    
    def _concatenate_running_video(self, chunk_idx: int):
        """Concatenate chunk videos using ffmpeg"""
        import subprocess
        
        chunk_paths = [os.path.join(self.session_dir, f"chunk_{i+1}.mp4") 
                       for i in range(chunk_idx + 1)]
        chunk_paths = [p for p in chunk_paths if os.path.exists(p)]
        
        if len(chunk_paths) == 1:
            import shutil
            running_path = os.path.join(self.session_dir, f"running_{chunk_idx + 1}.mp4")
            shutil.copy(chunk_paths[0], running_path)
        else:
            running_path = os.path.join(self.session_dir, f"running_{chunk_idx + 1}.mp4")
            concat_file = os.path.join(self.session_dir, "concat_temp.txt")
            with open(concat_file, 'w') as f:
                for path in chunk_paths:
                    f.write(f"file '{os.path.abspath(path)}'\n")
            
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", running_path]
            subprocess.run(cmd, capture_output=True, check=True)
            os.remove(concat_file)
    
    def _save_chunk_outputs(self, chunk_idx: int, user_prompt: str, processed_prompt: str):
        """Save chunk video and metadata"""
        start_frame = chunk_idx * self.frames_per_chunk
        end_frame = (chunk_idx + 1) * self.frames_per_chunk
        
        chunk_latents = self.full_latents[:, start_frame:end_frame].clone()
        chunk_video = self._decode_latents_chunked(chunk_latents)
        
        chunk_path = os.path.join(self.session_dir, f"chunk_{chunk_idx + 1}.mp4")
        write_video(chunk_path, chunk_video[0].to(torch.uint8), fps=self.fps)
        
        # Save prompts
        prompt_path = os.path.join(self.session_dir, f"chunk_{chunk_idx + 1}_prompts.json")
        with open(prompt_path, 'w') as f:
            json.dump({
                "user_prompt": user_prompt,
                "processed_prompt": processed_prompt
            }, f, indent=2)
        
        del chunk_video
        torch.cuda.empty_cache()
        
        self._concatenate_running_video(chunk_idx)
        self._save_history()
        
        print(f"  Saved: chunk_{chunk_idx + 1}.mp4, running_{chunk_idx + 1}.mp4")
    
    def _save_history(self):
        """Save session history"""
        history = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "grounding": self.enhancer.grounding,
            "settings": {
                "chunk_duration": self.chunk_duration,
                "latent_frames_per_chunk": self.latent_frames_per_chunk,
                "fps": self.fps,
            },
            "chunks": [
                {
                    "idx": idx,
                    "user_prompt": state.user_prompt,
                    "processed_prompt": state.processed_prompt,
                    "timestamp": state.timestamp,
                    "time_range": f"{idx * self.chunk_duration:.0f}-{(idx + 1) * self.chunk_duration:.0f}s",
                }
                for idx, state in sorted(self.states.items())
            ]
        }
        
        history_path = os.path.join(self.session_dir, "history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def list_chunks(self):
        """Print all chunks"""
        print("\n--- Generated Chunks ---")
        if self.enhancer.grounding:
            print(f"Grounding: {self.enhancer.grounding[:80]}...")
        print()
        for idx in range(self.current_chunk):
            if idx in self.states:
                state = self.states[idx]
                start_time = idx * self.chunk_duration
                end_time = (idx + 1) * self.chunk_duration
                user_preview = state.user_prompt[:50] + "..." if len(state.user_prompt) > 50 else state.user_prompt
                print(f"  [✓] Chunk {idx + 1} ({start_time:.0f}-{end_time:.0f}s)")
                print(f"      User: {user_preview}")
                proc_preview = state.processed_prompt[:60] + "..." if len(state.processed_prompt) > 60 else state.processed_prompt
                print(f"      AI:   {proc_preview}")
        print()
    
    def goto_chunk(self, target_chunk: int):
        """Go back to a specific chunk for regeneration"""
        if target_chunk < 1 or target_chunk > self.current_chunk:
            print(f"  Invalid chunk. Valid range: 1-{self.current_chunk}")
            return False
        
        target_idx = target_chunk - 1  # Convert to 0-based index
        
        # If going back to chunk N, we need state from chunk N-1 to restore
        if target_idx > 0 and target_idx - 1 in self.states:
            # Restore the latent state from the chunk before our target
            prev_state = self.states[target_idx - 1]
            context_frames = prev_state.end_latent.shape[1]
            # Restore end_latent to full_latents so recaching works
            self.full_latents[:, prev_state.current_frame - context_frames:prev_state.current_frame] = prev_state.end_latent
            # Zero out latents AFTER the restore point to prevent stale data
            self.full_latents[:, prev_state.current_frame:].zero_()
            print(f"  Restored state from chunk {target_idx}")
        else:
            # Going back to chunk 1 - zero everything
            self.full_latents.zero_()
        
        # CRITICAL: Regenerate noise for chunks being redone
        # This ensures regeneration produces different results
        start_frame = target_idx * self.latent_frames_per_chunk
        self.base_noise[:, start_frame:] = torch.randn(
            1, self.max_latent_frames - start_frame, 
            *self.base_noise.shape[2:],
            device=self.device, dtype=torch.bfloat16
        )
        print(f"  Regenerated noise from frame {start_frame}")
        
        # Remove states for chunks >= target
        for idx in list(self.states.keys()):
            if idx >= target_idx:
                del self.states[idx]
        
        # CRITICAL: Fully reinitialize caches to clean slate
        self._initialize_caches()
        
        # Reset model's block_mask to default state
        if hasattr(self.pipeline.generator.model, 'block_mask'):
            default_mask = self.pipeline.generator.model._prepare_blockwise_causal_attn_mask(
                device=self.device,
                num_frames=self.frames_per_chunk,
                frame_seqlen=self.pipeline.frame_seq_length,
                num_frame_per_block=self.pipeline.num_frame_per_block,
                local_attn_size=self.pipeline.local_attn_size
            )
            self.pipeline.generator.model.block_mask = default_mask
        
        # Revert prompt enhancer history
        self.enhancer.revert_to(target_idx - 1 if target_idx > 0 else 0)
        
        # Set warm-up flag - use previous chunk's prompt to reset model context
        self.needs_warmup = True
        if target_idx > 0 and target_idx - 1 in self.states:
            # Use the previous chunk's processed prompt for warm-up
            self.warmup_prompt = self.states[target_idx - 1].processed_prompt
        else:
            # For chunk 1, use the enhanced grounding
            self.warmup_prompt = self.enhancer.grounding
        
        self.current_chunk = target_idx
        print(f"  Rewound to chunk {target_chunk}. Will run warm-up pass before generating.")
        return True
    
    def finalize(self):
        """Save the final video"""
        if not self.states:
            print("  No chunks generated!")
            return
        
        max_chunk = max(self.states.keys())
        final_video_frames = (max_chunk + 1) * self.latent_frames_per_chunk * self.temporal_upsample
        
        print(f"\n  Finalizing video (~{final_video_frames} frames, ~{final_video_frames/self.fps:.1f}s)...")
        
        running_path = os.path.join(self.session_dir, f"running_{max_chunk + 1}.mp4")
        final_path = os.path.join(self.session_dir, "final_video.mp4")
        
        if os.path.exists(running_path):
            import shutil
            shutil.copy(running_path, final_path)
        
        self._save_history()
        print(f"  Final video saved: {final_path}")
        print(f"  Session saved to: {self.session_dir}")
    
    def run(self):
        """Main interactive loop"""
        self.setup()
        
        # Step 1: Get grounding prompt
        print("\n" + "="*70)
        print("Step 1: Set the GROUNDING (subject + setting)")
        print("This anchors all future prompts for consistency.")
        print("="*70)
        print("\nDescribe the main subject and setting.")
        print("Example: 'A young woman with red hair standing in a misty forest'")
        print("Tip: Start with 'No AI:' to skip enhancement")
        print()
        
        while True:
            grounding_input = input("Grounding: ").strip()
            if grounding_input:
                break
            print("  Please enter a grounding prompt.")
        
        # Check if user wants to skip AI
        if grounding_input.lower().startswith('no ai:') or grounding_input.lower().startswith('no ai '):
            enhanced_grounding = grounding_input[6:].strip() if grounding_input.lower().startswith('no ai:') else grounding_input[5:].strip()
            print(f"\n  [Using grounding directly without AI enhancement]")
            self.enhancer.grounding = enhanced_grounding
            self.enhancer.previous_prompts = [enhanced_grounding]
        else:
            print("\n  [Enhancing grounding with AI...]")
            enhanced_grounding = self.enhancer.set_grounding(grounding_input)
            
            print(f"\n  Your input: {grounding_input}")
            print(f"\n  ┌─ AI Enhanced Grounding ─────────────────────────────────")
            words = enhanced_grounding.split()
            line = "  │ "
            for word in words:
                if len(line) + len(word) + 1 > 80:
                    print(line)
                    line = "  │ " + word + " "
                else:
                    line += word + " "
            if line.strip() != "│":
                print(line)
            print("  └────────────────────────────────────────────────────────")
            
            # Interactive approval for grounding
            while True:
                print("\n  Options: [Enter]=Accept, [r]=Regenerate, [e]=Edit manually")
                choice = input("  Your choice: ").strip().lower()
                
                if choice == '' or choice == 'y' or choice == 'yes':
                    break
                elif choice == 'r' or choice == 'regenerate':
                    print("  [Regenerating...]")
                    self.enhancer.previous_prompts = []
                    enhanced_grounding = self.enhancer.set_grounding(grounding_input)
                    print(f"\n  ┌─ AI Enhanced Grounding (regenerated) ────────────────")
                    words = enhanced_grounding.split()
                    line = "  │ "
                    for word in words:
                        if len(line) + len(word) + 1 > 80:
                            print(line)
                            line = "  │ " + word + " "
                        else:
                            line += word + " "
                    if line.strip() != "│":
                        print(line)
                    print("  └────────────────────────────────────────────────────────")
                    continue
                elif choice == 'e' or choice == 'edit':
                    print("\n  Enter your edited grounding (press Enter twice to finish):")
                    lines = []
                    while True:
                        line = input("  > ")
                        if line == "":
                            if lines:
                                break
                        else:
                            lines.append(line)
                    enhanced_grounding = " ".join(lines)
                    self.enhancer.grounding = enhanced_grounding
                    self.enhancer.previous_prompts = [enhanced_grounding]
                    print(f"  [Using edited grounding]")
                    break
                else:
                    print("  Invalid choice. Enter, 'r', or 'e'")
        
        self.grounding_set = True
        print(f"\n  ✓ Grounding set: {enhanced_grounding[:60]}...")
        
        # Step 2: Interactive chunk generation
        print("\n" + "="*70)
        print("Step 2: Build your video chunk by chunk")
        print("="*70)
        print("\nCommands:")
        print("  - Enter a prompt/change for the next ~10s chunk")
        print("  - 'No AI: <prompt>' - Use prompt directly without AI enhancement")
        print("  - 'list' - Show all generated chunks")
        print("  - 'back' - Go back one chunk")
        print("  - 'goto N' - Go to chunk N for re-editing")
        print("  - 'done' - Finish and save final video")
        print("  - 'quit' - Exit without saving")
        print("\nAfter AI enhancement, you can: Accept (Enter), Regenerate (r), or Edit (e)")
        print()
        
        while self.current_chunk < self.max_chunks:
            chunk_num = self.current_chunk + 1
            start_time = self.current_chunk * self.chunk_duration
            end_time = start_time + self.chunk_duration
            
            print(f"\n[Chunk {chunk_num}] ({start_time:.0f}-{end_time:.0f}s)")
            user_input = input("Your prompt/change: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'done':
                if self.states:
                    self.finalize()
                else:
                    print("  No chunks generated yet!")
                break
            
            if user_input.lower() == 'quit':
                print("  Exiting without saving...")
                break
            
            if user_input.lower() == 'list':
                self.list_chunks()
                continue
            
            if user_input.lower() == 'back':
                if self.current_chunk > 0:
                    self.goto_chunk(self.current_chunk)
                else:
                    print("  Already at the first chunk!")
                continue
            
            if user_input.lower().startswith('goto '):
                try:
                    target = int(user_input.split()[1])
                    self.goto_chunk(target)
                except (ValueError, IndexError):
                    print("  Usage: goto N (e.g., 'goto 2')")
                continue
            
            # Check if user wants to skip AI enhancement
            if user_input.lower().startswith('no ai:') or user_input.lower().startswith('no ai '):
                # Remove the "No AI:" prefix and use the rest directly
                processed_prompt = user_input[6:].strip() if user_input.lower().startswith('no ai:') else user_input[5:].strip()
                print(f"\n  [Using prompt directly without AI enhancement]")
                print(f"  Prompt: {processed_prompt}")
                self.enhancer.add_to_history(processed_prompt)
            else:
                # Enhance the prompt with AI
                print("  [Enhancing prompt with AI...]")
                processed_prompt = self.enhancer.enhance_prompt(user_input, add_to_history=False)
                
                # Show full enhanced prompt and get user approval
                print(f"\n  ┌─ AI Enhanced Prompt ─────────────────────────────────")
                # Word wrap the prompt for display
                words = processed_prompt.split()
                line = "  │ "
                for word in words:
                    if len(line) + len(word) + 1 > 80:
                        print(line)
                        line = "  │ " + word + " "
                    else:
                        line += word + " "
                if line.strip() != "│":
                    print(line)
                print("  └────────────────────────────────────────────────────────")
                
                # Interactive approval loop
                while True:
                    print("\n  Options: [Enter]=Accept, [r]=Regenerate, [e]=Edit manually")
                    choice = input("  Your choice: ").strip().lower()
                    
                    if choice == '' or choice == 'y' or choice == 'yes':
                        # Accept the prompt
                        self.enhancer.add_to_history(processed_prompt)
                        break
                    elif choice == 'r' or choice == 'regenerate':
                        # Regenerate with AI
                        print("  [Regenerating...]")
                        processed_prompt = self.enhancer.enhance_prompt(user_input, add_to_history=False)
                        print(f"\n  ┌─ AI Enhanced Prompt (regenerated) ──────────────────")
                        words = processed_prompt.split()
                        line = "  │ "
                        for word in words:
                            if len(line) + len(word) + 1 > 80:
                                print(line)
                                line = "  │ " + word + " "
                            else:
                                line += word + " "
                        if line.strip() != "│":
                            print(line)
                        print("  └────────────────────────────────────────────────────────")
                        continue
                    elif choice == 'e' or choice == 'edit':
                        # Let user edit manually
                        print("\n  Enter your edited prompt (or paste the AI version and modify):")
                        print("  (Press Enter twice to finish)")
                        lines = []
                        while True:
                            line = input("  > ")
                            if line == "":
                                if lines:
                                    break
                            else:
                                lines.append(line)
                        processed_prompt = " ".join(lines)
                        self.enhancer.add_to_history(processed_prompt)
                        print(f"  [Using edited prompt]")
                        break
                    else:
                        print("  Invalid choice. Enter, 'r', or 'e'")
            
            # Generate the chunk (with warmup if coming back from later chunk)
            self.generate_chunk(
                self.current_chunk, 
                user_input, 
                processed_prompt,
                needs_warmup=self.needs_warmup,
                warmup_prompt=self.warmup_prompt
            )
            
            # Reset warmup flags after use
            self.needs_warmup = False
            self.warmup_prompt = None
            
            self.current_chunk += 1
        
        if self.current_chunk >= self.max_chunks:
            print(f"\n  Reached maximum chunks ({self.max_chunks}). Finalizing...")
            self.finalize()
        
        print(f"\nSession complete!")
        print(f"Files saved in: {self.session_dir}")


def main():
    parser = argparse.ArgumentParser(description="Interactive Video Builder with AI Prompt Enhancement")
    parser.add_argument("--config", type=str, default="configs/longlive_inference.yaml")
    parser.add_argument("--output_dir", type=str, default="videos/interactive_ai")
    parser.add_argument("--chunk_duration", type=float, default=10.0)
    parser.add_argument("--max_chunks", type=int, default=12, help="Maximum number of chunks (default: 12 = 2 minutes)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Get Anthropic API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try to source from bash_aliases
        bash_aliases = "/root/.bash_aliases"
        if os.path.exists(bash_aliases):
            with open(bash_aliases) as f:
                for line in f:
                    if "ANTHROPIC_API_KEY" in line and "export" in line:
                        api_key = line.split("=")[1].strip().strip('"\'')
                        break
    
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found!")
        print("Set it in environment or /root/.bash_aliases")
        sys.exit(1)
    
    builder = InteractiveVideoBuilderWithAI(
        config_path=args.config,
        output_dir=args.output_dir,
        anthropic_key=api_key,
        chunk_duration=args.chunk_duration,
        max_chunks=args.max_chunks,
        seed=args.seed
    )
    
    builder.run()


if __name__ == "__main__":
    main()
