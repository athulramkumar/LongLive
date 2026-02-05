"""
Video Builder State Manager for Web Interface

Refactored from interactive_demo_anthropic.py to be web-compatible.
Removes all input() calls and exposes clean methods for Gradio integration.
"""

import gc
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field

import torch
from omegaconf import OmegaConf
from einops import rearrange
from torchvision.io import write_video

import anthropic

# Add parent directory to path for LongLive imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.causal_inference import CausalInferencePipeline
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb


@dataclass
class ChunkState:
    """Represents the state after generating a chunk"""
    chunk_idx: int
    user_prompt: str
    processed_prompt: str
    end_latent: torch.Tensor  # Stored on CPU to save GPU memory
    current_frame: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class PromptEnhancer:
    """Uses Anthropic Claude to enhance prompts while maintaining grounding"""
    
    SYSTEM_PROMPT = """You are a video prompt engineer for LongLive, an AI video generation model.

Output ONLY valid JSON:
```json
{
  "edits": "Brief description of what you changed",
  "scene": "Environment, setting, lighting, weather, style/aesthetic",
  "subject": "Primary subject with visual traits (appearance, clothing, features)",
  "action": "What happens in this 10-second chunk (1-2 sentences)",
  "locks": ["Elements fixed across chunks"]
}
```

ANALYZE the user's request and INFER which fields to modify:
- Location/background/environment changes → update "scene"
- Lighting/weather/time of day changes → update "scene"
- Style/aesthetic/mood changes → update "scene" (style is part of scene)
- Character/object appearance changes → update "subject"
- Movement/event/action changes → update "action"
- Multiple aspects → update only the relevant fields

EXAMPLES:
- "add rain" → scene edit (weather)
- "she starts running" → action edit
- "change to sunset" → scene edit (lighting/time)
- "make it cinematic" → scene edit (style)
- "she puts on a hat" → subject + action edit
- "zoom out to reveal the city" → scene + action edit

RULES:
1. INFER the edit type from user's natural language - don't require explicit field names
2. MINIMAL CHANGES - only modify fields affected by the user's request
3. COPY unchanged fields exactly from the previous state
4. LOCKS preserve continuity - update only if the locked element itself changes

Output ONLY valid JSON."""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.grounding: Optional[str] = None
        self.previous_prompts: List[str] = []  # Flat text prompts for video model
        self.structured_states: List[Dict] = []  # Structured JSON states for continuity
    
    def set_grounding(self, grounding: str) -> str:
        """Set the initial grounding and return an enhanced version"""
        self.grounding = grounding
        self.previous_prompts = []
        self.structured_states = []
        
        # Generate initial structured state with simplified 5-field format
        structured_response = self._call_claude(
            f"""Create the initial GROUNDING state for a video.

User's description: {grounding}

Output JSON with ONLY these 5 fields:
```json
{{
  "edits": "initial",
  "scene": "Environment, setting, lighting, weather, visual style",
  "subject": "Primary subject with visual traits (appearance, clothing, features)",
  "action": "Initial state or subtle motion",
  "locks": ["subject identity", "key visual elements to preserve"]
}}
```

Output ONLY valid JSON."""
        )
        
        # Parse and store structured state
        structured_state = self._parse_json_response(structured_response)
        if structured_state:
            self.structured_states.append(structured_state)
            # Convert to flat text prompt for video model
            flat_prompt = self._structured_to_flat(structured_state)
        else:
            # Fallback: create a simple flat prompt from grounding
            flat_prompt = f"{grounding}. Scene establishing shot with natural lighting."
        
        self.previous_prompts.append(flat_prompt)
        
        return flat_prompt
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from Claude's response"""
        try:
            # Try to extract JSON from response
            import re
            # Look for JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            # Try parsing the whole response as JSON
            return json.loads(response)
        except (json.JSONDecodeError, AttributeError):
            # If parsing fails, return None
            return None
    
    def _structured_to_flat(self, state: Dict) -> str:
        """Convert structured state to display format (JSON for UI)."""
        if not state:
            return ""
        
        # Return formatted JSON for display
        return json.dumps(state, indent=2)
    
    def _structured_to_video_prompt(self, state: Dict) -> str:
        """Convert structured state to flat text prompt for video model."""
        if not state:
            return ""
        
        parts = []
        
        # Subject + Scene + Action
        if state.get("subject"):
            parts.append(state["subject"])
        if state.get("scene"):
            parts.append(f"in {state['scene']}")
        if state.get("action"):
            parts.append(state["action"])
        
        return ". ".join(filter(None, parts))
    
    def set_grounding_direct(self, grounding: str):
        """Set grounding directly without AI enhancement"""
        self.grounding = grounding
        self.previous_prompts = [grounding]
        self.structured_states = []  # No structured state for direct grounding
    
    def enhance_prompt(self, user_input: str, add_to_history: bool = True) -> str:
        """Enhance a user's short prompt while maintaining grounding"""
        if not self.grounding:
            raise ValueError("Grounding must be set first")
        
        # Get previous structured state if available
        prev_state = self.structured_states[-1] if self.structured_states else None
        
        if prev_state:
            # Pass the previous JSON and ask Claude to modify it
            prev_state_json = json.dumps(prev_state, indent=2)
            
            message = f"""PREVIOUS STATE:
```json
{prev_state_json}
```

USER REQUEST: "{user_input}"

Analyze what the user wants to change and output modified JSON with ONLY these 5 fields:
- "edits": what you changed
- "scene": environment, setting, lighting, style (update if user mentions location/weather/lighting/style)
- "subject": who/what with visual traits (update if user mentions character/appearance)
- "action": what happens in this chunk (update if user mentions movement/events)
- "locks": elements that stay fixed (copy from previous unless a locked element changes)

COPY unchanged fields exactly. Output ONLY valid JSON."""

        else:
            # First chunk - build from grounding
            message = f"""GROUNDING: {self.grounding}

USER ACTION: "{user_input}"

Output JSON with ONLY these 5 fields:
```json
{{
  "edits": "initial",
  "scene": "environment, setting, lighting, style",
  "subject": "who/what with visual traits",
  "action": "what happens",
  "locks": ["subject identity", "key visual elements"]
}}
```"""

        structured_response = self._call_claude(message)
        
        # Parse structured state
        new_state = self._parse_json_response(structured_response)
        
        # Convert to flat prompt for video model
        if new_state:
            flat_prompt = self._structured_to_flat(new_state)
            if add_to_history:
                self.structured_states.append(new_state)
                self.previous_prompts.append(flat_prompt)
            return flat_prompt
        else:
            # Fallback if JSON parsing fails - create prompt from previous + user request
            prev_prompt = self.previous_prompts[-1] if self.previous_prompts else self.grounding
            flat_prompt = f"[EDIT: {user_input}] {prev_prompt}"
            if add_to_history:
                self.previous_prompts.append(flat_prompt)
            return flat_prompt
    
    def add_to_history(self, prompt: str):
        """Add a prompt to history (for manually edited prompts)"""
        self.previous_prompts.append(prompt)
    
    def _call_claude(self, message: str) -> str:
        """Call Claude API to enhance prompt"""
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,  # Increased for JSON output
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": message}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            if self.grounding:
                return f"{self.grounding}, {message}"
            return message
    
    def revert_to(self, chunk_idx: int):
        """Revert prompt history when going back"""
        keep_count = chunk_idx + 1
        if keep_count < 1:
            keep_count = 1
        if keep_count < len(self.previous_prompts):
            self.previous_prompts = self.previous_prompts[:keep_count]
        if keep_count < len(self.structured_states):
            self.structured_states = self.structured_states[:keep_count]


class VideoBuilderState:
    """Web-compatible state manager for interactive video generation"""
    
    def __init__(
        self,
        config_path: str = "configs/longlive_inference.yaml",
        output_dir: str = "videos/interactive_web",
        anthropic_key: Optional[str] = None,
        chunk_duration: float = 10.0,
        max_chunks: int = 12,
        seed: int = 42,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        self.config_path = config_path
        self.output_dir = output_dir
        self.chunk_duration = chunk_duration
        self.max_chunks = max_chunks
        self.seed = seed
        self.progress_callback = progress_callback
        
        # Get API key
        if anthropic_key:
            self.anthropic_key = anthropic_key
        else:
            self.anthropic_key = self._get_anthropic_key()
        
        self.enhancer: Optional[PromptEnhancer] = None
        
        # Frame calculations
        self.fps = 16
        self.temporal_upsample = 4
        video_frames_per_chunk = int(chunk_duration * self.fps)
        self.latent_frames_per_chunk = video_frames_per_chunk // self.temporal_upsample
        self.latent_frames_per_chunk = (self.latent_frames_per_chunk // 3) * 3
        self.frames_per_chunk = self.latent_frames_per_chunk
        
        # State
        self.states: Dict[int, ChunkState] = {}
        self.current_chunk = 0
        self.pipeline = None
        self.device = None
        self.config = None
        self.grounding_set = False
        self.is_setup = False
        
        # Warm-up tracking
        self.needs_warmup = False
        self.warmup_prompt = None
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"session_{self.session_id}")
        
        # Tensors (initialized in setup)
        self.base_noise = None
        self.full_latents = None
        self.max_latent_frames = None
    
    def _get_anthropic_key(self) -> str:
        """Get Anthropic API key from environment or bash_aliases"""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            bash_aliases = "/root/.bash_aliases"
            if os.path.exists(bash_aliases):
                with open(bash_aliases) as f:
                    for line in f:
                        if "ANTHROPIC_API_KEY" in line and "export" in line:
                            api_key = line.split("=")[1].strip().strip('"\'')
                            break
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        return api_key
    
    def _report_progress(self, message: str, progress: float = 0.0):
        """Report progress to callback if available"""
        if self.progress_callback:
            self.progress_callback(message, progress)
    
    def setup(self) -> Dict:
        """Initialize the model and pipeline. Returns status dict."""
        if self.is_setup:
            return {"status": "already_setup", "session_dir": self.session_dir}
        
        self._report_progress("Creating session directory...", 0.05)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Initialize prompt enhancer
        self.enhancer = PromptEnhancer(self.anthropic_key)
        
        # Load config
        self._report_progress("Loading configuration...", 0.1)
        self.config = OmegaConf.load(self.config_path)
        self.config.num_output_frames = self.latent_frames_per_chunk * self.max_chunks
        self.config.seed = self.seed
        self.config.distributed = False
        
        # Setup device
        self.device = torch.device("cuda:0")
        set_seed(self.seed)
        torch.set_grad_enabled(False)
        
        gpu_name = torch.cuda.get_device_name(self.device)
        free_vram = get_cuda_free_memory_gb(self.device)
        
        # Load model
        self._report_progress("Loading model (this takes ~60s)...", 0.2)
        load_start = time.perf_counter()
        
        self.pipeline = CausalInferencePipeline(self.config, device=self.device)
        
        # Load generator checkpoint
        if self.config.generator_ckpt:
            self._report_progress("Loading generator checkpoint...", 0.4)
            state_dict = torch.load(self.config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict or "generator_ema" in state_dict:
                raw_gen_state_dict = state_dict["generator_ema" if self.config.use_ema else "generator"]
            elif "model" in state_dict:
                raw_gen_state_dict = state_dict["model"]
            else:
                raise ValueError("Generator state dict not found")
            self.pipeline.generator.load_state_dict(raw_gen_state_dict)
        
        # LoRA setup
        self._report_progress("Configuring LoRA...", 0.6)
        from utils.lora_utils import configure_lora_for_model
        from utils.memory import DynamicSwapInstaller
        import peft
        
        self.pipeline.is_lora_enabled = False
        if getattr(self.config, "adapter", None):
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
        self._report_progress("Moving model to GPU...", 0.8)
        self.pipeline = self.pipeline.to(dtype=torch.bfloat16)
        DynamicSwapInstaller.install_model(self.pipeline.text_encoder, device=self.device)
        self.pipeline.generator.to(device=self.device)
        self.pipeline.vae.to(device=self.device)
        
        load_time = time.perf_counter() - load_start
        memory_used = torch.cuda.memory_allocated(self.device) / 1e9
        
        # Initialize noise and latent buffers
        self._report_progress("Initializing buffers...", 0.9)
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
        
        self.is_setup = True
        self._report_progress("Setup complete!", 1.0)
        
        return {
            "status": "success",
            "session_id": self.session_id,
            "session_dir": self.session_dir,
            "gpu": gpu_name,
            "vram_free": f"{free_vram:.2f} GB",
            "load_time": f"{load_time:.1f}s",
            "memory_used": f"{memory_used:.2f} GB",
            "chunk_duration": self.chunk_duration,
            "max_chunks": self.max_chunks
        }
    
    def set_grounding(self, grounding: str, skip_ai: bool = False) -> Dict:
        """Set the grounding prompt. Returns enhanced version."""
        if not self.is_setup:
            raise RuntimeError("Call setup() first")
        
        if skip_ai:
            self.enhancer.set_grounding_direct(grounding)
            enhanced = grounding
        else:
            enhanced = self.enhancer.set_grounding(grounding)
        
        self.grounding_set = True
        
        return {
            "user_input": grounding,
            "enhanced": enhanced,
            "skip_ai": skip_ai
        }
    
    def regenerate_grounding(self, grounding: str) -> str:
        """Regenerate grounding enhancement"""
        if not self.is_setup:
            raise RuntimeError("Call setup() first")
        
        self.enhancer.previous_prompts = []
        return self.enhancer.set_grounding(grounding)
    
    def accept_grounding(self, enhanced: str):
        """Accept potentially edited grounding"""
        self.enhancer.grounding = enhanced
        self.enhancer.previous_prompts = [enhanced]
        self.grounding_set = True
    
    def enhance_chunk_prompt(self, user_input: str) -> str:
        """Enhance a chunk prompt without generating"""
        if not self.grounding_set:
            raise RuntimeError("Set grounding first")
        return self.enhancer.enhance_prompt(user_input, add_to_history=False)
    
    def regenerate_chunk_prompt(self, user_input: str) -> str:
        """Regenerate chunk prompt enhancement"""
        return self.enhancer.enhance_prompt(user_input, add_to_history=False)
    
    def generate_chunk(
        self,
        user_prompt: str,
        processed_prompt: str,
        skip_ai: bool = False
    ) -> Dict:
        """Generate a video chunk. Returns status and paths."""
        if not self.grounding_set:
            raise RuntimeError("Set grounding first")
        
        chunk_idx = self.current_chunk
        
        # Add to history if not already added
        if skip_ai:
            self.enhancer.add_to_history(processed_prompt)
        else:
            # Check if this prompt is already in history
            if processed_prompt not in self.enhancer.previous_prompts:
                self.enhancer.add_to_history(processed_prompt)
        
        # Generate
        start_time = time.perf_counter()
        self._generate_chunk_internal(
            chunk_idx,
            user_prompt,
            processed_prompt,
            needs_warmup=self.needs_warmup,
            warmup_prompt=self.warmup_prompt
        )
        gen_time = time.perf_counter() - start_time
        
        # Reset warmup flags
        self.needs_warmup = False
        self.warmup_prompt = None
        
        self.current_chunk += 1
        
        # Get video path
        chunk_video = os.path.join(self.session_dir, f"chunk_{chunk_idx + 1}.mp4")
        running_video = os.path.join(self.session_dir, f"running_{chunk_idx + 1}.mp4")
        
        return {
            "status": "success",
            "chunk_idx": chunk_idx,
            "chunk_num": chunk_idx + 1,
            "time_range": f"{chunk_idx * self.chunk_duration:.0f}-{(chunk_idx + 1) * self.chunk_duration:.0f}s",
            "generation_time": f"{gen_time:.1f}s",
            "chunk_video": chunk_video,
            "running_video": running_video,
            "user_prompt": user_prompt,
            "processed_prompt": processed_prompt
        }
    
    def _initialize_caches(self):
        """Initialize KV caches"""
        # Simple version matching main branch - let PyTorch handle cleanup
        local_attn_size = self.pipeline.local_attn_size
        if local_attn_size != -1:
            kv_cache_size = local_attn_size * self.pipeline.frame_seq_length
        else:
            kv_cache_size = self.latent_frames_per_chunk * self.max_chunks * self.pipeline.frame_seq_length
        
        self.pipeline._initialize_kv_cache(1, torch.bfloat16, self.device, kv_cache_size)
        self.pipeline._initialize_crossattn_cache(1, torch.bfloat16, self.device)
    
    def _copy_cache(self, cache_list):
        """Deep copy a cache list to CPU to save GPU memory"""
        return [{k: v.clone().cpu() if isinstance(v, torch.Tensor) else v 
                 for k, v in cache.items()} for cache in cache_list]
    
    def _clear_state_tensors(self, state: ChunkState):
        """Clear tensors from a ChunkState to free memory"""
        if state.end_latent is not None:
            del state.end_latent
    
    def _cleanup_old_states(self, keep_last_n: int = 2):
        """Remove old states to free GPU memory, keeping only last N for go-back"""
        if len(self.states) <= keep_last_n:
            return
        
        sorted_indices = sorted(self.states.keys())
        indices_to_remove = sorted_indices[:-keep_last_n]
        
        for idx in indices_to_remove:
            if idx in self.states:
                self._clear_state_tensors(self.states[idx])
                del self.states[idx]
        
        gc.collect()
        torch.cuda.empty_cache()
    
    def _recache_after_switch(self, current_start_frame: int, new_conditional_dict: dict):
        """Recache KV with NEW prompt for dramatic changes"""
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
        # Limit recache frames to avoid OOM - use 12 as default window size
        effective_attn_size = local_attn_size if local_attn_size != -1 else 12
        num_recache_frames = min(effective_attn_size, current_start_frame)
        
        recache_start_frame = current_start_frame - num_recache_frames
        frames_to_recache = self.full_latents[:, recache_start_frame:current_start_frame].to(self.device)
        
        self._report_progress(f"Recaching {num_recache_frames} frames...", 0.1)
        
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
                current_start=0,
                sink_recache_after_switch=not global_sink,
            )
        
        self.pipeline.generator.model.block_mask = old_block_mask
        
        for blk in self.pipeline.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
    
    def _run_dummy_generation(self, chunk_idx: int, prompt: str):
        """Run a dummy generation pass to reset model context"""
        start_frame = chunk_idx * self.frames_per_chunk
        end_frame = start_frame + self.frames_per_chunk
        num_blocks = self.frames_per_chunk // self.pipeline.num_frame_per_block
        
        self._report_progress("Running warm-up pass...", 0.05)
        
        conditional_dict = self.pipeline.text_encoder(text_prompts=[prompt])
        
        self._initialize_caches()
        
        if chunk_idx > 0:
            self._recache_after_switch(start_frame, conditional_dict)
        
        local_attn_size = self.pipeline.local_attn_size if self.pipeline.local_attn_size != -1 else 12
        recache_frames = min(local_attn_size, start_frame) if chunk_idx > 0 else 0
        recache_offset = start_frame - recache_frames if chunk_idx > 0 else 0
        
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
        
        # Clean up dummy generation memory
        del conditional_dict, dummy_noise, denoised_pred
        torch.cuda.empty_cache()
    
    def _generate_chunk_internal(
        self,
        chunk_idx: int,
        user_prompt: str,
        processed_prompt: str,
        needs_warmup: bool = False,
        warmup_prompt: str = None
    ):
        """Internal chunk generation logic"""
        start_frame = chunk_idx * self.frames_per_chunk
        end_frame = start_frame + self.frames_per_chunk
        num_blocks = self.frames_per_chunk // self.pipeline.num_frame_per_block
        
        did_warmup = False
        
        if needs_warmup and warmup_prompt and chunk_idx > 0:
            if chunk_idx - 1 in self.states:
                prev_state = self.states[chunk_idx - 1]
                context_frames = prev_state.end_latent.shape[1]
                self.full_latents[:, prev_state.current_frame - context_frames:prev_state.current_frame] = prev_state.end_latent
            
            self._run_dummy_generation(chunk_idx, warmup_prompt)
            did_warmup = True
            
            if chunk_idx - 1 in self.states:
                prev_state = self.states[chunk_idx - 1]
                context_frames = prev_state.end_latent.shape[1]
                self.full_latents[:, prev_state.current_frame - context_frames:prev_state.current_frame] = prev_state.end_latent
        
        self._report_progress(f"Generating frames {start_frame}-{end_frame}...", 0.2)
        
        conditional_dict = self.pipeline.text_encoder(text_prompts=[processed_prompt])
        
        recached = False
        recache_offset = 0
        
        if chunk_idx > 0:
            if chunk_idx - 1 in self.states:
                prev_state = self.states[chunk_idx - 1]
                context_frames = prev_state.end_latent.shape[1]
                self.full_latents[:, prev_state.current_frame - context_frames:prev_state.current_frame] = prev_state.end_latent
            
            if did_warmup:
                for blk in self.pipeline.crossattn_cache:
                    blk["k"].zero_()
                    blk["v"].zero_()
                    blk["is_init"] = False
                
                local_attn_size = self.pipeline.local_attn_size if self.pipeline.local_attn_size != -1 else 12
                recache_frames = min(local_attn_size, start_frame)
                recache_offset = start_frame - recache_frames
                
                recache_cache_pos = recache_frames * self.pipeline.frame_seq_length
                for cache in self.pipeline.kv_cache1:
                    cache["global_end_index"].fill_(recache_cache_pos)
                    cache["local_end_index"].fill_(recache_cache_pos % (local_attn_size * self.pipeline.frame_seq_length))
                
                recached = True
            else:
                self._initialize_caches()
                self._recache_after_switch(start_frame, conditional_dict)
                recached = True
                local_attn_size = self.pipeline.local_attn_size if self.pipeline.local_attn_size != -1 else 12
                recache_frames = min(local_attn_size, start_frame)
                recache_offset = start_frame - recache_frames
        else:
            self._initialize_caches()
        
        current_start_frame = start_frame
        
        for block_idx in range(num_blocks):
            progress = 0.2 + (0.6 * (block_idx + 1) / num_blocks)
            self._report_progress(f"Block {block_idx + 1}/{num_blocks}...", progress)
            
            current_num_frames = self.pipeline.num_frame_per_block
            noisy_input = self.base_noise[:, current_start_frame:current_start_frame + current_num_frames]
            
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
        
        # Save state
        context_frames = min(12, end_frame)
        end_latent = self.full_latents[:, end_frame - context_frames:end_frame].clone()
        
        self.states[chunk_idx] = ChunkState(
            chunk_idx=chunk_idx,
            user_prompt=user_prompt,
            processed_prompt=processed_prompt,
            end_latent=end_latent.cpu(),  # Store on CPU to save GPU memory
            current_frame=end_frame
        )
        
        # Save outputs
        self._report_progress("Saving video...", 0.85)
        self._save_chunk_outputs(chunk_idx, user_prompt, processed_prompt)
        
        # Clean up old states to free GPU memory (keep last 2 for go-back)
        self._cleanup_old_states(keep_last_n=2)
        
        # Aggressive memory cleanup after generation
        del conditional_dict
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        self._report_progress("Done!", 1.0)
    
    def _decode_latents_chunked(self, latents, chunk_size=80):
        """Decode latents in chunks to avoid OOM"""
        num_frames = latents.shape[1]
        all_video_chunks = []
        
        for start in range(0, num_frames, chunk_size):
            end = min(start + chunk_size, num_frames)
            chunk_latents = latents[:, start:end].to(self.device)
            chunk_video = self.pipeline.vae.decode_to_pixel(chunk_latents, use_cache=False)
            chunk_video = (chunk_video * 0.5 + 0.5).clamp(0, 1)
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
    
    def _save_history(self):
        """Save session history"""
        history = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "grounding": self.enhancer.grounding if self.enhancer else None,
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
    
    def get_chunks_info(self) -> List[Dict]:
        """Get info about all generated chunks"""
        chunks = []
        for idx in range(self.current_chunk):
            if idx in self.states:
                state = self.states[idx]
                chunks.append({
                    "chunk_num": idx + 1,
                    "time_range": f"{idx * self.chunk_duration:.0f}-{(idx + 1) * self.chunk_duration:.0f}s",
                    "user_prompt": state.user_prompt,
                    "processed_prompt": state.processed_prompt
                })
        return chunks
    
    def goto_chunk(self, target_chunk: int) -> Dict:
        """Go back to a specific chunk for regeneration"""
        if target_chunk < 1 or target_chunk > self.current_chunk:
            return {
                "status": "error",
                "message": f"Invalid chunk. Valid range: 1-{self.current_chunk}"
            }
        
        target_idx = target_chunk - 1
        
        if target_idx > 0 and target_idx - 1 in self.states:
            prev_state = self.states[target_idx - 1]
            context_frames = prev_state.end_latent.shape[1]
            self.full_latents[:, prev_state.current_frame - context_frames:prev_state.current_frame] = prev_state.end_latent
            self.full_latents[:, prev_state.current_frame:].zero_()
        else:
            self.full_latents.zero_()
        
        # Regenerate noise
        start_frame = target_idx * self.latent_frames_per_chunk
        self.base_noise[:, start_frame:] = torch.randn(
            1, self.max_latent_frames - start_frame,
            *self.base_noise.shape[2:],
            device=self.device, dtype=torch.bfloat16
        )
        
        # Remove states
        for idx in list(self.states.keys()):
            if idx >= target_idx:
                del self.states[idx]
        
        # Reinitialize caches
        self._initialize_caches()
        
        if hasattr(self.pipeline.generator.model, 'block_mask'):
            default_mask = self.pipeline.generator.model._prepare_blockwise_causal_attn_mask(
                device=self.device,
                num_frames=self.frames_per_chunk,
                frame_seqlen=self.pipeline.frame_seq_length,
                num_frame_per_block=self.pipeline.num_frame_per_block,
                local_attn_size=self.pipeline.local_attn_size
            )
            self.pipeline.generator.model.block_mask = default_mask
        
        # Revert prompt history
        self.enhancer.revert_to(target_idx - 1 if target_idx > 0 else 0)
        
        # Set warm-up
        self.needs_warmup = True
        if target_idx > 0 and target_idx - 1 in self.states:
            self.warmup_prompt = self.states[target_idx - 1].processed_prompt
        else:
            self.warmup_prompt = self.enhancer.grounding
        
        self.current_chunk = target_idx
        
        return {
            "status": "success",
            "message": f"Rewound to chunk {target_chunk}",
            "current_chunk": target_chunk
        }
    
    def go_back(self) -> Dict:
        """Go back one chunk"""
        if self.current_chunk > 0:
            return self.goto_chunk(self.current_chunk)
        return {"status": "error", "message": "Already at first chunk"}
    
    def finalize(self) -> Dict:
        """Save the final video"""
        if not self.states:
            return {"status": "error", "message": "No chunks generated!"}
        
        max_chunk = max(self.states.keys())
        
        running_path = os.path.join(self.session_dir, f"running_{max_chunk + 1}.mp4")
        final_path = os.path.join(self.session_dir, "final_video.mp4")
        
        if os.path.exists(running_path):
            import shutil
            shutil.copy(running_path, final_path)
        
        self._save_history()
        
        return {
            "status": "success",
            "final_video": final_path,
            "session_dir": self.session_dir,
            "total_chunks": max_chunk + 1,
            "duration": f"~{(max_chunk + 1) * self.chunk_duration:.0f}s"
        }
    
    def get_status(self) -> Dict:
        """Get current builder status"""
        return {
            "is_setup": self.is_setup,
            "grounding_set": self.grounding_set,
            "grounding": self.enhancer.grounding if self.enhancer else None,
            "current_chunk": self.current_chunk,
            "max_chunks": self.max_chunks,
            "session_id": self.session_id,
            "session_dir": self.session_dir,
            "chunks_generated": len(self.states)
        }
    
    def reset(self) -> Dict:
        """Reset the session for a new video generation.
        
        Clears all caches, prompts, and state while keeping the model loaded.
        """
        if not self.is_setup:
            return {"status": "error", "message": "Model not setup yet"}
        
        # Clear GPU tensors from all states before resetting
        for state in self.states.values():
            self._clear_state_tensors(state)
        
        self.states = {}
        self.current_chunk = 0
        self.grounding_set = False
        
        # Force garbage collection
        gc.collect()
        
        # Reset warm-up tracking
        self.needs_warmup = False
        self.warmup_prompt = None
        
        # Reset prompt enhancer
        if self.enhancer:
            self.enhancer.grounding = None
            self.enhancer.previous_prompts = []
            self.enhancer.structured_states = []
        
        # Create new session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.output_dir, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Reset latent buffer
        if self.full_latents is not None:
            self.full_latents.zero_()
        
        # Regenerate base noise with new seed offset
        if self.base_noise is not None:
            new_seed = self.seed + int(datetime.now().timestamp()) % 10000
            self.base_noise = torch.randn(
                [1, self.max_latent_frames, 16, 60, 104],
                device=self.device,
                dtype=torch.bfloat16,
                generator=torch.Generator(device=self.device).manual_seed(new_seed)
            )
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        return {
            "status": "success",
            "message": "Session reset. Ready for new video.",
            "new_session_id": self.session_id,
            "session_dir": self.session_dir
        }
