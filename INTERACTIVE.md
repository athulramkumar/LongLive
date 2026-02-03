# Interactive Video Generation with LongLive

This document explains how to use the interactive terminal-based video generation demo and provides a code walkthrough of the key components.

## Quick Start

### Prerequisites

1. Activate the conda environment:
```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /venv/longlive
```

2. Ensure you have an Anthropic API key in `/root/.bash_aliases` or set it manually:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Running the Demo

```bash
cd /workspace/longlive/LongLive
python interactive_demo_anthropic.py
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `configs/longlive_inference.yaml` | Model configuration file |
| `--output_dir` | `videos/interactive_ai` | Output directory for sessions |
| `--chunk_duration` | `10.0` | Duration of each video chunk in seconds |
| `--max_chunks` | `6` | Maximum number of chunks (total video length) |
| `--seed` | `42` | Random seed for reproducibility |

Example with custom settings:
```bash
python interactive_demo_anthropic.py --chunk_duration 5.0 --max_chunks 12 --seed 123
```

---

## Interactive Commands

Once the model is loaded, you'll be prompted to enter commands:

| Command | Description |
|---------|-------------|
| `<prompt>` | Enter any text - AI will enhance it, then you can review |
| `No AI: <prompt>` | Use prompt directly without AI enhancement |
| `list` | Show all generated chunks with their prompts |
| `back` | Go back one chunk to re-edit it |
| `goto N` | Go to chunk N (1-indexed) for re-editing |
| `done` | Finish and save the final concatenated video |
| `quit` | Exit without saving |

### Prompt Review Options

After AI enhances your prompt, you'll see the full enhanced text and can:

| Option | Description |
|--------|-------------|
| `Enter` | Accept the enhanced prompt and generate |
| `r` | Regenerate - get a new AI enhancement |
| `e` | Edit - manually modify the prompt |

### Workflow Example

```
Grounding: A majestic eagle soaring through mountain peaks
  [AI enhances...]
  ┌─ AI Enhanced Grounding ─────────────────────────────────
  │ A majestic bald eagle with golden-brown plumage soars 
  │ majestically through snow-capped mountain peaks...
  └────────────────────────────────────────────────────────
  Options: [Enter]=Accept, [r]=Regenerate, [e]=Edit manually
  Your choice: <Enter>

[Chunk 1] Your prompt: it glides through clouds
  [AI enhances...]
  ┌─ AI Enhanced Prompt ─────────────────────────────────
  │ The majestic bald eagle glides effortlessly through 
  │ wispy white clouds, wings fully extended...
  └────────────────────────────────────────────────────────
  Your choice: <Enter>

[Chunk 2] Your prompt: No AI: The eagle dives sharply toward a crystal lake below
  [Using prompt directly without AI enhancement]

[Chunk 3] Your prompt: back           <-- Go back to chunk 2
[Chunk 2] Your prompt: it transforms into a dragon
  [AI enhances...]
  Your choice: e                      <-- Edit the AI output
  > The eagle's form shimmers and morphs, feathers becoming scales...
  
[Chunk 3] Your prompt: done           <-- Finalize video
```

### Tips for Better AI Enhancement

The AI prompt enhancer is designed to be **concise and focused**. It follows this structure:

1. **GROUNDING**: Subject + current visual state (from initial or evolved state)
2. **CHANGE**: What is visually different - specific parts, colors, positions
3. **MOTION**: How things move temporally - action verbs, direction, speed

**Example output format:**
```
Red sports car's panels split and retract, hood folding into torso, wheels becoming legs. 
Transformation completes while vehicle maintains forward momentum.
```

**What the AI avoids:**
- Fluffy adjectives: "dramatic", "majestic", "beautiful", "elegant"
- Vague descriptions: "begins to transform", "starts changing"
- Atmosphere over action: "moody lighting", "sense of anticipation"

**What the AI focuses on:**
- Specific visual changes: "panels split", "hair turns gray", "wings unfold"
- Motion descriptions: "launches forward", "rotates outward", "strikes ground"
- Spatial details: "hood becomes torso", "wheels form legs"

If the AI isn't following your intent:
- Use `r` to regenerate
- Use `e` to edit manually  
- Use `No AI:` to bypass entirely

---

## Code Architecture

### File: `interactive_demo_anthropic.py`

The main script is organized into these key classes:

### 1. `ChunkState` (lines 37-47)

A dataclass that stores the state after generating each chunk:

```python
@dataclass
class ChunkState:
    chunk_idx: int              # 0-based chunk index
    user_prompt: str            # Original user input (e.g., "it flies faster")
    processed_prompt: str       # AI-enhanced prompt (detailed description)
    end_latent: torch.Tensor    # Last 12 frames of latent for continuity
    kv_cache: List[Dict]        # Self-attention KV cache state
    crossattn_cache: List[Dict] # Cross-attention cache state  
    current_frame: int          # Frame index at end of this chunk
    timestamp: str              # When this chunk was generated
```

### 2. `PromptEnhancer` (lines 50-147)

Handles Anthropic API integration for prompt enhancement:

- **`set_grounding(grounding)`**: Takes initial subject/setting, enhances it, stores as anchor
- **`enhance_prompt(user_input)`**: Expands short prompts using grounding context
- **`revert_to(chunk_idx)`**: Reverts prompt history when going back

### 3. `InteractiveVideoBuilderWithAI` (lines 149-880+)

Main class that orchestrates video generation:

---

## Key Functions Explained

### Initialization (`__init__` and `setup`)

**Location**: Lines 152-290

```python
def __init__(self, config_path, output_dir, anthropic_key, chunk_duration=10.0, max_chunks=12, seed=42):
    # Frame calculation (accounting for 4x temporal upsampling by VAE)
    self.fps = 16
    self.temporal_upsample = 4
    video_frames_per_chunk = int(chunk_duration * self.fps)  # e.g., 160 frames for 10s
    self.latent_frames_per_chunk = video_frames_per_chunk // self.temporal_upsample  # e.g., 40
    self.latent_frames_per_chunk = (self.latent_frames_per_chunk // 3) * 3  # Round to 39 (divisible by 3)
    
    # State tracking
    self.states: Dict[int, ChunkState] = {}  # Stores state for each chunk
    self.current_chunk = 0                    # Current chunk being generated
    self.needs_warmup = False                 # Flag for back/goto operations
    self.warmup_prompt = None                 # Prompt to use for warm-up pass
```

### Cache Initialization (`_initialize_caches`)

**Location**: Lines 282-291

Initializes the KV caches for the transformer model:

```python
def _initialize_caches(self):
    local_attn_size = self.pipeline.local_attn_size  # Usually 12 frames
    if local_attn_size != -1:
        kv_cache_size = local_attn_size * self.pipeline.frame_seq_length
    else:
        kv_cache_size = self.latent_frames_per_chunk * self.max_chunks * self.pipeline.frame_seq_length
    
    self.pipeline._initialize_kv_cache(1, torch.bfloat16, self.device, kv_cache_size)
    self.pipeline._initialize_crossattn_cache(1, torch.bfloat16, self.device)
```

### Recaching After Prompt Switch (`_recache_after_switch`)

**Location**: Lines 298-357

When switching prompts (either after going back or between chunks), this function:
1. Zeros the KV cache
2. Recaches the last `local_attn_size` frames (12 frames) with the new prompt conditioning
3. Prepares the cache for generating the next frames

```python
def _recache_after_switch(self, current_start_frame: int, new_conditional_dict: dict):
    # Zero all cache values and indices
    for cache in self.pipeline.kv_cache1:
        cache["k"].zero_()
        cache["v"].zero_()
        cache["global_end_index"].zero_()
        cache["local_end_index"].zero_()
    
    # Calculate how many frames to recache
    local_attn_size = self.pipeline.local_attn_size
    num_recache_frames = min(local_attn_size, current_start_frame)  # Usually 12
    
    # Get the frames from full_latents and run through model with new prompt
    recache_start_frame = current_start_frame - num_recache_frames
    frames_to_recache = self.full_latents[:, recache_start_frame:current_start_frame]
    
    # Run model forward pass to populate cache
    self.pipeline.generator(
        noisy_image_or_video=frames_to_recache,
        conditional_dict=new_conditional_dict,  # NEW prompt's text embeddings
        timestep=context_timestep,
        kv_cache=self.pipeline.kv_cache1,
        crossattn_cache=self.pipeline.crossattn_cache,
        current_start=0,
    )
```

### Warm-up Generation (`_run_dummy_generation`)

**Location**: Lines 358-423

When going back to re-edit a chunk, this function runs a "dummy" generation pass to reset the model's context:

```python
def _run_dummy_generation(self, chunk_idx: int, prompt: str):
    """Run a dummy generation pass to reset model context (output discarded)"""
    # Uses the PREVIOUS chunk's prompt (e.g., chunk 1's prompt when re-editing chunk 2)
    # This establishes a "clean" continuation context before generating with the new prompt
    
    # 1. Initialize fresh cache
    self._initialize_caches()
    
    # 2. Recache with previous chunk's prompt
    self._recache_after_switch(start_frame, conditional_dict)
    
    # 3. Generate full chunk (output discarded, but fills cache)
    for block_idx in range(num_blocks):
        # ... denoising loop ...
        # Cache is populated with continuation context
```

### Chunk Generation (`generate_chunk`)

**Location**: Lines 425-530

The main generation function:

```python
def generate_chunk(self, chunk_idx, user_prompt, processed_prompt, needs_warmup=False, warmup_prompt=None):
    # 1. If coming back from a later chunk, do warm-up first
    if needs_warmup and warmup_prompt and chunk_idx > 0:
        self._run_dummy_generation(chunk_idx, warmup_prompt)
        did_warmup = True
    
    # 2. Create text embeddings for this prompt
    conditional_dict = self.pipeline.text_encoder(text_prompts=[processed_prompt])
    
    # 3. Handle cache based on whether we did warm-up
    if chunk_idx > 0:
        if did_warmup:
            # Keep self-attention KV cache from warm-up
            # Only reset cross-attention for new text conditioning
            for blk in self.pipeline.crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False
            # Reset cache position indices to chunk start
        else:
            # Normal case: full reinit and recache
            self._initialize_caches()
            self._recache_after_switch(start_frame, conditional_dict)
    
    # 4. Generate frames block by block (3 frames per block)
    for block_idx in range(num_blocks):
        # Denoising loop for each block
        for step_idx, current_timestep in enumerate(self.pipeline.denoising_step_list):
            _, denoised_pred = self.pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=self.pipeline.kv_cache1,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=cache_pos
            )
        
        # Store denoised latent
        self.full_latents[:, current_start:current_start + 3] = denoised_pred.cpu()
        
        # Update cache with context for next block
        self.pipeline.generator(..., timestep=context_timestep)
    
    # 5. Save state for potential back/goto
    self.states[chunk_idx] = ChunkState(
        chunk_idx=chunk_idx,
        user_prompt=user_prompt,
        processed_prompt=processed_prompt,
        end_latent=self.full_latents[:, end_frame-12:end_frame].clone(),  # Last 12 frames
        kv_cache=self._copy_cache(self.pipeline.kv_cache1),
        crossattn_cache=self._copy_cache(self.pipeline.crossattn_cache),
        current_frame=end_frame
    )
```

---

## Back/Goto Logic

### `goto_chunk` Function

**Location**: Lines 673-740

This is the core function for rewinding to a previous chunk:

```python
def goto_chunk(self, target_chunk: int):
    """Go back to a specific chunk for regeneration"""
    target_idx = target_chunk - 1  # Convert 1-indexed to 0-indexed
    
    # 1. Restore latent state from chunk BEFORE the target
    if target_idx > 0 and target_idx - 1 in self.states:
        prev_state = self.states[target_idx - 1]
        # Restore the end_latent (last 12 frames) from previous chunk
        self.full_latents[:, prev_state.current_frame - 12:prev_state.current_frame] = prev_state.end_latent
        # Zero out everything after to prevent stale data
        self.full_latents[:, prev_state.current_frame:].zero_()
    else:
        # Going back to chunk 1 - zero everything
        self.full_latents.zero_()
    
    # 2. Regenerate noise for different results
    start_frame = target_idx * self.latent_frames_per_chunk
    self.base_noise[:, start_frame:] = torch.randn(...)
    
    # 3. Delete states for chunks >= target
    for idx in list(self.states.keys()):
        if idx >= target_idx:
            del self.states[idx]
    
    # 4. Reinitialize caches
    self._initialize_caches()
    
    # 5. Revert prompt history
    self.enhancer.revert_to(target_idx - 1 if target_idx > 0 else 0)
    
    # 6. Set warm-up flag for next generation
    self.needs_warmup = True
    if target_idx > 0 and target_idx - 1 in self.states:
        # Use previous chunk's prompt for warm-up
        self.warmup_prompt = self.states[target_idx - 1].processed_prompt
    else:
        self.warmup_prompt = self.enhancer.grounding
    
    self.current_chunk = target_idx
```

### Main Run Loop

**Location**: Lines 790-870

```python
def run(self):
    # ... grounding setup ...
    
    while self.current_chunk < self.max_chunks:
        user_input = input("Your prompt/change: ").strip()
        
        if user_input.lower() == 'back':
            if self.current_chunk > 0:
                self.goto_chunk(self.current_chunk)  # Go back to current (re-edit)
            continue
        
        if user_input.lower().startswith('goto '):
            target = int(user_input.split()[1])
            self.goto_chunk(target)
            continue
        
        if user_input.lower() == 'done':
            self.finalize()
            break
        
        # Enhance prompt and generate
        processed_prompt = self.enhancer.enhance_prompt(user_input)
        self.generate_chunk(
            self.current_chunk, 
            user_input, 
            processed_prompt,
            needs_warmup=self.needs_warmup,  # True if coming from back/goto
            warmup_prompt=self.warmup_prompt
        )
        
        # Reset warmup flags
        self.needs_warmup = False
        self.warmup_prompt = None
        
        self.current_chunk += 1
```

---

## What Gets Cached After Each Chunk

After each chunk generation, the following are stored in `ChunkState`:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_idx` | int | 0-based chunk index |
| `user_prompt` | str | Original user input (short) |
| `processed_prompt` | str | AI-enhanced detailed prompt |
| `end_latent` | Tensor `[1, 12, 16, 60, 104]` | Last 12 latent frames (for continuity) |
| `kv_cache` | List[Dict] | Self-attention KV cache state |
| `crossattn_cache` | List[Dict] | Cross-attention cache state |
| `current_frame` | int | Frame index at chunk end |
| `timestamp` | str | ISO timestamp |

### KV Cache Structure

Each layer's KV cache dict contains:
```python
{
    "k": Tensor,              # Key vectors
    "v": Tensor,              # Value vectors  
    "global_end_index": int,  # Global position index
    "local_end_index": int    # Local attention window index
}
```

### Cross-Attention Cache Structure

```python
{
    "k": Tensor,      # Text-projected keys
    "v": Tensor,      # Text-projected values
    "is_init": bool   # Whether cache has been initialized
}
```

---

## Output Files

Each session creates a folder: `videos/interactive_ai/session_YYYYMMDD_HHMMSS/`

| File | Description |
|------|-------------|
| `chunk_N.mp4` | Individual chunk N video |
| `chunk_N_prompts.json` | User and AI prompts for chunk N |
| `running_N.mp4` | Concatenated video up to chunk N |
| `final_video.mp4` | Final concatenated video |
| `history.json` | Complete session history |

### Example `history.json`

```json
{
  "session_id": "20260202_204718",
  "grounding": "A majestic eagle",
  "settings": {
    "chunk_duration": 10.0,
    "latent_frames_per_chunk": 39,
    "fps": 16
  },
  "chunks": [
    {
      "idx": 0,
      "user_prompt": "it glides through clouds",
      "processed_prompt": "A majestic eagle glides gracefully through wispy white clouds...",
      "time_range": "0-10s"
    }
  ]
}
```

---

## Technical Notes

### Frame Calculations

- **Video FPS**: 16 frames/second
- **VAE Temporal Upsampling**: 4x (39 latent frames → 156 video frames)
- **Chunk Duration**: 10s = 160 video frames ≈ 39 latent frames
- **Block Size**: 3 frames per denoising block
- **Local Attention Window**: 12 frames

### Memory Optimization

- Only the last 12 latent frames (`end_latent`) are stored per chunk, not all frames
- Video concatenation uses `ffmpeg` on CPU to avoid re-decoding all latents
- Large latent tensors are decoded in chunks to prevent OOM

### Warm-up Pass Rationale

When going back to re-edit a chunk, the model's attention patterns may be influenced by the previous generation. The warm-up pass:
1. Generates the chunk with the PREVIOUS chunk's prompt (discarded)
2. This "resets" the model to a clean continuation state
3. Then generates with the NEW prompt using the established context
