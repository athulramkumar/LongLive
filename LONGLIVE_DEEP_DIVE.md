# LongLive: Complete Technical Deep Dive

> A comprehensive guide to understanding the LongLive repository - from prompt to video generation, including all special recipes and innovations.

## Table of Contents

1. [Overview](#overview)
2. [Repository Architecture](#repository-architecture)
3. [The Complete Diffusion Pipeline: Prompt → Video](#the-complete-diffusion-pipeline-prompt--video)
4. [The Causal Transformer Architecture](#the-causal-transformer-architecture)
5. [Special Recipes: The 4 Key Innovations](#special-recipes-the-4-key-innovations)
6. [Flow Matching Scheduler](#flow-matching-scheduler)
7. [Training Pipeline](#training-pipeline)
8. [Inference Modes](#inference-modes)
9. [Key Numbers & Configuration](#key-numbers--configuration)
10. [Summary of Innovations](#summary-of-innovations)

---

## Overview

**LongLive** is a real-time interactive long video generation framework that enables:
- **Real-time generation** at 20.7 FPS on a single H100 GPU
- **Long video generation** up to 240 seconds
- **Interactive prompt switching** mid-video
- **Efficient fine-tuning** - extends a short-clip model to minute-long generation in 32 GPU-days

The key insight is transforming a **bidirectional diffusion model** (slow) into a **frame-level autoregressive (AR) causal model** (fast) with KV caching.

### Paper & Resources
- **Paper**: [LongLive: Real-time Interactive Long Video Generation](https://arxiv.org/abs/2509.22622)
- **Model**: [HuggingFace - LongLive-1.3B](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B)
- **Demo**: [Project Page](https://nvlabs.github.io/LongLive)

---

## Repository Architecture

```
LongLive/
├── inference.py                    # Single-prompt inference entry point
├── interactive_inference.py        # Multi-prompt interactive inference
├── profiled_inference.py           # Performance profiling
├── train.py                        # Training entry point
│
├── model/                          # DMD training models
│   ├── base.py                     # Base model with backward simulation
│   ├── dmd.py                      # Distribution Matching Distillation
│   └── dmd_switch.py               # DMD with prompt switching
│
├── pipeline/                       # Core inference/training pipelines
│   ├── causal_inference.py         # Single-prompt inference
│   ├── interactive_causal_inference.py  # Interactive inference with KV-recache
│   ├── self_forcing_training.py    # Step 1 training (initialization)
│   ├── streaming_training.py       # Step 2 streaming long tuning
│   └── streaming_switch_training.py # Prompt switch training
│
├── wan/                            # Modified Wan2.1 base model
│   ├── modules/
│   │   ├── causal_model.py         # Causal transformer with KV cache
│   │   ├── causal_model_infinity.py # Infinite-length generation
│   │   ├── model.py                # Original bidirectional model
│   │   ├── attention.py            # Flash/Flex attention
│   │   ├── vae.py                  # Video VAE
│   │   └── t5.py                   # UMT5-XXL text encoder
│   ├── text2video.py               # WanT2V model class
│   └── image2video.py              # WanI2V model class
│
├── trainer/
│   └── distillation.py             # Score Distillation Trainer
│
├── utils/
│   ├── wan_wrapper.py              # Wrappers for Wan components
│   ├── scheduler.py                # Flow matching scheduler
│   ├── loss.py                     # Loss functions
│   ├── dataset.py                  # Dataset classes
│   ├── distributed.py              # FSDP utilities
│   └── lora_utils.py               # LoRA configuration
│
├── configs/                        # YAML configurations
│   ├── longlive_inference.yaml     # Single-prompt inference
│   ├── longlive_interactive_inference.yaml # Interactive inference
│   ├── longlive_train_init.yaml    # Step 1 training config
│   └── longlive_train_long.yaml    # Step 2 streaming training config
│
└── example/                        # Example prompts
    ├── interactive_example.jsonl   # Multi-prompt examples
    └── long_example.txt            # Single-prompt examples
```

---

## The Complete Diffusion Pipeline: Prompt → Video

### Step 1: Text Encoding

```python
# utils/wan_wrapper.py - WanTextEncoder
class WanTextEncoder(torch.nn.Module):
    def __init__(self):
        # UMT5-XXL encoder (frozen, ~4.7B parameters)
        self.text_encoder = umt5_xxl(encoder_only=True)
        self.tokenizer = HuggingfaceTokenizer(seq_len=512)
    
    def forward(self, text_prompts: List[str]) -> dict:
        # 1. Tokenize text using UMT5-XXL tokenizer (max 512 tokens)
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        
        # 2. Encode through T5 text encoder
        context = self.text_encoder(ids, mask)  # Shape: [B, 512, 4096]
        
        # 3. Zero out padding positions
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0
        
        return {"prompt_embeds": context}
```

### Step 2: Noise Initialization

```python
# inference.py
# Generate random noise in latent space
sampled_noise = torch.randn(
    [batch_size, num_output_frames, 16, 60, 104],  # [B, T, C, H, W]
    device=device, 
    dtype=torch.bfloat16
)
# 16 channels, 60×104 spatial dims = 480×832 pixels after VAE decoding
# T = number of latent frames (e.g., 120 = 30 seconds at 4fps latent)
```

### Step 3: Causal Frame-by-Frame Denoising

This is the **core innovation** - instead of bidirectional attention across all frames, LongLive uses frame-level autoregressive generation with KV caching.

#### 3.1. Initialize KV Cache

```python
# pipeline/causal_inference.py
def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override):
    """
    KV Cache Structure:
    - One cache dict per transformer block (30 blocks for 1.3B model)
    - Each contains K and V tensors + index tracking
    """
    kv_cache = []
    for _ in range(30):  # 30 transformer blocks
        kv_cache.append({
            "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0]),  # Tracks total frames processed
            "local_end_index": torch.tensor([0])    # Tracks local window position
        })
    return kv_cache

# Cache size calculation:
# local_attn_size=12 frames × 1560 tokens/frame = 18,720 tokens
# Plus sink tokens = 3 frames × 1560 = 4,680 tokens
```

#### 3.2. Block-wise Temporal Denoising

```python
# pipeline/causal_inference.py - inference() method
def inference(self, noise, text_prompts, ...):
    # Encode text prompt
    conditional_dict = self.text_encoder(text_prompts)
    
    # Initialize caches
    self._initialize_kv_cache(batch_size, dtype, device, kv_cache_size)
    self._initialize_crossattn_cache(batch_size, dtype, device)
    
    # Configure attention
    self.generator.model.local_attn_size = self.local_attn_size  # 12 frames
    
    # Process frame blocks (e.g., 3 frames at a time)
    all_num_frames = [self.num_frame_per_block] * num_blocks  # [3, 3, 3, ...]
    
    for current_num_frames in all_num_frames:
        noisy_input = noise[:, current_start:current_start + current_num_frames]
        
        # Spatial denoising loop (e.g., 4 steps: 1000→750→500→250)
        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones([B, current_num_frames]) * current_timestep
            
            # Forward pass through generator with KV cache
            _, denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start * self.frame_seq_length  # 1560 tokens/frame
            )
            
            # Add noise for next step (except final step)
            if index < len(self.denoising_step_list) - 1:
                next_t = self.denoising_step_list[index + 1]
                noisy_input = self.scheduler.add_noise(denoised_pred, noise, next_t)
        
        # Record output
        output[:, current_start:current_start + current_num_frames] = denoised_pred
        
        # CRITICAL: Update KV cache with CLEAN context (Frame Sink mechanism)
        context_timestep = torch.ones_like(timestep) * self.args.context_noise  # Usually 0
        self.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=self.kv_cache1,  # Updates cache in-place
            crossattn_cache=self.crossattn_cache,
            current_start=current_start * self.frame_seq_length
        )
        
        current_start += current_num_frames
    
    # Decode latents to pixels
    video = self.vae.decode_to_pixel(output)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    return video
```

### Step 4: VAE Decoding

```python
# utils/wan_wrapper.py - WanVAEWrapper
class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        # Normalization constants for Wan VAE (16 channels)
        self.mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ])
        self.std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ])
        self.model = _video_vae(pretrained_path="...", z_dim=16)
    
    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Input: latent [B, T, 16, H_lat, W_lat] (e.g., [1, 120, 16, 60, 104])
        Output: pixels [B, T, 3, H, W] (e.g., [1, 120, 3, 480, 832])
        
        VAE upscales: 8× spatial, 4× temporal (compressed)
        """
        zs = latent.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        scale = [self.mean, 1.0 / self.std]
        
        output = []
        for u in zs:
            decoded = self.model.decode(u.unsqueeze(0), scale)
            output.append(decoded.float().clamp_(-1, 1).squeeze(0))
        
        output = torch.stack(output, dim=0)
        return output.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
```

---

## The Causal Transformer Architecture

### CausalWanModel Overview

Located in `wan/modules/causal_model.py`:

```python
class CausalWanModel(ModelMixin, ConfigMixin):
    """
    Modified Wan2.1 backbone with:
    - Causal (autoregressive) attention instead of bidirectional
    - KV caching for efficient sequential generation
    - Local attention window + frame sink for long sequences
    """
    
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),     # Temporal, Height, Width patch sizes
        text_len=512,             # Max text token length
        in_dim=16,                # VAE latent channels
        dim=2048,                 # Hidden dimension (1536 for 1.3B)
        ffn_dim=8192,             # FFN dimension (6144 for 1.3B)
        freq_dim=256,             # Time embedding dimension
        text_dim=4096,            # T5 embedding dimension
        out_dim=16,               # Output channels
        num_heads=16,             # Attention heads (12 for 1.3B)
        num_layers=32,            # Transformer blocks (30 for 1.3B)
        local_attn_size=-1,       # Local window size (-1 = global)
        sink_size=0,              # Frame sink size
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6
    ):
        super().__init__()
        
        # Patch embedding: [B, C, T, H, W] -> [B, T*H'*W', dim]
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        
        # Text embedding: [B, 512, 4096] -> [B, 512, dim]
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), 
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        
        # Time embedding: scalar -> [B, dim]
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), 
            nn.SiLU(), 
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(dim, dim * 6)  # 6 modulation parameters
        )
        
        # Causal attention blocks with local window + sink
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(
                cross_attn_type='t2v_cross_attn',
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                local_attn_size=local_attn_size,  # Default: 12 frames
                sink_size=sink_size,               # Default: 3 frames
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                eps=eps
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.head = CausalHead(dim, out_dim, patch_size, eps)
        
        # RoPE frequencies
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)
```

### Causal Self-Attention with KV Cache

```python
class CausalWanSelfAttention(nn.Module):
    """
    Self-attention with:
    - Causal masking (can only attend to past)
    - KV caching for autoregressive generation
    - Local attention window for memory efficiency
    - Frame sink for global consistency
    """
    
    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        
        # Compute max attention size
        # local_attn_size=12, frame_seq_length=1560 -> 18,720 tokens
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560
        
        # Q, K, V projections
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        
        # QK normalization (RMSNorm)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, 
                kv_cache=None, current_start=0, sink_recache_after_switch=False):
        """
        Args:
            x: Input tensor [B, SeqLen, dim]
            seq_lens: Actual sequence lengths [B]
            grid_sizes: [B, 3] containing (F, H, W) for each sample
            freqs: RoPE frequencies
            block_mask: FlexAttention block mask (for training)
            kv_cache: KV cache dict with keys 'k', 'v', 'global_end_index', 'local_end_index'
            current_start: Starting position in full sequence
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        
        # Compute Q, K, V
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        
        if kv_cache is None:
            # Training mode: use FlexAttention with block mask
            roped_query = rope_apply(q, grid_sizes, freqs)
            roped_key = rope_apply(k, grid_sizes, freqs)
            
            # Pad to multiple of 128 for FlexAttention
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            # ... padding ...
            
            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )
        else:
            # Inference mode: use KV cache
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()  # H * W after patching
            current_start_frame = current_start // frame_seqlen
            
            # Apply causal RoPE (with frame offset)
            roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame)
            roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame)
            
            # KV Cache management
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen  # 3 * 1560 = 4,680
            
            # Check if we need to roll (evict old tokens)
            if self.local_attn_size != -1 and need_to_roll:
                # Calculate tokens to evict and roll
                num_evicted = num_new_tokens + local_end - kv_cache_size
                num_rolled = local_end - num_evicted - sink_tokens
                
                # Roll cache: preserve sink, shift local window
                temp_k = kv_cache["k"].clone()
                temp_k[:, sink_tokens:sink_tokens + num_rolled] = \
                    temp_k[:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled]
                
                # Insert new K, V
                temp_k[:, write_start:write_end] = roped_key[:, ...]
                temp_v[:, write_start:write_end] = v[:, ...]
            else:
                # Direct insert (no rolling needed)
                temp_k = kv_cache["k"].clone()
                temp_k[:, local_start:local_end] = roped_key
                temp_v[:, local_start:local_end] = v
            
            # Compute attention with sink + local window
            if sink_tokens > 0:
                # Concatenate: [sink_tokens] + [local_window]
                local_budget = self.max_attention_size - sink_tokens
                k_sink = temp_k[:, :sink_tokens]
                v_sink = temp_v[:, :sink_tokens]
                
                if local_budget > 0:
                    local_start_for_window = max(sink_tokens, local_end - local_budget)
                    k_local = temp_k[:, local_start_for_window:local_end]
                    v_local = temp_v[:, local_start_for_window:local_end]
                    k_cat = torch.cat([k_sink, k_local], dim=1)
                    v_cat = torch.cat([v_sink, v_local], dim=1)
                
                x = attention(roped_query, k_cat, v_cat)  # FlashAttention
            else:
                # Local window only
                window_start = max(0, local_end - self.max_attention_size)
                x = attention(roped_query, temp_k[:, window_start:local_end], temp_v[:, window_start:local_end])
        
        return self.o(x.flatten(2)), cache_update_info
```

### Causal RoPE (Rotary Position Embedding)

```python
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """
    Apply RoPE with frame offset for causal generation.
    
    Standard RoPE: positions start at 0
    Causal RoPE: positions start at start_frame (for KV cache consistency)
    """
    n, c = x.size(2), x.size(3) // 2  # num_heads, head_dim//2
    
    # Split frequencies for 3D: temporal, height, width
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].reshape(seq_len, n, -1, 2))
        
        # Compute frequencies with frame offset
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),  # Temporal
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),  # Height
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)   # Width
        ], dim=-1).reshape(seq_len, 1, -1)
        
        # Apply rotation
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output.append(x_i)
    
    return torch.stack(output).type_as(x)
```

### Block-wise Causal Attention Mask

```python
@staticmethod
def _prepare_blockwise_causal_attn_mask(device, num_frames, frame_seqlen, num_frame_per_block, local_attn_size):
    """
    Creates attention mask for training where:
    - Tokens in each frame block can attend to all tokens in previous blocks
    - Plus all tokens within their own block (bidirectional within block)
    - With optional local attention window
    
    Example for 21 frames, 3 frames/block, local_attn=12:
    Block 0 (frames 0-2): attends to itself
    Block 1 (frames 3-5): attends to blocks 0-1
    Block 2 (frames 6-8): attends to blocks 0-2
    ...but with local window, only last 12 frames are attended
    """
    total_length = num_frames * frame_seqlen  # e.g., 21 * 1560 = 32,760 tokens
    padded_length = math.ceil(total_length / 128) * 128 - total_length
    
    # Compute block end indices
    ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    frame_indices = torch.arange(0, total_length, frame_seqlen * num_frame_per_block, device=device)
    
    for tmp in frame_indices:
        ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block
    
    def attention_mask(b, h, q_idx, kv_idx):
        if local_attn_size == -1:  # Global attention
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
        else:  # Local attention window
            return ((kv_idx < ends[q_idx]) & 
                    (kv_idx >= ends[q_idx] - local_attn_size * frame_seqlen)) | \
                   (q_idx == kv_idx)
    
    return create_block_mask(attention_mask, B=None, H=None, 
                            Q_LEN=total_length + padded_length,
                            KV_LEN=total_length + padded_length,
                            _compile=False, device=device)
```

---

## Special Recipes: The 4 Key Innovations

### 1. Frame Sink (Attention Sink)

**Problem:** In long video generation, early frames lose coherence as they move outside the local attention window.

**Solution:** Keep the **first N frames** (default 3) permanently in the KV cache, acting as "anchors" for global consistency.

```python
# wan/modules/causal_model.py - CausalWanSelfAttention
def forward(self, ..., kv_cache=None):
    sink_tokens = self.sink_size * frame_seqlen  # 3 * 1560 = 4,680 tokens
    
    # When rolling cache, NEVER evict sink tokens
    if need_roll:
        # Shift local window content, but preserve sink
        # [SINK | OLD_LOCAL | NEW] -> [SINK | SHIFTED_LOCAL | NEW]
        cache["k"][:, sink_tokens:sink_tokens + num_rolled] = \
            cache["k"][:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()
        cache["v"][:, sink_tokens:sink_tokens + num_rolled] = \
            cache["v"][:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()
    
    # Attention always includes: [SINK] + [LOCAL_WINDOW]
    k_sink = cache["k"][:, :sink_tokens]      # First 3 frames (permanent)
    k_local = cache["k"][:, local_start:local_end]  # Last 12 frames (sliding)
    k_cat = torch.cat([k_sink, k_local], dim=1)
    
    x = attention(query, k_cat, v_cat)
```

**Visual representation:**
```
Frame:     [0 1 2] [3 4 5 6 7 8 9 10 11 12 13 14] [15 16 17]
           |_SINK_| |_______LOCAL WINDOW________| |_NEW__|
           
After processing frame 17:
           [0 1 2] [6 7 8 9 10 11 12 13 14 15 16 17]
           |_SINK_| |________LOCAL WINDOW_________|
           
Frames 3-5 are evicted, but 0-2 remain forever.
```

**Effect:** First 3 frames act as visual anchors, maintaining global scene consistency even in very long videos.

### 2. KV-Recache for Prompt Switching

**Problem:** When user changes prompt mid-video, the KV cache contains features computed with the OLD prompt's cross-attention. This causes visual discontinuity.

**Solution:** Re-run the last N frames through the model with the NEW prompt to refresh both self-attention and cross-attention caches.

```python
# pipeline/interactive_causal_inference.py
def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
    """
    Called when prompt switches. Refreshes KV cache with new prompt context.
    """
    # 1. Reset cross-attention cache (prompt-dependent)
    for blk in self.crossattn_cache:
        blk["k"].zero_()
        blk["v"].zero_()
        blk["is_init"] = False
    
    # 2. Optionally reset self-attention KV cache
    if not self.global_sink:
        # Full reset - allows complete refresh
        for cache in self.kv_cache1:
            cache["k"].zero_()
            cache["v"].zero_()
    # If global_sink=True, keep sink tokens (first 3 frames)
    
    # 3. Determine frames to recache
    num_recache_frames = min(self.local_attn_size, current_start_frame)
    recache_start = current_start_frame - num_recache_frames
    frames_to_recache = output[:, recache_start:current_start_frame]
    
    # 4. Prepare causal attention mask for recaching
    block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
        device=device,
        num_frames=num_recache_frames,
        frame_seqlen=self.frame_seq_length,
        num_frame_per_block=self.num_frame_per_block,
        local_attn_size=self.local_attn_size
    )
    self.generator.model.block_mask = block_mask
    
    # 5. Re-compute KV cache with NEW prompt
    context_timestep = torch.zeros([B, num_recache_frames])  # Clean context (t=0)
    with torch.no_grad():
        self.generator(
            noisy_image_or_video=frames_to_recache,
            conditional_dict=new_conditional_dict,  # NEW prompt embeddings!
            timestep=context_timestep,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=recache_start * self.frame_seq_length,
            sink_recache_after_switch=not self.global_sink
        )
    
    # 6. Reset cross-attention again (will be recomputed during generation)
    for blk in self.crossattn_cache:
        blk["k"].zero_()
        blk["v"].zero_()
        blk["is_init"] = False
```

**Visual Timeline:**
```
Frame:      0  10  20  30  40  50  60  70  80
Prompt:     |----Prompt A----|----Prompt B----|
                             ^
                        Switch Point

At frame 40 (switch):
1. Reset cross-attention cache
2. Take frames 28-39 (last 12 frames)
3. Re-run through model with Prompt B
4. KV cache now contains Prompt B context
5. Continue generating frames 40+ with Prompt B
```

**Effect:** Smooth prompt transitions with semantic coherence to the new prompt.

### 3. Streaming Long Tuning (Train-Long-Test-Long)

**Problem:** Training on short clips (5 seconds) but testing on long videos (60+ seconds) causes distribution mismatch - the model never sees accumulated errors or KV cache states during training.

**Solution:** Training that mimics inference by reusing historical KV cache across chunks.

```python
# pipeline/streaming_training.py
class StreamingTrainingPipeline:
    """
    Training pipeline that:
    1. Generates video chunk-by-chunk (like inference)
    2. Reuses KV cache between chunks
    3. Computes loss only on new chunks
    """
    
    def generate_chunk_with_cache(
        self, 
        noise: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int,
        requires_grad: bool = True
    ):
        """
        Generate one chunk using existing KV cache.
        
        Args:
            noise: [B, chunk_frames, C, H, W] noise for this chunk
            conditional_dict: text embeddings
            current_start_frame: where this chunk starts in full sequence
            requires_grad: enable gradients for this chunk
        """
        batch_size, chunk_frames = noise.shape[:2]
        num_blocks = chunk_frames // self.num_frame_per_block
        
        output = torch.zeros_like(noise)
        
        # Generate block by block (same as inference!)
        local_start = 0
        for block_idx in range(num_blocks):
            noisy_input = noise[:, local_start:local_start + self.num_frame_per_block]
            
            # Spatial denoising with random early exit
            for step_idx, timestep in enumerate(self.denoising_step_list):
                exit_flag = (step_idx == exit_flags[block_idx])
                
                if not exit_flag:
                    with torch.no_grad():
                        _, denoised = self.generator(
                            noisy_input, conditional_dict, timestep,
                            kv_cache=self.kv_cache1,
                            current_start=(current_start_frame + local_start) * self.frame_seq_length
                        )
                        # Add noise for next step
                        noisy_input = self.scheduler.add_noise(denoised, noise, next_timestep)
                else:
                    # Final step - enable gradients if required
                    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
                    with ctx:
                        _, denoised = self.generator(
                            noisy_input, conditional_dict, timestep,
                            kv_cache=self.kv_cache1,
                            current_start=(current_start_frame + local_start) * self.frame_seq_length
                        )
                    break
            
            output[:, local_start:local_start + self.num_frame_per_block] = denoised
            
            # Update KV cache with generated content (CRITICAL!)
            context_timestep = torch.zeros_like(timestep)
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    current_start=(current_start_frame + local_start) * self.frame_seq_length
                )
            
            local_start += self.num_frame_per_block
        
        return output, denoised_timestep_from, denoised_timestep_to
```

**Training Loop:**
```python
# model/dmd_switch.py - simplified
for iteration in range(max_iters):
    # Initialize caches
    pipeline.clear_kv_cache()
    pipeline._initialize_kv_cache(batch_size, dtype, device)
    
    total_generated = 0
    accumulated_output = []
    
    # Generate up to streaming_max_length frames
    while total_generated < streaming_max_length:
        # Sample noise for this chunk
        chunk_noise = torch.randn([B, chunk_size, C, H, W])
        
        # Check for prompt switch
        if total_generated >= switch_frame and not switched:
            pipeline._recache_after_switch(output, total_generated, new_cond)
            current_cond = new_cond
            switched = True
        
        # Generate chunk (with KV cache reuse!)
        chunk_output, t_from, t_to = pipeline.generate_chunk_with_cache(
            noise=chunk_noise,
            conditional_dict=current_cond,
            current_start_frame=total_generated,
            requires_grad=(total_generated >= grad_start_frame)
        )
        
        accumulated_output.append(chunk_output)
        total_generated += chunk_size
    
    # Compute DMD loss on final chunk
    loss = model.compute_distribution_matching_loss(
        chunk_output, current_cond, uncond,
        denoised_timestep_from=t_from,
        denoised_timestep_to=t_to
    )
    
    loss.backward()
    optimizer.step()
```

**Effect:** Model learns to handle accumulated errors and long KV cache states, matching train-time and test-time behavior.

### 4. Distribution Matching Distillation (DMD)

**Problem:** Need to distill knowledge from a large teacher model (14B) to the fast student (1.3B) without paired video data.

**Solution:** Train using KL gradient between fake score (student) and real score (teacher) distributions.

```python
# model/dmd.py
class DMD(SelfForcingModel):
    """
    Distribution Matching Distillation:
    - Generator: 1.3B causal model (trainable)
    - Real Score: 14B teacher model (frozen) 
    - Fake Score: 1.3B critic model (trainable)
    """
    
    def __init__(self, args, device):
        super().__init__(args, device)
        
        # Generator - causal model we're training
        self.generator = WanDiffusionWrapper(..., is_causal=True)
        self.generator.model.requires_grad_(True)
        
        # Real score - teacher (frozen)
        self.real_score = WanDiffusionWrapper(model_name="Wan2.1-T2V-14B", is_causal=False)
        self.real_score.model.requires_grad_(False)
        
        # Fake score - critic (trainable)
        self.fake_score = WanDiffusionWrapper(model_name="Wan2.1-T2V-1.3B", is_causal=False)
        self.fake_score.model.requires_grad_(True)
    
    def _compute_kl_grad(self, noisy_input, estimated_clean, timestep, cond, uncond):
        """
        Compute KL gradient (Eq. 7 in DMD paper: https://arxiv.org/abs/2311.18828)
        
        grad = s_fake(x_t) - s_real(x_t)
        
        Where s_* is the score function (denoising direction)
        """
        # 1. Fake score prediction (student/critic)
        _, pred_fake = self.fake_score(noisy_input, cond, timestep)
        
        # 2. Real score prediction (teacher) with CFG
        _, pred_real_cond = self.real_score(noisy_input, cond, timestep)
        _, pred_real_uncond = self.real_score(noisy_input, uncond, timestep)
        pred_real = pred_real_cond + self.real_guidance_scale * (pred_real_cond - pred_real_uncond)
        
        # 3. KL gradient
        grad = pred_fake - pred_real
        
        # 4. Normalize by real prediction magnitude (Eq. 8 in DMD paper)
        p_real = estimated_clean - pred_real
        normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
        grad = grad / normalizer
        
        return torch.nan_to_num(grad)
    
    def compute_distribution_matching_loss(self, image_or_video, cond, uncond, gradient_mask=None):
        """
        DMD loss for generator training.
        
        Loss = ||x - (x - grad)||² / 2 = ||grad||² / 2
        
        But computed as MSE for stable gradients.
        """
        batch_size, num_frames = image_or_video.shape[:2]
        
        with torch.no_grad():
            # Sample random timestep
            timestep = torch.randint(min_step, max_step, [batch_size, num_frames], device=device)
            
            # Add noise to generated video
            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(image_or_video, noise, timestep)
            
            # Compute KL gradient
            grad = self._compute_kl_grad(noisy_latent, image_or_video, timestep, cond, uncond)
        
        # MSE loss with gradient as target direction
        if gradient_mask is not None:
            loss = 0.5 * F.mse_loss(
                image_or_video[gradient_mask],
                (image_or_video - grad).detach()[gradient_mask]
            )
        else:
            loss = 0.5 * F.mse_loss(image_or_video, (image_or_video - grad).detach())
        
        return loss
    
    def generator_loss(self, shape, cond, uncond, initial_latent=None):
        """
        Full generator loss computation:
        1. Generate video using backward simulation
        2. Compute DMD loss against teacher
        """
        # Step 1: Run generator with backward simulation
        pred_video, gradient_mask, t_from, t_to = self._run_generator(
            image_or_video_shape=shape,
            conditional_dict=cond,
            initial_latent=initial_latent
        )
        
        # Step 2: Compute DMD loss
        loss, log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_video,
            conditional_dict=cond,
            unconditional_dict=uncond,
            gradient_mask=gradient_mask,
            denoised_timestep_from=t_from,
            denoised_timestep_to=t_to
        )
        
        return loss, log_dict
    
    def critic_loss(self, shape, cond, uncond, initial_latent=None):
        """
        Critic (fake score) loss:
        Train fake_score to predict denoising direction on generated videos.
        """
        # Generate video (no gradients)
        with torch.no_grad():
            generated_video, _, t_from, t_to = self._run_generator(shape, cond, initial_latent)
        
        # Add noise
        timestep = torch.randint(min_step, max_step, [...])
        noise = torch.randn_like(generated_video)
        noisy_video = self.scheduler.add_noise(generated_video, noise, timestep)
        
        # Fake score prediction
        _, pred_fake = self.fake_score(noisy_video, cond, timestep)
        
        # Denoising loss (predict the generated video from noisy version)
        loss = self.denoising_loss_func(
            x=generated_video,
            x_pred=pred_fake,
            noise=noise,
            timestep=timestep
        )
        
        return loss
```

**Training alternation:**
```yaml
# configs/longlive_train_init.yaml
dfake_gen_update_ratio: 5  # Train critic 5x more than generator
```

```python
for iteration in range(max_iters):
    # Train critic more frequently
    for _ in range(dfake_gen_update_ratio):
        critic_loss = model.critic_loss(...)
        critic_loss.backward()
        critic_optimizer.step()
    
    # Train generator
    gen_loss = model.generator_loss(...)
    gen_loss.backward()
    gen_optimizer.step()
```

---

## Flow Matching Scheduler

LongLive uses **Flow Matching** instead of DDPM/DDIM:

```python
# utils/scheduler.py
class FlowMatchScheduler:
    """
    Flow Matching scheduler based on:
    - Rectified Flow: https://arxiv.org/abs/2209.03003
    - Flow Matching: https://arxiv.org/abs/2210.02747
    """
    
    def __init__(
        self, 
        num_train_timesteps=1000,
        shift=5.0,           # Noise schedule shift (higher = more noise early)
        sigma_min=0.003,     # Minimum noise level
        sigma_max=1.0,       # Maximum noise level
        extra_one_step=False
    ):
        self.shift = shift
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_train_timesteps = num_train_timesteps
    
    def set_timesteps(self, num_inference_steps, training=False):
        """
        Compute sigma schedule with optional shift.
        """
        # Linear interpolation between sigma_max and sigma_min
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, num_inference_steps)
        
        # Apply shift: σ' = shift * σ / (1 + (shift - 1) * σ)
        # This pushes more noise to early timesteps
        self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        
        # Convert to timesteps [0, 1000]
        self.timesteps = self.sigmas * self.num_train_timesteps
    
    def add_noise(self, original_samples, noise, timestep):
        """
        Flow matching forward process:
        
        x_t = (1 - σ_t) * x_0 + σ_t * ε
        
        vs DDPM: x_t = √α_t * x_0 + √(1-α_t) * ε
        
        Flow matching uses linear interpolation, DDPM uses sqrt scaling.
        """
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        
        # Linear interpolation
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
    
    def step(self, model_output, timestep, sample, to_final=False):
        """
        Euler step for flow matching:
        
        x_{t-1} = x_t + v * (σ_{t-1} - σ_t)
        
        Where v = ε - x_0 (velocity prediction)
        """
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        
        if to_final:
            sigma_next = 0
        else:
            sigma_next = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
        
        prev_sample = sample + model_output * (sigma_next - sigma)
        return prev_sample
    
    def training_target(self, sample, noise, timestep):
        """
        Flow matching training target: v = ε - x_0
        
        The model predicts the "velocity" from noise to clean.
        """
        return noise - sample
```

### Flow Prediction ↔ x0 Conversion

```python
# utils/wan_wrapper.py - WanDiffusionWrapper
def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
    """
    Convert flow/velocity prediction to x0 prediction.
    
    Given:
        v = ε - x_0  (model predicts velocity)
        x_t = (1 - σ) * x_0 + σ * ε  (flow interpolation)
    
    Solving for x_0:
        x_0 = x_t - σ * v
    
    Derivation:
        x_t = (1-σ)*x_0 + σ*ε
        x_t = x_0 - σ*x_0 + σ*ε
        x_t = x_0 + σ*(ε - x_0)
        x_t = x_0 + σ*v
        x_0 = x_t - σ*v
    """
    timestep_id = torch.argmin(
        (self.scheduler.timesteps - timestep.unsqueeze(1)).abs(), dim=1
    )
    sigma_t = self.scheduler.sigmas[timestep_id].reshape(-1, 1, 1, 1)
    
    x0_pred = xt - sigma_t * flow_pred
    return x0_pred

@staticmethod
def _convert_x0_to_flow_pred(scheduler, x0_pred, xt, timestep):
    """
    Convert x0 prediction back to flow prediction.
    
    v = (x_t - x_0) / σ
    """
    timestep_id = torch.argmin(
        (scheduler.timesteps - timestep.unsqueeze(1)).abs(), dim=1
    )
    sigma_t = scheduler.sigmas[timestep_id].reshape(-1, 1, 1, 1)
    
    flow_pred = (xt - x0_pred) / sigma_t
    return flow_pred
```

---

## Training Pipeline

### Step 1: Self-Forcing Initialization

**Purpose:** Initialize the causal model to match short-clip generation quality of the teacher.

**Config:** `configs/longlive_train_init.yaml`

```yaml
# Training frames
num_training_frames: 21        # 5.25 seconds at 4fps
min_num_training_frames: 21
slice_last_frames: 21          # Only compute gradients on last 21 frames

# Model architecture
num_frame_per_block: 3         # Denoise 3 frames together
model_kwargs:
  local_attn_size: 12          # 12-frame attention window
  sink_size: 3                 # 3-frame sink
  timestep_shift: 5.0          # Flow matching shift

# Denoising schedule
denoising_step_list: [1000, 750, 500, 250]  # 4-step denoising
warp_denoising_step: true

# Loss
distribution_loss: dmd
denoising_loss_type: flow
guidance_scale: 3.0

# Optimization
lr: 2.0e-06
lr_critic: 4.0e-07
dfake_gen_update_ratio: 5      # Train critic 5x per generator step
max_iters: 700
```

**Training script:** `train_init.sh`

```bash
torchrun --nproc_per_node=8 train.py --config_path configs/longlive_train_init.yaml
```

### Step 2: Streaming Long Tuning

**Purpose:** Train for long video consistency and prompt switching capability.

**Config:** `configs/longlive_train_long.yaml`

```yaml
# Streaming configuration
streaming_training: true
streaming_chunk_size: 21       # 5.25 seconds per chunk
streaming_max_length: 240      # Up to 60 seconds total
streaming_min_new_frame: 18    # Min new frames per iteration
train_first_chunk: true

# Prompt switching
distribution_loss: dmd_switch
switch_mode: random_choice
switch_choices: [21, 39, 57, 75, 93, 111, 129, 147, 165, 183, 201]
switch_prompt_path: prompts/vidprom_filtered_extended_switch.txt

# LoRA for efficient fine-tuning
adapter:
  type: "lora"
  rank: 256
  alpha: 256
  dropout: 0.0
  dtype: "bfloat16"
  apply_to_critic: true

# Training
lr: 1.0e-05
lr_critic: 2.0e-06
max_iters: 3000

# Model
global_sink: false             # Reset sink during switch
```

**Training script:** `train_long.sh`

```bash
torchrun --nproc_per_node=8 train.py --config_path configs/longlive_train_long.yaml
```

---

## Inference Modes

### 1. Single-Prompt Inference

**Script:** `inference.sh`

```bash
python inference.py --config_path configs/longlive_inference.yaml
```

**Config highlights:**

```yaml
denoising_step_list: [1000, 750, 500, 250]
num_frame_per_block: 3
num_output_frames: 120         # 30 seconds at 4fps

model_kwargs:
  local_attn_size: 12
  sink_size: 3
  timestep_shift: 5.0

generator_ckpt: longlive_models/models/longlive_base.pt
lora_ckpt: longlive_models/models/lora.pt
global_sink: true
context_noise: 0
```

### 2. Interactive Multi-Prompt Inference

**Script:** `interactive_inference.sh`

```bash
python interactive_inference.py --config_path configs/longlive_interactive_inference.yaml
```

**Config additions:**

```yaml
switch_frame_indices: "60,120,180"  # Switch at these frame indices
data_path: example/interactive_example.jsonl
```

**JSONL format:**

```json
{"prompts_list": ["A cat sitting on a red couch", "The cat stands up and stretches", "The cat jumps off the couch"]}
{"prompts_list": ["A serene lake at sunset", "Ripples appear on the water", "A fish jumps out of the water"]}
```

### 3. Infinite Length Generation

**Config:** `configs/longlive_inference_infinity.yaml`

Uses KV-cache relative RoPE (contributed by @qixinhu11) for theoretically unlimited video length.

---

## Key Numbers & Configuration

### Model Specifications

| Component | Value | Notes |
|-----------|-------|-------|
| Model size | 1.3B parameters | Student model |
| Teacher model | Wan2.1-T2V-14B | Used for DMD distillation |
| Transformer blocks | 30 | `num_layers` |
| Hidden dimension | 1536 | `dim` for 1.3B |
| FFN dimension | 6144 | `ffn_dim` |
| Attention heads | 12 | `num_heads` |
| Head dimension | 128 | `dim // num_heads` |
| Text encoder | UMT5-XXL | ~4.7B params, frozen |
| VAE | Wan2.1 Video VAE | 16 latent channels |

### Resolution & Tokens

| Metric | Value | Calculation |
|--------|-------|-------------|
| Output resolution | 480 × 832 pixels | After VAE decoding |
| Latent resolution | 60 × 104 | 480/8 × 832/8 |
| Latent channels | 16 | VAE z_dim |
| Patch size | (1, 2, 2) | Temporal, H, W |
| Tokens per frame | 1,560 | 60×104 / (2×2) = 1560 |
| Frame rate (latent) | 4 fps | VAE compresses 4× temporal |
| Frame rate (output) | 16 fps | 4 × 4 temporal upsampling |

### Attention Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `local_attn_size` | 12 | Frames in local attention window |
| `sink_size` | 3 | Frames permanently in cache |
| `max_attention_size` | 18,720 tokens | 12 × 1560 |
| Total attention | 23,400 tokens | sink (4,680) + local (18,720) |

### Generation Speed

| Metric | H100 | A100 | Notes |
|--------|------|------|-------|
| FPS (FP16) | 20.7 | ~15 | Frames per second |
| FPS (FP8) | 24.8 | N/A | With quantization |
| Max length | 240 seconds | 60 seconds | Memory dependent |

### Denoising Configuration

```yaml
denoising_step_list: [1000, 750, 500, 250]
# Warped timesteps (with shift=5.0):
# 1000 → 833.3
# 750 → 600.0  
# 500 → 357.1
# 250 → 166.7
```

---

## Summary of Innovations

| Innovation | Problem Solved | Key Technique |
|------------|----------------|---------------|
| **Frame-level AR with KV caching** | Slow bidirectional attention | Causal attention + KV cache reuse |
| **Frame Sink** | Long-range consistency loss | Keep first 3 frames in attention permanently |
| **KV-Recache** | Visual discontinuity on prompt switch | Re-compute KV cache with new prompt |
| **Streaming Long Tuning** | Train-test distribution mismatch | Train with KV cache reuse (like inference) |
| **DMD Training** | Need paired data for distillation | KL gradient between teacher/student scores |
| **Local + Global Attention** | Memory explosion on long videos | 12-frame window + 3-frame global sink |
| **Flow Matching** | Stable training | Linear interpolation noise schedule |
| **LoRA Fine-tuning** | Expensive full fine-tuning | 256-rank LoRA adapters |

---

## Quick Start Commands

```bash
# 1. Clone and setup
git clone https://github.com/NVlabs/LongLive
cd LongLive
conda create -n longlive python=3.10 -y
conda activate longlive
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# 2. Download models
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download Efficient-Large-Model/LongLive --local-dir longlive_models

# 3. Single-prompt inference
bash inference.sh

# 4. Interactive inference
bash interactive_inference.sh
```

---

## References

- **Paper**: [LongLive: Real-time Interactive Long Video Generation](https://arxiv.org/abs/2509.22622)
- **Self-Forcing**: [Self-Forcing: Bridging the Gap Between Training and Inference](https://github.com/guandeh17/Self-Forcing)
- **DMD**: [Diffusion Model Distillation](https://arxiv.org/abs/2311.18828)
- **DMD2**: [DMD2: Improved Diffusion Model Distillation](https://arxiv.org/abs/2405.14867)
- **Wan2.1**: [Wan Video Generation](https://github.com/Wan-Video/Wan2.1)
- **Flow Matching**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

---

*Document generated for LongLive repository deep dive. Last updated: February 2026.*
