#!/usr/bin/env python3
"""
Gradio Web Demo for LongLive Interactive Video Generation

Run with: python -m gradio.web_demo
Or: python gradio/web_demo.py
"""

import os
import sys
import gradio as gr
from typing import Optional, Tuple, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_ui.video_builder_state import VideoBuilderState


# Global state (single user mode)
builder: Optional[VideoBuilderState] = None


def get_gpu_memory_stats() -> str:
    """Get GPU memory statistics"""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            free = total - reserved
            return f"**GPU Memory:** {allocated:.1f}GB used / {total:.1f}GB total ({free:.1f}GB free)"
        return "GPU: Not available"
    except Exception as e:
        return f"GPU: Error getting stats - {e}"


def create_builder():
    """Create a new VideoBuilderState instance"""
    global builder
    builder = VideoBuilderState(
        config_path="configs/longlive_inference.yaml",
        output_dir="videos/interactive_web"
    )
    return builder


def setup_model(progress=gr.Progress()):
    """Initialize the model"""
    global builder
    
    if builder is None:
        builder = create_builder()
    
    def progress_callback(message: str, pct: float):
        progress(pct, desc=message)
    
    builder.progress_callback = progress_callback
    
    try:
        result = builder.setup()
        status_text = f"""Model loaded successfully!

**Session:** {result['session_id']}
**GPU:** {result['gpu']}
**VRAM:** {result['vram_free']}
**Load time:** {result['load_time']}

Ready to create videos. Enter a grounding prompt below."""
        
        return (
            status_text,
            gr.update(interactive=True),  # grounding_input
            gr.update(interactive=True),  # grounding_skip_ai
            gr.update(interactive=True),  # grounding_btn
            gr.update(interactive=False), # setup_btn
        )
    except Exception as e:
        return (
            f"Setup failed: {str(e)}",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
        )


def enhance_grounding(grounding: str, skip_ai: bool):
    """Enhance the grounding prompt"""
    global builder
    
    if not builder or not builder.is_setup:
        return "Error: Model not setup", "", gr.update(visible=False)
    
    if not grounding.strip():
        return "Please enter a grounding prompt", "", gr.update(visible=False)
    
    try:
        result = builder.set_grounding(grounding, skip_ai=skip_ai)
        enhanced = result["enhanced"]
        
        return (
            "Grounding enhanced! Review below:",
            enhanced,
            gr.update(visible=True),  # Show approval buttons
        )
    except Exception as e:
        return f"Error: {str(e)}", "", gr.update(visible=False)


def regenerate_grounding_handler(grounding: str):
    """Regenerate grounding enhancement"""
    global builder
    
    if not builder:
        return ""
    
    try:
        enhanced = builder.regenerate_grounding(grounding)
        return enhanced
    except Exception as e:
        return f"Error: {str(e)}"


def accept_grounding(enhanced: str):
    """Accept the grounding and move to chunk building"""
    global builder
    
    if not builder:
        return (
            "Error: Model not setup",
            gr.update(visible=True),   # grounding_section
            gr.update(visible=False),  # chunk_section
            [],  # history
        )
    
    builder.accept_grounding(enhanced)
    
    status = f"""Grounding accepted!

**Scene:** {enhanced[:100]}...

Now add prompts for each ~10s video chunk."""
    
    return (
        status,
        gr.update(visible=False),  # Hide grounding section
        gr.update(visible=True),   # Show chunk section
        [],  # Clear history
    )


def enhance_chunk_prompt(user_input: str, skip_ai: bool):
    """Enhance a chunk prompt"""
    global builder
    
    if not builder or not builder.grounding_set:
        return "Error: Set grounding first", "", gr.update(visible=False)
    
    if not user_input.strip():
        return "Please enter a prompt", "", gr.update(visible=False)
    
    if skip_ai:
        return (
            "Using prompt directly (no AI enhancement)",
            user_input,
            gr.update(visible=True),
        )
    
    try:
        enhanced = builder.enhance_chunk_prompt(user_input)
        return (
            "Prompt enhanced! Review below:",
            enhanced,
            gr.update(visible=True),
        )
    except Exception as e:
        return f"Error: {str(e)}", "", gr.update(visible=False)


def regenerate_chunk_prompt_handler(user_input: str):
    """Regenerate chunk prompt"""
    global builder
    
    if not builder:
        return ""
    
    try:
        return builder.regenerate_chunk_prompt(user_input)
    except Exception as e:
        return f"Error: {str(e)}"


def generate_chunk(user_input: str, processed_prompt: str, skip_ai: bool, progress=gr.Progress()):
    """Generate a video chunk"""
    global builder
    
    if not builder or not builder.grounding_set:
        return (
            "Error: Set grounding first",
            None,
            [],
            gr.update(visible=False),
            "",
        )
    
    def progress_callback(message: str, pct: float):
        progress(pct, desc=message)
    
    builder.progress_callback = progress_callback
    
    try:
        # If processed_prompt is JSON, convert to flat prompt for video model
        video_prompt = processed_prompt
        try:
            import json
            state = json.loads(processed_prompt)
            if isinstance(state, dict) and "subject" in state:
                # It's our structured JSON - convert to flat prompt
                video_prompt = builder.enhancer._structured_to_video_prompt(state)
                # Also store the structured state
                if state not in builder.enhancer.structured_states:
                    builder.enhancer.structured_states.append(state)
        except (json.JSONDecodeError, TypeError):
            # Not JSON, use as-is
            pass
        
        result = builder.generate_chunk(
            user_prompt=user_input,
            processed_prompt=video_prompt,
            skip_ai=skip_ai
        )
        
        # Get updated history
        chunks = builder.get_chunks_info()
        history_data = [
            [c["chunk_num"], c["time_range"], c["user_prompt"][:50] + "..." if len(c["user_prompt"]) > 50 else c["user_prompt"]]
            for c in chunks
        ]
        
        chunk_num = result["chunk_num"]
        next_chunk = builder.current_chunk + 1
        next_start = builder.current_chunk * builder.chunk_duration
        next_end = next_start + builder.chunk_duration
        
        gpu_stats = get_gpu_memory_stats()
        
        status = f"""Chunk {chunk_num} generated in {result['generation_time']}!

**Time range:** {result['time_range']}
**Video saved:** {result['chunk_video']}

{gpu_stats}

Ready for chunk {next_chunk} ({next_start:.0f}-{next_end:.0f}s)"""
        
        chunk_video_path = result["chunk_video"]  # Last chunk only
        full_video_path = result["running_video"]  # Full video so far
        
        return (
            status,
            chunk_video_path,  # Last chunk
            full_video_path,   # Full video
            history_data,
            gr.update(visible=False),  # Hide approval section
            "",  # Clear prompt input
        )
    except Exception as e:
        return (
            f"Generation failed: {str(e)}",
            None,
            None,
            [],
            gr.update(visible=True),
            user_input,
        )


def go_back_handler():
    """Go back one chunk"""
    global builder
    
    if not builder:
        return "Error: No session", [], None, None
    
    result = builder.go_back()
    
    if result["status"] == "error":
        return result["message"], get_history_data(), None, None
    
    chunks = builder.get_chunks_info()
    history_data = [
        [c["chunk_num"], c["time_range"], c["user_prompt"][:50] + "..."]
        for c in chunks
    ]
    
    # Get video paths for the previous chunk
    chunk_video_path = None
    full_video_path = None
    if builder.current_chunk > 0:
        chunk_video_path = os.path.join(builder.session_dir, f"chunk_{builder.current_chunk}.mp4")
        full_video_path = os.path.join(builder.session_dir, f"running_{builder.current_chunk}.mp4")
        if not os.path.exists(chunk_video_path):
            chunk_video_path = None
        if not os.path.exists(full_video_path):
            full_video_path = None
    
    return (
        f"Rewound to chunk {builder.current_chunk + 1}. Enter a new prompt.",
        history_data,
        chunk_video_path,
        full_video_path
    )


def goto_chunk_handler(target: int):
    """Go to a specific chunk"""
    global builder
    
    if not builder:
        return "Error: No session", [], None, None
    
    if not target or target < 1:
        return "Enter a valid chunk number", get_history_data(), None, None
    
    result = builder.goto_chunk(int(target))
    
    if result["status"] == "error":
        return result["message"], get_history_data(), None, None
    
    chunks = builder.get_chunks_info()
    history_data = [
        [c["chunk_num"], c["time_range"], c["user_prompt"][:50] + "..."]
        for c in chunks
    ]
    
    chunk_video_path = None
    full_video_path = None
    if builder.current_chunk > 0:
        chunk_video_path = os.path.join(builder.session_dir, f"chunk_{builder.current_chunk}.mp4")
        full_video_path = os.path.join(builder.session_dir, f"running_{builder.current_chunk}.mp4")
        if not os.path.exists(chunk_video_path):
            chunk_video_path = None
        if not os.path.exists(full_video_path):
            full_video_path = None
    
    return (
        f"Rewound to chunk {builder.current_chunk + 1}. Enter a new prompt.",
        history_data,
        chunk_video_path,
        full_video_path
    )


def finalize_video():
    """Finalize and export the video"""
    global builder
    
    if not builder:
        return "Error: No session", None
    
    result = builder.finalize()
    
    if result["status"] == "error":
        return result["message"], None
    
    status = f"""Video finalized!

**Duration:** {result['duration']}
**Chunks:** {result['total_chunks']}
**Saved to:** {result['final_video']}

Download the video using the button below."""
    
    return status, result["final_video"]


def get_history_data():
    """Get current history as table data"""
    global builder
    if not builder:
        return []
    chunks = builder.get_chunks_info()
    return [
        [c["chunk_num"], c["time_range"], c["user_prompt"][:50] + "..."]
        for c in chunks
    ]


def get_current_chunk_info():
    """Get info about the current chunk to generate"""
    global builder
    if not builder or not builder.grounding_set:
        return "Set up model and grounding first"
    
    chunk_num = builder.current_chunk + 1
    start_time = builder.current_chunk * builder.chunk_duration
    end_time = start_time + builder.chunk_duration
    
    return f"**Chunk {chunk_num}** ({start_time:.0f}-{end_time:.0f}s)"


def reset_session():
    """Reset the session for a new video"""
    global builder
    
    if not builder:
        return (
            "Error: Model not loaded yet",
            gr.update(visible=True),   # grounding_section
            gr.update(visible=False),  # chunk_section
            [],  # history
            None,  # chunk video
            None,  # full video
            "",  # grounding input
            "",  # enhanced grounding
            gr.update(visible=False),  # grounding_approval
            "",  # chunk input
            "",  # enhanced chunk
            gr.update(visible=False),  # chunk_approval
        )
    
    result = builder.reset()
    
    if result["status"] == "error":
        return (
            result["message"],
            gr.update(visible=True),
            gr.update(visible=False),
            [],
            None,
            None,
            "",
            "",
            gr.update(visible=False),
            "",
            "",
            gr.update(visible=False),
        )
    
    status = f"""Session reset!

**New Session:** {result['new_session_id']}
**Output:** {result['session_dir']}

Ready for a new video. Enter a grounding prompt below."""
    
    return (
        status,
        gr.update(visible=True),   # Show grounding section
        gr.update(visible=False),  # Hide chunk section
        [],  # Clear history
        None,  # Clear chunk video
        None,  # Clear full video
        "",  # Clear grounding input
        "",  # Clear enhanced grounding
        gr.update(visible=False),  # Hide grounding approval
        "",  # Clear chunk input
        "",  # Clear enhanced chunk
        gr.update(visible=False),  # Hide chunk approval
    )


# Build the Gradio interface
def create_demo():
    with gr.Blocks(title="LongLive Interactive Video Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# LongLive Interactive Video Generator

Build AI-generated videos chunk by chunk with natural language prompts.
        """)
        
        # Status area
        status_box = gr.Markdown("Click **Setup Model** to begin (takes ~60 seconds)")
        
        # Setup section
        with gr.Row():
            setup_btn = gr.Button("Setup Model", variant="primary", scale=1)
            reset_btn = gr.Button("Reset Session", variant="secondary", scale=1)
        
        # Grounding section
        with gr.Group(visible=True) as grounding_section:
            gr.Markdown("### Step 1: Set the Scene (Grounding)")
            gr.Markdown("Describe the main subject and setting. This anchors all future prompts.")
            
            with gr.Row():
                grounding_input = gr.Textbox(
                    label="Grounding Prompt",
                    placeholder="e.g., A young woman with red hair standing in a misty forest",
                    lines=2,
                    interactive=False
                )
            
            with gr.Row():
                grounding_skip_ai = gr.Checkbox(label="Skip AI enhancement", value=False, interactive=False)
                grounding_btn = gr.Button("Enhance", variant="primary", interactive=False)
            
            # Grounding approval
            with gr.Group(visible=False) as grounding_approval:
                enhanced_grounding = gr.Textbox(
                    label="Enhanced Grounding (edit if needed)", 
                    lines=10,
                    max_lines=20
                )
                with gr.Row():
                    grounding_accept_btn = gr.Button("Accept", variant="primary")
                    grounding_regen_btn = gr.Button("Regenerate")
        
        # Chunk building section
        with gr.Group(visible=False) as chunk_section:
            gr.Markdown("### Step 2: Build Video Chunks")
            
            chunk_info = gr.Markdown("**Chunk 1** (0-10s)")
            
            with gr.Row():
                with gr.Column(scale=2):
                    chunk_input = gr.Textbox(
                        label="What happens in this chunk?",
                        placeholder="e.g., she slowly turns to face the camera",
                        lines=2
                    )
                    with gr.Row():
                        chunk_skip_ai = gr.Checkbox(label="Skip AI enhancement", value=False)
                        chunk_enhance_btn = gr.Button("Enhance Prompt", variant="secondary")
                    
                    # Chunk approval
                    with gr.Group(visible=False) as chunk_approval:
                        enhanced_chunk = gr.Textbox(
                            label="Enhanced Prompt (edit if needed)", 
                            lines=12,
                            max_lines=25
                        )
                        with gr.Row():
                            chunk_accept_btn = gr.Button("Generate Chunk", variant="primary")
                            chunk_regen_btn = gr.Button("Regenerate")
                
                with gr.Column(scale=2):
                    gr.Markdown("**Last Generated Chunk:**")
                    chunk_video_player = gr.Video(label="Last Chunk", interactive=False, height=250)
                    gr.Markdown("**Full Video So Far:**")
                    full_video_player = gr.Video(label="Full Video", interactive=False, height=250)
            
            # History and navigation
            gr.Markdown("### Generated Chunks")
            history_table = gr.Dataframe(
                headers=["#", "Time", "Prompt"],
                datatype=["number", "str", "str"],
                interactive=False,
                wrap=True
            )
            
            with gr.Row():
                back_btn = gr.Button("← Back One Chunk")
                goto_input = gr.Number(label="Go to chunk #", precision=0)
                goto_btn = gr.Button("Go")
                finalize_btn = gr.Button("Done - Finalize Video", variant="primary")
        
        # Final video section
        with gr.Group(visible=False) as final_section:
            gr.Markdown("### Final Video")
            final_video = gr.Video(label="Final Video")
            download_btn = gr.File(label="Download")
        
        # Event handlers
        setup_btn.click(
            fn=setup_model,
            inputs=[],
            outputs=[status_box, grounding_input, grounding_skip_ai, grounding_btn, setup_btn]
        )
        
        grounding_btn.click(
            fn=enhance_grounding,
            inputs=[grounding_input, grounding_skip_ai],
            outputs=[status_box, enhanced_grounding, grounding_approval]
        )
        
        grounding_regen_btn.click(
            fn=regenerate_grounding_handler,
            inputs=[grounding_input],
            outputs=[enhanced_grounding]
        )
        
        grounding_accept_btn.click(
            fn=accept_grounding,
            inputs=[enhanced_grounding],
            outputs=[status_box, grounding_section, chunk_section, history_table]
        )
        
        chunk_enhance_btn.click(
            fn=enhance_chunk_prompt,
            inputs=[chunk_input, chunk_skip_ai],
            outputs=[status_box, enhanced_chunk, chunk_approval]
        )
        
        chunk_regen_btn.click(
            fn=regenerate_chunk_prompt_handler,
            inputs=[chunk_input],
            outputs=[enhanced_chunk]
        )
        
        chunk_accept_btn.click(
            fn=generate_chunk,
            inputs=[chunk_input, enhanced_chunk, chunk_skip_ai],
            outputs=[status_box, chunk_video_player, full_video_player, history_table, chunk_approval, chunk_input]
        )
        
        back_btn.click(
            fn=go_back_handler,
            inputs=[],
            outputs=[status_box, history_table, chunk_video_player, full_video_player]
        )
        
        goto_btn.click(
            fn=goto_chunk_handler,
            inputs=[goto_input],
            outputs=[status_box, history_table, chunk_video_player, full_video_player]
        )
        
        finalize_btn.click(
            fn=finalize_video,
            inputs=[],
            outputs=[status_box, final_video]
        )
        
        reset_btn.click(
            fn=reset_session,
            inputs=[],
            outputs=[status_box, grounding_section, chunk_section, history_table, chunk_video_player, full_video_player, grounding_input, enhanced_grounding, grounding_approval, chunk_input, enhanced_chunk, chunk_approval]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public URL
        show_error=True
    )
