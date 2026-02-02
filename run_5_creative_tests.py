#!/usr/bin/env python3
"""
Run 5 creative tests of the interactive terminal demo
Each test has dramatic changes between 10-second chunks

Per README: Include subject (who/what) and background/setting (where) in EVERY prompt
for better global coherence during prompt switches.
"""

import subprocess
import os
import time

# Test scenarios with 6 prompts each (60 seconds = 6 x 10s chunks)
# Each prompt includes the ANCHOR (subject + setting) for grounding

TESTS = [
    {
        "name": "1_seasons_changing",
        "description": "A woman standing in a field as seasons change dramatically around her",
        "anchor": "A beautiful young woman with long dark hair wearing a flowing white dress stands in an open meadow field",
        "variations": [
            "in lush green spring, cherry blossom petals floating in the air, bright sunshine, vibrant colors, photorealistic",
            "in blazing summer heat, golden wheat fields surrounding her, intense sunlight, heat waves visible, she shields her eyes from the sun, sweat on her skin",
            "in deep autumn, surrounded by brilliant red and orange fallen leaves swirling around her, overcast sky, she wraps her arms around herself, wind blowing her hair",
            "in a winter wonderland, heavy snow falling, the field completely white, she shivers, breath visible in the cold air, bare trees covered in frost and ice",
            "in a dramatic thunderstorm, dark purple clouds, lightning striking in the background, heavy rain soaking her dress and hair, wind whipping violently",
            "surrounded by magical aurora borealis at night, swirling green and purple lights in the sky, snow-covered ground glowing with ethereal light, peaceful and transcendent"
        ]
    },
    {
        "name": "2_aging_man",
        "description": "A man aging from young adult to elderly",
        "anchor": "A man standing in a warmly lit living room with family photos on the walls",
        "variations": [
            "in his early 20s with thick brown hair, smooth skin, bright energetic eyes, wearing a casual blue shirt, confident smile, full of youth and vitality",
            "in his mid-30s, slight stubble, a few subtle lines around his eyes, hair slightly shorter, wearing a professional suit, more mature expression, successful appearance",
            "in his late 40s, noticeable grey streaks in his hair, deeper wrinkles on his forehead, reading glasses perched on his nose, wearing a cardigan, thoughtful expression",
            "in his early 60s, predominantly grey hair thinning on top, pronounced wrinkles, slightly stooped posture, kind tired eyes, wearing comfortable clothes, sitting in an armchair",
            "in his mid-70s, white hair, deeply wrinkled face with age spots, using a walking cane, frail but dignified, warm gentle smile, same room now filled with decades of memories",
            "in his late 80s, very elderly, thin white hair, sitting peacefully in the armchair, eyes closed with serene expression, soft golden light streaming through the window"
        ]
    },
    {
        "name": "3_camera_pan_castle",
        "description": "Panning around a grand medieval castle from different angles",
        "anchor": "A majestic medieval castle with massive stone walls, tall towers with flags, surrounded by a moat",
        "variations": [
            "viewed from the front gate, drawbridge over the moat, guards standing at attention, morning sunlight, epic scale, wide establishing shot",
            "viewed from the left side, revealing a lush garden courtyard, ornate windows and balconies visible, servants walking through the grounds, birds flying overhead, mid-morning light",
            "from the rear view, showing the back towers and a secret garden maze, mountain backdrop visible, a waterfall cascading down nearby cliffs, dramatic clouds gathering",
            "aerial view from above, revealing its full layout, the surrounding village visible in the distance, fields and forests stretching to the horizon, late afternoon golden light casting long shadows",
            "at sunset viewed from across a lake, perfect reflection in the still water, orange and pink sky, silhouette of towers against the colorful clouds, romantic and majestic atmosphere",
            "at night from the front gate angle, illuminated by hundreds of torches and candles in windows, full moon overhead, mystical fog rolling through, magical and enchanting"
        ]
    },
    {
        "name": "4_city_time_lapse",
        "description": "A futuristic city transitioning through different times and weather",
        "anchor": "A stunning futuristic cityscape with sleek skyscrapers, holographic advertisements, and flying vehicles between buildings",
        "variations": [
            "at dawn, pink and orange sunrise reflecting off glass buildings, vehicles beginning to fill the sky, peaceful awakening, soft warm light",
            "at busy midday, intense activity, thousands of flying cars streaming between buildings, bustling crowds on elevated walkways, bright harsh sunlight, neon signs competing with daylight",
            "during a sudden afternoon thunderstorm, dark clouds rolling in, rain pouring down, lightning illuminating the metallic structures, people rushing for cover, dramatic weather",
            "at golden hour after the storm, wet streets reflecting the warm light, rainbow arcing over the skyline, steam rising from buildings, beautiful post-rain atmosphere",
            "at dusk transitioning to night, all the neon lights and holograms activating, the sky deep purple, stars becoming visible, city transforming into a glowing wonderland",
            "in deep night, full cyberpunk aesthetic, neon lights dominating, massive holographic projections in the sky, flying vehicles with glowing trails, light rain, noir atmosphere"
        ]
    },
    {
        "name": "5_metamorphosis",
        "description": "A creature undergoing dramatic transformations",
        "anchor": "A magical creature in an enchanted forest clearing with dappled sunlight filtering through ancient trees",
        "variations": [
            "appearing as a small colorful caterpillar on a green leaf, fuzzy body with bright stripes, multiple tiny legs, large cute eyes, eating hungrily, macro photography style, morning dew",
            "now wrapped in a glowing chrysalis attached to a branch, semi-transparent cocoon revealing movement inside, bioluminescent swirls of color, mysterious transformation beginning",
            "as the chrysalis cracks open, a magnificent butterfly beginning to emerge, wet crumpled wings starting to unfold, iridescent blue and gold colors revealed, dramatic moment of birth",
            "as a fully emerged butterfly with spectacular expanded wings, patterns like galaxies and nebulae, hovering in the air, surrounded by floating pollen and light particles, transcendent beauty",
            "transformed into a mythical phoenix-like creature, wings now made of flowing fire and light, much larger and more majestic, flames trailing behind it, flying through the forest",
            "completing its final transformation into a majestic dragon, massive wings of fire and starlight, ancient wise eyes, perched on a rock overlooking the forest, legendary and powerful"
        ]
    }
]


def build_prompts(test_config):
    """Build full prompts by combining anchor with variations"""
    anchor = test_config["anchor"]
    variations = test_config["variations"]
    # Each prompt = anchor + variation for consistent grounding
    return [f"{anchor}, {variation}" for variation in variations]


def run_test(test_config, output_base_dir):
    """Run a single test with the interactive demo"""
    name = test_config["name"]
    description = test_config["description"]
    prompts = build_prompts(test_config)
    
    print("\n" + "="*80)
    print(f"TEST: {name}")
    print(f"Description: {description}")
    print(f"Anchor: {test_config['anchor']}")
    print("="*80)
    
    # Show prompts
    print("Prompts:")
    for i, p in enumerate(prompts):
        preview = p[:100] + "..." if len(p) > 100 else p
        print(f"  [{i+1}] {preview}")
    
    # Prepare prompts input (prompts + done command)
    prompts_input = "\n".join(prompts) + "\ndone\n"
    
    # Run the interactive demo
    cmd = [
        "python", "interactive_terminal_demo.py",
        "--chunk_duration", "10",
        "--total_duration", "60",
        "--output_dir", f"{output_base_dir}/{name}",
        "--seed", "42"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    
    start_time = time.perf_counter()
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/workspace/longlive/LongLive"
    )
    
    stdout, _ = process.communicate(input=prompts_input, timeout=1800)  # 30 min timeout
    
    elapsed = time.perf_counter() - start_time
    
    # Print output
    print("\n--- Output ---")
    # Print last 30 lines
    lines = stdout.strip().split('\n')
    for line in lines[-30:]:
        print(line)
    
    print(f"\n--- Completed in {elapsed:.1f}s ---")
    
    return {
        "name": name,
        "elapsed_seconds": round(elapsed, 1),
        "success": process.returncode == 0
    }


def main():
    import json
    
    output_base = "videos/creative_tests"
    os.makedirs(output_base, exist_ok=True)
    
    print("="*80)
    print("       RUNNING 5 CREATIVE TESTS")
    print("       Each test: 60 seconds (6 x 10s chunks)")
    print("       Using anchor-based prompts for coherence")
    print("="*80)
    
    results = []
    total_start = time.perf_counter()
    
    for i, test in enumerate(TESTS):
        print(f"\n[{i+1}/5] Starting test: {test['name']}")
        result = run_test(test, output_base)
        results.append(result)
    
    total_elapsed = time.perf_counter() - total_start
    
    # Print summary
    print("\n" + "="*80)
    print("                    SUMMARY")
    print("="*80)
    print(f"{'Test':<30} {'Time':<15} {'Status'}")
    print("-"*60)
    for r in results:
        status = "✓ Success" if r["success"] else "✗ Failed"
        print(f"{r['name']:<30} {r['elapsed_seconds']:<15.1f} {status}")
    print("-"*60)
    print(f"{'Total':<30} {total_elapsed:<15.1f}")
    print("="*80)
    
    # Save results
    results_path = os.path.join(output_base, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "tests": results,
            "total_elapsed_seconds": round(total_elapsed, 1)
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # List output directories
    print("\nGenerated videos:")
    for test in TESTS:
        test_dir = os.path.join(output_base, test["name"])
        if os.path.exists(test_dir):
            sessions = [d for d in os.listdir(test_dir) if d.startswith("session_")]
            if sessions:
                latest = sorted(sessions)[-1]
                final_video = os.path.join(test_dir, latest, "final_video.mp4")
                if os.path.exists(final_video):
                    size_mb = os.path.getsize(final_video) / (1024*1024)
                    print(f"  {test['name']}: {final_video} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
