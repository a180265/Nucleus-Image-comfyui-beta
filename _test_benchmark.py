"""
Nucleus-Image sampler/scheduler benchmark with complex prompt.
Each output image has parameters + prompt overlaid directly.
Tests 5 sampler combos, generates labeled images + comparison grid.
"""
import os, sys, time, gc, math
import torch, numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

os.chdir(r"D:\AI\ComfyUI-aki-v3\ComfyUI")
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

OUT_DIR = r"D:\AI"

T0 = time.time()
def pt(msg):
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"[{time.time()-T0:.1f}s | VRAM {vram:.1f}G] {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# Image annotation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_labeled_image(img, prompt, sampler, scheduler, steps, cfg, seed, t_sample, t_decode):
    """Overlay prompt + parameters on a white strip below the image."""
    margin = 15
    line_h = 18
    info_lines = [
        f"Sampler: {sampler}    Scheduler: {scheduler}    Steps: {steps}    CFG: {cfg}    Seed: {seed}",
        f"Size: {img.width}x{img.height}    Sample: {t_sample:.0f}s    Decode: {t_decode:.0f}s    Total: {t_sample+t_decode:.0f}s",
    ]
    # Wrap prompt into lines
    prompt_prefix = "Prompt: "
    max_chars_per_line = 95
    prompt_text = prompt_prefix + prompt
    prompt_lines = []
    while prompt_text:
        if len(prompt_text) <= max_chars_per_line:
            prompt_lines.append(prompt_text)
            break
        idx = prompt_text[:max_chars_per_line].rfind(" ")
        if idx == -1:
            idx = max_chars_per_line
        prompt_lines.append(prompt_text[:idx])
        prompt_text = prompt_text[idx:].lstrip()
        if prompt_text:
            prompt_text = "         " + prompt_text

    total_text_lines = len(info_lines) + 1 + len(prompt_lines)
    text_area_h = margin * 2 + total_text_lines * line_h + 5

    canvas_w = img.width
    canvas_h = img.height + text_area_h
    canvas = PILImage.new("RGB", (canvas_w, canvas_h), (240, 240, 240))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_bold = ImageFont.truetype("arialbd.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_bold = font

    y = img.height + margin
    draw.text((margin, y), info_lines[0], fill=(20, 20, 20), font=font_bold)
    y += line_h
    draw.text((margin, y), info_lines[1], fill=(60, 60, 60), font=font)
    y += line_h + 5
    draw.line([(margin, y), (canvas_w - margin, y)], fill=(180, 180, 180))
    y += 5
    for pline in prompt_lines:
        draw.text((margin, y), pline, fill=(40, 40, 100), font=font)
        y += line_h

    return canvas


def _make_comparison_grid(results, prompt, seed, cfg, steps):
    """Create a side-by-side grid of all labeled images."""
    images = []
    for fname, sname, sched, t_s, t_d in results:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            images.append(PILImage.open(path))

    if not images:
        return PILImage.new("RGB", (800, 600), (200, 200, 200))

    n = len(images)
    cols = min(n, 3)
    rows = math.ceil(n / cols)

    img_w = images[0].width
    img_h = images[0].height
    gap = 6

    total_w = cols * img_w + (cols + 1) * gap
    total_h = rows * img_h + (rows + 1) * gap

    canvas = PILImage.new("RGB", (total_w, total_h), (30, 30, 30))
    for i, img in enumerate(images):
        col = i % cols
        row = i // cols
        x = gap + col * (img_w + gap)
        y = gap + row * (img_h + gap)
        canvas.paste(img, (x, y))

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Test configuration
# ══════════════════════════════════════════════════════════════════════════════

# Complex prompt: multiple people, detailed spatial layout, varied colors
PROMPT = (
    "A crowded medieval marketplace at sunset. On the left, a blacksmith in a leather apron "
    "is forging a sword at his anvil with orange sparks flying. In the center, a merchant woman "
    "in a red dress is selling fruit from a wooden cart filled with apples and oranges. On the "
    "right, two children in brown tunics are chasing a white cat through a stack of barrels. "
    "In the background, stone buildings with thatched roofs line a cobblestone street, and a "
    "tall church tower with a golden cross rises against an orange and purple sunset sky."
)
NEG_PROMPT = ""
SEED = 42
W, H = 1024, 1024
STEPS = 50
CFG = 4.0

COMBOS = [
    ("euler",       "normal"),
    ("dpmpp_2m",    "karras"),
    ("heun",        "normal"),
    ("dpmpp_2m_sde","karras"),
    ("dpmpp_3m_sde","exponential"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Main benchmark
# ══════════════════════════════════════════════════════════════════════════════

import importlib
nm = importlib.import_module("custom_nodes.ComfyUI-Nucleus-Image.nodes")

pt("Loading transformer...")
t0 = time.time()
loader = nm.NucleusImageTransformerLoader()
model = loader.load_model("nucleus_image_transformer_fp8.safetensors", "bf16", "offload_device")[0]
pt(f"Transformer loaded: {time.time()-t0:.1f}s")

pt("Loading TE...")
te_loader = nm.NucleusImageTextEncoderLoader()
te = te_loader.load_model("nucleus_image_text_encoder_fp8.safetensors", "bf16")[0]

pt("Loading VAE...")
vae_loader = nm.NucleusImageVAELoader()
vae = vae_loader.load_model("nucleus_image_vae.safetensors", "bf16")[0]
decoder = nm.NucleusImageVAEDecode()

# Dual encode
pt("Dual encoding...")
t_enc = time.time()
pos, neg = nm.NucleusImageTextEncodeDual().encode(te, PROMPT, NEG_PROMPT)
t_enc = time.time() - t_enc
pt(f"Encoding done: {t_enc:.1f}s")

# Generate images
sampler_node = nm.NucleusImageSampler()
results = []

for sname, sched in COMBOS:
    pt(f"\n--- {sname} + {sched} ({STEPS} steps) ---")
    t0 = time.time()
    result = sampler_node.sample(model, pos, neg, W, H, STEPS, CFG, SEED, sname, sched)
    t_sample = time.time() - t0

    t0d = time.time()
    img_tensor = decoder.decode(vae, result[0])[0]
    t_decode = time.time() - t0d

    img_np = (img_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img_pil = PILImage.fromarray(img_np)

    # Save labeled image
    labeled = _make_labeled_image(img_pil, PROMPT, sname, sched, STEPS, CFG, SEED, t_sample, t_decode)
    labeled_fname = f"nucleus_bench_{sname}_{sched}.png"
    labeled.save(os.path.join(OUT_DIR, labeled_fname))

    results.append((labeled_fname, sname, sched, t_sample, t_decode))
    pt(f"  Sample: {t_sample:.0f}s, Decode: {t_decode:.0f}s -> {labeled_fname}")

# Comparison grid
pt("\nGenerating comparison grid...")
grid = _make_comparison_grid(results, PROMPT, SEED, CFG, STEPS)
grid_path = os.path.join(OUT_DIR, "nucleus_bench_comparison.png")
grid.save(grid_path)
pt(f"Grid saved: {grid_path}")

pt(f"\nTotal script: {time.time()-T0:.0f}s")
