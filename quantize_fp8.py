# Quantize Nucleus-Image model to FP8 (float8_e4m3fn) single-file format.
#
# Usage:
#     python quantize_fp8.py
#
# Input:  D:\AI\Nucleus-Image        (bf16, sharded)
# Output: D:\AI\Nucleus-Image-FP8     (fp8, single file per component)

import os
import json
import gc
import torch
from safetensors.torch import load_file, save_file

SRC_DIR = r"D:\AI\Nucleus-Image"
DST_DIR = r"D:\AI\Nucleus-Image-FP8"

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0


def quantize_tensor(tensor: torch.Tensor):
    """Quantize a bf16/fp16/fp32 tensor to FP8 with per-tensor symmetric scaling.
    Returns (fp8_tensor, scale_float)."""
    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return tensor, None
    flat = tensor.float()
    max_val = flat.abs().amax().item()
    if max_val == 0:
        return torch.zeros_like(tensor, dtype=FP8_DTYPE), 1.0
    scale = FP8_MAX / max_val
    q = (flat * scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, 1.0 / scale  # store inv_scale so dequant = q * inv_scale


def load_sharded_state_dict(component_dir: str, prefix: str = ""):
    """Find and return list of shard file paths for a component."""
    files = sorted(f for f in os.listdir(component_dir) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files in {component_dir}")
    return [os.path.join(component_dir, f) for f in files]


def quantize_component(src_dir, dst_file, component_name):
    """Load shards one-by-one, quantize, and accumulate into single FP8 state dict."""
    print(f"\n{'='*60}")
    print(f"Quantizing: {component_name}")
    print(f"{'='*60}")

    shard_paths = load_sharded_state_dict(src_dir)
    print(f"Found {len(shard_paths)} shard(s)")

    fp8_state = {}
    scales = {}
    total_params = 0
    quantized_params = 0

    for i, shard_path in enumerate(shard_paths):
        shard_name = os.path.basename(shard_path)
        size_gb = os.path.getsize(shard_path) / 1024**3
        print(f"  [{i+1}/{len(shard_paths)}] Loading {shard_name} ({size_gb:.2f} GB)...")

        shard = load_file(shard_path)
        print(f"    {len(shard)} tensors loaded")

        for key, tensor in shard.items():
            total_params += tensor.numel()
            q_tensor, inv_scale = quantize_tensor(tensor)
            fp8_state[key] = q_tensor
            if inv_scale is not None:
                scales[key] = inv_scale
                quantized_params += tensor.numel()

        # Free the shard immediately
        del shard
        gc.collect()

    # Save as single FP8 safetensors file
    print(f"  Saving FP8 state dict ({len(fp8_state)} tensors)...")

    # Store scales as metadata
    metadata = {
        "quantization": json.dumps({
            "method": "fp8_e4m3fn_per_tensor",
            "scales": scales,
        }),
        "original_dtype": "bfloat16",
    }

    save_file(fp8_state, dst_file, metadata=metadata)

    out_size_gb = os.path.getsize(dst_file) / 1024**3
    print(f"  Saved: {dst_file} ({out_size_gb:.2f} GB)")
    print(f"  Quantized {quantized_params:,} / {total_params:,} parameters")

    del fp8_state
    gc.collect()
    return out_size_gb


def copy_config_files(src_dir, dst_dir, filenames):
    """Copy config/JSON/tokenizer files."""
    for f in filenames:
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, "r", encoding="utf-8") as rf:
                data = rf.read()
            with open(dst, "w", encoding="utf-8") as wf:
                wf.write(data)


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    total_input_gb = 0
    total_output_gb = 0

    # ---- 1. Transformer ----
    trans_src = os.path.join(SRC_DIR, "transformer")
    trans_dst = os.path.join(DST_DIR, "transformer")
    os.makedirs(trans_dst, exist_ok=True)

    # Calculate input size
    trans_input_gb = sum(
        os.path.getsize(os.path.join(trans_src, f))
        for f in os.listdir(trans_src) if f.endswith(".safetensors")
    ) / 1024**3

    trans_out_gb = quantize_component(
        trans_src,
        os.path.join(trans_dst, "diffusion_pytorch_model.safetensors"),
        "Transformer (MoE Diffusion)",
    )
    total_input_gb += trans_input_gb
    total_output_gb += trans_out_gb

    # Copy transformer config
    copy_config_files(trans_src, trans_dst, ["config.json"])

    # ---- 2. Text Encoder ----
    te_src = os.path.join(SRC_DIR, "text_encoder")
    te_dst = os.path.join(DST_DIR, "text_encoder")
    os.makedirs(te_dst, exist_ok=True)

    te_input_gb = sum(
        os.path.getsize(os.path.join(te_src, f))
        for f in os.listdir(te_src) if f.endswith(".safetensors")
    ) / 1024**3

    te_out_gb = quantize_component(
        te_src,
        os.path.join(te_dst, "model.safetensors"),
        "Text Encoder (Qwen3-VL-8B)",
    )
    total_input_gb += te_input_gb
    total_output_gb += te_out_gb

    # Copy text encoder configs and tokenizer
    te_files = [
        "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json",
        "preprocessor_config.json", "video_preprocessor_config.json",
        "chat_template.json", "vocab.json", "merges.txt",
        "model.safetensors.index.json",
    ]
    copy_config_files(te_src, te_dst, te_files)

    # ---- 3. VAE ----
    vae_src = os.path.join(SRC_DIR, "vae")
    vae_dst = os.path.join(DST_DIR, "vae")
    os.makedirs(vae_dst, exist_ok=True)

    vae_out_gb = quantize_component(
        vae_src,
        os.path.join(vae_dst, "diffusion_pytorch_model.safetensors"),
        "VAE (AutoencoderKLQwenImage)",
    )
    vae_input_gb = os.path.getsize(
        os.path.join(vae_src, "diffusion_pytorch_model.safetensors")
    ) / 1024**3
    total_input_gb += vae_input_gb
    total_output_gb += vae_out_gb

    copy_config_files(vae_src, vae_dst, ["config.json"])

    # ---- 4. Scheduler ----
    sched_dst = os.path.join(DST_DIR, "scheduler")
    os.makedirs(sched_dst, exist_ok=True)
    copy_config_files(
        os.path.join(SRC_DIR, "scheduler"),
        sched_dst,
        ["scheduler_config.json"],
    )

    # ---- 5. Processor ----
    proc_dst = os.path.join(DST_DIR, "processor")
    os.makedirs(proc_dst, exist_ok=True)
    proc_files = [
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "preprocessor_config.json", "video_preprocessor_config.json",
        "chat_template.json", "vocab.json", "merges.txt",
    ]
    copy_config_files(os.path.join(SRC_DIR, "processor"), proc_dst, proc_files)

    # ---- 6. model_index.json ----
    # Update to point to single files (remove index files references)
    with open(os.path.join(SRC_DIR, "model_index.json"), "r") as f:
        model_index = json.load(f)
    with open(os.path.join(DST_DIR, "model_index.json"), "w") as f:
        json.dump(model_index, f, indent=2)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"QUANTIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Input:  {total_input_gb:.2f} GB (bf16)")
    print(f"Output: {total_output_gb:.2f} GB (fp8)")
    print(f"Ratio:  {total_output_gb / total_input_gb:.2f}x")
    print(f"Output: {DST_DIR}")


if __name__ == "__main__":
    main()
