#!/usr/bin/env python3
"""
Block-Swap + FP8 Inference Test for Nucleus-Image (17B MoE, 24GB VRAM target)

Strategy:
  Stage 1 - Text encoding: Load Qwen3-VL from FP8 on GPU directly -> encode -> free
  Stage 2 - Denoising:     Load transformer with tiny expert weights, keep non-expert
                            on GPU, swap expert FP8 weights per-block from CPU
  Stage 3 - VAE decoding:  Load VAE from FP8 on GPU -> decode -> save
"""

import os, sys, time, json, gc, types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from safetensors import safe_open

# ── paths ────────────────────────────────────────────────────────────────────
FP8_DIR   = r"D:\AI\Nucleus-Image-FP8"
COMFY_PY  = r"D:\AI\ComfyUI-aki-v3\python"
sys.path.insert(0, os.path.join(COMFY_PY, "Lib", "site-packages"))

# ── helpers ──────────────────────────────────────────────────────────────────

def _set_nested_attr(model, name, value):
    parts = name.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    old = getattr(obj, parts[-1])
    if isinstance(old, nn.Parameter):
        setattr(obj, parts[-1], nn.Parameter(value))
    else:
        setattr(obj, parts[-1], value)


def _is_expert_param(name):
    return "experts.gate_up_proj" in name or "experts.down_proj" in name


def _dequant_fp8(tensor, scale):
    if tensor.dtype == torch.float8_e4m3fn and scale is not None:
        return (tensor.float() * scale).to(torch.bfloat16)
    if tensor.is_floating_point():
        return tensor.to(torch.bfloat16)
    return tensor


def load_fp8_into_model(model, fp8_path, device=None):
    """Load FP8 state dict into a model, dequantizing tensor-by-tensor.
    If device is set, each tensor is dequantized on that device then copied in."""
    # Metadata (scales) must be read via safe_open — load_file() doesn't include __metadata__
    scales = {}
    with safe_open(fp8_path, framework="pt") as f:
        meta = f.metadata()
        if meta and "quantization" in meta:
            scales = json.loads(meta["quantization"]).get("scales", {})
    sd = load_file(fp8_path)

    param_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())
    loaded = 0
    for key in list(sd.keys()):
        tensor = sd.pop(key)
        if key in param_map:
            if device is not None and tensor.is_floating_point():
                tensor = tensor.to(device)
            bf16 = _dequant_fp8(tensor, scales.get(key))
            param_map[key].data.copy_(bf16)
            loaded += 1
        elif key in buffer_map:
            buffer_map[key].copy_(tensor)
            loaded += 1
        del tensor
    del sd
    gc.collect()
    return loaded


# ── Stage 1: Text Encoding ──────────────────────────────────────────────────

def encode_text(prompt, device="cuda"):
    """Load Qwen3-VL from FP8, encode prompt, return (embeds, mask), free model."""
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, AutoConfig

    print("\n[Stage 1] Text Encoding")
    t0 = time.time()

    proc_dir = os.path.join(FP8_DIR, "processor")
    processor = Qwen3VLProcessor.from_pretrained(proc_dir)

    te_dir = os.path.join(FP8_DIR, "text_encoder")
    te_config = AutoConfig.from_pretrained(te_dir)

    # Create model directly on GPU to avoid 16GB CPU allocation
    with torch.device(device):
        text_encoder = Qwen3VLForConditionalGeneration._from_config(
            te_config, dtype=torch.bfloat16
        )
    print(f"  Model allocated on {device}, loading FP8 weights...")

    load_fp8_into_model(text_encoder, os.path.join(te_dir, "model.safetensors"), device=device)
    text_encoder.eval()
    print(f"  Text encoder loaded in {time.time()-t0:.1f}s")

    # Format prompt
    SYSTEM_PROMPT = (
        "You are an image generation assistant. Follow the user's prompt literally. "
        "Pay careful attention to spatial layout: objects described as on the left must "
        "appear on the left, on the right on the right. Match exact object counts and "
        "assign colors to the correct objects."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[formatted],
        padding="longest",
        pad_to_multiple_of=8,
        max_length=1024,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    ).to(device=device)

    with torch.no_grad():
        outputs = text_encoder(**inputs, use_cache=False, return_dict=True, output_hidden_states=True)
        prompt_embeds = outputs.hidden_states[-8].to(dtype=torch.bfloat16, device=device)
    prompt_mask = inputs.attention_mask
    if prompt_mask is not None and prompt_mask.all():
        prompt_mask = None

    del text_encoder, outputs, inputs
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Text encoding done in {time.time()-t0:.1f}s, embeds shape: {prompt_embeds.shape}")
    return prompt_embeds, prompt_mask


# ── Stage 2: Transformer with Block-Swap ────────────────────────────────────

def load_transformer_block_swap(device="cuda"):
    """Build transformer with tiny expert weights (to save RAM), load non-expert
    from FP8 onto GPU, keep expert FP8 on CPU for per-block swapping."""
    from diffusers.models import NucleusMoEImageTransformer2DModel

    print("\n[Stage 2] Loading transformer (block-swap)")
    t0 = time.time()

    trans_dir = os.path.join(FP8_DIR, "transformer")

    # ── Monkey-patch SwiGLUExperts.__init__ to allocate tiny expert weights ──
    from diffusers.models.transformers.transformer_nucleusmoe_image import SwiGLUExperts
    _orig_swiglu_init = SwiGLUExperts.__init__

    def _tiny_swiglu_init(self, hidden_size, moe_intermediate_dim, num_experts, use_grouped_mm=False):
        nn.Module.__init__(self)  # must call super before setting parameters
        self.num_experts = num_experts
        self.moe_intermediate_dim = moe_intermediate_dim
        self.hidden_size = hidden_size
        self.use_grouped_mm = use_grouped_mm
        # Allocate tiny placeholders instead of full (64, 2048, 2688) tensors
        self.gate_up_proj = nn.Parameter(torch.empty(1, 1, 1, dtype=torch.bfloat16))
        self.down_proj = nn.Parameter(torch.empty(1, 1, 1, dtype=torch.bfloat16))

    SwiGLUExperts.__init__ = _tiny_swiglu_init

    # Build transformer on CPU — now only ~1.8GB instead of ~48GB
    config = NucleusMoEImageTransformer2DModel.load_config(trans_dir)
    config["use_grouped_mm"] = False  # Test with for-loop instead of grouped_mm
    transformer = NucleusMoEImageTransformer2DModel.from_config(config)
    print(f"  Model created on CPU (tiny experts)")

    # Restore original init
    SwiGLUExperts.__init__ = _orig_swiglu_init

    # ── Load FP8 state dict to CPU ───────────────────────────────────────
    fp8_path = os.path.join(trans_dir, "diffusion_pytorch_model.safetensors")
    print(f"  Loading FP8 weights ({os.path.getsize(fp8_path)/1e9:.2f} GB)...")
    # Metadata (scales) must be read via safe_open — load_file() doesn't include __metadata__
    scales = {}
    with safe_open(fp8_path, framework="pt") as f:
        meta = f.metadata()
        if meta and "quantization" in meta:
            scales = json.loads(meta["quantization"]).get("scales", {})
    fp8_sd = load_file(fp8_path)

    # Separate expert vs non-expert weights
    expert_fp8 = {}    # {block_idx: {'gate_up_proj': fp8, 'down_proj': fp8}}
    expert_scales = {}  # {block_idx: {'gate_up_proj': scale, 'down_proj': scale}}
    non_expert_count = 0
    expert_count = 0

    for name in list(fp8_sd.keys()):
        tensor = fp8_sd.pop(name)
        if _is_expert_param(name):
            parts = name.split(".")
            block_idx = int(parts[1])
            ptype = "gate_up_proj" if "gate_up_proj" in name else "down_proj"
            expert_fp8.setdefault(block_idx, {})[ptype] = tensor
            if name in scales:
                expert_scales.setdefault(block_idx, {})[ptype] = scales[name]
            expert_count += 1
        else:
            bf16 = _dequant_fp8(tensor, scales.get(name))
            _set_nested_attr(transformer, name, nn.Parameter(bf16))
            non_expert_count += 1
        del tensor

    del fp8_sd, scales
    gc.collect()

    # Move non-expert params to GPU
    for name, param in transformer.named_parameters():
        if not _is_expert_param(name) and param.device.type != "cuda":
            # Already set by _set_nested_attr above but might still be on CPU
            pass
    # Move the whole model — expert params (tiny 1x1x1) go too but they're negligible
    transformer.to(device)
    print(f"  Non-expert params on GPU: {non_expert_count}")
    print(f"  Expert FP8 blocks on CPU: {sorted(expert_fp8.keys())} ({expert_count} tensors)")

    # ── Monkey-patch each MoE block's expert forward ──────────────────────
    for block_idx, block in enumerate(transformer.transformer_blocks):
        if not block.moe_enabled or block_idx not in expert_fp8:
            continue
        _patch_expert_forward(block.img_mlp.experts, block_idx,
                              expert_fp8[block_idx], expert_scales.get(block_idx, {}), device)

    elapsed = time.time() - t0
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Transformer loaded in {elapsed:.1f}s, VRAM: {vram:.1f} GB")
    return transformer, config


def _patch_expert_forward(experts_module, block_idx, fp8_data, fp8_scales, device):
    """Replace SwiGLUExperts.forward with block-swap version."""
    use_gmm = experts_module.use_grouped_mm
    gu_fp8 = fp8_data["gate_up_proj"]
    dp_fp8 = fp8_data["down_proj"]
    gu_scale = fp8_scales.get("gate_up_proj")
    dp_scale = fp8_scales.get("down_proj")

    if gu_scale is not None:
        gu_scale_t = torch.tensor(gu_scale, dtype=torch.float32)
    else:
        gu_scale_t = torch.tensor(1.0, dtype=torch.float32)
    if dp_scale is not None:
        dp_scale_t = torch.tensor(dp_scale, dtype=torch.float32)
    else:
        dp_scale_t = torch.tensor(1.0, dtype=torch.float32)

    _cache = {"gu": None, "dp": None}

    def _load():
        if _cache["gu"] is not None:
            return
        gu_gpu = gu_fp8.to(device)
        dp_gpu = dp_fp8.to(device)
        sc_gu = gu_scale_t.to(device)
        sc_dp = dp_scale_t.to(device)
        _cache["gu"] = (gu_gpu.float() * sc_gu).to(torch.bfloat16)
        _cache["dp"] = (dp_gpu.float() * sc_dp).to(torch.bfloat16)
        del gu_gpu, dp_gpu

    def _unload():
        _cache["gu"] = None
        _cache["dp"] = None

    if use_gmm:
        def _forward(self, x, num_tokens_per_expert):
            _load()
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            gate_up = F.grouped_mm(x, _cache["gu"], offs=offsets)
            gate, up = gate_up.chunk(2, dim=-1)
            out = F.grouped_mm(F.silu(gate) * up, _cache["dp"], offs=offsets)
            result = out.type_as(x)
            _unload()
            return result
    else:
        def _forward(self, x, num_tokens_per_expert):
            _load()
            ntl = num_tokens_per_expert.tolist()
            num_real = sum(ntl)
            num_pad = x.shape[0] - num_real
            gu, dp = _cache["gu"], _cache["dp"]
            x_per = torch.split(x[:num_real], ntl, dim=0)
            outs = []
            for ei, xe in enumerate(x_per):
                gu_out = torch.matmul(xe, gu[ei])
                g, u = gu_out.chunk(2, dim=-1)
                outs.append(torch.matmul(F.silu(g) * u, dp[ei]))
            result = torch.cat(outs, dim=0)
            if num_pad > 0:
                result = torch.vstack((result, result.new_zeros((num_pad, result.shape[-1]))))
            _unload()
            return result

    experts_module.forward = types.MethodType(_forward, experts_module)


# ── Stage 3: VAE Decoding ──────────────────────────────────────────────────

def decode_latents(latents_packed, height, width, patch_size, device="cuda"):
    from diffusers import AutoencoderKLQwenImage

    print("\n[Stage 3] VAE Decoding")
    t0 = time.time()

    vae_dir = os.path.join(FP8_DIR, "vae")
    vae_config = AutoencoderKLQwenImage.load_config(vae_dir)

    with torch.device(device):
        vae = AutoencoderKLQwenImage.from_config(vae_config)
    load_fp8_into_model(vae, os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"), device=device)
    vae.eval()

    # Unpack latents
    vae_sf = 2 ** len(vae.temperal_downsample)  # 8
    bs, np_, ch = latents_packed.shape
    hp = patch_size * (int(height) // (vae_sf * patch_size))
    wp = patch_size * (int(width) // (vae_sf * patch_size))
    latents = latents_packed.view(
        bs, hp // patch_size, wp // patch_size,
        ch // (patch_size * patch_size), patch_size, patch_size,
    )
    latents = latents.permute(0, 3, 1, 4, 2, 5).contiguous()
    latents = latents.reshape(bs, ch // (patch_size * patch_size), 1, hp, wp)

    z_dim = vae.config.z_dim
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents)
    latents_std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(latents)
    latents = latents * latents_std + latents_mean

    with torch.no_grad():
        decoded = vae.decode(latents.to(vae.dtype), return_dict=False)[0][:, :, 0]

    from diffusers.image_processor import VaeImageProcessor
    ip = VaeImageProcessor(vae_scale_factor=vae_sf * 2)
    image = ip.postprocess(decoded, output_type="pil")[0]

    del vae; gc.collect(); torch.cuda.empty_cache()
    print(f"  VAE decode done in {time.time()-t0:.1f}s")
    return image


# ── helpers from pipeline ───────────────────────────────────────────────────

def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _pack_latents(latents, batch_size, num_channels, height, width, patch_size):
    latents = latents.view(
        batch_size, num_channels, height // patch_size, patch_size,
        width // patch_size, patch_size,
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
    return latents.reshape(
        batch_size, (height // patch_size) * (width // patch_size),
        num_channels * patch_size * patch_size,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    total_t0 = time.time()

    prompt = "A fluffy orange cat sitting on a windowsill, looking outside at a sunset cityscape."
    height, width = 1024, 1024
    num_steps = 50
    seed = 42
    device = "cuda"
    patch_size = 2

    # ── Stage 1: Text Encoding (positive + negative for CFG) ─────────────
    cfg_scale = 8.0
    do_cfg = True
    prompt_embeds, prompt_mask = encode_text(prompt, device)
    if do_cfg:
        neg_prompt_embeds, neg_prompt_mask = encode_text("", device)  # empty negative

    # ── Stage 2: Transformer Block-Swap ──────────────────────────────────
    from diffusers import FlowMatchEulerDiscreteScheduler

    transformer, trans_config = load_transformer_block_swap(device)

    # Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(FP8_DIR, "scheduler")
    )

    # Prepare latents
    vae_sf = 8  # 2^3
    num_ch = trans_config["in_channels"] // 4  # 64 // 4 = 16
    h_lat = patch_size * (height // (vae_sf * patch_size))  # 128
    w_lat = patch_size * (width // (vae_sf * patch_size))   # 128

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((1, 1, num_ch, h_lat, w_lat),
                        generator=generator, device=device, dtype=torch.bfloat16)
    latents = _pack_latents(noise, 1, num_ch, h_lat, w_lat, patch_size)

    img_shapes = [(1, h_lat // patch_size, w_lat // patch_size)]

    # Timesteps
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.set_begin_index(0)
    scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device, mu=mu)
    timesteps = scheduler.timesteps
    num_train_ts = scheduler.config.num_train_timesteps  # 1000

    print(f"\n[Denoising] {len(timesteps)} steps, seq_len={image_seq_len}")
    print(f"  Latent shape: {latents.shape}, img_shapes={img_shapes}")
    print(f"  Timesteps: {timesteps[:5].tolist()} ... {timesteps[-3:].tolist()}")
    print(f"  Timestep / 1000 range: [{(timesteps[-1]/num_train_ts).item():.4f}, {(timesteps[0]/num_train_ts).item():.4f}]")
    print(f"  mu={mu:.4f}")

    # Denoising loop with CFG
    dt0 = time.time()
    for i, t in enumerate(timesteps):
        st0 = time.time()
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        # Positive prediction
        noise_pred = transformer(
            hidden_states=latents,
            timestep=timestep / num_train_ts,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]

        # Negative prediction (CFG)
        if do_cfg:
            neg_noise_pred = transformer(
                hidden_states=latents,
                timestep=timestep / num_train_ts,
                encoder_hidden_states=neg_prompt_embeds,
                encoder_hidden_states_mask=neg_prompt_mask,
                img_shapes=img_shapes,
                return_dict=False,
            )[0]

            # CFG with adaptive normalization
            comb_pred = neg_noise_pred + cfg_scale * (noise_pred - neg_noise_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

        noise_pred = -noise_pred

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if i == 0 or i == len(timesteps) - 1:
            print(f"  Step {i+1:3d}/{len(timesteps)}: {time.time()-st0:.2f}s  "
                  f"pos_range=[{noise_pred.min():.3f},{noise_pred.max():.3f}] "
                  f"lat_range=[{latents.min():.3f},{latents.max():.3f}] "
                  f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
        else:
            print(f"  Step {i+1:3d}/{len(timesteps)}: {time.time()-st0:.2f}s")

    print(f"\n  Denoising completed in {time.time()-dt0:.1f}s")
    del transformer; gc.collect(); torch.cuda.empty_cache()

    # ── Stage 3: VAE Decode ──────────────────────────────────────────────
    image = decode_latents(latents, height, width, patch_size, device)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output.png")
    image.save(out_path)
    print(f"\n  Image saved to: {out_path}")
    print(f"  TOTAL TIME: {time.time()-total_t0:.1f}s")


if __name__ == "__main__":
    main()
