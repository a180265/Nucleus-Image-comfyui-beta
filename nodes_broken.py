import os
import json
import gc
import types
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from safetensors import safe_open
import folder_paths
import comfy.model_management as mm
import comfy.samplers
import comfy.utils

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(NODE_DIR, "configs")
PROCESSOR_DIR = os.path.join(NODE_DIR, "processor")

SYSTEM_PROMPT = (
    "You are an image generation assistant. Follow the user's prompt literally. "
    "Pay careful attention to spatial layout: objects described as on the left must "
    "appear on the left, on the right on the right. Match exact object counts and "
    "assign colors to the correct objects."
)

VAE_SCALE_FACTOR = 8
PATCH_SIZE = 2
NUM_TRAIN_TIMESTEPS = 1000

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ══════════════════════════════════════════════════════════════════════════════
# FP8 / Weight Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _dequant_fp8(tensor, scale):
    """Dequantize FP8 tensor to bf16 via FP32 intermediate for max precision."""
    if tensor.dtype == torch.float8_e4m3fn and scale is not None:
        return (tensor.float() * torch.tensor(scale, dtype=torch.float32, device=tensor.device)).to(torch.bfloat16)
    if tensor.is_floating_point():
        return tensor.to(torch.bfloat16)
    return tensor


def _detect_weight_dtype(safetensors_path):
    """Detect weight format by checking weight-like tensors first."""
    with safe_open(safetensors_path, framework="pt") as f:
        keys = list(f.keys())
        # Prioritize weight-like keys (skip biases, norms, positional embeddings)
        weight_keys = [k for k in keys if any(w in k for w in ["weight", "proj"])]
        check_keys = weight_keys if weight_keys else keys[:5]
        for key in check_keys:
            tensor = f.get_tensor(key)
            if tensor.dtype == torch.float8_e4m3fn:
                return "fp8_e4m3fn"
            if tensor.dtype == torch.float8_e5m2:
                return "fp8_e5m2"
            return "bf16"  # First weight tensor is not FP8
    return "bf16"


def _load_fp8_scales(safetensors_path):
    """Read FP8 quantization scales from safetensors metadata."""
    scales = {}
    with safe_open(safetensors_path, framework="pt") as f:
        meta = f.metadata()
        if meta and "quantization" in meta:
            scales = json.loads(meta["quantization"]).get("scales", {})
    return scales


def _load_weights_into_model(model, safetensors_path, device=None, target_dtype=None):
    """Load safetensors weights into a model, handling FP8 dequantization.

    For FP8 weights on CPU: bulk-loads file with load_file (fast on Windows),
    then dequants each tensor on GPU and copies result back to CPU parameter.
    Peak extra GPU: one tensor at a time (~200MB), not the full model.
    """
    is_fp8 = _detect_weight_dtype(safetensors_path) == "fp8_e4m3fn"
    scales = _load_fp8_scales(safetensors_path) if is_fp8 else {}
    param_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())

    # GPU-assisted dequant for FP8: when model target is CPU, dequant each
    # tensor on GPU (fast native FP8 support) then copy back. This avoids
    # allocating the full model on GPU (saves ~16GB VRAM).
    gpu_dequant = False
    gpu_dev = None
    if is_fp8:
        gpu_dev = mm.get_torch_device()
        gpu_dequant = device is not None and str(device) != str(gpu_dev)

    sd = load_file(safetensors_path)
    for key in list(sd.keys()):
        tensor = sd.pop(key)
        if key in param_map:
            if is_fp8:
                if gpu_dequant:
                    tensor = _dequant_fp8(tensor.to(gpu_dev), scales.get(key))
                    param_map[key].data.copy_(tensor)  # GPU bf16 → CPU param
                    del tensor
                    continue
                elif device is not None:
                    tensor = tensor.to(device)
                tensor = _dequant_fp8(tensor, scales.get(key))
            else:
                if target_dtype and tensor.is_floating_point():
                    tensor = tensor.to(target_dtype)
                if device is not None and tensor.is_floating_point():
                    tensor = tensor.to(device)
            param_map[key].data.copy_(tensor)
        elif key in buffer_map:
            buffer_map[key].copy_(tensor)
        del tensor
    del sd
    gc.collect()


def _set_nested_attr(model, name, value):
    """Set a nested attribute on a model, handling Parameters."""
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
    """Check if a parameter name belongs to MoE expert weights."""
    return "experts.gate_up_proj" in name or "experts.down_proj" in name


def _get_block_idx(name):
    """Extract transformer block index from parameter name."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "transformer_blocks":
            return int(parts[i + 1])
    return -1


def _resolve_device(device_str):
    """Convert user-friendly device string to torch device."""
    if device_str == "GPU":
        return mm.get_torch_device()
    return mm.unet_offload_device()  # CPU


# ══════════════════════════════════════════════════════════════════════════════
# Block-Swap Expert Forward Patching
# ══════════════════════════════════════════════════════════════════════════════

def _patch_expert_forward(experts_module, block_idx, expert_data, device):
    """Patch an expert module's forward to load/unload weights per call."""
    gu_raw = expert_data["gate_up_proj"]
    dp_raw = expert_data["down_proj"]
    gu_scale = expert_data.get("gu_scale")
    dp_scale = expert_data.get("dp_scale")
    is_fp8 = expert_data.get("is_fp8", False)
    on_gpu = expert_data.get("on_gpu", False)

    _cache = {"gu": None, "dp": None}

    def _dequant_weight(raw, scale, dev):
        if is_fp8 and raw.dtype == torch.float8_e4m3fn:
            if scale:
                scale_t = torch.tensor(scale, dtype=torch.bfloat16, device=dev)
                return raw.to(dev).to(torch.bfloat16) * scale_t
            return raw.to(dev).to(torch.bfloat16)
        return raw.to(dev)

    # For on_gpu FP8: dequant fresh every call (no cache). This saves ~2x VRAM
    # since we only keep FP8 raw (~264MB/block) instead of FP8 + bf16 cache
    # (~792MB/block). GPU dequant is fast (~0.5ms per tensor).
    fp8_nocache = on_gpu and is_fp8

    if fp8_nocache:
        def _load():
            _cache["gu"] = _dequant_weight(gu_raw, gu_scale, device)
            _cache["dp"] = _dequant_weight(dp_raw, dp_scale, device)

        def _unload():
            _cache["gu"] = None
            _cache["dp"] = None
    else:
        def _load():
            if _cache["gu"] is not None:
                return
            _cache["gu"] = _dequant_weight(gu_raw, gu_scale, device)
            _cache["dp"] = _dequant_weight(dp_raw, dp_scale, device)

        def _unload():
            if not on_gpu:
                _cache["gu"] = None
                _cache["dp"] = None

    use_gmm = experts_module.use_grouped_mm

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
    experts_module._expert_cache = _cache  # expose for cleanup


def _clear_expert_caches(transformer, moe_blocks):
    """Clear GPU-cached expert weights to free VRAM."""
    for block_idx in moe_blocks:
        block = transformer.transformer_blocks[block_idx]
        if hasattr(block.img_mlp.experts, '_expert_cache'):
            block.img_mlp.experts._expert_cache["gu"] = None
            block.img_mlp.experts._expert_cache["dp"] = None


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler / Latent Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
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


def _format_prompt(processor, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ══════════════════════════════════════════════════════════════════════════════
# Node 1: Transformer Loader
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageTransformerLoader:
    """Load Nucleus-Image transformer from standard ComfyUI diffusion_models directory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "precision": (["bf16", "fp16", "fp32"],),
                "quantization": (["disabled", "fp8_e4m3fn"],),
                "device": (["CPU", "GPU"], {"default": "GPU",
                    "tooltip": "Device to load non-expert weights to. Expert weights follow block_swap config."}),
            },
            "optional": {
                "block_swap_args": ("BLOCKSWAPARGS",),
            },
        }

    RETURN_TYPES = ("NUCLEUS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Nucleus-Image"

    def load_model(self, model_name, precision, quantization, device, block_swap_args=None):
        from diffusers.models import NucleusMoEImageTransformer2DModel
        from diffusers.models.transformers.transformer_nucleusmoe_image import SwiGLUExperts

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        compute_dtype = DTYPE_MAP[precision]
        load_device = _resolve_device(device)

        # Auto-detect weight format
        weight_dtype = _detect_weight_dtype(model_path)
        is_fp8 = weight_dtype == "fp8_e4m3fn"
        if quantization == "disabled" and is_fp8:
            quantization = "fp8_e4m3fn"

        print(f"[Nucleus-Image] Loading transformer: {model_name}")
        print(f"[Nucleus-Image] Weight: {weight_dtype}, Compute: {precision}, Quant: {quantization}, Device: {device}")

        # Monkey-patch SwiGLUExperts to allocate tiny expert weights
        _orig_init = SwiGLUExperts.__init__
        def _tiny_init(self, hidden_size, moe_intermediate_dim, num_experts, use_grouped_mm=False):
            nn.Module.__init__(self)
            self.num_experts = num_experts
            self.moe_intermediate_dim = moe_intermediate_dim
            self.hidden_size = hidden_size
            self.use_grouped_mm = use_grouped_mm
            self.gate_up_proj = nn.Parameter(torch.empty(1, 1, 1, dtype=torch.bfloat16))
            self.down_proj = nn.Parameter(torch.empty(1, 1, 1, dtype=torch.bfloat16))

        SwiGLUExperts.__init__ = _tiny_init
        try:
            config_path = os.path.join(CONFIGS_DIR, "transformer_config.json")
            config = NucleusMoEImageTransformer2DModel.load_config(config_path)
            transformer = NucleusMoEImageTransformer2DModel.from_config(config)
        finally:
            SwiGLUExperts.__init__ = _orig_init

        # Load weights, separating expert vs non-expert
        fp8_scales = _load_fp8_scales(model_path) if is_fp8 else {}
        sd = load_file(model_path)

        expert_data = {}
        for name in list(sd.keys()):
            tensor = sd.pop(name)
            if _is_expert_param(name):
                block_idx = _get_block_idx(name)
                ptype = "gate_up_proj" if "gate_up_proj" in name else "down_proj"
                expert_data.setdefault(block_idx, {})[ptype] = tensor
                if name in fp8_scales:
                    expert_data[block_idx][f"{ptype}_scale"] = fp8_scales[name]
                if is_fp8:
                    expert_data[block_idx].setdefault("is_fp8", True)
            else:
                if is_fp8:
                    bf16 = _dequant_fp8(tensor, fp8_scales.get(name))
                else:
                    bf16 = tensor.to(compute_dtype) if tensor.is_floating_point() else tensor
                _set_nested_attr(transformer, name, nn.Parameter(bf16))
            del tensor
        del sd, fp8_scales
        gc.collect()

        # Parse block swap config
        bs_args = block_swap_args if block_swap_args and block_swap_args.get("enabled", True) else None
        blocks_to_swap = bs_args.get("blocks_to_swap", 0) if bs_args else 0
        num_blocks = len(transformer.transformer_blocks)
        swap_start = num_blocks - blocks_to_swap
        moe_blocks = [i for i, b in enumerate(transformer.transformer_blocks) if b.moe_enabled]

        gpu_device = mm.get_torch_device()

        # Move non-expert params to load_device
        for name, param in transformer.named_parameters():
            if not _is_expert_param(name):
                param.data = param.data.to(load_device).to(compute_dtype)

        # Patch expert forwards with block-swap awareness
        for block_idx in moe_blocks:
            if block_idx not in expert_data:
                continue
            block = transformer.transformer_blocks[block_idx]
            on_gpu = block_idx < swap_start

            ed = expert_data[block_idx]
            ed["on_gpu"] = on_gpu

            if on_gpu:
                # Move to GPU — closure holds the GPU tensor directly
                ed["gate_up_proj"] = ed["gate_up_proj"].to(gpu_device)
                ed["down_proj"] = ed["down_proj"].to(gpu_device)

            _patch_expert_forward(block.img_mlp.experts, block_idx, ed, gpu_device)

            # For CPU blocks: closure already references the same tensors in
            # expert_data, so no copy needed. For GPU blocks: the original CPU
            # tensors were replaced above, so expert_data entry is now lightweight.

        # expert_data is no longer needed — all tensors are held by closures.
        # Free the dict so ComfyUI doesn't cache ~30GB of expert weights.
        del expert_data
        gc.collect()

        return ({
            "transformer": transformer,
            "config": config,
            "precision": precision,
            "quantization": quantization,
            "blocks_to_swap": blocks_to_swap,
            "num_blocks": num_blocks,
            "swap_start": swap_start,
            "moe_blocks": moe_blocks,
        },)


# ══════════════════════════════════════════════════════════════════════════════
# Node 2: Text Encoder Loader
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageTextEncoderLoader:
    """Load Qwen3-VL text encoder from standard ComfyUI text_encoders directory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("text_encoders"),),
                "precision": (["bf16", "fp16", "fp32"],),
                "quantization": (["disabled", "fp8_e4m3fn"],),
                "device": (["CPU", "GPU"], {"default": "CPU",
                    "tooltip": "Device to load text encoder to. GPU is faster but uses more VRAM."}),
            },
        }

    RETURN_TYPES = ("NUCLEUS_TEXT_ENCODER",)
    RETURN_NAMES = ("text_encoder",)
    FUNCTION = "load_model"
    CATEGORY = "Nucleus-Image"

    def load_model(self, model_name, precision, quantization, device):
        from transformers import Qwen3VLForConditionalGeneration, AutoConfig
        from transformers import Qwen3VLProcessor

        model_path = folder_paths.get_full_path_or_raise("text_encoders", model_name)
        compute_dtype = DTYPE_MAP[precision]
        load_device = _resolve_device(device)

        weight_dtype = _detect_weight_dtype(model_path)
        is_fp8 = weight_dtype == "fp8_e4m3fn"
        if quantization == "disabled" and is_fp8:
            quantization = "fp8_e4m3fn"

        print(f"[Nucleus-Image] Loading text encoder: {model_name}")
        print(f"[Nucleus-Image] Weight: {weight_dtype}, Compute: {precision}, Quant: {quantization}, Device: {device}")

        processor = Qwen3VLProcessor.from_pretrained(PROCESSOR_DIR)

        te_config_path = os.path.join(CONFIGS_DIR, "text_encoder_config.json")
        te_config = AutoConfig.from_pretrained(te_config_path)
        # Create on meta device (instant, no allocation/initialization) then
        # materialize on target device with to_empty (fast, no random init).
        # This skips the slow ~10min CPU random init for a 16.5GB model.
        with torch.device("meta"):
            text_encoder = Qwen3VLForConditionalGeneration._from_config(
                te_config, dtype=compute_dtype
            )
        text_encoder = text_encoder.to_empty(device=load_device)
        _load_weights_into_model(text_encoder, model_path, device=load_device, target_dtype=compute_dtype)

        # Re-initialize rotary embedding buffers (not stored in safetensors,
        # left as garbage by to_empty). Without this, self-attention produces
        # extreme values and the model outputs garbage embeddings.
        for module in text_encoder.modules():
            if hasattr(module, 'inv_freq') and hasattr(module, 'rope_init_fn'):
                inv_freq, _ = module.rope_init_fn(module.config, device=load_device)
                module.inv_freq = inv_freq
                if hasattr(module, 'original_inv_freq'):
                    module.original_inv_freq = inv_freq

        text_encoder.eval()

        return ({
            "model": text_encoder,
            "processor": processor,
            "load_device": load_device,
            "current_device": load_device,
            "precision": precision,
            "quantization": quantization,
        },)


# ══════════════════════════════════════════════════════════════════════════════
# Node 3: VAE Loader
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageVAELoader:
    """Load Nucleus-Image VAE from standard ComfyUI vae directory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"),),
                "precision": (["bf16", "fp16", "fp32"],),
            },
        }

    RETURN_TYPES = ("NUCLEUS_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_model"
    CATEGORY = "Nucleus-Image"

    def load_model(self, model_name, precision):
        from diffusers import AutoencoderKLQwenImage

        model_path = folder_paths.get_full_path_or_raise("vae", model_name)
        compute_dtype = DTYPE_MAP[precision]
        device = mm.get_torch_device()

        weight_dtype = _detect_weight_dtype(model_path)
        print(f"[Nucleus-Image] Loading VAE: {model_name} (weight: {weight_dtype}, compute: {precision})")

        vae_config_path = os.path.join(CONFIGS_DIR, "vae_config.json")
        vae_config = AutoencoderKLQwenImage.load_config(vae_config_path)
        with torch.device(device):
            vae = AutoencoderKLQwenImage.from_config(vae_config)

        _load_weights_into_model(vae, model_path, device=device, target_dtype=compute_dtype)
        vae.eval()

        return ({"model": vae, "precision": precision},)


# ══════════════════════════════════════════════════════════════════════════════
# Node 4: Block Swap Config
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageBlockSwap:
    """Configure block swap for Nucleus-Image transformer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "blocks_to_swap": ("INT", {"default": 12, "min": 0, "max": 29, "step": 1,
                    "tooltip": "Number of MoE blocks to swap to CPU. 0 = all on GPU (fastest). 12 recommended for bf16 on 24GB VRAM."}),
                "prefetch_blocks": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
            },
        }

    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "configure"
    CATEGORY = "Nucleus-Image"

    def configure(self, enabled, blocks_to_swap, prefetch_blocks):
        return ({
            "enabled": enabled,
            "blocks_to_swap": blocks_to_swap,
            "prefetch_blocks": prefetch_blocks,
        },)


# ══════════════════════════════════════════════════════════════════════════════
# Node 5: Text Encode
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageTextEncode:
    """Encode text prompt using loaded Qwen3-VL text encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("NUCLEUS_TEXT_ENCODER",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "force_offload": ("BOOLEAN", {"default": True,
                    "tooltip": "Move text encoder back to CPU after encoding to free VRAM."}),
            },
        }

    RETURN_TYPES = ("NUCLEUS_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Nucleus-Image"

    def encode(self, text_encoder, text, force_offload):
        model = text_encoder["model"]
        processor = text_encoder["processor"]
        current_device = text_encoder.get("current_device", "cpu")
        gpu_device = mm.get_torch_device()

        # Move to GPU for encoding if not already there
        is_on_cpu = str(current_device) == "cpu"
        if is_on_cpu:
            # Estimate model size for progress feedback
            model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
            print(f"[Nucleus-Image] Moving text encoder to GPU ({model_size_gb:.1f} GB)...")
            if model_size_gb > 10:
                print(f"[Nucleus-Image] WARNING: Large text encoder on CPU ({model_size_gb:.1f} GB). "
                      f"If this hangs, use FP8 text encoder instead (8GB vs 16GB).")
        model.to(gpu_device)
        text_encoder["current_device"] = gpu_device
        if is_on_cpu:
            gc.collect()
            print(f"[Nucleus-Image] Text encoder on GPU, starting encoding...")

        formatted = _format_prompt(processor, text if text else "")

        inputs = processor(
            text=[formatted],
            padding="longest",
            pad_to_multiple_of=8,
            max_length=1024,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device=gpu_device)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=False, return_dict=True, output_hidden_states=True)
            prompt_embeds = outputs.hidden_states[-8].to(dtype=torch.bfloat16, device=gpu_device)

        prompt_mask = inputs.attention_mask
        if prompt_mask is not None and prompt_mask.all():
            prompt_mask = None

        # Offload text encoder to free VRAM
        if force_offload:
            offload_dev = mm.unet_offload_device()
            model.to(offload_dev)
            text_encoder["current_device"] = offload_dev
            gc.collect()
            torch.cuda.empty_cache()

        return ({"prompt_embeds": prompt_embeds, "prompt_mask": prompt_mask},)


# ══════════════════════════════════════════════════════════════════════════════
# k-diffusion Model Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class _NucleusDenoiser:
    """Wraps the transformer for k-diffusion sampler compatibility.

    Converts velocity predictions to x0 (denoised) predictions that
    k-diffusion samplers expect, handling CFG internally.
    """

    def __init__(self, transformer, pos_embeds, pos_mask, neg_embeds, neg_mask,
                 cfg, cfg_rescale, img_shapes, do_cfg):
        self.transformer = transformer
        self.pos_embeds = pos_embeds
        self.pos_mask = pos_mask
        self.neg_embeds = neg_embeds
        self.neg_mask = neg_mask
        self.cfg = cfg
        self.cfg_rescale = cfg_rescale
        self.img_shapes = img_shapes
        self.do_cfg = do_cfg
        # Compat: some k-diffusion samplers check model.inner_model
        self.inner_model = types.SimpleNamespace(
            inner_model=types.SimpleNamespace(model_sampling=None)
        )

    def __call__(self, x, sigma, **kwargs):
        # k-diffusion arithmetic (x + d * dt with float32 sigmas) can promote
        # x to float32 between steps. Cast back to model dtype each call.
        x = x.to(torch.bfloat16)

        # Ensure sigma is 1D tensor matching batch size
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma], device=x.device, dtype=x.dtype)
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.shape[0] < x.shape[0]:
            sigma = sigma.expand(x.shape[0])

        # sigma is in [0, 1] range for flow matching — pass directly as timestep
        # Cast to bf16 to match official pipeline: timestep.to(latents.dtype)
        timestep = sigma.to(torch.bfloat16)

        # Positive prediction
        output = self.transformer(
            hidden_states=x,
            timestep=timestep,
            encoder_hidden_states=self.pos_embeds,
            encoder_hidden_states_mask=self.pos_mask,
            img_shapes=self.img_shapes,
            return_dict=False,
        )[0]
        v_pos = -output  # negate: model outputs -v, k-diffusion needs x0

        if self.do_cfg:
            output_neg = self.transformer(
                hidden_states=x,
                timestep=timestep,
                encoder_hidden_states=self.neg_embeds,
                encoder_hidden_states_mask=self.neg_mask,
                img_shapes=self.img_shapes,
                return_dict=False,
            )[0]
            v_neg = -output_neg

            # CFG combination
            v_comb = v_neg + self.cfg * (v_pos - v_neg)

            # Adaptive normalization with cfg_rescale interpolation
            # rescale=1.0 → full normalization (official behavior)
            # rescale=0.0 → raw CFG (no normalization)
            if self.cfg_rescale > 0.0:
                cond_norm = torch.norm(v_pos, dim=-1, keepdim=True).clamp(min=1e-6)
                comb_norm = torch.norm(v_comb, dim=-1, keepdim=True).clamp(min=1e-6)
                v_rescaled = v_comb * (cond_norm / comb_norm)
                v_final = self.cfg_rescale * v_rescaled + (1.0 - self.cfg_rescale) * v_comb
            else:
                v_final = v_comb
        else:
            v_final = v_pos

        # Velocity → x0 (denoised prediction for k-diffusion)
        # x0 = x - sigma * v
        sigma_3d = sigma.reshape(-1, 1, 1)
        x0 = x - sigma_3d * v_final
        return x0


# ══════════════════════════════════════════════════════════════════════════════
# Node 6: Advanced Config
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageAdvancedConfig:
    """Configure advanced sampling parameters: CFG rescale and timestep shift."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cfg_rescale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "CFG adaptive normalization strength. 1.0 = official (recommended). 0.0 = raw CFG."}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Override timestep shift (mu). 0.0 = auto from resolution. "
                               "Higher values shift noise schedule toward low-noise range."}),
            },
        }

    RETURN_TYPES = ("NUCLEUS_ADVANCED_ARGS",)
    RETURN_NAMES = ("advanced_args",)
    FUNCTION = "configure"
    CATEGORY = "Nucleus-Image"

    def configure(self, cfg_rescale, shift):
        return ({"cfg_rescale": cfg_rescale, "shift": shift},)


# ══════════════════════════════════════════════════════════════════════════════
# Node 7: Sampler
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageSampler:
    """Sample latents with CFG, block-swap, and ComfyUI sampler/scheduler support."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("NUCLEUS_MODEL",),
                "positive": ("NUCLEUS_CONDITIONING",),
                "negative": ("NUCLEUS_CONDITIONING",),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS,),
                "force_offload": ("BOOLEAN", {"default": True,
                    "tooltip": "Move transformer to CPU after sampling to free VRAM for VAE decode."}),
            },
            "optional": {
                "advanced_args": ("NUCLEUS_ADVANCED_ARGS",),
            },
        }

    RETURN_TYPES = ("NUCLEUS_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "Nucleus-Image"
    OUTPUT_NODE = True

    @torch.no_grad()
    def sample(self, model, positive, negative, width, height, steps, cfg,
               seed, sampler_name, scheduler_name, force_offload, advanced_args=None):
        from comfy.k_diffusion import sampling as k_sampling

        device = mm.get_torch_device()
        transformer = model["transformer"]
        config = model["config"]

        # Parse advanced args (use defaults if not connected)
        adv = advanced_args or {}
        cfg_rescale = adv.get("cfg_rescale", 1.0)

        # Move transformer to GPU for sampling
        transformer.to(device)

        # Compute latent dimensions
        num_ch = config["in_channels"] // 4  # 64 // 4 = 16
        h_lat = PATCH_SIZE * (height // (VAE_SCALE_FACTOR * PATCH_SIZE))
        w_lat = PATCH_SIZE * (width // (VAE_SCALE_FACTOR * PATCH_SIZE))
        img_shapes = [(1, h_lat // PATCH_SIZE, w_lat // PATCH_SIZE)]

        # Generate noise
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn((1, 1, num_ch, h_lat, w_lat),
                            generator=generator, device=device, dtype=torch.bfloat16)
        latents = _pack_latents(noise, 1, num_ch, h_lat, w_lat, PATCH_SIZE)

        # Generate sigmas (matching official FlowMatchEulerDiscreteScheduler)
        sigmas = self._generate_sigmas(steps, scheduler_name, device)

        # Get conditioning
        pos_embeds = positive["prompt_embeds"]
        pos_mask = positive["prompt_mask"]
        neg_embeds = negative["prompt_embeds"]
        neg_mask = negative["prompt_mask"]
        do_cfg = cfg > 1.0

        # Create k-diffusion model wrapper (handles CFG + velocity→x0 conversion)
        denoiser = _NucleusDenoiser(
            transformer, pos_embeds, pos_mask, neg_embeds, neg_mask,
            cfg, cfg_rescale, img_shapes, do_cfg,
        )

        # Select sampler function (no silent fallback)
        sampler_func = self._get_sampler_func(sampler_name, k_sampling)

        # Progress callback (suppress k-diffusion's internal tqdm)
        pbar = comfy.utils.ProgressBar(steps)
        def callback(d):
            pbar.update_absolute(d["i"] + 1, steps)

        print(f"[Nucleus-Image] Sampling: {sampler_name} + {scheduler_name}, "
              f"{steps} steps, cfg={cfg}, cfg_rescale={cfg_rescale}")

        # Run sampling via k-diffusion
        latents = sampler_func(denoiser, latents, sigmas, callback=callback, disable=True)

        # Offload transformer + clear expert caches to free GPU for VAE decode
        if force_offload:
            _clear_expert_caches(transformer, model["moe_blocks"])
            transformer.to(mm.unet_offload_device())
            gc.collect()
            torch.cuda.empty_cache()

        return ({"latents": latents, "width": width, "height": height},)

    @staticmethod
    def _get_sampler_func(sampler_name, k_sampling):
        """Get the k-diffusion sampler function for the given name.

        Handles special-case samplers that need signature adaptation.
        No silent fallback — raises AttributeError if function not found.
        """
        if sampler_name == "dpm_fast":
            # dpm_fast takes (model, x, sigma_min, sigma_max, n, ...) not (model, x, sigmas, ...)
            def _wrap(model, x, sigmas, extra_args=None, callback=None, disable=None, **kw):
                sigma_min = sigmas[sigmas > 0].min().item()
                sigma_max = sigmas[0].item()
                n = len(sigmas) - 1
                return k_sampling.sample_dpm_fast(
                    model, x, sigma_min, sigma_max, n,
                    extra_args=extra_args or {}, callback=callback, disable=disable,
                )
            return _wrap

        if sampler_name == "dpm_adaptive":
            # dpm_adaptive takes (model, x, sigma_min, sigma_max, ...) not (model, x, sigmas, ...)
            def _wrap(model, x, sigmas, extra_args=None, callback=None, disable=None, **kw):
                sigma_min = sigmas[sigmas[sigmas > 0]].min().item()
                sigma_max = sigmas[0].item()
                return k_sampling.sample_dpm_adaptive(
                    model, x, sigma_min, sigma_max,
                    extra_args=extra_args or {}, callback=callback, disable=disable,
                )
            return _wrap

        if sampler_name == "ddim":
            # DDIM = Euler (no dedicated k-diffusion function)
            return k_sampling.sample_euler

        if sampler_name in ("uni_pc", "uni_pc_bh2"):
            # UniPC lives in a separate module
            from comfy.extra_samplers import uni_pc
            variant = "bh1" if sampler_name == "uni_pc" else "bh2"
            def _wrap(model, x, sigmas, extra_args=None, callback=None, disable=None, **kw):
                return uni_pc.sample_unipc(
                    model, x, sigmas,
                    extra_args=extra_args or {}, callback=callback, disable=disable,
                    variant=variant,
                )
            return _wrap

        # All other samplers: direct k-diffusion function lookup (no fallback)
        func = getattr(k_sampling, f"sample_{sampler_name}")
        return func

    @staticmethod
    def _generate_sigmas(steps, scheduler_name, device):
        """Generate sigmas matching the official FlowMatchEulerDiscreteScheduler.

        Official config: use_dynamic_shifting=false, shift=1.0, num_train_timesteps=1000.
        This means sigma_max=1.0, sigma_min=1/1000=0.001, no dynamic shift applied.
        """
        sigma_max = 1.0
        sigma_min = 1.0 / NUM_TRAIN_TIMESTEPS  # 0.001

        if scheduler_name == "normal":
            sigmas = np.linspace(sigma_max, sigma_min, steps)
        elif scheduler_name == "karras":
            rho = 7.0
            t_k = np.linspace(0, 1, steps)
            sigmas = (sigma_max ** (1 / rho) + t_k * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        elif scheduler_name == "exponential":
            sigmas = np.geomspace(sigma_max, sigma_min, steps)
        elif scheduler_name == "simple":
            sigmas = np.linspace(sigma_max, sigma_min, steps)
        else:
            raise ValueError(f"[Nucleus-Image] Unsupported scheduler: {scheduler_name}")

        sigmas = np.append(sigmas, 0.0)
        return torch.tensor(sigmas, dtype=torch.float32, device=device)


# ══════════════════════════════════════════════════════════════════════════════
# Node 8: VAE Decode
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageVAEDecode:
    """Decode latents to images using the Nucleus-Image VAE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("NUCLEUS_VAE",),
                "samples": ("NUCLEUS_LATENT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "Nucleus-Image"
    OUTPUT_NODE = True

    @torch.no_grad()
    def decode(self, vae, samples):
        vae_model = vae["model"]
        device = mm.get_torch_device()

        latents_packed = samples["latents"]
        height = samples["height"]
        width = samples["width"]

        vae_model.to(device)

        # Unpack latents: (B, seq_len, C*ps*ps) -> (B, z_dim, 1, H, W)
        bs, _, ch = latents_packed.shape
        hp = PATCH_SIZE * (int(height) // (VAE_SCALE_FACTOR * PATCH_SIZE))
        wp = PATCH_SIZE * (int(width) // (VAE_SCALE_FACTOR * PATCH_SIZE))
        latents = latents_packed.view(
            bs, hp // PATCH_SIZE, wp // PATCH_SIZE,
            ch // (PATCH_SIZE * PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5).contiguous()
        latents = latents.reshape(bs, ch // (PATCH_SIZE * PATCH_SIZE), 1, hp, wp)

        # Denormalize: latents * std + mean (MULTIPLY by std, not divide!)
        z_dim = vae_model.config.z_dim
        latents_mean = torch.tensor(vae_model.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents)
        latents_std = torch.tensor(vae_model.config.latents_std).view(1, z_dim, 1, 1, 1).to(latents)
        latents = latents * latents_std + latents_mean

        decoded = vae_model.decode(latents.to(vae_model.dtype), return_dict=False)[0][:, :, 0]

        # Offload VAE
        vae_model.to(mm.unet_offload_device())
        gc.collect()
        torch.cuda.empty_cache()

        # Convert to ComfyUI IMAGE: (B, H, W, 3) float32 [0,1]
        # Decoded values are in [-1, 1], denormalize to [0, 1]
        arr = (decoded.float() * 0.5 + 0.5).clamp(0, 1)
        if arr.dim() == 4:
            arr = arr.permute(0, 2, 3, 1)
        return (arr,)


# ══════════════════════════════════════════════════════════════════════════════
# Node Registrations
# ══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "NucleusImageTransformerLoader": NucleusImageTransformerLoader,
    "NucleusImageTextEncoderLoader": NucleusImageTextEncoderLoader,
    "NucleusImageVAELoader": NucleusImageVAELoader,
    "NucleusImageBlockSwap": NucleusImageBlockSwap,
    "NucleusImageTextEncode": NucleusImageTextEncode,
    "NucleusImageAdvancedConfig": NucleusImageAdvancedConfig,
    "NucleusImageSampler": NucleusImageSampler,
    "NucleusImageVAEDecode": NucleusImageVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NucleusImageTransformerLoader": "Nucleus-Image Transformer Loader",
    "NucleusImageTextEncoderLoader": "Nucleus-Image Text Encoder Loader",
    "NucleusImageVAELoader": "Nucleus-Image VAE Loader",
    "NucleusImageBlockSwap": "Nucleus-Image Block Swap",
    "NucleusImageTextEncode": "Nucleus-Image Text Encode",
    "NucleusImageAdvancedConfig": "Nucleus-Image Advanced Config",
    "NucleusImageSampler": "Nucleus-Image Sampler",
    "NucleusImageVAEDecode": "Nucleus-Image VAE Decode",
}
