import os
import json
import gc
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from safetensors import safe_open
import folder_paths
import comfy.utils
import comfy.samplers
import comfy.model_management as mm


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an image generation assistant. Follow the user's prompt literally. "
    "Pay careful attention to spatial layout: objects described as on the left must "
    "appear on the left, on the right on the right. Match exact object counts and "
    "assign colors to the correct objects."
)

VAE_SCALE_FACTOR = 8
PATCH_SIZE = 2
NUM_TRAIN_TIMESTEPS = 1000

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(THIS_DIR, "configs")
PROCESSOR_DIR = os.path.join(THIS_DIR, "processor")


# ══════════════════════════════════════════════════════════════════════════════
# FP8 Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_fp8_scales(path):
    scales = {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
        if meta and "quantization" in meta:
            scales = json.loads(meta["quantization"]).get("scales", {})
    return scales


def _dequant_fp8(tensor, scale):
    if tensor.dtype == torch.float8_e4m3fn and scale is not None:
        return (tensor.float() * scale).to(torch.bfloat16)
    if tensor.is_floating_point():
        return tensor.to(torch.bfloat16)
    return tensor


def _load_fp8_into_model(model, fp8_path, device=None):
    scales = _load_fp8_scales(fp8_path)
    sd = load_file(fp8_path)
    param_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())
    for key in list(sd.keys()):
        tensor = sd.pop(key)
        if key in param_map:
            if device is not None and tensor.is_floating_point():
                tensor = tensor.to(device)
            param_map[key].data.copy_(_dequant_fp8(tensor, scales.get(key)))
        elif key in buffer_map:
            buffer_map[key].copy_(tensor)
        del tensor
    del sd
    gc.collect()


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


# ══════════════════════════════════════════════════════════════════════════════
# Expert Forward — GPU-resident FP8, per-layer dequant
# ══════════════════════════════════════════════════════════════════════════════
# Memory layout for 24GB VRAM + FP8 model, block_swap=0:
#   GPU: non-expert bf16 (~3.2GB) + expert FP8 (~15.3GB) = ~18.5GB
#   Per forward: dequant one MoE layer's experts to bf16 (~1GB temp)
#   Peak: ~19.5GB — fits in 24GB with room for activations
#
# block_swap > 0: that many MoE layers keep FP8 on CPU instead of GPU.
# ══════════════════════════════════════════════════════════════════════════════

def _patch_expert_forward(experts_module, fp8_data, fp8_scales, device):
    """Patch expert forward. FP8 weights live on `device` (GPU or CPU).
    Per forward: dequant FP8→bf16 on compute device, compute, discard bf16."""
    use_gmm = experts_module.use_grouped_mm
    gu_fp8 = fp8_data["gate_up_proj"].to(device)   # stays on device
    dp_fp8 = fp8_data["down_proj"].to(device)
    gu_scale = torch.tensor(fp8_scales.get("gate_up_proj", 1.0), dtype=torch.float32, device=device)
    dp_scale = torch.tensor(fp8_scales.get("down_proj", 1.0), dtype=torch.float32, device=device)

    if use_gmm:
        def _forward(self, x, num_tokens_per_expert):
            gu = (gu_fp8.float() * gu_scale).to(torch.bfloat16)
            dp = (dp_fp8.float() * dp_scale).to(torch.bfloat16)
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            gate_up = F.grouped_mm(x, gu, offs=offsets)
            gate, up = gate_up.chunk(2, dim=-1)
            out = F.grouped_mm(F.silu(gate) * up, dp, offs=offsets)
            return out.type_as(x)
    else:
        def _forward(self, x, num_tokens_per_expert):
            gu = (gu_fp8.float() * gu_scale).to(torch.bfloat16)
            dp = (dp_fp8.float() * dp_scale).to(torch.bfloat16)
            ntl = num_tokens_per_expert.tolist()
            num_real = sum(ntl)
            num_pad = x.shape[0] - num_real
            x_per = torch.split(x[:num_real], ntl, dim=0)
            outs = []
            for ei, xe in enumerate(x_per):
                gu_out = torch.matmul(xe, gu[ei])
                g, u = gu_out.chunk(2, dim=-1)
                outs.append(torch.matmul(F.silu(g) * u, dp[ei]))
            result = torch.cat(outs, dim=0)
            if num_pad > 0:
                result = torch.vstack((result, result.new_zeros((num_pad, result.shape[-1]))))
            return result

    experts_module.forward = types.MethodType(_forward, experts_module)


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler / Latent helpers
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
# Node: Block Swap
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageBlockSwap:
    """blocks_to_swap=0 → all expert FP8 on GPU (~18.5GB total, fastest).
    blocks_to_swap=29 → all expert FP8 on CPU (~3.2GB GPU, slowest)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 29, "step": 1}),
            },
        }

    RETURN_TYPES = ("BLOCKSWAPARGS",)
    FUNCTION = "set_args"
    CATEGORY = "Nucleus-Image"

    def set_args(self, blocks_to_swap):
        return ({"blocks_to_swap": blocks_to_swap},)


# ══════════════════════════════════════════════════════════════════════════════
# Node: Model Shift
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageModelShift:
    """Override sigma shift. If not connected, uses scheduler config defaults."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("NUCLEUS_MODEL",)},
            "optional": {
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("NUCLEUS_MODEL",)
    FUNCTION = "set_shift"
    CATEGORY = "Nucleus-Image"

    def set_shift(self, model, base_shift=0.5, max_shift=1.15):
        model = dict(model)
        model["scheduler_config"] = dict(model.get("scheduler_config", {}))
        model["scheduler_config"]["base_shift"] = base_shift
        model["scheduler_config"]["max_shift"] = max_shift
        return (model,)


# ══════════════════════════════════════════════════════════════════════════════
# Node: CFG Rescale
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageCFGRescale:
    """CFG adaptive normalization strength. 1.0 = official pipeline default."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("NUCLEUS_MODEL",)},
            "optional": {
                "cfg_rescale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("NUCLEUS_MODEL",)
    FUNCTION = "set_cfg_rescale"
    CATEGORY = "Nucleus-Image"

    def set_cfg_rescale(self, model, cfg_rescale=1.0):
        model = dict(model)
        model["cfg_rescale"] = cfg_rescale
        return (model,)


# ══════════════════════════════════════════════════════════════════════════════
# Node: Transformer Loader
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageTransformerLoader:
    """Load transformer. Non-expert → CPU. Expert FP8 → GPU (block_swap=0)
    or CPU (block_swap=29). Sampler moves non-expert to GPU when sampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),
                               {"tooltip": "Models from ComfyUI/models/diffusion_models"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "load_device": (["offload_device", "main_device"], {"default": "offload_device"}),
            },
            "optional": {
                "block_swap_args": ("BLOCKSWAPARGS",),
            },
        }

    RETURN_TYPES = ("NUCLEUS_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nucleus-Image"

    def load_model(self, model_name, precision, load_device, block_swap_args=None):
        from diffusers.models import NucleusMoEImageTransformer2DModel
        from diffusers.models.transformers.transformer_nucleusmoe_image import SwiGLUExperts

        blocks_to_swap = block_swap_args.get("blocks_to_swap", 0) if block_swap_args else 0
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        gpu = mm.get_torch_device()
        offload = mm.unet_offload_device()

        # Tiny-init trick for MoE experts
        _orig = SwiGLUExperts.__init__
        def _tiny(self, hidden_size, moe_intermediate_dim, num_experts, use_grouped_mm=False):
            nn.Module.__init__(self)
            self.num_experts = num_experts
            self.moe_intermediate_dim = moe_intermediate_dim
            self.hidden_size = hidden_size
            self.use_grouped_mm = use_grouped_mm
            self.gate_up_proj = nn.Parameter(torch.empty(1, 1, 1, dtype=torch.bfloat16))
            self.down_proj = nn.Parameter(torch.empty(1, 1, 1, dtype=torch.bfloat16))
        SwiGLUExperts.__init__ = _tiny
        config = NucleusMoEImageTransformer2DModel.load_config(
            os.path.join(CONFIGS_DIR, "transformer_config.json"))
        transformer = NucleusMoEImageTransformer2DModel.from_config(config)
        SwiGLUExperts.__init__ = _orig

        # Load weights
        scales = _load_fp8_scales(model_path)
        sd = load_file(model_path)
        num_moe = sum(1 for b in transformer.transformer_blocks if b.moe_enabled)
        gpu_blocks = num_moe - blocks_to_swap

        expert_data = {}
        expert_scales = {}
        moe_idx = 0
        for name in list(sd.keys()):
            tensor = sd.pop(name)
            if _is_expert_param(name):
                parts = name.split(".")
                bi = int(parts[1])
                pt_ = "gate_up_proj" if "gate_up_proj" in name else "down_proj"
                expert_data.setdefault(bi, {})[pt_] = tensor
                if name in scales:
                    expert_scales.setdefault(bi, {})[pt_] = scales[name]
            else:
                bf16 = _dequant_fp8(tensor, scales.get(name))
                _set_nested_attr(transformer, name, nn.Parameter(bf16))
            del tensor
        del sd, scales
        gc.collect()

        # Patch experts: FP8 stays on GPU or CPU
        moe_idx = 0
        for bi, block in enumerate(transformer.transformer_blocks):
            if block.moe_enabled and bi in expert_data:
                expert_device = gpu if moe_idx < gpu_blocks else offload
                _patch_expert_forward(
                    block.img_mlp.experts, expert_data[bi],
                    expert_scales.get(bi, {}), expert_device)
                moe_idx += 1
        del expert_data, expert_scales
        gc.collect()

        transformer.to(offload).eval()

        sched_cfg_path = os.path.join(CONFIGS_DIR, "scheduler_config.json")
        scheduler_config = json.load(open(sched_cfg_path)) if os.path.exists(sched_cfg_path) else {}

        return ({
            "transformer": transformer, "config": config,
            "scheduler_config": scheduler_config,
        },)


# ══════════════════════════════════════════════════════════════════════════════
# Node: Text Encoder Loader
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageTextEncoderLoader:
    """Load text encoder. Stays on CPU; TextEncode moves to GPU temporarily."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("text_encoders"),
                               {"tooltip": "Models from ComfyUI/models/text_encoders"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("NUCLEUS_TE",)
    FUNCTION = "load_model"
    CATEGORY = "Nucleus-Image"

    def load_model(self, model_name, precision):
        # Only store config + path. Actual loading happens in TextEncode on GPU.
        model_path = folder_paths.get_full_path_or_raise("text_encoders", model_name)
        return ({"model_path": model_path, "model_name": model_name},)


# ══════════════════════════════════════════════════════════════════════════════
# Node: VAE Loader
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageVAELoader:
    """Load VAE. Stays on CPU; VAEDecode moves to GPU temporarily."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"),
                               {"tooltip": "Models from ComfyUI/models/vae"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("NUCLEUS_VAE",)
    FUNCTION = "load_model"
    CATEGORY = "Nucleus-Image"

    def load_model(self, model_name, precision):
        from diffusers import AutoencoderKLQwenImage

        model_path = folder_paths.get_full_path_or_raise("vae", model_name)
        offload = mm.unet_offload_device()

        vae_config = AutoencoderKLQwenImage.load_config(
            os.path.join(CONFIGS_DIR, "vae_config.json"))
        with torch.device(offload):
            vae = AutoencoderKLQwenImage.from_config(vae_config)
        _load_fp8_into_model(vae, model_path, device=offload)
        vae.eval()

        return ({"model": vae},)


# ══════════════════════════════════════════════════════════════════════════════
# Node: Text Encode — moves TE to GPU, encodes, moves back to CPU
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageTextEncode:
    """Encode text. Temporarily moves TE to GPU, then back to CPU."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("NUCLEUS_TE",),
                "text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("NUCLEUS_CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Nucleus-Image"

    def encode(self, text_encoder, text):
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, AutoConfig

        gpu = mm.get_torch_device()
        model_path = text_encoder["model_path"]

        # Load TE directly on GPU for fast FP8 dequant (much faster than CPU)
        te_config = AutoConfig.from_pretrained(
            os.path.join(CONFIGS_DIR, "text_encoder_config.json"))
        with torch.device(gpu):
            model = Qwen3VLForConditionalGeneration._from_config(te_config, dtype=torch.bfloat16)
        _load_fp8_into_model(model, model_path, device=gpu)
        model.eval()

        processor = Qwen3VLProcessor.from_pretrained(PROCESSOR_DIR)
        formatted = _format_prompt(processor, text)
        inputs = processor(
            text=[formatted], padding="longest", pad_to_multiple_of=8,
            max_length=1024, truncation=True, return_attention_mask=True,
            return_tensors="pt",
        ).to(device=gpu)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=False, return_dict=True, output_hidden_states=True)
            prompt_embeds = outputs.hidden_states[-8].to(dtype=torch.bfloat16, device=gpu)

        prompt_mask = inputs.attention_mask
        if prompt_mask is not None and prompt_mask.all():
            prompt_mask = None

        # Free TE from GPU
        del model, outputs, inputs
        gc.collect()
        mm.soft_empty_cache()

        return ({"prompt_embeds": prompt_embeds, "prompt_mask": prompt_mask},)


# ══════════════════════════════════════════════════════════════════════════════
# Node: Text Encode Dual — positive + negative in one TE load (2x faster)
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageTextEncodeDual:
    """Encode positive + negative text in one TE load session.
    Avoids loading the 8.2GB TE model twice — roughly halves encoding time."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("NUCLEUS_TE",),
                "positive_text": ("STRING", {"default": "", "multiline": True}),
                "negative_text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("NUCLEUS_CONDITIONING", "NUCLEUS_CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "Nucleus-Image"

    def encode(self, text_encoder, positive_text, negative_text):
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, AutoConfig

        gpu = mm.get_torch_device()
        model_path = text_encoder["model_path"]

        # Load TE once
        te_config = AutoConfig.from_pretrained(
            os.path.join(CONFIGS_DIR, "text_encoder_config.json"))
        with torch.device(gpu):
            model = Qwen3VLForConditionalGeneration._from_config(te_config, dtype=torch.bfloat16)
        _load_fp8_into_model(model, model_path, device=gpu)
        model.eval()

        processor = Qwen3VLProcessor.from_pretrained(PROCESSOR_DIR)

        def _do_encode(text):
            formatted = _format_prompt(processor, text)
            inputs = processor(
                text=[formatted], padding="longest", pad_to_multiple_of=8,
                max_length=1024, truncation=True, return_attention_mask=True,
                return_tensors="pt",
            ).to(device=gpu)
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False, return_dict=True, output_hidden_states=True)
                embeds = outputs.hidden_states[-8].to(dtype=torch.bfloat16, device=gpu)
            mask = inputs.attention_mask
            if mask is not None and mask.all():
                mask = None
            del outputs, inputs
            return embeds, mask

        pos_embeds, pos_mask = _do_encode(positive_text)
        neg_embeds, neg_mask = _do_encode(negative_text)

        # Free TE
        del model
        gc.collect()
        mm.soft_empty_cache()

        return (
            {"prompt_embeds": pos_embeds, "prompt_mask": pos_mask},
            {"prompt_embeds": neg_embeds, "prompt_mask": neg_mask},
        )


# ══════════════════════════════════════════════════════════════════════════════
# Node: Zero Conditioning — no TE needed (instant, quality trade-off)
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageZeroConditioning:
    """Generate zero conditioning without loading text encoder (instant).
    Creates a minimal zero embedding. Quality may differ slightly from real
    empty-string encoding. Use NucleusImageTextEncodeDual for best quality."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("NUCLEUS_CONDITIONING",)
    FUNCTION = "zero_out"
    CATEGORY = "Nucleus-Image"

    def zero_out(self):
        # Qwen3-VL hidden_size=4096, minimal sequence for the template tokens
        embeds = torch.zeros(1, 8, 4096, dtype=torch.bfloat16)
        return ({"prompt_embeds": embeds, "prompt_mask": None},)


# ══════════════════════════════════════════════════════════════════════════════
# Sampler stepping: uses ComfyUI's sampler_object for all sampler types
# Model wrapper converts flow-matching velocity → x0 for k_diffusion
# ══════════════════════════════════════════════════════════════════════════════

class _FlowMatchNoiseScaling:
    """Noise scaling for flow matching: identity transforms."""
    def __init__(self, sigmas):
        self._sigmas = sigmas

    @property
    def sigma_max(self):
        return self._sigmas[0]

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return noise

    def inverse_noise_scaling(self, sigma, latent):
        return latent


class _MockModelPatcher:
    """Provides get_model_object() for SDE samplers that query model_sampling."""
    def __init__(self, model_sampling):
        self._model_sampling = model_sampling

    def get_model_object(self, name):
        if name == "model_sampling":
            return self._model_sampling
        raise AttributeError(f"No model object named {name}")


class _InnerModel:
    def __init__(self, model_sampling):
        self.model_sampling = model_sampling
        self.model_patcher = _MockModelPatcher(model_sampling)


class _ModelWrap:
    """Wrapper providing the interface ComfyUI's KSAMPLER expects:
    callable + .inner_model.model_sampling for noise scaling."""
    def __init__(self, model_fn, model_sampling):
        self.model_fn = model_fn
        self.inner_model = _InnerModel(model_sampling)
        self.model_patcher = _MockModelPatcher(model_sampling)

    def __call__(self, x, sigma, **kwargs):
        return self.model_fn(x, sigma, **kwargs)


class NucleusImageSampler:
    """Sample latents using ComfyUI's sampler_object API.
    Supports ALL sampler/scheduler types available in ComfyUI (including custom installed ones).
    Model wrapper converts flow-matching velocity → x0 for k_diffusion."""

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
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
            },
        }

    RETURN_TYPES = ("NUCLEUS_LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Nucleus-Image"
    OUTPUT_NODE = True

    @torch.no_grad()
    def sample(self, model, positive, negative, width, height, steps, cfg, seed,
               sampler_name, scheduler_name):
        device = mm.get_torch_device()
        transformer = model["transformer"]
        config = model["config"]
        sched_cfg = model.get("scheduler_config", {})
        cfg_rescale = model.get("cfg_rescale", 1.0)

        transformer.to(device)

        # Generate sigmas via ComfyUI scheduler
        ms = _make_model_sampling(sched_cfg)
        sigmas = comfy.samplers.calculate_sigmas(ms, scheduler_name, steps).to(device)

        # Prepare latents
        num_ch = config["in_channels"] // 4
        h_lat = PATCH_SIZE * (height // (VAE_SCALE_FACTOR * PATCH_SIZE))
        w_lat = PATCH_SIZE * (width // (VAE_SCALE_FACTOR * PATCH_SIZE))
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn((1, 1, num_ch, h_lat, w_lat),
                            generator=generator, device=device, dtype=torch.bfloat16)
        latents = _pack_latents(noise, 1, num_ch, h_lat, w_lat, PATCH_SIZE)
        img_shapes = [(1, h_lat // PATCH_SIZE, w_lat // PATCH_SIZE)]

        pos_emb = positive["prompt_embeds"].to(device)
        pos_mask = positive["prompt_mask"].to(device) if positive.get("prompt_mask") is not None else None
        neg_emb = negative["prompt_embeds"].to(device)
        neg_mask = negative["prompt_mask"].to(device) if negative.get("prompt_mask") is not None else None
        do_cfg = cfg > 1.0

        pbar = comfy.utils.ProgressBar(steps)

        # Model function: velocity → x0 for k_diffusion
        def model_fn(x, sigma, **kwargs):
            # k_diffusion may pass float32 — convert to bfloat16 for transformer
            x_bf = x.to(torch.bfloat16)
            sigma_bf = sigma.to(torch.bfloat16)
            ts = sigma_bf.reshape(-1)[:1] if sigma_bf.numel() > 1 else sigma_bf.reshape(1)

            v_pos = transformer(
                hidden_states=x_bf, timestep=ts,
                encoder_hidden_states=pos_emb, encoder_hidden_states_mask=pos_mask,
                img_shapes=img_shapes, return_dict=False,
            )[0]
            if do_cfg:
                v_neg = transformer(
                    hidden_states=x_bf, timestep=ts,
                    encoder_hidden_states=neg_emb, encoder_hidden_states_mask=neg_mask,
                    img_shapes=img_shapes, return_dict=False,
                )[0]
                v = v_neg + cfg * (v_pos - v_neg)
                if cfg_rescale != 1.0:
                    pos_norm = torch.norm(v_pos, dim=-1, keepdim=True)
                    comb_norm = torch.norm(v, dim=-1, keepdim=True)
                    v = v * (pos_norm / comb_norm) * cfg_rescale + v * (1.0 - cfg_rescale)
            else:
                v = v_pos

            # velocity → x0:  x0 = x + sigma * v
            x0 = x_bf + sigma_bf * v
            return x0.to(x.dtype)

        # Wrap model for ComfyUI's KSAMPLER interface
        noise_scaling = _FlowMatchNoiseScaling(sigmas)
        model_wrap = _ModelWrap(model_fn, noise_scaling)

        # Use ComfyUI's sampler_object — supports ALL installed samplers
        sampler = comfy.samplers.sampler_object(sampler_name)

        def callback(step, denoised, x, total_steps):
            pbar.update_absolute(step, total_steps)

        latents = sampler.sample(
            model_wrap, sigmas, extra_args={"seed": seed},
            callback=callback, noise=latents, disable_pbar=True,
        )
        latents = latents.to(torch.bfloat16)

        # Offload
        transformer.to(mm.unet_offload_device())
        gc.collect()
        mm.soft_empty_cache()

        return ({"latents": latents, "width": width, "height": height},)


def _make_model_sampling(sched_cfg):
    import comfy.model_sampling as ms
    class _Cfg:
        sampling_settings = {
            "shift": sched_cfg.get("shift", 1.0),
            "multiplier": sched_cfg.get("num_train_timesteps", 1000),
        }
    return ms.ModelSamplingDiscreteFlow(_Cfg())


# ══════════════════════════════════════════════════════════════════════════════
# Node: VAE Decode
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageVAEDecode:
    """Decode latents. Moves VAE to GPU, decodes, offloads back."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("NUCLEUS_VAE",),
                "samples": ("NUCLEUS_LATENT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Nucleus-Image"
    OUTPUT_NODE = True

    @torch.no_grad()
    def decode(self, vae, samples):
        device = mm.get_torch_device()
        offload = mm.unet_offload_device()
        vae_model = vae["model"]
        latents_packed = samples["latents"]
        height, width = samples["height"], samples["width"]

        vae_model.to(device)

        bs, _, ch = latents_packed.shape
        hp = PATCH_SIZE * (int(height) // (VAE_SCALE_FACTOR * PATCH_SIZE))
        wp = PATCH_SIZE * (int(width) // (VAE_SCALE_FACTOR * PATCH_SIZE))
        latents = latents_packed.view(
            bs, hp // PATCH_SIZE, wp // PATCH_SIZE,
            ch // (PATCH_SIZE * PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5).contiguous()
        latents = latents.reshape(bs, ch // (PATCH_SIZE * PATCH_SIZE), 1, hp, wp)

        z_dim = vae_model.config.z_dim
        latents_mean = torch.tensor(vae_model.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents)
        latents_std_inv = 1.0 / torch.tensor(vae_model.config.latents_std).view(1, z_dim, 1, 1, 1).to(latents)
        latents = latents / latents_std_inv + latents_mean

        decoded = vae_model.decode(latents.to(vae_model.dtype), return_dict=False)[0][:, :, 0]

        vae_model.to(offload)
        gc.collect()
        mm.soft_empty_cache()

        arr = (decoded.float() * 0.5 + 0.5).clamp(0, 1)
        if arr.dim() == 4:
            arr = arr.permute(0, 2, 3, 1)
        return (arr,)


# ══════════════════════════════════════════════════════════════════════════════
# Registrations
# ══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "NucleusImageTransformerLoader": NucleusImageTransformerLoader,
    "NucleusImageTextEncoderLoader": NucleusImageTextEncoderLoader,
    "NucleusImageVAELoader": NucleusImageVAELoader,
    "NucleusImageTextEncode": NucleusImageTextEncode,
    "NucleusImageTextEncodeDual": NucleusImageTextEncodeDual,
    "NucleusImageZeroConditioning": NucleusImageZeroConditioning,
    "NucleusImageSampler": NucleusImageSampler,
    "NucleusImageVAEDecode": NucleusImageVAEDecode,
    "NucleusImageBlockSwap": NucleusImageBlockSwap,
    "NucleusImageModelShift": NucleusImageModelShift,
    "NucleusImageCFGRescale": NucleusImageCFGRescale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NucleusImageTransformerLoader": "Nucleus-Image Transformer Loader",
    "NucleusImageTextEncoderLoader": "Nucleus-Image Text Encoder Loader",
    "NucleusImageVAELoader": "Nucleus-Image VAE Loader",
    "NucleusImageTextEncode": "Nucleus-Image Text Encode",
    "NucleusImageTextEncodeDual": "Nucleus-Image Text Encode (Dual)",
    "NucleusImageZeroConditioning": "Nucleus-Image Zero Conditioning",
    "NucleusImageSampler": "Nucleus-Image Sampler",
    "NucleusImageVAEDecode": "Nucleus-Image VAE Decode",
    "NucleusImageBlockSwap": "Nucleus-Image Block Swap",
    "NucleusImageModelShift": "Nucleus-Image Model Shift",
    "NucleusImageCFGRescale": "Nucleus-Image CFG Rescale",
}
