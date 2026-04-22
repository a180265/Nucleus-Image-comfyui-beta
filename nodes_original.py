import os
import json
import gc
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import folder_paths
import comfy.utils


folder_paths.add_model_folder_path(
    "nucleus_image",
    os.path.join(folder_paths.models_dir, "nucleus_image"),
)

FP8_MODEL_DIR = r"D:\AI\Nucleus-Image-FP8"

SYSTEM_PROMPT = (
    "You are an image generation assistant. Follow the user's prompt literally. "
    "Pay careful attention to spatial layout: objects described as on the left must "
    "appear on the left, on the right on the right. Match exact object counts and "
    "assign colors to the correct objects."
)

VAE_SCALE_FACTOR = 8   # 2 ** len(vae.temperal_downsample)
PATCH_SIZE = 2
NUM_TRAIN_TIMESTEPS = 1000


# ══════════════════════════════════════════════════════════════════════════════
# FP8 Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _dequant_fp8(tensor, scale):
    if tensor.dtype == torch.float8_e4m3fn and scale is not None:
        return (tensor.float() * scale).to(torch.bfloat16)
    if tensor.is_floating_point():
        return tensor.to(torch.bfloat16)
    return tensor


def _load_fp8_into_model(model, fp8_path, device=None):
    """Load FP8 safetensors into a model, dequantizing tensor-by-tensor."""
    sd = load_file(fp8_path)
    metadata = sd.pop("__metadata__", None)
    scales = {}
    if metadata and "quantization" in metadata:
        scales = json.loads(metadata["quantization"]).get("scales", {})

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
# Block-Swap Expert Forward Patching
# ══════════════════════════════════════════════════════════════════════════════

def _patch_expert_forward(experts_module, block_idx, fp8_data, fp8_scales, device):
    use_gmm = experts_module.use_grouped_mm
    gu_fp8 = fp8_data["gate_up_proj"]
    dp_fp8 = fp8_data["down_proj"]
    gu_scale = torch.tensor(fp8_scales.get("gate_up_proj", 1.0), dtype=torch.float32)
    dp_scale = torch.tensor(fp8_scales.get("down_proj", 1.0), dtype=torch.float32)

    _cache = {"gu": None, "dp": None}

    def _load():
        if _cache["gu"] is not None:
            return
        gu_gpu = gu_fp8.to(device)
        dp_gpu = dp_fp8.to(device)
        _cache["gu"] = (gu_gpu.float() * gu_scale.to(device)).to(torch.bfloat16)
        _cache["dp"] = (dp_gpu.float() * dp_scale.to(device)).to(torch.bfloat16)
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
# Node 1: Model Loader
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageModelLoader:
    """Load Nucleus-Image transformer with FP8 block-swap (keeps weights on CPU)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": FP8_MODEL_DIR}),
            },
        }

    RETURN_TYPES = ("NUCLEUS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Nucleus-Image"

    def load_model(self, model_path):
        from diffusers.models import NucleusMoEImageTransformer2DModel
        from diffusers.models.transformers.transformer_nucleusmoe_image import SwiGLUExperts

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        trans_dir = os.path.join(model_path, "transformer")

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
        config = NucleusMoEImageTransformer2DModel.load_config(trans_dir)
        transformer = NucleusMoEImageTransformer2DModel.from_config(config)
        SwiGLUExperts.__init__ = _orig_init

        # Load FP8 state dict, separate expert vs non-expert
        fp8_sd = load_file(os.path.join(trans_dir, "diffusion_pytorch_model.safetensors"))
        metadata = fp8_sd.pop("__metadata__", None)
        scales = {}
        if metadata and "quantization" in metadata:
            scales = json.loads(metadata["quantization"]).get("scales", {})

        expert_fp8 = {}
        expert_scales = {}
        for name in list(fp8_sd.keys()):
            tensor = fp8_sd.pop(name)
            if _is_expert_param(name):
                parts = name.split(".")
                block_idx = int(parts[1])
                ptype = "gate_up_proj" if "gate_up_proj" in name else "down_proj"
                expert_fp8.setdefault(block_idx, {})[ptype] = tensor
                if name in scales:
                    expert_scales.setdefault(block_idx, {})[ptype] = scales[name]
            else:
                bf16 = _dequant_fp8(tensor, scales.get(name))
                _set_nested_attr(transformer, name, nn.Parameter(bf16))
            del tensor
        del fp8_sd, scales
        gc.collect()

        # Patch MoE block expert forwards (weights stay on CPU, loaded per-block to GPU)
        for block_idx, block in enumerate(transformer.transformer_blocks):
            if block.moe_enabled and block_idx in expert_fp8:
                _patch_expert_forward(
                    block.img_mlp.experts, block_idx,
                    expert_fp8[block_idx], expert_scales.get(block_idx, {}),
                    "cuda",
                )

        return ({"transformer": transformer, "config": config, "model_path": model_path},)


# ══════════════════════════════════════════════════════════════════════════════
# Node 2: Text Encode
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageTextEncode:
    """Encode text prompt using Qwen3-VL (loads FP8 to GPU, encodes, frees)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": FP8_MODEL_DIR}),
                "text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("NUCLEUS_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Nucleus-Image"

    def encode(self, model_path, text):
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, AutoConfig

        device = "cuda"
        proc_dir = os.path.join(model_path, "processor")
        processor = Qwen3VLProcessor.from_pretrained(proc_dir)

        te_dir = os.path.join(model_path, "text_encoder")
        te_config = AutoConfig.from_pretrained(te_dir)
        with torch.device(device):
            text_encoder = Qwen3VLForConditionalGeneration._from_config(
                te_config, dtype=torch.bfloat16
            )
        _load_fp8_into_model(text_encoder, os.path.join(te_dir, "model.safetensors"), device=device)
        text_encoder.eval()

        formatted = _format_prompt(processor, text)
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

        return ({"prompt_embeds": prompt_embeds, "prompt_mask": prompt_mask},)


# ══════════════════════════════════════════════════════════════════════════════
# Node 3: Sampler
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageSampler:
    """Sample latents with CFG and block-swap, showing progress in ComfyUI UI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("NUCLEUS_MODEL",),
                "positive": ("NUCLEUS_CONDITIONING",),
                "negative": ("NUCLEUS_CONDITIONING",),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("NUCLEUS_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "Nucleus-Image"
    OUTPUT_NODE = True

    @torch.no_grad()
    def sample(self, model, positive, negative, width, height, steps, cfg, seed):
        from diffusers import FlowMatchEulerDiscreteScheduler

        device = "cuda"
        transformer = model["transformer"]
        config = model["config"]

        # Move transformer to GPU
        transformer.to(device)

        # Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(model["model_path"], "scheduler")
        )

        # Prepare latents
        num_ch = config["in_channels"] // 4  # 64 // 4 = 16
        h_lat = PATCH_SIZE * (height // (VAE_SCALE_FACTOR * PATCH_SIZE))
        w_lat = PATCH_SIZE * (width // (VAE_SCALE_FACTOR * PATCH_SIZE))

        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn((1, 1, num_ch, h_lat, w_lat),
                            generator=generator, device=device, dtype=torch.bfloat16)
        latents = _pack_latents(noise, 1, num_ch, h_lat, w_lat, PATCH_SIZE)

        img_shapes = [(1, h_lat // PATCH_SIZE, w_lat // PATCH_SIZE)]

        # Compute timesteps
        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        image_seq_len = latents.shape[1]
        mu = _calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        scheduler.set_begin_index(0)
        scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device, mu=mu)
        timesteps = scheduler.timesteps

        pos_embeds = positive["prompt_embeds"]
        pos_mask = positive["prompt_mask"]
        neg_embeds = negative["prompt_embeds"]
        neg_mask = negative["prompt_mask"]
        do_cfg = cfg > 1.0

        pbar = comfy.utils.ProgressBar(steps)

        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # Positive prediction
            noise_pred = transformer(
                hidden_states=latents,
                timestep=timestep / NUM_TRAIN_TIMESTEPS,
                encoder_hidden_states=pos_embeds,
                encoder_hidden_states_mask=pos_mask,
                img_shapes=img_shapes,
                return_dict=False,
            )[0]

            # CFG: negative prediction + combine
            if do_cfg:
                neg_noise_pred = transformer(
                    hidden_states=latents,
                    timestep=timestep / NUM_TRAIN_TIMESTEPS,
                    encoder_hidden_states=neg_embeds,
                    encoder_hidden_states_mask=neg_mask,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]
                comb_pred = neg_noise_pred + cfg * (noise_pred - neg_noise_pred)
                # Adaptive normalization (matches original pipeline)
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

            noise_pred = -noise_pred
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            pbar.update_absolute(i + 1, steps)

        # Move transformer back to CPU to free GPU
        transformer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        return ({"latents": latents, "width": width, "height": height},)


# ══════════════════════════════════════════════════════════════════════════════
# Node 4: VAE Decode
# ══════════════════════════════════════════════════════════════════════════════

class NucleusImageVAEDecode:
    """Decode latents to images using the Nucleus-Image VAE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": FP8_MODEL_DIR}),
                "samples": ("NUCLEUS_LATENT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "Nucleus-Image"
    OUTPUT_NODE = True

    @torch.no_grad()
    def decode(self, model_path, samples):
        from diffusers import AutoencoderKLQwenImage

        device = "cuda"
        latents_packed = samples["latents"]
        height = samples["height"]
        width = samples["width"]

        vae_dir = os.path.join(model_path, "vae")
        vae_config = AutoencoderKLQwenImage.load_config(vae_dir)
        with torch.device(device):
            vae = AutoencoderKLQwenImage.from_config(vae_config)
        _load_fp8_into_model(vae, os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"), device=device)
        vae.eval()

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

        # Denormalize
        z_dim = vae.config.z_dim
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents)
        latents_std_inv = (1.0 / torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1)).to(latents)
        latents = latents * latents_std_inv + latents_mean

        decoded = vae.decode(latents.to(vae.dtype), return_dict=False)[0][:, :, 0]

        del vae
        gc.collect()
        torch.cuda.empty_cache()

        # Convert to ComfyUI IMAGE: (B, H, W, 3) float32 [0,1]
        arr = decoded.float().clamp(0, 1)
        if arr.dim() == 4:
            arr = arr.permute(0, 2, 3, 1)
        return (arr,)


# ══════════════════════════════════════════════════════════════════════════════
# Node Registrations
# ══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "NucleusImageModelLoader": NucleusImageModelLoader,
    "NucleusImageTextEncode": NucleusImageTextEncode,
    "NucleusImageSampler": NucleusImageSampler,
    "NucleusImageVAEDecode": NucleusImageVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NucleusImageModelLoader": "Nucleus-Image Model Loader",
    "NucleusImageTextEncode": "Nucleus-Image Text Encode",
    "NucleusImageSampler": "Nucleus-Image Sampler",
    "NucleusImageVAEDecode": "Nucleus-Image VAE Decode",
}
