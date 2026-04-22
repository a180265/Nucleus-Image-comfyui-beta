# ComfyUI-Nucleus-Image

> **⚠️ 测试版本 / Beta Version**
>
> 本节点为测试版本。目前仅验证了 FP8 模型的正常推理流程；FP16 因硬件限制未实际运行测试。参数无明确限制，采样器与调度器经个人测试 euler + normal 效果较好。块交换（Block Swap）、CFG 缩放（CFG Rescale）、模型偏移（Model Shift）等功能未进行实际运行测试，但代码已通过检查。
>
> This is a beta release. Only FP8 model inference has been tested; FP16 has not been run due to hardware limitations. There are no explicit parameter restrictions. Personally tested: euler + normal sampler/scheduler gives good results. Block Swap, CFG Rescale, and Model Shift features have not been runtime-tested but have passed code review.

Nucleus-Image 17B MoE 扩散模型的 ComfyUI 自定义节点。

## 环境要求

### 硬件

| 配置等级 | VRAM | RAM | 推荐精度 | blocks_to_swap |
|---------|------|-----|---------|---------------|
| 最低 | 8 GB | 16 GB | FP8 | 29（全交换） |
| 推荐 | 24 GB | 32 GB | FP8 | 0（全 GPU） |
| 高端 | 24 GB | 48 GB+ | bf16 | 12-14 |
| 最佳 | 48 GB+ | 64 GB+ | bf16 | 0（全 GPU） |

### 软件依赖

> 详细依赖说明见 [REQUIREMENTS.md](REQUIREMENTS.md)

| 依赖 | 最低版本 | 测试版本 | 说明 |
|------|---------|---------|------|
| Python | 3.11+ | 3.13.11 | |
| PyTorch | 2.11+ | 2.11.0+cu130 | 需要 `F.grouped_mm` 支持 |
| CUDA | 12.0+ | 13.0 | RTX 40/50 系列 |
| cuDNN | 9.0+ | 9.19.0 | |
| ComfyUI | 0.19+ | — | |
| diffusers | 0.38+ | 0.38.0.dev0 | NucleusMoE 管线支持 |
| transformers | 4.57+ | 4.57.6 | Qwen3-VL 支持 |
| safetensors | 0.5+ | 0.8.0-rc.0 | |
| accelerate | 1.0+ | 1.12.0 | 设备管理与 offload |
| numpy | 1.26+ | 2.3.5 | |
| Pillow | 10.0+ | 12.1.0 | |
| scipy | 1.13+ | 1.17.0 | 调度器数值计算 |

验证环境：
```bash
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"
python -c "import torch; print('grouped_mm:', hasattr(torch.nn.functional, 'grouped_mm'))"
python -c "import diffusers; print('diffusers:', diffusers.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
```

## 模型文件安装

### 文件放置

将模型文件放入 ComfyUI 标准目录（**单文件格式，无需分片**）：

```
ComfyUI/models/
├── diffusion_models/
│   ├── nucleus_image_transformer_fp8.safetensors    (~16.9 GB)
│   └── nucleus_image_transformer_bf16.safetensors   (~31.7 GB, 可选)
├── text_encoders/
│   ├── nucleus_image_text_encoder_fp8.safetensors   (~8.2 GB)
│   └── nucleus_image_text_encoder_bf16.safetensors  (~16.5 GB, 可选)
└── vae/
    ├── nucleus_image_vae.safetensors                 (~122 MB)
    └── nucleus_image_vae_bf16.safetensors            (~243 MB, 可选)
```

FP8 为推荐格式（速度快、资源占用低）。bf16 为可选高质量格式。

### 从原始模型准备

FP8 模型直接复制单文件即可。bf16 模型如果原始为分片格式，需先合并：

```bash
# Transformer FP8（直接复制）
copy "D:\AI\Nucleus-Image-FP8\transformer\diffusion_pytorch_model.safetensors" ^
  "ComfyUI\models\diffusion_models\nucleus_image_transformer_fp8.safetensors"

# Text Encoder FP8（直接复制）
copy "D:\AI\Nucleus-Image-FP8\text_encoder\model.safetensors" ^
  "ComfyUI\models\text_encoders\nucleus_image_text_encoder_fp8.safetensors"

# VAE（直接复制）
copy "D:\AI\Nucleus-Image-FP8\vae\diffusion_pytorch_model.safetensors" ^
  "ComfyUI\models\vae\nucleus_image_vae.safetensors"

# BF16 分片合并（需要足够 RAM，不足会使用虚拟内存）
python merge_shards.py --input "D:\AI\Nucleus-Image\transformer" ^
  --output "ComfyUI\models\diffusion_models\nucleus_image_transformer_bf16.safetensors"
python merge_shards.py --input "D:\AI\Nucleus-Image\text_encoder" ^
  --output "ComfyUI\models\text_encoders\nucleus_image_text_encoder_bf16.safetensors"
```

## 节点列表

### 数据流

```
[Nucleus Block Swap]──────BLOCKSWAPARGS──────┐
                                              ▼
[Nucleus Transformer Loader]──NUCLEUS_MODEL──┬──[Nucleus Model Shift]──NUCLEUS_MODEL
                                               └──[Nucleus CFG Rescale]──NUCLEUS_MODEL
                                                        │
[Nucleus Text Encoder Loader]──NUCLEUS_TE──┐           │
                                            │           │
[Nucleus Text Encode]──────────COND──┐      │           │
[Nucleus Text Encode (Dual)]──COND×2─┤      │           │
[Nucleus Zero Conditioning]───COND───┘      │           │
                     ▼                       ▼           ▼
                  [Nucleus Sampler]──NUCLEUS_LATENT──▶[Nucleus VAE Decode]──▶IMAGE
                                                               ▲
                  [Nucleus VAE Loader]────NUCLEUS_VAE─────────┘
```

共 11 个节点：

| 节点 | 说明 |
|------|------|
| **Transformer Loader** | 加载 Transformer（从 `diffusion_models/`） |
| **Text Encoder Loader** | 加载 Qwen3-VL 文本编码器（从 `text_encoders/`），延迟加载 |
| **VAE Loader** | 加载 VAE（从 `vae/`） |
| **Block Swap** | 配置块交换参数（可选） |
| **Text Encode** | 编码提示词（正向或负向） |
| **Text Encode (Dual)** | 同时编码正向+负向提示词，只加载一次 TE（更快） |
| **Zero Conditioning** | 零条件嵌入，无需加载 TE（即时生成） |
| **Model Shift** | 覆盖 sigma shift 参数（可选） |
| **CFG Rescale** | CFG 自适应归一化强度（可选） |
| **Sampler** | 去噪采样（支持 ComfyUI 全部采样器/调度器） |
| **VAE Decode** | 解码 latent 为图片 |

## 推荐参数（24GB VRAM）

### 方案 A：FP8 快速模式（推荐）

| 节点 | 参数 | 值 |
|------|------|---|
| **Transformer Loader** | model_name | `nucleus_image_transformer_fp8.safetensors` |
| | precision | `bf16` |
| | load_device | `offload_device` |
| **Block Swap** | blocks_to_swap | `0`（全部放 GPU） |
| **Text Encoder Loader** | model_name | `nucleus_image_text_encoder_fp8.safetensors` |
| | precision | `bf16` |
| **Text Encode (Dual)** | positive_text | 正向提示词 |
| | negative_text | 负向提示词 |
| **Sampler** | steps | `50` |
| | cfg | `4.0` |
| | sampler_name | `euler` |
| | scheduler_name | `normal` |
| **VAE Loader** | model_name | `nucleus_image_vae.safetensors` |
| | precision | `bf16` |
| **VAE Decode** | — | — |

### 方案 B：bf16 高质量模式

| 节点 | 参数 | 值 |
|------|------|---|
| **Transformer Loader** | model_name | `nucleus_image_transformer_bf16.safetensors` |
| | load_device | `offload_device` |
| **Block Swap** | blocks_to_swap | `14`（15 块在 GPU，14 块在 CPU） |
| 其余同方案 A | | |

## 节点参数详解

### Nucleus-Image Transformer Loader

| 参数 | 类型 | 说明 |
|------|------|------|
| model_name | 下拉 | `models/diffusion_models/` 中的权重文件 |
| precision | `bf16` / `fp16` / `fp32` | 计算精度（bf16 推荐） |
| load_device | `offload_device` / `main_device` | 非专家参数初始存放设备 |
| block_swap_args | BLOCKSWAPARGS | 块交换配置（可选） |

> FP8 模型文件会被自动检测并处理（expert FP8 权重按层反量化计算，无需手动指定 quantization）。

### Nucleus-Image Text Encoder Loader

| 参数 | 类型 | 说明 |
|------|------|------|
| model_name | 下拉 | `models/text_encoders/` 中的权重文件 |
| precision | `bf16` / `fp16` / `fp32` | 计算精度 |

> 延迟加载：此节点仅记录路径，实际加载在 Text Encode 节点执行时发生（加载到 GPU → 编码 → 自动释放）。

### Nucleus-Image VAE Loader

| 参数 | 类型 | 说明 |
|------|------|------|
| model_name | 下拉 | `models/vae/` 中的权重文件 |
| precision | `bf16` / `fp16` / `fp32` | 计算精度 |

### Nucleus-Image Block Swap

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| blocks_to_swap | INT | 0 | 交换到 CPU 的 MoE 块数量（0-29），0 = 全部放 GPU |

### Nucleus-Image Text Encode

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| text_encoder | NUCLEUS_TE | — | 来自 Text Encoder Loader |
| text | STRING | "" | 提示词（支持多行） |

> 编码完成后 TE 自动从 GPU 释放。

### Nucleus-Image Text Encode (Dual)

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| text_encoder | NUCLEUS_TE | — | 来自 Text Encoder Loader |
| positive_text | STRING | "" | 正向提示词 |
| negative_text | STRING | "" | 负向提示词 |

> 只加载一次 TE 模型同时编码正向和负向，避免重复加载，输出两个 `NUCLEUS_CONDITIONING`。

### Nucleus-Image Zero Conditioning

无输入参数。生成零嵌入作为条件（即时，无需加载 TE）。适用于无条件生成或作为负向条件的替代。

### Nucleus-Image Model Shift

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| model | NUCLEUS_MODEL | — | 来自 Transformer Loader |
| base_shift | FLOAT | 0.5 | 基础 shift 值 |
| max_shift | FLOAT | 1.15 | 最大 shift 值 |

> 如不连接此节点，使用调度器配置中的默认值。

### Nucleus-Image CFG Rescale

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| model | NUCLEUS_MODEL | — | 来自 Transformer Loader（或 Model Shift） |
| cfg_rescale | FLOAT | 1.0 | CFG 自适应归一化强度（0.0-1.0），1.0 = 官方默认 |

### Nucleus-Image Sampler

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| model | NUCLEUS_MODEL | — | 来自 Transformer Loader |
| positive | NUCLEUS_CONDITIONING | — | 正向条件 |
| negative | NUCLEUS_CONDITIONING | — | 负向条件 |
| width | INT | 1024 | 图片宽度（步长 64） |
| height | INT | 1024 | 图片高度（步长 64） |
| steps | INT | 50 | 去噪步数 |
| cfg | FLOAT | 4.0 | 引导系数 |
| seed | INT | 0 | 随机种子 |
| sampler_name | 下拉 | euler | ComfyUI 内置采样器 |
| scheduler_name | 下拉 | normal | ComfyUI 内置调度器 |

> 采样完成后 Transformer 自动 offload 到 CPU。

### Nucleus-Image VAE Decode

| 参数 | 类型 | 说明 |
|------|------|------|
| vae | NUCLEUS_VAE | 来自 VAE Loader |
| samples | NUCLEUS_LATENT | 来自 Sampler |

> 解码完成后 VAE 自动 offload 到 CPU。

## 推荐分辨率

| 分辨率 | 宽×高 | 适用场景 |
|--------|--------|---------|
| 1024×1024 | 方形 | 通用 |
| 1344×768 | 横屏 | 风景 |
| 768×1344 | 竖屏 | 人像 |
| 1280×720 | 16:9 | 宽屏 |

## 常见问题

**Q: 图片和提示词无关？**
A: 确保使用了负向提示词（Text Encode 负向连到 Sampler 的 negative 输入）。空字符串 `""` 也可以作为负向条件。推荐使用 Text Encode (Dual) 节点同时编码正负向。

**Q: 图片偏暗/发灰？**
A: 确认 VAE Loader 使用的是正确的模型文件，不要用其他模型的 VAE。

**Q: 显存不足 OOM？**
A: 增大 `blocks_to_swap` 值，或从 bf16 切换到 FP8 模型。

**Q: 为什么不能用 ComfyUI 内置的 CLIPLoader / VAELoader？**
A: Nucleus-Image 使用 Qwen3-VL（非标准 CLIP）和 AutoencoderKLQwenImage（16 通道 latent），架构与标准 SD/SDXL 不同，必须使用专用加载器。

**Q: Text Encoder Loader 没有加载设备选项？**
A: Text Encoder 采用延迟加载策略，在 Text Encode 节点执行时自动加载到 GPU，编码完成后自动释放，无需手动管理设备。
