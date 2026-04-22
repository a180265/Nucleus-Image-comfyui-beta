# ComfyUI-Nucleus-Image

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
[Nucleus Transformer Loader]──NUCLEUS_MODEL──┐
                                               │
[Nucleus Text Encoder Loader]──NUCLEUS_TE──┐  │
                                             │  │
[Nucleus Text Encode(正向)]──COND──┐         │  │
[Nucleus Text Encode(负向)]──COND──┘         │  │
                     ▼                        ▼  ▼
[Nucleus Advanced Config]──ADV_ARGS──┐  ┌──[Nucleus Sampler]──NUCLEUS_LATENT──▶[Nucleus VAE Decode]──▶IMAGE
                                      │  │                                        ▲
                                      └──┘                                        │
[Nucleus VAE Loader]────────────NUCLEUS_VAE──────────────────────────────────────┘
```

共 8 个节点（Advanced Config 为可选）：

| 节点 | 说明 |
|------|------|
| **Transformer Loader** | 加载 Transformer（从 `diffusion_models/`） |
| **Text Encoder Loader** | 加载 Qwen3-VL 文本编码器（从 `text_encoders/`） |
| **VAE Loader** | 加载 VAE（从 `vae/`） |
| **Block Swap** | 配置块交换参数（可选） |
| **Text Encode** | 编码提示词（支持正/负向） |
| **Advanced Config** | 配置 CFG 归一化与时间步偏移（可选） |
| **Sampler** | 去噪采样（支持 ComfyUI 采样器/调度器） |
| **VAE Decode** | 解码 latent 为图片 |

## 推荐参数（24GB VRAM + 32GB RAM）

### 方案 A：FP8 快速模式（推荐）

| 节点 | 参数 | 值 |
|------|------|---|
| **Transformer Loader** | model_name | `nucleus_image_transformer_fp8.safetensors` |
| | precision | `bf16` |
| | quantization | `disabled`（自动检测 FP8） |
| | device | `GPU` |
| **Block Swap** | blocks_to_swap | `0`（全部放 GPU） |
| **Text Encoder Loader** | model_name | `nucleus_image_text_encoder_fp8.safetensors` |
| | precision | `bf16` |
| | quantization | `disabled`（自动检测 FP8） |
| | device | `CPU` |
| **Text Encode** | force_offload | `true` |
| **Sampler** | steps | `50` |
| | cfg | `4.0` |
| | sampler_name | `euler` |
| | scheduler_name | `normal` |
| | force_offload | `true` |
| **VAE Loader** | model_name | `nucleus_image_vae.safetensors` |
| | precision | `bf16` |
| **VAE Decode** | — | — |

**预计资源占用：**
- GPU 峰值：~22 GB（采样阶段）
- CPU 占用：~4 GB（text encoder 编码后已卸载）
- 总耗时：~80 秒（50 步）

### 方案 B：bf16 高质量模式

| 节点 | 参数 | 值 |
|------|------|---|
| **Transformer Loader** | model_name | `nucleus_image_transformer_bf16.safetensors` |
| | device | `CPU` |
| **Block Swap** | blocks_to_swap | `14`（15 块在 GPU，14 块在 CPU） |
| 其余同方案 A | | |

**预计资源占用：**
- GPU 峰值：~21 GB
- CPU 占用：~15 GB（14 块 expert bf16）
- 总耗时：~120 秒（50 步，因 CPU↔GPU 交换）

## 内存占用参考表

### FP8 模式（blocks_to_swap 可调）

| blocks_to_swap | GPU 峰值 | CPU 占用 | 速度 |
|---------------|---------|---------|------|
| 0 | ~22 GB | ~4 GB | 最快 |
| 10 | ~17 GB | ~9 GB | 快 |
| 20 | ~12 GB | ~14 GB | 中等 |
| 29 | ~7 GB | ~15 GB | 慢 |

### bf16 模式（blocks_to_swap 可调）

| blocks_to_swap | GPU 峰值 | CPU 占用 | 速度 |
|---------------|---------|---------|------|
| 0 | ~34 GB | ~0 | 最快（需 48GB VRAM） |
| 14 | ~21 GB | ~15 GB | 中等（24GB VRAM 推荐） |
| 20 | ~15 GB | ~21 GB | 慢 |
| 29 | ~7 GB | ~31 GB | 最慢（需 48GB RAM） |

> 每个 MoE 层 expert：~528 MB (FP8) / ~1.06 GB (bf16)。29 个 MoE 层（层 3-31）。

## 节点参数详解

### Nucleus-Image Transformer Loader

| 参数 | 类型 | 说明 |
|------|------|------|
| model_name | 下拉 | `models/diffusion_models/` 中的权重文件 |
| precision | `bf16`/`fp16`/`fp32` | 计算精度（bf16 推荐） |
| quantization | `disabled`/`fp8_e4m3fn` | 权重量化（`disabled` 自动检测） |
| device | `CPU`/`GPU` | 非专家参数存放设备 |
| block_swap_args | BLOCKSWAPARGS | 块交换配置（可选） |

### Nucleus-Image Text Encoder Loader

| 参数 | 类型 | 说明 |
|------|------|------|
| model_name | 下拉 | `models/text_encoders/` 中的权重文件 |
| precision | `bf16`/`fp16`/`fp32` | 计算精度 |
| quantization | `disabled`/`fp8_e4m3fn` | 权重量化（`disabled` 自动检测） |
| device | `CPU`/`GPU` | 初始加载设备 |

### Nucleus-Image VAE Loader

| 参数 | 类型 | 说明 |
|------|------|------|
| model_name | 下拉 | `models/vae/` 中的权重文件 |
| precision | `bf16`/`fp16`/`fp32` | 计算精度 |

### Nucleus-Image Block Swap

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| enabled | BOOLEAN | true | 启用块交换 |
| blocks_to_swap | INT | 0 | 交换到 CPU 的 MoE 块数量（0-29） |
| prefetch_blocks | INT | 0 | 异步预取块数 |

### Nucleus-Image Text Encode

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| text_encoder | NUCLEUS_TEXT_ENCODER | — | 来自 Text Encoder Loader |
| text | STRING | "" | 提示词（支持多行） |
| force_offload | BOOLEAN | true | 编码后卸载到 CPU 释放 VRAM |

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
| force_offload | BOOLEAN | true | 采样后卸载到 CPU 释放 VRAM |

### Nucleus-Image VAE Decode

| 参数 | 类型 | 说明 |
|------|------|------|
| vae | NUCLEUS_VAE | 来自 VAE Loader |
| samples | NUCLEUS_LATENT | 来自 Sampler |

## 推荐分辨率

| 分辨率 | 宽×高 | 适用场景 |
|--------|--------|---------|
| 1024×1024 | 方形 | 通用 |
| 1344×768 | 横屏 | 风景 |
| 768×1344 | 竖屏 | 人像 |
| 1280×720 | 16:9 | 宽屏 |

## 常见问题

**Q: 图片和提示词无关？**
A: 确保使用了负向提示词（Text Encode 负向连到 Sampler 的 negative 输入）。空字符串 `""` 也可以作为负向条件。

**Q: 图片偏暗/发灰？**
A: 确认 VAE Loader 使用的是正确的模型文件，不要用其他模型的 VAE。

**Q: 显存不足 OOM？**
A: 增大 `blocks_to_swap` 值，或从 bf16 切换到 FP8 模型。

**Q: 为什么不能用 ComfyUI 内置的 CLIPLoader / VAELoader？**
A: Nucleus-Image 使用 Qwen3-VL（非标准 CLIP）和 AutoencoderKLQwenImage（16 通道 latent），架构与标准 SD/SDXL 不同，必须使用专用加载器。
