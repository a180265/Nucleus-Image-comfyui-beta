# Nucleus-Image ComfyUI 依赖说明

本节点基于 **秋葉 ComfyUI 整合包 (aki-v3)** 环境测试通过。

## 测试环境

| 项目 | 版本 |
|------|------|
| OS | Windows 11 |
| Python | 3.13.11 |
| GPU | NVIDIA RTX 5090 D v2 |
| CUDA Runtime | 13.0 |
| cuDNN | 9.19.0 |

## 核心依赖（必须满足最低版本）

| 包名 | 最低版本 | 测试版本 | 说明 |
|------|---------|---------|------|
| **torch** | 2.11.0+ | 2.11.0+cu130 | 需要 `F.grouped_mm` 支持（MoE 核心算子） |
| **diffusers** | 0.38.0 | 0.38.0.dev0 | NucleusMoE 管线支持 |
| **transformers** | 4.57.0 | 4.57.6 | Qwen3-VL 文本编码器 |
| **safetensors** | 0.5.0 | 0.8.0-rc.0 | 模型权重加载 |
| **accelerate** | 1.0.0 | 1.12.0 | 设备管理与 offload |
| **numpy** | 1.26.0 | 2.3.5 | 数值计算 |
| **Pillow** | 10.0.0 | 12.1.0 | 图像处理 |
| **scipy** | 1.13.0 | 1.17.0 | 调度器数值计算 |

## 辅助依赖（自动安装，列出供参考）

| 包名 | 版本 | 说明 |
|------|------|------|
| tokenizers | 0.22.2 | 分词器后端 |
| huggingface_hub | 0.36.0 | 模型下载 |
| sentencepiece | 0.2.1 | 分词 |
| regex | 2026.1.15 | 正则表达式 |
| tqdm | 4.67.1 | 进度条 |
| packaging | 25.0 | 版本解析 |
| filelock | 3.20.3 | 文件锁 |
| sympy | 1.14.0 | 符号计算（torch 依赖） |
| networkx | 3.6.1 | 图计算（torch 依赖） |

## PyTorch 关键特性要求

| 特性 | 要求 | 验证命令 |
|------|------|---------|
| `F.grouped_mm` | 必须 | `python -c "import torch; print(hasattr(torch.nn.functional, 'grouped_mm'))"` |
| CUDA BF16 | 必须 | `python -c "import torch; print(torch.cuda.is_bf16_supported())"` |
| SDPA | 推荐 | `python -c "import torch; print(hasattr(torch.nn.functional, 'scaled_dot_product_attention'))"` |

## 快速安装（非整合包用户）

```bash
# 安装 PyTorch（CUDA 13.0）
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# 安装核心依赖
pip install diffusers>=0.38.0 transformers>=4.57.0 safetensors>=0.5.0 accelerate>=1.0.0

# 如果 diffusers 0.38 尚未正式发布，安装 nightly
pip install git+https://github.com/huggingface/diffusers.git
```

## 整合包用户

秋葉 ComfyUI 整合包 (aki-v3) 已包含所有依赖，无需额外安装。直接将本节点目录放入 `ComfyUI/custom_nodes/` 即可。
