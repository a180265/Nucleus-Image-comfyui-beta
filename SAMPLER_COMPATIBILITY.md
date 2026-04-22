# Nucleus-Image 采样器与调度器兼容性文档

## 数据流

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

共 8 个节点。**Advanced Config** 为可选节点，不连接时采样器使用模型默认参数。

---

## 高级模型操作（Advanced Config 节点）

`Nucleus-Image Advanced Config` 独立节点，输出 `NUCLEUS_ADVANCED_ARGS`，连接到 Sampler 的可选输入 `advanced_args`。

**不连接 = 使用模型默认值（官方管线行为）。**

### 参数

| 参数 | 类型 | 默认 | 范围 | 说明 |
|------|------|------|------|------|
| cfg_rescale | FLOAT | 1.0 | 0.0~1.0 | CFG 自适应归一化强度 |
| shift | FLOAT | 0.0 | 0.0~10.0 | 时间步偏移覆盖，0 = 自动 |

### 1. CFG Adaptive Normalization（cfg_rescale）

控制 CFG 后的自适应归一化强度：

```python
# 标准 CFG
v_comb = v_neg + cfg * (v_pos - v_neg)

# 自适应归一化
v_rescaled = v_comb * ||v_pos|| / ||v_comb||
v_final = rescale * v_rescaled + (1 - rescale) * v_comb
```

**原理：** 高 CFG 时，CFG 组合向量的模远大于条件向量模（cfg=4 → 5倍，cfg=10 → 13.5倍）。归一化将模拉回与条件预测一致，防止去噪轨迹发散。`cfg_rescale` 控制归一化的混合比例。

| cfg_rescale | 效果 | 适用场景 |
|-------------|------|---------|
| 1.0（不连接=默认） | 官方管线行为，完整归一化 | 通用推荐 |
| 0.5 | 部分归一化 | 高 CFG 时减少过饱和 |
| 0.0 | 纯 CFG，无归一化 | 实验性，可能过曝 |

**安全范围：**

| cfg_rescale | 配合 cfg | 安全性 | 说明 |
|-------------|---------|--------|------|
| 0.8~1.0 | 任意 | 安全 | 推荐 |
| 0.3~0.8 | ≤5.0 | 较安全 | 可能轻微过饱和 |
| 0.0~0.3 | ≤2.0 | 需谨慎 | 高 CFG 时输出幅度失控 |
| 0.0 | ≤1.5 | 仅低 CFG | cfg>3.0 时去噪轨迹发散 → 图像崩塌 |

### 2. Timestep Shift（shift）

覆盖自动计算的指数时间偏移值 `mu`。与 Flux 的 shift 概念一致。

**不连接（shift=0）= 自动根据分辨率计算：**

```
mu = base_shift + (max_shift - base_shift) * (seq_len - base_seq_len) / (max_seq_len - base_seq_len)
```

默认参数来自 `configs/scheduler_config.json`（`base_shift=0.5`, `max_shift=1.15`）：

| 分辨率 | seq_len | mu（自动） | sigma 范围 |
|--------|---------|-----------|------------|
| 512×512 | 1024 | 0.63 | [1.000..0.037] |
| 768×768 | 2304 | 0.85 | [1.000..0.045] |
| 1024×1024 | 4096 | 1.15 | [1.000..0.061] |
| 1280×720 | 3600 | 1.07 | [1.000..0.056] |
| 1344×768 | 4032 | 1.14 | [1.000..0.060] |
| 1536×1024 | 6144 | 1.50 | [1.000..0.084] |

**手动覆盖效果与安全范围：**

| shift 值 | sigma 范围 | 效果 | 安全性 |
|----------|------------|------|--------|
| 0.0（不连接） | 自动 | 官方行为 | 安全 |
| 0.3~0.7 | [1.0..0.03~0.04] | 接近低分辨率自动值 | 安全 |
| 0.7~1.5 | [1.0..0.04~0.08] | 正常范围 | 安全 |
| 1.5~2.0 | [1.0..0.08~0.13] | 去噪范围略窄 | 较安全，细节可能损失 |
| 2.0~3.0 | [1.0..0.13~0.29] | 去噪范围窄 | 有风险，可能模糊 |
| 3.0~5.0 | [1.0..0.29~0.75] | 去噪范围极窄 | 高风险，噪声残留严重 |
| >5.0 | [1.0..>0.75] | 几乎无去噪 | 崩塌，输出纯噪声 |

---

## 采样器兼容性

### 兼容性说明

Nucleus-Image 通过 `_NucleusDenoiser` 包装器将 velocity 预测转换为 x0（去噪预测），接入 ComfyUI 的 k-diffusion 采样器系统。兼容性取决于采样器是否依赖 ComfyUI 的内部模型结构（`model.inner_model.model_patcher`、CFG++ 基础设施等）。

### 状态定义

| 状态 | 含义 |
|------|------|
| **兼容** | 直接可用，仅需 `model(x, sigma) -> x0` |
| **部分兼容** | 运行可用，但使用非最优路径（缺少 RF 专用优化） |
| **不兼容** | 运行时崩溃，需要 ComfyUI 完整模型包装链 |

### 兼容性总表

| 采样器 | 状态 | 阶数 | 类型 | 每步模型调用 | 说明 |
|--------|------|------|------|-------------|------|
| **euler** | 兼容 | 1阶 | 确定性 | 1 | 推荐，速度最快 |
| **heun** | 兼容 | 2阶 | 确定性 | 2 | 更细腻，步数可减半 |
| **heunpp2** | 兼容 | 2-3阶 | 确定性 | 2-3 | Heun 改进版 |
| **dpm_2** | 兼容 | 2阶 | 确定性 | 2 | DPM-Solver 中点 |
| **dpmpp_2m** | 兼容 | 2阶 | 确定性多步 | 1 | 推荐，质量/速度平衡 |
| **lms** | 兼容 | 1-4阶 | 确定性多步 | 1 | 线性多步法 |
| **ddpm** | 兼容 | 1阶 | 随机 | 1 | 标准 DDPM 步 |
| **ipndm** | 兼容 | 1-4阶 | 确定性多步 | 1 | 改进 PNDM |
| **ipndm_v** | 兼容 | 1-4阶 | 确定性多步 | 1 | PNDM 变步长 |
| **deis** | 兼容 | 1-3阶 | 确定性多步 | 1 | 预计算系数 |
| **res_multistep** | 兼容 | 2阶 | 确定性多步 | 1 | Residual 多步 |
| **res_multistep_ancestral** | 兼容 | 2阶 | 随机多步 | 1 | Residual 随机 |
| **gradient_estimation** | 兼容 | 1阶 | 确定性 | 1 | 梯度估计 |
| **dpm_fast** | 兼容 | 1-3阶 | 确定性自适应 | 1-3 | 自适应步长 |
| **dpm_adaptive** | 兼容 | 2-3阶 | 确定性自适应 | 2-3 | 自适应精度 |
| **ddim** | 兼容 | 1阶 | 确定性 | 1 | 等同 Euler |
| **uni_pc** | 兼容 | 1-3阶 | 确定性多步 | 1 | UniPC (bh1) |
| **uni_pc_bh2** | 兼容 | 1-3阶 | 确定性多步 | 1 | UniPC (bh2) |
| **euler_ancestral** | 部分兼容 | 1阶 | 随机 | 1 | 使用标准路径（非 RF 优化） |
| **dpm_2_ancestral** | 部分兼容 | 2阶 | 随机 | 2 | 使用标准路径 |
| **dpmpp_2s_ancestral** | 部分兼容 | 2阶 | 随机 | 2 | 使用标准路径 |
| euler_cfg_pp | 不兼容 | 1阶 | 确定性 | 1 | 需要 CFG++ 基础设施 |
| euler_ancestral_cfg_pp | 不兼容 | 1阶 | 随机 | 1 | 需要 CFG++ 基础设施 |
| dpmpp_2s_ancestral_cfg_pp | 不兼容 | 2阶 | 随机 | 2 | 需要 CFG++ 基础设施 |
| dpmpp_2m_cfg_pp | 不兼容 | 2阶 | 确定性多步 | 1 | 需要 CFG++ 基础设施 |
| res_multistep_cfg_pp | 不兼容 | 2阶 | 确定性多步 | 1 | 需要 CFG++ 基础设施 |
| res_multistep_ancestral_cfg_pp | 不兼容 | 2阶 | 随机多步 | 1 | 需要 CFG++ 基础设施 |
| gradient_estimation_cfg_pp | 不兼容 | 1阶 | 确定性 | 1 | 需要 CFG++ 基础设施 |
| dpmpp_sde | 不兼容 | 2阶 | 随机 | 2 | 需要 model_sampling（half-logSNR） |
| dpmpp_sde_gpu | 不兼容 | 2阶 | 随机 | 2 | 同 dpmpp_sde |
| dpmpp_2m_sde | 不兼容 | 2阶 | 随机多步 | 1 | 需要 model_sampling |
| dpmpp_2m_sde_gpu | 不兼容 | 2阶 | 随机多步 | 1 | 同 dpmpp_2m_sde |
| dpmpp_2m_sde_heun | 不兼容 | 2阶 | 随机多步 | 1 | 同 dpmpp_2m_sde |
| dpmpp_2m_sde_heun_gpu | 不兼容 | 2阶 | 随机多步 | 1 | 同 dpmpp_2m_sde |
| dpmpp_3m_sde | 不兼容 | 3阶 | 随机多步 | 1 | 需要 model_sampling |
| dpmpp_3m_sde_gpu | 不兼容 | 3阶 | 随机多步 | 1 | 同 dpmpp_3m_sde |
| lcm | 不兼容 | 1阶 | 确定性 | 1 | 需要 model_sampling.noise_scaling |
| er_sde | 不兼容 | 1-3阶 | 随机多阶段 | 1 | 需要 model_sampling |
| seeds_2 | 不兼容 | 2阶 | 随机 | 2 | 需要 model_sampling |
| seeds_3 | 不兼容 | 3阶 | 随机 | 3 | 需要 model_sampling |
| sa_solver | 不兼容 | 1-3阶 | 随机 | 1-2 | 需要 model_sampling |
| sa_solver_pece | 不兼容 | 1-3阶 | 随机 | 2 | 同 sa_solver |
| exp_heun_2_x0 | 不兼容 | 2阶 | 确定性 | 2 | 委托给 seeds_2 |
| exp_heun_2_x0_sde | 不兼容 | 2阶 | 随机 | 2 | 委托给 seeds_2 |

### 不兼容原因分类

| 原因 | 涉及采样器 | 说明 |
|------|-----------|------|
| **CFG++ 基础设施** | 所有 `_cfg_pp` 后缀 | 需要 ComfyUI 的 `set_model_options_post_cfg_function` 来捕获无条件去噪结果 |
| **half-logSNR** | dpmpp_sde/2m_sde/3m_sde, er_sde, seeds, sa_solver, exp_heun | 需要 `model_sampling` 对象进行 sigma→lambda 时间参数化转换 |
| **noise_scaling** | lcm | 直接调用 `model_sampling.noise_scaling()` 进行噪声缩放 |

### 统计

- **兼容**: 18 个（含 ddim、uni_pc）
- **部分兼容**: 3 个（euler_ancestral、dpm_2_ancestral、dpmpp_2s_ancestral）
- **不兼容**: 23 个

---

## 调度器兼容性

所有 9 个 ComfyUI 调度器均兼容。它们在 Nucleus-Image 指数偏移确定的 `[sigma_max, sigma_min]` 范围内重新排列 sigma 间距：

| 调度器 | 策略 | 特点 |
|--------|------|------|
| **normal** | Nucleus-Image 偏移后的线性间距 | 官方推荐，已验证 |
| **karras** | Karras 公式（rho=7.0） | 末端加密，细节更好 |
| **exponential** | 几何间距 | 高噪声区更密 |
| **sgm_uniform** | SGM 风格均匀间距 | 略微偏移起始点 |
| **simple** | 线性间距（sigma 空间） | 最简单的间距 |
| **ddim_uniform** | 线性间距 | 同 simple |
| **beta** | Beta 分布间距 | 两端集中，实验性 |
| **linear_quadratic** | 先线性后二次 | 前半线性后半加速，来自 Mochi |
| **kl_optimal** | 反正切间距 | KL 最优，理论最优 |

---

## 推荐组合

### 通用推荐（质量优先）

| 参数 | 值 |
|------|---|
| sampler_name | `dpmpp_2m` |
| scheduler_name | `normal` |
| steps | 28-30 |
| cfg | 4.0 |
| Advanced Config | 不连接（默认） |

### 快速生成（速度优先）

| 参数 | 值 |
|------|---|
| sampler_name | `euler` |
| scheduler_name | `normal` |
| steps | 20-25 |
| cfg | 4.0 |
| Advanced Config | 不连接（默认） |

### 高质量（步数充裕）

| 参数 | 值 |
|------|---|
| sampler_name | `heun` |
| scheduler_name | `karras` |
| steps | 30-40 |
| cfg | 4.0 |
| Advanced Config | cfg_rescale=0.7~1.0 |

### 细节增强

| 参数 | 值 |
|------|---|
| sampler_name | `dpmpp_2m` |
| scheduler_name | `karras` |
| steps | 30 |
| cfg | 4.0 |
| Advanced Config | 不连接（默认） |

### 实验性（随机采样）

| 参数 | 值 | 说明 |
|------|---|------|
| sampler_name | `euler_ancestral` | 部分兼容，使用标准路径 |
| scheduler_name | `normal` | |
| steps | 30-50 | 随机采样通常需要更多步数 |
| cfg | 4.0 | |
| Advanced Config | cfg_rescale=0.7~1.0 | 可尝试降低减少过饱和 |

---

## 错误处理

当选择不兼容的采样器时，采样器会直接报错（**不自动回退**），错误信息类似：

```
AttributeError: 'SimpleNamespace' object has no attribute 'model_patcher'
```

这表示该采样器需要 ComfyUI 的完整模型包装基础设施，当前不支持。选择表中标记为"兼容"或"部分兼容"的采样器即可。

如果 ComfyUI 未来新增采样器/调度器，它们会自动出现在下拉列表中。采样器会尝试直接调用，兼容则正常运行，不兼容则报错——方便你测试新出现的选项。
