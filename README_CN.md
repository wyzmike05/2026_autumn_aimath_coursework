# CIFAR-10 IR-SDE 与传统去噪实验

## 1. 实验简介

本目录在现有 MNIST 实验旁新增一套 CIFAR-10 实验，用于比较条件生成式 IR-SDE 去噪与六种传统方法。CIFAR-10 代码独立实现；现有 MNIST 文件保持不变。

## 2. 新增文件

| 文件 | 用途 |
|---|---|
| `ir_sde/model_cifar.py` | 创建适用于 32×32 RGB 图像的 CIFAR-10 UNet。 |
| `ir_sde/data_cifar.py` | CIFAR-10 数据加载、在线噪声生成及确定性测试样本。 |
| `main_cifar.py` | IR-SDE 训练、评估、SSIM 计算和模型/指标保存。 |
| `ir_sde/pipeline_cifar.py` | RGB Pipeline 适配层，沿用共享 Scheduler/Pipeline 与采样接口。 |
| `traditional_baselines_cifar.py` | 六种传统去噪方法、级联/调参策略、协议自检及对比图。 |
| `merge_results_cifar.py` | 合并 IR-SDE 与传统方法指标为 CSV 和 PSNR 透视表。 |

## 3. 与 MNIST 实验对照

| 项目 | MNIST | CIFAR-10 |
|---|---|---|
| IR-SDE 入口 | `main.py` | `main_cifar.py` |
| 传统基线入口 | `traditional_baselines.py` | `traditional_baselines_cifar.py` |
| 数据集 | MNIST 灰度图，28×28 | CIFAR-10 RGB，32×32 |
| UNet | 2 通道输入、1 通道输出 | 6 通道输入、3 通道输出 |
| 输出目录 | `outputs/ir_sde`、`outputs/traditional_single` | `outputs/cifar10/ir_sde`、`outputs/cifar10/traditional_single` |
| 数据位置 | `data/MNIST/raw` | `data/cifar-10-batches-py` |

## 4. 环境要求

- Python 3.9 或更新版本。
- 支持 CUDA 的 PyTorch 安装，以及可用的 NVIDIA CUDA 驱动。
- 项目所用依赖：torch、diffusers、torchvision、opencv-python、scikit-image、bm3d、pandas、tqdm、matplotlib。CIFAR 代码没有增加第三方依赖。
- 推荐环境为 RTX 3050 Ti Laptop 4G。训练默认 batch size 128、UNet 通道规模 `(64, 128)`，使用 AMP；实际峰值随软件版本和运行环境变化，应以训练日志为准。

## 5. 启动步骤

请在项目根目录执行以下命令。

### 5.1 冒烟测试（小样本，预计 2–5 分钟）

```bash
python main_cifar.py --epochs 1 --batch-size 16 --train-samples 128 --test-samples 64 --eval-samples 64 --device cuda --output-dir ./outputs/cifar10_smoke/ir_sde
```

### 5.2 完整训练 IR-SDE（3050 Ti 推荐折中档，预计 1.5–3 小时）

```bash
python main_cifar.py --epochs 10 --batch-size 128 --samples-per-image 2 --inference-steps 20 --eval-every 2 --num-workers 4 --device cuda --output-dir ./outputs/cifar10/ir_sde
```

### 5.3 传统基线（CPU，预计 25–50 分钟）

```bash
python traditional_baselines_cifar.py --data-root ./data --ir-sde-output ./outputs/cifar10/ir_sde --output-dir ./outputs/cifar10/traditional_single --seed 42 --device cpu --strategy single
```

`--strategy` 还可设为 `cascade` 或 `tuned`，策略仅在评估噪声包含 `mixed` 时生效。可用 `--methods` 选择六种基础方法的子集，也可用 `--noise-types` 显式选择噪声。

### 5.4 合并结果

```bash
python merge_results_cifar.py
```

## 6. 输出目录结构

```text
outputs/
└── cifar10/
    ├── ir_sde/
    │   ├── fixed_eval.json
    │   ├── config.json
    │   ├── history.json / history.csv
    │   ├── batch_history.json / batch_history.csv
    │   ├── curves.png
    │   ├── comparison.png
    │   ├── comparison_{noise}.png
    │   ├── checkpoint.pt
    │   ├── metrics.json
    │   ├── unet/
    │   └── scheduler/
    ├── traditional_single/
    │   ├── metrics.json
    │   ├── summary.csv
    │   └── compare_{noise}.png
    └── combined/
        └── summary_combined.csv
```

冒烟测试使用独立目录 `outputs/cifar10_smoke/ir_sde`，不会覆盖完整训练的默认输出。
若只需在无 CUDA 的环境检查逻辑，可将冒烟命令中的 `--device cuda` 改为 `--device cpu`；CPU 运行不打印 GPU 显存峰值。

## 7. 关键设计说明

- **平行实现而非泛化：** MNIST 与 RGB CIFAR-10 的张量通道数、分辨率、SSIM 口径和显存预算不同；分开入口减少对既有 MNIST 实验的行为影响。
- **复用共享采样组件：** 直接复用 `IRSdeScheduler`，并以现有 `IRSdePipeline` 为父类。由于原 Pipeline 的输入校验固定要求单通道，新增的 CIFAR Pipeline 适配类使用同一 scheduler 和跳步公式，接受 RGB 张量。
- **`in_channels=6`：** 网络输入为 `cat([x_t, y], dim=1)`，当前状态 `x_t` 和条件观测 `y` 各含 3 个 RGB 通道。
- **较温和的 Gaussian 噪声：** CIFAR 默认 sigma 范围设为 `(0.03, 0.15)`，避免彩色图使用更大的 MNIST 默认噪声范围导致退化过强；传统基线使用相同范围以保持协议对齐。
- **BM3D 局限：** BM3D 库按灰度图工作，因此逐 RGB 通道处理；可能产生伪彩，这是已知局限，应在结果讨论中说明。
- **SSIM：** 除 PSNR 外，SSIM 衡量恢复图像与干净图像的结构相似度，有助于评估图像结构保留情况。
- **AMP：** CUDA 训练使用 autocast 与 GradScaler，降低激活显存并争取提升吞吐；数值有限性检查和梯度裁剪仍保留。

## 8. RTX 3050 Ti Laptop 4G 常见问题

- **CUDA OOM：** 将 `--batch-size` 降到 64；仍不足时，将 `ir_sde/model_cifar.py` 中 `block_out_channels` 从 `(64, 128)` 降为 `(32, 64)`。脚本会在训练 OOM 时输出对应提示。
- **CIFAR 下载慢：** 首次运行由 torchvision 自动下载，数据约 170 MB；下载完成后复用本地缓存。确保网络能访问 torchvision 数据源，或稍后重试。
- **传统基线运行较慢：** NLM、TV、BM3D 等在 CPU 上逐图像/逐通道运行，耗时主要受 CPU 性能影响，GPU 不会加速这些 NumPy/scikit-image 算法。
- **笔记本散热：** 训练时建议接通电源、使用高性能电源模式并保持通风；长时间训练注意温度和降频。显存峰值由每个 epoch 的日志给出。

## 9. 数据下载与路径

`NoisyCIFAR10Dataset(root="./data", download=True)` 使用 torchvision 默认目录，自动下载到 `data/cifar-10-batches-py`。这与 MNIST 的 `data/MNIST/raw` 同属 `data` 父目录；不需要手动设置子目录或单独下载数据。
