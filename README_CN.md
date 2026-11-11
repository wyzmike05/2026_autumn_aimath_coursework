# 基于 IR-SDE 与传统方法的图像去噪实验

## 项目概述

本项目实现并比较两类图像去噪方法：

- **IR-SDE**：基于条件随机微分方程的概率生成式去噪方法。
- **六种传统方法**：高斯模糊（Gaussian blur）、中值滤波（median blur）、小波去噪（wavelet）、非局部均值（NLM）、全变分（TV）和 BM3D。

实验分别在两个数据集上完成：

| 数据集 | 图像类型 | 分辨率 | 实验内容 |
|---|---|---:|---|
| MNIST | 单通道灰度图 | 28×28 | IR-SDE 与六种传统方法在四类合成噪声下的去噪对比 |
| CIFAR-10 | 三通道 RGB 彩色图 | 32×32 | IR-SDE 与六种传统方法在四类合成噪声下的去噪对比 |

四类噪声为高斯、椒盐、散斑和混合噪声。项目关注的不只是降低像素误差，也包括去噪时对边缘、纹理和结构的保留。代码记录 PSNR；CIFAR-10 的 IR-SDE 与传统方法还记录 SSIM。**当前 MNIST IR-SDE 评估未计算 SSIM**，所以 MNIST 合并表中 IR-SDE 的 SSIM 项为空，不应作为 0 或与传统方法的 SSIM 作直接比较。

请在项目根目录执行下列命令。

## 项目结构与实验入口

| 功能 | MNIST | CIFAR-10 |
|---|---|---|
| IR-SDE 训练与评估 | `main.py` | `main_cifar.py` |
| 数据集和在线噪声 | `ir_sde/data.py` | `ir_sde/data_cifar.py` |
| UNet 模型构造 | `ir_sde/model.py` 中的 `make_mnist_unet()` | `ir_sde/model_cifar.py` 中的 `make_cifar_unet()` |
| 推理管线 | `ir_sde/model.py` 中的共享 `IRSdePipeline` | `ir_sde/pipeline_cifar.py` 中的 RGB 适配管线 |
| 六种传统基线 | `traditional_baselines.py` | `traditional_baselines_cifar.py` |
| 结果合并 | `merge_results.py` | `merge_results_cifar.py` |
| 默认 IR-SDE 输出目录 | `outputs/ir_sde`（示例命令使用 `outputs/mnist/ir_sde`） | `outputs/cifar10/ir_sde` |
| 传统基线输出目录 | 示例命令使用 `outputs/mnist/traditional_single` | `outputs/cifar10/traditional_single` |

`ir_sde/model.py` 中的 `IRSdeScheduler` 在两套实验间复用。MNIST 使用 2 通道输入、1 通道输出；CIFAR UNet 将三通道状态 `x_t` 与三通道观测 `y` 拼接，使用 6 通道输入、3 通道输出。

## 环境依赖

- Python 3.9 或更新版本。
- PyTorch 与相匹配的 torchvision；如需 GPU 训练，请安装与本机驱动兼容的 **CUDA 版 PyTorch**。
- `diffusers`
- `opencv-python`
- `scikit-image`
- `bm3d`
- `pandas`
- `tqdm`
- `matplotlib`

可在所选 Python 环境中安装项目依赖：

```bash
python -m pip install torch torchvision diffusers opencv-python scikit-image bm3d pandas tqdm matplotlib
```

PyTorch 的 CUDA 安装方式请按 PyTorch 官网提供的命令选择匹配本机 CUDA/驱动的构建版本。建议使用支持 CUDA 的 NVIDIA GPU 训练；例如 RTX 3050 Ti Laptop 4G 可运行 CIFAR 默认轻量 UNet，并使用 AMP。显存实际使用量会随模型、batch size 和软件版本变化。传统方法主要通过 CPU 上的 NumPy/OpenCV/scikit-image 运算，不依赖 GPU 加速。

## 数据与输出目录说明

仓库的 `.gitignore` 忽略了 `data/` 和 `outputs/`。因此从 GitHub 克隆后，这两个目录及其数据/结果文件不会由 Git 下载；目录可能不存在，也可能因本地操作而为空。

### 数据集自动下载

两个数据集由 `torchvision.datasets` 加载并设置 `download=True`。首次执行相应实验时会自动下载到共同的数据根目录 `./data/`，下载后可复用缓存：

```text
data/
├── MNIST/
│   └── raw/
└── cifar-10-batches-py/
```

MNIST 与 CIFAR-10 的下载由各自实验第一次构造数据集时触发。需要可用网络和足够磁盘空间；如果下载失败，可检查数据源连通性后重试。

### 实验输出

运行脚本时会自动创建所需输出目录。推荐的完整结果结构如下：

```text
outputs/
├── mnist/
│   ├── ir_sde/
│   │   ├── config.json
│   │   ├── fixed_eval.json
│   │   ├── history.json / history.csv
│   │   ├── batch_history.json / batch_history.csv
│   │   ├── checkpoint.pt
│   │   ├── metrics.json
│   │   ├── comparison.png
│   │   ├── comparison_{noise}.png
│   │   ├── curves.png
│   │   ├── unet/
│   │   └── scheduler/
│   ├── traditional_single/
│   │   ├── metrics.json
│   │   ├── summary.csv
│   │   └── compare_{noise}.png
│   └── combined/
│       └── summary_combined.csv
└── cifar10/
    ├── ir_sde/
    │   └── （与 MNIST ir_sde 类似的配置、模型、指标、曲线和图片）
    ├── traditional_single/
    │   ├── metrics.json
    │   ├── summary.csv
    │   └── compare_{noise}.png
    └── combined/
        └── summary_combined.csv
```

其中 `comparison.png` 是 IR-SDE 的最终视觉对比图，图像排列为三排：**干净原图、含噪图、恢复图**。传统方法的 `compare_{noise}.png` 展示相应噪声下的干净图、含噪图和最佳传统方法恢复结果。`fixed_eval.json` 保存固定评估索引及协议参数；`metrics.json` 保存指标，传统结果的指标按方法组织在 `per_method` 中。

## MNIST 实验启动步骤

### 1. 训练 IR-SDE

```bash
python main.py --epochs 10 --device cuda --output-dir ./outputs/mnist/ir_sde
```

Windows 用户若遇到 DataLoader 多进程问题，可添加 `--num-workers 0`：

```bash
python main.py --epochs 10 --device cuda --num-workers 0 --output-dir ./outputs/mnist/ir_sde
```

### 2. 运行传统基线

传统方法在 CPU 上运行，并读取 IR-SDE 生成的 `fixed_eval.json` 以对齐评估样本：

```bash
python traditional_baselines.py --ir-sde-output ./outputs/mnist/ir_sde --output-dir ./outputs/mnist/traditional_single
```

Windows 下如遇多进程问题，可以添加 `--num-workers 0`（如果该脚本版本提供此参数）；基线本身主要使用 CPU NumPy/OpenCV 运算。

### 3. 合并结果

```bash
python merge_results.py
```

运行前请确认 `merge_results.py` 中的路径指向：

```text
outputs/mnist/ir_sde/metrics.json
outputs/mnist/traditional_single/metrics.json
outputs/mnist/combined/summary_combined.csv
```

当前仓库版本已将 MNIST 合并脚本配置为上述 `outputs/mnist/` 路径。

## CIFAR-10 实验启动步骤

### 1. 冒烟测试

使用较小样本验证训练、RGB 推理和结果保存流程：

```bash
python main_cifar.py --epochs 1 --batch-size 16 --train-samples 128 --test-samples 64 --eval-samples 64 --num-workers 0 --device cuda --output-dir ./outputs/cifar10_smoke/ir_sde
```

冒烟测试输出放在独立的 `outputs/cifar10_smoke/` 下，不会覆盖正式结果。

### 2. 完整训练 IR-SDE

```bash
python main_cifar.py --epochs 10 --batch-size 128 --samples-per-image 2 --inference-steps 20 --eval-every 2 --num-workers 0 --device cuda --output-dir ./outputs/cifar10/ir_sde
```

CIFAR-10 默认 UNet 的 `block_out_channels=(64,128)`，训练启用 CUDA AMP。若发生显存不足，可降低 `--batch-size`（例如 64），或按代码提示将通道配置降为 `(32,64)`。

### 3. 运行六种传统基线

```bash
python traditional_baselines_cifar.py --ir-sde-output ./outputs/cifar10/ir_sde --output-dir ./outputs/cifar10/traditional_single --device cpu --strategy single
```

传统基线会读取 `outputs/cifar10/ir_sde/fixed_eval.json`，在固定样本和噪声协议下评估四种噪声，并生成协议自检结果与 `compare_{noise}.png`。需先成功运行 IR-SDE，生成相应固定评估文件和指标。

### 4. 合并结果

```bash
python merge_results_cifar.py
```

## 结果查看指南

**最终跨方法对比表**为：

- MNIST：`outputs/mnist/combined/summary_combined.csv`
- CIFAR-10：`outputs/cifar10/combined/summary_combined.csv`

在表中按 `noise_type` 筛选噪声类型，按 `method` 比较方法；主要指标包括 noisy/restored PSNR、SSIM 和方法可用状态。详细指标可查各数据集的 `ir_sde/metrics.json` 与 `traditional_single/metrics.json`。

**IR-SDE 视觉结果图**为：

- MNIST：`outputs/mnist/ir_sde/comparison.png`
- CIFAR-10：`outputs/cifar10/ir_sde/comparison.png`

报告中的实验分析和结果说明可继续查看项目根目录的 `analyze.md`，目录结构与指标口径说明可查看 `outputs/README.md`。

### 指标解释与比较注意事项

- PSNR 由整个评估集合的全局 MSE 计算，数值越高通常表示像素误差越小。
- SSIM 衡量图像结构相似度，有助于讨论结构保留，但不能单独证明纹理或颜色伪影。
- MNIST 的当前 IR-SDE `evaluate` 未计算 SSIM，因此合并表中相应值为空；请勿把空值当成 0。
- MNIST 和 CIFAR-10 的分辨率、通道数及噪声强度范围不同。跨数据集比较时应说明实验配置差异。

## 技术亮点与局限

- **条件概率生成去噪：** IR-SDE 以干净图像、含噪观测及随机过程状态构造训练目标，并通过条件采样生成恢复结果。
- **CIFAR RGB 联合输入：** CIFAR UNet 将当前状态 `x_t` 和观测 `y` 的 3 个 RGB 通道拼接，形成 `in_channels=6` 的输入，并输出 3 个通道。网络可在特征中联合利用 RGB 信息。
- **AMP 混合精度：** CIFAR CUDA 训练使用 autocast 与 GradScaler，降低激活显存占用并提高 GPU 训练效率；脚本也记录每个 epoch 的显存峰值。
- **BM3D 的多通道局限：** 所用 BM3D 算法按 RGB 通道分别进行灰度去噪，没有显式建模通道间关系，理论上可能导致颜色不一致或伪彩。应结合 `compare_{noise}.png` 讨论实际视觉效果，而不要只凭 PSNR/SSIM 断言已出现颜色伪影。
- **噪声与传统先验的匹配：** 中值滤波对椒盐噪声具有机制上的优势，BM3D 常用于加性高斯噪声；不同方法的表现取决于噪声类型、参数与数据。

## Windows 使用提示

- 本 README 中的训练命令均建议使用 `--num-workers 0`，以降低 Windows DataLoader 多进程启动问题；训练速度或数据加载情况稳定后可自行调整。
- CIFAR-10 脚本默认设备为 CUDA；在无可用 CUDA 的环境中可显式传 `--device cpu` 进行逻辑测试，但完整训练会明显变慢。MNIST 也推荐在支持 CUDA 的 GPU 上运行。
- Windows 终端运行脚本时，建议从项目根目录启动，以确保 `./data`、`./outputs/...` 等相对路径解析正确。
