# 传统去噪基线

## 简介

本脚本评估高斯模糊(gaussian_blur)、中值滤波(median_blur)、小波软阈值(wavelet)、非局部均值(nlm)、全变分(tv)和 BM3D 六种传统去噪方法，输出 PSNR、SSIM、对比图及协议自检结果。

## 输入文件

- `outputs/ir_sde/fixed_eval.json`：提供固定评估索引。
- `outputs/ir_sde/metrics.json`：提供 IR-SDE 的 `per_noise` 和 noisy-PSNR 参考值。

未传 `--noise-types` 时，脚本自动采用 `metrics.json` 中 `per_noise` 包含的噪声类型；传入后以命令行选择为准，但协议自检仍读取 `metrics.json`。

## 执行命令

```bash
python traditional_baselines.py --data-root ./data --ir-sde-output outputs/ir_sde --output-dir outputs/traditional_single --seed 42 --device cpu --strategy single
python traditional_baselines.py --data-root ./data --ir-sde-output outputs/ir_sde --output-dir outputs/traditional_cascade --seed 42 --device cpu --strategy cascade
python traditional_baselines.py --data-root ./data --ir-sde-output outputs/ir_sde --output-dir outputs/traditional_tuned --seed 42 --device cpu --strategy tuned
```

`cascade` 和 `tuned` 仅在所选噪声包含 `mixed` 时生效。

## 可选参数

| 参数 | 说明 |
|---|---|
| `--methods` | 方法子集，默认全部六种；空格分隔。 |
| `--strategy` | `single`、`cascade` 或 `tuned`，默认 `single`。 |
| `--noise-types` | 支持 `all`、位掩码、全名或唯一前缀；默认自动对齐 IR-SDE。 |
| `--device` | 计算设备，默认 `cpu`。 |
| `--seed` | 随机种子，默认 `42`。 |

## 输出文件

- `metrics.json`：`config` 保存运行配置，`best_per_noise` 保存各噪声最优方法与 PSNR，`per_method` 保存逐方法逐噪声指标；`cascade` 和 `tuned` 结果也写入此处。
- `summary.csv`：按噪声类型和方法排序的扁平指标表。
- `compare_{noise}.png`：3×8 的干净、含噪、最优传统方法恢复结果对比图。

运行后会进行协议自检，验证 noisy_psnr 与 IR-SDE 一致。

## 指标口径

- `noisy_psnr`：含噪图像的 PSNR。
- `restored_psnr`：传统方法恢复图像的 PSNR。
- `ssim`：恢复图像与干净图像的平均 SSIM。
- `available`：对应方法是否可用。