# 2026_autumn_aimath_coursework
Course work for Foundational Mathematics and Algorithms for Artificial Intelligence, autumn, 2026

```bash
python main.py --output-dir "./outputs/ir_sde" --epochs 1 --batch-size 256 --lr 1e-3 --timesteps 100 --lam 0.1 --noise-probs 0.5 0.5 0 0 --inference-steps 50 --samples-per-image 1 --seed 42 --device cpu
```

```bash
python main.py --output-dir "./outputs/ir_sde_all" --epochs 1 --batch-size 256 --lr 1e-3 --timesteps 100 --lam 0.1 --noise-probs 0.25 0.25 0.25 0.25 --inference-steps 50 --samples-per-image 1 --seed 42 --device cpu
```



# 项目完整运行复现指南

本项目包含 MNIST 和 CIFAR-10 两个数据集的完整实验。运行顺序如下：

## 第一部分：MNIST 实验（灰度 28x28）
```bash
# 1. 训练 IR-SDE (使用 GPU)
python main.py --epochs 10 --batch-size 64 --samples-per-image 4 --inference-steps 20 --eval-every 2 --num-workers 0 --device cuda --output-dir ./outputs/mnist/ir_sde

# 2. 跑传统基线对比 (使用 CPU)
python traditional_baselines.py --data-root ./data --ir-sde-output ./outputs/mnist/ir_sde --output-dir ./outputs/mnist/traditional_single --seed 42 --device cpu --strategy single

# 3. 合并结果 (前提：确保 merge_results.py 中的路径已指向 outputs/mnist/)
python merge_results.py
```

## 第二部分：CIFAR-10 实验（彩色 32x32）
```bash
# 0. (这个不需要参考) 冒烟测试，验证显存是否够用
python main_cifar.py --epochs 1 --batch-size 16 --train-samples 128 --test-samples 64 --eval-samples 64 --device cuda --output-dir ./outputs/cifar10_smoke/ir_sde

# 1. 训练 IR-SDE (使用 GPU，开启 AMP)
python main_cifar.py --epochs 10 --batch-size 128 --samples-per-image 2 --inference-steps 20 --eval-every 2 --num-workers 0 --device cuda --output-dir ./outputs/cifar10/ir_sde

# 2. 跑传统基线对比 (使用 CPU)
python traditional_baselines_cifar.py --data-root ./data --ir-sde-output ./outputs/cifar10/ir_sde --output-dir ./outputs/cifar10/traditional_single --seed 42 --device cpu --strategy single

# 3. 合并结果
python merge_results_cifar.py
```

