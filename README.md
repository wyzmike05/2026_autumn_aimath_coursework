# 2026_autumn_aimath_coursework
Course work for Foundational Mathematics and Algorithms for Artificial Intelligence, autumn, 2026

```bash
python main.py --output-dir "./outputs/ir_sde" --epochs 1 --batch-size 256 --lr 1e-3 --timesteps 100 --lam 0.1 --noise-probs 0.5 0.5 0 0 --inference-steps 50 --samples-per-image 1 --seed 42 --device cpu
```

```bash
python main.py --output-dir "./outputs/ir_sde_all" --epochs 1 --batch-size 256 --lr 1e-3 --timesteps 100 --lam 0.1 --noise-probs 0.25 0.25 0.25 0.25 --inference-steps 50 --samples-per-image 1 --seed 42 --device cpu
```
