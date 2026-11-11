# merge_results.py
"""Merge IR-SDE and traditional baseline metrics into one summary table."""

import json
from pathlib import Path

import pandas as pd

IR_METRICS   = Path("outputs/mnist/ir_sde/metrics.json")
TRAD_METRICS = Path("outputs/mnist/traditional_single/metrics.json")
OUT_DIR      = Path("outputs/mnist/combined")
OUT_CSV      = OUT_DIR / "summary_combined.csv"

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ir   = json.loads(IR_METRICS.read_text(encoding="utf-8"))["per_noise"]
    trad = json.loads(TRAD_METRICS.read_text(encoding="utf-8"))["per_method"]

    rows = []
    for noise, m in ir.items():
        rows.append({
            "method": "IR-SDE", "noise_type": noise,
            "noisy_psnr": m["noisy_psnr"], "restored_psnr": m["restored_psnr"],
            "ssim": m.get("ssim"),          # IR-SDE 目前没有 ssim，会是 None
            "available": True, "note": "irsde",
        })
    for method, per_noise in trad.items():
        for noise, m in per_noise.items():
            if not m.get("available"):
                continue
            rows.append({
                "method": method, "noise_type": noise,
                "noisy_psnr": m["noisy_psnr"], "restored_psnr": m["restored_psnr"],
                "ssim": m["ssim"],
                "available": True, "note": m.get("note", ""),
            })

    df = pd.DataFrame(rows, columns=[
        "method", "noise_type", "noisy_psnr", "restored_psnr", "ssim", "available", "note",
    ])
    order = {n: i for i, n in enumerate(["gaussian", "salt_pepper", "speckle", "mixed"])}
    df["_o"] = df["noise_type"].map(order)
    df = df.sort_values(["_o", "method"], kind="mergesort").drop(columns="_o")
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")

    # 顺手出 markdown 透视表，直接贴报告
    table = df.pivot(index="method", columns="noise_type", values="restored_psnr")
    print(table.round(2).to_string())

if __name__ == "__main__":
    main()