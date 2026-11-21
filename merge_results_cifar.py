"""Merge CIFAR-10 IR-SDE and traditional metrics into one summary table."""

import json
from pathlib import Path

import pandas as pd

IR_METRICS = Path("outputs/cifar10/ir_sde/metrics.json")
TRAD_METRICS = Path("outputs/cifar10/traditional_single/metrics.json")
OUT_DIR = Path("outputs/cifar10/combined")
OUT_CSV = OUT_DIR / "summary_combined.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ir = json.loads(IR_METRICS.read_text(encoding="utf-8"))["per_noise"]
    trad = json.loads(TRAD_METRICS.read_text(encoding="utf-8"))["per_method"]

    rows = []
    for noise, metrics in ir.items():
        rows.append(
            {
                "method": "IR-SDE",
                "noise_type": noise,
                "noisy_psnr": metrics["noisy_psnr"],
                "restored_psnr": metrics["restored_psnr"],
                "ssim": metrics.get("restored_ssim"),
                "available": True,
                "note": "irsde",
            }
        )
    for method, per_noise in trad.items():
        for noise, metrics in per_noise.items():
            if not metrics.get("available"):
                continue
            rows.append(
                {
                    "method": method,
                    "noise_type": noise,
                    "noisy_psnr": metrics["noisy_psnr"],
                    "restored_psnr": metrics["restored_psnr"],
                    "ssim": metrics["ssim"],
                    "available": True,
                    "note": metrics.get("note", ""),
                }
            )

    frame = pd.DataFrame(
        rows,
        columns=[
            "method",
            "noise_type",
            "noisy_psnr",
            "restored_psnr",
            "ssim",
            "available",
            "note",
        ],
    )
    order = {
        noise: index
        for index, noise in enumerate(
            ["gaussian", "salt_pepper", "speckle", "mixed"]
        )
    }
    frame["_o"] = frame["noise_type"].map(order)
    frame = frame.sort_values(["_o", "method"], kind="mergesort").drop(
        columns="_o"
    )
    frame.to_csv(OUT_CSV, index=False, float_format="%.6f")

    table = frame.pivot(index="method", columns="noise_type", values="restored_psnr")
    print(table.round(2).to_string())
    print(f"\nSaved combined summary to {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
