"""Recompute per-noise MNIST IR-SDE SSIM on the fixed evaluation set."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from diffusers import UNet2DModel
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from ir_sde.data import NoisyMNISTDataset
from ir_sde.model import IRSdePipeline, IRSdeScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="./outputs/mnist/ir_sde")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    fixed_eval = json.loads(
        (output_dir / "fixed_eval.json").read_text(encoding="utf-8")
    )
    indices = fixed_eval["indices"]
    if len(indices) != 256:
        raise ValueError(f"Expected 256 fixed evaluation indices, got {len(indices)}")

    seed = int(fixed_eval["seed"])
    if seed != int(config["seed"]):
        raise ValueError("fixed_eval.json seed does not match config.json")
    samples_per_image = int(fixed_eval["test_samples_per_image"])
    if samples_per_image != int(config["test_samples_per_image"]):
        raise ValueError("Fixed evaluation sample count does not match config.json")

    device = torch.device(args.device or config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment")
    if device.type == "cuda":
        print(f"Device: {torch.cuda.get_device_name(device)}", flush=True)
    else:
        print(f"Device: {device}", flush=True)

    data_root = args.data_root or config["data_root"]
    data_root = Path(data_root)
    if not data_root.is_absolute():
        data_root = Path(__file__).resolve().parent / data_root
    noise_ranges = config.get("noise_ranges")
    batch_size = int(config["batch_size"])
    inference_steps = int(fixed_eval["inference_steps"])

    unet = UNet2DModel.from_pretrained(output_dir / "unet").to(device)
    scheduler = IRSdeScheduler.from_pretrained(output_dir / "scheduler")
    pipeline = IRSdePipeline(unet=unet, scheduler=scheduler).to(device)
    pipeline.set_progress_bar_config(disable=True)

    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_noise = metrics.get("per_noise")
    if not isinstance(per_noise, dict):
        raise ValueError("metrics.json is missing the per_noise object")

    for kind in NoisyMNISTDataset.NOISE_TYPES:
        if kind not in per_noise:
            raise ValueError(f"metrics.json is missing per_noise[{kind!r}]")
        dataset = NoisyMNISTDataset(
            root=str(data_root),
            train=False,
            noise_probs={kind: 1.0},
            seed=seed,
            samples_per_image=samples_per_image,
            noise_ranges=noise_ranges,
        )
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        sample_ssim = []
        pipeline.unet.eval()
        with torch.no_grad():
            for noisy, clean in tqdm(loader, desc=f"SSIM {kind}", unit="batch"):
                noisy = noisy.to(device)
                restored = pipeline(
                    noisy,
                    num_inference_steps=inference_steps,
                    generator=generator,
                ).images
                restored = restored.cpu().squeeze(1).numpy()
                clean = clean.squeeze(1).numpy()
                sample_ssim.extend(
                    structural_similarity(
                        target,
                        prediction,
                        data_range=1.0,
                    )
                    for target, prediction in zip(clean, restored)
                )
        if len(sample_ssim) != 256:
            raise RuntimeError(
                f"{kind}: expected 256 SSIM values, got {len(sample_ssim)}"
            )
        score = float(sum(sample_ssim) / len(sample_ssim))
        per_noise[kind]["ssim"] = score
        print(f"{kind}: SSIM={score:.6f} (n={len(sample_ssim)})", flush=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=metrics_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            json.dump(metrics, temp_file, indent=2)
            temp_path = Path(temp_file.name)
        os.replace(temp_path, metrics_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    print(f"Updated {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
