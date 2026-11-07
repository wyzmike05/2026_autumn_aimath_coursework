"""Train and evaluate conditional IR-SDE on paired noisy MNIST."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm.auto import tqdm

from ir_sde.data import NoisyMNISTDataset
from ir_sde.model import IRSdePipeline, IRSdeScheduler, make_mnist_unet


def train_epoch(
    unet,
    scheduler,
    loader,
    optimizer,
    device,
    description="Training",
    batch_history=None,
    epoch=1,
    lr_scheduler=None,
):
    unet.train()
    total, total_noise, count = 0.0, 0.0, 0
    progress = tqdm(loader, desc=description, unit="batch", dynamic_ncols=True)
    for y, x0 in progress:
        y, x0 = y.to(device), x0.to(device)
        t = torch.randint(1, scheduler.T + 1, (x0.size(0),), device=device)
        noise = torch.randn_like(x0)
        xt = scheduler.add_noise(x0, noise, t, y)
        prediction = unet(torch.cat([xt, y], dim=1), t).sample
        target = x0 if scheduler.config.prediction_type == "sample" else noise
        loss = F.mse_loss(prediction, target)
        with torch.no_grad():
            _, predicted_noise = scheduler.prediction_to_x0_noise(prediction, xt, t, y)
            noise_mse = F.mse_loss(predicted_noise, noise).item()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        current_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        batch_loss = loss.item()
        if batch_history is not None:
            batch_history.append(
                {
                    "global_batch": len(batch_history) + 1,
                    "epoch": epoch,
                    "train_loss": batch_loss,
                    "train_noise_mse": noise_mse,
                    "lr": current_lr,
                }
            )
        total += batch_loss * x0.size(0)
        total_noise += noise_mse * x0.size(0)
        count += x0.size(0)
        progress.set_postfix(
            loss=f"{batch_loss:.6f}",
            noise_mse=f"{noise_mse:.6f}",
            avg_loss=f"{total / count:.6f}",
            lr=f"{current_lr:.3e}",
            refresh=True,
        )
    if count == 0:
        raise ValueError("Training loader must not be empty")
    return total / count, total_noise / count


@torch.no_grad()
def evaluate(unet, scheduler, loader, device, steps, seed, image_path):
    unet.eval()
    pipeline = IRSdePipeline(unet=unet, scheduler=scheduler).to(device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    noisy_error, restored_error, pixels = 0.0, 0.0, 0
    progress = tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True)
    for index, (y, x0) in enumerate(progress):
        y, x0 = y.to(device), x0.to(device)
        restored = pipeline(y, num_inference_steps=steps, generator=generator).images
        if not torch.isfinite(restored).all():
            raise RuntimeError("Non-finite restoration; check SDE parameters")
        noisy_error += F.mse_loss(y, x0, reduction="sum").item()
        restored_error += F.mse_loss(restored, x0, reduction="sum").item()
        pixels += x0.numel()
        progress.set_postfix(restored_mse=f"{restored_error / pixels:.6f}")
        if index == 0:
            n = min(8, x0.size(0))
            # Rows: clean targets, noisy observations, restored images.
            save_image(
                torch.cat([x0[:n], y[:n], restored[:n]]).cpu(), image_path, nrow=n
            )
    if pixels == 0:
        raise ValueError("Evaluation loader must not be empty")
    noisy_mse, restored_mse = noisy_error / pixels, restored_error / pixels
    return {
        "noisy_mse": noisy_mse,
        "restored_mse": restored_mse,
        # PSNR from dataset-wide MSE with pixel range [0, 1].
        "noisy_psnr": -10 * torch.tensor(noisy_mse).log10().item(),
        "restored_psnr": -10 * torch.tensor(restored_mse).log10().item(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="./outputs/ir_sde")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Rate per integer step; default: 5.0 / timesteps",
    )
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument(
        "--noise-probs", type=float, nargs=4, default=None,
        metavar=("GAUSSIAN", "SALT_PEPPER", "SPECKLE", "MIXED"),
        help="Noise probabilities in this order: gaussian salt_pepper speckle mixed; sum must be 1",
    )
    parser.add_argument("--prediction-type", choices=("sample", "epsilon"), default="sample",
                        help="Predict clean image (stable default) or diffusion noise")
    parser.add_argument("--gaussian-sigma", type=float, nargs=2, default=(0.05, 0.20),
                        metavar=("MIN", "MAX"), help="Gaussian corruption standard deviation range")
    parser.add_argument("--salt-pepper-amount", type=float, nargs=2, default=(0.01, 0.08),
                        metavar=("MIN", "MAX"), help="Fraction of pixels selected for salt/pepper replacement")
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate fixed test samples every N epochs",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=256,
        help="Number of fixed test samples for intermediate evaluation",
    )
    parser.add_argument(
        "--samples-per-image",
        type=int,
        default=4,
        help="Independent noisy samples per training image (default: 4)",
    )
    parser.add_argument(
        "--test-samples-per-image",
        type=int,
        default=1,
        help="Fixed noisy samples per test image (default: 1)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    try:
        args.noise_probs = NoisyMNISTDataset.validate_noise_probs(
            None if args.noise_probs is None else
            dict(zip(NoisyMNISTDataset.NOISE_TYPES, args.noise_probs))
        )
    except ValueError as error:
        parser.error(str(error))
    if not all(math.isfinite(v) for v in (args.lr, args.lam)) or (
        args.theta is not None and not math.isfinite(args.theta)
    ):
        parser.error("lr, lam and theta must be finite")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")
    if args.timesteps < 1 or not 1 <= args.inference_steps <= args.timesteps:
        parser.error("require 1 <= inference-steps <= timesteps")
    if args.lam <= 0 or (args.theta is not None and args.theta <= 0):
        parser.error("theta and lam must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    if args.eval_every < 1 or args.eval_samples < 1:
        parser.error("eval-every and eval-samples must be positive")
    for name, bounds in (("gaussian-sigma", args.gaussian_sigma),
                         ("salt-pepper-amount", args.salt_pepper_amount)):
        if not all(math.isfinite(v) for v in bounds) or not 0 <= bounds[0] <= bounds[1]:
            parser.error(f"{name} must be a finite nonnegative increasing range")
    if args.salt_pepper_amount[1] > 1:
        parser.error("salt-pepper-amount must be <= 1")
    if args.samples_per_image < 1 or args.test_samples_per_image < 1:
        parser.error("samples-per-image and test-samples-per-image must be positive")
    for limit in (args.train_samples, args.test_samples):
        if limit is not None and limit < 1:
            parser.error("sample limits must be positive")
    return args


def save_history(history, batch_history, output):
    """Persist raw measurements and redraw curves after each epoch."""
    fields = [
        "epoch",
        "global_batch",
        "train_noise_mse",
        "train_loss",
        "test_noisy_mse",
        "test_restored_mse",
        "test_noisy_psnr",
        "test_restored_psnr",
    ]
    with (output / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    with (output / "batch_history.json").open("w", encoding="utf-8") as file:
        json.dump(batch_history, file, indent=2)
    with (output / "batch_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["global_batch", "epoch", "train_loss", "train_noise_mse", "lr"]
        )
        writer.writeheader()
        writer.writerows(batch_history)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        [r["global_batch"] for r in batch_history],
        [r["train_loss"] for r in batch_history],
        linewidth=0.8,
        alpha=0.65,
        label="Train loss (each batch)",
    )
    for index, row in enumerate(history):
        step = row["global_batch"]
        ax.axvline(step, color="gray", linestyle=":", alpha=0.4)
        ax.scatter(
            step,
            batch_history[step - 1]["train_loss"],
            color="tab:blue",
            s=25,
            zorder=4,
            label="Epoch end" if index == 0 else None,
        )
        ax.annotate(
            f"E{row['epoch']}",
            (step, 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    evaluated = [r for r in history if "test_restored_mse" in r]
    steps = [r["global_batch"] for r in evaluated]
    psnr_ax = ax.twinx()
    for kind, color in (("noisy", "tab:gray"), ("restored", "tab:orange")):
        ax.plot(
            steps,
            [r[f"test_{kind}_mse"] for r in evaluated],
            marker="o",
            linestyle="--",
            color=color,
            label=f"Fixed test {kind} MSE",
        )
        psnr_ax.plot(
            steps,
            [r[f"test_{kind}_psnr"] for r in evaluated],
            marker="s",
            linestyle=":",
            color=color,
            label=f"Fixed test {kind} PSNR",
        )
    ax.set_xlabel("Global training batch (continuous across epochs)")
    ax.set_ylabel("MSE / training loss")
    psnr_ax.set_ylabel("PSNR (dB)")
    ax.set_title("Training and fixed test history", pad=30)
    ax.grid(alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    right_lines, right_labels = psnr_ax.get_legend_handles_labels()
    ax.legend(lines + right_lines, labels + right_labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    theta = 5.0 / args.timesteps if args.theta is None else args.theta
    scheduler = IRSdeScheduler(T=args.timesteps, theta=theta, lam=args.lam,
                              prediction_type=args.prediction_type)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    noise_ranges = {"gaussian": args.gaussian_sigma,
                    "salt_pepper": args.salt_pepper_amount}
    train_data = NoisyMNISTDataset(
        root=args.data_root,
        train=True,
        seed=args.seed,
        samples_per_image=args.samples_per_image,
        noise_probs=args.noise_probs,
        noise_ranges=noise_ranges,
    )
    test_data = NoisyMNISTDataset(
        root=args.data_root,
        train=False,
        seed=args.seed,
        samples_per_image=args.test_samples_per_image,
        noise_probs=args.noise_probs,
        noise_ranges=noise_ranges,
    )
    if args.train_samples is not None:
        train_data = Subset(train_data, range(min(args.train_samples, len(train_data))))
    if args.test_samples is not None:
        test_data = Subset(test_data, range(min(args.test_samples, len(test_data))))
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    # Fixed indices, corruption seeds and sampling seed keep epoch comparisons fair.
    eval_indices = torch.randperm(
        len(test_data), generator=torch.Generator().manual_seed(args.seed)
    )[: min(args.eval_samples, len(test_data))].tolist()
    eval_loader = DataLoader(
        Subset(test_data, eval_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    with (output / "fixed_eval.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "indices": eval_indices,
                "seed": args.seed,
                "noise_probs": args.noise_probs,
                "inference_steps": args.inference_steps,
                "test_samples_per_image": args.test_samples_per_image,
            },
            file,
            indent=2,
        )
    with (output / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                **vars(args),
                "theta": theta,
                "noise_probs": args.noise_probs,
                "noise_ranges": {**NoisyMNISTDataset.DEFAULT_NOISE_RANGES, **noise_ranges},
                "lr_schedule": "cosine",
            },
            file,
            indent=2,
        )
    unet = make_mnist_unet().to(device)
    # Integer time: theta*T=5 corresponds to terminal decay exp(-5).
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    # Step after each optimizer update; decay continuously across epoch boundaries.
    total_steps = args.epochs * len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0.0
    )
    print(
        f"Device: {device}; train: {len(train_data)}; test: {len(test_data)}",
        flush=True,
    )
    print(f"Noise probabilities: {args.noise_probs}", flush=True)
    history = []
    batch_history = []
    for epoch in range(1, args.epochs + 1):
        loss, noise_mse = train_epoch(
            unet,
            scheduler,
            train_loader,
            optimizer,
            device,
            description=f"Epoch {epoch}/{args.epochs}",
            batch_history=batch_history,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
        )
        print(f"Epoch {epoch}/{args.epochs}: loss = {loss:.6f}, noise MSE = {noise_mse:.6f}", flush=True)
        history.append(
            {
                "epoch": epoch,
                "global_batch": len(batch_history),
                "train_loss": loss,
                "train_noise_mse": noise_mse,
            }
        )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            fixed_metrics = evaluate(
                unet,
                scheduler,
                eval_loader,
                device,
                args.inference_steps,
                args.seed,
                output / f"comparison_epoch_{epoch:04d}.png",
            )
            history[-1].update(
                {f"test_{key}": value for key, value in fixed_metrics.items()}
            )
            print(
                f"Fixed test: MSE = {fixed_metrics['restored_mse']:.6f}, "
                f"PSNR = {fixed_metrics['restored_psnr']:.2f} dB",
                flush=True,
            )
        save_history(history, batch_history, output)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": unet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "scheduler_config": dict(scheduler.config),
                "args": vars(args),
                "history": history,
                "batch_history": batch_history,
            },
            output / "checkpoint.pt",
        )
    # Evaluate the held-out test set once, after training.
    metrics = evaluate(
        unet,
        scheduler,
        test_loader,
        device,
        args.inference_steps,
        args.seed,
        output / "comparison.png",
    )
    unet.save_pretrained(output / "unet")
    scheduler.save_pretrained(output / "scheduler")
    per_noise = {}
    for kind, probability in args.noise_probs.items():
        if probability == 0:
            continue
        dataset = NoisyMNISTDataset(
            root=args.data_root, train=False, noise_probs={kind: 1.0}, seed=args.seed,
            samples_per_image=args.test_samples_per_image, noise_ranges=noise_ranges,
        )
        loader = DataLoader(Subset(dataset, eval_indices), batch_size=args.batch_size,
                            generator=torch.Generator().manual_seed(args.seed))
        per_noise[kind] = evaluate(
            unet, scheduler, loader, device, args.inference_steps,
            args.seed, output / f"comparison_{kind}.png",
        )
    with (output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"history": history, **metrics, "per_noise": per_noise}, file, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
