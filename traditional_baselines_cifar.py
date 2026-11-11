from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import cv2 as cv2_module
import numpy as np
import pandas as pd
import torch
from bm3d import BM3DStages, bm3d
from skimage.metrics import structural_similarity
from skimage.restoration import (
    denoise_nl_means,
    denoise_tv_chambolle,
    denoise_wavelet,
)
from torch.utils.data import Subset
from torchvision.utils import save_image
from tqdm.auto import tqdm

from ir_sde.data_cifar import NoisyCIFAR10Dataset

cv2_module.setNumThreads(1)

NOISE_TYPES = ("gaussian", "salt_pepper", "speckle", "mixed")
METHOD_NAMES = ("gaussian_blur", "median_blur", "wavelet", "nlm", "tv", "bm3d")
STRATEGIES = ("single", "cascade", "tuned")
SUMMARY_COLUMNS = (
    "method",
    "noise_type",
    "noisy_psnr",
    "restored_psnr",
    "ssim",
    "available",
    "note",
)
Restorer = Callable[[np.ndarray], np.ndarray]


def _to_float32(image: np.ndarray) -> np.ndarray:
    """将输入图像转为连续的 float32 [C,H,W]，支持单通道和 RGB。"""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3 or array.shape[0] not in (1, 3):
        raise ValueError(f"Expected [C, H, W] with C in (1, 3), got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Input image contains non-finite values")
    return np.ascontiguousarray(array)


def _match_input_shape(output: np.ndarray, input_shape: tuple[int, ...]) -> np.ndarray:
    """裁剪去噪结果到 [0,1] 并恢复输入形状。"""
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 2 and len(input_shape) == 3 and input_shape[0] == 1:
        array = array[np.newaxis, ...]
    if array.shape != input_shape:
        raise ValueError(
            f"Denoiser changed image shape from {input_shape} to {array.shape}"
        )
    if not np.isfinite(array).all():
        raise RuntimeError("Denoiser produced non-finite values")
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)


def _apply_opencv(image: np.ndarray, operation: Callable) -> np.ndarray:
    """将 [C,H,W] 转 HWC，经 OpenCV 处理后恢复通道优先格式。"""
    input_shape = np.asarray(image).shape
    source = _to_float32(image)
    hwc = np.transpose(source, (1, 2, 0))
    quantized = np.rint(hwc * 255.0).clip(0, 255).astype(np.uint8)
    filtered = operation(quantized)
    if filtered.ndim == 2:
        filtered = filtered[..., np.newaxis]
    result = np.transpose(filtered.astype(np.float32) / 255.0, (2, 0, 1))
    return _match_input_shape(result, input_shape)


def _build_gaussian_blur() -> Restorer:
    """构建固定参数的 OpenCV 高斯模糊；按 HWC 通道轴处理。"""

    def restore(image: np.ndarray) -> np.ndarray:
        return _apply_opencv(
            image, lambda source: cv2_module.GaussianBlur(source, (3, 3), sigmaX=0)
        )

    return restore


def _build_median_blur() -> Restorer:
    """构建固定参数的 OpenCV 中值滤波；按 HWC 通道轴处理。"""

    def restore(image: np.ndarray) -> np.ndarray:
        return _apply_opencv(
            image, lambda source: cv2_module.medianBlur(source, ksize=3)
        )

    return restore


def _build_wavelet(sigma: float = 0.1) -> Restorer:
    """小波去噪优先使用 channel_axis=0，skimage 不兼容时逐通道回退。"""

    def restore(image: np.ndarray) -> np.ndarray:
        input_shape = np.asarray(image).shape
        source = _to_float32(image)
        try:
            filtered = denoise_wavelet(
                source,
                sigma=sigma,
                mode="soft",
                wavelet="db1",
                channel_axis=0,
            )
        except (TypeError, ValueError):
            filtered = np.stack(
                [
                    denoise_wavelet(
                        channel,
                        sigma=sigma,
                        mode="soft",
                        wavelet="db1",
                        channel_axis=None,
                    )
                    for channel in source
                ],
                axis=0,
            )
        return _match_input_shape(filtered, input_shape)

    return restore


def _build_nlm(h: float = 0.1) -> Restorer:
    """NLM 优先使用 channel_axis=0，不支持时逐通道执行 fast_mode。"""

    def restore(image: np.ndarray) -> np.ndarray:
        input_shape = np.asarray(image).shape
        source = _to_float32(image)
        try:
            filtered = denoise_nl_means(
                source,
                h=h,
                patch_size=3,
                patch_distance=5,
                fast_mode=True,
                channel_axis=0,
            )
        except (TypeError, ValueError):
            filtered = np.stack(
                [
                    denoise_nl_means(
                        channel,
                        h=h,
                        patch_size=3,
                        patch_distance=5,
                        fast_mode=True,
                        channel_axis=None,
                    )
                    for channel in source
                ],
                axis=0,
            )
        return _match_input_shape(filtered, input_shape)

    return restore


def _build_tv(weight: float = 0.1) -> Restorer:
    """TV 去噪优先使用 channel_axis=0，skimage 不兼容时逐通道回退。"""

    def restore(image: np.ndarray) -> np.ndarray:
        input_shape = np.asarray(image).shape
        source = _to_float32(image)
        try:
            filtered = denoise_tv_chambolle(source, weight=weight, channel_axis=0)
        except (TypeError, ValueError):
            filtered = np.stack(
                [
                    denoise_tv_chambolle(channel, weight=weight, channel_axis=None)
                    for channel in source
                ],
                axis=0,
            )
        return _match_input_shape(filtered, input_shape)

    return restore


def add_bm3d_method() -> Restorer:
    """bm3d 库仅支持灰度，逐通道可能引入伪彩，这是已知局限，将在报告中讨论。"""

    def restore(image: np.ndarray) -> np.ndarray:
        input_shape = np.asarray(image).shape
        source = _to_float32(image)
        filtered = []
        for channel in source:
            result = bm3d(
                channel,
                sigma_psd=0.1,
                stage_arg=BM3DStages.ALL_STAGES,
            )
            if isinstance(result, tuple):
                result = result[0]
            filtered.append(result)
        return _match_input_shape(np.stack(filtered, axis=0), input_shape)

    return restore


METHOD_BUILDERS = {
    "gaussian_blur": _build_gaussian_blur,
    "median_blur": _build_median_blur,
    "wavelet": _build_wavelet,
    "nlm": _build_nlm,
    "tv": _build_tv,
    "bm3d": add_bm3d_method,
}

TUNED_SPECS = {
    "tv_tuned": ("weight", (0.05, 0.1, 0.2), _build_tv),
    "nlm_tuned": ("h", (0.05, 0.1, 0.2), _build_nlm),
    "wavelet_tuned": ("sigma", (0.05, 0.1, 0.2), _build_wavelet),
}


def _mse_to_psnr(mse: float) -> float:
    """将数据集级 MSE 转换为 PSNR。"""
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def _mean_ssim(clean: np.ndarray, restored: np.ndarray) -> float:
    """使用 channel_axis=0 计算单张灰度或 RGB 图像的 SSIM。"""
    clean_channels = _to_float32(clean)
    restored_channels = _to_float32(restored)
    return float(
        structural_similarity(
            clean_channels,
            restored_channels,
            data_range=1.0,
            channel_axis=0,
        )
    )


def evaluate_method(
    loader: Subset,
    restorer: Restorer | None,
    device: torch.device,
    description: str,
) -> dict:
    """在固定噪声子集上评估恢复函数并逐图计算 SSIM。"""
    noisy_error = 0.0
    restored_error = 0.0
    pixels = 0
    ssim_sum = 0.0
    ssim_count = 0
    progress = tqdm(loader, desc=description, unit="batch", dynamic_ncols=True)
    for y, x0 in progress:
        y = y.to(device)
        x0 = x0.to(device)
        clean_np = x0.detach().cpu().numpy()
        noisy_error += ((y - x0) ** 2).sum().item()
        pixels += x0.numel()
        noisy_np = y.detach().cpu().numpy()
        restored_np = restorer(noisy_np)
        restored = torch.from_numpy(restored_np).to(device=device, dtype=x0.dtype)
        if restored.shape != x0.shape:
            raise RuntimeError(
                f"Restorer changed shape from {tuple(x0.shape)} to "
                f"{tuple(restored.shape)}"
            )
        restored_error += ((restored - x0) ** 2).sum().item()
        if clean_np.ndim == 3:
            ssim_sum += _mean_ssim(clean_np, restored_np)
            ssim_count += 1
        else:
            for clean_image, restored_image in zip(clean_np, restored_np):
                ssim_sum += _mean_ssim(clean_image, restored_image)
                ssim_count += 1
        progress.set_postfix(restored_mse=f"{restored_error / pixels:.6f}")
    if pixels == 0:
        raise ValueError("Evaluation loader must not be empty")
    noisy_mse = noisy_error / pixels
    restored_mse = restored_error / pixels
    return {
        "noisy_mse": noisy_mse,
        "restored_mse": restored_mse,
        "noisy_psnr": _mse_to_psnr(noisy_mse),
        "restored_psnr": _mse_to_psnr(restored_mse),
        "ssim": ssim_sum / ssim_count,
        "available": True,
    }


def load_eval_indices(path: Path) -> list[int]:
    """读取并校验 IR-SDE 固定评估索引。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 {path}；请先运行 main_cifar.py 生成 fixed_eval.json"
        )
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    indices = payload.get("indices") if isinstance(payload, dict) else None
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"{path} 中缺少非空 'indices' 列表")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError(f"{path} 的 'indices' 必须全部是整数")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{path} 的 'indices' 包含重复索引")
    return indices


def _main_noise_ranges() -> dict[str, tuple[float, float]]:
    """返回与 main_cifar.py 一致的噪声强度范围。"""
    ranges = dict(NoisyCIFAR10Dataset.DEFAULT_NOISE_RANGES)
    ranges["gaussian"] = (0.03, 0.15)
    ranges["salt_pepper"] = (0.01, 0.08)
    return ranges


def parse_noise_selection(text: str) -> tuple[str, ...]:
    """解析 all、四位掩码、全名或唯一前缀形式的噪声选择。"""
    if not isinstance(text, str):
        raise ValueError("--noise-types 必须是字符串")
    value = text.strip()
    if not value:
        raise ValueError("--noise-types 不能为空")
    if value.lower() == "all":
        return NOISE_TYPES
    if len(value) == 4 and all(character in "01" for character in value):
        selected = tuple(kind for kind, bit in zip(NOISE_TYPES, value) if bit == "1")
        if not selected:
            raise ValueError("没有选中任何噪声")
        return selected
    tokens = [token.strip().lower() for token in value.split(",")]
    if any(not token for token in tokens):
        raise ValueError("--noise-types 含有空项")
    aliases = {
        "".join(part[0] for part in kind.split("_")): kind
        for kind in NOISE_TYPES
        if "_" in kind
    }
    selected_names = []
    for token in tokens:
        alias_candidates = [kind for alias, kind in aliases.items() if alias == token]
        candidates = alias_candidates or [
            kind for kind in NOISE_TYPES if kind.startswith(token)
        ]
        if len(candidates) > 1:
            raise ValueError(
                f"噪声前缀 '{token}' 有歧义，候选: {', '.join(candidates)}"
            )
        if not candidates:
            raise ValueError(
                f"未知噪声类型或前缀: '{token}'；可选: {', '.join(NOISE_TYPES)}"
            )
        if candidates[0] not in selected_names:
            selected_names.append(candidates[0])
    return tuple(kind for kind in NOISE_TYPES if kind in selected_names)


def resolve_noise_types(
    args: argparse.Namespace,
    ir_sde_output: Path,
) -> tuple[str, ...]:
    """确定显式选择或自动对齐 IR-SDE metrics.json 的噪声类型。"""
    if args.noise_types is not None:
        return parse_noise_selection(args.noise_types)
    metrics_path = ir_sde_output / "metrics.json"
    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        per_noise = metrics.get("per_noise", {})
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        return NOISE_TYPES
    if not isinstance(per_noise, dict):
        return NOISE_TYPES
    selected = tuple(kind for kind in NOISE_TYPES if kind in per_noise)
    if not selected:
        return NOISE_TYPES
    print(f"[auto] IR-SDE 只包含 {selected}，仅评估这些噪声", flush=True)
    return selected


def _build_selected_restorers(methods: list[str]) -> dict[str, Restorer]:
    """为请求的基础方法构建恢复器。"""
    return {method: METHOD_BUILDERS[method]() for method in methods}


def run_single_methods(
    methods: list[str],
    loaders: dict[str, Subset],
    device: torch.device,
    restorers: dict[str, Restorer],
    noise_types: tuple[str, ...],
) -> dict[str, dict[str, dict]]:
    """在所选噪声类型上运行各基础方法。"""
    per_method: dict[str, dict[str, dict]] = {method: {} for method in methods}
    for method in methods:
        restorer = restorers.get(method)
        for kind in noise_types:
            per_method[method][kind] = evaluate_method(
                loaders[kind],
                restorer,
                device,
                description=f"{method}/{kind}",
            )
    return per_method


def _make_cascade_restorer() -> Restorer:
    """构建中值滤波后接小波去噪的级联恢复器。"""
    median_restorer = _build_median_blur()
    wavelet_restorer = _build_wavelet(sigma=0.1)

    def restore(image: np.ndarray) -> np.ndarray:
        return wavelet_restorer(median_restorer(image))

    return restore


def run_cascade(loader: Subset, device: torch.device) -> dict:
    """在混合噪声上评估固定级联策略。"""
    result = evaluate_method(
        loader, _make_cascade_restorer(), device, "cascade_median_wavelet/mixed"
    )
    result["note"] = "cascade"
    return result


def run_tuned(loader: Subset, device: torch.device) -> dict[str, dict]:
    """在混合噪声上搜索 TV、NLM 和小波的最优参数。"""
    results: dict[str, dict] = {}
    for output_name, (parameter, values, builder) in TUNED_SPECS.items():
        best_result = None
        best_value = None
        best_psnr = -math.inf
        for value in values:
            description = f"{output_name}/{parameter}={value:g}/mixed"
            result = evaluate_method(
                loader,
                builder(**{parameter: value}),
                device,
                description,
            )
            print(
                f"[tuned] {output_name}: {parameter}={value:g}, "
                f"restored_psnr={result['restored_psnr']:.6f} dB",
                flush=True,
            )
            if result["restored_psnr"] > best_psnr:
                best_psnr = result["restored_psnr"]
                best_result = result
                best_value = value
        print(
            f"[tuned] selected {output_name}: {parameter}={best_value:g}, "
            f"restored_psnr={best_psnr:.6f} dB",
            flush=True,
        )
        best_result["note"] = "tuned"
        best_result["best_param"] = {parameter: best_value}
        results[output_name] = best_result
    return results


def _best_entry(per_method: dict, noise_type: str) -> tuple[str | None, float | None]:
    """返回指定噪声下 restored_psnr 最高的可用方法。"""
    best_method = None
    best_psnr = None
    for method, results in per_method.items():
        result = results.get(noise_type)
        if not result or not result.get("available"):
            continue
        psnr = result.get("restored_psnr")
        if psnr is None or not math.isfinite(psnr):
            continue
        if best_psnr is None or psnr > best_psnr:
            best_method = method
            best_psnr = float(psnr)
    return best_method, best_psnr


def compute_best_per_noise(
    per_method: dict,
    noise_types: tuple[str, ...],
) -> dict[str, dict]:
    """汇总每种所选噪声的最优方法和 PSNR。"""
    return {
        kind: dict(
            zip(
                ("method", "restored_psnr"),
                _best_entry(per_method, kind),
            )
        )
        for kind in noise_types
    }


def write_summary_csv(
    per_method: dict,
    output_path: Path,
    noise_types: tuple[str, ...],
) -> pd.DataFrame:
    """写出按噪声类型和方法排序的汇总 CSV。"""
    rows = []
    for method, results in per_method.items():
        for kind in noise_types:
            result = results.get(kind)
            if result is None:
                continue
            rows.append(
                {
                    "method": method,
                    "noise_type": kind,
                    "noisy_psnr": result.get("noisy_psnr"),
                    "restored_psnr": result.get("restored_psnr"),
                    "ssim": result.get("ssim"),
                    "available": result.get("available"),
                    "note": result.get("note", ""),
                }
            )
    frame = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    noise_order = {kind: index for index, kind in enumerate(noise_types)}
    frame["_noise_order"] = frame["noise_type"].map(noise_order)
    frame = frame.sort_values(
        ["_noise_order", "method"], kind="mergesort"
    ).drop(columns="_noise_order")
    frame.to_csv(output_path, index=False, float_format="%.6f")
    return frame


def _resolve_restorer(
    method: str,
    result: dict,
    base_restorers: dict[str, Restorer],
) -> Restorer:
    """将基础、级联或调参结果解析为恢复函数。"""
    if method == "cascade_median_wavelet":
        return _make_cascade_restorer()
    if method in TUNED_SPECS:
        parameter, _, builder = TUNED_SPECS[method]
        return builder(**{parameter: result["best_param"][parameter]})
    return base_restorers[method]


def save_compare_figures(
    loaders: dict[str, Subset],
    per_method: dict,
    best_per_noise: dict,
    base_restorers: dict[str, Restorer],
    output_dir: Path,
    noise_types: tuple[str, ...],
) -> list[Path]:
    """为每种所选噪声保存干净、含噪、最佳恢复结果的对比图。"""
    saved_paths = []
    for kind in noise_types:
        best = best_per_noise[kind]
        method = best["method"]
        if method is None:
            raise ValueError(f"{kind}: 没有可用的传统去噪结果，无法生成对比图")
        restorer = _resolve_restorer(method, per_method[method][kind], base_restorers)
        clean_images = []
        noisy_images = []
        restored_images = []
        for y, x0 in loaders[kind]:
            clean_images.append(x0.detach().cpu())
            noisy_images.append(y.detach().cpu())
            restored_images.append(
                torch.from_numpy(restorer(y.detach().cpu().numpy())).cpu()
            )
            if len(clean_images) == 8:
                break
        if len(clean_images) < 8:
            raise ValueError(
                f"{kind}: 对比图需要至少 8 个评估样本，当前只有 {len(clean_images)} 个"
            )
        tensor = torch.cat(
            [
                torch.stack(clean_images),
                torch.stack(noisy_images),
                torch.stack(restored_images),
            ],
            dim=0,
        )
        if tuple(tensor.shape) != (24, 3, 32, 32):
            raise RuntimeError(f"对比图张量形状异常: {tuple(tensor.shape)}")
        image_path = output_dir / f"compare_{kind}.png"
        save_image(tensor, image_path, nrow=8, padding=0)
        saved_paths.append(image_path)
    return saved_paths


def _load_ir_metrics(path: Path) -> dict:
    """读取 IR-SDE 指标文件。"""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def protocol_self_check(
    per_method: dict,
    ir_sde_metrics_path: Path,
    noise_types: tuple[str, ...],
) -> None:
    """打印协议对齐、逐方法检查及最优方法表。"""
    print("\n[协议自检] traditional noisy_psnr vs IR-SDE noisy_psnr", flush=True)
    ir_per_noise = _load_ir_metrics(ir_sde_metrics_path)["per_noise"]
    print(
        "noise_type | noisy_psnr(trad) | noisy_psnr(irsde) | diff | OK/WARN",
        flush=True,
    )
    for kind in noise_types:
        values = [
            float(result["noisy_psnr"])
            for results in per_method.values()
            if (result := results.get(kind))
            and result.get("noisy_psnr") is not None
            and math.isfinite(result["noisy_psnr"])
        ]
        if not values:
            print(f"{kind} | unavailable | unavailable | n/a | WARN", flush=True)
            continue
        reference = float(ir_per_noise[kind]["noisy_psnr"])
        differences = [value - reference for value in values]
        max_abs_diff = max(abs(value) for value in differences)
        status = "OK" if max_abs_diff < 0.01 else "WARN"
        print(
            f"{kind} | {values[0]:.6f} | {reference:.6f} | "
            f"{differences[0]:+.6f} | {status}",
            flush=True,
        )
        if status == "WARN":
            print(
                f"WARNING: {kind}: noisy_psnr 最大绝对差值 "
                f"{max_abs_diff:.6f} dB >= 0.01 dB，请检查协议对齐",
                flush=True,
            )

    print(
        "\nnoisy_psnr per-method check "
        "(max |diff| over available noise types):",
        flush=True,
    )
    print("method | max|diff| | OK/WARN", flush=True)
    for method, results in per_method.items():
        differences = [
            abs(float(result["noisy_psnr"]) - float(ir_per_noise[kind]["noisy_psnr"]))
            for kind, result in results.items()
            if kind in ir_per_noise and result.get("noisy_psnr") is not None
        ]
        if not differences:
            print(f"{method} | n/a | WARN", flush=True)
            continue
        max_diff = max(differences)
        status = "OK" if max_diff < 0.01 else "WARN"
        print(f"{method} | {max_diff:.6f} | {status}", flush=True)

    print("\nbest_per_noise:", flush=True)
    print("noise_type | best_method | restored_psnr", flush=True)
    for kind, best in compute_best_per_noise(per_method, noise_types).items():
        print(
            f"{kind} | {best['method']} | {best['restored_psnr']}",
            flush=True,
        )


def _normalize_methods(methods: list[str]) -> list[str]:
    """去重并保持命令行方法顺序。"""
    normalized = []
    for method in methods:
        if method not in normalized:
            normalized.append(method)
    return normalized


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--ir-sde-output", default="./outputs/cifar10/ir_sde")
    parser.add_argument(
        "--output-dir", default="./outputs/cifar10/traditional_single"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_NAMES,
        default=list(METHOD_NAMES),
        help="Methods to run; default: all six methods",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="single",
        help="Mixed-noise strategy; pure noise types always use single logic",
    )
    parser.add_argument(
        "--noise-types",
        default=None,
        help="all, a 4-bit mask, comma-separated names, or unique prefixes",
    )
    return parser.parse_args()


def main() -> None:
    """执行噪声选择、评估、汇总和可视化主流程。"""
    args = parse_args()
    methods = _normalize_methods(args.methods)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    ir_sde_output = Path(args.ir_sde_output)
    try:
        noise_types = resolve_noise_types(args, ir_sde_output)
    except ValueError as error:
        raise SystemExit(f"--noise-types 解析失败: {error}") from error
    fixed_eval_path = ir_sde_output / "fixed_eval.json"
    eval_indices = load_eval_indices(fixed_eval_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_ranges = _main_noise_ranges()
    base_restorers = _build_selected_restorers(
        [method for method in methods if method != "bm3d"]
    )
    if "bm3d" in methods:
        base_restorers["bm3d"] = add_bm3d_method()
    datasets = {
        kind: NoisyCIFAR10Dataset(
            root=args.data_root,
            train=False,
            seed=args.seed,
            samples_per_image=1,
            noise_probs={kind: 1.0},
            noise_ranges=noise_ranges,
        )
        for kind in noise_types
    }
    for kind, dataset in datasets.items():
        if any(index < 0 or index >= len(dataset) for index in eval_indices):
            raise IndexError(f"{kind} 数据集上存在越界的 fixed_eval 索引")
    loaders = {
        kind: Subset(dataset, eval_indices) for kind, dataset in datasets.items()
    }

    per_method = run_single_methods(
        methods,
        loaders,
        device,
        base_restorers,
        noise_types,
    )
    if args.strategy == "cascade":
        if "mixed" not in noise_types:
            print(
                "WARNING: strategy=cascade 需要 mixed 噪声；当前未选中，跳过 cascade",
                flush=True,
            )
        else:
            single_best_method, single_best_psnr = _best_entry(per_method, "mixed")
            cascade_result = run_cascade(loaders["mixed"], device)
            per_method["cascade_median_wavelet"] = {"mixed": cascade_result}
            if cascade_result["restored_psnr"] < single_best_psnr:
                print(
                    "WARNING: cascade restored_psnr "
                    f"{cascade_result['restored_psnr']:.6f} dB 低于 best single "
                    f"{single_best_method} {single_best_psnr:.6f} dB",
                    flush=True,
                )
    elif args.strategy == "tuned":
        if "mixed" not in noise_types:
            print(
                "WARNING: strategy=tuned 需要 mixed 噪声；当前未选中，跳过 tuned",
                flush=True,
            )
        else:
            for method, result in run_tuned(loaders["mixed"], device).items():
                per_method[method] = {"mixed": result}

    best_per_noise = compute_best_per_noise(per_method, noise_types)
    payload = {
        "config": {
            "strategy": args.strategy,
            "eval_indices_source": fixed_eval_path.as_posix(),
            "seed": args.seed,
            "eval_samples": len(eval_indices),
            "methods": methods,
            "bm3d_available": "bm3d" in methods,
            "noise_types": list(noise_types),
        },
        "best_per_noise": best_per_noise,
        "per_method": per_method,
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    summary_path = output_dir / "summary.csv"
    write_summary_csv(per_method, summary_path, noise_types)
    save_compare_figures(
        loaders,
        per_method,
        best_per_noise,
        base_restorers,
        output_dir,
        noise_types,
    )
    protocol_self_check(
        per_method,
        ir_sde_output / "metrics.json",
        noise_types,
    )
    print(f"\nSaved metrics to {metrics_path.resolve()}", flush=True)
    print(f"Saved summary to {summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
