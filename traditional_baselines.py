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

from ir_sde.data import NoisyMNISTDataset

cv2_module.setNumThreads(1)

NOISE_TYPES = ("gaussian", "salt_pepper", "speckle", "mixed")  # 定义噪声类型
METHOD_NAMES = ("gaussian_blur", "median_blur", "wavelet", "nlm", "tv", "bm3d")  # 方法名
STRATEGIES = ("single", "cascade", "tuned")  # 策略，仅对 mixed 生效
SUMMARY_COLUMNS = (
    "method",
    "noise_type",
    "noisy_psnr",
    "restored_psnr",
    "ssim",
    "available",
    "note",
)  # CSV 列名
Restorer = Callable[[np.ndarray], np.ndarray]


# -------------------通用工具函数----------------
def _to_float32_2d(image: np.ndarray) -> np.ndarray:
    """将输入图像转换为连续的 float32 二维数组。"""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"Expected a grayscale image, got shape {array.shape}")
        array = array[0]
    elif array.ndim != 2:
        raise ValueError(f"Expected [H, W] or [1, H, W], got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Input image contains non-finite values")
    return np.ascontiguousarray(array)


def _match_input_shape(output: np.ndarray, input_shape: tuple[int, ...]) -> np.ndarray:
    """将去噪结果裁剪到 [0,1] 并恢复输入形状。"""
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != input_shape:
        array = array.reshape(input_shape)
    if not np.isfinite(array).all():
        raise RuntimeError("Denoiser produced non-finite values")
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)


# -------------------去噪方法----------------
def _build_gaussian_blur() -> Restorer:
    """构建固定参数的 OpenCV 高斯模糊去噪器。"""

    def restore(image: np.ndarray) -> np.ndarray:
        """对单张图像执行高斯模糊。"""
        input_shape = np.asarray(image).shape
        source = _to_float32_2d(image)
        # OpenCV uint8 quantization introduces at most half a level of error.
        quantized = np.rint(source * 255.0).clip(0, 255).astype(np.uint8)
        filtered = cv2_module.GaussianBlur(quantized, ksize=(3, 3), sigmaX=0)
        result = filtered.astype(np.float32) / 255.0
        return _match_input_shape(result, input_shape)

    return restore


def _build_median_blur() -> Restorer:
    """构建固定参数的 OpenCV 中值滤波去噪器。"""

    def restore(image: np.ndarray) -> np.ndarray:
        """对单张图像执行中值滤波。"""
        input_shape = np.asarray(image).shape
        source = _to_float32_2d(image)
        # OpenCV uint8 quantization introduces at most half a level of error.
        quantized = np.rint(source * 255.0).clip(0, 255).astype(np.uint8)
        filtered = cv2_module.medianBlur(quantized, ksize=3)
        result = filtered.astype(np.float32) / 255.0
        return _match_input_shape(result, input_shape)

    return restore


def _build_wavelet(sigma: float = 0.1) -> Restorer:
    """构建固定参数的小波软阈值去噪器。"""

    def restore(image: np.ndarray) -> np.ndarray:
        """对单张图像执行小波去噪。"""
        input_shape = np.asarray(image).shape
        source = _to_float32_2d(image)
        filtered = denoise_wavelet(
            source,
            sigma=sigma,
            mode="soft",
            wavelet="db1",
            channel_axis=None,
        )
        return _match_input_shape(filtered, input_shape)

    return restore


def _build_nlm(h: float = 0.1) -> Restorer:
    """构建固定参数的非局部均值去噪器。"""

    def restore(image: np.ndarray) -> np.ndarray:
        """对单张图像执行非局部均值去噪。"""
        input_shape = np.asarray(image).shape
        source = _to_float32_2d(image)
        filtered = denoise_nl_means(
            source,
            h=h,
            patch_size=3,
            patch_distance=5,
            fast_mode=True,
        )
        return _match_input_shape(filtered, input_shape)

    return restore


def _build_tv(weight: float = 0.1) -> Restorer:
    """构建固定参数的全变分去噪器。"""

    def restore(image: np.ndarray) -> np.ndarray:
        """对单张图像执行全变分去噪。"""
        input_shape = np.asarray(image).shape
        source = _to_float32_2d(image)
        filtered = denoise_tv_chambolle(source, weight=weight)
        return _match_input_shape(filtered, input_shape)

    return restore


def add_bm3d_method() -> Restorer:
    """构建 BM3D 去噪器。"""

    def restore(image: np.ndarray) -> np.ndarray:
        """对单张图像执行 BM3D 去噪。"""
        input_shape = np.asarray(image).shape
        source = _to_float32_2d(image)
        filtered = bm3d(
            source,
            sigma_psd=0.1,
            stage_arg=BM3DStages.ALL_STAGES,
        )
        if isinstance(filtered, tuple):
            filtered = filtered[0]
        return _match_input_shape(filtered, input_shape)

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

#-------------评估指标----------------------------
def _mse_to_psnr(mse: float) -> float:
    """将数据集级 MSE 转换为 PSNR。"""
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def _mean_ssim(clean: np.ndarray, restored: np.ndarray) -> float:
    """计算单张灰度图的 SSIM。"""
    clean_2d = _to_float32_2d(clean)
    restored_2d = _to_float32_2d(restored)
    return float(structural_similarity(clean_2d, restored_2d, data_range=1.0))


def evaluate_method(
    loader: Subset,
    restorer: Restorer | None,
    device: torch.device,
    description: str,
) -> dict:
    """在固定噪声子集上评估一个恢复函数。"""
    noisy_error = 0.0
    restored_error = 0.0
    pixels = 0
    ssim_sum = 0.0
    ssim_count = 0

    progress = tqdm(loader, desc=description, unit="sample", dynamic_ncols=True)
    for y, x0 in progress:
        y = y.to(device)
        x0 = x0.to(device)
        clean = x0.detach().cpu().numpy()

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
        ssim_sum += _mean_ssim(clean, restored_np)
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
            f"找不到 {path}；请先运行 main.py 生成 fixed_eval.json"
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
    """返回与 main.py 一致的噪声强度范围。"""
    ranges = dict(NoisyMNISTDataset.DEFAULT_NOISE_RANGES)
    ranges["gaussian"] = (0.05, 0.20)
    ranges["salt_pepper"] = (0.01, 0.08)
    return ranges


def parse_noise_selection(text: str) -> tuple[str, ...]:
    """解析全部、位掩码、全名或唯一前缀形式的噪声选择。"""
    if not isinstance(text, str):
        raise ValueError("--noise-types 必须是字符串")

    value = text.strip()
    if not value:
        raise ValueError("--noise-types 不能为空")

    if value.lower() == "all":
        return NOISE_TYPES

    if len(value) == 4 and all(character in "01" for character in value):
        selected = tuple(
            kind for kind, bit in zip(NOISE_TYPES, value) if bit == "1"
        )
        if not selected:
            raise ValueError("没有选中任何噪声")
        return selected

    tokens = [token.strip().lower() for token in value.split(",")]
    if any(not token for token in tokens):
        raise ValueError("--noise-types 含有空项")

    selected_names = []
    multiword_aliases = {
        "".join(part[0] for part in kind.split("_")): kind
        for kind in NOISE_TYPES
        if "_" in kind
    }
    for token in tokens:
        # Multi-word names accept their initials (salt_pepper -> sp).
        alias_candidates = [
            kind for kind in NOISE_TYPES if multiword_aliases.get(token) == kind
        ]
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

    if not selected_names:
        raise ValueError("没有选中任何噪声")
    return tuple(kind for kind in NOISE_TYPES if kind in selected_names)


def resolve_noise_types(
    args: argparse.Namespace,
    ir_sde_output: Path,
) -> tuple[str, ...]:
    """确定显式选择或自动对齐 IR-SDE 的噪声类型。"""
    selection = getattr(args, "noise_types", None)
    if selection is not None:
        return parse_noise_selection(selection)

    metrics_path = Path(ir_sde_output) / "metrics.json"
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

    print(
        f"[auto] IR-SDE 只包含 {selected}，仅评估这些噪声",
        flush=True,
    )
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
        """对单张图像执行中值滤波后接小波去噪。"""
        return wavelet_restorer(median_restorer(image))

    return restore


def run_cascade(loader: Subset, device: torch.device) -> dict:
    """在混合噪声上评估固定级联策略。"""
    restorer = _make_cascade_restorer()
    result = evaluate_method(loader, restorer, device, "cascade_median_wavelet/mixed")
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
            restorer = builder(**{parameter: value})
            result = evaluate_method(loader, restorer, device, description)
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
    best = {}
    for kind in noise_types:
        method, psnr = _best_entry(per_method, kind)
        best[kind] = {"method": method, "restored_psnr": psnr}
    return best

#---------------输出与可视化---------------------
def write_summary_csv(
    per_method: dict,
    output_path: Path,
    noise_types: tuple[str, ...],
) -> pd.DataFrame:
    """写出按噪声类型和方法排序的汇总 CSV。"""
    rows = []
    for method in per_method:
        for kind in noise_types:
            result = per_method[method].get(kind)
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
        value = result["best_param"][parameter]
        return builder(**{parameter: value})
    return base_restorers[method]


def save_compare_figures(
    loaders: dict[str, Subset],
    per_method: dict,
    best_per_noise: dict,
    base_restorers: dict[str, Restorer],
    output_dir: Path,
    noise_types: tuple[str, ...],
) -> list[Path]:
    """为每种所选噪声保存 3x8 对比图。"""
    saved_paths = []
    for kind in noise_types:
        best = best_per_noise[kind]
        method = best["method"]
        result = per_method[method][kind]
        restorer = _resolve_restorer(method, result, base_restorers)

        clean_images = []
        noisy_images = []
        restored_images = []
        for y, x0 in loaders[kind]:
            clean_images.append(x0.detach().cpu())
            noisy_images.append(y.detach().cpu())
            restored_np = restorer(y.detach().cpu().numpy())
            restored_images.append(torch.from_numpy(restored_np).cpu())
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
        if tuple(tensor.shape) != (24, 1, 28, 28):
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
    ir_metrics = _load_ir_metrics(ir_sde_metrics_path)
    ir_per_noise = ir_metrics["per_noise"]

    print(
        "noise_type | noisy_psnr(trad) | noisy_psnr(irsde) | diff | OK/WARN",
        flush=True,
    )
    for kind in noise_types:
        values = []
        for method in per_method:
            result = per_method[method].get(kind)
            if not result:
                continue
            psnr = result.get("noisy_psnr")
            if psnr is not None and math.isfinite(psnr):
                values.append(float(psnr))

        reference = float(ir_per_noise[kind]["noisy_psnr"])
        traditional = values[0]
        differences = [value - reference for value in values]
        max_abs_diff = max(abs(value) for value in differences)
        status = "OK" if max_abs_diff < 0.01 else "WARN"
        print(
            f"{kind} | {traditional:.6f} | {reference:.6f} | "
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
    for method in per_method:
        differences = [
            abs(float(result["noisy_psnr"]) - float(ir_per_noise[kind]["noisy_psnr"]))
            for kind, result in per_method[method].items()
        ]
        max_diff = max(differences)
        status = "OK" if max_diff < 0.01 else "WARN"
        print(f"{method} | {max_diff:.6f} | {status}", flush=True)

    print("\nbest_per_noise:", flush=True)
    print("noise_type | best_method | restored_psnr", flush=True)
    best_per_noise = compute_best_per_noise(per_method, noise_types)
    for kind in noise_types:
        best = best_per_noise[kind]
        print(
            f"{kind} | {best['method']} | {best['restored_psnr']:.6f}",
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
    parser.add_argument("--ir-sde-output", default="./outputs/ir_sde")
    parser.add_argument("--output-dir", default="./outputs/traditional")
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

    # Fixed global seeds mirror the deterministic IR-SDE setup.
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

    bm3d_restorer = add_bm3d_method()
    bm3d_available = True
    base_restorers = _build_selected_restorers(
        [method for method in methods if method != "bm3d"]
    )
    if "bm3d" in methods:
        base_restorers["bm3d"] = bm3d_restorer

    # Each single-noise Dataset is constructed exactly as in main.py's
    # per-noise evaluation, including the same seed and samples_per_image.
    datasets = {
        kind: NoisyMNISTDataset(
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
        kind: Subset(dataset, eval_indices)
        for kind, dataset in datasets.items()
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
            single_best_mixed_method, single_best_mixed_psnr = _best_entry(
                per_method, "mixed"
            )
            cascade_result = run_cascade(loaders["mixed"], device)
            per_method["cascade_median_wavelet"] = {"mixed": cascade_result}
            if cascade_result["restored_psnr"] < single_best_mixed_psnr:
                print(
                    "WARNING: cascade restored_psnr "
                    f"{cascade_result['restored_psnr']:.6f} dB 低于 best single "
                    f"{single_best_mixed_method} {single_best_mixed_psnr:.6f} dB",
                    flush=True,
                )
    elif args.strategy == "tuned":
        if "mixed" not in noise_types:
            print(
                "WARNING: strategy=tuned 需要 mixed 噪声；当前未选中，跳过 tuned",
                flush=True,
            )
        else:
            tuned_results = run_tuned(loaders["mixed"], device)
            for method, result in tuned_results.items():
                per_method[method] = {"mixed": result}

    best_per_noise = compute_best_per_noise(per_method, noise_types)
    payload = {
        "config": {
            "strategy": args.strategy,
            "eval_indices_source": fixed_eval_path.as_posix(),
            "seed": args.seed,
            "eval_samples": len(eval_indices),
            "methods": methods,
            "bm3d_available": bm3d_available,
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
