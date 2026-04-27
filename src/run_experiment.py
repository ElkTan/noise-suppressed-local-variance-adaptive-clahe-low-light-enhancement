from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from degradation import batch_synthesize_directory, read_image_rgb, write_image_rgb
from baselines import (
    run_clahe,
    run_denoise_clahe,
    run_he,
    run_raw_variance_adaptive_clahe,
    run_raw_variance_adaptive_clahe_with_aux,
)
from metrics import compute_mean_y, compute_nar, compute_psnr, compute_ssim
from proposed import build_ablation_variants, run_ns_lva_clahe


def save_clip_map_image(clip_map: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = cv2.normalize(clip_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enlarged = cv2.resize(normalized, (256, 256), interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(enlarged, cv2.COLORMAP_VIRIDIS)
    if not cv2.imwrite(str(path), colored):
        raise IOError(f"Failed to save clip map image: {path}")


def save_crop_image(image: np.ndarray, path: str | Path, ratio: float = 0.35) -> None:
    height, width = image.shape[:2]
    crop_h = max(int(height * ratio), 32)
    crop_w = max(int(width * ratio), 32)
    y0 = max((height - crop_h) // 2, 0)
    x0 = max((width - crop_w) // 2, 0)
    crop = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    write_image_rgb(path, crop)


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file())


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    csv_path = ensure_parent(path)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, object]], metric_fields: list[str], group_key: str = "method") -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)

    summary: list[dict[str, object]] = []
    for method, items in grouped.items():
        result: dict[str, object] = {group_key: method}
        for field in metric_fields:
            values = [float(item[field]) for item in items]
            result[field] = round(float(np.mean(values)), 6)
        summary.append(result)
    return summary


def run_methods_on_synthetic(
    clean_dir: str | Path,
    synthetic_dir: str | Path,
    output_dir: str | Path,
) -> None:
    methods: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "low_light_input": lambda image: image,
        "he": run_he,
        "clahe": run_clahe,
        "denoise_clahe": run_denoise_clahe,
        "proposed": lambda image: run_ns_lva_clahe(image)[0],
    }
    rows: list[dict[str, object]] = []

    clean_images = {path.name: path for path in list_images(clean_dir)}
    synthetic_images = list_images(synthetic_dir)
    for low_path in synthetic_images:
        if low_path.name not in clean_images:
            continue
        gt = read_image_rgb(clean_images[low_path.name])
        low = read_image_rgb(low_path)
        for method_name, method_fn in methods.items():
            pred = method_fn(low)
            rows.append(
                {
                    "image": low_path.name,
                    "method": method_name,
                    "psnr": compute_psnr(gt, pred),
                    "ssim": compute_ssim(gt, pred, channel="rgb"),
                    "nar": compute_nar(low, pred),
                    "mean_y": compute_mean_y(pred),
                }
            )
            if method_name == "proposed":
                _, aux = run_ns_lva_clahe(low)
                save_clip_map_image(aux["clip_map"], Path(output_dir) / "visual" / f"{low_path.stem}_clip_map.png")

    metric_fields = ["psnr", "ssim", "nar", "mean_y"]
    write_csv(Path(output_dir) / "metrics" / "synthetic_main.csv", rows, ["image", "method", *metric_fields])
    summary = summarize_rows(rows, metric_fields)
    write_csv(Path(output_dir) / "metrics" / "synthetic_main_summary.csv", summary, ["method", *metric_fields])


def run_visual_on_real(real_dir: str | Path, output_dir: str | Path) -> None:
    methods: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "input": lambda image: image,
        "he": run_he,
        "clahe": run_clahe,
        "denoise_clahe": run_denoise_clahe,
        "proposed": lambda image: run_ns_lva_clahe(image)[0],
    }
    for image_path in list_images(real_dir):
        image = read_image_rgb(image_path)
        for method_name, method_fn in methods.items():
            result = method_fn(image)
            write_image_rgb(Path(output_dir) / "visual" / f"{image_path.stem}_{method_name}.png", result)
            save_crop_image(result, Path(output_dir) / "visual" / f"{image_path.stem}_{method_name}_crop.png")

        _, aux = run_ns_lva_clahe(image)
        save_clip_map_image(aux["clip_map"], Path(output_dir) / "visual" / f"{image_path.stem}_real_clip_map.png")


def run_ablation(clean_dir: str | Path, synthetic_dir: str | Path, output_dir: str | Path) -> None:
    rows: list[dict[str, object]] = []
    clean_images = {path.name: path for path in list_images(clean_dir)}
    synthetic_images = list_images(synthetic_dir)
    for low_path in synthetic_images:
        if low_path.name not in clean_images:
            continue
        gt = read_image_rgb(clean_images[low_path.name])
        low = read_image_rgb(low_path)

        fixed = run_clahe(low)
        raw, raw_aux = run_raw_variance_adaptive_clahe_with_aux(low)
        variants = build_ablation_variants(low)
        outputs = {
            "fixed_clahe": (fixed, None),
            "raw_variance_adaptive": (raw, raw_aux),
            **variants,
        }
        for method_name, (pred, aux) in outputs.items():
            rows.append(
                {
                    "image": low_path.name,
                    "method": method_name,
                    "ssim": compute_ssim(gt, pred, channel="rgb"),
                    "nar": compute_nar(low, pred),
                    "mean_y": compute_mean_y(pred),
                }
            )
            if aux is not None:
                save_clip_map_image(
                    aux["clip_map"],
                    Path(output_dir) / "ablation" / f"{low_path.stem}_{method_name}_clip_map.png",
                )

    metric_fields = ["ssim", "nar", "mean_y"]
    write_csv(Path(output_dir) / "ablation" / "ablation_metrics.csv", rows, ["image", "method", *metric_fields])
    summary = summarize_rows(rows, metric_fields)
    write_csv(Path(output_dir) / "ablation" / "ablation_summary.csv", summary, ["method", *metric_fields])


def run_sensitivity(clean_dir: str | Path, synthetic_dir: str | Path, output_dir: str | Path) -> None:
    clean_images = {path.name: path for path in list_images(clean_dir)}
    synthetic_images = list_images(synthetic_dir)
    beta_values = [0.5, 1.0, 2.0, 4.0]
    cmax_values = [2.0, 3.0, 4.0, 5.0]
    rows: list[dict[str, object]] = []

    for low_path in synthetic_images:
        if low_path.name not in clean_images:
            continue
        gt = read_image_rgb(clean_images[low_path.name])
        low = read_image_rgb(low_path)

        for beta in beta_values:
            pred, _ = run_ns_lva_clahe(low, beta=beta)
            rows.append(
                {
                    "image": low_path.name,
                    "sweep_type": "beta",
                    "value": beta,
                    "ssim": compute_ssim(gt, pred, channel="rgb"),
                    "nar": compute_nar(low, pred),
                }
            )
        for cmax in cmax_values:
            pred, _ = run_ns_lva_clahe(low, cmax=cmax)
            rows.append(
                {
                    "image": low_path.name,
                    "sweep_type": "cmax",
                    "value": cmax,
                    "ssim": compute_ssim(gt, pred, channel="rgb"),
                    "nar": compute_nar(low, pred),
                }
            )

    write_csv(
        Path(output_dir) / "metrics" / "sensitivity.csv",
        rows,
        ["image", "sweep_type", "value", "ssim", "nar"],
    )

    summary_rows: list[dict[str, object]] = []
    for sweep_type in ("beta", "cmax"):
        values = sorted({float(row["value"]) for row in rows if row["sweep_type"] == sweep_type})
        for value in values:
            matched = [row for row in rows if row["sweep_type"] == sweep_type and float(row["value"]) == value]
            summary_rows.append(
                {
                    "sweep_type": sweep_type,
                    "value": value,
                    "ssim": round(float(np.mean([float(row["ssim"]) for row in matched])), 6),
                    "nar": round(float(np.mean([float(row["nar"]) for row in matched])), 6),
                }
            )
    write_csv(Path(output_dir) / "metrics" / "sensitivity_summary.csv", summary_rows, ["sweep_type", "value", "ssim", "nar"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NS-LVA-CLAHE experiments.")
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--synthetic-dir", default="data/synthetic_low")
    parser.add_argument("--real-dir", default="data/real_low")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--generate-synthetic", action="store_true")
    parser.add_argument("--run-main", action="store_true")
    parser.add_argument("--run-real", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--run-sensitivity", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.generate_synthetic:
        batch_synthesize_directory(
            input_dir=args.clean_dir,
            output_dir=args.synthetic_dir,
            alpha=args.alpha,
            gamma=args.gamma,
            sigma=args.sigma,
        )
    if args.run_main:
        run_methods_on_synthetic(args.clean_dir, args.synthetic_dir, args.output_dir)
    if args.run_real:
        run_visual_on_real(args.real_dir, args.output_dir)
    if args.run_ablation:
        run_ablation(args.clean_dir, args.synthetic_dir, args.output_dir)
    if args.run_sensitivity:
        run_sensitivity(args.clean_dir, args.synthetic_dir, args.output_dir)


if __name__ == "__main__":
    main()
