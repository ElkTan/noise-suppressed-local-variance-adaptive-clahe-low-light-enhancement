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
from metrics import compute_loe, compute_mean_y, compute_nar, compute_niqe_like, compute_psnr, compute_ssim, fit_naturalness_model
from proposed import AdaptiveClaheConfig, build_ablation_variants, run_adaptive_clahe_variant, run_ns_lva_clahe


def save_clip_map_image(clip_map: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = cv2.normalize(clip_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enlarged = cv2.resize(normalized, (256, 256), interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(enlarged, cv2.COLORMAP_VIRIDIS)
    if not cv2.imwrite(str(path), colored):
        raise IOError(f"Failed to save clip map image: {path}")


def find_focus_crop_box(reference_image: np.ndarray, ratio: float = 0.35) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = gray.shape
    crop_h = max(int(height * ratio), 64)
    crop_w = max(int(width * ratio), 64)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)

    best_score = None
    best_box = (max((width - crop_w) // 2, 0), max((height - crop_h) // 2, 0), crop_w, crop_h)
    step_y = max(crop_h // 4, 16)
    step_x = max(crop_w // 4, 16)
    for y0 in range(0, max(height - crop_h, 0) + 1, step_y):
        for x0 in range(0, max(width - crop_w, 0) + 1, step_x):
            patch = gray[y0 : y0 + crop_h, x0 : x0 + crop_w]
            patch_grad = gradient[y0 : y0 + crop_h, x0 : x0 + crop_w]
            darkness = 255.0 - float(np.mean(patch))
            texture = float(np.mean(patch_grad))
            score = darkness + 2.5 * texture
            if best_score is None or score > best_score:
                best_score = score
                best_box = (x0, y0, crop_w, crop_h)
    return best_box


def save_crop_image(image: np.ndarray, path: str | Path, ratio: float = 0.35, crop_box: tuple[int, int, int, int] | None = None) -> None:
    height, width = image.shape[:2]
    if crop_box is None:
        crop_w = max(int(width * ratio), 32)
        crop_h = max(int(height * ratio), 32)
        x0 = max((width - crop_w) // 2, 0)
        y0 = max((height - crop_h) // 2, 0)
    else:
        x0, y0, crop_w, crop_h = crop_box
    crop = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    write_image_rgb(path, crop)


def save_comparison_strip(images: list[tuple[str, np.ndarray]], path: str | Path) -> None:
    if not images:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    labeled = []
    target_h = min(img.shape[0] for _, img in images)
    target_w = min(img.shape[1] for _, img in images)
    for label, image in images:
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((target_h + 36, target_w, 3), 255, dtype=np.uint8)
        canvas[36:] = resized
        cv2.putText(canvas, label, (8, 24), font, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
        labeled.append(canvas)
    strip = np.concatenate(labeled, axis=1)
    write_image_rgb(path, strip)


def export_display_version(
    input_path: str | Path,
    output_path: str | Path,
    gamma: float = 0.7,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> None:
    image = read_image_rgb(input_path)
    display = _make_display_image(image, gamma=gamma, percentile_low=percentile_low, percentile_high=percentile_high)
    write_image_rgb(output_path, display)


def _make_display_image(
    image: np.ndarray,
    gamma: float = 0.7,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    image_f = image.astype(np.float32)
    value = np.max(image_f, axis=2)
    lo = float(np.percentile(value, percentile_low))
    hi = float(np.percentile(value, percentile_high))
    scale = max(hi - lo, 1.0)
    stretched = np.clip((image_f - lo) / scale, 0.0, 1.0)
    display = np.power(stretched, gamma)
    return np.clip(display * 255.0, 0, 255).astype(np.uint8)


def export_paper_crop(
    input_path: str | Path,
    output_path: str | Path,
    crop_box: tuple[int, int, int, int],
    target_size: tuple[int, int] = (480, 360),
    display: bool = False,
    gamma: float = 0.7,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> None:
    image = read_image_rgb(input_path)
    x0, y0, crop_w, crop_h = crop_box
    crop = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)
    if display:
        resized = _make_display_image(
            resized,
            gamma=gamma,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
    write_image_rgb(output_path, resized)


def export_paper_crop_from_array(
    image: np.ndarray,
    output_path: str | Path,
    crop_box: tuple[int, int, int, int],
    target_size: tuple[int, int] = (480, 360),
    display: bool = False,
    gamma: float = 0.7,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> None:
    x0, y0, crop_w, crop_h = crop_box
    crop = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)
    if display:
        resized = _make_display_image(
            resized,
            gamma=gamma,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
    write_image_rgb(output_path, resized)


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


def _main_methods() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "low_light_input": lambda image: image,
        "he": run_he,
        "clahe": run_clahe,
        "denoise_clahe": run_denoise_clahe,
        "proposed": lambda image: run_ns_lva_clahe(image)[0],
    }


def _real_methods() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "input": lambda image: image,
        "he": run_he,
        "clahe": run_clahe,
        "denoise_clahe": run_denoise_clahe,
        "proposed": lambda image: run_ns_lva_clahe(image, cmax=6.0, eta=0.3)[0],
    }


def run_methods_on_synthetic(
    clean_dir: str | Path,
    synthetic_dir: str | Path,
    output_dir: str | Path,
    metrics_filename: str = "synthetic_main.csv",
    summary_filename: str = "synthetic_main_summary.csv",
    clip_prefix: str = "",
) -> None:
    methods = _main_methods()
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
                clip_name = f"{clip_prefix}{low_path.stem}_clip_map.png" if clip_prefix else f"{low_path.stem}_clip_map.png"
                save_clip_map_image(aux["clip_map"], Path(output_dir) / "visual" / clip_name)

    metric_fields = ["psnr", "ssim", "nar", "mean_y"]
    write_csv(Path(output_dir) / "metrics" / metrics_filename, rows, ["image", "method", *metric_fields])
    summary = summarize_rows(rows, metric_fields)
    write_csv(Path(output_dir) / "metrics" / summary_filename, summary, ["method", *metric_fields])


def run_visual_on_real(real_dir: str | Path, output_dir: str | Path) -> None:
    methods = _real_methods()
    for image_path in list_images(real_dir):
        image = read_image_rgb(image_path)
        crop_box = find_focus_crop_box(image, ratio=0.3)
        comparison_images: list[tuple[str, np.ndarray]] = []
        for method_name, method_fn in methods.items():
            result = method_fn(image)
            comparison_images.append((method_name, result))
            write_image_rgb(Path(output_dir) / "visual" / f"{image_path.stem}_{method_name}.png", result)
            save_crop_image(
                result,
                Path(output_dir) / "visual" / f"{image_path.stem}_{method_name}_crop.png",
                crop_box=crop_box,
            )

        _, aux = run_ns_lva_clahe(image, cmax=6.0, eta=0.3)
        save_clip_map_image(aux["clip_map"], Path(output_dir) / "visual" / f"{image_path.stem}_real_clip_map.png")
        save_comparison_strip(comparison_images, Path(output_dir) / "visual" / f"{image_path.stem}_comparison_strip.png")


def run_ablation(clean_dir: str | Path, synthetic_dir: str | Path, output_dir: str | Path) -> None:
    rows: list[dict[str, object]] = []
    clean_images = {path.name: path for path in list_images(clean_dir)}
    synthetic_images = list_images(synthetic_dir)
    for low_path in synthetic_images:
        if low_path.name not in clean_images:
            continue
        gt = read_image_rgb(clean_images[low_path.name])
        low = read_image_rgb(low_path)
        crop_box = find_focus_crop_box(low, ratio=0.28)

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
            if low_path.stem in {"kodim01", "kodim05", "kodim13"}:
                save_crop_image(
                    pred,
                    Path(output_dir) / "ablation" / f"{low_path.stem}_{method_name}_crop.png",
                    crop_box=crop_box,
                )
        if low_path.stem in {"kodim01", "kodim05", "kodim13"}:
            save_comparison_strip(
                [(name, pred) for name, (pred, _aux) in outputs.items()],
                Path(output_dir) / "ablation" / f"{low_path.stem}_ablation_strip.png",
            )

    metric_fields = ["ssim", "nar", "mean_y"]
    write_csv(Path(output_dir) / "ablation" / "ablation_metrics.csv", rows, ["image", "method", *metric_fields])
    summary = summarize_rows(rows, metric_fields)
    write_csv(Path(output_dir) / "ablation" / "ablation_summary.csv", summary, ["method", *metric_fields])


def run_fusion_ablation(clean_dir: str | Path, synthetic_dir: str | Path, output_dir: str | Path) -> None:
    rows: list[dict[str, object]] = []
    clean_images = {path.name: path for path in list_images(clean_dir)}
    synthetic_images = list_images(synthetic_dir)
    variants = {
        "hard_stitching": AdaptiveClaheConfig(
            cmax=5.0,
            beta=1.0,
            lambda_=0.5,
            smooth_clip_map=False,
            blend_tiles=False,
            chroma_compensation=False,
        ),
        "clipmap_smoothing_only": AdaptiveClaheConfig(
            cmax=5.0,
            beta=1.0,
            lambda_=0.5,
            smooth_clip_map=True,
            blend_tiles=False,
            chroma_compensation=False,
        ),
        "bilinear_blending": AdaptiveClaheConfig(
            cmax=5.0,
            beta=1.0,
            lambda_=0.5,
            smooth_clip_map=True,
            blend_tiles=True,
            chroma_compensation=False,
        ),
    }

    for low_path in synthetic_images:
        if low_path.name not in clean_images:
            continue
        gt = read_image_rgb(clean_images[low_path.name])
        low = read_image_rgb(low_path)
        crop_box = find_focus_crop_box(low, ratio=0.28)
        outputs: list[tuple[str, np.ndarray, dict[str, object]]] = []
        for method_name, config in variants.items():
            pred, aux = run_adaptive_clahe_variant(low, config)
            outputs.append((method_name, pred, aux))
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
            if low_path.stem in {"kodim01", "kodim05", "kodim13"}:
                save_crop_image(
                    pred,
                    Path(output_dir) / "fusion" / f"{low_path.stem}_{method_name}_crop.png",
                    crop_box=crop_box,
                )
                save_clip_map_image(aux["clip_map"], Path(output_dir) / "fusion" / f"{low_path.stem}_{method_name}_clip_map.png")
        if low_path.stem in {"kodim01", "kodim05", "kodim13"}:
            save_comparison_strip(
                [(name, pred) for name, pred, _aux in outputs],
                Path(output_dir) / "fusion" / f"{low_path.stem}_fusion_strip.png",
            )

    metric_fields = ["psnr", "ssim", "nar", "mean_y"]
    write_csv(Path(output_dir) / "fusion" / "fusion_metrics.csv", rows, ["image", "method", *metric_fields])
    summary = summarize_rows(rows, metric_fields)
    write_csv(Path(output_dir) / "fusion" / "fusion_summary.csv", summary, ["method", *metric_fields])


def export_paper_display(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    pairs = [
        (
            output_dir / "visual" / "1d787b65ee7c7de9c9f254e4f90f04e8_comparison_strip.png",
            output_dir / "visual" / "1d787b65ee7c7de9c9f254e4f90f04e8_comparison_strip_display.png",
        ),
        (
            output_dir / "visual" / "f5e0d8ea21bae5152af7abcf44ab6f38_comparison_strip.png",
            output_dir / "visual" / "f5e0d8ea21bae5152af7abcf44ab6f38_comparison_strip_display.png",
        ),
        (
            output_dir / "ablation" / "kodim13_ablation_strip.png",
            output_dir / "ablation" / "kodim13_ablation_strip_display.png",
        ),
        (
            output_dir / "fusion" / "kodim01_fusion_strip.png",
            output_dir / "fusion" / "kodim01_fusion_strip_display.png",
        ),
    ]
    for src, dst in pairs:
        export_display_version(src, dst)


def export_paper_crops(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    figures_dir = Path("paper/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    real_source = output_dir / "visual" / "7f79ba01baf9bbd7215861656b73271c_proposed.png"
    real_crop_box = (300, 360, 480, 360)
    export_paper_crop(real_source, figures_dir / "fig4_real_crop_raw.png", real_crop_box, target_size=(480, 360), display=False)
    export_paper_crop(real_source, figures_dir / "fig4_real_crop_display.png", real_crop_box, target_size=(480, 360), display=True)

    synth_input = read_image_rgb("data/synthetic_low/kodim13.png")
    synth_output, _ = run_ns_lva_clahe(synth_input)
    synth_crop_box = (120, 120, 480, 360)
    export_paper_crop_from_array(synth_output, figures_dir / "fig4_synth_crop_raw.png", synth_crop_box, target_size=(480, 360), display=False)
    export_paper_crop_from_array(synth_output, figures_dir / "fig4_synth_crop_display.png", synth_crop_box, target_size=(480, 360), display=True)


def run_real_no_reference(real_dir: str | Path, clean_dir: str | Path, output_dir: str | Path) -> None:
    methods = _real_methods()
    clean_paths = list_images(clean_dir)
    model_mean, model_cov = fit_naturalness_model(clean_paths)
    rows: list[dict[str, object]] = []
    for image_path in list_images(real_dir):
        image = read_image_rgb(image_path)
        for method_name, method_fn in methods.items():
            pred = method_fn(image)
            rows.append(
                {
                    "image": image_path.name,
                    "method": method_name,
                    "loe": compute_loe(image, pred),
                    "niqe_like": compute_niqe_like(pred, model_mean, model_cov),
                    "mean_y": compute_mean_y(pred),
                }
            )
    metric_fields = ["loe", "niqe_like", "mean_y"]
    write_csv(Path(output_dir) / "metrics" / "real_no_reference.csv", rows, ["image", "method", *metric_fields])
    summary = summarize_rows(rows, metric_fields)
    write_csv(Path(output_dir) / "metrics" / "real_no_reference_summary.csv", summary, ["method", *metric_fields])


def run_hard_synthetic(clean_dir: str | Path, synthetic_dir: str | Path, output_dir: str | Path) -> None:
    batch_synthesize_directory(
        input_dir=clean_dir,
        output_dir=synthetic_dir,
        alpha=0.2,
        gamma=2.2,
        sigma=0.03,
    )
    run_methods_on_synthetic(
        clean_dir=clean_dir,
        synthetic_dir=synthetic_dir,
        output_dir=output_dir,
        metrics_filename="synthetic_hard.csv",
        summary_filename="synthetic_hard_summary.csv",
        clip_prefix="hard_",
    )


def run_sensitivity(clean_dir: str | Path, synthetic_dir: str | Path, output_dir: str | Path) -> None:
    clean_images = {path.name: path for path in list_images(clean_dir)}
    synthetic_images = list_images(synthetic_dir)
    beta_values = [0.5, 1.0, 2.0, 4.0]
    cmax_values = [4.0, 8.0, 12.0, 16.0]
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
    parser.add_argument("--run-fusion", action="store_true")
    parser.add_argument("--run-real-no-reference", action="store_true")
    parser.add_argument("--run-hard-synthetic", action="store_true")
    parser.add_argument("--run-sensitivity", action="store_true")
    parser.add_argument("--export-paper-display", action="store_true")
    parser.add_argument("--export-paper-crops", action="store_true")
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
    if args.run_fusion:
        run_fusion_ablation(args.clean_dir, args.synthetic_dir, args.output_dir)
    if args.run_real_no_reference:
        run_real_no_reference(args.real_dir, args.clean_dir, args.output_dir)
    if args.run_hard_synthetic:
        run_hard_synthetic(args.clean_dir, "data/synthetic_low_hard", args.output_dir)
    if args.run_sensitivity:
        run_sensitivity(args.clean_dir, args.synthetic_dir, args.output_dir)
    if args.export_paper_display:
        export_paper_display(args.output_dir)
    if args.export_paper_crops:
        export_paper_crops(args.output_dir)


if __name__ == "__main__":
    main()
