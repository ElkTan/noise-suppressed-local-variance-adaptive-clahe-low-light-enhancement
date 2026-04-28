from __future__ import annotations

import argparse
import os
from pathlib import Path

import csv

_CACHE_DIR = Path(".cache") / "matplotlib"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path(".cache")).resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_metric_bars(rows: list[dict[str, str]], output_dir: str | Path) -> None:
    if not rows:
        return
    output_dir = ensure_dir(output_dir)
    methods = [row["method"] for row in rows]
    metric_specs = [
        ("psnr", "PSNR", "higher"),
        ("ssim", "SSIM", "higher"),
        ("nar", "NAR", "lower"),
        ("mean_y", "Mean Y", "higher"),
    ]
    for key, title, _direction in metric_specs:
        if key not in rows[0]:
            continue
        values = [float(row[key]) for row in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(methods, values, color="#2f6b7c")
        ax.set_title(f"{title} comparison")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(output_dir / f"main_{key}.png", dpi=200)
        plt.close(fig)


def plot_ablation(rows: list[dict[str, str]], output_dir: str | Path) -> None:
    if not rows:
        return
    output_dir = ensure_dir(output_dir)
    methods = [row["method"] for row in rows]
    for key, title in [("ssim", "SSIM"), ("nar", "NAR"), ("mean_y", "Mean Y")]:
        values = [float(row[key]) for row in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(methods, values, marker="o", linewidth=2, color="#c65d1e")
        ax.set_title(f"Ablation on {title}")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / f"ablation_{key}.png", dpi=200)
        plt.close(fig)


def plot_sensitivity(rows: list[dict[str, str]], output_dir: str | Path) -> None:
    if not rows:
        return
    output_dir = ensure_dir(output_dir)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["sweep_type"], []).append(row)

    for sweep_type, values in grouped.items():
        sorted_rows = sorted(values, key=lambda row: float(row["value"]))
        x = [float(row["value"]) for row in sorted_rows]
        for key, title in [("ssim", "SSIM"), ("nar", "NAR")]:
            y = [float(row[key]) for row in sorted_rows]
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.plot(x, y, marker="o", linewidth=2, color="#3d8c40")
            ax.set_title(f"{sweep_type} vs {title}")
            ax.set_xlabel(sweep_type)
            ax.set_ylabel(title)
            fig.tight_layout()
            fig.savefig(output_dir / f"sensitivity_{sweep_type}_{key}.png", dpi=200)
            plt.close(fig)


def plot_fusion(rows: list[dict[str, str]], output_dir: str | Path) -> None:
    if not rows:
        return
    output_dir = ensure_dir(output_dir)
    methods = [row["method"] for row in rows]
    for key, title in [("psnr", "PSNR"), ("ssim", "SSIM"), ("nar", "NAR"), ("mean_y", "Mean Y")]:
        if key not in rows[0]:
            continue
        values = [float(row[key]) for row in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(methods, values, color="#6a4c93")
        ax.set_title(f"Fusion Strategy Comparison on {title}")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / f"fusion_{key}.png", dpi=200)
        plt.close(fig)


def plot_real_no_reference(rows: list[dict[str, str]], output_dir: str | Path) -> None:
    if not rows:
        return
    output_dir = ensure_dir(output_dir)
    methods = [row["method"] for row in rows]
    for key, title in [("loe", "LOE"), ("niqe_like", "NIQE-like"), ("mean_y", "Mean Y")]:
        if key not in rows[0]:
            continue
        values = [float(row[key]) for row in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(methods, values, color="#a44a3f")
        ax.set_title(f"Real Low-Light No-Reference Comparison on {title}")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / f"real_no_reference_{key}.png", dpi=200)
        plt.close(fig)


def plot_synthetic_hard(rows: list[dict[str, str]], output_dir: str | Path) -> None:
    if not rows:
        return
    output_dir = ensure_dir(output_dir)
    methods = [row["method"] for row in rows]
    for key, title in [("psnr", "PSNR"), ("ssim", "SSIM"), ("nar", "NAR"), ("mean_y", "Mean Y")]:
        if key not in rows[0]:
            continue
        values = [float(row[key]) for row in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(methods, values, color="#486581")
        ax.set_title(f"Hard Degradation Comparison on {title}")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / f"synthetic_hard_{key}.png", dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures from experiment CSV files.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figure-dir", default="paper/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figure_dir = Path(args.figure_dir)

    plot_metric_bars(
        read_csv_rows(results_dir / "metrics" / "synthetic_main_summary.csv"),
        figure_dir,
    )
    plot_ablation(
        read_csv_rows(results_dir / "ablation" / "ablation_summary.csv"),
        figure_dir,
    )
    plot_sensitivity(
        read_csv_rows(results_dir / "metrics" / "sensitivity_summary.csv"),
        figure_dir,
    )
    plot_fusion(
        read_csv_rows(results_dir / "fusion" / "fusion_summary.csv"),
        figure_dir,
    )
    plot_real_no_reference(
        read_csv_rows(results_dir / "metrics" / "real_no_reference_summary.csv"),
        figure_dir,
    )
    plot_synthetic_hard(
        read_csv_rows(results_dir / "metrics" / "synthetic_hard_summary.csv"),
        figure_dir,
    )


if __name__ == "__main__":
    main()
