from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from degradation import batch_synthesize_directory, read_image_rgb


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES)


def inspect_directory(directory: str | Path, split_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image_path in list_images(directory):
        image = read_image_rgb(image_path)
        height, width = image.shape[:2]
        rows.append(
            {
                "split": split_name,
                "filename": image_path.name,
                "width": width,
                "height": height,
                "channels": image.shape[2] if image.ndim == 3 else 1,
            }
        )
    return rows


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_pairing(clean_dir: str | Path, synthetic_dir: str | Path) -> list[str]:
    clean_names = {path.name for path in list_images(clean_dir)}
    synthetic_names = {path.name for path in list_images(synthetic_dir)}
    missing = sorted(clean_names - synthetic_names)
    return missing


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for row in rows:
        split = str(row["split"])
        info = summary.setdefault(
            split,
            {"split": split, "count": 0, "min_width": None, "max_width": None, "min_height": None, "max_height": None},
        )
        width = int(row["width"])
        height = int(row["height"])
        info["count"] = int(info["count"]) + 1
        info["min_width"] = width if info["min_width"] is None else min(int(info["min_width"]), width)
        info["max_width"] = width if info["max_width"] is None else max(int(info["max_width"]), width)
        info["min_height"] = height if info["min_height"] is None else min(int(info["min_height"]), height)
        info["max_height"] = height if info["max_height"] is None else max(int(info["max_height"]), height)
    return list(summary.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare image data for NS-LVA-CLAHE experiments.")
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--real-dir", default="data/real_low")
    parser.add_argument("--synthetic-dir", default="data/synthetic_low")
    parser.add_argument("--manifest-out", default="paper/figures/data_manifest.csv")
    parser.add_argument("--summary-out", default="paper/figures/data_summary.csv")
    parser.add_argument("--generate-synthetic", action="store_true")
    parser.add_argument("--check-pairs", action="store_true")
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

    manifest_rows = [
        *inspect_directory(args.clean_dir, "clean"),
        *inspect_directory(args.synthetic_dir, "synthetic_low"),
        *inspect_directory(args.real_dir, "real_low"),
    ]
    write_csv(
        args.manifest_out,
        manifest_rows,
        ["split", "filename", "width", "height", "channels"],
    )
    write_csv(
        args.summary_out,
        summarize(manifest_rows),
        ["split", "count", "min_width", "max_width", "min_height", "max_height"],
    )

    if args.check_pairs:
        missing = validate_pairing(args.clean_dir, args.synthetic_dir)
        if missing:
            print("Missing synthetic counterparts:")
            for name in missing:
                print(name)
        else:
            print("All clean images have synthetic counterparts.")


if __name__ == "__main__":
    main()

