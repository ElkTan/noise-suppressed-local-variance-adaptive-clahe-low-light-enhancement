from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_image_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"Failed to write image: {path}")


def synthesize_low_light(
    image: np.ndarray,
    alpha: float = 0.3,
    gamma: float = 2.0,
    sigma: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a synthetic low-light image from an RGB uint8 image."""
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image.")
    if rng is None:
        rng = np.random.default_rng(0)

    image_f = image.astype(np.float32) / 255.0
    degraded = np.power(np.clip(alpha * image_f, 0.0, 1.0), gamma)
    noise = rng.normal(loc=0.0, scale=sigma, size=degraded.shape).astype(np.float32)
    degraded = np.clip(degraded + noise, 0.0, 1.0)
    return (degraded * 255.0).round().astype(np.uint8)


def batch_synthesize_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    alpha: float = 0.3,
    gamma: float = 2.0,
    sigma: float = 0.02,
) -> list[Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for image_path in sorted(p for p in input_dir.iterdir() if p.is_file()):
        image = read_image_rgb(image_path)
        low_light = synthesize_low_light(image, alpha=alpha, gamma=gamma, sigma=sigma)
        output_path = output_dir / image_path.name
        write_image_rgb(output_path, low_light)
        saved_paths.append(output_path)
    return saved_paths

