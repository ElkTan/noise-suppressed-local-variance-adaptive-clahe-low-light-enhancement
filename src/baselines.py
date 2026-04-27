from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _split_ycrcb(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    return y, cr, cb


def _merge_ycrcb(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.merge(
        [np.clip(y, 0, 255).astype(np.uint8), np.clip(cr, 0, 255).astype(np.uint8), np.clip(cb, 0, 255).astype(np.uint8)]
    )
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _tile_slices(shape: tuple[int, int], tile_grid: tuple[int, int]) -> list[tuple[slice, slice]]:
    height, width = shape
    tile_rows, tile_cols = tile_grid
    row_edges = np.linspace(0, height, tile_rows + 1, dtype=int)
    col_edges = np.linspace(0, width, tile_cols + 1, dtype=int)
    return [
        (slice(row_edges[r], row_edges[r + 1]), slice(col_edges[c], col_edges[c + 1]))
        for r in range(tile_rows)
        for c in range(tile_cols)
    ]


def _compute_variance_map(y: np.ndarray, tile_grid: tuple[int, int], smooth: bool) -> np.ndarray:
    source = cv2.GaussianBlur(y, (5, 5), 0) if smooth else y
    source_f = source.astype(np.float32)
    clip_values: list[float] = []
    for rows, cols in _tile_slices(y.shape, tile_grid):
        clip_values.append(float(np.var(source_f[rows, cols])))
    return np.array(clip_values, dtype=np.float32).reshape(tile_grid)


def _variance_to_clip_map(
    variance_map: np.ndarray,
    cmin: float = 1.0,
    cmax: float = 4.0,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    mean_var = float(np.mean(variance_map))
    k = beta * mean_var + eps
    return cmin + (cmax - cmin) * (variance_map / (variance_map + k + eps))


def _apply_tile_clahe(y: np.ndarray, clip_map: np.ndarray, tile_grid: tuple[int, int]) -> np.ndarray:
    output = np.zeros_like(y)
    slices = _tile_slices(y.shape, tile_grid)
    for index, (rows, cols) in enumerate(slices):
        r = index // tile_grid[1]
        c = index % tile_grid[1]
        tile = y[rows, cols]
        clip_limit = max(float(clip_map[r, c]), 0.01)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        output[rows, cols] = clahe.apply(tile)
    return output


def run_he(image: np.ndarray) -> np.ndarray:
    y, cr, cb = _split_ycrcb(image)
    y_out = cv2.equalizeHist(y)
    return _merge_ycrcb(y_out, cr, cb)


def run_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    y, cr, cb = _split_ycrcb(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    y_out = clahe.apply(y)
    return _merge_ycrcb(y_out, cr, cb)


def run_denoise_clahe(
    image: np.ndarray,
    denoise: str = "bilateral",
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    y, cr, cb = _split_ycrcb(image)
    if denoise == "bilateral":
        y_denoised = cv2.bilateralFilter(y, d=5, sigmaColor=25, sigmaSpace=25)
    elif denoise == "median":
        y_denoised = cv2.medianBlur(y, 5)
    else:
        raise ValueError(f"Unsupported denoise method: {denoise}")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    y_out = clahe.apply(y_denoised)
    return _merge_ycrcb(y_out, cr, cb)


def run_raw_variance_adaptive_clahe(
    image: np.ndarray,
    tile_grid: tuple[int, int] = (8, 8),
    cmin: float = 1.0,
    cmax: float = 4.0,
    beta: float = 1.0,
) -> np.ndarray:
    enhanced, _ = run_raw_variance_adaptive_clahe_with_aux(
        image=image,
        tile_grid=tile_grid,
        cmin=cmin,
        cmax=cmax,
        beta=beta,
    )
    return enhanced


def run_raw_variance_adaptive_clahe_with_aux(
    image: np.ndarray,
    tile_grid: tuple[int, int] = (8, 8),
    cmin: float = 1.0,
    cmax: float = 4.0,
    beta: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    y, cr, cb = _split_ycrcb(image)
    variance_map = _compute_variance_map(y, tile_grid=tile_grid, smooth=False)
    clip_map = _variance_to_clip_map(variance_map, cmin=cmin, cmax=cmax, beta=beta)
    y_out = _apply_tile_clahe(y, clip_map=clip_map, tile_grid=tile_grid)
    aux = {"clip_map": clip_map, "structure_map": variance_map, "noise_sigma": 0.0}
    return _merge_ycrcb(y_out, cr, cb), aux

