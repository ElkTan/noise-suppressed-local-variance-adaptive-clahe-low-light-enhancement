from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class AdaptiveClaheConfig:
    tile_grid: tuple[int, int] = (8, 8)
    cmin: float = 1.0
    cmax: float = 4.0
    beta: float = 1.0
    lambda_: float = 0.5
    delta: float = 0.5
    eta: float = 1.0
    eps: float = 1e-6
    smooth_variance: bool = True
    suppress_noise: bool = True
    smooth_clip_map: bool = True
    chroma_compensation: bool = True


def _split_ycrcb(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    return y, cr, cb


def _merge_ycrcb(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    merged = cv2.merge(
        [np.clip(y, 0, 255).astype(np.uint8), np.clip(cr, 0, 255).astype(np.uint8), np.clip(cb, 0, 255).astype(np.uint8)]
    )
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)


def _tile_edges(length: int, count: int) -> np.ndarray:
    return np.linspace(0, length, count + 1, dtype=int)


def _tile_slices(shape: tuple[int, int], tile_grid: tuple[int, int]) -> list[tuple[slice, slice]]:
    height, width = shape
    row_edges = _tile_edges(height, tile_grid[0])
    col_edges = _tile_edges(width, tile_grid[1])
    return [
        (slice(row_edges[r], row_edges[r + 1]), slice(col_edges[c], col_edges[c + 1]))
        for r in range(tile_grid[0])
        for c in range(tile_grid[1])
    ]


def estimate_noise_sigma(y: np.ndarray) -> float:
    lap = cv2.Laplacian(y.astype(np.float32), cv2.CV_32F)
    median = float(np.median(lap))
    mad = float(np.median(np.abs(lap - median)))
    return mad / 0.6745 if mad > 0 else 0.0


def build_structure_map(
    y: np.ndarray,
    tile_grid: tuple[int, int] = (8, 8),
    smooth_variance: bool = True,
    suppress_noise: bool = True,
    lambda_: float = 0.5,
) -> tuple[np.ndarray, float]:
    source = cv2.GaussianBlur(y, (5, 5), 0) if smooth_variance else y
    source_f = source.astype(np.float32)
    noise_sigma = estimate_noise_sigma(y)
    noise_var = noise_sigma ** 2

    values: list[float] = []
    for rows, cols in _tile_slices(y.shape, tile_grid):
        variance = float(np.var(source_f[rows, cols]))
        if suppress_noise:
            variance = max(variance - lambda_ * noise_var, 0.0)
        values.append(variance)
    return np.array(values, dtype=np.float32).reshape(tile_grid), noise_sigma


def build_clip_map(
    structure_map: np.ndarray,
    cmin: float = 1.0,
    cmax: float = 4.0,
    beta: float = 1.0,
    eps: float = 1e-6,
    smooth_clip_map: bool = True,
) -> np.ndarray:
    k = beta * float(np.mean(structure_map)) + eps
    clip_map = cmin + (cmax - cmin) * (structure_map / (structure_map + k + eps))
    if smooth_clip_map:
        clip_map = cv2.GaussianBlur(clip_map.astype(np.float32), (3, 3), 0)
    return clip_map.astype(np.float32)


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


def compensate_chroma(
    y: np.ndarray,
    y_enhanced: np.ndarray,
    cr: np.ndarray,
    cb: np.ndarray,
    eta: float = 1.0,
    delta: float = 0.5,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    y_f = y.astype(np.float32)
    y_enhanced_f = y_enhanced.astype(np.float32)
    gain = np.power((y_enhanced_f + eps) / (y_f + eps), delta)
    cr_new = 128.0 + eta * gain * (cr.astype(np.float32) - 128.0)
    cb_new = 128.0 + eta * gain * (cb.astype(np.float32) - 128.0)
    return np.clip(cr_new, 0, 255).astype(np.uint8), np.clip(cb_new, 0, 255).astype(np.uint8)


def run_adaptive_clahe_variant(image: np.ndarray, config: AdaptiveClaheConfig) -> tuple[np.ndarray, dict[str, Any]]:
    y, cr, cb = _split_ycrcb(image)
    structure_map, noise_sigma = build_structure_map(
        y,
        tile_grid=config.tile_grid,
        smooth_variance=config.smooth_variance,
        suppress_noise=config.suppress_noise,
        lambda_=config.lambda_,
    )
    clip_map = build_clip_map(
        structure_map,
        cmin=config.cmin,
        cmax=config.cmax,
        beta=config.beta,
        eps=config.eps,
        smooth_clip_map=config.smooth_clip_map,
    )
    y_enhanced = _apply_tile_clahe(y, clip_map=clip_map, tile_grid=config.tile_grid)
    if config.chroma_compensation:
        cr_out, cb_out = compensate_chroma(
            y=y,
            y_enhanced=y_enhanced,
            cr=cr,
            cb=cb,
            eta=config.eta,
            delta=config.delta,
            eps=config.eps,
        )
    else:
        cr_out, cb_out = cr, cb

    aux = {
        "clip_map": clip_map,
        "structure_map": structure_map,
        "noise_sigma": noise_sigma,
        "config": config,
    }
    return _merge_ycrcb(y_enhanced, cr_out, cb_out), aux


def run_ns_lva_clahe(
    image: np.ndarray,
    tile_grid: tuple[int, int] = (8, 8),
    cmin: float = 1.0,
    cmax: float = 4.0,
    beta: float = 1.0,
    lambda_: float = 0.5,
    delta: float = 0.5,
    eta: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = AdaptiveClaheConfig(
        tile_grid=tile_grid,
        cmin=cmin,
        cmax=cmax,
        beta=beta,
        lambda_=lambda_,
        delta=delta,
        eta=eta,
        smooth_variance=True,
        suppress_noise=True,
        smooth_clip_map=True,
        chroma_compensation=True,
    )
    return run_adaptive_clahe_variant(image, config)


def build_ablation_variants(
    image: np.ndarray,
    tile_grid: tuple[int, int] = (8, 8),
    cmin: float = 1.0,
    cmax: float = 4.0,
    beta: float = 1.0,
    lambda_: float = 0.5,
    delta: float = 0.5,
    eta: float = 1.0,
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    variants = {
        "smoothed_variance_adaptive": AdaptiveClaheConfig(
            tile_grid=tile_grid,
            cmin=cmin,
            cmax=cmax,
            beta=beta,
            lambda_=lambda_,
            delta=delta,
            eta=eta,
            smooth_variance=True,
            suppress_noise=False,
            smooth_clip_map=False,
            chroma_compensation=False,
        ),
        "noise_suppressed_adaptive": AdaptiveClaheConfig(
            tile_grid=tile_grid,
            cmin=cmin,
            cmax=cmax,
            beta=beta,
            lambda_=lambda_,
            delta=delta,
            eta=eta,
            smooth_variance=True,
            suppress_noise=True,
            smooth_clip_map=False,
            chroma_compensation=False,
        ),
        "full_method": AdaptiveClaheConfig(
            tile_grid=tile_grid,
            cmin=cmin,
            cmax=cmax,
            beta=beta,
            lambda_=lambda_,
            delta=delta,
            eta=eta,
            smooth_variance=True,
            suppress_noise=True,
            smooth_clip_map=True,
            chroma_compensation=True,
        ),
    }
    return {name: run_adaptive_clahe_variant(image, config) for name, config in variants.items()}

