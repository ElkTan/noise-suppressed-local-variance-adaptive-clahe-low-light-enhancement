from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class AdaptiveClaheConfig:
    tile_grid: tuple[int, int] = (8, 8)
    cmin: float = 1.0
    cmax: float = 5.0
    beta: float = 1.0
    lambda_: float = 0.5
    delta: float = 0.5
    eta: float = 0.0
    eps: float = 1e-6
    smooth_variance: bool = True
    suppress_noise: bool = True
    smooth_clip_map: bool = True
    chroma_compensation: bool = True
    blend_tiles: bool = True


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


def _tile_edges_1d(length: int, count: int) -> np.ndarray:
    return np.linspace(0, length, count + 1, dtype=int)


def estimate_noise_sigma(y: np.ndarray) -> float:
    y_f = y.astype(np.float32)
    residual = y_f - cv2.GaussianBlur(y_f, (5, 5), 0)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
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
    cmax: float = 5.0,
    beta: float = 1.0,
    eps: float = 1e-6,
    smooth_clip_map: bool = True,
) -> np.ndarray:
    if np.allclose(structure_map, 0.0):
        clip_map = np.full_like(structure_map, cmin, dtype=np.float32)
    else:
        p10 = float(np.percentile(structure_map, 10))
        p90 = float(np.percentile(structure_map, 90))
        normalized = np.clip((structure_map - p10) / (p90 - p10 + eps), 0.0, 1.0)
        # beta controls how aggressively high-structure regions receive larger clip limits.
        gamma = max(0.35, 1.0 / max(beta, eps))
        clip_map = cmin + (cmax - cmin) * np.power(normalized, gamma)
    if smooth_clip_map:
        clip_map = cv2.GaussianBlur(clip_map.astype(np.float32), (3, 3), 0)
    return clip_map.astype(np.float32)


def _build_clahe_lut(tile: np.ndarray, clip_limit: float, bins: int = 256) -> np.ndarray:
    hist = np.bincount(tile.ravel(), minlength=bins).astype(np.float32)
    clip_threshold = max(clip_limit * tile.size / bins, 1.0)
    excess = np.maximum(hist - clip_threshold, 0.0)
    hist = np.minimum(hist, clip_threshold)
    redistribute = float(excess.sum())
    hist += redistribute / bins
    cdf = np.cumsum(hist)
    cdf_min = float(cdf[0])
    denom = max(float(cdf[-1] - cdf_min), 1.0)
    lut = np.round((cdf - cdf_min) / denom * 255.0)
    return np.clip(lut, 0, 255).astype(np.uint8)


def _build_tile_luts(y: np.ndarray, clip_map: np.ndarray, tile_grid: tuple[int, int]) -> np.ndarray:
    row_edges = _tile_edges_1d(y.shape[0], tile_grid[0])
    col_edges = _tile_edges_1d(y.shape[1], tile_grid[1])
    luts = np.zeros((tile_grid[0], tile_grid[1], 256), dtype=np.uint8)
    for r in range(tile_grid[0]):
        for c in range(tile_grid[1]):
            tile = y[row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]]
            luts[r, c] = _build_clahe_lut(tile, max(float(clip_map[r, c]), 0.01))
    return luts


def _interpolate_luts(
    y: np.ndarray,
    luts: np.ndarray,
    tile_grid: tuple[int, int],
) -> np.ndarray:
    height, width = y.shape
    row_edges = _tile_edges_1d(height, tile_grid[0])
    col_edges = _tile_edges_1d(width, tile_grid[1])
    row_centers = (row_edges[:-1] + row_edges[1:]) / 2.0
    col_centers = (col_edges[:-1] + col_edges[1:]) / 2.0

    out = np.zeros((height, width), dtype=np.float32)

    for r in range(tile_grid[0]):
        r0, r1 = row_edges[r], row_edges[r + 1]
        rr = np.arange(r0, r1, dtype=np.float32)
        if tile_grid[0] == 1:
            top_idx = bottom_idx = 0
            wy_bottom = np.zeros_like(rr)
        elif r == 0:
            top_idx, bottom_idx = 0, 1
            wy_bottom = (rr - row_centers[0]) / max(row_centers[1] - row_centers[0], 1.0)
        elif r == tile_grid[0] - 1:
            top_idx, bottom_idx = tile_grid[0] - 2, tile_grid[0] - 1
            wy_bottom = (rr - row_centers[top_idx]) / max(row_centers[bottom_idx] - row_centers[top_idx], 1.0)
        else:
            top_idx, bottom_idx = r, r + 1
            wy_bottom = (rr - row_centers[top_idx]) / max(row_centers[bottom_idx] - row_centers[top_idx], 1.0)
        wy_bottom = np.clip(wy_bottom, 0.0, 1.0)[:, None]
        wy_top = 1.0 - wy_bottom

        for c in range(tile_grid[1]):
            c0, c1 = col_edges[c], col_edges[c + 1]
            cc = np.arange(c0, c1, dtype=np.float32)
            if tile_grid[1] == 1:
                left_idx = right_idx = 0
                wx_right = np.zeros_like(cc)
            elif c == 0:
                left_idx, right_idx = 0, 1
                wx_right = (cc - col_centers[0]) / max(col_centers[1] - col_centers[0], 1.0)
            elif c == tile_grid[1] - 1:
                left_idx, right_idx = tile_grid[1] - 2, tile_grid[1] - 1
                wx_right = (cc - col_centers[left_idx]) / max(col_centers[right_idx] - col_centers[left_idx], 1.0)
            else:
                left_idx, right_idx = c, c + 1
                wx_right = (cc - col_centers[left_idx]) / max(col_centers[right_idx] - col_centers[left_idx], 1.0)
            wx_right = np.clip(wx_right, 0.0, 1.0)[None, :]
            wx_left = 1.0 - wx_right

            tile = y[r0:r1, c0:c1]
            tl = luts[top_idx, left_idx][tile]
            tr = luts[top_idx, right_idx][tile]
            bl = luts[bottom_idx, left_idx][tile]
            br = luts[bottom_idx, right_idx][tile]

            out[r0:r1, c0:c1] = (
                wy_top * wx_left * tl
                + wy_top * wx_right * tr
                + wy_bottom * wx_left * bl
                + wy_bottom * wx_right * br
            )
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_tile_clahe(
    y: np.ndarray,
    clip_map: np.ndarray,
    tile_grid: tuple[int, int],
    blend_tiles: bool = True,
) -> np.ndarray:
    luts = _build_tile_luts(y, clip_map=clip_map, tile_grid=tile_grid)
    if blend_tiles:
        return _interpolate_luts(y, luts=luts, tile_grid=tile_grid)

    output = np.zeros_like(y)
    slices = _tile_slices(y.shape, tile_grid)
    for index, (rows, cols) in enumerate(slices):
        r = index // tile_grid[1]
        c = index % tile_grid[1]
        tile = y[rows, cols]
        output[rows, cols] = luts[r, c][tile]
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
    y_enhanced = _apply_tile_clahe(
        y,
        clip_map=clip_map,
        tile_grid=config.tile_grid,
        blend_tiles=config.blend_tiles,
    )
    if config.chroma_compensation and config.eta > 0:
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
        "y_enhanced": y_enhanced,
    }
    return _merge_ycrcb(y_enhanced, cr_out, cb_out), aux


def run_ns_lva_clahe(
    image: np.ndarray,
    tile_grid: tuple[int, int] = (8, 8),
    cmin: float = 1.0,
    cmax: float = 5.0,
    beta: float = 1.0,
    lambda_: float = 0.5,
    delta: float = 0.5,
    eta: float = 0.0,
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
        chroma_compensation=eta > 0,
        blend_tiles=True,
    )
    return run_adaptive_clahe_variant(image, config)


def build_ablation_variants(
    image: np.ndarray,
    tile_grid: tuple[int, int] = (8, 8),
    cmin: float = 1.0,
    cmax: float = 5.0,
    beta: float = 1.0,
    lambda_: float = 0.5,
    delta: float = 0.5,
    eta: float = 0.0,
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
            blend_tiles=False,
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
            blend_tiles=False,
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
            chroma_compensation=eta > 0,
            blend_tiles=True,
        ),
    }
    return {name: run_adaptive_clahe_variant(image, config) for name, config in variants.items()}
