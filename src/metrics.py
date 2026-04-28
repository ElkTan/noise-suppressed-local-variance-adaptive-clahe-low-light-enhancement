from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32)


def compute_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_f = gt.astype(np.float32)
    pred_f = pred.astype(np.float32)
    mse = float(np.mean((gt_f - pred_f) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def _ssim_single_channel(x: np.ndarray, y: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def compute_ssim(gt: np.ndarray, pred: np.ndarray, channel: str = "rgb") -> float:
    if channel == "y":
        gt_y = _to_gray_float(gt)
        pred_y = _to_gray_float(pred)
        return _ssim_single_channel(gt_y, pred_y)
    if channel != "rgb":
        raise ValueError("channel must be 'rgb' or 'y'")

    scores = [_ssim_single_channel(gt[..., idx], pred[..., idx]) for idx in range(gt.shape[2])]
    return float(np.mean(scores))


def compute_nar(input_low: np.ndarray, enhanced: np.ndarray, tl: int = 50, tg: int = 10) -> float:
    y_low = _to_gray_float(input_low)
    y_enhanced = _to_gray_float(enhanced)

    grad_x = cv2.Sobel(y_low, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(y_low, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(grad_x, grad_y)
    mask = (y_low < tl) & (gradient_mag < tg)

    if int(mask.sum()) < 16:
        mask = np.ones_like(y_low, dtype=bool)

    low_var = float(np.var(y_low[mask]))
    enhanced_var = float(np.var(y_enhanced[mask]))
    return enhanced_var / (low_var + 1e-6)


def compute_mean_y(image: np.ndarray) -> float:
    y = _to_gray_float(image)
    return float(np.mean(y))


def compute_loe(input_low: np.ndarray, enhanced: np.ndarray, max_size: int = 64) -> float:
    """Lightness Order Error. Lower is better."""
    low_gray = _to_gray_float(input_low)
    enhanced_gray = _to_gray_float(enhanced)

    height, width = low_gray.shape
    scale = min(max_size / max(height, 1), max_size / max(width, 1), 1.0)
    if scale < 1.0:
        new_size = (max(int(width * scale), 8), max(int(height * scale), 8))
        low_gray = cv2.resize(low_gray, new_size, interpolation=cv2.INTER_AREA)
        enhanced_gray = cv2.resize(enhanced_gray, new_size, interpolation=cv2.INTER_AREA)

    low_vec = low_gray.reshape(-1)
    enhanced_vec = enhanced_gray.reshape(-1)
    low_order = low_vec[:, None] >= low_vec[None, :]
    enhanced_order = enhanced_vec[:, None] >= enhanced_vec[None, :]
    disagreements = np.logical_xor(low_order, enhanced_order)
    return float(np.mean(disagreements))


def _mscn(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32) / 255.0
    mu = cv2.GaussianBlur(image_f, (7, 7), 7 / 6)
    sigma = cv2.GaussianBlur((image_f - mu) ** 2, (7, 7), 7 / 6)
    sigma = np.sqrt(np.maximum(sigma, 1e-6))
    return (image_f - mu) / (sigma + 1e-6)


def _naturalness_features(gray: np.ndarray) -> np.ndarray:
    mscn = _mscn(gray)
    feats: list[float] = []
    feats.extend(
        [
            float(np.mean(mscn)),
            float(np.std(mscn)),
            float(np.mean(np.abs(mscn))),
            float(np.mean(mscn**2)),
            float(np.mean(mscn**4)),
        ]
    )
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        shifted = np.roll(mscn, shift=(dy, dx), axis=(0, 1))
        pair = mscn * shifted
        feats.extend(
            [
                float(np.mean(pair)),
                float(np.std(pair)),
                float(np.mean(np.abs(pair))),
            ]
        )
    hist, _ = np.histogram(gray.astype(np.uint8), bins=16, range=(0, 255), density=True)
    feats.extend(hist.astype(np.float32).tolist())
    return np.array(feats, dtype=np.float32)


def fit_naturalness_model(image_paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        features.append(_naturalness_features(gray))
    if not features:
        raise ValueError("No valid clean reference images found for naturalness model.")
    feature_matrix = np.stack(features, axis=0)
    mean = np.mean(feature_matrix, axis=0)
    cov = np.cov(feature_matrix, rowvar=False)
    cov += np.eye(cov.shape[0], dtype=np.float32) * 1e-4
    return mean.astype(np.float32), cov.astype(np.float32)


def compute_niqe_like(image: np.ndarray, model_mean: np.ndarray, model_cov: np.ndarray) -> float:
    """A lightweight NIQE-style naturalness distance fitted from clean images. Lower is better."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    feature = _naturalness_features(gray)
    delta = feature - model_mean
    inv_cov = np.linalg.pinv(model_cov)
    score = float(np.sqrt(np.maximum(delta @ inv_cov @ delta.T, 0.0)))
    return score
