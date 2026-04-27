from __future__ import annotations

import cv2
import numpy as np


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

