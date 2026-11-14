"""Suy giảm ảnh KHÔNG thuộc họ gamma/log để tạo cặp (ảnh gốc, ảnh đầu vào)."""
import numpy as np


def _q(x):
    return np.rint(np.clip(x, 0, 255)).astype(np.uint8)


def darken_offset(img, alpha=0.4, beta=10.0):
    return _q(alpha * img.astype(np.float64) - beta)


def darken_noisy(img, alpha=0.4, beta=10.0, sigma=4.0, seed=0):
    rng = np.random.default_rng(seed)
    x = alpha * img.astype(np.float64) - beta + rng.normal(0, sigma, img.shape)
    return _q(x)


def brighten_offset(img, alpha=1.6, beta=40.0):
    return _q(alpha * img.astype(np.float64) + beta)


DEGRADES = {
    "Thiếu sáng (offset)": darken_offset,
    "Thiếu sáng + nhiễu": darken_noisy,
    "Quá sáng (bright)": brighten_offset,
}
