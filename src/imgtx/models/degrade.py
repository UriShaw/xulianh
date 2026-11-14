"""Mô phỏng ảnh thiếu/quá sáng để có cặp (ảnh gốc, ảnh đầu vào) – mục 8.2.

Quy tắc 1: suy giảm phải NẰM NGOÀI họ biến đổi được khảo sát (gamma × c, log × gain),
nên dùng tuyến tính + độ lệch đen / cắt ngưỡng / nhiễu. `darken_gamma_control` chỉ để đối chứng (TN8).
"""
import numpy as np


def _q(x):
    return np.rint(np.clip(x, 0, 255)).astype(np.uint8)


def darken_offset(img, alpha=0.4, beta=10.0):
    """Thiếu sáng: s = round(clip(alpha*r - beta)). Vùng tối nhất bị ép về 0."""
    return _q(alpha * np.asarray(img, np.float64) - beta)


def darken_noisy(img, alpha=0.4, beta=10.0, sigma=4.0, seed=0):
    """Như offset + nhiễu Gauss (seed cố định) trước khi lượng tử hoá."""
    rng = np.random.default_rng(seed)
    x = alpha * np.asarray(img, np.float64) - beta + rng.normal(0.0, sigma, np.shape(img))
    return _q(x)


def brighten_offset(img, alpha=1.6, beta=40.0):
    """Quá sáng: s = round(clip(alpha*r + beta)); vùng sáng bị cắt ngưỡng 255."""
    return _q(alpha * np.asarray(img, np.float64) + beta)


def darken_gamma_control(img, gamma=2.5):
    """CHỈ ĐỐI CHỨNG (TN8): làm tối bằng gamma -> gamma 1/2.5 khôi phục gần hoàn hảo (thiên vị)."""
    return _q(255.0 * (np.asarray(img, np.float64) / 255.0) ** gamma)


DEGRADES = {
    "offset": darken_offset,
    "noisy": darken_noisy,
    "bright": brighten_offset,
    "control_gamma": darken_gamma_control,
}

DEGRADE_LABELS = {
    "offset": "Thiếu sáng (offset)",
    "noisy": "Thiếu sáng + nhiễu",
    "bright": "Quá sáng (bright)",
    "control_gamma": "Đối chứng gamma 2.5 (TN8)",
}

DEFAULT_PARAMS = {
    "offset": {"alpha": 0.4, "beta": 10},
    "noisy": {"alpha": 0.4, "beta": 10, "sigma": 4.0, "seed": 0},
    "bright": {"alpha": 1.6, "beta": 40},
    "control_gamma": {"gamma": 2.5},
}


def build_pair(img, kind=None, **params):
    """Trả (ref_goc, ref_dau_vao). kind=None/'none' -> ảnh thật: (None, img)."""
    if kind in (None, "none"):
        return None, img
    p = dict(DEFAULT_PARAMS.get(kind, {}))
    p.update(params)
    return img, DEGRADES[kind](img, **p)
