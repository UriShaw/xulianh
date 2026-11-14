"""Ba phép biến đổi cường độ (hàm thuần, ảnh uint8) và bảng tra LUT."""
import numpy as np

LOG_C = 255.0 / np.log(256.0)


def _check(img):
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise ValueError("Ảnh phải có dtype uint8.")
    if a.ndim not in (1, 2):
        raise ValueError("Ảnh phải là mảng 2-D (ảnh xám).")
    return a


def _to_u8(x):
    return np.rint(np.clip(x, 0, 255)).astype(np.uint8)


def identity(img):
    return _check(img).copy()


def negative(img):
    """s = 255 - r."""
    return (255 - _check(img)).astype(np.uint8)


def log_transform(img, gain=1.0):
    """s = gain * (255/ln256) * ln(1+r); cắt ngưỡng 255."""
    a = _check(img)
    if not gain > 0:
        raise ValueError("gain phải > 0.")
    return _to_u8(gain * LOG_C * np.log1p(a.astype(np.float64)))


def gamma_transform(img, gamma, c=1.0):
    """s = 255 * c * (r/255)^gamma; cắt ngưỡng 255."""
    a = _check(img)
    if not gamma > 0:
        raise ValueError("gamma phải > 0.")
    if not c > 0:
        raise ValueError("c phải > 0.")
    return _to_u8(255.0 * c * (a.astype(np.float64) / 255.0) ** gamma)


TRANSFORMS = {
    "identity": identity,
    "negative": negative,
    "log": log_transform,
    "gamma": gamma_transform,
}


def apply_lut(img, func, **kw):
    """Dựng LUT 256 phần tử từ `func` rồi tra bảng: O(MN)."""
    a = _check(img)
    lut = func(np.arange(256, dtype=np.uint8), **kw)
    return lut[a]
