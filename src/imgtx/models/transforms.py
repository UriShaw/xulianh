"""Ba phép biến đổi cường độ (âm bản, logarit, gamma) cho ảnh xám 8-bit + bảng tra LUT.

Mọi hàm là hàm thuần: nhận mảng NumPy uint8, trả mảng uint8 cùng kích thước,
không đọc/ghi file (mục 5.1 tài liệu thiết kế).
"""
import numpy as np

L = 256                                  # số mức xám
LOG_C = (L - 1) / np.log(L)              # c = 255 / ln 256  -> T(255) = 255 khi gain = 1


def _check(img):
    """Kiểm tra ảnh: dtype uint8, 1-D hoặc 2-D (ảnh xám). Ném ValueError nếu sai (T09)."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise ValueError("Ảnh phải có dtype uint8.")
    if a.ndim not in (1, 2):
        raise ValueError("Ảnh phải là mảng 2-D (ảnh xám).")
    return a


def _to_u8(x):
    """Cắt ngưỡng [0, 255], làm tròn rồi ép uint8 (tránh tràn số)."""
    return np.rint(np.clip(x, 0, L - 1)).astype(np.uint8)


# ----------------------------------------------------------------- registry (NFR4)
TRANSFORMS = {}


def register(name):
    """Đăng ký một phép biến đổi mới: chỉ cần gắn @register("ten") lên hàm."""
    def deco(fn):
        TRANSFORMS[name] = fn
        return fn
    return deco


@register("identity")
def identity(img):
    """s = r (dòng mốc của mọi thí nghiệm)."""
    return _check(img).copy()


@register("negative")
def negative(img):
    """Âm bản: s = (L-1) - r."""
    return (L - 1 - _check(img)).astype(np.uint8)


@register("log")
def log_transform(img, gain=1.0):
    """Logarit: s = gain * (255/ln256) * ln(1 + r); cắt ngưỡng 255 khi gain > 1."""
    a = _check(img)
    if not gain > 0:
        raise ValueError("gain phải > 0.")
    return _to_u8(gain * LOG_C * np.log1p(a.astype(np.float64)))


@register("gamma")
def gamma_transform(img, gamma, c=1.0):
    """Gamma: s = 255 * c * (r/255)^gamma; cắt ngưỡng 255 khi c > 1."""
    a = _check(img)
    if not gamma > 0:
        raise ValueError("gamma phải > 0.")
    if not c > 0:
        raise ValueError("c phải > 0.")
    return _to_u8((L - 1) * c * np.power(a.astype(np.float64) / (L - 1), gamma))


def build_lut(func, **kw):
    """Bảng tra 256 phần tử của phép biến đổi `func`."""
    return func(np.arange(L, dtype=np.uint8), **kw)


def apply_lut(img, func, **kw):
    """Áp dụng phép biến đổi bằng LUT: O(MN), kết quả bằng đúng công thức trực tiếp (T07)."""
    a = _check(img)
    return build_lut(func, **kw)[a]


def apply_color_v(rgb, func, **kw):
    """Mở rộng sang ảnh màu (FR12): biến đổi kênh V của HSV, giữ nguyên H, S."""
    import cv2
    hsv = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2HSV)
    hsv[..., 2] = apply_lut(hsv[..., 2], func, **kw)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
