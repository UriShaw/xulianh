"""Nạp ảnh, đổi xám/kênh V, ảnh mẫu."""
import cv2
import numpy as np


def decode_image(data: bytes):
    """Giải mã byte ảnh -> RGB uint8 (có áp dụng xoay EXIF)."""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Không đọc được ảnh (định dạng không hợp lệ).")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_max(rgb, max_side=512):
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return rgb
    return cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def to_working(rgb, mode):
    """Ảnh làm việc 2-D uint8: 'gray' = độ chói, 'v' = kênh V của HSV."""
    if mode == "v":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2].copy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def to_display(rgb, work, mode):
    """Dựng ảnh hiển thị: chế độ 'v' giữ nguyên H,S và thay V."""
    if mode == "v":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv[..., 2] = work
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return work


def png_bytes(img):
    arr = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    return buf.tobytes()


def load_sample(name):
    from skimage import data
    img = getattr(data, name)()
    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


SAMPLES = ["camera", "astronaut", "coins"]
