"""Lớp IO: nạp/lưu ảnh (an toàn với đường dẫn tiếng Việt trên Windows), chuyển xám/kênh V, CSV, ảnh tổng hợp."""
import glob
import os
import sys

import cv2
import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def resource_path(*parts):
    """Đường dẫn tới tài nguyên đi kèm (chạy mã nguồn hoặc chạy trong file exe PyInstaller)."""
    base = getattr(sys, "_MEIPASS", None) or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base, *parts)


def read_rgb(path):
    """Đọc ảnh -> RGB uint8. Dùng np.fromfile để đọc được đường dẫn có dấu."""
    data = np.fromfile(path, dtype=np.uint8)
    return decode_image(data.tobytes())


def decode_image(data: bytes):
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("Không đọc được ảnh (định dạng không hợp lệ).")
    if arr.dtype != np.uint8:                      # ảnh 16-bit -> 8-bit
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def resize_max(rgb, max_side=512):
    """Thu nhỏ để cạnh dài ≤ max_side (không phóng to)."""
    if not max_side:
        return rgb
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return rgb
    return cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb


def to_working(rgb, mode="gray"):
    """Ảnh làm việc 2-D uint8: 'gray' = độ chói, 'v' = kênh V của HSV."""
    if mode == "v" and rgb.ndim == 3:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2].copy()
    return to_gray(rgb)


def to_display(rgb, work, mode="gray"):
    """Ảnh hiển thị: chế độ 'v' giữ H, S của ảnh gốc và thay kênh V."""
    if mode == "v" and rgb.ndim == 3:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv[..., 2] = work
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return work


def save_image(path, img):
    """Lưu PNG/JPG an toàn với đường dẫn Unicode."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, arr)
    if not ok:
        raise ValueError("Không mã hoá được ảnh.")
    buf.tofile(path)


def list_images(patterns, root="."):
    """Mở rộng danh sách glob -> danh sách file ảnh (đã sắp xếp, không trùng)."""
    if isinstance(patterns, str):
        patterns = [patterns]
    out = []
    for p in patterns:
        p = p if os.path.isabs(p) else os.path.join(root, p)
        if os.path.isdir(p):
            p = os.path.join(p, "*")
        out += [f for f in glob.glob(p) if f.lower().endswith(IMG_EXT)]
    return sorted(dict.fromkeys(os.path.normpath(f) for f in out))


def save_csv(df, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")   # utf-8-sig để Excel đọc đúng tiếng Việt


def sample_images():
    """Ảnh mẫu đi kèm ứng dụng: {tên: đường dẫn}."""
    return {os.path.splitext(os.path.basename(f))[0]: f
            for f in list_images(resource_path("data", "raw", "*"))}


# ----------------------------------------------------------------- ảnh tổng hợp (mục 4.2)
def synthetic_images(size=256):
    """Ảnh tổng hợp biết trước vị trí biên: thang xám, bậc thang, nửa đen–trắng, hình học."""
    ramp = np.tile(np.linspace(0, 255, size).astype(np.uint8), (size, 1))
    steps = np.tile(np.repeat(np.linspace(0, 255, 8).astype(np.uint8), size // 8), (size, 1))
    half = np.zeros((size, size), np.uint8)
    half[:, size // 2:] = 255
    shapes = np.full((size, size), 60, np.uint8)
    cv2.circle(shapes, (size // 3, size // 3), size // 6, 200, -1)
    cv2.rectangle(shapes, (size // 2, size // 2), (size - 20, size - 20), 120, -1)
    cv2.line(shapes, (10, size - 10), (size - 10, 10), 240, 3)
    return {"syn_ramp": ramp, "syn_steps": steps, "syn_half": half, "syn_shapes": shapes}


def write_synthetic(folder, size=256):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for name, img in synthetic_images(size).items():
        p = os.path.join(folder, name + ".png")
        save_image(p, img)
        paths.append(p)
    return paths
