"""Histogram, thống kê, Sobel, độ đo bảo toàn biên, MSE/PSNR và hàm evaluate (mục 2.2–2.5, 6.4)."""
import cv2
import numpy as np


def histogram(img):
    """(h, p): số điểm ảnh mỗi mức xám và histogram chuẩn hoá."""
    h = np.bincount(np.asarray(img).ravel(), minlength=256).astype(np.int64)
    return h, h / h.sum()


def hist_stats(img):
    """mean, std, entropy (bit), P1, P99, dải động P99−P1, tỉ lệ bão hoà 0/255, skewness."""
    _, p = histogram(img)
    r = np.arange(256)
    mean = float((r * p).sum())
    std = float(np.sqrt(((r - mean) ** 2 * p).sum()))
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum()) + 0.0
    cdf = np.cumsum(p)
    p1 = int(np.searchsorted(cdf, 0.01))
    p99 = int(np.searchsorted(cdf, 0.99))
    skew = float(((r - mean) ** 3 * p).sum() / std ** 3) if std > 0 else 0.0
    return dict(mean=mean, std=std, entropy=entropy, p1=p1, p99=p99, dyn_range=p99 - p1,
                sat_low=float(p[0]), sat_high=float(p[255]), skew=skew)


# ----------------------------------------------------------------- Sobel TỰ CÀI ĐẶT (thuật toán cốt lõi)
SOBEL_GX = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64)     # đạo hàm theo x (ngang) – slide 33, chương 2
SOBEL_GY = SOBEL_GX.T.copy()                             # [[-1,-2,-1],[0,0,0],[1,2,1]] – đạo hàm theo y


def correlate3x3(f, kernel):
    """Tương quan (correlation) ảnh float với nhân 3×3, tự cài bằng NumPy.

    g(x, y) = Σ_{i=-1..1} Σ_{j=-1..1} w(i, j) · f(x+i, y+j)

    Biên ảnh được đệm bằng phản xạ không lặp điểm mép (…c b | a b c | b a…), giống BORDER_REFLECT_101
    mặc định của OpenCV, nên kết quả trùng cv2.Sobel tới sai số dấu phẩy động. Thay vì 2 vòng for
    qua từng điểm ảnh (rất chậm trong Python), cộng dồn 9 ảnh dịch chuyển có trọng số – cùng công thức,
    nhanh hơn hàng trăm lần.
    """
    h, w = f.shape
    pad = np.pad(f, 1, mode="reflect")
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(3):              # hàng của nhân  (dịch theo y)
        for j in range(3):          # cột của nhân   (dịch theo x)
            if kernel[i, j] != 0:
                out += kernel[i, j] * pad[i:i + h, j:j + w]
    return out


def sobel(img, ksize=3, method="manual", norm="l2"):
    """Gradient Sobel: trả (độ lớn M, hướng θ).

    method : "manual" – tự cài (correlate3x3, mặc định); "opencv" – cv2.Sobel để đối chiếu.
    norm   : "l2" – M = sqrt(Gx² + Gy²);  "l1" – xấp xỉ M ≈ |Gx| + |Gy| (slide 33).
    Tính trên float64 để không tràn số uint8.
    """
    f = np.asarray(img).astype(np.float64)
    if method == "manual" and ksize == 3:
        gx, gy = correlate3x3(f, SOBEL_GX), correlate3x3(f, SOBEL_GY)
    else:
        gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.abs(gx) + np.abs(gy) if norm == "l1" else np.hypot(gx, gy)
    return mag, np.arctan2(gy, gx)


def edge_map_topk(mag, density=0.10):
    """Giữ đúng round(density·N) điểm có gradient lớn nhất (cùng mật độ ⇒ P = R = F1).

    Sắp xếp ổn định (stable) nên khi nhiều điểm bằng nhau ở ngưỡng, điểm có chỉ số nhỏ hơn được giữ:
    kết quả tất định, không đổi theo phiên bản NumPy (tái lập – NFR3).
    """
    flat = mag.ravel()
    n = max(1, min(int(round(density * flat.size)), flat.size))
    idx = np.argsort(-flat, kind="stable")[:n]
    m = np.zeros(flat.shape, bool)
    m[idx] = True
    return m.reshape(mag.shape)


def otsu_threshold(img_u8):
    """Ngưỡng Otsu TỰ CÀI từ histogram: chọn t cực đại phương sai giữa hai lớp.

    σ_B²(t) = [μ_T·ω(t) − μ(t)]² / [ω(t)·(1 − ω(t))],  ω(t) = Σ_{k≤t} p_k,  μ(t) = Σ_{k≤t} k·p_k.
    Lớp 1 = {r ≤ t}, lớp 2 = {r > t}. Trả t (0..255), trùng cv2.threshold(..., THRESH_OTSU).
    """
    _, p = histogram(img_u8)
    k = np.arange(256)
    omega = np.cumsum(p)
    mu = np.cumsum(k * p)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return int(np.argmax(sigma_b))


def edge_map_otsu(mag):
    """Nhị phân hoá Otsu (tự cài) trên ảnh gradient đã chuẩn hoá về 0–255."""
    mx = mag.max()
    if mx <= 0:
        return np.zeros(mag.shape, bool)
    g = np.rint(mag / mx * 255).astype(np.uint8)
    return g > otsu_threshold(g)


def edge_scores(e_ref, e_test):
    tp = float(np.logical_and(e_ref, e_test).sum())
    fp = float(np.logical_and(~e_ref, e_test).sum())
    fn = float(np.logical_and(e_ref, ~e_test).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return dict(precision=pr, recall=rc, f1=f1, iou=iou)


def grad_corr(a, b):
    """Tương quan Pearson giữa hai bản đồ độ lớn gradient (không cần ngưỡng)."""
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def mse(a, b):
    if a.shape != b.shape:
        raise ValueError("Hai ảnh phải cùng kích thước.")
    d = a.astype(np.float64) - b.astype(np.float64)
    return float((d * d).mean())


def psnr(a, b, peak=255.0, cap=100.0):
    """PSNR (dB); MSE = 0 -> cap (100 dB)."""
    m = mse(a, b)
    return cap if m == 0 else float(10.0 * np.log10(peak ** 2 / m))


def evaluate(ref_goc, ref_dau_vao, out, density=0.10, ksize=3):
    """So sánh ảnh kết quả `out` với ảnh tham chiếu (mục 6.4).

    Tham chiếu = ref_goc (ảnh gốc) nếu có, ngược lại ref_dau_vao; ghi ref_type tương ứng.
    MSE/PSNR chỉ tính khi có ảnh gốc, còn lại NaN.
    """
    has_gt = ref_goc is not None
    ref = ref_goc if has_gt else ref_dau_vao
    row = dict(ref_type="goc" if has_gt else "dau_vao")
    row.update(hist_stats(out))
    mag_ref, _ = sobel(ref, ksize)
    mag_out, _ = sobel(out, ksize)
    sc = edge_scores(edge_map_topk(mag_ref, density), edge_map_topk(mag_out, density))
    sd = float(out.std())
    row.update(mean_grad=float(mag_out.mean()),
               mean_grad_norm=float(mag_out.mean() / sd) if sd > 0 else float("nan"),
               edge_density_otsu=float(edge_map_otsu(mag_out).mean()),
               grad_corr=grad_corr(mag_ref, mag_out),
               precision_topk=sc["precision"], recall_topk=sc["recall"],
               f1_topk=sc["f1"], iou_topk=sc["iou"])
    if has_gt:
        row.update(mse=mse(ref_goc, out), psnr=psnr(ref_goc, out))
    else:
        row.update(mse=float("nan"), psnr=float("nan"))
    return row
