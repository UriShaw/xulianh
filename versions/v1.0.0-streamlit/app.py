"""Khảo sát các phép biến đổi âm bản, logarit, gamma + so sánh biên Sobel.

Chạy:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from imgtx import io_utils as iou
from imgtx import viz
from imgtx.analysis import edge_map_topk, sobel
from imgtx.degrade import DEGRADES
from imgtx.experiment import W_SCORE, add_score, grid_gamma_c, run_one, sweep_gamma, sweep_log

st.set_page_config(page_title="Khảo sát âm bản – logarit – gamma", layout="wide")

GAMMAS = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
GAINS = [0.5, 0.75, 1.0, 1.25, 1.5]
CS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
COLS = {"Phép": "Phép", "mean": "Mean", "std": "Std", "entropy": "Entropy", "dyn_range": "Dải động",
        "sat_high": "Bão hoà 255", "f1_topk": "F1 biên", "iou_topk": "IoU biên",
        "grad_corr": "Tương quan ∇", "mean_grad_norm": "∇ TB chuẩn hoá",
        "edge_density_otsu": "Mật độ biên Otsu", "mse": "MSE", "psnr": "PSNR (dB)"}


@st.cache_data(show_spinner=False)
def c_gamma(goc, vao, c, k):
    return sweep_gamma(goc, vao, GAMMAS, c, k)


@st.cache_data(show_spinner=False)
def c_log(goc, vao, k):
    return sweep_log(goc, vao, GAINS, k)


@st.cache_data(show_spinner=False)
def c_grid(goc, vao, k):
    return grid_gamma_c(goc, vao, GAMMAS, CS, k)


# ------------------------------------------------------------------ Sidebar
sb = st.sidebar
sb.header("1. Nguồn ảnh")
src = sb.radio("Chọn nguồn", ["Thư viện (tải ảnh lên)", "Camera (chụp ảnh)", "Ảnh mẫu"])
rgb = None
try:
    if src.startswith("Thư viện"):
        up = sb.file_uploader("Chọn ảnh trong thư viện", type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"])
        if up:
            rgb = iou.decode_image(up.getvalue())
    elif src.startswith("Camera"):
        shot = sb.camera_input("Chụp ảnh")
        if shot:
            rgb = iou.decode_image(shot.getvalue())
    else:
        rgb = iou.load_sample(sb.selectbox("Ảnh mẫu", iou.SAMPLES))
except ValueError as e:
    sb.error(str(e))

sb.header("2. Tiền xử lý")
max_side = sb.slider("Cạnh dài tối đa (px)", 256, 1024, 512, 64)
cmode = sb.radio("Chế độ màu", ["Xám", "Màu (áp dụng lên kênh V)"])
mode = "v" if cmode.startswith("Màu") else "gray"

sb.header("3. Ảnh tham chiếu")
sim = sb.checkbox("Mô phỏng suy giảm (có ảnh gốc)", value=False,
                  help="Ảnh bạn chọn được coi là ảnh GỐC; ảnh đầu vào = ảnh gốc đã bị suy giảm. "
                       "Khi đó F1/IoU/MSE/PSNR đo mức bảo toàn so với ảnh gốc.")
deg_fn, deg_kw = None, {}
if sim:
    kind = sb.selectbox("Kiểu suy giảm", list(DEGRADES))
    if "Quá sáng" in kind:
        deg_kw = dict(alpha=sb.slider("α", 1.0, 2.5, 1.6, 0.1), beta=sb.slider("β", 0, 100, 40, 5))
    else:
        deg_kw = dict(alpha=sb.slider("α", 0.1, 0.9, 0.4, 0.05), beta=sb.slider("β", 0, 40, 10, 1))
        if "nhiễu" in kind:
            deg_kw["sigma"] = sb.slider("σ nhiễu", 0.0, 15.0, 4.0, 0.5)
    deg_fn = DEGRADES[kind]

sb.header("4. Tham số biến đổi")
gain = sb.slider("Log: gain", 0.2, 2.0, 1.0, 0.05)
gamma = sb.slider("Gamma: γ", 0.1, 4.0, 1.0, 0.05)
cc = sb.slider("Gamma: hệ số c", 0.3, 3.0, 1.0, 0.05)
k = sb.slider("Mật độ biên top-k", 0.02, 0.30, 0.10, 0.01)

# ------------------------------------------------------------------ Chuẩn bị ảnh
st.title("Khảo sát phép biến đổi âm bản, logarit và gamma")
st.caption("Histogram + biên Sobel để đánh giá mức độ bảo toàn chi tiết.")
if rgb is None:
    st.info("Hãy chọn ảnh từ thư viện, chụp bằng camera hoặc dùng ảnh mẫu ở thanh bên.")
    st.stop()

rgb = iou.resize_max(rgb, max_side)
gray = iou.to_working(rgb, mode)
if deg_fn is not None:
    ref_goc, ref_in = gray, deg_fn(gray, **deg_kw)
    st.success("Có ảnh gốc: biên, gradient và MSE/PSNR được so với **ảnh gốc**.")
else:
    ref_goc, ref_in = None, gray
    st.warning("Không có ảnh gốc: tham chiếu là **ảnh đầu vào**. F1/IoU/tương quan ∇ chỉ đo "
               "“độ ổn định biên so với ảnh đầu vào” (identity luôn = 1), không chứng minh chi tiết "
               "được giữ hay khôi phục; MSE/PSNR không có. Bật “Mô phỏng suy giảm” để có ảnh gốc.")


def show(img):
    return iou.to_display(rgb, img, mode)


# ------------------------------------------------------------------ Tính 4 dòng
items = [("Đầu vào (identity)", "identity", {}), ("Âm bản", "negative", {}),
         (f"Log (gain={gain:g})", "log", {"gain": gain}),
         (f"Gamma (γ={gamma:g}, c={cc:g})", "gamma", {"gamma": gamma, "c": cc})]
outs, rows = {}, []
for label, name, p in items:
    o, r = run_one(name, ref_goc, ref_in, k, **p)
    outs[label] = o
    rows.append({"Phép": label, **r})
table = pd.DataFrame(rows)

t1, t2, t3 = st.tabs(["So sánh ba phép", "Ảnh hưởng tham số", "Đường cong T(r)"])

# ------------------------------------------------------------------ Tab 1
with t1:
    cols_img = ([("Ảnh gốc", ref_goc)] if ref_goc is not None else []) + list(outs.items())
    cs = st.columns(len(cols_img))
    for col, (lab, img) in zip(cs, cols_img):
        col.markdown(f"**{lab}**")
        col.image(show(img))
    st.markdown("**Histogram** (đường đỏ = mean)")
    st.pyplot(viz.plot_hists({l: i for l, i in cols_img}))
    st.markdown(f"**Biên Sobel** (top-{k:.0%} điểm có gradient lớn nhất)")
    cs = st.columns(len(cols_img))
    for col, (lab, img) in zip(cs, cols_img):
        mag, _ = sobel(img)
        col.image(edge_map_topk(mag, k).astype(np.uint8) * 255, caption=lab)

    st.markdown("**Bảng chỉ số**")
    st.dataframe(table[list(COLS)].rename(columns=COLS).round(4), hide_index=True)
    st.download_button("Tải bảng chỉ số (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
                       "ket_qua.csv", "text/csv")

    idr, neg = table.iloc[0], table.iloc[1]
    st.markdown("**Kiểm chứng H1 (âm bản bảo toàn gradient)**")
    st.write(f"|F1(âm bản) − F1(identity)| = {abs(neg.f1_topk - idr.f1_topk):.2e}; "
             f"|tương quan ∇(âm bản) − tương quan ∇(identity)| = {abs(neg.grad_corr - idr.grad_corr):.2e}. "
             "Gần 0 ⇒ ủng hộ H1 (|∇(255−f)| = |∇f|).")
    pick = st.selectbox("Tải ảnh kết quả", list(outs))
    st.download_button("Tải PNG", iou.png_bytes(show(outs[pick])), "ket_qua.png", "image/png")

# ------------------------------------------------------------------ Tab 2
with t2:
    has_gt = ref_goc is not None
    p3 = ("psnr", "PSNR (dB)") if has_gt else ("dyn_range", "Dải động")
    panels = [[("mean", "mean")], [("entropy", "entropy")],
              [("f1_topk", "F1"), ("grad_corr", "grad_corr")], [p3], [("sat_high", "tỉ lệ =255")]]
    idrow = table.iloc[0]

    st.subheader(f"Gamma: quét γ (c = {cc:g})")
    dg = c_gamma(ref_goc, ref_in, cc, k)
    st.pyplot(viz.plot_sweep(dg, "gamma", panels, idrow, gamma))
    st.caption("Đường chấm = dòng identity (mốc). Đường đứt xám = γ hiện tại.")
    ov = {f"γ={g:g}": run_one("gamma", ref_goc, ref_in, k, gamma=g, c=cc)[0] for g in (0.3, 0.5, 1.0, 2.0, 3.0)}
    st.pyplot(viz.plot_hist_overlay(ov))

    st.subheader("Logarit: quét gain")
    dl = c_log(ref_goc, ref_in, k)
    st.pyplot(viz.plot_sweep(dl, "gain", panels, idrow, gain))

    st.subheader("Gamma: lưới γ × c")
    dgrid = c_grid(ref_goc, ref_in, k)
    opts = {"F1 biên": "f1_topk", "Tương quan ∇": "grad_corr", "Mean": "mean",
            "Tỉ lệ bão hoà 255": "sat_high"}
    if has_gt:
        opts["PSNR (dB)"] = "psnr"
    h = st.selectbox("Chỉ số", list(opts))
    st.pyplot(viz.plot_heatmap(dgrid, opts[h], h))

    st.subheader("Gợi ý tham số")
    sc = add_score(pd.concat([dgrid, dl], ignore_index=True))
    cols = ["transform", "gamma", "c", "gain", "mean", "dyn_range", "f1_topk", "grad_corr", "entropy",
            "score"] + (["psnr"] if has_gt else [])
    st.write(f"Điểm tổng hợp = {W_SCORE[0]}·F1 + {W_SCORE[1]}·grad_corr + {W_SCORE[2]}·entropy/8 "
             "(trọng số khởi điểm, hãy chỉnh theo mục tiêu).")
    st.dataframe(sc.sort_values("score", ascending=False).head(5)[cols].round(4), hide_index=True)
    if has_gt:
        best = dgrid.loc[dgrid.psnr.idxmax()]
        st.write(f"PSNR cao nhất trên lưới: γ={best.gamma:g}, c={best.c:g} → {best.psnr:.2f} dB "
                 f"(identity: {idrow.psnr:.2f} dB, F1 {best.f1_topk:.3f} so với {idrow.f1_topk:.3f}). "
                 "Nếu tối ưu nằm ở biên lưới, hãy mở rộng lưới.")
    st.caption("F1 so với ảnh gốc đo mức làm hỏng biên, không đo mức cải thiện; "
               "mức cải thiện hiển thị đọc qua PSNR, mean và dải động.")

# ------------------------------------------------------------------ Tab 3
with t3:
    a, b = st.columns([1, 1])
    a.pyplot(viz.plot_curves(gain, gamma, cc))
    b.markdown(
        "- **Âm bản** s = 255 − r, |T′| = 1 ⇒ |∇s| = |∇f|: biên Sobel không đổi.\n"
        "- **Logarit** T′ = c/(1+r) giảm theo r: khuếch đại biên vùng tối, nén biên vùng sáng.\n"
        "- **Gamma** T′ = γ(r/255)^(γ−1): γ<1 khuếch đại vùng tối, γ>1 khuếch đại vùng sáng.\n"
        "- **c / gain** chỉ nhân thang gradient khi chưa bão hoà; c>1 gây cắt ngưỡng 255 và mất biên.\n"
        "- |∇s| = |T′(f)|·|∇f| là xấp xỉ liên tục; lượng tử hoá 8-bit và bão hoà còn làm mất chi tiết thực sự.")
