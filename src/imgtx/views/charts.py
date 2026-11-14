"""Biểu đồ Matplotlib (dùng matplotlib.figure.Figure để nhúng được vào giao diện và lưu PNG)."""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..models import transforms as T
from ..models.analysis import edge_map_topk, histogram, sobel

BLUE, ORANGE, GREEN, RED, GRAY = "#2a6fdb", "#e8833a", "#2e9e6b", "#d64545", "#888888"


def _fig(w, h):
    return Figure(figsize=(w, h), dpi=100, layout="tight")


def save_fig(fig, path, dpi=150):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi)


# ----------------------------------------------------------------- đường cong T(r) (Hình 1)
def plot_curves(gain=1.0, gamma=1.0, c=1.0, fig=None):
    fig = fig or _fig(5, 4)
    fig.clear()
    ax = fig.add_subplot(111)
    r = np.arange(256, dtype=np.uint8)
    ax.plot(r, r, ":", color=GRAY, lw=1, label="identity")
    ax.plot(r, T.negative(r), color=RED, label="âm bản")
    ax.plot(r, T.log_transform(r, gain), color=GREEN, label=f"log (gain={gain:g})")
    ax.plot(r, T.gamma_transform(r, gamma, c), color=BLUE, lw=2, label=f"gamma (γ={gamma:g}, c={c:g})")
    for g in (0.3, 0.5, 2.0, 3.0):
        ax.plot(r, T.gamma_transform(r, g), lw=0.8, alpha=0.45, ls="--", color=BLUE)
    ax.set(xlabel="r (mức xám vào)", ylabel="s = T(r)", xlim=(0, 255), ylim=(0, 255),
           title="Đường cong biến đổi (nét đứt: γ = 0.3, 0.5, 2, 3)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    return fig


# ----------------------------------------------------------------- histogram
def plot_hist_overlay(images: dict, fig=None):
    fig = fig or _fig(7, 3)
    fig.clear()
    ax = fig.add_subplot(111)
    colors = [GRAY, ORANGE, BLUE, GREEN, RED]
    for (name, img), col in zip(images.items(), colors):
        _, p = histogram(img)
        ax.fill_between(np.arange(256), p, step="mid", alpha=0.25, color=col)
        ax.plot(np.arange(256), p, lw=1.1, color=col, label=f"{name} (μ={img.mean():.0f}, σ={img.std():.0f})")
    ax.set(xlim=(0, 255), xlabel="mức xám", ylabel="tần suất p(r)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    return fig


def plot_hists(images: dict, fig=None):
    n = len(images)
    fig = fig or _fig(3.0 * n, 2.4)
    fig.clear()
    for i, (name, img) in enumerate(images.items()):
        ax = fig.add_subplot(1, n, i + 1)
        _, p = histogram(img)
        ax.bar(np.arange(256), p, width=1.0, color=BLUE)
        ax.axvline(img.mean(), color=RED, ls="--", lw=1)
        ax.set_title(name, fontsize=8)
        ax.set_xlim(0, 255)
        ax.tick_params(labelsize=7)
    return fig


# ----------------------------------------------------------------- TN1: lưới ảnh – histogram – biên
def plot_tn1_grid(ref_in, outs: dict, density=0.10, ref_goc=None, fig=None):
    """Hàng 1: ảnh; hàng 2: histogram; hàng 3: biên Sobel top-k."""
    cols = ({"Ảnh gốc": ref_goc} if ref_goc is not None else {}) | {"Ảnh đầu vào": ref_in} | outs
    n = len(cols)
    fig = fig or _fig(2.6 * n, 7.2)
    fig.clear()
    for j, (name, img) in enumerate(cols.items()):
        ax = fig.add_subplot(3, n, j + 1)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
        ax = fig.add_subplot(3, n, n + j + 1)
        _, p = histogram(img)
        ax.bar(np.arange(256), p, width=1.0, color=BLUE)
        ax.set_xlim(0, 255)
        ax.tick_params(labelsize=6)
        ax = fig.add_subplot(3, n, 2 * n + j + 1)
        ax.imshow(edge_map_topk(sobel(img)[0], density), cmap="gray")
        ax.axis("off")
    return fig


# ----------------------------------------------------------------- quét tham số
def plot_sweep(df, xcol, panels, baseline=None, current=None, fig=None, logx=False):
    """panels: list các list (cột, nhãn); baseline: dòng identity (vẽ đường chấm)."""
    fig = fig or _fig(3.8 * len(panels), 3)
    fig.clear()
    for i, cols in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1)
        for col, label in cols:
            ln = ax.plot(df[xcol], df[col], marker="o", ms=3, label=label)[0]
            if baseline is not None and col in baseline and pd.notna(baseline[col]):
                ax.axhline(baseline[col], ls=":", lw=1, color=ln.get_color())
        if current is not None:
            ax.axvline(current, color=GRAY, ls="--", lw=0.8)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xcol, fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    return fig


def plot_heatmap(df, value, title, fig=None, ax=None, cmap="viridis"):
    pv = df.pivot_table(index="gamma", columns="c", values=value, aggfunc="mean")
    if ax is None:
        fig = fig or _fig(5.2, 4)
        fig.clear()
        ax = fig.add_subplot(111)
    im = ax.imshow(pv.values, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(len(pv.columns)), [f"{x:g}" for x in pv.columns], fontsize=7)
    ax.set_yticks(range(len(pv.index)), [f"{x:g}" for x in pv.index], fontsize=7)
    ax.set_xlabel("c", fontsize=8)
    ax.set_ylabel("γ", fontsize=8)
    ax.set_title(title, fontsize=9)
    vmax = np.nanmax(pv.values) if np.isfinite(pv.values).any() else 1
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}", ha="center", va="center",
                        fontsize=5.5, color="w" if v < 0.6 * vmax else "k")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return ax.figure


def plot_heatmaps(df, values, fig=None):
    """TN7: nhiều heatmap (γ × c) cạnh nhau."""
    fig = fig or _fig(4.2 * len(values), 3.8)
    fig.clear()
    for i, (col, title) in enumerate(values):
        ax = fig.add_subplot(1, len(values), i + 1)
        plot_heatmap(df, col, title, ax=ax)
    return fig


def plot_metric_by_param(df, xcol, metrics, by="degrade", fig=None):
    """TN2/TN3: giá trị trung bình theo tham số, mỗi đường là một kiểu suy giảm."""
    fig = fig or _fig(3.8 * len(metrics), 3.1)
    fig.clear()
    for i, (col, label) in enumerate(metrics):
        ax = fig.add_subplot(1, len(metrics), i + 1)
        for key, g in df.groupby(by):
            m = g.groupby(xcol)[col].mean()
            ax.plot(m.index, m.values, marker="o", ms=3, label=str(key))
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(xcol, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    fig.axes[0].legend(fontsize=7)
    return fig


def plot_bar(series: pd.Series, title, ylabel, fig=None):
    fig = fig or _fig(5, 3)
    fig.clear()
    ax = fig.add_subplot(111)
    ax.bar([str(i) for i in series.index], series.values, color=BLUE)
    for i, v in enumerate(series.values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    return fig
