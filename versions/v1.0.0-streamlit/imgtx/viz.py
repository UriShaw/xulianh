"""Biểu đồ Matplotlib: histogram, đường cong T(r), quét tham số, heatmap."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import transforms as T
from .analysis import histogram


def plot_hists(images: dict):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.4))
    axes = np.atleast_1d(axes)
    for ax, (name, img) in zip(axes, images.items()):
        _, p = histogram(img)
        ax.bar(np.arange(256), p, width=1.0, color="#4c78a8")
        ax.axvline(img.mean(), color="red", ls="--", lw=1)
        ax.set_title(name, fontsize=8)
        ax.set_xlim(0, 255)
        ax.set_xlabel("mức xám", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("tần suất", fontsize=8)
    fig.tight_layout()
    return fig


def plot_hist_overlay(images: dict):
    fig, ax = plt.subplots(figsize=(7, 3))
    for name, img in images.items():
        _, p = histogram(img)
        ax.plot(np.arange(256), p, lw=1.2, label=name)
    ax.set_xlim(0, 255)
    ax.set_xlabel("mức xám")
    ax.set_ylabel("tần suất")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_curves(gain, gamma, c):
    r = np.arange(256, dtype=np.uint8)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(r, r, "k:", lw=1, label="identity")
    ax.plot(r, T.negative(r), label="âm bản")
    ax.plot(r, T.log_transform(r, gain), label=f"log (gain={gain:g})")
    ax.plot(r, T.gamma_transform(r, gamma, c), label=f"gamma (γ={gamma:g}, c={c:g})")
    for g in (0.3, 2.0):
        ax.plot(r, T.gamma_transform(r, g), lw=0.8, alpha=0.5, ls="--", label=f"gamma γ={g:g}, c=1")
    ax.set_xlabel("r (vào)")
    ax.set_ylabel("s = T(r) (ra)")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sweep(df, xcol, panels, baseline=None, current=None):
    """panels: list các list (cột, nhãn); baseline: dòng identity (đường chấm)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 3))
    axes = np.atleast_1d(axes)
    for ax, cols in zip(axes, panels):
        for col, label in cols:
            ln = ax.plot(df[xcol], df[col], marker="o", ms=3, label=label)[0]
            if baseline is not None and col in baseline and pd.notna(baseline[col]):
                ax.axhline(baseline[col], ls=":", lw=1, color=ln.get_color())
        if current is not None:
            ax.axvline(current, color="gray", ls="--", lw=0.8)
        ax.set_xlabel(xcol, fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_heatmap(df, value, title):
    pv = df.pivot(index="gamma", columns="c", values=value)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    im = ax.imshow(pv.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(pv.columns)), [f"{x:g}" for x in pv.columns])
    ax.set_yticks(range(len(pv.index)), [f"{x:g}" for x in pv.index])
    ax.set_xlabel("c")
    ax.set_ylabel("γ")
    ax.set_title(title, fontsize=9)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
