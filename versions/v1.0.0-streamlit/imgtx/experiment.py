"""Chạy một phép biến đổi / quét tham số và trả về DataFrame."""
import numpy as np
import pandas as pd

from .analysis import evaluate
from .transforms import TRANSFORMS, apply_lut

NAN = float("nan")


def run_one(name, ref_goc, ref_in, density=0.10, **params):
    out = apply_lut(ref_in, TRANSFORMS[name], **params)
    row = evaluate(ref_goc, ref_in, out, density)
    if name == "negative":  # đảo mức xám là chủ ý: MSE/PSNR vô nghĩa
        row["mse"] = row["psnr"] = NAN
    row.update(transform=name, **params)
    return out, row


def sweep_log(ref_goc, ref_in, gains, density=0.10):
    return pd.DataFrame([run_one("log", ref_goc, ref_in, density, gain=g)[1] for g in gains])


def sweep_gamma(ref_goc, ref_in, gammas, c=1.0, density=0.10):
    return pd.DataFrame([run_one("gamma", ref_goc, ref_in, density, gamma=g, c=c)[1] for g in gammas])


def grid_gamma_c(ref_goc, ref_in, gammas, cs, density=0.10):
    return pd.DataFrame([run_one("gamma", ref_goc, ref_in, density, gamma=g, c=c)[1]
                         for g in gammas for c in cs])


W_SCORE = (0.4, 0.4, 0.2)  # F1, grad_corr, entropy/8


def add_score(df, w=W_SCORE):
    df = df.copy()
    df["score"] = w[0] * df["f1_topk"] + w[1] * df["grad_corr"] + w[2] * df["entropy"] / 8.0
    return df
