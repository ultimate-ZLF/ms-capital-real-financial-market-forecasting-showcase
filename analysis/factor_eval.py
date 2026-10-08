"""因子检验：逐月 corr(target)/corr(sign)/corr(|target|) + 符号稳定性 + cos 得分代理。

用法：python factor_eval.py
读取 factors/factors_m*.parquet，输出汇总表。
"""
import os
import glob
import numpy as np
import pandas as pd
import polars as pl

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
OUT = os.path.join(BASE, "factors")

files = sorted(glob.glob(os.path.join(OUT, "factors_m*.parquet")))
if not files:
    raise SystemExit("no factor files found")
df = pl.concat([pl.read_parquet(f) for f in files])
print(f"months: {sorted(df['month'].unique().to_list())}")
print(f"rows: {df.height}")

FACTORS = ["t_imb", "t_imb_w30", "t_imb_w10", "t_imb_2s", "t_imb_5s", "t_size_imb",
           "ofi", "ofi_w30", "ofi_w10", "ofi_norm", "ofi_w30_norm", "ofi_w10_norm", "ofi_5s",
           "d60_mid", "d600_mid", "d60_micro",
           "book_imb0", "d_book_imb", "spread0", "spread_mean60",
           "mid_vol60", "mid_vol600", "sig_txn",
           "o_n", "o_vol", "t_n", "t_vol",
           # v2 非共识
           "imp_depth_60", "imp_depth_5", "aggr_imb", "spread_cross_frac",
           "cancel_imb", "patience_diff",
           "range_pos", "curv", "vwap_dev60", "vwap_dev600", "vol_skew",
           "last_gap", "burst5", "accel",
           "spread_sqz", "depth_flow", "vol_ratio"]

# 波动率归一化版本（理论：pred = β·imb·σ_local）
for f in ["t_imb", "ofi", "ofi_norm", "t_imb_w30", "t_imb_2s"]:
    df = df.with_columns((pl.col(f) * pl.col("sig_txn")).alias(f + "_x_sig"))
    df = df.with_columns((pl.col(f) * pl.col("mid_vol60")).alias(f + "_x_mv60"))
FACTORS = FACTORS + [f + "_x_sig" for f in ["t_imb", "ofi", "ofi_norm", "t_imb_w30", "t_imb_2s"]] \
                  + [f + "_x_mv60" for f in ["t_imb", "ofi", "ofi_norm", "t_imb_w30", "t_imb_2s"]]

months = sorted(df["month"].unique().to_list())
target = df["target"].to_numpy()
sign = np.sign(target)

def r(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 100:
        return np.nan
    return np.corrcoef(x[m], y[m])[0, 1]

rows = []
for f in FACTORS:
    x = df[f].to_numpy()
    cs, c_sign, c_abs = [], [], []
    cos_l = []
    for mm in months:
        mask = (df["month"].to_numpy() == mm)
        xm, tm = x[mask], target[mask]
        cs.append(r(xm, tm))
        c_sign.append(r(xm, sign[mask]))
        c_abs.append(r(xm, np.abs(tm)))
        m2 = ~(np.isnan(xm) | np.isnan(tm))
        if m2.sum() > 100 and np.sqrt((xm[m2]**2).sum()) > 0:
            cos_l.append(float((xm[m2] * tm[m2]).sum() / np.sqrt((xm[m2]**2).sum() * (tm[m2]**2).sum())))
        else:
            cos_l.append(np.nan)
    rows.append({
        "factor": f,
        "corr_mean": np.nanmean(cs), "corr_std": np.nanstd(cs),
        "pos_frac": np.nanmean([c > 0 for c in cs]),
        "cos_mean": np.nanmean(cos_l), "cos_std": np.nanstd(cos_l),
        "corr_sign_mean": np.nanmean(c_sign),
        "corr_abs_mean": np.nanmean(c_abs),
    })

res = pd.DataFrame(rows).sort_values("corr_mean", ascending=False)
pd.set_option("display.width", 200, "display.max_columns", 20)
print("\n### 因子汇总（按月统计）###")
print(res.round(4).to_string(index=False))

# t_imb 逐月 cos 详情（锚定基线）
print("\n### t_imb 逐月 cos ###")
x = df["t_imb"].to_numpy()
for mm in months:
    mask = (df["month"].to_numpy() == mm)
    xm, tm = x[mask], target[mask]
    m2 = ~(np.isnan(xm) | np.isnan(tm))
    c = (xm[m2] * tm[m2]).sum() / np.sqrt((xm[m2]**2).sum() * (tm[m2]**2).sum())
    print(f"month {mm}: cos={c:.5f}  corr={r(xm, tm):.4f}")
