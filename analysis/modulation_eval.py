"""幅度调制实验：pred = combo_C × (1 + k·σ̂)，σ̂ 为 |target| 预测器的秩组合。
另计算"完美调制"上界：combo × rank(|target|)。"""
import os
import glob
import numpy as np
import polars as pl
from scipy.stats import rankdata

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
OUT = os.path.join(BASE, "factors")
files = sorted(glob.glob(os.path.join(OUT, "factors_m*.parquet")))
df = pl.concat([pl.read_parquet(f) for f in files])
months = sorted(df["month"].unique().to_list())
tgt = df["target"].to_numpy()
marr = df["month"].to_numpy()

def ranknorm(x):
    out = np.full(len(x), np.nan)
    m = ~np.isnan(x)
    n = m.sum()
    if n > 1:
        r = rankdata(x[m])
        out[m] = 2 * (r - 1) / (n - 1) - 1
    return out

def monthly_cos(pred):
    cos_l = []
    for mm in months:
        mask = marr == mm
        p, t = pred[mask], tgt[mask]
        ok = ~(np.isnan(p) | np.isnan(t))
        if ok.sum() > 100 and np.sqrt((p[ok]**2).sum()) > 0:
            cos_l.append(float((p[ok] * t[ok]).sum() / np.sqrt((p[ok]**2).sum() * (t[ok]**2).sum())))
        else:
            cos_l.append(np.nan)
    return np.array(cos_l)

def report(name, pred):
    cl = monthly_cos(pred)
    print(f"{name:<34} cos_mean={np.nanmean(cl):.5f} cos_min={np.nanmin(cl):.5f} neg={int((cl < 0).sum())}")

# 组合 C（方向基）
Xc = np.column_stack([ranknorm(df[f].to_numpy()) for f in ["t_imb_2s", "t_imb_w10", "ofi_w10_norm"]])
combo = np.nanmean(Xc, axis=1)
report("C (baseline)", combo)

# 完美调制上界
report("C × rank(|target|) [上界]", combo * ranknorm(np.abs(tgt)))

# |target| 预测器
POS = ["mid_vol60", "sig_txn", "t_n", "o_n", "spread_cross_frac", "vol_ratio", "vwap_dev60"]
NEG = ["burst5", "last_gap"]

def sig_hat(fs_pos, fs_neg):
    cols = [ranknorm(df[f].to_numpy()) for f in fs_pos] + \
           [-ranknorm(df[f].to_numpy()) for f in fs_neg]
    return np.nanmean(np.column_stack(cols), axis=1)

print()
for label, pos, neg in [("σ̂=mid_vol60", ["mid_vol60"], []),
                        ("σ̂=t_n", ["t_n"], []),
                        ("σ̂=stack_pos7", POS, []),
                        ("σ̂=stack_pos7+neg2", POS, NEG)]:
    sig = sig_hat(pos, neg)
    for k in [0.25, 0.5, 1.0]:
        pred = combo * (1 + k * sig)
        report(f"C × (1+{k}·{label})", pred)

# 只用幅度预测器本身（纯波动率预测，验证能否单独贡献 cos）
report("σ̂=stack_pos7 alone", sig_hat(POS, []))

# --- 第二轮：更多 σ̂ 候选 + k 微调 ---
print()
for label, pos, neg in [("σ̂=o_n", ["o_n"], []),
                        ("σ̂=t_vol", ["t_vol"], []),
                        ("σ̂=t_n+o_n", ["t_n", "o_n"], []),
                        ("σ̂=spread_cross_frac", ["spread_cross_frac"], []),
                        ("σ̂=vol_ratio", ["vol_ratio"], []),
                        ("σ̂=t_n+vol_ratio", ["t_n", "vol_ratio"], [])]:
    sig = sig_hat(pos, neg)
    for k in [0.5]:
        pred = combo * (1 + k * sig)
        report(f"C × (1+{k}·{label})", pred)

# k 微调（t_n）
sig = sig_hat(["t_n"], [])
for k in [0.35, 0.45, 0.55, 0.65, 0.8]:
    report(f"C × (1+{k}·σ̂=t_n)", combo * (1 + k * sig))

# C4（含 v2 因子）作为方向基 + t_n 调制
Xc4 = np.column_stack([ranknorm(df[f].to_numpy()) for f in
                       ["t_imb_2s", "t_imb_w10", "ofi_w10_norm", "patience_diff", "imp_depth_5"]])
combo4 = np.nanmean(Xc4, axis=1)
report("C4 (v2 扩展) baseline", combo4)
report("C4 × (1+0.5·σ̂=t_n)", combo4 * (1 + 0.5 * sig))

# --- 第三轮：t_vol 的 k 微调 + 组合 ---
print()
sigv = sig_hat(["t_vol"], [])
for k in [0.35, 0.45, 0.55, 0.65, 0.8]:
    report(f"C × (1+{k}·σ̂=t_vol)", combo * (1 + k * sigv))
sig_stack = sig_hat(["t_vol", "t_n"], [])
report("C × (1+0.5·σ̂=t_vol+t_n)", combo * (1 + 0.5 * sig_stack))
