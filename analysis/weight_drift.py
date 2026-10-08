"""诊断：逐月 OLS 权重是否随时间/市场状态漂移。
若漂移 → 拟合权重是 regime 特化的，解释了 v3 的 CV→LB 崩塌。"""
import os
import glob
import numpy as np
import polars as pl

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
OUT = os.path.join(BASE, "factors")
files = sorted(glob.glob(os.path.join(OUT, "factors_m*.parquet")))
df = pl.concat([pl.read_parquet(f) for f in files])
months = sorted(df["month"].unique().to_list())
FS = ["t_imb_2s", "t_imb_w10", "ofi_w10_norm", "patience_diff"]

# 逐月 rank（组内秩，避免跨月量纲差异）
df = df.with_columns([
    pl.col(f).rank().over("month") for f in FS
])
X = df.select(FS).to_numpy()
tgt = df["target"].to_numpy()
marr = df["month"].to_numpy()

W = []
for mm in months:
    mask = marr == mm
    ok = ~np.isnan(X[mask]).any(axis=1)
    w, *_ = np.linalg.lstsq(X[mask][ok], tgt[mask][ok], rcond=None)
    W.append(w)
W = np.array(W)
Wn = W / W.sum(axis=1, keepdims=True)  # 归一化看比例

print("逐月 OLS 权重（归一化比例，71 个月）：")
print(f"{'factor':<18} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} {'负权重月数':>8}")
for i, f in enumerate(FS):
    print(f"{f:<18} {Wn[:, i].mean():7.3f} {Wn[:, i].std():7.3f} "
          f"{Wn[:, i].min():7.3f} {Wn[:, i].max():7.3f} {int((W[:, i] < 0).sum()):>8}")

# 权重与市场状态的相关：活跃度（t_n 均值）与 w10 权重
t_n = df["t_n"].to_numpy()
act = np.array([np.nanmean(t_n[marr == mm]) for mm in months])
for i, f in enumerate(FS):
    c = np.corrcoef(Wn[:, i], act)[0, 1]
    print(f"corr(w({f}), 月均成交笔数) = {c:+.3f}")

# 前 20 月 vs 后 20 月的权重均值对比
print("\n前 20 月 vs 后 20 月权重均值：")
for i, f in enumerate(FS):
    print(f"  {f:<18} {Wn[:20, i].mean():.3f}  vs  {Wn[-20:, i].mean():.3f}")
