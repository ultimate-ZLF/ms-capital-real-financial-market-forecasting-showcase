"""因子组合检验：等权 rank 组合、OLS 组合、交互项，逐月 cos。"""
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
print("months:", months)

# 交互项：失衡在价差窄时更有效（流动性好，信息冲击传导快）
for f in ["t_imb_2s", "t_imb_5s", "t_imb_w10", "ofi_w10_norm"]:
    df = df.with_columns((pl.col(f) / pl.col("spread0")).alias(f + "_div_spread"))

def ranknorm(x):
    """秩归一化到 [-1, 1]（零均值），NaN 保持 NaN。
    分母必须用 n-1 而非 r.max()-1（极端值并列时 r.max() < n）。"""
    out = np.full(len(x), np.nan)
    m = ~np.isnan(x)
    n = m.sum()
    if n < 100:
        return out
    from scipy.stats import rankdata
    r = rankdata(x[m])
    out[m] = 2 * (r - 1) / (n - 1) - 1
    return out

def monthly_cos(pred, tgt, months_arr):
    cos_l = []
    for mm in months:
        m = (months_arr == mm)
        p, t = pred[m], tgt[m]
        ok = ~(np.isnan(p) | np.isnan(t))
        if ok.sum() > 100 and np.sqrt((p[ok]**2).sum()) > 0:
            cos_l.append(float((p[ok] * t[ok]).sum() / np.sqrt((p[ok]**2).sum() * (t[ok]**2).sum())))
        else:
            cos_l.append(np.nan)
    return np.array(cos_l)

tgt = df["target"].to_numpy()
marr = df["month"].to_numpy()

def prep(fs, suffix=""):
    return np.column_stack([ranknorm(df[f].to_numpy()) for f in fs])

combos = {
    "A: t_imb_2s only": ["t_imb_2s"],
    "B: t_imb_2s + t_imb_w10": ["t_imb_2s", "t_imb_w10"],
    "C: + ofi_w10_norm": ["t_imb_2s", "t_imb_w10", "ofi_w10_norm"],
    "C2: C + patience_diff": ["t_imb_2s", "t_imb_w10", "ofi_w10_norm", "patience_diff"],
    "C3: C2 + aggr_imb": ["t_imb_2s", "t_imb_w10", "ofi_w10_norm", "patience_diff", "aggr_imb"],
    "C4: C2 + imp_depth_5": ["t_imb_2s", "t_imb_w10", "ofi_w10_norm", "patience_diff", "imp_depth_5"],
    "C5: C2 + aggr + imp": ["t_imb_2s", "t_imb_w10", "ofi_w10_norm", "patience_diff",
                            "aggr_imb", "imp_depth_5"],
    "V: patience_diff only": ["patience_diff"],
    "F: 交互项 set": ["t_imb_2s_div_spread", "t_imb_w10_div_spread", "ofi_w10_norm_div_spread"],
    "H: 方向投票(sign平均)": ["t_imb_2s", "t_imb_w10", "ofi_w10_norm", "patience_diff"],
}

# 正交性检查：patience_diff 与 C 组合各因子的 pooled 相关
print("\n### patience_diff 正交性（pooled corr）###")
for f in ["t_imb_2s", "t_imb_w10", "ofi_w10_norm", "aggr_imb", "imp_depth_5"]:
    x = df["patience_diff"].to_numpy()
    y = df[f].to_numpy()
    ok = ~(np.isnan(x) | np.isnan(y))
    print(f"  corr(patience_diff, {f}) = {np.corrcoef(x[ok], y[ok])[0,1]:.4f}")

print("\n### 等权 rank 组合（cos 逐月）###")
print(f"{'combo':<30} {'cos_mean':>9} {'cos_std':>8} {'cos_min':>8} {'neg_months':>10}")
results = {}
for name, fs in combos.items():
    X = prep(fs)
    pred = np.nanmean(X, axis=1) if name != "H: 方向投票(sign平均)" else np.nanmean(np.sign(X), axis=1)
    cl = monthly_cos(pred, tgt, marr)
    results[name] = cl
    print(f"{name:<30} {np.nanmean(cl):9.5f} {np.nanstd(cl):8.5f} {np.nanmin(cl):8.5f} {int((cl < 0).sum()):>10}")

# OLS 权重（全部月份 pooled 拟合，看系数是否与理论符号一致）
print("\n### OLS 拟合权重（pooled，非严格样本外，仅看符号）###")
OLS_FS = ["t_imb_2s", "t_imb_w10", "ofi_w10_norm", "patience_diff", "aggr_imb"]
X = prep(OLS_FS)
ok = ~np.isnan(X).any(axis=1)
Xc, tc = X[ok], tgt[ok]
from numpy.linalg import lstsq
w, *_ = lstsq(Xc, tc, rcond=None)
for f, wi in zip(OLS_FS, w):
    print(f"  {f:<18} w={wi:+.4f}")

# OLS 组合的逐月 cos（in-sample 参考）
pred = np.full(len(tgt), np.nan)
pred[ok] = Xc @ w
cl = monthly_cos(pred, tgt, marr)
print(f"OLS 组合 cos: mean={np.nanmean(cl):.5f} min={np.nanmin(cl):.5f}")

# 组合 C 逐月明细
print("\n### 组合 C 逐月 cos ###")
for mm, c in zip(months, results["C: + ofi_w10_norm"]):
    print(f"month {mm}: cos={c:.5f}")
