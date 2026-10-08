"""new_factor_eval.py — P0 新因子检验：逐月 corr/sign/abs + 符号稳定性 + 冗余度。

读取 factors/new_m*.parquet（新因子）与 factors/factors_m*.parquet（已有因子锚点），
输出：
1. 新因子汇总表：corr_mean/std、pos_frac（71 月符号稳定性）、cos_mean/std、
   corr_sign_mean、corr_abs_mean —— 与 factor_eval.py 同口径
2. 冗余度：每个新因子与锚点因子的全样本 |corr|（>0.7 标 ⚠）
用法：python new_factor_eval.py
"""
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
OUT = os.path.join(BASE, "factors")

NEW_FACTORS = [
    # L2 盘口档位
    "rel_spread2_0", "imb2_0", "slope_0",
    "ask_slope_mean60", "bid_slope_mean60", "slope_imb60",
    # 秒级静默
    "t_has_data_15", "t_miss_max", "o_has_data_15", "o_miss_max",
    # near/far 半窗（tx）
    "t_price_nf_15", "t_price_nf_30", "t_price_nf_60",
    "t_vol_nf_30", "t_vol_nf_60",
    "t_rc_nf_15", "t_rc_nf_30", "t_rc_nf_60",
    # near/far 半窗（order）
    "o_vol_nf_15", "o_vol_nf_30", "o_vol_nf_60",
    "o_rc_nf_15", "o_rc_nf_30", "o_rc_nf_60",
    "o_cancel_nf_15", "o_cancel_nf_30", "o_cancel_nf_60",
    # 跨流方向差
    "x_to_buy_diff", "x_to_buy_diff_15",
    # 大单计数
    "t_large_buy", "t_large_sell", "x_large_imb",
    # RV 波动率
    "m_rv60", "m_rv600", "x_rv_60_600",
]

# 冗余度锚点（覆盖我们的主要信号族）
ANCHORS = ["t_imb", "ofi", "ofi_norm", "book_imb0", "spread0", "mid_vol60",
           "t_vol", "t_n", "o_vol", "o_n", "cancel_imb", "d60_mid", "vwap_dev60"]

files = sorted(glob.glob(os.path.join(OUT, "new_m*.parquet")))
if not files:
    raise SystemExit("no new factor files found — 先跑 new_factors.py")
df = pl.concat([pl.read_parquet(f) for f in files])
print(f"新因子月份: {sorted(df['month'].unique().to_list())}")
print(f"行数: {df.height}")

# 冗余度：join 已有因子（同 sample_id+month 精确对齐）
anchor_files = sorted(glob.glob(os.path.join(OUT, "factors_m*.parquet")))
if anchor_files:
    anc = pl.concat([pl.read_parquet(f) for f in anchor_files])
    shared_months = sorted(set(df["month"].unique().to_list()) & set(anc["month"].unique().to_list()))
    a_sel = anc.filter(pl.col("month").is_in(shared_months)) \
        .select(["sample_id", "month"] + [c for c in ANCHORS if c in anc.columns])
    df_a = df.filter(pl.col("month").is_in(shared_months)) \
        .join(a_sel, on=["sample_id", "month"], how="left")
    anchors_found = [c for c in ANCHORS if c in anc.columns]
else:
    df_a, anchors_found = df, []

months = sorted(df["month"].unique().to_list())
target = df["target"].to_numpy()
sign = np.sign(target)


def r(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 100:
        return np.nan
    return np.corrcoef(x[m], y[m])[0, 1]


rows = []
for f in NEW_FACTORS:
    if f not in df.columns:
        print(f"⚠ 因子 {f} 不存在于缓存，跳过")
        continue
    x = df[f].to_numpy()
    cs, c_sign, c_abs, cos_l = [], [], [], []
    for mm in months:
        mask = (df["month"].to_numpy() == mm)
        xm, tm = x[mask], target[mask]
        cs.append(r(xm, tm))
        c_sign.append(r(xm, sign[mask]))
        c_abs.append(r(xm, np.abs(tm)))
        m2 = ~(np.isnan(xm) | np.isnan(tm))
        if m2.sum() > 100 and np.sqrt((xm[m2] ** 2).sum()) > 0:
            cos_l.append(float((xm[m2] * tm[m2]).sum()
                               / np.sqrt((xm[m2] ** 2).sum() * (tm[m2] ** 2).sum())))
        else:
            cos_l.append(np.nan)
    # 冗余度
    red = {}
    for a in anchors_found:
        xa = df_a[a].to_numpy()
        c = r(x, xa)
        if c is not None and not np.isnan(c) and abs(c) > 0.7:
            red[a] = round(float(c), 2)
    rows.append({
        "factor": f,
        "corr_mean": np.nanmean(cs), "corr_std": np.nanstd(cs),
        "pos_frac": np.nanmean([c > 0 for c in cs]),
        "cos_mean": np.nanmean(cos_l), "cos_std": np.nanstd(cos_l),
        "corr_sign_mean": np.nanmean(c_sign),
        "corr_abs_mean": np.nanmean(c_abs),
        "null_frac": float(df[f].null_count() / df.height),
        "high_corr_anchors": red,
    })

res = pd.DataFrame(rows).sort_values("cos_mean", ascending=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
print("\n### 新因子汇总（按月统计，cos 降序）###")
print(res.round(4).to_string(index=False))

print("\n### 锚点参考（同口径）###")
for a in anchors_found:
    x = df_a[a].to_numpy()
    cs = [r(x[df["month"].to_numpy() == mm], target[df["month"].to_numpy() == mm])
          for mm in months]
    print(f"{a:16s} corr_mean={np.nanmean(cs):+.4f}  pos_frac={np.nanmean([c > 0 for c in cs]):.2f}")
