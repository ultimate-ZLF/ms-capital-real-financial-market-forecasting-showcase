"""v2 因子正确性抽查：asof join 方向 + 月 66 新因子相关性。"""
import polars as pl
import numpy as np

import os
BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)

# --- asof 方向验证：样本 0 的成交 → 匹配快照 sec 应 >= 成交 sec ---
mt = pl.read_parquet(BASE + r"/train/market.parquet")
tt = pl.read_parquet(BASE + r"/train/transaction.parquet")
t0 = tt.filter(pl.col("sample_id") == 0).sort("seconds_before_predict")
m0 = (mt.filter(pl.col("sample_id") == 0)
      .select(["sample_id", "seconds_before_predict", "ask_price_1", "bid_price_1"])
      .sort("seconds_before_predict"))
j = t0.join_asof(m0, on="seconds_before_predict", by="sample_id", strategy="forward")
print("columns:", j.columns)
print(j.head(8))
# 验证：结果中的 seconds_before_predict 是否仍是成交时刻（= t0 原值）
print("kept sec == original trade sec:",
      bool((j["seconds_before_predict"].head(8).to_numpy()
            == t0["seconds_before_predict"].head(8).to_numpy()).all()))

# --- 新因子相关性（月 66）---
df = pl.read_parquet(BASE + r"/factors/factors_m66.parquet")
new_f = ["imp_depth_60", "imp_depth_5", "aggr_imb", "spread_cross_frac",
         "cancel_imb", "patience_diff",
         "range_pos", "curv", "vwap_dev60", "vwap_dev600", "vol_skew",
         "last_gap", "burst5", "accel",
         "spread_sqz", "depth_flow", "vol_ratio"]
t = df["target"].to_numpy()
print(f"\n{'factor':<20} {'corr(tgt)':>10} {'corr(sign)':>10} {'corr(abs)':>10} {'nan%':>7}")
for f in new_f:
    x = df[f].to_numpy()
    ok = ~(np.isnan(x) | np.isnan(t))
    if ok.sum() < 100:
        print(f"{f:<20} insufficient")
        continue
    c1 = np.corrcoef(x[ok], t[ok])[0, 1]
    c2 = np.corrcoef(x[ok], np.sign(t[ok]))[0, 1]
    c3 = np.corrcoef(x[ok], np.abs(t[ok]))[0, 1]
    print(f"{f:<20} {c1:10.4f} {c2:10.4f} {c3:10.4f} {np.isnan(x).mean()*100:7.2f}")
