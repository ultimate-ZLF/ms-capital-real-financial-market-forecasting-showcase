"""lgb_newfeat_all.py — LGBM 全 35 新因子对照实验（不做单因子筛选）。

假设（用户提出）：单因子 corr/符号稳定性是公式阶段的闸门，对树模型过于严格——
弱相关因子可能通过条件信号/因子交互提供增量（我们自己的先例：d600_mid 线性≈0
但 LGBM 中有实权）。本实验把全部 35 个新因子喂给树，让 12 折 CV 当裁判。
对比：11 因子筛选版 CV 0.1290（lgb_newfeat.py）。
"""
import glob
import os

import lightgbm as lgb
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
newf = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new_m*.parquet")))])
months = sorted(df["month"].unique().to_list())
print(f"loaded: {df.height} rows, {len(months)} months", flush=True)

# 全部 35 个新因子（来自 new_factors.py 六族）
NEW_FEATS = [
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
assert len(NEW_FEATS) == 35

df = df.join(newf.select(["sample_id", "month"] + NEW_FEATS),
             on=["sample_id", "month"], how="left")

FEATS = [
    "t_imb", "t_imb_w30", "t_imb_w10", "t_imb_2s", "t_imb_5s", "t_size_imb",
    "ofi", "ofi_w30", "ofi_w10", "ofi_norm", "ofi_w30_norm", "ofi_w10_norm", "ofi_5s",
    "book_imb0", "book_imb60", "d_book_imb", "spread0", "spread_mean60",
    "d60_mid", "d600_mid", "d60_micro",
    "mid_vol60", "mid_vol600", "sig_txn",
    "t_n", "t_vol", "o_n", "o_vol",
    "imp_depth_60", "imp_depth_5", "aggr_imb", "spread_cross_frac",
    "cancel_imb", "patience_diff",
    "range_pos", "curv", "vwap_dev60", "vwap_dev600", "vol_skew",
    "last_gap", "burst5", "accel",
    "spread_sqz", "depth_flow", "vol_ratio",
] + NEW_FEATS

df = df.with_columns([
    (pl.col("micro") - pl.col("mid")).alias("micro_imb0"),
    (pl.col("micro60") - pl.col("mid60")).alias("micro_imb60"),
])
FEATS += ["micro_imb0", "micro_imb60"]

X = df.select(FEATS).to_numpy()
y = df["target"].to_numpy()
marr = df["month"].to_numpy()
sids = df["sample_id"].to_numpy()
print(f"features: {len(FEATS)}", flush=True)

params = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=500,
    feature_fraction=0.5,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
    seed=42,
)

VAL_MONTHS = months[2::6]
results = []
imps = []
oof_rows = []
for vm in VAL_MONTHS:
    tr = marr != vm
    va = marr == vm
    dtr = lgb.Dataset(X[tr], y[tr])
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = model.predict(X[va], num_iteration=model.best_iteration)
    yv = y[va]
    cos = float((pred * yv).sum() / np.sqrt((pred**2).sum() * (yv**2).sum()))
    results.append((vm, cos, model.best_iteration))
    imps.append(model.feature_importance("gain"))
    oof_rows.append(pl.DataFrame({
        "sample_id": sids[va], "month": marr[va], "pred": pred.astype(np.float32)}))
    print(f"[fold] val month={vm:3d} cos={cos:.5f} iters={model.best_iteration}", flush=True)

cos_arr = np.array([r[1] for r in results])
print(f"\n### LGBM+全35新因子 逐月 cos（{len(results)} 折）###")
print(f"mean={cos_arr.mean():.5f} std={cos_arr.std():.5f} "
      f"min={cos_arr.min():.5f} neg={int((cos_arr < 0).sum())}")
print(f"参考：11 因子筛选版 = 0.1290；v4 基线（47 因子）= 0.1277")

oof = pl.concat(oof_rows)
oof.write_parquet(os.path.join(OUT, "lgb_all35_oof.parquet"))
print(f"OOF 已存 factors/lgb_all35_oof.parquet（{oof.height} 行）", flush=True)

imp = np.array(imps)
imp_n = imp / imp.sum(axis=1, keepdims=True)
print("\n### 特征重要性 top 30（gain，12 折均值）###")
order = np.argsort(imp_n.mean(axis=0))[::-1][:30]
for i in order:
    tag = "  <- 新" if FEATS[i] in NEW_FEATS else ""
    print(f"  {FEATS[i]:<24} {imp_n.mean(axis=0)[i]:.4f}{tag}")
