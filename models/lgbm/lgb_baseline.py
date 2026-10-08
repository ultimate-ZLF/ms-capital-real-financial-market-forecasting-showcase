"""LGBM 基线 v1：全因子回归，按月分组 CV（每 6 月取 1 验证月，12 折），强正则化。

闸门（v3 教训）：
1. 逐月 cos 均值 ≥ 公式 v2 的 0.0868
2. 逐月 cos 全为正、std 不高于公式的 0.014
3. 特征重要性结构与因子检验一致（w10/t_imb 系应靠前）
通过闸门才生成 test 提交。
"""
import os
import glob
import numpy as np
import polars as pl
import lightgbm as lgb

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
OUT = os.path.join(BASE, "factors")
files = sorted(glob.glob(os.path.join(OUT, "factors_m*.parquet")))
df = pl.concat([pl.read_parquet(f) for f in files])
months = sorted(df["month"].unique().to_list())
print(f"loaded: {df.height} rows, {len(months)} months")

FEATS = [
    # v1 失衡族
    "t_imb", "t_imb_w30", "t_imb_w10", "t_imb_2s", "t_imb_5s", "t_size_imb",
    "ofi", "ofi_w30", "ofi_w10", "ofi_norm", "ofi_w30_norm", "ofi_w10_norm", "ofi_5s",
    # v1 盘口/动量/波动
    "book_imb0", "book_imb60", "d_book_imb", "spread0", "spread_mean60",
    "d60_mid", "d600_mid", "d60_micro",
    "mid_vol60", "mid_vol600", "sig_txn",
    # v1 活动量
    "t_n", "t_vol", "o_n", "o_vol",
    # v2 非共识
    "imp_depth_60", "imp_depth_5", "aggr_imb", "spread_cross_frac",
    "cancel_imb", "patience_diff",
    "range_pos", "curv", "vwap_dev60", "vwap_dev600", "vol_skew",
    "last_gap", "burst5", "accel",
    "spread_sqz", "depth_flow", "vol_ratio",
]

# 微价失衡（盘口加权 mid 偏移）
df = df.with_columns([
    (pl.col("micro") - pl.col("mid")).alias("micro_imb0"),
    (pl.col("micro60") - pl.col("mid60")).alias("micro_imb60"),
])
FEATS += ["micro_imb0", "micro_imb60"]

X = df.select(FEATS).to_numpy()
y = df["target"].to_numpy()
marr = df["month"].to_numpy()
print(f"features: {len(FEATS)}")

params = dict(
    objective="regression",      # L2 → 拟合 E[target|X]，cos 下方向正确
    metric="rmse",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=500,       # 强正则：防 v3 式过拟合
    feature_fraction=0.5,        # 抗共线
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
    seed=42,
)

VAL_MONTHS = months[2::6]  # 3, 9, 15, ..., 69 —— 12 折
results = []
imps = []
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
    print(f"[fold] val month={vm:3d} cos={cos:.5f} iters={model.best_iteration}", flush=True)

cos_arr = np.array([r[1] for r in results])
print(f"\n### LGBM 逐月 cos（{len(results)} 折）###")
print(f"mean={cos_arr.mean():.5f} std={cos_arr.std():.5f} "
      f"min={cos_arr.min():.5f} neg={int((cos_arr < 0).sum())}")
print(f"公式基准：v1C=0.0815 / v2调制=0.0868")

imp = np.array(imps)
imp_n = imp / imp.sum(axis=1, keepdims=True)
print("\n### 特征重要性 top 20（gain，12 折均值）###")
order = np.argsort(imp_n.mean(axis=0))[::-1][:20]
for i in order:
    print(f"  {FEATS[i]:<22} {imp_n.mean(axis=0)[i]:.4f}")
