"""LGBM 增量实验：47 因子 + P0 新因子批（11 个通过检验的），同 v4 协议 12 折。

与 lgb_baseline.py 唯一差异 = FEATS 加入 new_m*.parquet 的 11 个新因子；
参数/CV 协议/闸门完全一致。OOF 存档 factors/lgb_newfeat_oof.parquet（供三方集成）。
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

NEW_FEATS = [
    # 跨流方向差（71/71 稳定，本批最强）
    "x_to_buy_diff_15", "x_to_buy_diff",
    # near/far 价格半窗
    "t_price_nf_30", "t_price_nf_15",
    # 大单计数/失衡
    "t_large_buy", "x_large_imb",
    # L2 两档失衡（弱但无冗余）
    "imb2_0",
    # RV 波动率（|target| 预测 0.255/0.324）
    "m_rv60", "m_rv600",
    # 静默结构（稳定负相关，树条件信号）
    "t_miss_max", "o_miss_max",
]

df = df.join(newf.select(["sample_id", "month"] + NEW_FEATS),
             on=["sample_id", "month"], how="left")

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
] + NEW_FEATS

# 微价失衡（同 lgb_baseline.py:41-45）
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

VAL_MONTHS = months[2::6]  # 同 v4：12 折
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
print(f"\n### LGBM+新因子 逐月 cos（{len(results)} 折）###")
print(f"mean={cos_arr.mean():.5f} std={cos_arr.std():.5f} "
      f"min={cos_arr.min():.5f} neg={int((cos_arr < 0).sum())}")
print(f"参考：v4 基线（47 因子）= 0.1277")

oof = pl.concat(oof_rows)
oof.write_parquet(os.path.join(OUT, "lgb_newfeat_oof.parquet"))
print(f"OOF 已存 factors/lgb_newfeat_oof.parquet（{oof.height} 行）", flush=True)

imp = np.array(imps)
imp_n = imp / imp.sum(axis=1, keepdims=True)
print("\n### 特征重要性 top 25（gain，12 折均值）###")
order = np.argsort(imp_n.mean(axis=0))[::-1][:25]
for i in order:
    tag = "  <- 新" if FEATS[i] in NEW_FEATS else ""
    print(f"  {FEATS[i]:<22} {imp_n.mean(axis=0)[i]:.4f}{tag}")
