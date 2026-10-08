"""lgb_all35_submit.py — LGBM 82 特征（47+全35）提交：12 折模型对 test 平均预测。

N_ROUNDS = 199（lgb_newfeat_all.py CV 早停轮数中位数）。
输出：submissions/submission_lgb_v3.parquet
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
    "micro_imb0", "micro_imb60",
] + NEW_FEATS


def prep(df: pl.DataFrame) -> np.ndarray:
    df = df.with_columns([
        (pl.col("micro") - pl.col("mid")).alias("micro_imb0"),
        (pl.col("micro60") - pl.col("mid60")).alias("micro_imb60"),
    ])
    return df.select(FEATS).to_numpy()


# ---------- train ----------
files = sorted(glob.glob(os.path.join(OUT, "factors_m*.parquet")))
train = pl.concat([pl.read_parquet(f) for f in files])
newm = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new_m*.parquet")))])
train = train.join(newm.select(["sample_id", "month"] + NEW_FEATS),
                   on=["sample_id", "month"], how="left")
X = prep(train)
y = train["target"].to_numpy()
marr = train["month"].to_numpy()
months = sorted(train["month"].unique().to_list())

# ---------- test ----------
test = pl.read_parquet(os.path.join(OUT, "factors_test.parquet"))
newt = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new_test_c*.parquet")))])
test = test.join(newt.select(["sample_id"] + NEW_FEATS), on="sample_id", how="left")
Xt = prep(test)
print(f"train {X.shape}, test {Xt.shape}", flush=True)

params = dict(
    objective="regression", metric="rmse",
    learning_rate=0.05, num_leaves=63,
    min_child_samples=500, feature_fraction=0.5,
    bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=10.0, verbose=-1, seed=42,
)

VAL_MONTHS = months[2::6]
N_ROUNDS = 199  # lgb_newfeat_all.py CV 早停中位数
preds = []
for vm in VAL_MONTHS:
    tr = marr != vm
    dtr = lgb.Dataset(X[tr], y[tr])
    model = lgb.train(params, dtr, num_boost_round=N_ROUNDS)
    preds.append(model.predict(Xt))
    print(f"[fold {vm}] trained, pred mean={preds[-1].mean():.4f}", flush=True)

pred = np.mean(preds, axis=0)
print(f"\nfinal pred: mean={pred.mean():.5f} std={pred.std():.5f} "
      f"min={pred.min():.4f} max={pred.max():.4f}")

sub = pl.DataFrame({"sample_id": test["sample_id"], "prediction": pred})
out_path = os.path.join(BASE, "submissions", "submission_lgb_v3.parquet")
sub.write_parquet(out_path)
print(f"saved -> {out_path}")
