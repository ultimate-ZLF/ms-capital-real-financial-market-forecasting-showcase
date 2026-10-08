"""lgb_prune_submit.py — top-150 剪枝集提交：12 折模型（OOF + test 预测一次产出）。

特征集 = lgb_prune1.py 存盘的 gain 排名 top-150。
N_ROUNDS = 197（196 特征 CV 早停中位数）。
输出：factors/lgb_prune_oof.parquet + submissions/submission_lgb_v5.parquet

⚠️ **重跑会覆盖两样东西（2026-09-21 标注）**：
  1. `submission_lgb_v5.parquet` —— v21 blend 的 LGBM 成员（LB 0.133 的组成部分）；
  2. `factors/lgb_prune_oof.parquet` —— 本脚本写的是**固定 197 轮、无早停**版，而正确版由
     `lgb_prune_oof2.py` 以逐折早停协议重跑后覆盖（FACTORS「协议教训」：混合协议 0.14832
     vs 修正后 0.14977）。重跑本脚本 = 把修正后的 OOF 打回劣化版。
     **只需要 OOF 时请跑 `lgb_prune_oof2.py`，不要跑本脚本。**
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

NEW1_FEATS = [
    "rel_spread2_0", "imb2_0", "slope_0",
    "ask_slope_mean60", "bid_slope_mean60", "slope_imb60",
    "t_has_data_15", "t_miss_max", "o_has_data_15", "o_miss_max",
    "t_price_nf_15", "t_price_nf_30", "t_price_nf_60",
    "t_vol_nf_30", "t_vol_nf_60",
    "t_rc_nf_15", "t_rc_nf_30", "t_rc_nf_60",
    "o_vol_nf_15", "o_vol_nf_30", "o_vol_nf_60",
    "o_rc_nf_15", "o_rc_nf_30", "o_rc_nf_60",
    "o_cancel_nf_15", "o_cancel_nf_30", "o_cancel_nf_60",
    "x_to_buy_diff", "x_to_buy_diff_15",
    "t_large_buy", "t_large_sell", "x_large_imb",
    "m_rv60", "m_rv600", "x_rv_60_600",
]

NEW2_FEATS = [
    "m_mid_last", "m_mid_mean", "m_mid_std", "m_mid_skew",
    "m_sp_mean", "m_sp_std",
    "m_imb_mean", "m_imb_std", "m_imb2_mean60",
    "m_dofi", "m_dofi_60", "m_dofi_ewm120",
    "m_mid_ewm30", "m_mid_ewm120", "m_imb_ewm30", "m_imb_ewm120",
    "x_imb_ewm_sl", "x_dofi_ewm_sl",
    "x_mid_zscore", "x_vwap_mid_ratio",
    "t_sd", "t_sd_15", "t_sd_45",
    "t_px_std", "t_px_skew", "t_px_last",
    "t_lv_mean",
    "t_gap_mean", "t_gap_std", "t_gap_max", "t_gap_cv",
    "t_autocorr_lag1", "t_autocorr_lag5",
    "t_firsthalf_price", "t_firsthalf_buyratio", "t_firstthird_vol", "t_firstlast_px",
    "x_t_signed_ratio", "x_t_signed_w15_diff",
    "t_pxdiff1_mean", "t_pxdiff1_std", "t_pxdiff7_mean", "t_pxdiff7_std",
    "t_voldiff1_mean", "t_voldiff7_mean",
    "t_vwap",
    "o_px_std", "o_px_skew",
    "o_bid_depth", "o_ask_depth",
    "o_sv", "o_av",
    "o_gap_mean", "o_gap_std", "o_gap_cv",
    "o_cancel_new_ratio",
    "o_firsthalf_price", "o_firstthird_vol",
    "o_pxdiff1_mean", "o_pxdiff1_std", "o_pxdiff7_std", "o_voldiff1_mean",
    "x_sp_imb", "x_trans_order_vol_ratio",
]

NEW3_FEATS = [
    "t_pxdiff14_mean", "t_pxdiff14_std", "t_pxdiff21_std", "t_pxdiff30_std",
    "t_voldiff14_mean", "t_voldiff21_mean",
    "o_pxdiff14_std", "o_voldiff14_mean",
    "m_dofi2_sum60", "m_dofi2_ewm120",
    "m_imb_med", "m_imb_q25", "m_sp_q75",
    "m_mid_ret_abs_mean60",
    "m_mid_slope_180", "m_mid_slope_60",
    "m_imb_mean_15", "x_imb_15_180",
    "m_txv_std30",
    "m_ask_gap_mean60", "m_bid_gap_mean60",
    "t_sv_5", "t_sv_60", "x_t_sv_5_60",
    "t_vol_5", "x_t_vol_5_60",
    "t_n_60", "x_tn_15_60",
    "t_sd_ewm45", "t_vol_max30",
    "t_buy_sell_vol_ratio",
    "t_large_buy_15", "t_large_sell_15", "x_large_imb_15",
    "o_sv_ewm15", "o_av_ewm45", "o_add_ratio", "o_vol_max30",
]

PAIRS = [
    ("t_imb_w10", "m_mid_std"), ("t_imb_w10", "m_rv600"),
    ("ofi_w10", "m_mid_std"), ("ofi_w10", "m_dofi_ewm120"),
    ("t_sd_15", "m_rv600"), ("m_imb_mean", "m_rv600"),
    ("x_to_buy_diff_15", "t_vol"), ("m_mid_std", "t_n"),
    ("d_book_imb", "m_rv60"), ("m_dofi_ewm120", "m_sp_std"),
    ("t_imb_2s", "m_imb_mean"), ("m_mid_std", "m_sp_mean"),
]
PAIR_FEATS = [f"xp_{a}__{b}" for a, b in PAIRS]

BASE_FEATS = [
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
]
ALL_FEATS = BASE_FEATS + NEW1_FEATS + NEW2_FEATS + NEW3_FEATS + PAIR_FEATS

K = 150
PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.05, num_leaves=63,
    min_child_samples=500, feature_fraction=0.5,
    bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=10.0, verbose=-1, seed=42,
)
N_ROUNDS = 197


def prep(df: pl.DataFrame) -> np.ndarray:
    df = df.with_columns([
        (pl.col("micro") - pl.col("mid")).alias("micro_imb0"),
        (pl.col("micro60") - pl.col("mid60")).alias("micro_imb60"),
    ])
    return df.select(ALL_FEATS).to_numpy()


def main():
    gains = np.load(os.path.join(OUT, "lgb_prune_gains.npy"))
    order = np.argsort(gains)[::-1]
    keep = [ALL_FEATS[i] for i in order[:K]]
    idx = [ALL_FEATS.index(f) for f in keep]

    files = sorted(glob.glob(os.path.join(OUT, "factors_m*.parquet")))
    train = pl.concat([pl.read_parquet(f) for f in files])
    new1 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new_m*.parquet")))])
    new2 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new2_m*.parquet")))])
    new3 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new3_m*.parquet")))])
    train = (train.join(new1.select(["sample_id", "month"] + NEW1_FEATS), on=["sample_id", "month"], how="left")
                  .join(new2.select(["sample_id", "month"] + NEW2_FEATS), on=["sample_id", "month"], how="left")
                  .join(new3.select(["sample_id", "month"] + NEW3_FEATS), on=["sample_id", "month"], how="left"))
    train = train.with_columns([
        (pl.col(a) * pl.col(b)).alias(f"xp_{a}__{b}") for a, b in PAIRS
    ])
    X = prep(train)[:, idx]
    y = train["target"].to_numpy()
    marr = train["month"].to_numpy()
    sids = train["sample_id"].to_numpy()
    months = sorted(train["month"].unique().to_list())

    test = pl.read_parquet(os.path.join(OUT, "factors_test.parquet"))
    nt1 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new_test_c*.parquet")))])
    nt2 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new2_test_c*.parquet")))])
    nt3 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new3_test_c*.parquet")))])
    test = (test.join(nt1.select(["sample_id"] + NEW1_FEATS), on="sample_id", how="left")
                .join(nt2.select(["sample_id"] + NEW2_FEATS), on="sample_id", how="left")
                .join(nt3.select(["sample_id"] + NEW3_FEATS), on="sample_id", how="left"))
    test = test.with_columns([
        (pl.col(a) * pl.col(b)).alias(f"xp_{a}__{b}") for a, b in PAIRS
    ])
    Xt = prep(test)[:, idx]
    print(f"train {X.shape}, test {Xt.shape}", flush=True)

    VAL = months[2::6]
    preds, oof_rows = [], []
    for vm in VAL:
        tr, va = marr != vm, marr == vm
        dtr = lgb.Dataset(X[tr], y[tr])
        m = lgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)
        preds.append(m.predict(Xt))
        oof_rows.append(pl.DataFrame({
            "sample_id": sids[va], "month": marr[va],
            "pred": m.predict(X[va]).astype(np.float32)}))
        print(f"[fold {vm}] trained", flush=True)

    oof = pl.concat(oof_rows)
    oof.write_parquet(os.path.join(OUT, "lgb_prune_oof.parquet"))
    print(f"OOF 已存 factors/lgb_prune_oof.parquet（{oof.height} 行）", flush=True)

    pred = np.mean(preds, axis=0)
    print(f"final pred: mean={pred.mean():.5f} std={pred.std():.5f} "
          f"min={pred.min():.4f} max={pred.max():.4f}")
    sub = pl.DataFrame({"sample_id": test["sample_id"], "prediction": pred})
    out = os.path.join(BASE, "submissions", "submission_lgb_v5.parquet")
    sub.write_parquet(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
