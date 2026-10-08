"""lgb_prune1.py — 减法阶段第一轮：gain 驱动的特征剪枝曲线。

流程：
1. 全 196 特征 12 折跑一遍 → 各特征 mean gain 排名（存 factors/lgb_prune_gains.npy）
2. 按 gain 保留 top-150 / top-120 → 同协议重跑
3. 输出 CV 曲线：196（已知 0.13885）/ 150 / 120 对比 146 基线 0.13968
判定：剪枝后 CV 回升或持平 → 剪枝有效，保留最小有效集；否则回 146。
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


def build_df():
    files = sorted(glob.glob(os.path.join(OUT, "factors_m*.parquet")))
    df = pl.concat([pl.read_parquet(f) for f in files])
    new1 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new_m*.parquet")))])
    new2 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new2_m*.parquet")))])
    new3 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new3_m*.parquet")))])
    df = (df.join(new1.select(["sample_id", "month"] + NEW1_FEATS), on=["sample_id", "month"], how="left")
            .join(new2.select(["sample_id", "month"] + NEW2_FEATS), on=["sample_id", "month"], how="left")
            .join(new3.select(["sample_id", "month"] + NEW3_FEATS), on=["sample_id", "month"], how="left"))
    df = df.with_columns([
        (pl.col(a) * pl.col(b)).alias(f"xp_{a}__{b}") for a, b in PAIRS
    ])
    df = df.with_columns([
        (pl.col("micro") - pl.col("mid")).alias("micro_imb0"),
        (pl.col("micro60") - pl.col("mid60")).alias("micro_imb60"),
    ])
    return df


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

PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.05, num_leaves=63,
    min_child_samples=500, feature_fraction=0.5,
    bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=10.0, verbose=-1, seed=42,
)


def run_cv(feats: list[str], X_full: np.ndarray, y, marr, months,
           return_imp=False):
    idx = [ALL_FEATS.index(f) for f in feats]
    X = X_full[:, idx]
    VAL = months[2::6]
    coses, imps = [], []
    for vm in VAL:
        tr, va = marr != vm, marr == vm
        dtr = lgb.Dataset(X[tr], y[tr])
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(PARAMS, dtr, num_boost_round=800, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = m.predict(X[va], num_iteration=m.best_iteration)
        yv = y[va]
        coses.append(float((pred * yv).sum() / np.sqrt((pred**2).sum() * (yv**2).sum())))
        if return_imp:
            imps.append(m.feature_importance("gain"))
    out = (np.mean(coses), np.std(coses))
    if return_imp:
        return out, np.array(imps)
    return out


def main():
    df = build_df()
    months = sorted(df["month"].unique().to_list())
    X_full = df.select(ALL_FEATS).to_numpy()
    y = df["target"].to_numpy()
    marr = df["month"].to_numpy()
    print(f"X {X_full.shape}", flush=True)

    # Pass 1：全量 → gain 排名
    (cv_all, _), imps = run_cv(ALL_FEATS, X_full, y, marr, months, return_imp=True)
    imp_n = imps / imps.sum(axis=1, keepdims=True)
    mean_gain = imp_n.mean(axis=0)
    order = np.argsort(mean_gain)[::-1]
    np.save(os.path.join(OUT, "lgb_prune_gains.npy"), mean_gain)
    print(f"\n全 196：CV {cv_all:.5f}（对照 0.13885，一致即可信）", flush=True)

    # Pass 2/3：top-K
    for K in [150, 120]:
        keep = [ALL_FEATS[i] for i in order[:K]]
        cv, sd = run_cv(keep, X_full, y, marr, months)
        n_new3 = len([f for f in keep if f in NEW3_FEATS])
        n_pairs = len([f for f in keep if f in PAIR_FEATS])
        print(f"top-{K}：CV {cv:.5f} std {sd:.5f}（新3 {n_new3} / 交互 {n_pairs}）", flush=True)

    print("\n参考：146 特征 = 0.13968；196 = 0.13885")
    print("判定：top-K 回升到 0.1397+ → 剪枝有效，可继续下探；否则回 146 集")


if __name__ == "__main__":
    main()
