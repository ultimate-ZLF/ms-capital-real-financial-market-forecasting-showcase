"""TabM 共享数据管线：FEATS 定义、数据加载、per-fold 预处理、cos 指标。

与 lgb_baseline.py / lgb_submit.py 保持同一套 FEATS 与 CV 协议（照抄复制，
lgb 脚本是顶层执行代码不可 import——本模块是 TabM 两个脚本的公共依赖）。

预处理设计（v1）：
- NaN：per-fold 训练折中位数填充（NN 不能吃 NaN；LGBM 原生处理，TabM 必须显式做）
- 缩放：z-score（训练折统计）+ clip ±5（防 t_vol/t_n 等重尾计数特征打爆激活）
- target ×1000：纯数值卫生（MSE 从 ~7e-6 提到 O(1)），cos 尺度无关不受影响
"""
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
    # 微价失衡（衍生列，prep 中计算）
    "micro_imb0", "micro_imb60",
]

CLIP = 5.0  # z-score 后 clip 边界（重尾防御）


def prep(df: pl.DataFrame) -> pl.DataFrame:
    """加 2 个衍生列并选出 47 特征列（与 lgb_baseline.py:41-45 同逻辑）。"""
    df = df.with_columns([
        (pl.col("micro") - pl.col("mid")).alias("micro_imb0"),
        (pl.col("micro60") - pl.col("mid60")).alias("micro_imb60"),
    ])
    return df.select(FEATS)


def load_train() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray]:
    """返回 (X float32(N,47), y float32(N), month int32(N), months sorted, sample_id int32(N))。"""
    files = sorted(glob.glob(os.path.join(OUT, "factors_m*.parquet")))
    df = pl.concat([pl.read_parquet(f) for f in files])
    X = prep(df).to_numpy().astype(np.float32)  # Int 可空列 null → NaN
    y = df["target"].to_numpy().astype(np.float32)
    marr = df["month"].to_numpy()
    sids = df["sample_id"].to_numpy()
    return X, y, marr, sorted(df["month"].unique().to_list()), sids


def load_test() -> tuple[np.ndarray, pl.Series]:
    """返回 (Xt float32(N,47), sample_id Series)。"""
    test = pl.read_parquet(os.path.join(OUT, "factors_test.parquet"))
    return prep(test).to_numpy().astype(np.float32), test["sample_id"]


def cos_score(pred: np.ndarray, y: np.ndarray) -> float:
    """整月一个标量 cos（与 lgb_baseline.py:78 完全一致）。"""
    return float((pred * y).sum() / np.sqrt((pred**2).sum() * (y**2).sum()))


def fit_preprocess(X_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从训练折统计拟合 (med, mu, sd)。只允许用训练折数据（验证/test 零泄漏）。"""
    med = np.nanmedian(X_tr, axis=0)
    filled = np.where(np.isnan(X_tr), med, X_tr)
    mu = filled.mean(axis=0)
    sd = filled.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)  # 常数特征防除零
    return med, mu, sd


def apply_preprocess(X: np.ndarray, med, mu, sd) -> np.ndarray:
    """NaN→中位数填充 + z-score + clip。返回 float32。"""
    filled = np.where(np.isnan(X), med, X)
    scaled = np.clip((filled - mu) / sd, -CLIP, CLIP)
    assert np.isfinite(scaled).all(), "预处理后仍有非有限值（NaN/inf 漏处理）"
    return scaled.astype(np.float32)
