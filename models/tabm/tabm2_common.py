"""tabm2_common.py — TabM 重做共享管线：top-150 特征加载 + tick 单位化预处理。

设计（v2，替代 tabm_common.py 的 z-score 管线；依据避雷 #13 + WALKTHROUGH4 的
snap-CNN 实证：NN 跨 regime 衰减根因是预处理，tick 单位化衰减 0.946 vs z-score 0.797）：

- 特征集：lgb_prune1.py 存盘的 gain 排名 top-150（196 = BASE 47 + NEW1 35 + NEW2 64
  + NEW3 38 + PAIRS 12 交互）。列序保持 ALL_FEATS 规范相对顺序
  （与 lgb_prune_submit.py 的 idx 切片一致），不按 gain 重排。
- 预处理三类单位变换（跨 regime 市场机制不变量）：
  TICK       ÷U_s（per-sample tick 单位，snap_cache index）+ clip ±256
             —— 价差/动量/波动/斜率类（test tick 0.1226% vs train 0.1504%，÷U_s 后
             两边都是"tick 数"，与 snap-CNN 的 mid_dev/spread_n 同一约定）
  TICK_ASINH ÷U_s 后 asinh —— 有符号价格×量（t_sd 族，符号保留、重尾压缩）
  ASINH      asinh —— 计数/成交量/秒数/深度/订单流量类
             （test 活跃度 +32%：乘性 regime 漂移 → asinh 后近似加性偏移，bias 可吸收）
  RAW        原样 —— 无量纲比值/比率/相关系数/归一价格水平（≈1.0，两市场同尺度）
- NaN：per-fold 训练折中位数填充（变换前，与 v1 同机制）；U_s null →
  TICK_FALLBACK（train 0.0015037 / test 0.0012257，DATA.md tick 常数）
- 无 z-score、无 ±5 clip；变换后统一 clip ±256 兜底（防 raw 类病态值打爆激活）

单位分类依据：FACTORS.md 因子定义 + factors.py/new_factors*.py 计算式，
四大类精确划分 196 特征（模块加载时 assert 验证）。
"""
import glob
import os

import numpy as np
import polars as pl

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
OUT = os.path.join(BASE, "factors")
CACHE = os.path.join(BASE, "snap_cache")

# 跨目录复用（models/lgbm）——目录结构见 README
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lgbm"))

# lgb_prune_submit 仅常量+受保护 main()，import 无副作用（lgb_prune_oof2.py 已有先例）
from lgb_prune_submit import (ALL_FEATS, BASE_FEATS, NEW1_FEATS, NEW2_FEATS,
                              NEW3_FEATS, PAIRS, PAIR_FEATS)  # noqa: E402

K = 150
TICK_CLIP = 256.0      # 与 snap-CNN mid_dev 同一裁剪约定
TICK_FALLBACK = {"train": 0.0015037, "test": 0.0012257}  # DATA.md tick 常数

# —— 单位类别映射（RAW = 不在前三个列表中的其余特征）——
TICK_FEATS = [
    # BASE：价差/动量/波动/曲率/vwap 偏差/微价偏差
    "spread0", "spread_mean60", "d60_mid", "d600_mid", "d60_micro",
    "mid_vol60", "mid_vol600", "sig_txn", "curv",
    "vwap_dev60", "vwap_dev600", "micro_imb0", "micro_imb60",
    # NEW1：档位斜率（价格/量）、已实现波动
    "ask_slope_mean60", "bid_slope_mean60", "slope_imb60",
    "m_rv60", "m_rv600",
    # NEW2：mid/spread 分布、成交价 std、价差序列、首末价差
    "m_mid_std", "m_sp_mean", "m_sp_std",
    "t_px_std", "t_firstlast_px",
    "t_pxdiff1_mean", "t_pxdiff1_std", "t_pxdiff7_mean", "t_pxdiff7_std",
    "o_px_std", "o_pxdiff1_mean", "o_pxdiff1_std", "o_pxdiff7_std",
    # NEW3：更长窗价差、spread 分位、绝对波动、mid 斜率、档间距
    "t_pxdiff14_mean", "t_pxdiff14_std", "t_pxdiff21_std", "t_pxdiff30_std",
    "o_pxdiff14_std", "m_sp_q75", "m_mid_ret_abs_mean60",
    "m_mid_slope_180", "m_mid_slope_60",
    "m_ask_gap_mean60", "m_bid_gap_mean60",
]
TICK_ASINH_FEATS = [
    # 有符号价格×量：Σ sign·price·vol（"tick·股"单位，符号保留）
    "t_sd", "t_sd_15", "t_sd_45", "t_sd_ewm45",
]
ASINH_FEATS = [
    # BASE：订单流失衡（有符号量）、活动计数、最后事件间隔
    "ofi", "ofi_w30", "ofi_w10", "ofi_5s",
    "t_n", "t_vol", "o_n", "o_vol", "last_gap",
    # NEW1：数据存在计数、缺失秒数、大单计数
    "t_has_data_15", "t_miss_max", "o_has_data_15", "o_miss_max",
    "t_large_buy", "t_large_sell",
    # NEW2：深度失衡流、事件间隔、成交量差、深度、订单量
    "m_dofi", "m_dofi_60", "m_dofi_ewm120", "x_dofi_ewm_sl",
    "t_gap_mean", "t_gap_std", "t_gap_max",
    "t_firstthird_vol", "t_voldiff1_mean", "t_voldiff7_mean",
    "o_bid_depth", "o_ask_depth", "o_sv", "o_av",
    "o_gap_mean", "o_gap_std", "o_firstthird_vol", "o_voldiff1_mean",
    # NEW3：dofi2、成交/订单量统计、大单 15s 窗、量差
    "m_dofi2_sum60", "m_dofi2_ewm120", "m_txv_std30",
    "t_sv_5", "t_sv_60", "t_vol_5", "t_n_60", "t_vol_max30",
    "t_voldiff14_mean", "t_voldiff21_mean", "o_voldiff14_mean",
    "t_large_buy_15", "t_large_sell_15",
    "o_sv_ewm15", "o_av_ewm45", "o_vol_max30",
    # PAIRS：交互乘积（无界，asinh 压缩）
    *PAIR_FEATS,
]
RAW_FEATS = [
    f for f in ALL_FEATS
    if f not in set(TICK_FEATS) | set(TICK_ASINH_FEATS) | set(ASINH_FEATS)
]
# 四大类必须精确划分 196 特征（写错一个名字会在这里立刻暴露）
assert sorted(TICK_FEATS + TICK_ASINH_FEATS + ASINH_FEATS + RAW_FEATS) == \
    sorted(ALL_FEATS), "单位类别映射未精确划分 ALL_FEATS"


def get_kept_names() -> list[str]:
    """top-150 特征名（规范列序），首次重建后缓存到 lgb_prune150_names.txt。"""
    names_path = os.path.join(OUT, "lgb_prune150_names.txt")
    if os.path.exists(names_path):
        with open(names_path, encoding="utf-8") as f:
            keep = [ln.strip() for ln in f if ln.strip()]
        if len(keep) == K:
            return keep
    gains = np.load(os.path.join(OUT, "lgb_prune_gains.npy"))
    idx = sorted(np.argsort(gains)[::-1][:K])  # 升序 = ALL_FEATS 规范相对顺序
    keep = [ALL_FEATS[i] for i in idx]
    with open(names_path, "w", encoding="utf-8") as f:
        f.write("\n".join(keep) + "\n")
    return keep


KEEP = get_kept_names()
# 150 宽矩阵内各单位的列下标（供向量化变换）
_CLS = {"tick": TICK_FEATS, "tick_asinh": TICK_ASINH_FEATS,
        "asinh": ASINH_FEATS, "raw": RAW_FEATS}
CLS_IDX = {c: np.array([KEEP.index(f) for f in names if f in KEEP], dtype=np.int64)
           for c, names in _CLS.items()}


BASE_SRC = ["sample_id", "month", "target", "micro", "mid", "micro60", "mid60"] + \
           [f for f in BASE_FEATS if f not in ("micro_imb0", "micro_imb60")]


def _read_batch(prefix: str, cols: list[str], months_keep=None) -> pl.DataFrame:
    """读一批因子 parquet（train 按月过滤）。

    列投影在 scan 层做（parquet 谓词/投影下推），因子列就地 cast f32——
    f64 列会让 df 峰值 ~2GB，f32 减半（本机无页面文件，避雷 #15）。
    """
    if prefix == "base":  # train: factors_m*.parquet；test 单文件在 load_test 里单独处理
        files = sorted(glob.glob(os.path.join(OUT, "factors_m*.parquet")))
        keep_cols = BASE_SRC
        cast_cols = [c for c in BASE_SRC if c not in ("sample_id", "month", "target")]
    else:
        files = sorted(glob.glob(os.path.join(OUT, f"{prefix}_m*.parquet")))
        keep_cols = ["sample_id", "month"] + cols
        cast_cols = cols
    lf = pl.scan_parquet(files)
    if months_keep is not None:
        lf = lf.filter(pl.col("month").is_in(months_keep))
    df = lf.select(keep_cols).collect()
    if cast_cols:
        df = df.with_columns([pl.col(c).cast(pl.Float32) for c in cast_cols])
    return df


def _join_us(df: pl.DataFrame, split: str) -> np.ndarray:
    """per-sample tick 单位 U_s（null → 该 split 的 TICK_FALLBACK），返回与行对齐的 f64 数组。"""
    meta = (pl.read_parquet(os.path.join(CACHE, f"{split}_snap_index.parquet"))
              .select(["sample_id", "U_s"]))
    df = df.join(meta, on="sample_id", how="left")
    us = df["U_s"].fill_null(TICK_FALLBACK[split]).to_numpy().astype(np.float64)
    return np.clip(us, 1e-9, None)  # 除零防护（U_s 理论上 >0）


def prep(df: pl.DataFrame) -> np.ndarray:
    """加 2 个微价衍生列并直接选出 top-150 特征（f32）。

    直接在 polars 里 select 150 列并 cast f32（避免先做 196 宽 f64 矩阵再切片的
    ~2GB 峰值——本机无页面文件，进程峰值过 ~4GB 会被系统击杀，避雷 #15）。
    """
    df = df.with_columns([
        (pl.col("micro") - pl.col("mid")).alias("micro_imb0"),
        (pl.col("micro60") - pl.col("mid60")).alias("micro_imb60"),
    ])
    return df.select(KEEP).cast(pl.Float32).to_numpy()


def load_train(months_keep=None) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                           list[int], np.ndarray, np.ndarray]:
    """返回 (X f32(N,150), y f32(N), month(N), months sorted, sample_id, U_s f64(N))。"""
    train = _read_batch("base", None, months_keep)
    for prefix, cols in [("new", NEW1_FEATS), ("new2", NEW2_FEATS), ("new3", NEW3_FEATS)]:
        part = _read_batch(prefix, cols, months_keep)
        train = train.join(part, on=["sample_id", "month"], how="left")
    train = train.with_columns([
        (pl.col(a) * pl.col(b)).alias(f"xp_{a}__{b}") for a, b in PAIRS
    ])
    us = _join_us(train, "train")
    X = prep(train)
    y = train["target"].to_numpy().astype(np.float32)
    marr = train["month"].to_numpy()
    sids = train["sample_id"].to_numpy()
    months = sorted(train["month"].unique().to_list())
    del train
    return X, y, marr, months, sids, us


def load_test() -> tuple[np.ndarray, pl.Series, np.ndarray]:
    """返回 (Xt f32(N,150), sample_id Series, U_s f64(N))。"""
    test_cols = ["sample_id", "micro", "mid", "micro60", "mid60"] + \
                [f for f in BASE_FEATS if f not in ("micro_imb0", "micro_imb60")]
    test = pl.read_parquet(os.path.join(OUT, "factors_test.parquet"),
                           columns=test_cols).with_columns(
        [pl.col(c).cast(pl.Float32) for c in test_cols if c != "sample_id"])
    for prefix, cols in [("new", NEW1_FEATS), ("new2", NEW2_FEATS), ("new3", NEW3_FEATS)]:
        # 逐块 select+cast 再 concat（避免整批 f64 峰值）
        parts = [pl.read_parquet(f, columns=["sample_id"] + cols)
                   .with_columns([pl.col(c).cast(pl.Float32) for c in cols])
                 for f in sorted(glob.glob(os.path.join(OUT, f"{prefix}_test_c*.parquet")))]
        test = test.join(pl.concat(parts), on="sample_id", how="left")
    test = test.with_columns([
        (pl.col(a) * pl.col(b)).alias(f"xp_{a}__{b}") for a, b in PAIRS
    ])
    us = _join_us(test, "test")
    Xt = prep(test)
    return Xt, test["sample_id"], us


def cos_score(pred: np.ndarray, y: np.ndarray) -> float:
    """整月一个标量 cos（与 tabm_common.cos_score 完全一致）。"""
    return float((pred * y).sum() / np.sqrt((pred**2).sum() * (y**2).sum()))


# —— memmap 缓存（prep_tabm2_cache.py 构建；文件映射不占 commit，降峰值 ~1.1GB）——
CACHE_DIR = os.path.join(OUT, "tabm2_cache")
X_DAT = os.path.join(CACHE_DIR, "X.dat")
META_NPZ = os.path.join(CACHE_DIR, "meta.npz")
XT_DAT = os.path.join(CACHE_DIR, "Xt.dat")
TEST_META_NPZ = os.path.join(CACHE_DIR, "test_meta.npz")


def cache_ready() -> bool:
    return all(os.path.exists(p) for p in (X_DAT, META_NPZ, XT_DAT, TEST_META_NPZ))


def load_train_cached() -> tuple:
    """从 memmap 缓存载入（shape 存于 meta.npz；X 为只读映射，切片才是 RAM 副本）。"""
    meta = dict(np.load(META_NPZ))
    shape = tuple(meta.pop("shape").tolist())
    X = np.memmap(X_DAT, dtype=np.float32, mode="r", shape=shape)
    return X, meta["y"], meta["marr"], meta["months"].tolist(), meta["sids"], meta["us"]


def load_test_cached() -> tuple:
    meta = dict(np.load(TEST_META_NPZ))
    shape = tuple(meta.pop("shape").tolist())
    Xt = np.memmap(XT_DAT, dtype=np.float32, mode="r", shape=shape)
    return Xt, pl.Series("sample_id", meta["sids"]), meta["us"]


def fit_preprocess(X_tr: np.ndarray) -> np.ndarray:
    """逐列中位数（原始空间，只用训练折）——单位变换本身是 per-sample 的，无需拟合。"""
    return np.nanmedian(X_tr, axis=0).astype(np.float32)


def apply_preprocess(A: np.ndarray, med: np.ndarray, us: np.ndarray) -> np.ndarray:
    """NaN→训练折中位数填充 → 三类单位变换 → 全局 clip ±256。返回 float32。

    严格就地变换（连 np.where 的第二份 (N,150) 副本都不建）——本机无页面文件，
    峰值内存过 ~4GB 会被系统击杀（避雷 #15）。**调用方必须传入可丢弃的副本**
    （布尔切片 X[mask] / 显式 .copy() 均为独立数组，可安全就地改写）。
    """
    # 1) NaN 按列填充（逐列 mask 避免全矩阵 bool 副本；~150 列 × 1 趟 ≈ 1-2s）
    for j in range(A.shape[1]):
        m = np.isnan(A[:, j])
        if m.any():
            A[m, j] = med[j]
    # 2) 三类单位变换（就地）
    u = us.astype(np.float32)[:, None]
    if len(CLS_IDX["tick"]):
        A[:, CLS_IDX["tick"]] = np.clip(A[:, CLS_IDX["tick"]] / u,
                                        -TICK_CLIP, TICK_CLIP)
    if len(CLS_IDX["tick_asinh"]):
        A[:, CLS_IDX["tick_asinh"]] = np.arcsinh(A[:, CLS_IDX["tick_asinh"]] / u)
    if len(CLS_IDX["asinh"]):
        A[:, CLS_IDX["asinh"]] = np.arcsinh(A[:, CLS_IDX["asinh"]])
    np.clip(A, -TICK_CLIP, TICK_CLIP, out=A)  # 兜底 raw 类病态值（就地）
    assert np.isfinite(A).all(), "预处理后仍有非有限值（NaN/inf 漏处理）"
    return A
