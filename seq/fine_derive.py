"""fine_derive.py — 细段**派生通道**：纯内存、不落盘、**不改 `flow_feats` 的 16 条基通道**。

## 这是什么

细段基础通道（`flow_feats.FEAT_NAMES`）是每 (样本×秒) 的**组内聚合量**。
本模块在内存里把它扩展成一组**组内分布 / 跨秒状态 / 条件加权**的通道。

## ⚠️ 选特征的判据：只做**卷积主干做不到**的

**卷积 = 线性滤波 + 逐点非线性。** 它做不到的只有三类：

| 做不到的 | 本模块的算子族 | 本次的候选 |
|---|---|---|
| **比较 / 排序** | 组内极值、顺序统计 | `o_px_disp` `t_intragap` `t_imb_peak` `txn_burst_10s` |
| **计数 / 状态** | `since` / `run_signed` | `since_big_txn` `t_imb_run` |
| **相乘 / 相除** | 条件聚合、比值、相关 | `o_cancel_imb` `o_cancel_ratio_buy/sell` `t_big_imb` `vol_px_corr` |

反过来，**滚动均值 / 标准差 / EWM / 斜率 / 曲率全是线性滤波**（细段塔感受野 63 > 60，
覆盖全窗）——模型自己学得出来，做成通道只是替它省力、**不构成新信息**。

## ⚠️ 细段特有的两条约束

1. **末步读出**：细段塔只读 **bin 59**（`tcn_baseline.py` 的 `n_valid−1` 对应细段的 `59`）。
   所以任何**纯靠 bin 索引取值**的通道（如秒索引 `bin_t`）在读出点上贡献恒定，是空操作
   ——本模块**不产生**这类通道。
2. **无 pad、无 mask**：60 个 bin 对每个样本都存在，空 bin = **真实零活动**
   （`flow_feats.py` 文档 §3）。所以全窗均值就是 `mean(axis=1)`，**不需要 `n_valid`**。

## 为什么不在 `flow_feats.py` 里加通道

基通道数写进了三个模型的检查点指纹（`"fine": [60, K]`），改 K 会让已归档的权重失配
（`snap_cache/ckpt_*` 里的 v22/v24/v25）。本模块**只读不写**，默认关：
不显式传 `--derive` 时全链路与今天逐位一致。

## 数据源：回到原始表，**不读** 16 通道缓存

本次候选里有 4 条需要**逐笔事件级**的量——组内价格 std、组内相邻成交间隔、大单方向、
组内事件数。这些在 1 秒聚合里**已经被丢掉**（`flow_feats` 只留 VWAP / 总量 / 笔数），
从 16 通道缓存**反解不回来**。所以本模块重扫 `order.parquet` / `transaction.parquet`，
代价与 `flow_feats` 同量级（train ≈1.5–3 min）。

bin 约定与 `flow_feats` **必须逐位一致**：`bin = 59 − floor(sec)`，clip 到 `0..59`。
`screen_fine_channels.py` 把 `t_imb` 与缓存版对着核，防止两套 binning 静默漂移。

## 已知取舍（不是 bug）

- **`t_imb_peak` 在时间轴上是常量**（整窗顺序统计 → 逐样本一个标量）。
  它在因果卷积下等价于"往 head 注入一个 per-sample 标量"，机制上等同表格特征——
  是有意的，不是写错。
- **`t_intragap` 用 `(max−min)/(cnt−1)`**（相邻间隔的均值可由望远镜求和化简），
  **不需要排序**——比 `diff().over()` 便宜一个数量级，且完全等价。
- **`o_px_disp` 用样本标准差**（polars `std` 默认 ddof=1）；组内只有 1 笔时返回 null → 填 0。
  "一笔"本来就谈不上离散。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flow_feats  # noqa: E402  借它的 FINE_T / bin 约定 / load_u_s，保证两套 binning 同源
from seq_common import BASE  # noqa: E402

FINE_T = flow_feats.FINE_T
EPS = 1e-6

BIG_MULT = 2.0        # "大单" = 单笔量 > 本样本单笔均量 × BIG_MULT
CAP_RUN = 10.0        # `t_imb_run` 的连续长度截断（bin）
CAP_SINCE = 30.0      # `since_big_txn` 的间隔截断（bin）
W_BURST = 10          # `txn_burst_10s` 的回看窗（bin）
W_CORR = 20           # `vol_px_corr` 的滚动窗（bin）
CLIP_DISP = 50.0      # `o_px_disp` 的上界（tick）

# 候选池（11 条，2026-09-27 用户审定的「构思 → 自我批判」通过清单）
POOL = [
    "o_cancel_imb",        # 0  撤单方向失衡（16 条里完全没有这个量）
    "o_cancel_ratio_buy",  # 1  买侧撤单/(撤单+新挂)
    "o_cancel_ratio_sell", # 2  卖侧同上
    "t_big_imb",           # 3  (大买单量−大卖单量)/总成交 —— 需逐笔事件级阈值
    "t_imb_run",           # 4  失衡连续同向的带符号秒数
    "since_big_txn",       # 5  距上次大单的 bin 数
    "txn_burst_10s",       # 6  最近 10 bin 内最大事件数
    "t_imb_peak",          # 7  窗内 |t_imb| 最大那一 bin 的带符号值
    "vol_px_corr",         # 8  滚动 20 bin 的量价相关
    "t_intragap",          # 9  组内相邻成交平均间隔（亚秒）—— 需逐笔时间戳
    "o_px_disp",           # 10 组内挂单价离散 —— 需逐笔价格
]

# 基通道：**逐字复现** `flow_feats` 的 16 条（不参与模型输入，只做两件事）：
#   1. **闸门 −1**：把两套 binning 从 2 条通道扩到 16 条逐位对照；
#   2. **正交化控制组**：它们就是模型**已经看到**的东西，
#      `--ctrl` 默认取全部 16 条 → "增量 cos" 才是真正的增量。
BASE_NAMES = list(flow_feats.FEAT_NAMES)

# 参照通道（⊂ BASE_NAMES）：筛网用它们做**真阳性验证**（CLAUDE.md #15）。
# 期望值来自 FACTORS.md 的 71 月因子表（`t_imb_2s` 0.0733 / `x_to_buy_diff` 0.0214）。
REFS = ["t_imb", "x_to_buy_diff"]
ALL_NAMES = POOL + BASE_NAMES

ORDER_COLS = ["nb", "ns", "cb", "cs", "opx_std", "ocnt", "obcnt", "opxv", "ovol"]
TX_COLS = ["tb", "ts", "tvol", "tpxv", "tcnt", "tbcnt", "t_maxv",
           "tsec_max", "tsec_min", "big_buy", "big_sell"]


def parse_spec(spec) -> list:
    """`"a,b"` / `["a","b"]` / `None` / `"none"` → 名字列表（空 = 不派生）。"""
    if spec is None:
        return []
    if isinstance(spec, str):
        s = spec.strip()
        if s.lower() in ("", "none"):
            return []
        names = [x.strip() for x in s.split(",") if x.strip()]
    else:
        names = list(spec)
    bad = [x for x in names if x not in ALL_NAMES]
    if bad:
        raise SystemExit(f"未知通道 {bad}（可选：{ALL_NAMES}）")
    return names


def _bin_expr():
    """与 `flow_feats.build_chunk` 逐位同源的 bin 定义。"""
    return (59 - pl.col("seconds_before_predict").floor().cast(pl.Int16)) \
        .clip(0, FINE_T - 1).alias("bin")


def _scatter(frame, cols, sids_chunk):
    """group_by 结果 → 每列一个 `(n, 60)` f64 数组（缺席的 (样本,bin) 填 0）。"""
    n = sids_chunk.size
    sid = frame["sample_id"].to_numpy()
    row = np.searchsorted(sids_chunk, sid)
    assert (sids_chunk[row] == sid).all(), "sample_id 不在本块范围内"
    b = frame["bin"].to_numpy()
    out = {}
    for c in cols:
        a = np.zeros((n, FINE_T), dtype=np.float64)
        a[row, b] = frame[c].cast(pl.Float64).fill_null(0.0).to_numpy()
        out[c] = a
    return out


def build_raw_chunk(split, lo, hi, sids_chunk):
    """一个样本区间的**每秒原始聚合**（order + transaction）。

    ⚠️ 这里刻意**不做任何 asinh / 比值**——那是 `flow_feats` 的 16 通道在做的事，
    本模块只在原始量上做卷积做不到的算子。
    """
    # ---------------- order 表 ----------------
    o = (pl.scan_parquet(os.path.join(BASE, split, "order.parquet"))
         .filter(pl.col("sample_id").is_between(lo, hi))
         .select(["sample_id", "seconds_before_predict", "price", "volume",
                  "side", "order_action"])
         .collect())
    o = o.with_columns([
        _bin_expr(),
        (pl.col("order_action") == 0).alias("_is_new"),
        (pl.col("side") == 0).alias("_is_buy"),
    ])
    og = o.group_by(["sample_id", "bin"]).agg([
        pl.col("volume").filter(pl.col("_is_new") & pl.col("_is_buy")).sum().alias("nb"),
        pl.col("volume").filter(pl.col("_is_new") & (pl.col("side") == 1)).sum().alias("ns"),
        pl.col("volume").filter((pl.col("order_action") == 1) & pl.col("_is_buy")).sum().alias("cb"),
        pl.col("volume").filter((pl.col("order_action") == 1) & (pl.col("side") == 1)).sum().alias("cs"),
        pl.col("price").std().alias("opx_std"),   # 组内 1 笔 → null → fill_null(0)
        pl.len().alias("ocnt"),
        pl.col("_is_buy").sum().alias("obcnt"),
        (pl.col("price") * pl.col("volume")).sum().alias("opxv"),
        pl.col("volume").sum().alias("ovol"),
    ])
    del o
    raw = _scatter(og, ORDER_COLS, sids_chunk)
    del og

    # ---------------- transaction 表 ----------------
    t = (pl.scan_parquet(os.path.join(BASE, split, "transaction.parquet"))
         .filter(pl.col("sample_id").is_between(lo, hi))
         .select(["sample_id", "seconds_before_predict", "price", "volume", "side"])
         .collect())
    t = t.with_columns([_bin_expr(), (pl.col("side") == 0).alias("_is_buy")])
    # "大单"阈值是 **per-sample** 的（项目纪律：量级类一律 per-sample 归一，消 regime 差异）。
    # 用一次小 join 而不是 `.over()`：join 的右表只有 n_samples 行，内存可控。
    tstat = t.group_by("sample_id").agg(pl.col("volume").mean().alias("_vmean"))
    t = t.join(tstat, on="sample_id", how="left")
    t = t.with_columns((pl.col("volume") > BIG_MULT * pl.col("_vmean")).alias("_big"))
    tg = t.group_by(["sample_id", "bin"]).agg([
        pl.col("volume").filter(pl.col("_is_buy")).sum().alias("tb"),
        pl.col("volume").filter(~pl.col("_is_buy")).sum().alias("ts"),
        pl.col("volume").sum().alias("tvol"),
        (pl.col("price") * pl.col("volume")).sum().alias("tpxv"),
        pl.len().alias("tcnt"),
        pl.col("_is_buy").sum().alias("tbcnt"),
        pl.col("volume").max().alias("t_maxv"),
        pl.col("seconds_before_predict").max().alias("tsec_max"),
        pl.col("seconds_before_predict").min().alias("tsec_min"),
        pl.col("volume").filter(pl.col("_big") & pl.col("_is_buy")).sum().alias("big_buy"),
        pl.col("volume").filter(pl.col("_big") & ~pl.col("_is_buy")).sum().alias("big_sell"),
    ])
    del t, tstat
    raw.update(_scatter(tg, TX_COLS, sids_chunk))
    del tg

    assert set(raw.keys()) == set(ORDER_COLS + TX_COLS), "原始聚合列缺失"
    return raw


# --------------------------------------------------------------------------- #
# 算子：卷积做不到的四类（比值 / 状态 / 顺序统计 / 相关）
# --------------------------------------------------------------------------- #

def _ratio(num, den):
    """`den > 0` 处 `num/den`，否则 0（与 `flow_feats` 的 `np.divide(where=…)` 同约定）。"""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.zeros(np.broadcast_shapes(num.shape, den.shape), dtype=np.float64)
    return np.divide(num, den, out=out, where=den > 0)


def _run_signed(v, cap):
    """逐 bin 的「连续同向长度」带符号版：同号累加、异号或 0 重置。

    conv 无状态，做不到"已经连续几个了"。
    """
    n, T = v.shape
    out = np.zeros_like(v)
    sgn = np.sign(v)
    prev = np.zeros(n)
    run = np.zeros(n)
    for k in range(T):
        s = sgn[:, k]
        nz = s != 0
        run = np.where(nz, np.where(nz & (s == prev), run + 1.0, 1.0), 0.0)
        prev = np.where(nz, s, 0.0)
        out[:, k] = s * np.minimum(run, cap) / cap
    return out


def _since(cond, cap):
    """距上一次 `cond` 为真的 bin 数（`cap` 截断、归一化）；从未发生 → 1.0。"""
    n, T = cond.shape
    pos = np.arange(T)[None, :]
    last = np.maximum.accumulate(np.where(cond, pos, -1), axis=1)
    d = np.where(last < 0, float(cap), (pos - last).astype(np.float64))
    return np.minimum(d, cap) / cap


def _roll_max(v, w):
    """最近 w 个 bin 的最大值（含当前）。60 步逐列取 max，代价可忽略。"""
    n, T = v.shape
    out = np.empty_like(v)
    for k in range(T):
        out[:, k] = v[:, max(0, k - w + 1):k + 1].max(axis=1)
    return out


def _roll_sum(v, w):
    """最近 w 个 bin 的和（前缀和，无循环）。"""
    n, T = v.shape
    c = np.concatenate([np.zeros((n, 1)), np.cumsum(v, axis=1)], axis=1)   # (n, T+1)
    idx = np.arange(T)
    return c[:, idx + 1] - c[:, np.maximum(idx - w + 1, 0)]


def _peak_signed(v):
    """`|v|` 最大那一 bin 的**带符号值**，广播到全部 bin（整窗顺序统计 → 逐样本常量）。"""
    k = np.argmax(np.abs(v), axis=1)
    return np.repeat(v[np.arange(v.shape[0]), k][:, None], v.shape[1], axis=1)


def _roll_corr(x, y, w):
    """最近 w 个 bin 的 Pearson 相关（滚动和，无循环）。窗口未满 → 0。"""
    n, T = x.shape
    sx, sy = _roll_sum(x, w), _roll_sum(y, w)
    sxx, syy, sxy = _roll_sum(x * x, w), _roll_sum(y * y, w), _roll_sum(x * y, w)
    idx = np.arange(T)
    cnt = np.minimum(idx + 1, w).astype(np.float64)[None, :]
    num = cnt * sxy - sx * sy
    den = np.sqrt(np.maximum(cnt * sxx - sx * sx, 0.0) *
                  np.maximum(cnt * syy - sy * sy, 0.0))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return np.where((idx + 1 >= w)[None, :], r, 0.0)


# --------------------------------------------------------------------------- #
# 基通道复现（16 条）—— 只作对照与去冗余，不进模型
# --------------------------------------------------------------------------- #

def derive_base(raw, u_s):
    """**逐字复现** `flow_feats.build_chunk` 的 16 条基通道（归一约定见该文件 §3b/§4）。

    ⚠️ 这里一旦与 `flow_feats` 不一致，两个用途同时失效，而且**不会报错**：
      1. `screen_fine_channels.verify_binning` 拿它做闸门 −1（两套 binning 逐位对照）；
      2. 筛网的 `--ctrl` 默认取全部 16 条当正交化控制组——控制组错了，
         "增量 cos" 就是错的，而它看起来完全正常。
    `test_fine_derive.py` 守不住这条（要真实数据），**闸门 −1 是唯一的执法者**。
    """
    o_cnt, o_bcnt, t_cnt, t_bcnt = raw["ocnt"], raw["obcnt"], raw["tcnt"], raw["tbcnt"]
    nb, ns, cb, cs = raw["nb"], raw["ns"], raw["cb"], raw["cs"]
    tb, ts, tvol, tpxv = raw["tb"], raw["ts"], raw["tvol"], raw["tpxv"]
    U = np.asarray(u_s, dtype=np.float64)[:, None]

    o_vol_tot, o_pxv_tot = raw["ovol"].sum(axis=1, keepdims=True), raw["opxv"].sum(axis=1, keepdims=True)
    t_tot, t_pxv_tot, t_cnt_tot = (tvol.sum(axis=1, keepdims=True),
                                   tpxv.sum(axis=1, keepdims=True),
                                   t_cnt.sum(axis=1, keepdims=True))
    # 量级类：÷ per-sample 每秒均值（分母下限 1，防极稀疏样本放大）后再 asinh
    o_unit = np.maximum((nb + ns).sum(axis=1, keepdims=True) / FINE_T, 1.0)
    o_can_unit = np.maximum((cb + cs).sum(axis=1, keepdims=True) / FINE_T, 1.0)
    t_unit = np.maximum(t_tot / FINE_T, 1.0)
    o_vwap, t_vwap = _ratio(o_pxv_tot, o_vol_tot), _ratio(t_pxv_tot, t_tot)
    t_size_smp = _ratio(t_tot, t_cnt_tot)

    d = {
        "o_new_buy": np.arcsinh(nb / o_unit),
        "o_new_sell": np.arcsinh(ns / o_unit),
        "o_cancel_buy": np.arcsinh(cb / o_can_unit),
        "o_cancel_sell": np.arcsinh(cs / o_can_unit),
        "o_new_imb": _ratio(nb - ns, nb + ns),
        "o_cancel_ratio": _ratio(cb + cs, nb + ns + cb + cs),
        "o_px_dev": np.where(raw["ovol"] > 0,
                             np.clip((_ratio(raw["opxv"], raw["ovol"]) - o_vwap) / U, -50, 50), 0.0),
        "t_vol_buy": np.arcsinh(tb / t_unit),
        "t_vol_sell": np.arcsinh(ts / t_unit),
        "t_imb": _ratio(tb - ts, tb + ts),
        "t_px_dev": np.where(tvol > 0,
                             np.clip((_ratio(tpxv, tvol) - t_vwap) / U, -50, 50), 0.0),
        "t_size_rel": np.where(t_cnt > 0,
                               np.clip(_ratio(tvol, t_cnt) / (t_size_smp + EPS) - 1.0, -5, 5), 0.0),
        "x_to_buy_diff": np.where((t_cnt > 0) & (o_cnt > 0),
                                  _ratio(t_bcnt, t_cnt) - _ratio(o_bcnt, o_cnt), 0.0),
        "flow_cnt": np.arcsinh(o_cnt + t_cnt),
        "has_event": (o_cnt + t_cnt > 0).astype(np.float64),
    }
    pos = np.arange(FINE_T)[None, :]      # x_flow_gap 要沿 bin 轴累进
    last = np.maximum.accumulate(np.where(d["has_event"] > 0, pos, -1), axis=1)
    d["x_flow_gap"] = np.clip(pos - last, 0, 10) / 10.0
    return d


# --------------------------------------------------------------------------- #
# 11 条候选
# --------------------------------------------------------------------------- #

def derive_cand(raw, u_s):
    """11 条候选（`POOL`）：每一条都必须是卷积做不到的（比较/排序、计数/状态、相乘/相除）。"""
    tb, ts, tvol, tcnt = raw["tb"], raw["ts"], raw["tvol"], raw["tcnt"]
    cb, cs, nb, ns = raw["cb"], raw["cs"], raw["nb"], raw["ns"]
    imb = _ratio(tb - ts, tb + ts)

    dev = {}
    # --- 相乘 / 相除 ---
    dev["o_cancel_imb"] = _ratio(cb - cs, cb + cs)
    dev["o_cancel_ratio_buy"] = _ratio(cb, cb + nb)
    dev["o_cancel_ratio_sell"] = _ratio(cs, cs + ns)
    dev["t_big_imb"] = _ratio(raw["big_buy"] - raw["big_sell"], tvol)
    vwap = _ratio(raw["tpxv"], tvol)
    dpx = np.zeros_like(vwap)
    dpx[:, 1:] = np.abs(np.diff(vwap, axis=1))
    dev["vol_px_corr"] = _roll_corr(tvol, dpx * (tvol > 0), W_CORR)

    # --- 计数 / 状态 ---
    dev["t_imb_run"] = _run_signed(imb, CAP_RUN)
    vmean = _ratio(tvol.sum(axis=1), tcnt.sum(axis=1))[:, None]
    has_big = raw["t_maxv"] > np.where(tcnt > 0, BIG_MULT * vmean, np.inf)
    dev["since_big_txn"] = _since(has_big, CAP_SINCE)

    # --- 比较 / 排序 ---
    dev["txn_burst_10s"] = np.arcsinh(_roll_max(raw["ocnt"] + tcnt, W_BURST))
    dev["t_imb_peak"] = _peak_signed(imb)
    span = raw["tsec_max"] - raw["tsec_min"]
    dev["t_intragap"] = np.where(tcnt >= 2, np.clip(_ratio(span, tcnt - 1), 0.0, 1.0), 0.0)
    dev["o_px_disp"] = np.clip(raw["opx_std"] / np.asarray(u_s, dtype=np.float64)[:, None],
                               0.0, CLIP_DISP)
    return dev


def derive_all(raw, u_s, names):
    """原始聚合 → `(n, 60, K)` f32，K = len(names)，顺序与 names 一致。

    `names` 可以混用候选与基通道名（筛网要把基通道当控制组一起算出来）。
    """
    dev = derive_base(raw, u_s)
    dev.update(derive_cand(raw, u_s))
    missing = [n for n in names if n not in dev]
    if missing:
        raise SystemExit(f"未实现的通道 {missing}")
    out = np.stack([dev[nm] for nm in names], axis=2)
    assert out.shape == (raw["tcnt"].shape[0], FINE_T, len(names)), f"形状 {out.shape} 不符"
    assert np.isfinite(out).all(), "派生通道含 NaN/inf"
    return out.astype(np.float32)


def reduce_chunk(split, lo, hi, sids_chunk, u_s_chunk, names):
    """一个样本区间 → 每条通道两个 per-sample 标量 `(full (B,K), last2 (B,K))`。

    口径照 `screen_fine_channels.py` 的说明：`last2` 对齐**模型的末步读出**，
    `full` 对齐**表格因子的窗口定义**。细段无 pad，所以全窗就是 `mean(axis=1)`。
    """
    raw = build_raw_chunk(split, lo, hi, sids_chunk)
    d = derive_all(raw, u_s_chunk, names)
    full = d.mean(axis=1)
    last2 = (d[:, -1] + d[:, -2]) / 2.0
    return full.astype(np.float64), last2.astype(np.float64)


def _row_layout(split):
    """`(all_sids, cid→行起点)`——与 `flow_feats.build_split` **同源**（同一套月起点算法）。

    行序错了**不会报错**，只会让派生通道与基通道错配。两处必须一致。
    """
    if split == "train":
        lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).sort("sample_id")
        counts = lab.group_by("month").len().sort("month")["len"].to_numpy()
        return (lab["sample_id"].to_numpy(),
                {f"m{m}": int(counts[:m].sum()) for m in range(len(counts))})
    sids = pl.read_csv(os.path.join(BASE, "submissions/submission.csv"),
                       columns=["sample_id"]).sort("sample_id")["sample_id"].to_numpy()
    lo_all = int(next(iter(flow_feats.iter_chunks("test", None)))[1])
    return sids, {cid: lo - lo_all for cid, lo, _ in flow_feats.iter_chunks("test", None)}


def build_full(split, names, verbose=True):
    """全量派生通道 `(N, 60, K)` f16；`names` 为空时返回 `(0, 60, 0)`。

    行序与 `flow_feats.build_split` **逐行一致**（`load_train` 会把结果直接拼在基通道后面）。
    串行——`fork` 池在这里不可用，见 `screen_fine_channels.gather` 的警告与 CLAUDE.md #17。
    train 约 2–3 min（与 `flow_feats` 同量级的一次重扫描）。
    """
    names = list(names)
    if not names:
        return np.zeros((0, FINE_T, 0), dtype=np.float16)
    all_sids, cid2start = _row_layout(split)
    u_s_all = flow_feats.load_u_s(split, all_sids)
    out = np.empty((all_sids.size, FINE_T, len(names)), dtype=np.float16)
    t0 = time.time()
    for cid, lo, hi in flow_feats.iter_chunks(split, None):
        sel = (all_sids >= lo) & (all_sids <= hi)
        sc, uc = all_sids[sel], u_s_all[sel]
        d = derive_all(build_raw_chunk(split, lo, hi, sc), uc, names)
        rs = cid2start[cid]
        assert rs + d.shape[0] <= out.shape[0], f"{cid} 行区间越界"
        out[rs:rs + d.shape[0]] = d.astype(np.float16)
        if verbose:
            print(f"[fine_derive] {cid} {d.shape[0]} 样本（{time.time()-t0:.0f}s）", flush=True)
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="细段派生通道：自检（不训练）")
    p.add_argument("--months", default="0", help="用哪些月抽一段自检")
    p.add_argument("--pool", default=None)
    args = p.parse_args()

    names = parse_spec(args.pool) if args.pool else list(ALL_NAMES)
    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).sort("sample_id")
    sids_all = lab["sample_id"].to_numpy()
    chunks = list(flow_feats.iter_chunks("train", args.months))
    u_all = flow_feats.load_u_s("train", sids_all)
    print(f"自检 {len(names)} 条通道，月份 {args.months}")
    for cid, lo, hi in chunks:
        sel = (sids_all >= lo) & (sids_all <= hi)
        t0 = time.time()
        full, last2 = reduce_chunk("train", lo, hi, sids_all[sel], u_all[sel], names)
        print(f"  {cid}: {full.shape[0]} 样本 × {len(names)} 通道  "
              f"({time.time()-t0:.1f}s)")
        print(f"    {'通道':<20}{'非零率':>9}{'std':>11}{'全窗均值':>12}{'末2格均值':>12}")
        for k, nm in enumerate(names):
            print(f"    {nm:<20}{np.mean(full[:, k] != 0):>9.3f}{full[:, k].std():>11.5f}"
                  f"{full[:, k].mean():>12.5f}{last2[:, k].mean():>12.5f}")
