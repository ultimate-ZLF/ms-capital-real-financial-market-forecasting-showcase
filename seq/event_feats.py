"""event_feats.py — 细段**事件轴**缓存（**扁平存储版**，2026-09-30）：

    order 事件流：V_o (M_o, 6) f16 + offs_o (N+1,) int64
    txn   事件流：V_t (M_t, 5) f16 + offs_t (N+1,) int64

`offs[s]..offs[s+1]` 就是第 s 条样本（按 sample_id 升序）的事件，**样本内旧→新**。

## 为什么从定长 `(N, L, C)` 改成扁平

定长版必须挑一个全局 `L`，而 **L 由长尾决定、浪费由中位付**，两者相差 4~7 倍：

| 流 | 中位事件数 | p99 | 定长 L | 中位样本的 pad | 截断（train/test）|
|---|---|---|---|---|---|
| order | 88 | 716 | 384 | **77%** | 6.3% / 12.5% |
| txn | 48 | 495 | 320 | **85%** | 3.6% / 6.8% |

（桶式臂的 60 格天生没有这两个病——它的格号是"距现在第 k 秒"，跨样本对齐，
空着就是**真实的零活动**；事件轴的格号是"第 k 个事件"，跨样本不对齐，只能补 pad。）

扁平存储**两个病一起消掉**：`n_kept == n_ev`（**零截断**），批内只 pad 到批内最长
（配合按长度排序的采样器，pad ≈ 10%）。内存也从 10.8GB 降到 3.6GB。

**⚠️ 前缀 `valid` 通道已删**：扁平存储里没有"补位行"，`valid` 恒为 1 是死通道；
批内 pad 由上层 collate 补零产生，而因果卷积下 pad 落在读出点之后、**对读出零影响**。

## 两条流分开（分塔）

依据 `HYPOTHESES C-13`（2026-09-17 已裁决）：在"粗段 vs 细段"上实测
**④ 先分再合（各自走完整主干、head 处 concat）= 0.12280** 压倒 ① 全合 = 0.10626
（**+0.01654，6/6 折为正**），机理是**分源让每条塔按自己源的时间结构去读**。

⚠️ 但 2026-09-29 在 order/tx 这一对上**分塔没有复现该增益**（单模型 0.13893 vs
合并 0.13976，打平），只提高了**集成价值**（作为第二成员 +0.00532 vs +0.00361）。
记录在此，别把 C-13 的结论无脑外推。

## 平局：一个事件序列里 40% 的相邻顺序问题

时间戳是**1 毫秒网格**（唯一值 60020 ≈ 60001 = 60s ÷ 1ms），而事件**成簇到达**：
相邻事件的到达间隔**中位 2.0 毫秒**、**38.74% 的相邻对时间戳完全相同**。
只看 `(sample_id, seconds_before_predict)` 排序时约 40% 的相邻顺序未定义
（polars 默认非稳定排序）。

修法：`_row`（**原始文件行号**）。依据：实测两个文件里同一样本的行都**严格按时间单调**
（0 个逆序样本）⇒ **文件行序 = 模拟器的发出顺序**，这是真实信息。
两流分开后**不再需要跨表约定**（`kind` 通道也随之消失）。

⚠️ **合同**：`o` / `t` 必须**按原始文件行序**传入（`build_chunk` 用 `filter + select` 读，保序）。
违反会**报错**（样本内 `seconds_before_predict` 非递增的断言），不会静默降级。

## 通道（order 6 / txn 5）

| order | txn | 定义 |
|---|---|---|
| `dt_log` | `dt_log` | `log1p(距本流上一个事件的秒数)`；每流首个用**它自己的 `sec`** |
| `t_pos` | `t_pos` | `sec / 60`（0=现在，1=窗口起点）|
| `side` | `side` | order=挂单买卖方向；txn=**主动方**（分开后各自语义唯一）|
| `action` | — | 0=挂，1=撤（txn 没有这个概念，该通道不存在）|
| `px_dev` | `px_dev` | `clip((price − mid0)/U_s, ±64)`——两流**共用**基准，故可比 |
| `vol_rel` | `vol_rel` | `asinh(volume / 本样本**本流**每秒均量)`，分母下限 1 |

⚠️ `px_dev` 用 `mid0`（窗口起点附近的 mid）而不是"当时的 mid"——后者要 asof-join
`market` 表，本版没做。这是**近似**，别当成"已经做对了"。

**空样本**（某条流 60s 内零事件）：在扁平数组里给它**一行全零哨兵**（`n_kept = 1`），
保证上层读出下标 `n_kept−1 = 0` 恒合法（`-1` 会静默环绕）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flow_feats  # noqa: E402  借它的 chunk 协议 / tick 回退常数（单一来源）
from seq_common import BASE  # noqa: E402

CACHE = os.path.join(BASE, "event_cache")
SNAP_CACHE = os.path.join(BASE, "snap_cache")

O_NAMES = ["dt_log", "t_pos", "side", "action", "px_dev", "vol_rel"]
T_NAMES = ["dt_log", "t_pos", "side", "px_dev", "vol_rel"]
C_O = len(O_NAMES)
C_T = len(T_NAMES)
IX_O = {n: i for i, n in enumerate(O_NAMES)}
IX_T = {n: i for i, n in enumerate(T_NAMES)}

# 数据布局版本（进检查点指纹，防换版本后静默复用另一套布局的权重）：
#   1 = 合并单序列定长 (N,L,8)，平局顺序未定义
#   2 = 合并单序列定长 + `_row` 平局裁判
#   3 = 分塔定长 (N,384,7)+(N,320,6) + `_row`
#   4 = **分塔 + 扁平存储**（零截断、无 valid 通道、批内 pad）（本版）
LAYOUT_VERSION = 4

EVT_SECONDS = 60.0
NA_ACTION = 0.5         # transaction 没有 order_action（本版不给它建通道，常量仅供测试引用）
CLIP_PX = 64.0

SPLIT_N = {"train": 1_257_637, "test": 647_896}
O_COLS = ["sample_id", "seconds_before_predict", "price", "volume", "side", "order_action"]
T_COLS = ["sample_id", "seconds_before_predict", "price", "volume", "side"]


def load_ref(split, sids):
    """(U_s, mid0) —— per-sample 的 tick 单位与价格基准，取自 snap index（零拟合）。"""
    p = os.path.join(SNAP_CACHE, f"{split}_snap_index.parquet")
    if not os.path.exists(p):
        print(f"[warn] 缺 {p}，U_s 回退 tick 常数 {flow_feats.TICK_FALLBACK[split]}，mid0 回退 1.0",
              flush=True)
        return (np.full(sids.size, flow_feats.TICK_FALLBACK[split], dtype=np.float64),
                np.ones(sids.size, dtype=np.float64))
    idx = pl.read_parquet(p).sort("sample_id")
    assert idx.height == sids.size, f"index 行数 {idx.height} != {sids.size}"
    assert (idx["sample_id"].to_numpy() == sids).all(), "index 与 sample_id 行序不一致"
    u_s = idx["U_s"].fill_null(flow_feats.TICK_FALLBACK[split]).to_numpy().astype(np.float64)
    mid0 = idx["mid0"].fill_null(1.0).to_numpy().astype(np.float64)
    return u_s, mid0


def _flat_stream(d, sids_chunk, u_s, mid0, names, name):
    """**单条流** → `(V (M,C) f16, offs (n+1,) int64, n_ev (n,) int32)`。两条流共用这一处逻辑。

    `d` 必须**按原始文件行序**传入（行号是平局裁判）；违反会被下面的断言拦住。
    `offs[s]..offs[s+1]` = 第 s 条样本的事件（旧→新）；**空样本给 1 行全零哨兵**。
    """
    n = sids_chunk.size
    C = len(names)
    IX = {k: i for i, k in enumerate(names)}

    # ---- 合同检查：文件里同一样本的行必须按时间单调（否则 `_row` 不再是「发出顺序」）----
    if d.height:
        _g = d.select(pl.col("seconds_before_predict").cast(pl.Float64)
                      .diff().over("sample_id").alias("g")).get_column("g").to_numpy()
        _bad = int((_g > 0).sum())
        assert _bad == 0, (
            f"{name} 表未按时间单调传入（{_bad} 处逆序）：`_row` 会失去「发出顺序」的含义。"
            f"上游读取必须保序（filter+select），不能加会重排的算子")

    if not d.height:
        n_ev = np.zeros(n, dtype=np.int32)
        offs = np.arange(n + 1, dtype=np.int64)
        return np.zeros((n, C), dtype=np.float16), offs, n_ev

    # ---- 贴标签 + 行号（**排序前**取，就是文件里的发出顺序）----
    sel = [pl.col("sample_id").cast(pl.Int64), pl.col("seconds_before_predict").cast(pl.Float64),
           pl.col("price").cast(pl.Float64), pl.col("volume").cast(pl.Float64),
           pl.col("side").cast(pl.Float64)]
    if "action" in IX:
        sel.append(pl.col("order_action").cast(pl.Float64).alias("action"))
    ev = d.select(sel).with_columns(pl.int_range(pl.len()).alias("_row"))
    # 行序 = 旧→新；同刻按**发出顺序**（`_row`）
    ev = ev.sort(["sample_id", "seconds_before_predict", "_row"],
                 descending=[False, True, False])

    sid = ev["sample_id"].to_numpy()
    sec = ev["seconds_before_predict"].to_numpy()
    m = sid.size

    # ---- 逐样本分组（sids_chunk 与 ev 都按 sample_id 升序 → searchsorted 即可）----
    starts = np.searchsorted(sid, sids_chunk, side="left")
    ends = np.searchsorted(sid, sids_chunk, side="right")
    n_ev = (ends - starts).astype(np.int32)
    grp = np.repeat(np.arange(n), n_ev)
    idx_in = np.arange(m, dtype=np.int64) - np.repeat(starts, n_ev)
    assert m == int(n_ev.sum()), f"事件行数 {m} != 各样本事件数之和 {int(n_ev.sum())}"

    # ---- 本样本本流每秒均量（分母下限 1.0；无事件的样本 → 1.0）----
    u = np.ones(n, dtype=np.float64)
    st = ev.group_by("sample_id").agg(pl.col("volume").sum().alias("vsum")).sort("sample_id")
    s_sid = st["sample_id"].to_numpy()
    r2 = np.searchsorted(sids_chunk, s_sid)
    assert (sids_chunk[r2] == s_sid).all(), "聚合的 sample_id 不在本块范围内"
    u[r2] = np.maximum(st["vsum"].to_numpy() / EVT_SECONDS, 1.0)

    # ---- dt：本流内与前一个事件的秒差；每流首个事件 = 它自己的 sec ----
    same = np.zeros(m, dtype=bool)
    same[1:] = sid[1:] == sid[:-1]
    dt = np.maximum(np.where(same, np.roll(sec, 1) - sec, sec), 0.0)

    # ---- 扁平布局：空样本占 1 行哨兵（保证 offs 单调、读出下标恒合法）----
    nk = np.maximum(n_ev, 1).astype(np.int64)
    offs = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(nk, out=offs[1:])
    dst = offs[grp] + idx_in
    V = np.zeros((int(offs[-1]), C), dtype=np.float64)
    V[dst, IX["dt_log"]] = np.log1p(dt)
    V[dst, IX["t_pos"]] = np.clip(sec / EVT_SECONDS, 0.0, 1.0)
    V[dst, IX["side"]] = ev["side"].to_numpy()
    if "action" in IX:
        V[dst, IX["action"]] = ev["action"].to_numpy()
    V[dst, IX["px_dev"]] = np.clip((ev["price"].to_numpy() - mid0[grp]) / u_s[grp], -CLIP_PX, CLIP_PX)
    V[dst, IX["vol_rel"]] = np.arcsinh(ev["volume"].to_numpy() / u[grp])

    if not np.isfinite(V).all():
        raise AssertionError(f"{name} 有 {int((~np.isfinite(V)).sum())} 个非有限值")
    return V.astype(np.float16), offs, n_ev


def build_from_events(o, t, sids_chunk, u_s, mid0):
    """**纯函数**：两张事件表 → `((Vo, offs_o, n_ev_o), (Vt, offs_t, n_ev_t))`。

    `o` 需含 `O_COLS`；`t` 需含 `T_COLS`。**两者都必须按原始文件行序传入**（见模块头「平局」）。
    """
    return (_flat_stream(o, sids_chunk, u_s, mid0, O_NAMES, "order"),
            _flat_stream(t, sids_chunk, u_s, mid0, T_NAMES, "transaction"))


def build_chunk(split, lo, hi, sids_chunk, u_s_chunk, mid0_chunk):
    """读一个块的原始两表 → 调 `build_from_events`。"""
    o = (pl.scan_parquet(os.path.join(BASE, split, "order.parquet"))
         .filter(pl.col("sample_id").is_between(lo, hi)).select(O_COLS).collect())
    t = (pl.scan_parquet(os.path.join(BASE, split, "transaction.parquet"))
         .filter(pl.col("sample_id").is_between(lo, hi)).select(T_COLS).collect())
    return build_from_events(o, t, sids_chunk, u_s_chunk, mid0_chunk)


def event_counts(split):
    """只扫两条流的事件数分布（诊断用）→ ((o, t), sids)。"""
    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet"))
    sids = np.sort(lab["sample_id"].to_numpy())
    out = []
    for tbl in ("order", "transaction"):
        c = (pl.scan_parquet(os.path.join(BASE, split, f"{tbl}.parquet"))
             .select(["sample_id"]).group_by("sample_id").len().collect().sort("sample_id"))
        v = np.zeros(sids.size, dtype=np.int64)
        r = np.searchsorted(sids, c["sample_id"].to_numpy())
        assert (sids[r] == c["sample_id"].to_numpy()).all()
        v[r] = c["len"].to_numpy()
        out.append(v)
    return tuple(out), sids


def build_flat(split, months_arg=None, to_memory=True):
    """建两条流的**扁平**缓存 → `(Vo, offs_o, Vt, offs_t, df_index)`。

    ⚠️ 刻意没有 `--resume`（CLAUDE.md #4 的"静默全零"陷阱）；整块重建只要几分钟。
    """
    os.makedirs(CACHE, exist_ok=True)
    N = SPLIT_N[split]

    if split == "train":
        lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).sort("sample_id")
        all_sids = lab["sample_id"].to_numpy()
        counts = lab.group_by("month").len().sort("month")["len"].to_numpy()
        cid2start = {f"m{m}": int(counts[:m].sum()) for m in range(len(counts))}
    else:
        all_sids = pl.read_csv(os.path.join(BASE, "submissions/submission.csv"),
                               columns=["sample_id"]).sort("sample_id")["sample_id"].to_numpy()
        chunks0 = list(flow_feats.iter_chunks(split, None))
        cid2start = {cid: lo - chunks0[0][1] for cid, lo, _ in chunks0}

    u_s_all, mid0_all = load_ref(split, all_sids)

    chunks = list(flow_feats.iter_chunks(split, months_arg))
    Vo_parts, offs_o_parts = [], [np.zeros(1, dtype=np.int64)]
    Vt_parts, offs_t_parts = [], [np.zeros(1, dtype=np.int64)]
    nev_o, nev_t = [], []
    base_o = base_t = 0
    t_all = time.time()
    for cid, lo, hi in chunks:
        t0 = time.time()
        sel = (all_sids >= lo) & (all_sids <= hi)
        sids_chunk = all_sids[sel]
        (Vc_o, oc_o, nc_o), (Vc_t, oc_t, nc_t) = build_chunk(
            split, lo, hi, sids_chunk, u_s_all[sel], mid0_all[sel])
        rs = cid2start[cid]
        assert rs + sids_chunk.size <= N, f"{cid}: {rs}+{sids_chunk.size} > {N}"
        Vo_parts.append(Vc_o); offs_o_parts.append(oc_o[1:] + base_o); base_o += Vc_o.shape[0]
        Vt_parts.append(Vc_t); offs_t_parts.append(oc_t[1:] + base_t); base_t += Vc_t.shape[0]
        nev_o.append(nc_o); nev_t.append(nc_t)
        print(f"[{split}] {cid} samples={sids_chunk.size} order {Vc_o.shape[0]:,} / "
              f"txn {Vc_t.shape[0]:,} {time.time()-t0:.1f}s | {len(nev_o)}/{len(chunks)}", flush=True)

    Vo = np.concatenate(Vo_parts); offs_o = np.concatenate(offs_o_parts)
    Vt = np.concatenate(Vt_parts); offs_t = np.concatenate(offs_t_parts)
    del Vo_parts, Vt_parts
    assert offs_o.shape == (N + 1,) and offs_o[-1] == Vo.shape[0], \
        f"order offs {offs_o.shape}/{offs_o[-1]} != V {Vo.shape}"
    assert offs_t.shape == (N + 1,) and offs_t[-1] == Vt.shape[0], "txn offs 与 V 不匹配"
    assert (np.diff(offs_o) >= 1).all() and (np.diff(offs_t) >= 1).all(), "有样本占 0 行"

    df = pl.DataFrame({"sample_id": all_sids,
                       "n_ev_o": np.concatenate(nev_o), "n_ev_t": np.concatenate(nev_t)}).sort("sample_id")
    assert df.height == N, f"index 行数 {df.height} != {N}"
    mf = os.path.join(CACHE, f"{split}_evt_index.parquet")
    df.write_parquet(mf)

    if to_memory:
        print(f"[{split}] 内存模式：order {Vo.nbytes/2**30:.2f}GB + txn {Vt.nbytes/2**30:.2f}GB "
              f"= {(Vo.nbytes+Vt.nbytes)/2**30:.2f}GB", flush=True)
    else:
        np.save(os.path.join(CACHE, f"{split}_evt_o.npy"), Vo)
        np.save(os.path.join(CACHE, f"{split}_evt_t.npy"), Vt)
        np.save(os.path.join(CACHE, f"{split}_evt_offs_o.npy"), offs_o)
        np.save(os.path.join(CACHE, f"{split}_evt_offs_t.npy"), offs_t)
    for tag, v in (("order", df["n_ev_o"].to_numpy()), ("txn", df["n_ev_t"].to_numpy())):
        q = {p: int(np.percentile(v, p)) for p in (50, 90, 99)}
        print(f"[{split}] {tag}: 事件数 中位/90/99 = {q[50]}/{q[90]}/{q[99]}，max={int(v.max())}，"
              f"**零截断**（扁平存储）", flush=True)
    print(f"[{split}] index → {mf}\n[{split}] 构建完成，总耗时 {(time.time()-t_all)/60:.1f}min",
          flush=True)
    return Vo, offs_o, Vt, offs_t, df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "test"])
    p.add_argument("--months", default=None)
    p.add_argument("--stats", action="store_true", help="只打印两条流的事件数分布")
    args = p.parse_args()
    if args.stats:
        (o, t), _ = event_counts(args.split)
        for tag, v in (("order", o), ("txn", t)):
            print(f"[{args.split}] {tag}: mean={v.mean():.1f} 中位={int(np.median(v))}", end="")
            for q in (90, 99, 99.9):
                print(f" q{q}={int(np.percentile(v, q))}", end="")
            print(f" max={int(v.max())}")
    else:
        build_flat(args.split, args.months, to_memory=True)
