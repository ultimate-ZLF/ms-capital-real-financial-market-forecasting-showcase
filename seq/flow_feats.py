"""flow_feats.py — 细段缓存：最近 60s × 1s 的成交流/挂单流特征 (N,60,16) f16。

设计见 DESIGN_CNN_V2.md §4.2 / §4.3。要点（新会话避雷）：
1. bin 编号 `59 - floor(seconds_before_predict)`：bin 0 = 60s 前，bin 59 = 现在。
   与粗段同向（旧→新），这是因果卷积的前提。
2. 所有通道 = 比值 / asinh / per-sample 归一。**严禁** z-score、分位数分箱、全局分位阈值
   （三者都在本项目的衰减比上有实证折损：0.797 / 0.822 / v3）。
3. **空 bin = 真实零活动，不是缺失** → 0 填充，细段不需要 mask（60 个 bin 对每个样本都存在）。
3b. 量级类通道（挂单/撤单/成交量）用 **per-sample 每秒均值**归一后再 asinh：
   实测 test 的单笔挂单量是 train 的约 3 倍（asinh 中位 5.30 vs 6.40），
   而每秒事件数两边相同（flow_cnt 中位都是 3）——是订单规模差异不是活跃度差异。
   除以本样本自身的每秒均值把这一层 regime 差异彻底消掉（零拟合）；
   活跃度的绝对水平由 flow_cnt / has_event 承担，不受影响。
4. 价格基准用 per-sample 60s VWAP（自包含在 flow 表里，不读 market）。
5. tick 单位 U_s 取自 snap_cache index（已存在）；缺失时回退 DATA.md 的 tick 常数。
6. order_action 的 0=挂/1=撤 是未验证假设（DATA.md）——撤单类通道依赖它。

用法：
  python flow_feats.py --split test              # 13 块
  python flow_feats.py --split train             # 71 个月
  python flow_feats.py --split train --months 0,1,2
  python flow_feats.py --split train --resume    # 断点续跑（读 progress parquet）
"""
import argparse
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seq_common import BASE  # noqa: E402  缓存根由数据管道统一解析

CACHE = os.path.join(BASE, "flow_cache")
SNAP_CACHE = os.path.join(BASE, "snap_cache")
FINE_T = 60
K = 16
EPS = 1e-6
SPLIT_N = {"train": 1_257_637, "test": 647_896}
TICK_FALLBACK = {"train": 0.0015037, "test": 0.0012257}

FEAT_NAMES = [
    "o_new_buy",     # 0  asinh(每秒新挂买量 / 本样本每秒均量)
    "o_new_sell",    # 1  asinh(每秒新挂卖量 / 本样本每秒均量)
    "o_cancel_buy",  # 2  asinh(每秒买侧撤单量 / 本样本每秒撤单均量)
    "o_cancel_sell", # 3  asinh(每秒卖侧撤单量 / 本样本每秒撤单均量)
    "o_new_imb",     # 4  (新挂买−新挂卖)/(新挂买+新挂卖+eps)
    "o_cancel_ratio",# 5  撤单量/(新挂量+撤单量+eps)
    "o_px_dev",      # 6  (本秒挂单 VWAP − 全窗挂单 VWAP)/U_s
    "t_vol_buy",     # 7  asinh(每秒买方主动成交量 / 本样本每秒均量)
    "t_vol_sell",    # 8  asinh(每秒卖方主动成交量 / 本样本每秒均量)
    "t_imb",         # 9  (买量−卖量)/(买量+卖量+eps)
    "t_px_dev",      # 10 (本秒成交 VWAP − 全窗成交 VWAP)/U_s
    "t_size_rel",    # 11 本秒单笔均量/全窗单笔均量 − 1
    "x_to_buy_diff", # 12 t 买方笔数占比 − o 买方笔数占比（跨流方向差）
    "x_flow_gap",    # 13 min(距上一个事件 bin 数, 10)/10
    "flow_cnt",      # 14 asinh(本秒 order+tx 事件总数)
    "has_event",     # 15 本秒是否有任何 flow 事件 0/1
]
IDX = {n: i for i, n in enumerate(FEAT_NAMES)}


def load_u_s(split, sids):
    """U_s（per-sample 正 spread 中位数 ≈ 1 tick）来自 snap_cache index；缺失回退 tick 常数。"""
    p = os.path.join(SNAP_CACHE, f"{split}_snap_index.parquet")
    if not os.path.exists(p):
        print(f"[warn] 缺 {p}，U_s 全部回退 tick 常数 {TICK_FALLBACK[split]}", flush=True)
        return np.full(sids.size, TICK_FALLBACK[split], dtype=np.float64)
    idx = pl.read_parquet(p).sort("sample_id")
    assert idx.height == sids.size, f"index 行数 {idx.height} != {sids.size}"
    assert (idx["sample_id"].to_numpy() == sids).all(), "index 与 sample_id 行序不一致"
    return idx["U_s"].fill_null(TICK_FALLBACK[split]).to_numpy().astype(np.float64)


def iter_chunks(split, months_arg):
    """产出 (chunk_id, lo, hi)。train 按月；test 按 5 万样本切块（与 snap_feats 同协议）。"""
    if split == "train":
        lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet"))
        months = sorted(lab["month"].unique().to_list()) if months_arg is None \
            else [int(x) for x in months_arg.split(",")]
        for m in months:
            sub = lab.filter(pl.col("month") == m)
            yield f"m{m}", int(sub["sample_id"].min()), int(sub["sample_id"].max())
    else:
        sub = pl.read_csv(os.path.join(BASE, "submissions/submission.csv"),
                          columns=["sample_id"])
        lo, hi_all = int(sub["sample_id"].min()), int(sub["sample_id"].max())
        while lo <= hi_all:
            hi = min(lo + 49_999, hi_all)
            yield f"c{lo}", lo, hi
            lo = hi + 1


def build_chunk(split, lo, hi, sids_chunk, u_s_chunk):
    """构建一个块的 (n,60,16) 特征矩阵。"""
    n = sids_chunk.size
    X = np.zeros((n, FINE_T, K), dtype=np.float64)

    # ---------- order 表 ----------
    o = (pl.scan_parquet(os.path.join(BASE, split, "order.parquet"))
         .filter(pl.col("sample_id").is_between(lo, hi))
         .select(["sample_id", "seconds_before_predict", "price", "volume",
                  "side", "order_action"])
         .collect())
    o = o.with_columns([
        (59 - pl.col("seconds_before_predict").floor().cast(pl.Int16))
          .clip(0, FINE_T - 1).alias("bin"),
        (pl.col("order_action") == 0).alias("_is_new"),
        (pl.col("side") == 0).alias("_is_buy"),
    ]).with_columns([
        (pl.col("_is_new") & pl.col("_is_buy")).alias("_new_buy"),
        (pl.col("_is_new") & (pl.col("side") == 1)).alias("_new_sell"),
        ((pl.col("order_action") == 1) & pl.col("_is_buy")).alias("_cancel_buy"),
        ((pl.col("order_action") == 1) & (pl.col("side") == 1)).alias("_cancel_sell"),
    ])
    og = o.group_by(["sample_id", "bin"]).agg([
        pl.col("volume").filter(pl.col("_new_buy")).sum().alias("new_buy"),
        pl.col("volume").filter(pl.col("_new_sell")).sum().alias("new_sell"),
        pl.col("volume").filter(pl.col("_cancel_buy")).sum().alias("cancel_buy"),
        pl.col("volume").filter(pl.col("_cancel_sell")).sum().alias("cancel_sell"),
        (pl.col("price") * pl.col("volume")).sum().alias("px_vol"),
        pl.col("volume").sum().alias("vol"),
        pl.len().alias("cnt"),
        pl.col("_is_buy").sum().alias("buy_cnt"),
    ])
    del o
    sid_o = og["sample_id"].to_numpy()
    row = np.searchsorted(sids_chunk, sid_o)
    assert (sids_chunk[row] == sid_o).all(), "order 的 sample_id 不在本块范围内"
    b = og["bin"].to_numpy()
    o_new_buy = np.zeros((n, FINE_T)); o_new_sell = np.zeros((n, FINE_T))
    o_can_buy = np.zeros((n, FINE_T)); o_can_sell = np.zeros((n, FINE_T))
    o_vol = np.zeros((n, FINE_T)); o_px = np.zeros((n, FINE_T)); o_pxv = np.zeros((n, FINE_T))
    o_cnt = np.zeros((n, FINE_T)); o_bcnt = np.zeros((n, FINE_T))
    o_new_buy[row, b] = og["new_buy"].to_numpy()
    o_new_sell[row, b] = og["new_sell"].to_numpy()
    o_can_buy[row, b] = og["cancel_buy"].to_numpy()
    o_can_sell[row, b] = og["cancel_sell"].to_numpy()
    o_vol[row, b] = og["vol"].to_numpy()
    o_pxv[row, b] = og["px_vol"].to_numpy()
    o_cnt[row, b] = og["cnt"].to_numpy()
    o_bcnt[row, b] = og["buy_cnt"].to_numpy()
    del og

    # ---------- transaction 表 ----------
    t = (pl.scan_parquet(os.path.join(BASE, split, "transaction.parquet"))
         .filter(pl.col("sample_id").is_between(lo, hi))
         .select(["sample_id", "seconds_before_predict", "price", "volume", "side"])
         .collect())
    t = t.with_columns([
        (59 - pl.col("seconds_before_predict").floor().cast(pl.Int16))
          .clip(0, FINE_T - 1).alias("bin"),
        (pl.col("side") == 0).alias("_is_buy"),
    ])
    tg = t.group_by(["sample_id", "bin"]).agg([
        pl.col("volume").filter(pl.col("_is_buy")).sum().alias("buy"),
        pl.col("volume").filter(~pl.col("_is_buy")).sum().alias("sell"),
        (pl.col("price") * pl.col("volume")).sum().alias("px_vol"),
        pl.col("volume").sum().alias("vol"),
        pl.len().alias("cnt"),
        pl.col("_is_buy").sum().alias("buy_cnt"),
    ])
    del t
    sid_t = tg["sample_id"].to_numpy()
    row = np.searchsorted(sids_chunk, sid_t)
    assert (sids_chunk[row] == sid_t).all(), "transaction 的 sample_id 不在本块范围内"
    b = tg["bin"].to_numpy()
    t_buy = np.zeros((n, FINE_T)); t_sell = np.zeros((n, FINE_T))
    t_vol = np.zeros((n, FINE_T)); t_pxv = np.zeros((n, FINE_T))
    t_cnt = np.zeros((n, FINE_T)); t_bcnt = np.zeros((n, FINE_T))
    t_buy[row, b] = tg["buy"].to_numpy()
    t_sell[row, b] = tg["sell"].to_numpy()
    t_vol[row, b] = tg["vol"].to_numpy()
    t_pxv[row, b] = tg["px_vol"].to_numpy()
    t_cnt[row, b] = tg["cnt"].to_numpy()
    t_bcnt[row, b] = tg["buy_cnt"].to_numpy()
    del tg

    # ---------- per-sample 归一化基准（全部来自本样本自身） ----------
    U = u_s_chunk[:, None]
    o_new_tot = (o_new_buy + o_new_sell).sum(axis=1, keepdims=True)
    o_can_tot = (o_can_buy + o_can_sell).sum(axis=1, keepdims=True)
    o_vol_tot = o_vol.sum(axis=1, keepdims=True)
    o_pxv_tot = o_pxv.sum(axis=1, keepdims=True)
    o_vwap = np.divide(o_pxv_tot, o_vol_tot, out=np.zeros_like(o_pxv_tot), where=o_vol_tot > 0)
    t_vol_tot = t_vol.sum(axis=1, keepdims=True)
    t_cnt_tot = t_cnt.sum(axis=1, keepdims=True)
    t_pxv_tot = t_pxv.sum(axis=1, keepdims=True)
    t_vwap = np.divide(t_pxv_tot, t_vol_tot, out=np.zeros_like(t_pxv_tot), where=t_vol_tot > 0)
    t_size_smp = np.divide(t_vol_tot, t_cnt_tot, out=np.zeros_like(t_vol_tot), where=t_cnt_tot > 0)
    t_size_bin = np.divide(t_vol, t_cnt, out=np.zeros_like(t_vol), where=t_cnt > 0)

    # ---------- 16 通道 ----------
    # 量级类：÷ per-sample 每秒均值（分母下限 1，防极稀疏样本放大）后再 asinh
    o_unit = np.maximum((o_new_buy + o_new_sell).sum(axis=1, keepdims=True) / FINE_T, 1.0)
    o_can_unit = np.maximum((o_can_buy + o_can_sell).sum(axis=1, keepdims=True) / FINE_T, 1.0)
    t_unit = np.maximum((t_buy + t_sell).sum(axis=1, keepdims=True) / FINE_T, 1.0)
    X[:, :, IDX["o_new_buy"]] = np.arcsinh(o_new_buy / o_unit)
    X[:, :, IDX["o_new_sell"]] = np.arcsinh(o_new_sell / o_unit)
    X[:, :, IDX["o_cancel_buy"]] = np.arcsinh(o_can_buy / o_can_unit)
    X[:, :, IDX["o_cancel_sell"]] = np.arcsinh(o_can_sell / o_can_unit)
    den = o_new_buy + o_new_sell
    X[:, :, IDX["o_new_imb"]] = np.divide(o_new_buy - o_new_sell, den + EPS,
                                          out=np.zeros_like(den), where=den > 0)
    den2 = o_new_buy + o_new_sell + o_can_buy + o_can_sell
    X[:, :, IDX["o_cancel_ratio"]] = np.divide(o_can_buy + o_can_sell, den2 + EPS,
                                               out=np.zeros_like(den2), where=den2 > 0)
    o_px_bin = np.divide(o_pxv, o_vol, out=np.zeros_like(o_pxv), where=o_vol > 0)
    X[:, :, IDX["o_px_dev"]] = np.where(o_vol > 0,
                                        np.clip((o_px_bin - o_vwap) / U, -50, 50), 0.0)
    X[:, :, IDX["t_vol_buy"]] = np.arcsinh(t_buy / t_unit)
    X[:, :, IDX["t_vol_sell"]] = np.arcsinh(t_sell / t_unit)
    den3 = t_buy + t_sell
    X[:, :, IDX["t_imb"]] = np.divide(t_buy - t_sell, den3 + EPS,
                                      out=np.zeros_like(den3), where=den3 > 0)
    t_px_bin = np.divide(t_pxv, t_vol, out=np.zeros_like(t_pxv), where=t_vol > 0)
    X[:, :, IDX["t_px_dev"]] = np.where(t_vol > 0,
                                        np.clip((t_px_bin - t_vwap) / U, -50, 50), 0.0)
    X[:, :, IDX["t_size_rel"]] = np.where(t_cnt > 0,
                                          np.clip(t_size_bin / (t_size_smp + EPS) - 1.0, -5, 5), 0.0)
    t_buy_r = np.divide(t_bcnt, t_cnt, out=np.zeros_like(t_bcnt), where=t_cnt > 0)
    o_buy_r = np.divide(o_bcnt, o_cnt, out=np.zeros_like(o_bcnt), where=o_cnt > 0)
    X[:, :, IDX["x_to_buy_diff"]] = np.where((t_cnt > 0) & (o_cnt > 0),
                                             t_buy_r - o_buy_r, 0.0)
    X[:, :, IDX["flow_cnt"]] = np.arcsinh(o_cnt + t_cnt)
    X[:, :, IDX["has_event"]] = (o_cnt + t_cnt > 0).astype(np.float64)

    # ---------- x_flow_gap：要在整块上沿 bin 轴累进 ----------
    pos = np.arange(FINE_T)[None, :]
    last = np.maximum.accumulate(np.where(X[:, :, IDX["has_event"]] > 0, pos, -1), axis=1)
    X[:, :, IDX["x_flow_gap"]] = np.clip(pos - last, 0, 10) / 10.0

    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).sum())
        raise AssertionError(f"块 {lo}-{hi} 有 {bad} 个非有限值")
    return X.astype(np.float16), np.stack([o_cnt.sum(axis=1), t_cnt.sum(axis=1)], axis=1)


def build_split(split, months_arg, resume, to_memory=False):
    """to_memory=True：建到**内存数组**返回，不写 `.dat`/progress/index。

    提交时现算 test 特征用（test 缓存只用几次，存盘不划算——见 snap_feats 同名参数）。
    索引 `{split}_flow_index.parquet` 在内存模式下不重建，沿用磁盘上已有的那份。
    """
    os.makedirs(CACHE, exist_ok=True)
    dat_path = os.path.join(CACHE, f"{split}_flow_X.dat")
    prog_path = os.path.join(CACHE, f"progress_{split}.parquet")
    done = set()
    if resume and not to_memory and os.path.exists(prog_path):
        done = set(pl.read_parquet(prog_path)["chunk"].to_list())
        print(f"[resume] 已完成 {len(done)} 块", flush=True)

    N = SPLIT_N[split]
    if to_memory:
        X = np.zeros((N, FINE_T, K), dtype=np.float16)
        print(f"[{split}] 内存模式：{X.nbytes/2**30:.2f}GB 常驻，不写 .dat", flush=True)
    elif os.path.exists(dat_path):
        X = np.lib.format.open_memmap(dat_path, mode="r+", dtype=np.float16)
        assert X.shape == (N, FINE_T, K), f"{dat_path} 形状 {X.shape} 不符"
    else:
        X = np.lib.format.open_memmap(dat_path, mode="w+", dtype=np.float16,
                                      shape=(N, FINE_T, K))

    chunks = list(iter_chunks(split, months_arg))
    if split == "train":
        lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).sort("sample_id")
        all_sids = lab["sample_id"].to_numpy()
        counts = lab.group_by("month").len().sort("month")["len"].to_numpy()
        cid2start = {f"m{m}": int(counts[:m].sum()) for m in range(len(counts))}
    else:
        all_sids = pl.read_csv(os.path.join(BASE, "submissions/submission.csv"),
                               columns=["sample_id"]).sort("sample_id")["sample_id"].to_numpy()
        lo_all = int(chunks[0][1])
        cid2start = {cid: lo - lo_all for cid, lo, _ in chunks}
    u_s_all = load_u_s(split, all_sids)

    meta_rows = []
    t_all = time.time()
    for cid, lo, hi in chunks:
        if cid in done:
            print(f"[{split}] {cid} 已跳过", flush=True)
            continue
        t0 = time.time()
        sel = (all_sids >= lo) & (all_sids <= hi)
        sids_chunk = all_sids[sel]
        Xc, meta = build_chunk(split, lo, hi, sids_chunk, u_s_all[sel])
        rs = cid2start[cid]
        assert rs + Xc.shape[0] <= N and Xc.shape[0] == sids_chunk.size
        X[rs:rs + Xc.shape[0]] = Xc
        if not to_memory:          # 内存模式不落盘、无断点（一次跑完，失败重来）
            X.flush()
            meta_rows.append(pl.DataFrame({
                "sample_id": sids_chunk,
                "o_n": meta[:, 0].astype(np.int32),
                "t_n": meta[:, 1].astype(np.int32),
            }))
            new_prog = pl.DataFrame({"chunk": [cid], "n_samples": [Xc.shape[0]],
                                     "done_at": [time.strftime("%H:%M:%S")]})
            if os.path.exists(prog_path):
                new_prog = pl.concat([pl.read_parquet(prog_path), new_prog])
            new_prog.write_parquet(prog_path)
        done.add(cid)
        print(f"[{split}] {cid} samples={Xc.shape[0]} {time.time()-t0:.1f}s | "
              f"{len(done)}/{len(chunks)}", flush=True)
    print(f"[{split}] 构建完成，总耗时 {(time.time()-t_all)/60:.1f}min", flush=True)
    if to_memory:
        print(f"[{split}] 内存构建完成（{X.nbytes/2**30:.2f}GB，未落盘）", flush=True)
        return X
    if meta_rows:
        mf = os.path.join(CACHE, f"{split}_flow_index.parquet")
        pl.concat(meta_rows).sort("sample_id").write_parquet(mf)
        print(f"[{split}] index → {mf}", flush=True)
    return X


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "test"])
    p.add_argument("--months", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--memory", action="store_true",
                   help="建到内存返回、不落盘（提交时现算 test 特征用）。"
                        "⚠️ 不带本开关 = 落盘写 .dat **和 progress_*.parquet**——细段的 .dat 自 "
                        "2026-09-16 起按设计不留；一旦落了盘，日后的 --resume 会踩 CLAUDE.md #4"
                        "「静默产出全零数据集」陷阱")
    args = p.parse_args()
    build_split(args.split, args.months, args.resume, to_memory=args.memory)
