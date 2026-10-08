"""snap_feats.py — 构建盘口快照序列缓存（f16 memmap (N,224,17) + index parquet）。

通道 = seq_common.FEAT_NAMES（14 个盘口/成交通道 + dt）。dt 见 DESIGN_CNN_V2.md §4.1。

用法：
  python snap_feats.py --split train            # 71 个月全量（实测 ~6min，可后台）
  python snap_feats.py --split test             # 13 块（实测 ~2min）
  python snap_feats.py --split train --months 0,1,2   # 只构建部分月（调试）
  python snap_feats.py --split train --resume   # 断点续跑（读 progress parquet）
  python snap_feats.py --split train --finalize # 合并 meta_parts → index parquet + 断言

坑（新会话避雷）：
1. 排序方向（最高危）：sort seconds_before_predict 降序 → t=0 最旧；断言 diff≤0 全成立。
   搞反 = ffill 用未来信息回填 = 泄漏，且因果卷积语义全错。
2. 空档侧行（ask1==0/bid1==0，0.55%）：8 列盘口先 ffill（over sample_id，旧→新）再算一切派生量。
3. pad 区特征全 0；rel_t 只在有效区非零。
4. RAM 上限 8GB → 绝不能整表 read_ipc；scan_ipc + sample_id 范围过滤按月 collect。
5. 重建成本（2026-09-14 实测，parquet 路径）：train 4.6~4.9s/月 × 71、test 10.3s/5万样本
   × 13 → 纯计算 ~8min。旧文档的 "55min/25min" 是 feather 时代数字（单批次压缩无行组下推，
   每月都要整文件解压 ~4GB），已转 parquet，勿再据此判断重建成本。
   但墙钟常由内存而非 CPU 决定：build_split 每块前等 ≥7GB 空闲，最多干等 30min/块。
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import polars as pl

import sys

# 同目录（直接跑本脚本时 Python 不保证 seq/ 在 sys.path 上）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seq_common import BASE, CACHE, F, FEAT_NAMES, T, avail_mem_bytes  # noqa: E402

TICK_FALLBACK = {"train": 0.0015037, "test": 0.0012257}  # DATA.md 实测 tick 等差
EPS = 1e-6
SPLIT_N = {"train": 1_257_637, "test": 647_896}
BOOK_COLS = ["ask_price_1", "bid_price_1", "ask_price_2", "bid_price_2",
             "ask_volume_1", "bid_volume_1", "ask_volume_2", "bid_volume_2"]


def get_market_scan(split):
    """market.parquet 的惰性 scan（行组下推，单月只解压 ~80MB）。"""
    pq = os.path.join(BASE, split, "market.parquet")
    if not os.path.exists(pq):
        raise FileNotFoundError(f"缺 {pq}——原始数据 2026-09-14 起全部是 parquet（CLAUDE.md 目录约定）")
    return pl.scan_parquet(pq)


def build_month(scan, lo, hi, split):
    """一个样本块（月/chunk）→ (X_block f16 (n,T,F), meta polars)。

    内存警告（实测）：单次 collect 峰值 ~5GB 且 polars 分配器不归还 OS（压缩
    feather 无法 mmap/谓词下推 → 整文件解压）→ 跨月累计必然 OOM。
    因此本函数必须在独立子进程里跑（见 build_split 的 --worker 模式）。
    """
    # 文件天然有序（实测：sample_id 全局非降 + 每样本内秒数非增）→ 不 sort，
    # 只在下方断言验证。sort 会额外吃 ~1GB 内存，而且每次 collect 都要整文件
    # 解压（压缩 feather 单批次文件无谓词下推），能省就省。
    df = scan.filter(pl.col("sample_id").is_between(lo, hi)).collect()

    # —— 顺序断言（最高危，见顶部注释）——
    sid = df["sample_id"].to_numpy()
    sec = df["seconds_before_predict"].to_numpy()
    assert (np.diff(sid) >= 0).all(), "sample_id 非全局升序——文件顺序假设不成立！"
    check = df.group_by("sample_id").agg([
        (pl.col("seconds_before_predict").diff() <= 0).all().alias("desc_ok"),
        pl.col("seconds_before_predict").first().alias("sec_first"),
        pl.col("seconds_before_predict").last().alias("sec_last"),
    ])
    assert check["desc_ok"].all(), "秒数未严格非增——排序方向错了！"
    assert (check["sec_last"] <= 120).all(), "最新快照距 predict 超过 120s，数据异常"
    n_block = check.height

    # —— 空档侧标记 + ffill（只用过去信息，零泄漏）——
    df = df.with_columns(
        ((pl.col("ask_price_1") > 0) & (pl.col("bid_price_1") > 0)).alias("_valid"))
    for c in BOOK_COLS:
        # polars 不支持链式窗口（forward_fill.over().backward_fill.over()）→ 拆两步
        tmp = f"_{c}_a"
        df = df.with_columns(
            pl.when(pl.col("_valid")).then(pl.col(c)).otherwise(None)
            .forward_fill().over("sample_id").alias(tmp))
        df = df.with_columns(
            pl.col(tmp).backward_fill().over("sample_id").alias(c)).drop(tmp)
    # 极端样本（全月无有效行）兜底：价格 1.0、量 0
    df = df.with_columns([
        pl.col("ask_price_1").fill_null(1.0), pl.col("bid_price_1").fill_null(1.0),
        pl.col("ask_price_2").fill_null(1.0), pl.col("bid_price_2").fill_null(1.0),
        pl.col("ask_volume_1").fill_null(0), pl.col("bid_volume_1").fill_null(0),
        pl.col("ask_volume_2").fill_null(0), pl.col("bid_volume_2").fill_null(0),
    ])

    # —— 基础派生列 ——
    # 注意：polars 除零得 inf（非 null）→ 分母为 0 的公式必须用 when 守卫，fill_null 救不了 inf
    df = df.with_columns([
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2.0).alias("mid"),
        pl.when(pl.col("ask_volume_1") + pl.col("bid_volume_1") > 0)
          .then((pl.col("ask_price_1") * pl.col("bid_volume_1")
                 + pl.col("bid_price_1") * pl.col("ask_volume_1"))
                / (pl.col("ask_volume_1") + pl.col("bid_volume_1")))
          .otherwise(None).alias("micro"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread"),
        (pl.col("ask_volume_1") + pl.col("bid_volume_1")
         + pl.col("ask_volume_2") + pl.col("bid_volume_2")).alias("depth"),
        pl.col("transaction_volume").fill_null(0).alias("txn_vol"),
        pl.col("transaction_avgprice").fill_null(0.0).alias("txn_px"),
    ]).with_columns(
        pl.col("micro").fill_null(pl.col("mid")),  # 双侧量全 0 → 微价退化为 mid
        (pl.col("txn_vol") > 0).cast(pl.Float32).alias("has_txn"),
    )

    # —— per-sample 统计（U_s = 有效行正 spread 中位数 ≈ 1 tick）——
    g = df.group_by("sample_id").agg([
        pl.col("spread").filter(pl.col("_valid") & (pl.col("spread") > 0))
          .median().alias("U_s"),
        pl.col("depth").median().alias("med_depth"),
        pl.col("txn_vol").mean().alias("mean_vol"),
        pl.col("mid").first().alias("mid0"),
        pl.len().alias("n_valid"),
    ])
    df = df.join(g, on="sample_id")

    # —— 14 通道（tick 单位/比值化，见 seq_common.FEAT_NAMES）——
    U_s = pl.col("U_s")
    den_imb = pl.col("ask_volume_1") + pl.col("bid_volume_1")
    den_imb2 = (pl.col("ask_volume_1") + pl.col("bid_volume_1")
                + pl.col("ask_volume_2") + pl.col("bid_volume_2"))
    df = df.with_columns([
        pl.col("U_s").fill_null(TICK_FALLBACK[split]),
        pl.col("med_depth").fill_null(1.0),
        pl.col("mean_vol").fill_null(0.0),
    ]).with_columns([
        ((pl.col("mid") - pl.col("mid0")) / U_s).clip(-256.0, 256.0).alias("mid_dev"),
        (pl.col("mid").diff().over("sample_id").fill_null(0.0) / U_s).alias("dmid"),
        (pl.col("spread") / U_s).alias("spread_n"),
        pl.when(den_imb > 0).then((pl.col("bid_volume_1") - pl.col("ask_volume_1"))
                                  / den_imb).otherwise(0.0).alias("book_imb"),
        ((pl.col("micro") - pl.col("mid")) / U_s).alias("micro_off"),
        pl.when(den_imb2 > 0).then(
            (pl.col("bid_volume_1") + pl.col("bid_volume_2")
             - pl.col("ask_volume_1") - pl.col("ask_volume_2")) / den_imb2
        ).otherwise(0.0).alias("d_imb2"),
        pl.when(pl.col("med_depth") > 0)
          .then((pl.col("depth") - pl.col("med_depth")) / pl.col("med_depth"))
          .otherwise(0.0).alias("depth_rel"),
        (((pl.col("txn_px") - pl.col("mid")).sign() * pl.col("txn_vol")).arcsinh()).alias("t_svol"),
        (pl.col("txn_vol") / (pl.col("mean_vol") + 0.5)).alias("t_vol_rel"),
        pl.when(pl.col("has_txn") > 0)
          .then((pl.col("txn_px") - pl.col("mid")) / U_s).otherwise(0.0).alias("dpx"),
        pl.col("has_txn"),
        (pl.col("txn_vol") / (pl.col("ask_volume_1") + pl.col("bid_volume_1") + 1.0))
          .arcsinh().alias("tvol_dep"),
        (~pl.col("_valid")).cast(pl.Float32).alias("eside"),
        ((600.0 - pl.col("seconds_before_predict")) / 600.0).clip(0.0, 1.0).alias("rel_t"),
        # dt：距上一次观测的秒数 / 30（每样本首行 = 1.0；pad 区 = 0，被 masked pool 排除）。间隔不规则（中位 3.0s、q99 12s），
        # 差分/波动率类通道要靠它把"跨了多久"与"动了多少"分开。per-sample、零拟合。
        ((-pl.col("seconds_before_predict").diff().over("sample_id"))
         .clip(0.0, 30.0) / 30.0).fill_null(1.0).alias("dt"),
    ])

    # —— numpy 定长化装配（scatter 一次写入）——
    df = df.with_columns(pl.int_range(pl.len()).over("sample_id").alias("_pos"))
    narrow = df.select(FEAT_NAMES + ["sample_id", "_pos"])
    del df  # 释放派生列，防内存爬升
    arr = narrow.select(FEAT_NAMES).to_numpy().astype(np.float32)
    if not np.isfinite(arr).all():
        for i, f in enumerate(FEAT_NAMES):
            n_bad = int((~np.isfinite(arr[:, i])).sum())
            if n_bad:
                print(f"  {f}: {n_bad} 非有限值")
        raise AssertionError("特征含 NaN/inf！")
    sids = narrow["sample_id"].to_numpy()
    pos = narrow["_pos"].to_numpy()
    del narrow
    uniq = np.unique(sids)                        # 已排序
    assert uniq.shape[0] == n_block
    block_row = np.searchsorted(uniq, sids)
    X_block = np.zeros((n_block, T, F), dtype=np.float16)
    flat = (block_row.astype(np.int64) * T + pos) * F
    for f in range(F):
        X_block.ravel()[flat + f] = arr[:, f].astype(np.float16)
    assert np.isfinite(X_block.astype(np.float32)).all(), "f16 写入产生非有限值"

    meta = g.select(["sample_id", "n_valid", "U_s", "mid0", "med_depth"]).with_columns(
        pl.col("n_valid").cast(pl.Int16),
        pl.col("U_s").fill_null(TICK_FALLBACK[split]),
    ).sort("sample_id")
    return X_block, meta, check


def iter_chunks(split, months_arg):
    """产出 (chunk_id, lo, hi)。train 按月；test 按 5 万样本切块。"""
    if split == "train":
        lab = pl.read_parquet(os.path.join(BASE, r"train/label.parquet"))
        if months_arg is None:
            months = sorted(lab["month"].unique().to_list())
        else:
            months = [int(x) for x in months_arg.split(",")]
        for m in months:
            lo = int(lab.filter(pl.col("month") == m)["sample_id"].min())
            hi = int(lab.filter(pl.col("month") == m)["sample_id"].max())
            yield f"m{m}", lo, hi
    else:
        sub = pl.read_csv(os.path.join(BASE, r"submissions/submission.csv"),
                          columns=["sample_id"])
        lo_all, hi_all = int(sub["sample_id"].min()), int(sub["sample_id"].max())
        lo = lo_all
        while lo <= hi_all:
            hi = min(lo + 49_999, hi_all)
            yield f"c{lo}", lo, hi
            lo = hi + 1


def build_split(split, months_arg, resume, finalize, to_memory=False):
    """to_memory=True：整份特征建到**内存数组**返回，不写 `.dat`、不写 progress、不调 `_finalize`。

    用途是**提交时现算 test 特征**：test 缓存只在生成提交时用（一天几次），
    缓存的价值来自"算一次、复用多次"，在 test 侧不成立（train 侧成立——每次实验都要读，
    复用几百次，所以 train 仍然走磁盘缓存）。删掉三份 test `.dat` 回收 6.1GB。

    ⚠️ 索引 parquet（`{split}_snap_index.parquet`，含 U_s/mid0/n_valid）**不重建**，
    沿用磁盘上已有的那份——细段的构建要读它，而它只有几 MB。
    **删 `.dat` 时不要连索引一起删。**（缺失时 load_ref 会回退 tick 常数并打警告）
    """
    os.makedirs(CACHE, exist_ok=True)
    if finalize:
        _finalize(split)
        return
    dat_path = os.path.join(CACHE, f"{split}_snap_X.dat")
    meta_dir = os.path.join(CACHE, "meta_parts")
    os.makedirs(meta_dir, exist_ok=True)
    prog_path = os.path.join(CACHE, f"progress_{split}.parquet")
    done = set()
    if resume and not to_memory and os.path.exists(prog_path):
        done = set(pl.read_parquet(prog_path)["chunk"].to_list())
        print(f"[resume] 已完成 {len(done)} 块")

    N = SPLIT_N[split]
    if to_memory:
        X = np.zeros((N, T, F), dtype=np.float16)   # 常驻内存，不落盘
        print(f"[{split}] 内存模式：{X.nbytes/2**30:.2f}GB 常驻，不写 .dat")
    elif os.path.exists(dat_path):
        X = np.lib.format.open_memmap(dat_path, mode="r+", dtype=np.float16)
        assert X.shape == (N, T, F)
    else:
        X = np.lib.format.open_memmap(dat_path, mode="w+", dtype=np.float16,
                                      shape=(N, T, F))

    # —— 行定位：确定性计算（不依赖构建顺序，续跑安全）——
    chunks = list(iter_chunks(split, months_arg))
    if split == "train":
        lab = pl.read_parquet(os.path.join(BASE, r"train/label.parquet"))
        counts = lab.group_by("month").len().sort("month")["len"].to_numpy()
        cid2start = {f"m{m}": int(counts[:m].sum()) for m in range(len(counts))}
    else:
        lo_all = int(chunks[0][1])
        cid2start = {cid: lo - lo_all for cid, lo, _ in chunks}  # test id 连续已验证

    t_all = time.time()
    os.environ.setdefault("POLARS_MAX_THREADS", "4")  # 子进程继承，限排序/解压内存
    py = os.path.abspath(__file__)
    py_exe = sys.executable  # 父进程自己就是 pytorch_env 的 python
    tmp_dir = os.path.join(CACHE, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    for cid, lo, hi in chunks:
        if cid in done:
            print(f"[{split}] {cid} 已跳过")
            continue
        t0 = time.time()
        # 子进程跑单块（内存随进程退出释放，防 polars 分配器跨月累计 OOM）。
        # 子进程峰值（2026-09-14 实测 RSS）：train 单月 2.4GB、test 单块 6.4GB —— 来自
        # parquet 行组整体解压（100 万行/组），与块内样本数不成正比（test 块样本多但峰值
        # 反而更高）。系统内存紧张时可能被杀 → 启动前等内存水位（空闲 ≥7GB，按 test 峰值
        # 留余量），失败重试 3 次（间隔 20s）。
        # ⚠️ 必须用 avail_mem_bytes()（读 cgroup），不能用 psutil——容器限额 72GB 时
        # psutil 报的是宿主机的 629GB，守卫永远不会触发。见 seq_common.avail_mem_bytes。
        waited = 0
        while avail_mem_bytes() < 7e9 and waited < 1800:
            print(f"[{split}] 容器可用内存 {avail_mem_bytes()/1e9:.1f}GB < 7GB，"
                  f"等待 30s…", flush=True)
            time.sleep(30)
            waited += 30
        print(f"[{split}] 容器可用内存 {avail_mem_bytes()/1e9:.1f}GB，启动 {cid}", flush=True)
        tag = f"{split}_{cid}"
        tmp_block = os.path.join(tmp_dir, f"{tag}_block.npy")
        tmp_meta = os.path.join(tmp_dir, f"{tag}_meta.parquet")
        ret = None
        for attempt in range(3):
            ret = subprocess.run(
                [py_exe, py, "--split", split, "--worker", str(lo), str(hi),
                 "--tmp-block", tmp_block, "--tmp-meta", tmp_meta],
                capture_output=True, text=True, timeout=900)
            if ret.returncode == 0:
                break
            print(f"[{split}] {cid} 子进程失败（第 {attempt + 1} 次）："
                  f"{ret.stderr[-300:]}，20s 后重试", flush=True)
            time.sleep(20)
        if ret.returncode != 0:
            print(f"[{split}] {cid} 重试 3 次仍失败：\n{ret.stdout[-2000:]}\n{ret.stderr[-2000:]}")
            raise RuntimeError(f"worker {cid} failed")
        block = np.load(tmp_block, mmap_mode="r")
        meta = pl.read_parquet(tmp_meta)
        n_blk = block.shape[0]
        assert meta.height == n_blk, f"{cid} meta 与 block 行数不符"
        rs = cid2start[cid]
        assert rs + n_blk <= N, f"{cid} 行越界"
        X[rs:rs + n_blk] = block
        if not to_memory:          # 内存模式不落盘、无断点（一次跑完，失败重来）
            X.flush()
            meta.write_parquet(os.path.join(meta_dir, f"{split}_{cid}.parquet"))
            new_prog = pl.DataFrame({"chunk": [cid], "n_samples": [n_blk],
                                     "done_at": [time.strftime("%H:%M:%S")]})
            if os.path.exists(prog_path):
                old_prog = pl.read_parquet(prog_path)
                new_prog = pl.concat([old_prog, new_prog])
            new_prog.write_parquet(prog_path)
        try:
            mm = block._mmap  # 关闭 mmap 句柄，Windows 才能删文件
            block = None
            del mm
        except Exception:
            pass
        for f in (tmp_block, tmp_meta):
            try:
                os.remove(f)
            except OSError:
                pass  # 残留 tmp 文件不影响续跑
        done.add(cid)
        print(f"[{split}] {cid} samples={n_blk} "
              f"{time.time()-t0:.0f}s | {len(done)}/{len(chunks)}", flush=True)
    print(f"[{split}] 构建完成，总耗时 {(time.time()-t_all)/60:.1f}min")
    if to_memory:
        print(f"[{split}] 内存构建完成（{X.nbytes/2**30:.2f}GB，未落盘）；"
              f"索引沿用磁盘上已有的 {split}_snap_index.parquet")
        return X
    if len(done) >= len(chunks):
        _finalize(split)
    else:
        print(f"[{split}] 部分构建（{len(done)}/{len(chunks)} 块），"
              f"全部完成后用 --finalize 合并 index")
    return X


def _finalize(split):
    meta_dir = os.path.join(CACHE, "meta_parts")
    parts = []
    names = [f for f in os.listdir(meta_dir) if f.startswith(f"{split}_")]
    # 文件名数字排序（m0,m1,m2,...,m10...；字典序会把 m10 排在 m2 前 → 行序错位）
    names.sort(key=lambda f: int("".join(ch for ch in f.split("_")[-1] if ch.isdigit())))
    for f in names:
        parts.append(pl.read_parquet(os.path.join(meta_dir, f)))
    idx = pl.concat(parts)
    N = SPLIT_N[split]
    assert idx.height == N, f"index {idx.height} != 预期 {N}"
    sids = idx["sample_id"].to_numpy()
    assert (np.diff(sids) > 0).all(), "sample_id 非严格递增——行序错位！"
    assert idx["n_valid"].min() >= 1 and idx["n_valid"].max() <= T
    if split == "train":
        lab = pl.read_parquet(os.path.join(BASE, r"train/label.parquet"))
        assert (lab["sample_id"].to_numpy() == sids).all(), "与 label 行序不一致！"
    idx.write_parquet(os.path.join(CACHE, f"{split}_snap_index.parquet"))
    print(f"[finalize {split}] index {idx.height} 行 OK，"
          f"n_valid min/med/max = {idx['n_valid'].min()}/{int(idx['n_valid'].median())}/{idx['n_valid'].max()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "test"])
    p.add_argument("--months", default=None, help="逗号分隔月号；默认全部")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--memory", action="store_true",
                   help="建到内存返回、不落盘（提交时现算 test 特征用）。"
                        "⚠️ 不带本开关 = 落盘：`--split test` 会重建已删的 test_snap_X.dat（4.35GB）")
    p.add_argument("--worker", type=int, nargs=2, default=None,
                   help="子进程模式：只构建 [lo,hi] 一个块，写入 --tmp-block/--tmp-meta")
    p.add_argument("--tmp-block", default=None)
    p.add_argument("--tmp-meta", default=None)
    args = p.parse_args()

    if args.worker is not None:
        lo, hi = args.worker
        os.makedirs(os.path.dirname(args.tmp_block), exist_ok=True)
        scan = get_market_scan(args.split)
        block, meta, check = build_month(scan, lo, hi, args.split)
        np.save(args.tmp_block, block)
        meta.write_parquet(args.tmp_meta)
        print(f"worker done: samples={block.shape[0]} "
              f"sec_first_med={check['sec_first'].median():.0f}")
        sys.exit(0)

    build_split(args.split, args.months, args.resume, args.finalize,
                to_memory=args.memory)
