"""check_snap_gaps.py — 量 market 快照的时间连续性（train/test），为「1s 网格拼表」方案提供依据。

只读 train|test/market.parquet 的两列（sample_id, seconds_before_predict），回答：
1. 相邻快照间隔的分布：正常节奏 ~2.9s，长尾空洞占多少
2. 每样本的时间结构：行数、跨度、最大空洞、尾部空洞（最后一次观测距 predict 的秒数）
3. 最近 60s（flow 表所在区间）内的观测密度
4. train 的逐月漂移（gap 结构是否随月份变化）

用法：
  python data_prep/check_snap_gaps.py            # train + test
  python data_prep/check_snap_gaps.py train      # 只跑 train

工程约定：先 collect 两列再转 numpy，用全局 diff + 边界掩码代替 polars over() ——
221M 行的窗口函数既慢又吃内存，全局 diff 是本机的常数级替代。
"""
import os
import sys
import time

import numpy as np
import polars as pl

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)

NOMINAL = 3.0          # 标称快照间隔（实测中位数 3.001s，q90=3.09-3.13s）
WINDOW = 600.0         # 标称窗口
TAIL = 60.0            # flow 表覆盖的最近 60s


def rz(name, s):
    print(f"\n--- {name} ---\n{s}", flush=True)


def pct(x, q):
    return float(np.quantile(x, q))


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def analyse(split, months_lab=None):
    pq = os.path.join(BASE, split, "market.parquet")
    t0 = time.time()
    df = pl.scan_parquet(pq).select(["sample_id", "seconds_before_predict"]).collect()
    sid = df["sample_id"].to_numpy()
    sec = df["seconds_before_predict"].to_numpy()
    del df
    print(f"\n=== {split} === 加载 {sec.size:,} 行 / {time.time()-t0:.0f}s", flush=True)

    assert np.all(np.diff(sid) >= 0), "sample_id 非全局非降——文件顺序假设不成立"
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    n_smp = starts.size
    cnt = np.diff(np.r_[starts, sid.size]).astype(np.int64)
    print(f"样本数 {n_smp:,}", flush=True)

    # ---- 区间：组内相邻观测的秒差（文件内秒数递减 → 取负）----
    d = -np.diff(sec.astype(np.float64))
    same = sid[1:] == sid[:-1]
    d_in = d[same]
    n_bad = int((d_in < -1e-6).sum())          # 组内秒数不递减 = 顺序异常
    n_dup = int((np.abs(d_in) < 1e-9).sum())   # 同一时间戳重复
    rz("区间分布（秒，组内相邻观测）", "\n".join([
        f"n={d_in.size:,}  mean={fmt(d_in.mean())}  "
        f"q50={fmt(pct(d_in,0.5))} q90={fmt(pct(d_in,0.9))} "
        f"q99={fmt(pct(d_in,0.99))} q999={fmt(pct(d_in,0.999))} max={fmt(d_in.max())}",
        f"非递减异常 {n_bad:,}  重复时间戳 {n_dup:,} ({100*n_dup/d_in.size:.3f}%)",
        "  分桶： " + "  ".join(
            f"[{lo}-{hi})={100*np.mean((d_in>=lo)&(d_in<hi)):.2f}%"
            for lo, hi in [(0, 3.5), (3.5, 4.5), (4.5, 6), (6, 10), (10, 30), (30, 60), (60, 1e9)]),
        f"  >4.5s 占 {100*np.mean(d_in>4.5):.2f}%   >6s 占 {100*np.mean(d_in>6):.2f}%",
    ]))

    # ---- 每样本聚合（reduceat：组内求和/取最大）----
    dp = np.zeros(sid.size, dtype=np.float64)
    dp[:-1] = np.where(same, d, 0.0)          # 跨样本边界置 0（每个组末尾一格不计）
    gap_sum = np.add.reduceat(dp, starts)
    gap_max = np.maximum.reduceat(dp, starts)
    excess_sum = np.add.reduceat(np.maximum(dp - NOMINAL, 0.0), starts)
    sec_first = sec[starts].astype(np.float64)
    sec_last = sec[np.r_[starts[1:] - 1, sid.size - 1]].astype(np.float64)
    span = sec_first - sec_last
    n_tail = np.add.reduceat((sec <= TAIL).astype(np.float64), starts)
    # 最近 60s 内的最大空洞（细段设计用：flow 表只覆盖这一区间）
    dp_tail = np.zeros(sid.size, dtype=np.float64)
    dp_tail[:-1] = np.where(same & (sec[:-1] <= TAIL), d, 0.0)
    tail_gap_max = np.maximum.reduceat(dp_tail, starts)

    rz("每样本行数", "\n".join([
        f"mean={fmt(cnt.mean(),1)} median={fmt(np.median(cnt),1)} p1={fmt(pct(cnt,0.01),1)} "
        f"min={cnt.min()} max={cnt.max()}",
        f"行数 <100 的样本 {100*np.mean(cnt<100):.2f}%   <50 {100*np.mean(cnt<50):.2f}%   "
        f"<10 {100*np.mean(cnt<10):.2f}%",
        f"最大行数 212 占比（无空洞样本）{100*np.mean(cnt>=207):.2f}%",
    ]))

    rz("每样本最大空洞（秒）", "\n".join([
        f"q50={fmt(pct(gap_max,0.5))} q90={fmt(pct(gap_max,0.9))} "
        f"q99={fmt(pct(gap_max,0.99))} max={fmt(gap_max.max())}",
        f"含 >4.5s 空洞的样本 {100*np.mean(gap_max>4.5):.2f}%   >6s {100*np.mean(gap_max>6):.2f}%   "
        f">10s {100*np.mean(gap_max>10):.2f}%   >30s {100*np.mean(gap_max>30):.2f}%   "
        f">60s {100*np.mean(gap_max>60):.2f}%",
        f"空洞总时长 >60s 的样本 {100*np.mean(gap_sum>60):.2f}%   "
        f"超出标称节奏的总缺失 mean={fmt(excess_sum.mean(),1)}s/样本 "
        f"（= {100*excess_sum.mean()/WINDOW:.1f}% 的时间轴）",
    ]))

    rz("窗口两端", "\n".join([
        f"首观测 sec: q01={fmt(pct(sec_first,0.01),1)} q50={fmt(pct(sec_first,0.5),1)} min={fmt(sec_first.min(),1)}",
        f"末观测 sec（尾部空洞）: q50={fmt(pct(sec_last,0.5))} q90={fmt(pct(sec_last,0.9))} "
        f"q99={fmt(pct(sec_last,0.99))} max={fmt(sec_last.max())}",
        f"尾部空洞 >5s {100*np.mean(sec_last>5):.2f}%   >10s {100*np.mean(sec_last>10):.2f}%   "
        f">30s {100*np.mean(sec_last>30):.2f}%   >60s {100*np.mean(sec_last>60):.2f}%",
        f"实际跨度 span: q50={fmt(pct(span,0.5),1)} q01={fmt(pct(span,0.01),1)}",
    ]))

    rz(f"最近 {TAIL:.0f}s 内的观测数（flow 表区间）", "\n".join([
        f"mean={fmt(n_tail.mean(),1)} median={fmt(np.median(n_tail),1)} p1={fmt(pct(n_tail,0.01),1)} min={n_tail.min()} max={n_tail.max()}",
        f"<20 个观测 {100*np.mean(n_tail<20):.2f}%   <10 {100*np.mean(n_tail<10):.2f}%   "
        f"<5 {100*np.mean(n_tail<5):.2f}%",
        f"该区间内最大空洞 q50={fmt(pct(tail_gap_max,0.5))} q90={fmt(pct(tail_gap_max,0.9))} "
        f"max={fmt(tail_gap_max.max())}   >4.5s 占 {100*np.mean(tail_gap_max>4.5):.2f}%   "
        f">10s 占 {100*np.mean(tail_gap_max>10):.2f}%",
    ]))

    if months_lab is not None:
        lab_sid, lab_m = months_lab
        idx = np.searchsorted(lab_sid, sid[starts])
        assert idx.max() < lab_sid.size and (lab_sid[idx] == sid[starts]).all(), \
            "label 与 market 的 sample_id 对不上"
        m = lab_m[idx]
        um = np.unique(m)
        rows = np.bincount(m)
        mean_cnt = np.bincount(m, weights=cnt) / rows
        mean_gapmax = np.bincount(m, weights=gap_max) / rows
        frac_gap10 = np.bincount(m, weights=(gap_max > 10).astype(np.float64)) / rows
        mean_tail = np.bincount(m, weights=sec_last) / rows
        print(f"\n--- 逐月漂移（{um.size} 个月）---", flush=True)
        print("month rows  mean_cnt  mean_max_gap  %samples_gap>10s  mean_tailsec")
        for i in [0, 1, 2, um.size // 2, um.size - 3, um.size - 2, um.size - 1]:
            print(f"  {um[i]:2d}  {rows[i]:6d}   {mean_cnt[i]:7.1f}      {mean_gapmax[i]:6.2f}"
                  f"          {100*frac_gap10[i]:5.2f}%         {mean_tail[i]:5.2f}")
        print(f"  全月 mean_cnt 范围 [{mean_cnt.min():.1f}, {mean_cnt.max():.1f}]  "
              f"mean_max_gap 范围 [{mean_gapmax.min():.2f}, {mean_gapmax.max():.2f}]  "
              f"gap>10s 样本占比范围 [{100*frac_gap10.min():.2f}%, {100*frac_gap10.max():.2f}%]")


def main():
    splits = sys.argv[1:] or ["train", "test"]
    months_lab = None
    if "train" in splits:
        lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).sort("sample_id")
        months_lab = (lab["sample_id"].to_numpy(), lab["month"].to_numpy())
    for s in splits:
        analyse(s, months_lab if s == "train" else None)


if __name__ == "__main__":
    main()
