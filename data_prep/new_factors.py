"""new_factors.py — P0 新因子批（2026-09-07 公开 notebook 侦察后，全部自己实现）。

思路出处与公式定义见 reference/factor_comparison.md §1；本文件只借鉴概念，
实现与命名走我们自己的约定（沿用 factors.py 的符号/窗口惯例）。

因子清单（35 个）：
L2 盘口档位族（tj-transformer / ysx 思路，我们 L2 列首次建模）:
  rel_spread2_0    预测时刻快照 (ask2-bid2)/mid
  imb2_0           预测时刻快照 (bv1+bv2-av1-av2)/(bv1+bv2+av1+av2)
  slope_0          预测时刻快照 ((a2-a1)+(b1-b2))/mid（1→2 档间距）
  ask_slope_mean60 60s 内每快照 (a2-a1)/(av2-av1) 均值（av2==av1 → 该行 null）
  bid_slope_mean60 60s 内每快照 (b1-b2)/(bv1-bv2) 均值
  slope_imb60      bid_slope_mean60 - ask_slope_mean60
秒级静默结构（ysx 思路，61 秒网格 0..60）:
  t_has_data_15    ≤15s 有成交的秒数（floor(sec) 去重计数）
  t_miss_max       0..60 网格最长连续无成交段（max(首个空段, 60-最后秒, max diff-1)；无事件=61）
  o_has_data_15 / o_miss_max   订单流同款
near/far 半窗（ysx 思路：窗口 w 内再分近半 sec≤w/2 与远半 w/2<sec≤w）:
  t_price_nf_{15,30,60}  近半价均值/远半价均值 - 1
  t_vol_nf_{30,60}       近半量/远半量
  t_rc_nf_{15,30,60}     近半笔数/远半笔数
  o_vol_nf_{15,30,60} / o_rc_nf_{15,30,60} / o_cancel_nf_{15,30,60}   订单流同款
  （o_cancel = order_action==1，沿用 DATA.md 假设 0=挂/1=撤）
跨流方向差（ysx 思路）:
  x_to_buy_diff         t_buy_ratio - o_buy_ratio（全窗）
  x_to_buy_diff_15      ≤15s 版
大单计数（ysx 思路；train 阈值=按月全量分位，test=全 test 集一次分位）:
  t_large_buy           (side==0 & vol>q90) 笔数
  t_large_sell          (side==1 & vol>q95) 笔数
  x_large_imb           (t_large_buy - t_large_sell)/(t_n_15+1.0)
RV 波动率（Rib/ysx 思路；与 mid_vol std 版互补）:
  m_rv60 / m_rv600      sqrt(Σ dmid²)，60s / 全 600s（dmid=相邻快照 mid 差，边界填 0）
  x_rv_60_600           m_rv60 / m_rv600

用法：
  python new_factors.py                            # train 全量 71 月（断点续跑）
  python new_factors.py 0 66                       # 只算指定 train 月（调试）
  python new_factors.py --split test               # test 全量（5 万样本/块）
  python new_factors.py --split test --only-chunks 0,1    # 只算指定块（调试）
  （子进程 worker 模式为内部实现，见 main）

工程约定（沿用 snap_feats.py 教训）：
- 子进程逐块 + 内存门控（空闲 <7GB 等待）→ 防 polars 分配器跨块累计 OOM
- 数据源优先 *_*.parquet（行组下推），回退 feather
- 断点续跑：输出文件已存在即跳过
"""
import argparse
import os
import subprocess
import sys
import time

import polars as pl

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
OUT = os.path.join(BASE, "factors")

NF_WINS = [15, 30, 60]          # near/far 全窗
TEST_CHUNK = 50_000             # test 分块大小（样本数）


def get_scan(split, name):
    """优先 parquet（行组下推），回退 feather。"""
    pq = os.path.join(BASE, split, f"{name}.parquet")
    if os.path.exists(pq):
        return pl.scan_parquet(pq)
    return pl.scan_ipc(os.path.join(BASE, split, f"{name}.feather"))


def sec_grid(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """(sample, sec) 去重行 → has_data_15 + miss_max。"""
    sdf = df.group_by(["sample_id", "sec"]).len()
    g = sdf.group_by("sample_id").agg(pl.col("sec").sort().alias("_secs"))
    g = g.with_columns([
        pl.col("_secs").list.filter(pl.element() <= 15).list.len()
          .alias(f"{prefix}_has_data_15"),
    ]).with_columns([
        pl.col("_secs").list.first().alias("_s0"),
        pl.col("_secs").list.last().alias("_s1"),
        pl.col("_secs").list.diff(1, null_behavior="drop").list.max().alias("_sd"),
    ]).with_columns([
        pl.when(pl.col("_secs").list.len() == 0)
          .then(61)
          .otherwise(pl.max_horizontal([
              pl.col("_s0").fill_null(61),
              (60 - pl.col("_s1")).fill_null(61),
              (pl.col("_sd") - 1).fill_null(0),
          ])).alias(f"{prefix}_miss_max"),
    ]).drop(["_secs", "_s0", "_s1", "_sd"])
    return g


def near_far(df: pl.DataFrame, prefix: str, extra: set) -> dict[str, pl.DataFrame]:
    """半窗聚合；extra ⊆ {"price","vol","rc","cancel"}。"""
    parts = {}
    for w in NF_WINS:
        sub = df.filter(pl.col("sec") <= w)
        h = float(w) / 2.0
        near = pl.col("sec") <= h
        aggs = []
        if "price" in extra:
            aggs += [
                pl.col("price").filter(near).mean().alias("_pn"),
                pl.col("price").filter(~near).mean().alias("_pf"),
            ]
        if "vol" in extra:
            aggs += [
                pl.col("volume").filter(near).sum().alias("_vn"),
                pl.col("volume").filter(~near).sum().alias("_vf"),
            ]
        if "rc" in extra:
            aggs += [
                pl.col("sec").filter(near).len().alias("_rn"),
                pl.col("sec").filter(~near).len().alias("_rf"),
            ]
        if "cancel" in extra:
            aggs += [
                pl.col("is_cancel").filter(near).cast(pl.Int32).sum().alias("_cn"),
                pl.col("is_cancel").filter(~near).cast(pl.Int32).sum().alias("_cf"),
            ]
        g = sub.group_by("sample_id").agg(aggs)
        cols = []
        if "price" in extra:
            g = g.with_columns((pl.col("_pn") / pl.col("_pf") - 1.0).alias(f"{prefix}_price_nf_{w}"))
            cols.append(f"{prefix}_price_nf_{w}")
        if "vol" in extra:
            g = g.with_columns((pl.col("_vn") / pl.col("_vf")).alias(f"{prefix}_vol_nf_{w}"))
            cols.append(f"{prefix}_vol_nf_{w}")
        if "rc" in extra:
            g = g.with_columns((pl.col("_rn") / pl.col("_rf")).alias(f"{prefix}_rc_nf_{w}"))
            cols.append(f"{prefix}_rc_nf_{w}")
        if "cancel" in extra:
            g = g.with_columns((pl.col("_cn") / pl.col("_cf")).alias(f"{prefix}_cancel_nf_{w}"))
            cols.append(f"{prefix}_cancel_nf_{w}")
        parts[f"{prefix}_nf_{w}"] = g.select(["sample_id"] + cols)
    return parts


# ---------------------------------------------------------------------------
# 因子计算（一块样本：train=一个月，test=5 万样本）
# ---------------------------------------------------------------------------
def compute_block(lo: int, hi: int, split: str, q90=None, q95=None) -> pl.DataFrame:
    mdf = (get_scan(split, "market")
           .filter(pl.col("sample_id").is_between(lo, hi)).collect())
    tdf = (get_scan(split, "transaction")
           .filter(pl.col("sample_id").is_between(lo, hi)).collect())
    odf = (get_scan(split, "order")
           .filter(pl.col("sample_id").is_between(lo, hi)).collect())

    # 统一排序：sample 升序 + sec 升序（idx 0 = 最新，与 factors.py 一致）
    mdf = mdf.sort(["sample_id", "seconds_before_predict"])
    tdf = tdf.sort(["sample_id", "seconds_before_predict"])
    odf = odf.sort(["sample_id", "seconds_before_predict"])

    # ================= L2 盘口档位族（market） =================
    mdf = mdf.with_columns([
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2.0).alias("mid"),
        ((pl.col("ask_price_1") > 0) & (pl.col("bid_price_1") > 0)).alias("_valid"),
    ])
    last = mdf.group_by("sample_id").agg([
        pl.col("ask_price_1").first(), pl.col("bid_price_1").first(),
        pl.col("ask_price_2").first(), pl.col("bid_price_2").first(),
        pl.col("ask_volume_1").first(), pl.col("bid_volume_1").first(),
        pl.col("ask_volume_2").first(), pl.col("bid_volume_2").first(),
        pl.col("mid").first(),
    ])
    l2f = last.with_columns([
        pl.when((pl.col("ask_price_2") > 0) & (pl.col("bid_price_2") > 0) & (pl.col("mid") > 0))
          .then((pl.col("ask_price_2") - pl.col("bid_price_2")) / pl.col("mid"))
          .otherwise(None).alias("rel_spread2_0"),
        pl.when(pl.col("ask_volume_1") + pl.col("bid_volume_1")
                + pl.col("ask_volume_2") + pl.col("bid_volume_2") > 0)
          .then((pl.col("bid_volume_1") + pl.col("bid_volume_2")
                 - pl.col("ask_volume_1") - pl.col("ask_volume_2"))
                / (pl.col("ask_volume_1") + pl.col("bid_volume_1")
                   + pl.col("ask_volume_2") + pl.col("bid_volume_2")))
          .otherwise(None).alias("imb2_0"),
        pl.when((pl.col("ask_price_2") > 0) & (pl.col("bid_price_2") > 0) & (pl.col("mid") > 0))
          .then(((pl.col("ask_price_2") - pl.col("ask_price_1"))
                 + (pl.col("bid_price_1") - pl.col("bid_price_2"))) / pl.col("mid"))
          .otherwise(None).alias("slope_0"),
    ]).select(["sample_id", "rel_spread2_0", "imb2_0", "slope_0"])

    # 60s 内档位斜率均值
    w60 = mdf.filter(pl.col("seconds_before_predict") <= 60).with_columns([
        pl.when(pl.col("ask_volume_2") != pl.col("ask_volume_1"))
          .then((pl.col("ask_price_2") - pl.col("ask_price_1"))
                / (pl.col("ask_volume_2") - pl.col("ask_volume_1")))
          .otherwise(None).alias("_ask_slope"),
        pl.when(pl.col("bid_volume_1") != pl.col("bid_volume_2"))
          .then((pl.col("bid_price_1") - pl.col("bid_price_2"))
                / (pl.col("bid_volume_1") - pl.col("bid_volume_2")))
          .otherwise(None).alias("_bid_slope"),
    ])
    slope_f = w60.group_by("sample_id").agg([
        pl.col("_ask_slope").filter(pl.col("_valid")).mean().alias("ask_slope_mean60"),
        pl.col("_bid_slope").filter(pl.col("_valid")).mean().alias("bid_slope_mean60"),
    ]).with_columns(
        (pl.col("bid_slope_mean60") - pl.col("ask_slope_mean60")).alias("slope_imb60"))

    # ================= RV 波动率（market） =================
    # 空档侧行 mid 置 null → diff 跳过该行（损失 2 个差分，0.55% 行影响可忽略）
    mdf = mdf.with_columns(
        pl.when(pl.col("_valid")).then(pl.col("mid")).otherwise(None).alias("mid_n"))
    rv = mdf.with_columns(
        pl.col("mid_n").diff().over("sample_id").fill_null(0.0).alias("dmid")
    ).group_by("sample_id").agg([
        (pl.col("dmid").filter(pl.col("seconds_before_predict") <= 60)
         .pow(2).sum().sqrt()).alias("m_rv60"),
        (pl.col("dmid").pow(2).sum().sqrt()).alias("m_rv600"),
    ]).with_columns(
        pl.when(pl.col("m_rv600") > 0)
          .then(pl.col("m_rv60") / pl.col("m_rv600")).otherwise(None).alias("x_rv_60_600"))

    # ================= 成交/订单：秒网格 + near/far + 跨流 + 大单 =================
    tdf = tdf.with_columns(
        pl.col("seconds_before_predict").floor().cast(pl.Int16).alias("sec"),
        (pl.col("side") == 0).alias("is_buy"))
    odf = odf.with_columns(
        pl.col("seconds_before_predict").floor().cast(pl.Int16).alias("sec"),
        (pl.col("side") == 0).alias("is_buy"),
        (pl.col("order_action") == 1).alias("is_cancel"))

    # 大单阈值：默认块内全量分位（train 月 = 按月；test 由父进程传入全 test 分位）
    if q90 is None:
        q90 = float(tdf["volume"].quantile(0.90))
        q95 = float(tdf["volume"].quantile(0.95))
    else:
        # ⚠️ 缺 `q95 is None` 守卫：q90/q95 必须**成对**传入，单独传 --q90 会 float(None) → TypeError。
        #    现有调用（本文件 :405）都成对传，故一直没暴露。
        q95 = float(q95)

    tg = tdf.group_by("sample_id").agg([
        pl.col("is_buy").cast(pl.Float32).mean().alias("t_buy_ratio"),
        pl.col("is_buy").filter(pl.col("sec") <= 15).cast(pl.Float32).mean()
          .alias("t_buy_ratio_15"),
        pl.col("sec").filter(pl.col("sec") <= 15).len().alias("t_n_15"),
        (pl.col("is_buy") & (pl.col("volume") > q90)).cast(pl.Int32).sum().alias("t_large_buy"),
        ((~pl.col("is_buy")) & (pl.col("volume") > q95)).cast(pl.Int32).sum().alias("t_large_sell"),
    ]).with_columns(
        ((pl.col("t_large_buy") - pl.col("t_large_sell")) / (pl.col("t_n_15") + 1.0))
        .alias("x_large_imb"))

    og = odf.group_by("sample_id").agg([
        pl.col("is_buy").cast(pl.Float32).mean().alias("o_buy_ratio"),
        pl.col("is_buy").filter(pl.col("sec") <= 15).cast(pl.Float32).mean()
          .alias("o_buy_ratio_15"),
    ])

    t_grid = sec_grid(tdf, "t")
    o_grid = sec_grid(odf, "o")

    t_nf = near_far(tdf, "t", {"price", "vol", "rc"})
    o_nf = near_far(odf, "o", {"vol", "rc", "cancel"})

    # ================= 合并 =================
    df = l2f
    for part in [slope_f, rv, tg, og, t_grid, o_grid] + list(t_nf.values()) + list(o_nf.values()):
        df = df.join(part, on="sample_id", how="left")

    df = df.with_columns([
        (pl.col("t_buy_ratio") - pl.col("o_buy_ratio")).alias("x_to_buy_diff"),
        (pl.col("t_buy_ratio_15") - pl.col("o_buy_ratio_15")).alias("x_to_buy_diff_15"),
    ])

    # 静默结构缺失样本填充（无事件 → 0 个有数据秒、61 秒全缺）
    df = df.with_columns([
        pl.col("t_has_data_15").fill_null(0), pl.col("t_miss_max").fill_null(61),
        pl.col("o_has_data_15").fill_null(0), pl.col("o_miss_max").fill_null(61),
    ])

    # 安全网：任何 ±inf/NaN → null（0 除等；与 factors.py 末尾同款）
    df = df.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
        for c in df.columns
        if c != "sample_id" and isinstance(df.schema[c], (pl.Float32, pl.Float64))
    ])
    return df


def compute_month(m: int) -> pl.DataFrame:
    """train 单月：compute_block + label join。"""
    lab = pl.read_parquet(os.path.join(BASE, r"train/label.parquet"))
    lm = lab.filter(pl.col("month") == m)
    lo = int(lm["sample_id"].min())
    hi = int(lm["sample_id"].max())
    df = compute_block(lo, hi, "train")
    df = df.join(lm.select(["sample_id", "target", "month"]), on="sample_id", how="left")
    return df


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_worker(m: int, out_path: str) -> None:
    df = compute_month(m)
    df.write_parquet(out_path)
    null_frac = {c: round(float(df[c].null_count() / df.height), 3)
                 for c in df.columns if c not in ("sample_id", "target", "month")
                 and df[c].null_count() > 0.2 * df.height}
    print(f"[m{m}] saved {df.height} rows -> {out_path}；高缺失列: {null_frac}", flush=True)


def run_test_worker(lo: int, hi: int, q90: float, q95: float, out_path: str) -> None:
    df = compute_block(lo, hi, "test", q90, q95)
    df.write_parquet(out_path)
    print(f"[test {lo}-{hi}] saved {df.height} rows -> {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("months", nargs="*", type=int, default=None)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--only-chunks", default=None, help="test 模式：逗号分隔块号（调试）")
    p.add_argument("--worker", type=int, default=None, help="子进程模式：train 单月")
    p.add_argument("--worker-test", type=int, nargs=2, default=None, help="子进程模式：test [lo hi]")
    p.add_argument("--q90", type=float, default=None)
    p.add_argument("--q95", type=float, default=None)
    p.add_argument("--out", default=None, help="worker 输出路径")
    args = p.parse_args()

    os.environ.setdefault("POLARS_MAX_THREADS", "4")

    if args.worker is not None:
        run_worker(args.worker, args.out)
        return
    if args.worker_test is not None:
        run_test_worker(args.worker_test[0], args.worker_test[1], args.q90, args.q95, args.out)
        return

    import psutil
    py = os.path.abspath(__file__)
    py_exe = sys.executable
    t_all = time.time()

    def wait_mem():
        waited = 0
        while psutil.virtual_memory().available < 7e9 and waited < 1800:
            print(f"空闲内存 {psutil.virtual_memory().available/1e9:.1f}GB < 7GB，等待 30s…", flush=True)
            time.sleep(30)
            waited += 30

    def run_sub(args_list):
        for attempt in range(3):
            ret = subprocess.run([py_exe, py] + args_list,
                                 capture_output=True, text=True, timeout=3600,
                                 encoding="utf-8", errors="replace")
            if ret.returncode == 0:
                return ret
            print(f"子进程失败（第 {attempt + 1} 次）：{ret.stderr[-300:]}，20s 后重试", flush=True)
            time.sleep(20)
        raise RuntimeError(f"worker failed: {args_list}\n{ret.stdout[-2000:]}\n{ret.stderr[-2000:]}")

    if args.split == "train":
        lab = pl.read_parquet(os.path.join(BASE, r"train/label.parquet"))
        months = args.months if args.months else sorted(lab["month"].unique().to_list())
        todo = []
        for m in months:
            out = os.path.join(OUT, f"new_m{m}.parquet")
            if os.path.exists(out):
                print(f"[m{m}] 已存在，跳过")
                continue
            todo.append(m)
        for i, m in enumerate(todo):
            wait_mem()
            out = os.path.join(OUT, f"new_m{m}.parquet")
            t0 = time.time()
            run_sub(["--worker", str(m), "--out", out])
            print(f"[m{m}] 完成 {time.time()-t0:.0f}s | {i+1}/{len(todo)} | 累计 {(time.time()-t_all)/60:.1f}min", flush=True)
    else:
        # test：全 test 集大单阈值（一次分位，跨块一致 = 单一 regime）
        print("计算 test 全量 q90/q95 大单阈值…", flush=True)
        tvol = get_scan("test", "transaction").select("volume").collect()["volume"]
        q90 = float(tvol.quantile(0.90))
        q95 = float(tvol.quantile(0.95))
        print(f"q90={q90:.4g} q95={q95:.4g}", flush=True)
        del tvol

        sub = pl.read_csv(os.path.join(BASE, r"submissions/submission.csv"),
                          columns=["sample_id"])
        lo_all, hi_all = int(sub["sample_id"].min()), int(sub["sample_id"].max())
        chunks = []
        lo = lo_all
        ci = 0
        while lo <= hi_all:
            hi = min(lo + TEST_CHUNK - 1, hi_all)
            chunks.append((ci, lo, hi))
            ci += 1
            lo = hi + 1
        if args.only_chunks:
            keep = {int(x) for x in args.only_chunks.split(",")}
            chunks = [c for c in chunks if c[0] in keep]
        for ci, lo, hi in chunks:
            out = os.path.join(OUT, f"new_test_c{ci}.parquet")
            if os.path.exists(out):
                print(f"[test c{ci}] 已存在，跳过")
                continue
            wait_mem()
            t0 = time.time()
            run_sub(["--worker-test", str(lo), str(hi),
                     "--q90", repr(q90), "--q95", repr(q95), "--out", out])
            print(f"[test c{ci}] 完成 {time.time()-t0:.0f}s | {ci+1}/{len(chunks)} | 累计 {(time.time()-t_all)/60:.1f}min", flush=True)
    print(f"全部完成，总耗时 {(time.time()-t_all)/60:.1f}min")


if __name__ == "__main__":
    main()
