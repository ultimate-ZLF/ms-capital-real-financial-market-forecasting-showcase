"""new_factors3.py — 第三批因子（加法收官，~38 个原始因子；12 个交互乘积在 lgb 脚本算）。

来源：ysx 事件级 lag 全谱 / yangq369 v10 因子名（平方/分位/窗口比值/EWM）/ 自补档位统计。
输出：factors/new3_m{m}.parquet（train 月）、factors/new3_test_c{c}.parquet（test 块）。

清单（38）：
事件级 lag 谱扩展（8）:
  t_pxdiff14_mean/std, t_pxdiff21_std, t_pxdiff30_std   事件序 gap 14/21/30 价格差分
  t_voldiff14_mean, t_voldiff21_mean                   同上 volume
  o_pxdiff14_std, o_voldiff14_mean                     订单流同款
market 平方/分位/斜率/档位（13）:
  m_dofi2_sum60, m_dofi2_ewm120      dofi² 和（平方 OFI 非线性，v10 ofi2 思路）
  m_imb_med, m_imb_q25, m_sp_q75     分位统计（分布形状）
  m_mid_ret_abs_mean60               mean|dmid| 60s（已实现绝对波动）
  m_mid_slope_180, m_mid_slope_60    mid 斜率（(last−mid_w)/w，v10 mid_slope）
  m_imb_mean_15, x_imb_15_180        短窗 imb 均值 + 长短差（v10 imb_15_180）
  m_txv_std30                        market 表聚合成交列的 30s std（v10）
  m_ask_gap_mean60, m_bid_gap_mean60 60s 内 a2−a1 / b1−b2 均值（档间距，v10 gap 族）
成交流窗口比值/大单（13）:
  t_sv_5, t_sv_60, x_t_sv_5_60      符号量 5s/60s 及比值（v10 sv_5_60）
  t_vol_5, x_t_vol_5_60             量 5s 及比值
  t_n_60, x_tn_15_60                笔数 60s 及比值（v10 tn_15_180 思路）
  t_sd_ewm45                        exp(−sec/45) 加权有向金额量（v10 sd_ewm_45）
  t_vol_max30                        ≤30s 内单秒最大成交量（v10 vol_max）
  t_buy_sell_vol_ratio              买量/卖量（全窗）
  t_large_buy_15, t_large_sell_15, x_large_imb_15   大单 15s 窗（阈值同批 1：train=月分位，test=全 test 分位）
挂单流（4）:
  o_sv_ewm15, o_av_ewm45            exp 加权符号量/行动量（v10 sv_ewm/av_ewm）
  o_add_ratio                       action==0 占比（ysx market_ratio）
  o_vol_max30                       ≤30s 内单秒最大挂单量

用法：python new_factors3.py [months...] / --split test [--only-chunks i]
"""
import argparse
import os
import subprocess
import sys
import time

import polars as pl

from new_factors import get_scan

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
OUT = os.path.join(BASE, "factors")
TEST_CHUNK = 50_000


def compute_block3(lo: int, hi: int, split: str, q90=None, q95=None) -> pl.DataFrame:
    mdf = (get_scan(split, "market")
           .filter(pl.col("sample_id").is_between(lo, hi)).collect())
    tdf = (get_scan(split, "transaction")
           .filter(pl.col("sample_id").is_between(lo, hi)).collect())
    odf = (get_scan(split, "order")
           .filter(pl.col("sample_id").is_between(lo, hi)).collect())

    mdf = mdf.sort(["sample_id", "seconds_before_predict"])
    tdf = tdf.sort(["sample_id", "seconds_before_predict"])
    odf = odf.sort(["sample_id", "seconds_before_predict"])

    # ================= market =================
    mdf = mdf.with_columns([
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2.0).alias("mid"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread"),
        ((pl.col("ask_price_1") > 0) & (pl.col("bid_price_1") > 0)).alias("_valid"),
    ]).with_columns([
        pl.when(pl.col("ask_volume_1") + pl.col("bid_volume_1") > 0)
          .then((pl.col("bid_volume_1") - pl.col("ask_volume_1"))
                / (pl.col("ask_volume_1") + pl.col("bid_volume_1")))
          .otherwise(None).alias("imb"),
        pl.when(pl.col("_valid")).then(pl.col("mid")).otherwise(None).alias("mid_v"),
        (pl.col("ask_price_2") - pl.col("ask_price_1")).alias("ask_gap"),
        (pl.col("bid_price_1") - pl.col("bid_price_2")).alias("bid_gap"),
        pl.col("transaction_volume").fill_null(0).alias("txv"),
    ]).with_columns([
        (pl.col("ask_volume_1") - pl.col("bid_volume_1")).diff().over("sample_id")
          .fill_null(0.0).alias("dofi_d"),
        pl.col("mid_v").diff().over("sample_id").fill_null(0.0).alias("dmid"),
    ])
    sec60 = pl.col("seconds_before_predict") <= 60
    sec15 = pl.col("seconds_before_predict") <= 15
    sec30 = pl.col("seconds_before_predict") <= 30
    sec180 = pl.col("seconds_before_predict") <= 180
    m1 = mdf.group_by("sample_id").agg([
        pl.col("dofi_d").filter(sec60).pow(2).sum().alias("m_dofi2_sum60"),
        ((pl.col("dofi_d").pow(2) * (-pl.col("seconds_before_predict") / 120.0).exp()).sum()
         / (-pl.col("seconds_before_predict") / 120.0).exp().sum()).alias("m_dofi2_ewm120"),
        pl.col("imb").median().alias("m_imb_med"),
        pl.col("imb").quantile(0.25).alias("m_imb_q25"),
        pl.col("spread").filter(pl.col("_valid")).quantile(0.75).alias("m_sp_q75"),
        pl.col("dmid").filter(sec60).abs().mean().alias("m_mid_ret_abs_mean60"),
        pl.col("mid_v").filter(sec180).last().alias("_mid180"),
        pl.col("mid_v").filter(sec60).last().alias("_mid60"),
        pl.col("mid_v").first().alias("_mid0"),
        pl.col("imb").filter(sec15).mean().alias("m_imb_mean_15"),
        pl.col("imb").filter(sec180).mean().alias("_imb180"),
        pl.col("txv").filter(sec30).std().alias("m_txv_std30"),
        pl.col("ask_gap").filter(pl.col("_valid") & (pl.col("ask_price_2") > 0) & sec60)
          .mean().alias("m_ask_gap_mean60"),
        pl.col("bid_gap").filter(pl.col("_valid") & (pl.col("bid_price_2") > 0) & sec60)
          .mean().alias("m_bid_gap_mean60"),
    ]).with_columns([
        ((pl.col("_mid0") - pl.col("_mid180")) / 180.0).alias("m_mid_slope_180"),
        ((pl.col("_mid0") - pl.col("_mid60")) / 60.0).alias("m_mid_slope_60"),
        (pl.col("m_imb_mean_15") - pl.col("_imb180")).alias("x_imb_15_180"),
    ]).drop(["_mid180", "_mid60", "_mid0", "_imb180"])

    # ================= 成交流 =================
    tdf = tdf.with_columns([
        pl.col("seconds_before_predict").floor().cast(pl.Int16).alias("sec"),
        (pl.col("side") == 0).alias("is_buy"),
    ]).with_columns([
        (pl.col("is_buy").cast(pl.Int8) * 2 - 1).alias("sgn"),
    ]).with_columns([
        (pl.col("sgn") * pl.col("volume")).alias("sv"),
        (pl.col("sgn") * pl.col("price") * pl.col("volume")).alias("sd"),
        pl.col("price").diff(14).over("sample_id").alias("dpx14"),
        pl.col("price").diff(21).over("sample_id").alias("dpx21"),
        pl.col("price").diff(30).over("sample_id").alias("dpx30"),
        pl.col("volume").diff(14).over("sample_id").alias("dv14"),
        pl.col("volume").diff(21).over("sample_id").alias("dv21"),
    ])
    if q90 is None:
        q90 = float(tdf["volume"].quantile(0.90))
        q95 = float(tdf["volume"].quantile(0.95))
    else:
        # ⚠️ 缺 `q95 is None` 守卫：q90/q95 必须**成对**传入，单独传 --q90 会 float(None) → TypeError。
        #    现有调用（本文件 :306）都成对传，故一直没暴露。
        q95 = float(q95)
    t1 = tdf.group_by("sample_id").agg([
        pl.col("sv").filter(sec15).sum().alias("t_sv_5"),
        pl.col("sv").sum().alias("t_sv_60"),
        pl.col("volume").filter(sec15).sum().alias("t_vol_5"),
        pl.col("volume").sum().alias("t_vol_60v"),
        pl.col("sec").filter(sec15).len().alias("t_n_15"),
        pl.col("sec").len().alias("t_n_60"),
        (pl.col("sd") * (-pl.col("seconds_before_predict") / 45.0).exp()).sum()
          .alias("t_sd_ewm45"),
        (pl.col("is_buy") & (pl.col("volume") > q90)).cast(pl.Int32).sum().alias("t_large_buy"),
        ((~pl.col("is_buy")) & (pl.col("volume") > q95)).cast(pl.Int32).sum().alias("t_large_sell"),
        (pl.col("is_buy") & (pl.col("volume") > q90) & sec15).cast(pl.Int32).sum()
          .alias("t_large_buy_15"),
        ((~pl.col("is_buy")) & (pl.col("volume") > q95) & sec15).cast(pl.Int32).sum()
          .alias("t_large_sell_15"),
        pl.col("dpx14").mean().alias("t_pxdiff14_mean"),
        pl.col("dpx14").std().alias("t_pxdiff14_std"),
        pl.col("dpx21").std().alias("t_pxdiff21_std"),
        pl.col("dpx30").std().alias("t_pxdiff30_std"),
        pl.col("dv14").mean().alias("t_voldiff14_mean"),
        pl.col("dv21").mean().alias("t_voldiff21_mean"),
    ]).with_columns([
        (pl.col("t_sv_5") / pl.col("t_sv_60")).alias("x_t_sv_5_60"),
        (pl.col("t_vol_5") / pl.col("t_vol_60v")).alias("x_t_vol_5_60"),
        (pl.col("t_n_15") / pl.col("t_n_60")).alias("x_tn_15_60"),
        ((pl.col("t_large_buy_15") - pl.col("t_large_sell_15")) / (pl.col("t_n_15") + 1.0))
          .alias("x_large_imb_15"),
    ]).drop("t_n_15")
    # 买量/卖量比（全窗）
    bv = tdf.group_by("sample_id").agg([
        pl.col("volume").filter(pl.col("is_buy")).sum().alias("_bv"),
        pl.col("volume").filter(~pl.col("is_buy")).sum().alias("_svv"),
    ]).with_columns((pl.col("_bv") / (pl.col("_svv") + 1)).alias("t_buy_sell_vol_ratio"))
    t1 = t1.join(bv.select(["sample_id", "t_buy_sell_vol_ratio"]), on="sample_id", how="left")

    # 单秒最大量（≤30s）
    tmax = (tdf.filter(sec30).group_by(["sample_id", "sec"]).agg(
        pl.col("volume").sum().alias("_vs"))
        .group_by("sample_id").agg(pl.col("_vs").max().alias("t_vol_max30")))
    t1 = t1.join(tmax, on="sample_id", how="left")

    # ================= 挂单流 =================
    odf = odf.with_columns([
        pl.col("seconds_before_predict").floor().cast(pl.Int16).alias("sec"),
        (pl.col("side") == 0).alias("is_buy"),
        (pl.col("order_action") == 0).alias("is_add"),
    ]).with_columns([
        (pl.col("is_buy").cast(pl.Int8) * 2 - 1).alias("sgn"),
        (pl.col("is_add").cast(pl.Int8) * 2 - 1).alias("act"),
    ]).with_columns([
        pl.col("price").diff(14).over("sample_id").alias("dpx14"),
        pl.col("volume").diff(14).over("sample_id").alias("dv14"),
    ])
    o1 = odf.group_by("sample_id").agg([
        ((pl.col("sgn") * pl.col("volume") * (-pl.col("seconds_before_predict") / 15.0).exp()).sum()
         ).alias("o_sv_ewm15"),
        ((pl.col("act") * pl.col("volume") * (-pl.col("seconds_before_predict") / 45.0).exp()).sum()
         ).alias("o_av_ewm45"),
        pl.col("is_add").cast(pl.Float32).mean().alias("o_add_ratio"),
        pl.col("dpx14").std().alias("o_pxdiff14_std"),
        pl.col("dv14").mean().alias("o_voldiff14_mean"),
    ])
    omax = (odf.filter(sec30).group_by(["sample_id", "sec"]).agg(
        pl.col("volume").sum().alias("_vs"))
        .group_by("sample_id").agg(pl.col("_vs").max().alias("o_vol_max30")))
    o1 = o1.join(omax, on="sample_id", how="left")

    # ================= 合并 =================
    tcols = ["t_sv_5", "t_sv_60", "x_t_sv_5_60", "t_vol_5", "x_t_vol_5_60",
             "t_n_60", "x_tn_15_60", "t_sd_ewm45", "t_vol_max30",
             "t_buy_sell_vol_ratio", "t_large_buy_15", "t_large_sell_15",
             "x_large_imb_15", "t_pxdiff14_mean", "t_pxdiff14_std",
             "t_pxdiff21_std", "t_pxdiff30_std", "t_voldiff14_mean",
             "t_voldiff21_mean"]
    ocols = ["o_sv_ewm15", "o_av_ewm45", "o_add_ratio", "o_pxdiff14_std",
             "o_voldiff14_mean", "o_vol_max30"]
    df = (m1.join(t1.select(["sample_id"] + tcols), on="sample_id", how="left")
          .join(o1.select(["sample_id"] + ocols), on="sample_id", how="left"))

    df = df.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
        for c in df.columns
        if c != "sample_id" and isinstance(df.schema[c], (pl.Float32, pl.Float64))
    ])
    return df


def compute_month3(m: int) -> pl.DataFrame:
    lab = pl.read_parquet(os.path.join(BASE, r"train/label.parquet"))
    lm = lab.filter(pl.col("month") == m)
    df = compute_block3(int(lm["sample_id"].min()), int(lm["sample_id"].max()), "train")
    return df.join(lm.select(["sample_id", "target", "month"]), on="sample_id", how="left")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("months", nargs="*", type=int, default=None)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--only-chunks", default=None)
    p.add_argument("--worker", type=int, default=None)
    p.add_argument("--worker-test", type=int, nargs=2, default=None)
    p.add_argument("--q90", type=float, default=None)
    p.add_argument("--q95", type=float, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    os.environ.setdefault("POLARS_MAX_THREADS", "4")

    if args.worker is not None:
        df = compute_month3(args.worker)
        df.write_parquet(args.out)
        print(f"[m{args.worker}] saved {df.height} rows -> {args.out}", flush=True)
        return
    if args.worker_test is not None:
        df = compute_block3(args.worker_test[0], args.worker_test[1], "test", args.q90, args.q95)
        df.write_parquet(args.out)
        print(f"[test {args.worker_test[0]}-{args.worker_test[1]}] saved {df.height} rows", flush=True)
        return

    py = os.path.abspath(__file__)
    py_exe = sys.executable
    t_all = time.time()

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
            out = os.path.join(OUT, f"new3_m{m}.parquet")
            if os.path.exists(out):
                print(f"[m{m}] 已存在，跳过")
                continue
            todo.append(m)
        for i, m in enumerate(todo):
            out = os.path.join(OUT, f"new3_m{m}.parquet")
            t0 = time.time()
            run_sub(["--worker", str(m), "--out", out])
            print(f"[m{m}] 完成 {time.time()-t0:.0f}s | {i+1}/{len(todo)} | 累计 {(time.time()-t_all)/60:.1f}min", flush=True)
    else:
        tvol = get_scan("test", "transaction").select("volume").collect()["volume"]
        q90 = float(tvol.quantile(0.90))
        q95 = float(tvol.quantile(0.95))
        print(f"test 大单阈值 q90={q90:.4g} q95={q95:.4g}", flush=True)
        del tvol
        sub = pl.read_csv(os.path.join(BASE, r"submissions/submission.csv"),
                          columns=["sample_id"])
        lo_all, hi_all = int(sub["sample_id"].min()), int(sub["sample_id"].max())
        chunks, ci, lo = [], 0, lo_all
        while lo <= hi_all:
            hi = min(lo + TEST_CHUNK - 1, hi_all)
            chunks.append((ci, lo, hi))
            ci += 1
            lo = hi + 1
        if args.only_chunks:
            keep = {int(x) for x in args.only_chunks.split(",")}
            chunks = [c for c in chunks if c[0] in keep]
        for ci, lo, hi in chunks:
            out = os.path.join(OUT, f"new3_test_c{ci}.parquet")
            if os.path.exists(out):
                print(f"[test c{ci}] 已存在，跳过")
                continue
            t0 = time.time()
            run_sub(["--worker-test", str(lo), str(hi),
                     "--q90", repr(q90), "--q95", repr(q95), "--out", out])
            print(f"[test c{ci}] 完成 {time.time()-t0:.0f}s | {ci+1}/{len(chunks)} | 累计 {(time.time()-t_all)/60:.1f}min", flush=True)
    print(f"全部完成，总耗时 {(time.time()-t_all)/60:.1f}min")


if __name__ == "__main__":
    main()
