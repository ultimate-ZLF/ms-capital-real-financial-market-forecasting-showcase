"""new_factors2.py — 第二批因子（加法阶段：补齐侦察中所有未实现概念，~64 个）。

来源：Rib lgb-baseline / ysx 0726data / yangq369 v10 因子名（只借概念自实现）。
输出：factors/new2_m{m}.parquet（train 月）、factors/new2_test_c{c}.parquet（test 块）。

清单（64 个）：
market 快照级（20）:
  m_mid_last/mean/std/skew        mid 分布（Rib 的 level 族）
  m_sp_mean / m_sp_std            L1 价差全窗分布
  m_imb_mean / m_imb_std / m_imb2_mean60    L1 失衡均值/std + 平方项 60s（v10 非线性）
  m_dofi / m_dofi_60 / m_dofi_ewm120        快照级深度差分 OFI（Δav1−Δbv1，无 order 表）
  m_mid_ewm{30,120} / m_imb_ewm{30,120}     快照行级 exp 衰减均值（τ=30/120，Σw·x/Σw）
  x_imb_ewm_sl / x_dofi_ewm_sl    EWM 长短差（τ30−τ120）
  x_mid_zscore                    (mid_last−mid_mean)/mid_std（区间位置）
  x_vwap_mid_ratio                (vwap60−mid_last)/mid_last
成交流（26）:
  t_sd / t_sd_15 / t_sd_45        Σ sgn·price·vol 有向金额量（Rib；v10 保留）
  t_px_std / t_px_skew / t_px_last 成交价分布
  t_lv_mean                       mean(log1p(vol)) 单笔规模
  t_gap_mean/ std/ max/ cv        事件间隔分布（gap_cv=std/mean 稳定性）
  t_autocorr_lag{1,5}             秒级价格自相关（ysx）
  t_firsthalf_price / _buyratio / t_firstthird_vol / t_firstlast_px   事件序分段
  x_t_signed_ratio                Σsgn·vol/Σvol（全窗）
  x_t_signed_w15_diff             15s 内 exp 加权失衡 − 等权失衡（新鲜度偏离）
  t_pxdiff{1,7}_mean/std         事件序 lag 1/7 价格差分统计（ysx 事件级 lag 族简化版）
  t_voldiff{1,7}_mean             同上 volume 版
  t_vwap                          VWAP（x_vwap_mid_ratio 原料 + 独立因子）
挂单流（16）:
  o_px_std / o_px_skew            挂单价分布（tj/ysx）
  o_bid_depth / o_ask_depth       Σvol 分侧全窗
  o_sv / o_av                     Σsgn·vol（不分 action）/ Σact·vol（action0=+1 else −1）
  o_gap_mean/ std/ cv             挂单事件间隔
  o_cancel_new_ratio              action==1 笔数 / action==0 笔数
  o_firsthalf_price / o_firstthird_vol
  o_pxdiff1_mean/std / o_pxdiff7_std / o_voldiff1_mean
跨表（2）:
  x_sp_imb                       m_sp_mean × m_imb_mean（显式交互，Rib）
  x_trans_order_vol_ratio        t_vol / o_vol

用法：python new_factors2.py [months...]  /  python new_factors2.py --split test [--only-chunks i]
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


def ew_m(col_x, tau):
    """快照行级 exp(−sec/τ) 归一加权均值表达式。"""
    return (col_x * (-pl.col("seconds_before_predict") / tau).exp()).sum() \
        / (-pl.col("seconds_before_predict") / tau).exp().sum()


def compute_block2(lo: int, hi: int, split: str) -> pl.DataFrame:
    mdf = (get_scan(split, "market")
           .filter(pl.col("sample_id").is_between(lo, hi)).collect())
    tdf = (get_scan(split, "transaction")
           .filter(pl.col("sample_id").is_between(lo, hi)).collect())
    odf = (get_scan(split, "order")
           .filter(pl.col("sample_id").is_between(lo, hi)).collect())

    mdf = mdf.sort(["sample_id", "seconds_before_predict"])
    tdf = tdf.sort(["sample_id", "seconds_before_predict"])
    odf = odf.sort(["sample_id", "seconds_before_predict"])

    # ================= market 快照级 =================
    mdf = mdf.with_columns([
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2.0).alias("mid"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread"),
        ((pl.col("ask_price_1") > 0) & (pl.col("bid_price_1") > 0)).alias("_valid"),
    ])
    # imb（我方方向：bid−ask）/ dofi（快照行差分）
    mdf = mdf.with_columns([
        pl.when(pl.col("ask_volume_1") + pl.col("bid_volume_1") > 0)
          .then((pl.col("bid_volume_1") - pl.col("ask_volume_1"))
                / (pl.col("ask_volume_1") + pl.col("bid_volume_1")))
          .otherwise(None).alias("imb"),
        pl.when(pl.col("_valid"))
          .then(pl.col("mid")).otherwise(None).alias("mid_v"),
    ]).with_columns(
        (pl.col("ask_volume_1") - pl.col("bid_volume_1")).diff().over("sample_id")
          .fill_null(0.0).alias("dofi_d"))

    m1 = mdf.group_by("sample_id").agg([
        pl.col("mid_v").first().alias("m_mid_last"),
        pl.col("mid_v").mean().alias("m_mid_mean"),
        pl.col("mid_v").std().alias("m_mid_std"),
        pl.col("mid_v").skew().alias("m_mid_skew"),
        pl.col("spread").filter(pl.col("_valid")).mean().alias("m_sp_mean"),
        pl.col("spread").filter(pl.col("_valid")).std().alias("m_sp_std"),
        pl.col("imb").mean().alias("m_imb_mean"),
        pl.col("imb").std().alias("m_imb_std"),
        pl.col("imb").filter(pl.col("seconds_before_predict") <= 60)
          .pow(2).mean().alias("m_imb2_mean60"),
        pl.col("dofi_d").sum().alias("m_dofi"),
        pl.col("dofi_d").filter(pl.col("seconds_before_predict") <= 60)
          .sum().alias("m_dofi_60"),
        ew_m(pl.col("dofi_d"), 120.0).alias("m_dofi_ewm120"),
        ew_m(pl.col("mid_v"), 30.0).alias("m_mid_ewm30"),
        ew_m(pl.col("mid_v"), 120.0).alias("m_mid_ewm120"),
        ew_m(pl.col("imb"), 30.0).alias("m_imb_ewm30"),
        ew_m(pl.col("imb"), 120.0).alias("m_imb_ewm120"),
    ]).with_columns([
        (pl.col("m_imb_ewm30") - pl.col("m_imb_ewm120")).alias("x_imb_ewm_sl"),
        ((pl.col("m_mid_last") - pl.col("m_mid_mean"))
         / (pl.col("m_mid_std") + 1e-12)).alias("x_mid_zscore"),
    ])
    # dofi 长短差 = ewm30 − ewm120（在 agg 里直接算）
    m1b = mdf.group_by("sample_id").agg([
        (ew_m(pl.col("dofi_d"), 30.0) - ew_m(pl.col("dofi_d"), 120.0))
          .alias("x_dofi_ewm_sl"),
    ])
    m1 = m1.join(m1b, on="sample_id", how="left")

    # ================= 成交流 =================
    tdf = tdf.with_columns([
        pl.col("seconds_before_predict").floor().cast(pl.Int16).alias("sec"),
        (pl.col("side") == 0).alias("is_buy"),
    ]).with_columns([
        (pl.col("is_buy").cast(pl.Int8) * 2 - 1).alias("sgn"),
        pl.int_range(pl.len()).over("sample_id").alias("_r"),
        pl.len().over("sample_id").alias("_n"),
    ]).with_columns([
        (pl.col("sgn") * pl.col("price") * pl.col("volume")).alias("sd"),
        (pl.col("price") * pl.col("volume")).alias("pv"),
        pl.col("price").diff().over("sample_id").alias("dpx1"),
        pl.col("price").diff(7).over("sample_id").alias("dpx7"),
        pl.col("volume").diff().over("sample_id").alias("dv1"),
        pl.col("volume").diff(7).over("sample_id").alias("dv7"),
        pl.col("seconds_before_predict").diff().over("sample_id").alias("gap"),
    ])
    half_n = pl.col("_r") < pl.col("_n") / 2
    third = pl.col("_r") < pl.col("_n") / 3
    lastthird = pl.col("_r") >= 2 * pl.col("_n") / 3
    t1 = tdf.group_by("sample_id").agg([
        pl.col("sd").sum().alias("t_sd"),
        pl.col("sd").filter(pl.col("sec") <= 15).sum().alias("t_sd_15"),
        pl.col("sd").filter(pl.col("sec") <= 45).sum().alias("t_sd_45"),
        pl.col("price").std().alias("t_px_std"),
        pl.col("price").skew().alias("t_px_skew"),
        pl.col("price").first().alias("t_px_last"),
        pl.col("volume").log1p().mean().alias("t_lv_mean"),
        pl.col("gap").mean().alias("t_gap_mean"),
        pl.col("gap").std().alias("t_gap_std"),
        pl.col("gap").max().alias("t_gap_max"),
        pl.col("price").filter(half_n).mean().alias("_ph"),
        pl.col("price").filter(~half_n).mean().alias("_ps"),
        pl.col("is_buy").filter(half_n).cast(pl.Float32).mean().alias("_bh"),
        pl.col("is_buy").filter(~half_n).cast(pl.Float32).mean().alias("_bs"),
        pl.col("volume").filter(third).sum().alias("_v1"),
        pl.col("volume").filter(lastthird).sum().alias("_v3"),
        pl.col("price").first().alias("_p0"),
        pl.col("price").last().alias("_pL"),
        pl.col("sgn").cast(pl.Float32).mean().alias("_sr"),
        pl.col("dpx1").mean().alias("t_pxdiff1_mean"),
        pl.col("dpx1").std().alias("t_pxdiff1_std"),
        pl.col("dpx7").mean().alias("t_pxdiff7_mean"),
        pl.col("dpx7").std().alias("t_pxdiff7_std"),
        pl.col("dv1").mean().alias("t_voldiff1_mean"),
        pl.col("dv7").mean().alias("t_voldiff7_mean"),
        (pl.col("pv").sum() / pl.col("volume").sum()).alias("t_vwap"),
        pl.col("volume").sum().alias("t_vol"),
    ]).with_columns([
        (pl.col("t_gap_std") / pl.col("t_gap_mean")).alias("t_gap_cv"),
        (pl.col("_ph") / pl.col("_ps") - 1.0).alias("t_firsthalf_price"),
        (pl.col("_bh") - pl.col("_bs")).alias("t_firsthalf_buyratio"),
        (pl.col("_v1") / pl.col("_v3")).alias("t_firstthird_vol"),
        (pl.col("_p0") - pl.col("_pL")).alias("t_firstlast_px"),
        (pl.col("_sr")).alias("x_t_signed_ratio"),
    ])
    # 加权−等权失衡差（15s，exp(−sec/15) 不归一与归一之比）
    t2 = tdf.filter(pl.col("sec") <= 15).group_by("sample_id").agg([
        ((pl.col("sgn") * pl.col("volume") * (-pl.col("seconds_before_predict") / 15.0).exp()).sum()
         / (pl.col("volume") * (-pl.col("seconds_before_predict") / 15.0).exp()).sum()).alias("_sw"),
        (pl.col("sgn").cast(pl.Float32).mean()).alias("_se"),
    ]).with_columns((pl.col("_sw") - pl.col("_se")).alias("x_t_signed_w15_diff"))
    t1 = t1.join(t2.select(["sample_id", "x_t_signed_w15_diff"]), on="sample_id", how="left")

    # 秒级自相关（lag 1/5）：(sample, sec) 价格均值网格 + shift
    sdf = tdf.group_by(["sample_id", "sec"]).agg(
        pl.col("price").mean().alias("px")).sort(["sample_id", "sec"])
    spm = sdf.group_by("sample_id").agg(pl.col("px").mean().alias("pxm"))
    sdf = sdf.with_columns(
        pl.col("px").shift(1).over("sample_id").alias("px1"),
        pl.col("px").shift(5).over("sample_id").alias("px5"),
    ).join(spm, on="sample_id")
    ac = sdf.group_by("sample_id").agg([
        (((pl.col("px") - pl.col("pxm")) * (pl.col("px1") - pl.col("pxm")))
         .filter(pl.col("px1").is_not_null()).sum()
         / ((pl.col("px") - pl.col("pxm")).pow(2)).sum()).alias("t_autocorr_lag1"),
        (((pl.col("px") - pl.col("pxm")) * (pl.col("px5") - pl.col("pxm")))
         .filter(pl.col("px5").is_not_null()).sum()
         / ((pl.col("px") - pl.col("pxm")).pow(2)).sum()).alias("t_autocorr_lag5"),
    ])
    t1 = t1.join(ac, on="sample_id", how="left")

    # ================= 挂单流 =================
    odf = odf.with_columns([
        (pl.col("side") == 0).alias("is_buy"),
        (pl.col("order_action") == 0).alias("is_add"),
        (pl.col("order_action") == 1).alias("is_cancel"),
        pl.int_range(pl.len()).over("sample_id").alias("_r"),
        pl.len().over("sample_id").alias("_n"),
    ]).with_columns([
        (pl.col("is_buy").cast(pl.Int8) * 2 - 1).alias("sgn"),
        (pl.col("is_add").cast(pl.Int8) * 2 - 1).alias("act"),
        pl.col("price").diff().over("sample_id").alias("dpx1"),
        pl.col("price").diff(7).over("sample_id").alias("dpx7"),
        pl.col("volume").diff().over("sample_id").alias("dv1"),
        pl.col("seconds_before_predict").diff().over("sample_id").alias("gap"),
    ])
    ohalf = pl.col("_r") < pl.col("_n") / 2
    othird = pl.col("_r") < pl.col("_n") / 3
    olast = pl.col("_r") >= 2 * pl.col("_n") / 3
    o1 = odf.group_by("sample_id").agg([
        pl.col("price").std().alias("o_px_std"),
        pl.col("price").skew().alias("o_px_skew"),
        pl.col("volume").filter(pl.col("side") == 0).sum().alias("o_bid_depth"),
        pl.col("volume").filter(pl.col("side") == 1).sum().alias("o_ask_depth"),
        (pl.col("sgn") * pl.col("volume")).sum().alias("o_sv"),
        (pl.col("act") * pl.col("volume")).sum().alias("o_av"),
        pl.col("gap").mean().alias("o_gap_mean"),
        pl.col("gap").std().alias("o_gap_std"),
        pl.col("is_cancel").cast(pl.Int32).sum().alias("_nc"),
        pl.col("is_add").cast(pl.Int32).sum().alias("_na"),
        pl.col("price").filter(ohalf).mean().alias("_ph"),
        pl.col("price").filter(~ohalf).mean().alias("_ps"),
        pl.col("volume").filter(othird).sum().alias("_v1"),
        pl.col("volume").filter(olast).sum().alias("_v3"),
        pl.col("dpx1").mean().alias("o_pxdiff1_mean"),
        pl.col("dpx1").std().alias("o_pxdiff1_std"),
        pl.col("dpx7").std().alias("o_pxdiff7_std"),
        pl.col("dv1").mean().alias("o_voldiff1_mean"),
    ]).with_columns([
        (pl.col("o_gap_std") / pl.col("o_gap_mean")).alias("o_gap_cv"),
        (pl.col("_nc") / (pl.col("_na") + 1)).alias("o_cancel_new_ratio"),
        (pl.col("_ph") / pl.col("_ps") - 1.0).alias("o_firsthalf_price"),
        (pl.col("_v1") / pl.col("_v3")).alias("o_firstthird_vol"),
    ])

    # ================= 合并 + 跨表 =================
    tcols = ["t_sd", "t_sd_15", "t_sd_45", "t_px_std", "t_px_skew", "t_px_last",
             "t_lv_mean", "t_gap_mean", "t_gap_std", "t_gap_max", "t_gap_cv",
             "t_firsthalf_price", "t_firsthalf_buyratio", "t_firstthird_vol",
             "t_firstlast_px", "x_t_signed_ratio", "x_t_signed_w15_diff",
             "t_pxdiff1_mean", "t_pxdiff1_std", "t_pxdiff7_mean", "t_pxdiff7_std",
             "t_voldiff1_mean", "t_voldiff7_mean", "t_vwap", "t_vol",
             "t_autocorr_lag1", "t_autocorr_lag5"]
    ocols = ["o_px_std", "o_px_skew", "o_bid_depth", "o_ask_depth", "o_sv", "o_av",
             "o_gap_mean", "o_gap_std", "o_gap_cv", "o_cancel_new_ratio",
             "o_firsthalf_price", "o_firstthird_vol",
             "o_pxdiff1_mean", "o_pxdiff1_std", "o_pxdiff7_std", "o_voldiff1_mean"]
    df = (m1.join(t1.select(["sample_id"] + tcols), on="sample_id", how="left")
          .join(o1.select(["sample_id"] + ocols), on="sample_id", how="left"))
    ovol = odf.group_by("sample_id").agg(pl.col("volume").sum().alias("o_vol"))
    df = df.join(ovol, on="sample_id", how="left")
    df = df.with_columns([
        (pl.col("m_sp_mean") * pl.col("m_imb_mean")).alias("x_sp_imb"),
        ((pl.col("t_vwap") - pl.col("m_mid_last"))
         / pl.col("m_mid_last")).alias("x_vwap_mid_ratio"),
        (pl.col("t_vol") / pl.col("o_vol")).alias("x_trans_order_vol_ratio"),
    ])

    # 安全网：±inf/NaN → null
    df = df.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
        for c in df.columns
        if c != "sample_id" and isinstance(df.schema[c], (pl.Float32, pl.Float64))
    ])
    return df


def compute_month2(m: int) -> pl.DataFrame:
    lab = pl.read_parquet(os.path.join(BASE, r"train/label.parquet"))
    lm = lab.filter(pl.col("month") == m)
    df = compute_block2(int(lm["sample_id"].min()), int(lm["sample_id"].max()), "train")
    return df.join(lm.select(["sample_id", "target", "month"]), on="sample_id", how="left")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("months", nargs="*", type=int, default=None)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--only-chunks", default=None)
    p.add_argument("--worker", type=int, default=None)
    p.add_argument("--worker-test", type=int, nargs=2, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    os.environ.setdefault("POLARS_MAX_THREADS", "4")

    if args.worker is not None:
        df = compute_month2(args.worker)
        df.write_parquet(args.out)
        print(f"[m{args.worker}] saved {df.height} rows -> {args.out}", flush=True)
        return
    if args.worker_test is not None:
        df = compute_block2(args.worker_test[0], args.worker_test[1], "test")
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
            out = os.path.join(OUT, f"new2_m{m}.parquet")
            if os.path.exists(out):
                print(f"[m{m}] 已存在，跳过")
                continue
            todo.append(m)
        for i, m in enumerate(todo):
            out = os.path.join(OUT, f"new2_m{m}.parquet")
            t0 = time.time()
            run_sub(["--worker", str(m), "--out", out])
            print(f"[m{m}] 完成 {time.time()-t0:.0f}s | {i+1}/{len(todo)} | 累计 {(time.time()-t_all)/60:.1f}min", flush=True)
    else:
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
            out = os.path.join(OUT, f"new2_test_c{ci}.parquet")
            if os.path.exists(out):
                print(f"[test c{ci}] 已存在，跳过")
                continue
            t0 = time.time()
            run_sub(["--worker-test", str(lo), str(hi), "--out", out])
            print(f"[test c{ci}] 完成 {time.time()-t0:.0f}s | {ci+1}/{len(chunks)} | 累计 {(time.time()-t_all)/60:.1f}min", flush=True)
    print(f"全部完成，总耗时 {(time.time()-t_all)/60:.1f}min")


if __name__ == "__main__":
    main()
