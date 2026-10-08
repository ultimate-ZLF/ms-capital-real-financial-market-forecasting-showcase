"""因子挖掘 v1（polars 实现）：逐月计算微观结构因子，缓存到 factors/。

因子清单（理论出处见注释）：
- t_imb / t_imb_w30 / t_imb_5s  成交流失衡（Kyle 价格冲击），衰减加权版（信息新鲜度）
- t_size_imb                    大单方向（均值成交规模失衡）
- ofi / ofi_w30 / ofi_norm      订单流失衡（Cont-Kukanov-Stoikov 2014），区分挂单/撤单
- d60_mid / d600_mid            中间价动量/长窗反转
- d60_micro                     微价漂移（Stoikov microprice）
- book_imb0 / d_book_imb        盘口失衡及其变化
- spread0 / spread_mean60       价差
- sig_txn / mid_vol60 / mid_vol600  局部波动率（用于归一化和波动率预测）
- o_n / o_vol / t_n / t_vol     活动量

用法：python factors.py 0 5 10 20 30 40 50 55 60 63 66 68 70
"""
import os
import sys
import polars as pl

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
OUT = os.path.join(BASE, "factors")
os.makedirs(OUT, exist_ok=True)


def compute_batch(ids, mt, ot, tt):
    """对一组 sample_id 计算因子，返回 polars DataFrame（不含 target/month）。"""
    ids = list(ids)
    print(f"batch n={len(ids)}", flush=True)

    # ---------- transaction（60s 成交流） ----------
    tdf = tt.filter(pl.col("sample_id").is_in(ids)).with_columns([
        (pl.col("volume") * (pl.col("side") == 0)).alias("buy_v"),
        (pl.col("volume") * (pl.col("side") == 1)).alias("sell_v"),
    ]).with_columns([
        (pl.col("buy_v") * (-pl.col("seconds_before_predict") / 30.0).exp()).alias("buy_vw30"),
        (pl.col("sell_v") * (-pl.col("seconds_before_predict") / 30.0).exp()).alias("sell_vw30"),
        (pl.col("buy_v") * (-pl.col("seconds_before_predict") / 10.0).exp()).alias("buy_vw10"),
        (pl.col("sell_v") * (-pl.col("seconds_before_predict") / 10.0).exp()).alias("sell_vw10"),
    ])
    tg = tdf.group_by("sample_id").agg([
        pl.col("buy_v").sum(), pl.col("sell_v").sum(),
        pl.col("buy_vw30").sum(), pl.col("sell_vw30").sum(),
        pl.col("buy_vw10").sum(), pl.col("sell_vw10").sum(),
        (pl.col("buy_v") > 0).sum().alias("buy_n"),
        (pl.col("sell_v") > 0).sum().alias("sell_n"),
        pl.col("volume").sum().alias("t_vol"),
        pl.len().alias("t_n"),
    ]).with_columns([
        ((pl.col("buy_v") - pl.col("sell_v")) / (pl.col("buy_v") + pl.col("sell_v"))).alias("t_imb"),
        ((pl.col("buy_vw30") - pl.col("sell_vw30")) / (pl.col("buy_vw30") + pl.col("sell_vw30"))).alias("t_imb_w30"),
        ((pl.col("buy_vw10") - pl.col("sell_vw10")) / (pl.col("buy_vw10") + pl.col("sell_vw10"))).alias("t_imb_w10"),
    ])
    # 大单方向：买卖平均单笔规模失衡
    bs, ss = pl.col("buy_v") / pl.col("buy_n"), pl.col("sell_v") / pl.col("sell_n")
    tg = tg.with_columns(((bs - ss) / (bs + ss)).alias("t_size_imb"))
    # 短窗口失衡（扫描结论：窗口越短越强，2s 最强）
    for W, tag in [(2, "2s"), (5, "5s")]:
        wg = tdf.filter(pl.col("seconds_before_predict") <= W).group_by("sample_id").agg([
            pl.col("buy_v").sum().alias("buy_v"), pl.col("sell_v").sum().alias("sell_v")])
        wg = wg.with_columns(
            ((pl.col("buy_v") - pl.col("sell_v")) / (pl.col("buy_v") + pl.col("sell_v"))).alias(f"t_imb_{tag}"))
        tg = tg.join(wg.select(["sample_id", f"t_imb_{tag}"]), on="sample_id", how="left")
    # 成交价局部波动率：连续成交价差分的 std
    sig_txn = tdf.sort(["sample_id", "seconds_before_predict"]).with_columns(
        pl.col("price").diff().over("sample_id").alias("dpx")
    ).group_by("sample_id").agg(pl.col("dpx").std().alias("sig_txn"))

    # ---------- order（60s 订单流） ----------
    # OFI: 买挂+ 买撤- 卖挂- 卖撤+（假设 side0=买, action0=挂, action1=撤；方向由检验校正）
    ofi_c = (
        pl.when((pl.col("side") == 0) & (pl.col("order_action") == 0)).then(1.0)
        .when((pl.col("side") == 0) & (pl.col("order_action") == 1)).then(-1.0)
        .when((pl.col("side") == 1) & (pl.col("order_action") == 0)).then(-1.0)
        .otherwise(1.0)
    )
    odf = ot.filter(pl.col("sample_id").is_in(ids)).with_columns([
        (ofi_c * pl.col("volume")).alias("ofi_c"),
        ((-pl.col("seconds_before_predict") / 30.0).exp()).alias("w30"),
        ((-pl.col("seconds_before_predict") / 10.0).exp()).alias("w10"),
    ]).with_columns([
        (pl.col("ofi_c") * pl.col("w30")).alias("ofi_cw30"),
        (pl.col("volume") * pl.col("w30")).alias("volw30"),
        (pl.col("ofi_c") * pl.col("w10")).alias("ofi_cw10"),
        (pl.col("volume") * pl.col("w10")).alias("volw10"),
    ])
    og = odf.group_by("sample_id").agg([
        pl.col("ofi_c").sum().alias("ofi"),
        pl.col("ofi_cw30").sum().alias("ofi_w30"),
        pl.col("volume").sum().alias("o_vol"),
        pl.col("volw30").sum().alias("o_vol_w30"),
        pl.col("ofi_cw10").sum().alias("ofi_w10"),
        pl.col("volw10").sum().alias("o_vol_w10"),
        pl.len().alias("o_n"),
    ]).with_columns([
        (pl.col("ofi") / pl.col("o_vol")).alias("ofi_norm"),
        (pl.col("ofi_w30") / pl.col("o_vol_w30")).alias("ofi_w30_norm"),
        (pl.col("ofi_w10") / pl.col("o_vol_w10")).alias("ofi_w10_norm"),
    ])
    o5 = odf.filter(pl.col("seconds_before_predict") <= 5).group_by("sample_id").agg(
        pl.col("ofi_c").sum().alias("ofi_5s"))
    og = og.join(o5, on="sample_id", how="left")

    # ---------- market（600s 盘口快照） ----------
    mdf = mt.filter(pl.col("sample_id").is_in(ids)).with_columns([
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2.0).alias("mid"),
        ((pl.col("ask_price_1") * pl.col("bid_volume_1")
          + pl.col("bid_price_1") * pl.col("ask_volume_1"))
         / (pl.col("ask_volume_1") + pl.col("bid_volume_1"))).alias("micro"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread"),
    ]).sort(["sample_id", "seconds_before_predict"])  # sec 升序 → idx 0 = 最新
    mdf = mdf.with_columns([
        pl.int_range(pl.len()).over("sample_id").alias("idx"),
    ]).with_columns([
        (pl.len().over("sample_id") - 1 - pl.col("idx")).alias("idx_rev"),
    ])
    first = mdf.filter(pl.col("idx") == 0)          # sec≈0-3（预测时刻）
    w60 = mdf.filter(pl.col("idx") == 19)           # 第 20 个快照 ≈ sec≈57-63
    oldest = mdf.filter(pl.col("idx_rev") == 0)     # sec≈596（10 分钟前）
    w60m = mdf.filter(pl.col("idx") < 20)           # 最近 ~60 秒

    # join 三张快照表（idx0=预测时刻, idx19≈60s前, idx_rev0≈600s前）
    key = first.select(["sample_id", "mid", "micro",
                        pl.col("ask_volume_1").alias("av1"), pl.col("bid_volume_1").alias("bv1"),
                        pl.col("spread").alias("spread0")])
    key60 = w60.select(["sample_id",
                        pl.col("mid").alias("mid60"), pl.col("micro").alias("micro60"),
                        pl.col("ask_volume_1").alias("av1_60"), pl.col("bid_volume_1").alias("bv1_60")])
    key600 = oldest.select(["sample_id", pl.col("mid").alias("mid600")])
    feat = key.join(key60, on="sample_id", how="left").join(key600, on="sample_id", how="left")
    feat = feat.with_columns([
        (pl.col("mid") - pl.col("mid60")).alias("d60_mid"),
        (pl.col("mid") - pl.col("mid600")).alias("d600_mid"),
        (pl.col("micro") - pl.col("micro60")).alias("d60_micro"),
        ((pl.col("bv1") - pl.col("av1")) / (pl.col("bv1") + pl.col("av1"))).alias("book_imb0"),
        ((pl.col("bv1_60") - pl.col("av1_60")) / (pl.col("bv1_60") + pl.col("av1_60"))).alias("book_imb60"),
        pl.col("spread0"),
    ]).with_columns([
        (pl.col("book_imb0") - pl.col("book_imb60")).alias("d_book_imb"),
    ])
    spread_mean60 = w60m.group_by("sample_id").agg(pl.col("spread").mean().alias("spread_mean60"))
    w60m = w60m.with_columns(pl.col("mid").diff().over("sample_id").alias("dmid"))
    mid_vol60 = w60m.group_by("sample_id").agg(pl.col("dmid").std().alias("mid_vol60"))
    mid_vol600 = mdf.with_columns(pl.col("mid").diff().over("sample_id").alias("dmid")) \
        .group_by("sample_id").agg(pl.col("dmid").std().alias("mid_vol600"))

    # ========== 非共识因子 v2（五个视角） ==========
    # A 族：深度归一化冲击（Kyle λ 微观版）——每笔成交用成交前最近一张快照的盘口深度归一化
    #   join_asof forward：找 sec >= 成交 sec 的最小快照（绝对时间上恰在成交之前）
    snap = mdf.select(["sample_id", "seconds_before_predict", "ask_price_1", "bid_price_1",
                       "ask_volume_1", "bid_volume_1"])
    j = tdf.sort(["sample_id", "seconds_before_predict"]).join_asof(
        snap, on="seconds_before_predict", by="sample_id", strategy="forward")
    j = j.with_columns([
        (pl.col("volume") * pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)
         / (pl.col("ask_volume_1") + pl.col("bid_volume_1"))).alias("ic"),
        pl.when((pl.col("price") > pl.col("ask_price_1")) | (pl.col("price") < pl.col("bid_price_1")))
          .then(1.0).otherwise(0.0).alias("cross"),
    ]).with_columns([
        (pl.col("volume") * pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)
         * pl.col("cross")).alias("acv"),
        (pl.col("volume") * pl.col("cross")).alias("av"),
    ])
    imp_agg = j.group_by("sample_id").agg([
        pl.col("ic").sum().alias("imp_depth_60"),
        pl.col("ic").filter(pl.col("seconds_before_predict") <= 5).sum().alias("imp_depth_5"),
        pl.col("acv").sum().alias("ac_sum"),
        pl.col("av").sum().alias("av_sum"),
        pl.col("cross").mean().alias("spread_cross_frac"),
    ]).with_columns(
        (pl.col("ac_sum") / pl.col("av_sum")).alias("aggr_imb")).drop(["ac_sum", "av_sum"])

    # B 族：撤单行为与耐心不对称（谁会先放弃）
    odf = odf.with_columns([
        (pl.col("volume") * ((pl.col("side") == 0) & (pl.col("order_action") == 0))).alias("ba"),
        (pl.col("volume") * ((pl.col("side") == 0) & (pl.col("order_action") == 1))).alias("bc"),
        (pl.col("volume") * ((pl.col("side") == 1) & (pl.col("order_action") == 0))).alias("sa"),
        (pl.col("volume") * ((pl.col("side") == 1) & (pl.col("order_action") == 1))).alias("sc"),
    ])
    bfam = odf.group_by("sample_id").agg([
        pl.col("ba").sum(), pl.col("bc").sum(), pl.col("sa").sum(), pl.col("sc").sum(),
    ]).with_columns([
        ((pl.col("bc") - pl.col("sc")) / (pl.col("bc") + pl.col("sc"))).alias("cancel_imb"),
        ((pl.col("ba") - pl.col("bc")) / (pl.col("ba") + pl.col("bc"))).alias("pat_bid"),
        ((pl.col("sa") - pl.col("sc")) / (pl.col("sa") + pl.col("sc"))).alias("pat_ask"),
    ]).with_columns((pl.col("pat_bid") - pl.col("pat_ask")).alias("patience_diff"))

    # C 族：价格路径形态（均值回复视角）
    w120 = mdf.filter(pl.col("idx") == 39)  # 第 40 个快照 ≈ sec≈117-123
    mm = mdf.group_by("sample_id").agg([
        pl.col("mid").min().alias("mid_min600"), pl.col("mid").max().alias("mid_max600")])
    vwap60 = tdf.group_by("sample_id").agg(
        ((pl.col("price") * pl.col("volume")).sum() / pl.col("volume").sum()).alias("vwap60"))
    vwap600 = mdf.group_by("sample_id").agg(
        ((pl.col("transaction_avgprice") * pl.col("transaction_volume")).sum()
         / pl.col("transaction_volume").sum()).alias("vwap600"))
    cfam = (first.select(["sample_id", pl.col("mid").alias("mid0")])
            .join(w60.select(["sample_id", pl.col("mid").alias("mid60")]), on="sample_id", how="left")
            .join(w120.select(["sample_id", pl.col("mid").alias("mid120")]), on="sample_id", how="left")
            .join(mm, on="sample_id", how="left")
            .join(vwap60, on="sample_id", how="left")
            .join(vwap600, on="sample_id", how="left"))
    cfam = cfam.with_columns([
        pl.when(pl.col("mid_max600") > pl.col("mid_min600"))
          .then((pl.col("mid0") - pl.col("mid_min600")) / (pl.col("mid_max600") - pl.col("mid_min600")))
          .otherwise(None).alias("range_pos"),
        (pl.col("mid0") - 2 * pl.col("mid60") + pl.col("mid120")).alias("curv"),
        (pl.col("mid0") - pl.col("vwap60")).alias("vwap_dev60"),
        (pl.col("mid0") - pl.col("vwap600")).alias("vwap_dev600"),
    ])
    sk = w60m.group_by("sample_id").agg([
        pl.col("dmid").filter(pl.col("dmid") > 0).std().alias("up_vol"),
        pl.col("dmid").filter(pl.col("dmid") < 0).std().alias("down_vol"),
    ]).with_columns(
        pl.when((pl.col("up_vol") > 0) & (pl.col("down_vol") > 0))
          .then(pl.col("up_vol") / pl.col("down_vol")).otherwise(None).alias("vol_skew"))

    # D 族：时间节奏（clock）
    last_gap = tdf.group_by("sample_id").agg(
        pl.col("seconds_before_predict").min().alias("last_gap"))
    buckets = tdf.with_columns((pl.col("seconds_before_predict") // 5).cast(pl.Int32).alias("b5"))
    bcount = buckets.group_by(["sample_id", "b5"]).len()
    burst5 = bcount.group_by("sample_id").agg(
        (pl.col("len").max() / pl.col("len").sum()).alias("burst5"))
    n_near = buckets.filter(pl.col("seconds_before_predict") <= 15) \
        .group_by("sample_id").len().rename({"len": "n_near"})
    n_far = buckets.filter(pl.col("seconds_before_predict") >= 45) \
        .group_by("sample_id").len().rename({"len": "n_far"})
    accel = n_near.join(n_far, on="sample_id", how="left").with_columns(
        ((pl.col("n_near") - pl.col("n_far")) / (pl.col("n_near") + pl.col("n_far"))).alias("accel"))

    # E 族：盘口演化（流动性与波动率状态变化）
    efam = (first.select(["sample_id",
                          (pl.col("ask_volume_1") + pl.col("bid_volume_1")).alias("d0")])
            .join(w60.select(["sample_id",
                              (pl.col("ask_volume_1") + pl.col("bid_volume_1")).alias("d60")]),
                  on="sample_id", how="left")
            .join(spread_mean60, on="sample_id", how="left")
            .join(feat.select(["sample_id", "spread0"]), on="sample_id", how="left"))
    efam = efam.with_columns([
        ((pl.col("spread0") - pl.col("spread_mean60")) / pl.col("spread_mean60")).alias("spread_sqz"),
        ((pl.col("d0") - pl.col("d60")) / (pl.col("d0") + pl.col("d60"))).alias("depth_flow"),
    ])
    vol_ratio = mid_vol60.join(mid_vol600, on="sample_id", how="left").with_columns(
        pl.when(pl.col("mid_vol600") > 0)
          .then(pl.col("mid_vol60") / pl.col("mid_vol600")).otherwise(None).alias("vol_ratio"))

    # ---------- 合并 ----------
    tcols = ["t_imb", "t_imb_w30", "t_imb_w10", "t_size_imb", "t_imb_2s", "t_imb_5s", "t_vol", "t_n"]
    ocols = ["ofi", "ofi_w30", "ofi_w10", "ofi_norm", "ofi_w30_norm", "ofi_w10_norm", "ofi_5s", "o_vol", "o_n"]
    df = (feat
          .join(tg.select(["sample_id"] + tcols), on="sample_id", how="left")
          .join(og.select(["sample_id"] + ocols), on="sample_id", how="left")
          .join(spread_mean60, on="sample_id", how="left")
          .join(mid_vol60, on="sample_id", how="left")
          .join(mid_vol600, on="sample_id", how="left")
          .join(sig_txn, on="sample_id", how="left")
          # v2 非共识因子
          .join(imp_agg, on="sample_id", how="left")
          .join(bfam.select(["sample_id", "cancel_imb", "patience_diff"]), on="sample_id", how="left")
          .join(cfam.select(["sample_id", "range_pos", "curv", "vwap_dev60", "vwap_dev600"]),
                on="sample_id", how="left")
          .join(sk.select(["sample_id", "vol_skew"]), on="sample_id", how="left")
          .join(last_gap, on="sample_id", how="left")
          .join(burst5, on="sample_id", how="left")
          .join(accel.select(["sample_id", "accel"]), on="sample_id", how="left")
          .join(efam.select(["sample_id", "spread_sqz", "depth_flow"]), on="sample_id", how="left")
          .join(vol_ratio.select(["sample_id", "vol_ratio"]), on="sample_id", how="left")
          .drop(["av1", "bv1", "av1_60", "bv1_60"]))
    # 安全网：任何 +-inf → null（0 除：imp_depth 盘口深度为 0、t_size_imb 无买单、depth_flow/spread_sqz 等）
    df = df.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
        for c in df.columns
        if isinstance(df.schema[c], (pl.Float32, pl.Float64))
    ])
    return df


def compute_month(m, label, mt, ot, tt):
    """训练集单月：因子 + target 标签，保存到 factors/。"""
    ids = label.filter(pl.col("month") == m).get_column("sample_id").to_list()
    df = compute_batch(ids, mt, ot, tt)
    df = (df.join(label.filter(pl.col("month") == m).select(["sample_id", "target"]),
                  on="sample_id", how="left")
          .with_columns(pl.lit(m).alias("month")))
    out = os.path.join(OUT, f"factors_m{m}.parquet")
    df.write_parquet(out)
    print(f"[month {m}] saved {df.height} rows -> {out}", flush=True)


if __name__ == "__main__":
    months = [int(x) for x in sys.argv[1:]]
    print("loading files...", flush=True)
    label = pl.read_parquet(os.path.join(BASE, r"train/label.parquet"))
    mt = pl.read_parquet(os.path.join(BASE, r"train/market.parquet"))
    ot = pl.read_parquet(os.path.join(BASE, r"train/order.parquet"))
    tt = pl.read_parquet(os.path.join(BASE, r"train/transaction.parquet"))
    print("loaded", flush=True)
    for m in months:
        compute_month(m, label, mt, ot, tt)
    print("done")
