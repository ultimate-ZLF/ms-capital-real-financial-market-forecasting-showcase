"""快速扫描：t_imb 的最优回看窗口和衰减常数（只处理 transaction 文件）。"""
import os
import numpy as np
import polars as pl

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
months = [0, 5, 10, 20, 30, 40, 50, 55, 60, 63, 66, 68, 70]

label = pl.read_parquet(os.path.join(BASE, r"train/label.parquet"))
tt = pl.read_parquet(os.path.join(BASE, r"train/transaction.parquet"))

frames = []
for m in months:
    ids = label.filter(pl.col("month") == m).get_column("sample_id").to_list()
    tdf = tt.filter(pl.col("sample_id").is_in(ids)).with_columns([
        (pl.col("volume") * (pl.col("side") == 0)).alias("buy_v"),
        (pl.col("volume") * (pl.col("side") == 1)).alias("sell_v"),
    ])
    g = tdf.group_by("sample_id").agg([
        pl.col("buy_v").sum(), pl.col("sell_v").sum(),
        (pl.col("side") == 0).sum().alias("buy_n"),
        (pl.col("side") == 1).sum().alias("sell_n"),
    ])
    # 回看窗口 W：仅保留 sec<=W 的事件
    for W in [2, 5, 10, 15, 30, 60]:
        sub = tdf.filter(pl.col("seconds_before_predict") <= W)
        sg = sub.group_by("sample_id").agg([
            pl.col("buy_v").sum().alias(f"b{W}"), pl.col("sell_v").sum().alias(f"s{W}")])
        g = g.join(sg, on="sample_id", how="left")
    # 衰减常数 tau
    for tau in [10, 20, 60]:
        w = (-pl.col("seconds_before_predict") / tau).exp()
        sub = tdf.with_columns((pl.col("buy_v") * w).alias("bw"), (pl.col("sell_v") * w).alias("sw"))
        sg = sub.group_by("sample_id").agg([pl.col("bw").sum().alias(f"bw{tau}"), pl.col("sw").sum().alias(f"sw{tau}")])
        g = g.join(sg, on="sample_id", how="left")
    g = g.join(label.filter(pl.col("month") == m).select(["sample_id", "target"]), on="sample_id", how="left")
    g = g.with_columns(pl.lit(m).alias("month"))
    frames.append(g)

df = pl.concat(frames)
tgt = df["target"].to_numpy()

def corr(x):
    m = ~(np.isnan(x) | np.isnan(tgt))
    return np.corrcoef(x[m], tgt[m])[0, 1]

print(f"{'variant':<22} {'corr':>9} {'pos_frac':>9}")
variants = []
for W in [2, 5, 10, 15, 30, 60]:
    x = ((df[f"b{W}"] - df[f"s{W}"]) / (df[f"b{W}"] + df[f"s{W}"])).to_numpy()
    variants.append((f"t_imb_{W}s", x))
for tau in [10, 20, 60]:
    x = ((df[f"bw{tau}"] - df[f"sw{tau}"]) / (df[f"bw{tau}"] + df[f"sw{tau}"])).to_numpy()
    variants.append((f"t_imb_w{tau}", x))
x = ((df["buy_n"] - df["sell_n"]) / (df["buy_n"] + df["sell_n"])).to_numpy()
variants.append(("t_count_imb", x))

for name, x in variants:
    c = corr(x)
    # 逐月符号稳定性
    pos = 0
    for m in months:
        mask = (df["month"].to_numpy() == m)
        xx, yy = x[mask], tgt[mask]
        ok = ~(np.isnan(xx) | np.isnan(yy))
        cc = np.corrcoef(xx[ok], yy[ok])[0, 1]
        pos += int(cc > 0)
    print(f"{name:<22} {c:9.4f} {pos:>6}/{len(months)}")
