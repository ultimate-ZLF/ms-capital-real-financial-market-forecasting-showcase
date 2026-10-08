"""v2: vol-scaling horizon estimate + signal correlations (fixed: newest snapshot = first after ascending sort)."""
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
import pandas as pd

import os
BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
label = pd.read_parquet(BASE + r"/train/label.parquet")
month = 66
lbl = label[label.month == month].reset_index(drop=True)
ids = lbl.sample_id.values

with open(BASE + r"/train/market.parquet", "rb") as f:
    mtbl = pq.ParquetFile(f).read()
mask = pc.is_in(mtbl.column("sample_id"), pa.array(ids))
mdf = mtbl.filter(mask).to_pandas()
mdf["mid"] = (mdf.ask_price_1 + mdf.bid_price_1) / 2.0
mdf["micro"] = (mdf.ask_price_1 * mdf.bid_volume_1 + mdf.bid_price_1 * mdf.ask_volume_1) / (
    mdf.ask_volume_1 + mdf.bid_volume_1)

# --- per-episode: newest-snapshot features + mid change over horizons H ---
Hs = [3, 15, 30, 60, 120, 300, 600]
rows = []
for sid, g in mdf.groupby("sample_id"):
    g = g.sort_values("seconds_before_predict")  # ascending → g.iloc[0] is nearest to predict
    s, m = g.seconds_before_predict.values, g.mid.values
    first = g.iloc[0]
    r = {"sample_id": sid, "mid0": m[0]}
    for H in Hs:
        j = np.argmin(np.abs(s - H))
        r[f"d{H}"] = m[0] - m[j]
    r["d3s_std"] = np.diff(m).std()
    r["spread"] = first.ask_price_1 - first.bid_price_1
    r["micro_imb"] = first.micro - first.mid
    r["book_imb"] = (first.bid_volume_1 - first.ask_volume_1) / (
        first.bid_volume_1 + first.ask_volume_1)
    r["last_txv"] = first.transaction_volume
    rows.append(r)
feat = pd.DataFrame(rows)

tstd = lbl.target.std()
print(f"σ(target) = {tstd:.2e}")
print("### σ(mid change over H) vs H (month %d) ###" % month)
print(f"{'H':>5} {'σ':>10} {'σ/√H':>10}")
for H in Hs:
    sd = feat[f"d{H}"].std()
    print(f"{H:5d} {sd:10.2e} {sd/np.sqrt(H):10.2e}")

# --- signal correlations ---
with open(BASE + r"/train/order.parquet", "rb") as f:
    otbl = pq.ParquetFile(f).read()
with open(BASE + r"/train/transaction.parquet", "rb") as f:
    ttbl = pq.ParquetFile(f).read()
masko = pc.is_in(otbl.column("sample_id"), pa.array(ids))
odf = otbl.filter(masko).to_pandas()
maskt = pc.is_in(ttbl.column("sample_id"), pa.array(ids))
tdf = ttbl.filter(maskt).to_pandas()

o_agg = odf.groupby("sample_id").apply(
    lambda g: pd.Series({
        "o_buy_v": g[g.side == 0].volume.sum(),
        "o_sell_v": g[g.side == 1].volume.sum(),
        "o_n": len(g),
    }), include_groups=False).reset_index()
t_agg = tdf.groupby("sample_id").apply(
    lambda g: pd.Series({
        "t_buy_v": g[g.side == 0].volume.sum(),
        "t_sell_v": g[g.side == 1].volume.sum(),
        "t_n": len(g),
    }), include_groups=False).reset_index()

df = feat.merge(o_agg, on="sample_id").merge(t_agg, on="sample_id").merge(
    lbl[["sample_id", "target"]], on="sample_id")
df["o_imb"] = (df.o_buy_v - df.o_sell_v) / (df.o_buy_v + df.o_sell_v)
df["t_imb"] = (df.t_buy_v - df.t_sell_v) / (df.t_buy_v + df.t_sell_v)

print(f"\n### signal vs target correlations (month {month}, n={len(df)}) ###")
print(f"{'signal':<22} {'corr(target)':>12} {'corr(sign)':>10} {'corr(|tgt|)':>10}")
for c in ["d3", "d15", "d60", "d300", "spread", "micro_imb", "book_imb", "last_txv",
          "o_imb", "t_imb", "o_n", "t_n", "o_buy_v", "t_buy_v"]:
    x = df[c]
    print(f"{c:<22} {np.corrcoef(x, df.target)[0,1]:>12.4f} "
          f"{np.corrcoef(x, np.sign(df.target))[0,1]:>10.4f} "
          f"{np.corrcoef(x, df.target.abs())[0,1]:>10.4f}")
