"""Fine-grained horizon scan: σ(mid change over H) and P(zero change) vs H,
compared with σ(target)=5.0e-3 and P(target==0)."""
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
print("month %d: n=%d  σ(target)=%.3e  P(target==0)=%.4f"
      % (month, len(lbl), lbl.target.std(), (lbl.target == 0).mean()))

with open(BASE + r"/train/market.parquet", "rb") as f:
    mtbl = pq.ParquetFile(f).read()
mask = pc.is_in(mtbl.column("sample_id"), pa.array(ids))
mdf = mtbl.filter(mask).to_pandas()
mdf["mid"] = (mdf.ask_price_1 + mdf.bid_price_1) / 2.0
mdf["micro"] = (mdf.ask_price_1 * mdf.bid_volume_1 + mdf.bid_price_1 * mdf.ask_volume_1) / (
    mdf.ask_volume_1 + mdf.bid_volume_1)

Hs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
out = {}
for sid, g in mdf.groupby("sample_id"):
    g = g.sort_values("seconds_before_predict")
    s, m, mi = g.seconds_before_predict.values, g.mid.values, g.micro.values
    for H in Hs:
        j = np.argmin(np.abs(s - H))
        out.setdefault(H, {"mid": [], "micro": []})
        out[H]["mid"].append(m[0] - m[j])
        out[H]["micro"].append(mi[0] - mi[j])

tstd, tzero = lbl.target.std(), (lbl.target == 0).mean()
print(f"\n{'H':>4} {'σ(mid)':>10} {'σ(micro)':>10} {'P0(mid)':>8} {'P0(micro)':>8}")
print(f"tgt {'':>5} {'':>9} {tstd:10.2e} {'':>8} {tzero:8.4f}")
for H in Hs:
    dmid = np.array(out[H]["mid"])
    dmic = np.array(out[H]["micro"])
    print(f"{H:4d} {dmid.std():10.2e} {dmic.std():10.2e} "
          f"{(dmid == 0).mean():8.4f} {(dmic == 0).mean():8.4f}")
