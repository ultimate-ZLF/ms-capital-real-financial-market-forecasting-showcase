"""Large-H scan for txn price and market transaction_avgprice:
where do σ(H) and P0(H) cross σ(target)=5.0e-3 / P0(target)=3.5%?"""
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

Hs = [60, 90, 120, 180, 240, 300, 360, 480, 600]

# --- txn price ---
with open(BASE + r"/train/transaction.parquet", "rb") as f:
    ttbl = pq.ParquetFile(f).read()
mask = pc.is_in(ttbl.column("sample_id"), pa.array(ids))
tdf = ttbl.filter(mask).to_pandas()
out = {}
for sid, g in tdf.groupby("sample_id"):
    g = g.sort_values("seconds_before_predict")
    s, p = g.seconds_before_predict.values, g.price.values
    p0 = p[0]
    for H in Hs:
        j = np.argmin(np.abs(s - H))
        out.setdefault(H, []).append(p0 - p[j])
print("### txn price ###")
print(f"{'H':>4} {'σ(dH)':>10} {'P0':>8}")
for H in Hs:
    d = np.array(out[H])
    print(f"{H:4d} {d.std():10.2e} {(d == 0).mean():8.4f}")

# --- market transaction_avgprice ---
with open(BASE + r"/train/market.parquet", "rb") as f:
    mtbl = pq.ParquetFile(f).read()
mask = pc.is_in(mtbl.column("sample_id"), pa.array(ids))
mdf = mtbl.filter(mask).to_pandas()
out = {}
for sid, g in mdf.groupby("sample_id"):
    g = g.sort_values("seconds_before_predict")
    s, p = g.seconds_before_predict.values, g.transaction_avgprice.values
    # first non-NaN nearest predict
    if np.isnan(p).all():
        continue
    p0 = None
    for i in range(len(p)):
        if not np.isnan(p[i]):
            p0 = p[i]
            break
    if p0 is None:
        continue
    for H in Hs:
        j = np.argmin(np.abs(s - H))
        # nearest non-NaN to H
        pj = None
        for off in range(0, 5):
            jj = j + off
            if jj < len(p) and not np.isnan(p[jj]):
                pj = p[jj]
                break
            jj = j - off
            if jj >= 0 and not np.isnan(p[jj]):
                pj = p[jj]
                break
        if pj is not None:
            out.setdefault(H, []).append(p0 - pj)
print("\n### market transaction_avgprice ###")
print(f"{'H':>4} {'σ(dH)':>10} {'P0':>8}")
for H in Hs:
    d = np.array(out[H])
    print(f"{H:4d} {d.std():10.2e} {(d == 0).mean():8.4f}")
