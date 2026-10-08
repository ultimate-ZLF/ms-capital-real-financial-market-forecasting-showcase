"""Test: is target = change in TRANSACTION price over horizon H?
σ(dH_txn) and P0(dH_txn) vs σ(target)=5.0e-3, P0(target)=3.5%."""
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

with open(BASE + r"/train/transaction.parquet", "rb") as f:
    ttbl = pq.ParquetFile(f).read()
mask = pc.is_in(ttbl.column("sample_id"), pa.array(ids))
tdf = ttbl.filter(mask).to_pandas()
print("txn rows:", len(tdf))

Hs = [3, 5, 10, 15, 30, 60]
out = {}
for sid, g in tdf.groupby("sample_id"):
    g = g.sort_values("seconds_before_predict")
    s, p = g.seconds_before_predict.values, g.price.values
    p0 = p[0]  # nearest trade to predict time
    for H in Hs:
        j = np.argmin(np.abs(s - H))
        out.setdefault(H, []).append(p0 - p[j])

print(f"\n{'H':>4} {'σ(dH_txn)':>12} {'P0(dH_txn)':>12}")
print(f"tgt {'':>5} {lbl.target.std():12.2e} {(lbl.target == 0).mean():12.4f}")
for H in Hs:
    d = np.array(out[H])
    print(f"{H:4d} {d.std():12.2e} {(d == 0).mean():12.4f}")

# also: per-episode vol of consecutive trade price changes (30s window)
print("\nper-episode trade price: σ(consecutive chg), first 8 episodes:")
for sid in ids[:8]:
    g = tdf[tdf.sample_id == sid].sort_values("seconds_before_predict")
    if len(g) < 2:
        print(f"  {sid}: n={len(g)}")
        continue
    chg = g.price.diff().dropna()
    print(f"  {sid}: n={len(g)} σ(chg)={chg.std():.2e} P0={ (chg==0).mean():.3f}")
