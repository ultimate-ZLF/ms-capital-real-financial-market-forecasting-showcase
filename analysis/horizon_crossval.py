"""Cross-month validation of H≈240s hypothesis:
per month, σ(avgprice change over H) should ≈ σ(target) of that month if H is fixed."""
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

with open(BASE + r"/train/market.parquet", "rb") as f:
    mtbl = pq.ParquetFile(f).read()

Hs = [180, 240, 300, 600]
print(f"{'month':>5} {'σ(tgt)':>10} " + "".join(f"{'σ(H=%d)' % H:>12}" for H in Hs))
for month in [66, 67, 70]:
    lbl = label[label.month == month].reset_index(drop=True)
    ids = lbl.sample_id.values
    mask = pc.is_in(mtbl.column("sample_id"), pa.array(ids))
    mdf = mtbl.filter(mask).to_pandas()
    out = {}
    for sid, g in mdf.groupby("sample_id"):
        g = g.sort_values("seconds_before_predict")
        s, p = g.seconds_before_predict.values, g.transaction_avgprice.values
        if np.isnan(p).all():
            continue
        p0 = next((v for v in p if not np.isnan(v)), None)
        if p0 is None:
            continue
        for H in Hs:
            j = np.argmin(np.abs(s - H))
            pj = None
            for off in range(0, 5):
                if j + off < len(p) and not np.isnan(p[j + off]):
                    pj = p[j + off]
                    break
                if j - off >= 0 and not np.isnan(p[j - off]):
                    pj = p[j - off]
                    break
            if pj is not None:
                out.setdefault(H, []).append(p0 - pj)
    sds = "".join(f"{np.array(out[H]).std():12.2e}" for H in Hs)
    print(f"{month:5d} {lbl.target.std():10.2e} {sds}")

# distribution shape: kurtosis of d240 vs target (month 66)
lbl = label[label.month == 66].reset_index(drop=True)
ids = lbl.sample_id.values
mask = pc.is_in(mtbl.column("sample_id"), pa.array(ids))
mdf = mtbl.filter(mask).to_pandas()
d = []
for sid, g in mdf.groupby("sample_id"):
    g = g.sort_values("seconds_before_predict")
    s, p = g.seconds_before_predict.values, g.transaction_avgprice.values
    if np.isnan(p).all():
        continue
    p0 = next((v for v in p if not np.isnan(v)), None)
    j = np.argmin(np.abs(s - 240))
    pj = None
    for off in range(0, 5):
        if j + off < len(p) and not np.isnan(p[j + off]):
            pj = p[j + off]
            break
        if j - off >= 0 and not np.isnan(p[j - off]):
            pj = p[j - off]
            break
    if p0 is not None and pj is not None:
        d.append(p0 - pj)
d = np.array(d)
print("\n### month 66 distribution shape ###")
for name, x in [("d240_avgprice", d), ("target", lbl.target.values)]:
    x = x[~np.isnan(x)]
    print(f"{name:<16} kurt={pd.Series(x).kurtosis():8.2f} skew={pd.Series(x).skew():8.2f} "
          f"q99={np.quantile(abs(x), .99):.2e} q999={np.quantile(abs(x), .999):.2e}")
