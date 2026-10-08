"""Decisive continuity tests:
A) truncation check: do low-row samples cluster at month starts?
B) order-event exact matching between (k, k+1) for Δ in [0.5, 60]
C) book-state exact matching between (k, k+1) for Δ in [2, 600]
"""
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

# ---- A) truncation check ----
for m in [65, 66, 67]:
    ids_m = label[label.month == m].sample_id.values
    first30, last30 = ids_m[:30], ids_m[-30:]
    with open(BASE + r"/train/market.parquet", "rb") as f:
        tbl = pq.ParquetFile(f).read()
    counts = {}
    for grp, name in [(first30, "first30"), (last30, "last30")]:
        mask = pc.is_in(tbl.column("sample_id"), pa.array(grp))
        sub = tbl.filter(mask)
        agg = sub.group_by("sample_id").aggregate([("sample_id", "count")])
        arr = agg.column("sample_id_count").to_pandas()
        counts[name] = (arr.min(), arr.median(), arr.max())
    print(f"month {m}: rows/sample — first30 (min,med,max)={counts['first30']}  "
          f"last30 (min,med,max)={counts['last30']}")

# ---- B & C: pairs within month 66 ----
lbl66 = label[label.month == 66].sample_id.values
pairs = [(lbl66[0], lbl66[1]), (lbl66[5], lbl66[6])]

with open(BASE + r"/train/order.parquet", "rb") as f:
    otbl = pq.ParquetFile(f).read()
with open(BASE + r"/train/market.parquet", "rb") as f:
    mtbl = pq.ParquetFile(f).read()

for (k, k1) in pairs:
    print(f"\n===== pair ({k}, {k1}) =====")
    # B) order events
    mask = pc.is_in(otbl.column("sample_id"), pa.array([k, k1]))
    odf = otbl.filter(mask).to_pandas()
    ok = odf[odf.sample_id == k].sort_values("seconds_before_predict")
    ok1 = odf[odf.sample_id == k1].sort_values("seconds_before_predict")
    def ev(df):
        return set(zip((df.seconds_before_predict * 10).round().astype(int),
                       df.price.round(5), df.volume, df.side, df.order_action))
    ev_k, ev_k1 = ev(ok), ev(ok1)
    # for each candidate Δ, count k-events with sec>=Δ whose (sec-Δ, ...) appears in k1
    best = (0, None)
    for d in np.arange(0.5, 60.0, 0.5):
        cnt = tot = 0
        for t, p, v, s, a in ev_k:
            if t / 10 < d:
                continue
            tot += 1
            if (int(round(t - d * 10)), p, v, s, a) in ev_k1:
                cnt += 1
        if tot and cnt / tot > best[0]:
            best = (cnt / tot, d)
    print(f"B) order-event match: best Δ={best[1]} match_rate={best[0]:.3f} (true Δ in [0,60] → ~1.0)")

    # C) book state
    mask = pc.is_in(mtbl.column("sample_id"), pa.array([k, k1]))
    mdf = mtbl.filter(mask).to_pandas()
    cols = ["ask_price_1", "bid_price_1", "ask_volume_1", "bid_volume_1",
            "ask_price_2", "bid_price_2", "ask_volume_2", "bid_volume_2"]
    mk = mdf[mdf.sample_id == k].sort_values("seconds_before_predict").reset_index(drop=True)
    mk1 = mdf[mdf.sample_id == k1].sort_values("seconds_before_predict").reset_index(drop=True)
    st_k = list(zip(*[mk[c] for c in cols]))
    st_k1 = list(zip(*[mk1[c] for c in cols]))
    sec_k, sec_k1 = mk.seconds_before_predict.values, mk1.seconds_before_predict.values
    best = (0, None)
    for d in np.arange(2.0, 599.0, 1.0):
        cnt = tot = 0
        for i, s in enumerate(sec_k):
            target = s + d
            if target < sec_k1[0] or target > sec_k1[-1]:
                continue
            j = np.argmin(np.abs(sec_k1 - target))
            tot += 1
            if st_k[i] == st_k1[j]:
                cnt += 1
        if tot and cnt / tot > best[0]:
            best = (cnt / tot, d)
    print(f"C) book-state match: best Δ={best[1]} match_rate={best[0]:.3f} (true Δ → high, else ~0)")
