"""Deeper probe v2: target stats, per-sample time coverage, book structure.
Uses pyarrow Table (read_all, memory-mapped) group-by."""
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import numpy as np

import os
BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)

# --- label ---
label = pd.read_parquet(BASE + r"/train/label.parquet")
print("### label ###")
print("shape:", label.shape, "| months:", label.month.nunique())
print("target == 0 fraction:", (label.target == 0).mean().round(4))
print("distinct target values:", label.target.nunique())
print("top 8 most common target values:")
print(label.target.value_counts().head(8))
g = label.groupby("month").target.agg(["mean", "std", "count", "min", "max"])
print("\nper-month target stats (first 3 / last 3 months):")
print(pd.concat([g.head(3), g.tail(3)]).round(5))
m = label.groupby("month").sample_id.agg(["min", "max"])
print("\nper-month sample_id range (first 3 / last 3):")
print(pd.concat([m.head(3), m.tail(3)]))

# --- per-file coverage ---
def probe(rel, extra=None):
    path = BASE + "/" + rel
    with open(path, "rb") as f:
        reader = pq.ParquetFile(f)
        tbl = reader.read()
        sid = tbl.column("sample_id")
        n_rows, n_samples = tbl.num_rows, pc.count_distinct(sid).as_py()
        agg = tbl.group_by("sample_id").aggregate([
            ("seconds_before_predict", "min"),
            ("seconds_before_predict", "max"),
            ("seconds_before_predict", "count"),
        ])
        print(f"\n### {rel} ###  rows={n_rows:,} samples={n_samples:,} rows/sample={n_rows/n_samples:.1f}")
        for name in ("seconds_before_predict_min", "seconds_before_predict_max", "seconds_before_predict_count"):
            col = agg.column(name).to_pandas()
            print(f"  {name}: min={col.min():.2f} max={col.max():.2f} mean={col.mean():.2f} median={col.median():.2f}")
        if extra == "book":
            s0 = tbl.filter(pc.equal(sid, 0))
            sec = s0.column("seconds_before_predict").to_pandas()
            d = np.diff(np.sort(sec)).round(4)
            print("  sample0: n rows:", len(sec), "| sec range:", round(sec.min(), 1), "->", round(sec.max(), 1))
            print("  sample0: unique sec deltas:", sorted(set(d))[:6])
            ap1, bp1 = s0.column("ask_price_1").to_pandas(), s0.column("bid_price_1").to_pandas()
            sp = (ap1 - bp1)
            print("  sample0 spread(ask1-bid1): min=%.5f max=%.5f median=%.5f" % (sp.min(), sp.max(), sp.median()))
            av1, bv1 = s0.column("ask_volume_1").to_pandas(), s0.column("bid_volume_1").to_pandas()
            print("  sample0 ask1 vol median:", av1.median(), "| bid1 vol median:", bv1.median())
            print("  sample0 ask1 price unique:", s0.column("ask_price_1").unique().to_pylist()[:10])
        if extra == "side":
            side = tbl.column("side")
            print("  side value counts:", pc.value_counts(side).to_pandas().to_dict())
            if "order" in rel:
                oa = tbl.column("order_action")
                print("  order_action value counts:", pc.value_counts(oa).to_pandas().to_dict())
                print("  side x order_action crosstab:")
                df = tbl.select(["side", "order_action"]).to_pandas()
                print(pd.crosstab(df.side, df.order_action))

probe(r"train/market.parquet", extra="book")
probe(r"train/order.parquet", extra="side")
probe(r"train/transaction.parquet", extra="side")
probe(r"test/market.parquet", extra="book")
probe(r"test/order.parquet", extra="side")
probe(r"test/transaction.parquet", extra="side")

# --- do transaction events appear in order events? ---
print("\n### txn vs order overlap (train, sample 0) ###")
with open(BASE + r"/train/order.parquet", "rb") as f:
    o = pq.ParquetFile(f).read()
with open(BASE + r"/train/transaction.parquet", "rb") as f:
    t = pq.ParquetFile(f).read()
o0 = o.filter(pc.equal(o.column("sample_id"), 0)).to_pandas()
t0 = t.filter(pc.equal(t.column("sample_id"), 0)).to_pandas()
key_o = set(zip(o0.seconds_before_predict.round(3), o0.price.round(5), o0.volume, o0.side))
key_t = set(zip(t0.seconds_before_predict.round(3), t0.price.round(5), t0.volume, t0.side))
print(f"sample0: order events={len(o0)}, txn events={len(t0)}, txn keys found in order keys: {len(key_t & key_o)}/{len(key_t)}")
print("order rows with order_action==0:", (o0.order_action == 0).sum(), "| ==1:", (o0.order_action == 1).sum())
