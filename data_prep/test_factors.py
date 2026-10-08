"""计算 test 集因子（复用 factors.compute_batch），分块处理。"""
import os
import polars as pl
from factors import compute_batch

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
OUT = os.path.join(BASE, "factors")

print("loading test files...", flush=True)
mt = pl.read_parquet(os.path.join(BASE, r"test/market.parquet"))
ot = pl.read_parquet(os.path.join(BASE, r"test/order.parquet"))
tt = pl.read_parquet(os.path.join(BASE, r"test/transaction.parquet"))
sub = pl.read_csv(os.path.join(BASE, "submissions", "submission.csv"))
ids_all = sub["sample_id"].to_list()
print(f"loaded, test samples: {len(ids_all)}", flush=True)

CHUNK = 100_000
parts = []
for i in range(0, len(ids_all), CHUNK):
    ids = ids_all[i:i + CHUNK]
    print(f"[chunk {i // CHUNK}] n={len(ids)}", flush=True)
    parts.append(compute_batch(ids, mt, ot, tt))

df = pl.concat(parts)
out = os.path.join(OUT, "factors_test.parquet")
df.write_parquet(out)
print(f"saved {df.height} rows -> {out}")
