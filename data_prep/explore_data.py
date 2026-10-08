"""Explore MSCapital data: schema + first rows of each parquet file (no full load)."""
import os
import pyarrow.parquet as pq
import pandas as pd

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)

files = [
    "train/label.parquet",
    "train/market.parquet",
    "train/order.parquet",
    "train/transaction.parquet",
    "test/market.parquet",
    "test/order.parquet",
    "test/transaction.parquet",
]

for rel in files:
    path = os.path.join(BASE, rel)
    size_mb = os.path.getsize(path) / 1e6
    print("=" * 80)
    print(f"FILE: {rel}  ({size_mb:.1f} MB)")
    with open(path, "rb") as f:
        reader = pq.ParquetFile(f)
        schema = reader.schema_arrow
        print(f"num_row_groups: {reader.metadata.num_row_groups}")
        print("SCHEMA:")
        for i, field in enumerate(schema):
            print(f"  [{i}] {field.name}: {field.type}")
        # row group 0 = first rows
        batch = reader.read_row_group(0)
        print(f"first batch rows: {batch.num_rows}")
        df = batch.to_pandas()
        print("HEAD:")
        print(df.head(10).to_string(max_colwidth=30))
        print("DTYPES:")
        print(df.dtypes)

print("=" * 80)
print("submission.csv")
sub = pd.read_csv(os.path.join(BASE, "submissions", "submission.csv"))
print(sub.head(10).to_string(max_colwidth=30))
print(sub.dtypes)
print(f"rows: {len(sub)}")

print("=" * 80)
print("label.parquet FULL (it's small)")
label = pd.read_parquet(os.path.join(BASE, "train/label.parquet"))
print(label.shape)
print(label.head(10).to_string(max_colwidth=30))
print(label.dtypes)
print("nulls:", label.isnull().sum().sum())
print("unique count per col:")
for c in label.columns:
    print(f"  {c}: {label[c].nunique()}")
