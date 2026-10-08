"""convert_tables.py — order/transaction feather → parquet（行组 100 万、zstd）。

一次性转换；之后按月随机访问走行组下推（~20MB/月），避免每次整文件解压。
用法：python convert_tables.py [train|test] [order|transaction|all]
"""
import os
import sys
import time

import polars as pl

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "all"
    name = sys.argv[2] if len(sys.argv) > 2 else "all"
    splits = ["train", "test"] if split == "all" else [split]
    names = ["order", "transaction"] if name == "all" else [name]
    os.environ.setdefault("POLARS_MAX_THREADS", "4")
    for s in splits:
        for n in names:
            src = os.path.join(BASE, s, f"{n}.feather")
            dst = os.path.join(BASE, s, f"{n}.parquet")
            if not os.path.exists(src):
                print(f"{src} 不存在，跳过", flush=True)
                continue
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                print(f"{dst} 已存在，跳过", flush=True)
                continue
            t0 = time.time()
            print(f"{src} -> {dst} 开始（峰值内存 ~3GB，请勿并行重任务）", flush=True)
            (pl.scan_ipc(src)
               .sink_parquet(dst, row_group_size=1_000_000, compression="zstd"))
            sz = os.path.getsize(dst) / 1e9
            print(f"{dst} 完成 {time.time()-t0:.0f}s，{sz:.2f}GB", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
