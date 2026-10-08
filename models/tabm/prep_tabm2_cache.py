"""prep_tabm2_cache.py — 把 tabm2 的 X/Xt 从 parquet 转成磁盘 memmap 缓存（一次性）。

动机（避雷 #15 实战）：本机长期内存超承诺（多个常驻进程 + 无页面文件），
python 进程峰值过 ~3GB 就会被系统击杀。np.memmap 是文件映射、不占 commit 额度，
X(0.75GB)+Xt(0.39GB) 退出 commit → 进程峰值降 ~1.1GB。

缓存布局（factors/tabm2_cache/）：
- X.dat / meta.npz（y, marr, months, sids, us, shape）——train 全量 150 特征 f32
- Xt.dat / test_meta.npz（sids, us, shape）——test 全量

注意：因子缓存（factors_m*/new*_m* 等）有变化时必须重跑本脚本。
"""
import os
import sys

import numpy as np

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
sys.path.insert(0, BASE)

from tabm2_common import OUT, load_test, load_train  # noqa: E402

CACHE_DIR = os.path.join(OUT, "tabm2_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def main():
    X, y, marr, months, sids, us = load_train()
    Xm = np.memmap(os.path.join(CACHE_DIR, "X.dat"), dtype=np.float32,
                   mode="w+", shape=X.shape)
    Xm[:] = X
    Xm.flush()
    del Xm
    np.savez(os.path.join(CACHE_DIR, "meta.npz"),
             y=y, marr=marr, months=np.array(months), sids=sids, us=us,
             shape=np.array(X.shape))
    print(f"train 缓存完成：X {X.shape}，NaN 占比 {float(np.isnan(X).mean()):.4f}")
    del X, y, marr, sids, us

    Xt, test_ids, us_test = load_test()
    Xtm = np.memmap(os.path.join(CACHE_DIR, "Xt.dat"), dtype=np.float32,
                    mode="w+", shape=Xt.shape)
    Xtm[:] = Xt
    Xtm.flush()
    del Xtm
    np.savez(os.path.join(CACHE_DIR, "test_meta.npz"),
             sids=test_ids.to_numpy(), us=us_test, shape=np.array(Xt.shape))
    print(f"test 缓存完成：Xt {Xt.shape}")


if __name__ == "__main__":
    main()
