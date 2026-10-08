"""check_memory_build.py — 验证「内存构建」与「磁盘缓存」逐位一致。

动机：提交时改成现算 test 特征、不落盘（test 缓存只在生成提交时用，缓存"算一次复用多次"
的价值在 test 侧不成立，白占 6.1GB）。但**如果内存路径与当初建缓存时的路径有任何差异，
test 特征就会与 train 特征分布不一致**——这是不会报错的错：模型只是悄悄变差，不崩。

做法：在 **train** 上比对（磁盘缓存还在，test 的已删）。每条管线只用 `--months 0,1,2`
建到内存，再与磁盘缓存里**同样的行**逐位比较。

两条管线：snap（粗段 224×15）/ flow（细段 60×16）。（tri60 管线已于 2026-09-26 随 tri60 代码移除）
**比对的先决条件是磁盘缓存还在**——flow 的 `.dat` 已于 2026-09-16 按"现建到内存、不留盘"
的决定删除，所以现在**只有 snap 那条能跑**（缺 `.dat` 的管线会自动跳过并说明原因）。

用法：python models/cnn/check_memory_build.py
"""
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flow_feats      # noqa: E402
import snap_feats      # noqa: E402
from seq_common import BASE   # noqa: E402

MONTHS = "0,1,2"
N_MONTHS = 3


def _n_rows(n_months):
    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet"))
    counts = lab.group_by("month").len().sort("month")["len"].to_numpy()
    return int(counts[:n_months].sum())


def check(tag, mem, dat_path, n):
    disk = np.load(dat_path, mmap_mode="r")
    assert mem.shape == disk.shape, f"{tag} 形状不符 {mem.shape} vs {disk.shape}"
    a = np.asarray(mem[:n])
    b = np.asarray(disk[:n])
    n_bad = int((a != b).sum())
    ok = n_bad == 0
    print(f"  {'PASS' if ok else 'FAIL'}  {tag:8s} 前 {n:,} 行逐位比对"
          f"（{a.nbytes/2**20:.0f}MB）"
          + ("" if ok else f"  ← {n_bad} 个元素不同，最大差 "
             f"{np.abs(a.astype(np.float32)-b.astype(np.float32)).max():.6g}"))
    return ok


def main():
    n = _n_rows(N_MONTHS)
    print(f"验证内存构建 == 磁盘缓存（train 前 {N_MONTHS} 个月 = {n:,} 行）\n")
    ok = True

    # 注意签名不同：snap_feats.build_split 多一个 finalize 参数
    cases = (
        ("snap", lambda: snap_feats.build_split("train", MONTHS, False, False,
                                                to_memory=True),
         os.path.join(snap_feats.CACHE, "train_snap_X.dat")),
        ("flow", lambda: flow_feats.build_split("train", MONTHS, False, to_memory=True),
         os.path.join(flow_feats.CACHE, "train_flow_X.dat")),
    )
    for tag, build, dat in cases:
        if not os.path.exists(dat):
            # 磁盘缓存已删的管线无法比对——**这是正常状态**：flow 的 `.dat` 于 2026-09-16
            # 按"每次运行现建到内存"的决定删掉了（见 MEMORY / CLAUDE.md #4）。
            # 只保留 snap 那条能跑（它的 `.dat` 还在，被 cnn_baseline.py 直接读）。
            print(f"  [{tag}] 跳过：缺 {dat}（该管线已改为现建到内存、不留盘）")
            continue
        t0 = time.time()
        mem = build()
        print(f"  [{tag}] 内存构建 {time.time()-t0:.0f}s，{mem.nbytes/2**30:.2f}GB")
        ok &= check(tag, mem, dat, n)
        del mem

    print(f"\n{'全部一致' if ok else '存在不一致'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
