"""check_cache.py — 两份序列缓存的体检（拼表产物验收）。

检查项：
1. 形状 / dtype / 行序（与 label、submission 的 sample_id 严格一致）
2. 抽样读到内存后无 NaN/inf（memmap 里写坏会静默传播）
3. 逐通道 train/test 统计量对照——**这是跨 regime 对齐的证据**：
   比值类应几乎重合；量级类经 per-sample 归一后也应重合
4. 粗段 dt 通道的分布与陈旧率

用法：python check_cache.py

⚠️ 缺缓存时**优雅跳过**（2026-09-21）：`test_snap_X.dat` 已按设计删除 → 自动跳过 test 粗段体检、
只打 train 中位；**归档 OOF 重算与细段 index 检查照常执行**（原先会崩在 test 上，
把这两项一起带掉——MEMORY 里"逐折 cos 用本脚本重算"那条命令因此失效过）。
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seq_common import (CACHE as SNAP_CACHE, F, FEAT_NAMES, T)   # noqa: E402

import flow_feats                                                     # noqa: E402

BASE = flow_feats.BASE
N_SAMPLE = 20_000


def _rows(n):
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(n, size=min(N_SAMPLE, n), replace=False))


def check_coarse(split):
    dat = os.path.join(SNAP_CACHE, f"{split}_snap_X.dat")
    X = np.load(dat, mmap_mode="r")
    idx = pl.read_parquet(os.path.join(SNAP_CACHE, f"{split}_snap_index.parquet"))
    lab = (pl.read_parquet(os.path.join(BASE, "train/label.parquet")).sort("sample_id")
           if split == "train" else
           pl.read_csv(os.path.join(BASE, "submissions/submission.csv"),
                       columns=["sample_id"]).sort("sample_id"))
    n_expect = flow_feats.SPLIT_N[split]
    assert X.shape == (n_expect, T, F), f"{split} 形状 {X.shape} != {(n_expect, T, F)}"
    assert (idx["sample_id"].to_numpy() == lab["sample_id"].to_numpy()).all(), \
        f"{split} index 与 label/submission 行序不一致"
    a = np.asarray(X[_rows(X.shape[0])], dtype=np.float32)
    assert np.isfinite(a).all(), f"{split} 抽样含 NaN/inf"
    flat = a.reshape(-1, F)
    nv = idx["n_valid"].to_numpy()
    dt = np.asarray(X[:, :, FEAT_NAMES.index("dt")], dtype=np.float32)
    print(f"\n[{split}] 粗段 {X.shape} {X.dtype}  n_valid 中位 {int(np.median(nv))} "
          f"(min {nv.min()} / max {nv.max()})")
    print(f"  抽样非有限值 0；dt 中位 {np.median(dt):.3f}  "
          f"dt>0.5（陈旧≥15s）占比 {100*np.mean(dt > 0.5):.2f}%  "
          f"dt=1.0 占比 {100*np.mean(dt >= 1.0):.2f}%")
    return np.median(flat, axis=0)


def archived_oof_reference():
    """归档 OOF（旧 14 通道模型的 12 折预测）逐折 cos —— 新模型的 A/B 参照。

    不引用文档里四舍五入过的 CV 数字（WALKTHROUGH4 写 0.111 等），
    直接重算旧模型亲手产出的预测，这样"新模型有没有变好"的判据不依赖任何转述。

    目录：`oof_parts_v1_14ch/`（CLAUDE.md 记载的正式备份，**唯一有效的 CNN 参照**）。
    """
    d = os.path.join(SNAP_CACHE, "oof_parts_v1_14ch")
    if not (os.path.isdir(d) and any(f.endswith(".parquet") for f in os.listdir(d))):
        print(f"\n[warn] 缺 {d}，跳过归档 OOF 参照")
        return
    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).select(["sample_id", "target"])
    files = sorted([f for f in os.listdir(d) if f.endswith(".parquet")],
                   key=lambda x: int("".join(c for c in x if c.isdigit())))
    print(f"\n归档 OOF 逐折 cos（旧 14 通道模型，{d}）：")
    vals = []
    for f in files:
        p = pl.read_parquet(os.path.join(d, f))
        m = p.join(lab, on="sample_id", how="inner")
        pred = m["pred"].to_numpy().astype(np.float64)
        tgt = m["target"].to_numpy().astype(np.float64)
        cos = float((pred * tgt).sum() / np.sqrt((pred ** 2).sum() * (tgt ** 2).sum()))
        vals.append(cos)
        print(f"  验证月 {int(m['month'][0]):3d}  n={m.height:6d}  cos={cos:.5f}")
    if vals:
        print(f"  12 折平均（未加权）= {np.mean(vals):.5f}")


def main():
    # ⚠️ test 的粗段 .dat 自 2026-09-16 起已删（提交时现算到内存）→ 优雅跳过，
    # 不让它把后面的归档 OOF 重算与细段 index 检查一起带崩
    # （MEMORY「旧 12 折归档」那条 cos 重算命令靠的就是本脚本）。
    medians = {}
    for s in ("train", "test"):
        if not os.path.exists(os.path.join(SNAP_CACHE, f"{s}_snap_X.dat")):
            print(f"\n[warn] 缺 {s}_snap_X.dat（粗段 test 缓存按设计不落盘），跳过 {s} 粗段体检")
            continue
        medians[s] = check_coarse(s)
    archived_oof_reference()
    # 细段 index 的行序一致性（细段 .dat 无其他元数据，靠 index 对齐）
    for split in ("train", "test"):
        p = os.path.join(flow_feats.CACHE, f"{split}_flow_index.parquet")
        if not os.path.exists(p):
            print(f"[warn] 缺 {p}，跳过 {split} 细段 index 检查")
            continue
        fi = pl.read_parquet(p)
        lab = (pl.read_parquet(os.path.join(BASE, "train/label.parquet")).sort("sample_id")
               if split == "train" else
               pl.read_csv(os.path.join(BASE, "submissions/submission.csv"),
                           columns=["sample_id"]).sort("sample_id"))
        assert fi.height == flow_feats.SPLIT_N[split]
        assert (fi["sample_id"].to_numpy() == lab["sample_id"].to_numpy()).all(), \
            f"{split} 细段 index 与 label/submission 行序不一致"
        print(f"[{split}] 细段 index 行序 OK（{fi.height:,} 行），"
              f"事件数中位 order {int(fi['o_n'].median())} / tx {int(fi['t_n'].median())}")
    if "test" in medians:
        print("\n  粗段通道            train中位    test中位   比值")
        for i, nm in enumerate(FEAT_NAMES):
            tr, te = medians["train"][i], medians["test"][i]
            r = te / tr if abs(tr) > 1e-9 else float("nan")
            print(f"  {nm:16s} {tr:+10.4f} {te:+10.4f}   {r:6.3f}")
        print("\n（比值接近 1 = 该通道跨 regime 对齐良好；dt 本身是时间结构，比值无意义）")
    else:
        print("\n  粗段通道            train中位")
        for i, nm in enumerate(FEAT_NAMES):
            print(f"  {nm:16s} {medians['train'][i]:+10.4f}")
        print("\n（缺 test_snap_X.dat → 只打印 train 中位；跨 regime 比值需该缓存）")


if __name__ == "__main__":
    main()
