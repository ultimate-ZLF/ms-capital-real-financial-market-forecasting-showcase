"""check_flow_signal.py — flow 缓存的**函数级**体检（E1 的闸门 0）。

为什么需要它：`check_cache.py` 只查**结构**（形状/行序/有限性/跨 regime 中位），
**不查"这一列能不能预测 y"**。而 CLAUDE.md 工程教训 #4 记着一次事故：
`--resume` 在缺 `.dat` 时静默产出**全零**数据集——训练照跑、断言全过、训出来是常数预测，
报错里什么都看不出来。

后果有多严重：如果缓存坏了而没查出来，flow_baseline 会跑出 ≈0 的 cos，
而我们很可能把它读成 **"flow 冗余" —— 正好是与事实相反的结论**。

做法：
  1. **结构硬检查**：形状、有限性、每通道非零率、每通道 std、空 bin 占比；
  2. **信号软检查**：几个"必然会预测 y"的简单聚合量与 target 的 cos，与 `FACTORS.md`
     的因子表对照。

⚠️ 为什么信号检查只能要求"同量级"，不能对齐到小数点：
flow 缓存的 `t_imb` 在**空 bin 上写 0**（`np.divide(..., where=den>0)`，见 flow_feats.py），
而因子表的 `t_imb_2s` 在没有成交的样本上是 **NaN——整个样本被丢掉**。零填充会**拉低** cos
（p=0 的样本对分子没贡献，但它们的 target 方差仍在分母里）。所以缓存版的 cos 应当**略低于**
因子表的 0.0733，低多少取决于空 bin 比例。

用法（服务器）：
    python models/cnn/check_flow_signal.py                    # 现建 7 个月（跨年取样，约 15 秒）
    python models/cnn/check_flow_signal.py --from-disk        # 读 .dat，随机抽 10 万行
    python models/cnn/check_flow_signal.py --months 0,1,2     # 指定月份
"""
import argparse
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flow_feats  # noqa: E402  提供 build_split / IDX / CACHE
from seq_common import BASE  # noqa: E402

# 参照值（FACTORS.md 的 71 月单因子表；本脚本的信号是"缓存版"的粗略对应物，只要求同量级）
REF = [
    ("t_imb 最后 2 格均值", "t_imb_2s", 0.0733, "⚠️ 零填充，应略低"),
    ("t_imb 全窗均值", "t_imb（60s 等权）", 0.0541, "统计量不同（逐秒比值的均值 vs 量加权），量级参照"),
    ("x_to_buy_diff 全窗均值", "x_to_buy_diff", 0.0214, ""),
]


def cos(a, b):
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main():
    p = argparse.ArgumentParser(description="flow 缓存的函数级体检（E1 闸门 0）")
    p.add_argument("--months", default="0,12,24,36,48,60,70",
                   help="现建模式下取哪些月（跨 regime 取样，比连取 0-6 月更有代表性）")
    p.add_argument("--from-disk", action="store_true",
                   help="读 flow_cache/train_flow_X.dat——⚠️ 该缓存 2026-09-16 已按设计删除，"
                        "本开关只剩 FileNotFoundError 一条路（保留作警示）")
    p.add_argument("--rows", type=int, default=100_000,
                   help="⚠️ 只在 --from-disk（已死）分支生效；默认内存路径下无效")
    p.add_argument("--seed", type=int, default=0,
                   help="⚠️ 同上：只在 --from-disk（已死）分支生效")
    args = p.parse_args()

    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).sort("sample_id")
    y_all = lab["target"].to_numpy()
    marr = lab["month"].to_numpy()
    t0 = time.time()

    if args.from_disk:
        path = os.path.join(flow_feats.CACHE, "train_flow_X.dat")
        X = np.load(path, mmap_mode="r")
        assert X.shape == (lab.height, flow_feats.FINE_T, flow_feats.K), f"{path} 形状 {X.shape}"
        rng = np.random.default_rng(args.seed)
        rows = np.sort(rng.choice(X.shape[0], size=min(args.rows, X.shape[0]), replace=False))
        print(f"[disk] {path} {X.shape} f16，随机抽 {rows.size} 行")
    else:
        sel = [int(x) for x in args.months.split(",")]
        X = flow_feats.build_split("train", args.months, False, to_memory=True)
        rows = np.where(np.isin(marr, sel))[0]
        print(f"[memory] 现建 {sel} 共 {rows.size} 行（{time.time()-t0:.1f}s）")

    a = np.asarray(X[rows], dtype=np.float32)      # (n,60,16)
    y = y_all[rows]
    n = a.shape[0]
    print(f"取数完成 {a.shape}（{time.time()-t0:.1f}s，占用约 {a.nbytes/2**20:.0f}MB）\n")

    # ---------------- 1. 结构硬检查 ----------------
    ok = True
    if not np.isfinite(a).all():
        ok = False
        print(f"  FAIL  含 NaN/inf：{int((~np.isfinite(a)).sum())} 个")
    else:
        print("  PASS  有限性：无 NaN/inf")

    has = a[:, :, flow_feats.IDX["has_event"]] > 0
    print(f"  空 bin 占比 {100 * (~has).mean():.1f}%（flow 侧事件稀疏，理应有一批空格）")
    print("\n  通道              非零率   std      （非零率 ≈0 或 std=0 说明该列是死的）")
    dead = []
    for i, nm in enumerate(flow_feats.FEAT_NAMES):
        v = a[:, :, i].reshape(-1)
        nz = float((v != 0).mean())
        sd = float(v.std())
        if sd == 0.0:
            dead.append(nm)
            print(f"  {nm:16s} {100*nz:6.1f}%  {sd:8.5f}  ← **整列常数**")
    if dead:
        ok = False
        print(f"  FAIL  有整列常数：{dead}")
    else:
        print("  PASS  16 个通道都有变化（无一列是常数）")

    # ---------------- 2. 信号软检查 ----------------
    imb = a[:, :, flow_feats.IDX["t_imb"]]
    xdiff = a[:, :, flow_feats.IDX["x_to_buy_diff"]]
    sigs = [imb[:, -2:].mean(1), imb.mean(1), xdiff.mean(1)]

    assert lab.height == X.shape[0], "行数对不上"
    print(f"\n  信号（与 target 的 cos，n={n:,}）")
    passes = []
    for (name, ref_name, ref, note), s in zip(REF, sigs):
        c = cos(s, y)
        good = c >= 0.6 * ref
        passes.append(good)
        print(f"    {'PASS' if good else 'FAIL'}  {name:22s} cos={c:+.5f}   "
              f"参照 {ref_name} {ref:.4f}  {note}")

    # ---------------- 结论 ----------------
    print()
    if not ok:
        print("### 结论：**结构检查未过**——先修缓存，不要拿它跑 E1。")
    elif passes[0]:
        print("### 结论：缓存是活的，可以用它跑 E1。")
    elif all(c > 0.01 for c in [cos(s, y) for s in sigs]):
        print("### 结论：信号偏弱但非零——先看空 bin 占比与月份取样（7 个月的月间波动 "
              "σ≈0.013），必要时用 --months 多取几个月再判。")
    else:
        print("### 结论：**信号≈0 —— 高度可疑**。不要跑 E1：那会把'缓存坏'读成'flow 冗余'，"
              "结论正好反掉。先查行序（label 是否按 sample_id 升序）与缓存重建过程。")


if __name__ == "__main__":
    main()
