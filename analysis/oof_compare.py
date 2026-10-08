"""oof_compare.py — 两个 OOF 的 A/B 对照：**逐折配对差 + 切片内检查**（不训练，几秒）。

为什么要有这个脚本（WALKTHROUGH10 §7.1 / §8 的判据落地）
--------------------------------------------------------
2026-09-23 的 `U_s` 尺度臂在 train 上"7/7 模型一致、跨 regime 迁移稳定、6 折留出 +0.00286"，
却在 **LB 上完全归零**。定位到的原因是：**它的收益 100% 来自切片之间的权重再分配**，
而不是"预测变准了"——在每一个 `U_s` 切片**内部**，重标定几乎不改变 cos。

所以凡是"重标定 / 加权 / 校准 / 任何只在集合层面生效"的改动，
**下结论前必须在切片内部量一遍**。本脚本把这一步做成标准动作。

**⚠️ 但它两个方向都会错——2026-09-23 与 2026-09-28 各给出一个反例**：

> 本检查算在 **OOF** 上，而 OOF 留出的是**月份**，所有月份**来自同一个市场**。
> 因此它只能排除"跨切片再分配"这一种伪收益，**不能证明跨市场迁移**。
>
> | 案例 | 切片内改善 | LB 结果 | 判对? |
> |---|---|---|---|
> | `U_s` 尺度臂 | **0/10** → 判定"再分配" | 0.133 → 0.133（归零）| ✅ |
> | Y3 样本加权 | 9/10 → 判定"真变准" | 0.091 → 0.092 | ✅ |
> | **MLP 的 cos 损失臂** | **8/10** → 判定"真变准" | 0.091 → **0.090** | ❌ **假阳性** |
> | **TCN 的 cos 损失臂** | **2/10** → 会被判"再分配" | 0.122 → **0.128**（全额兑现）| ❌ **假阴性** |
>
> **2/4，而且两个方向都错过。** 第二条尤其要记住：**本项目最强的一分（+0.006）恰好会被它筛掉**。
> 原因是那 +0.0069 由**跨 |y| 带的幅度轮廓**承载（按 |y| 分片的切片内检查把"轮廓变化"
> 一律读成"再分配"）。**该条规则因此不能单独用来否定一个改动**——
> 分层恒等式与三条已测边界见 `CLAUDE.md #15` 第 5 条、`MEMORY.md`「幅度轴」。
>
> ⇒ **它的价值在"筛掉"那一侧，但筛掉本身有成本。** 用它做**归因**（回答"这个改动动了什么"），
> **不要用它单独否定一个改动**——否定需要"归因 + 该机制在你手上确实失效过的证据"。

    整体涨 + 切片内**不动**  →  **纯再分配** → 直接不必花 LB 名额（**这条最有用**）
    整体**不涨**             →  无效 → 也不必花 LB 名额
    整体涨 + 切片内也涨      →  **只说明不是再分配；仍需过 LB**（CLAUDE.md #13 照旧适用）

用法
----
    python analysis/oof_compare.py --base snap_cache/oof_parts_mlp_crop128 \
                                   --new  snap_cache/oof_parts_mlp_yw99 \
                                   --name-base 基线(crop128) --name-new Y3(yw99)

    # 换切片依据（默认按 |y| 等频十分位；也可 --by pred / month）
    python analysis/oof_compare.py --base A --new B --by pred --bins 20

读：`$MSC_BASE/train/label.parquet` + 两个 OOF 目录（各自优先 `oof_all.parquet`，否则拼 `f*.parquet`）。
"""
import os
import glob
import argparse

import numpy as np
import polars as pl

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)


def hdr(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}", flush=True)


def cos_of(p, y):
    return float((p * y).sum() / np.sqrt((p ** 2).sum() * (y ** 2).sum()))


def qbin(x, n):
    q = np.quantile(x, np.linspace(0, 1, n + 1))
    q[-1] += 1.0
    return np.clip(np.searchsorted(q[1:-1], x), 0, n - 1).astype(np.int32)


def read_oof(d):
    """优先 oof_all.parquet，否则按 f*.parquet 拼。返回 (sample_id, pred, 逐折表)。"""
    d = d if os.path.isabs(d) else os.path.join(BASE, d)
    allp = os.path.join(d, "oof_all.parquet")
    parts = sorted(glob.glob(os.path.join(d, "f[0-9]*.parquet")))
    per_fold = {}
    for f in parts:
        per_fold[os.path.basename(f)[:-8]] = pl.read_parquet(f).select("sample_id", "pred")
    if os.path.exists(allp):
        return pl.read_parquet(allp).select("sample_id", "pred"), per_fold
    if not parts:
        raise FileNotFoundError(f"{d} 下没有 oof_all.parquet 也没有 f*.parquet")
    return pl.concat(list(per_fold.values())), per_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基线 OOF 目录")
    ap.add_argument("--new", required=True, help="新方案 OOF 目录")
    ap.add_argument("--name-base", default="base")
    ap.add_argument("--name-new", default="new")
    ap.add_argument("--by", choices=["abs_y", "pred", "month"], default="abs_y",
                    help="切片依据（abs_y = 按 |target| 等频分箱，**判据默认**）")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).select(
        "sample_id", "month", "target").sort("sample_id")
    ab, fb = read_oof(args.base)
    an, fn = read_oof(args.new)

    print(f"BASE {args.name_base}: {args.base}  ({ab.height:,} 行)")
    print(f"NEW  {args.name_new}: {args.new}  ({an.height:,} 行)")

    # —— 逐折配对差（项目纪律：单折/均值对比都要配配对差，CLAUDE.md #8）——
    hdr("1. 逐折 cos（配对；月交错 6 折）")
    folds = sorted(set(fb) & set(fn))
    if folds:
        db = fb[folds[0]].join(lab, on="sample_id", how="inner")
        dn = fn[folds[0]].join(lab, on="sample_id", how="inner")
        print(f"{'折':>4}{args.name_base:>16}{args.name_new:>16}{'配对差':>12}")
        diffs = []
        for f in folds:
            jb = fb[f].join(lab, on="sample_id", how="inner")
            jn = fn[f].join(lab, on="sample_id", how="inner")
            cb = cos_of(jb["pred"].to_numpy(), jb["target"].to_numpy())
            cn = cos_of(jn["pred"].to_numpy(), jn["target"].to_numpy())
            diffs.append(cn - cb)
            print(f"{f:>4}{cb:>16.5f}{cn:>16.5f}{cn - cb:>+12.5f}")
        diffs = np.array(diffs)
        print(f"{'均值':>4}{'':>16}{'':>16}{diffs.mean():>+12.5f}"
              f"   折间 std {diffs.std():.5f}｜{int((diffs > 0).sum())}/{len(diffs)} 折为正")
    else:
        print("（没有成对的 f*.parquet，跳过——只有 oof_all 时无法配对）")

    # —— 全量 cos ——
    hdr("2. 全量 cos（覆盖全部样本）")
    d = ab.join(an, on="sample_id", how="inner", suffix="_new").join(lab, on="sample_id")
    y = d["target"].to_numpy().astype(np.float64)
    pb = d["pred"].to_numpy().astype(np.float64)
    pn = d["pred_new"].to_numpy().astype(np.float64)
    cb, cn = cos_of(pb, y), cos_of(pn, y)
    print(f"  {args.name_base:<22}{cb:.5f}")
    print(f"  {args.name_new:<22}{cn:.5f}   Δ {cn - cb:+.5f}")
    print(f"  两预测相关 {np.corrcoef(pb, pn)[0, 1]:.5f}")

    # —— 切片内检查（本脚本的核心）——
    byname = {"abs_y": "按 |target| 等频分箱", "pred": "按 BASE 预测幅度等频分箱",
              "month": "按月份"}
    key = {"abs_y": np.abs(y), "pred": np.abs(pb), "month": d["month"].to_numpy().astype(float)}[args.by]
    k = qbin(key, args.bins) if args.by != "month" else \
        np.clip(key.astype(np.int32), 0, int(key.max())).astype(np.int32)
    hdr(f"3. 切片内检查 — {byname[args.by]}（{k.max() + 1} 片）")
    print("  ⚠️ 组内 cos 的变化**不能相加**还原总变化（全局 cos 不是组内 cos 的加权平均）——")
    print("     它回答的是另一个问题：**预测质量本身有没有变好**。\n")
    print(f"  {'片':>4}{'n':>10}{args.name_base:>12}{args.name_new:>12}{'Δ':>10}")
    up = 0
    for i in range(int(k.max()) + 1):
        s = k == i
        if s.sum() < 100:
            continue
        a, b = cos_of(pb[s], y[s]), cos_of(pn[s], y[s])
        up += (b > a)
        print(f"  {i:>4}{int(s.sum()):>10,}{a:>12.5f}{b:>12.5f}{b - a:>+10.5f}")
    n_slices = int(sum(1 for i in range(int(k.max()) + 1) if (k == i).sum() >= 100))
    print(f"\n  → {up}/{n_slices} 片内部改善")

    hdr("判读")
    print("  ⚠️ 本检查算在 OOF 上，而 OOF 留出的是**月份**（同一市场）——")
    print("     所以它只能**排除**『纯再分配』，**不能证明**跨市场迁移。")
    print("     cos 损失臂（2026-09-23）就是反例：8/10 片改善，LB 0.091 → 0.090。")
    print("  · 整体涨 + 切片内**不动** → **纯再分配** → **不必花 LB 名额**（这条最有用）")
    print("  · 整体**不涨**            → 无效 → 也不必花 LB 名额")
    print("  · 整体涨 + 切片内也涨     → **只说明不是再分配；仍需过 LB**（CLAUDE.md #13）")


if __name__ == "__main__":
    main()
