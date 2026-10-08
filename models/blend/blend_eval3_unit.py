"""blend_eval3_unit.py — 三方集成（unit() L2 归一混合法，LB142 思路自实现）。

unit(x) = (x − mean(x)) / ||x − mean(x)||₂：每个成员去均值+L2 归一，
等"能量"贡献后再加权。解决原始尺度下 β 权重追逐尺度差的问题。
pred = u_tabm + α·u_snap + β·u_lgbm，2D 网格逐月最优 → pooled（中位数）。
"""
import os
import sys

import numpy as np
import polars as pl

# 跨目录复用（models/tabm）——目录结构见 README
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tabm"))

from tabm_common import BASE, cos_score, load_train

OOF_DIR = os.path.join(BASE, "factors")
LGB_OOF = sys.argv[1] if len(sys.argv) > 1 else "lgb_newfeat"
TABM_OOF = sys.argv[2] if len(sys.argv) > 2 else "tabm"  # tabm / tabm2 / tabm2emb


def load_oof(name):
    return pl.read_parquet(os.path.join(OOF_DIR, f"{name}_oof.parquet"))


def unit(x: np.ndarray) -> np.ndarray:
    xc = x - x.mean()
    n = np.sqrt((xc ** 2).sum())
    return xc / n if n > 0 else np.zeros_like(xc)


def main():
    _, y, marr, months, _ = load_train()
    tabm = load_oof(TABM_OOF)
    snap = load_oof("snap")
    lgbm = load_oof(LGB_OOF)
    print(f"TabM 成员 OOF: {TABM_OOF} / LGBM 成员 OOF: {LGB_OOF}")
    merged = (tabm.rename({"pred": "p_tabm"})
              .join(snap.rename({"pred": "p_snap"}), on=["sample_id", "month"])
              .join(lgbm.rename({"pred": "p_lgbm"}), on=["sample_id", "month"]))
    vm = sorted(merged["month"].unique().to_list())

    # 全局 unit 归一（LB142 做法：每成员全样本去均值+L2 归一）
    merged = merged.with_columns([
        pl.Series("u_tabm", unit(merged["p_tabm"].to_numpy())),
        pl.Series("u_snap", unit(merged["p_snap"].to_numpy())),
        pl.Series("u_lgbm", unit(merged["p_lgbm"].to_numpy())),
    ])

    agrid = np.arange(0.0, 2.01, 0.25)
    bgrid = np.arange(0.0, 2.01, 0.25)
    print(f"{'月':>4} {'cos(α*,β*)':>10} {'α*':>6} {'β*':>6}")
    best_ab = []
    cos_star = []
    by_month = {}
    for m in vm:
        mm = merged.filter(pl.col("month") == m)
        t = mm["u_tabm"].to_numpy()
        s = mm["u_snap"].to_numpy()
        l_ = mm["u_lgbm"].to_numpy()
        mask = marr == m
        ym = y[mask]
        best = (0.0, 0.0, -1.0)
        for a in agrid:
            for b in bgrid:
                cc = cos_score(t + a * s + b * l_, ym)
                if cc > best[2]:
                    best = (a, b, cc)
        best_ab.append(best)
        cos_star.append(best[2])
        by_month[m] = (t, s, l_, ym)
        print(f"{m:>4} {best[2]:10.5f} {best[0]:6.3f} {best[1]:6.3f}")

    ab = np.array([(b[0], b[1]) for b in best_ab])
    pa, pb = float(np.median(ab[:, 0])), float(np.median(ab[:, 1]))
    print(f"\n逐月最优 α: median={np.median(ab[:,0]):.3f} std={ab[:,0].std():.3f}")
    print(f"逐月最优 β: median={np.median(ab[:,1]):.3f} std={ab[:,1].std():.3f}")

    cos_pooled = []
    for m in vm:
        t, s, l_, ym = by_month[m]
        cos_pooled.append(cos_score(t + pa * s + pb * l_, ym))
    print(f"\nunit 三方逐月最优 = {np.mean(cos_star):.5f}（上界）")
    print(f"unit 三方 pooled α={pa:.3f} β={pb:.3f} = {np.mean(cos_pooled):.5f}")
    print(f"参考：v7 二方 pooled 0.14272；原始尺度三方 pooled 0.14272")

    # —— 权重敏感性（pooled 口径，α×β 粗网格）——
    print("\n### 权重敏感性（12 月 mean cos）###")
    header = "a/b"
    print(f"{header:>8}" + "".join(f"{b:>9.3f}" for b in [0.0, 0.25, 0.375, 0.5, 1.0]))
    for a in [0.5, 0.75, 1.0]:
        line = f"{a:>8.3f}"
        for b in [0.0, 0.25, 0.375, 0.5, 1.0]:
            vals = []
            for m in vm:
                t, s, l_, ym = by_month[m]
                vals.append(cos_score(t + a * s + b * l_, ym))
            line += f"{np.mean(vals):>9.5f}"
        print(line)


if __name__ == "__main__":
    main()
