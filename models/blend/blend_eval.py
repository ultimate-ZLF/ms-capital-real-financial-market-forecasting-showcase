"""OOF 集成评估：TabM × snap-CNN 加权（LGBM 可选，需先生成其 OOF）。

指标 cos 尺度无关 → 集成 = 归一化后加权：pred = p_tabm + α·p_cnn（α 网格拟合）。
v3 教训：拟合权重会跨 regime 衰减 → 用逐月稳定性检验 α 是否可信：
各月最优 α 的 std 大 = 不可信（退化成等权/定比）；std 小且符号一致 = 可用 pooled α。

输出：
- 各模型逐月 cos 对比表
- 最优 α（pooled）与逐月 α 分布
- 集成逐月 cos（pooled α 与逐月最优 α 两个口径）
"""
import os

import numpy as np
import polars as pl

# 跨目录复用（models/tabm）——目录结构见 README
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tabm"))

from tabm_common import BASE, cos_score, load_train

OOF_DIR = os.path.join(BASE, "factors")


def load_oof(name):
    return pl.read_parquet(os.path.join(OOF_DIR, f"{name}_oof.parquet"))


def main():
    _, y, marr, months, _ = load_train()
    tabm = load_oof("tabm")
    snap = load_oof("snap")
    assert tabm.height == snap.height
    merged = (tabm.rename({"pred": "p_tabm"})
              .join(snap.rename({"pred": "p_snap"}), on="sample_id"))
    print(f"OOF 样本数 {merged.height}；两模型 OOF 相关 = "
          f"{np.corrcoef(merged['p_tabm'], merged['p_snap'])[0, 1]:.4f}")

    # —— 逐月 cos 基准 ——
    print(f"\n{'月':>4} {'tabm':>8} {'snap':>8} {'α*':>8} {'cos(α*)':>8} {'cos(pooled)':>10}")
    alpha_grid = np.linspace(0.0, 2.0, 81)   # α = snap 相对 tabm 的权重比
    alphas, cos_base_t, cos_base_s, cos_star, cos_pooled = [], [], [], [], []
    by_month = {}
    for m in sorted(merged["month"].unique().to_list()):
        mm = merged.filter(pl.col("month") == m)
        t = mm["p_tabm"].to_numpy()
        s = mm["p_snap"].to_numpy()
        mask = marr == m
        ct, cs = cos_score(t, y[mask]), cos_score(s, y[mask])
        cos_base_t.append(ct)
        cos_base_s.append(cs)
        # 逐月最优 α（cos 对 α 是拟凹的，网格足够）
        best_a, best_c = 0.0, -1.0
        for a in alpha_grid:
            c = cos_score(t + a * s, y[mask])
            if c > best_c:
                best_a, best_c = a, c
        alphas.append(best_a)
        cos_star.append(best_c)
        by_month[m] = (t, s, y[mask])

    alphas = np.array(alphas)
    pooled_a = float(np.median(alphas))
    for m, a, ct, cs, cstar in zip(sorted(merged["month"].unique().to_list()),
                                   alphas, cos_base_t, cos_base_s, cos_star):
        t, s, ym = by_month[m]
        cp = cos_score(t + pooled_a * s, ym)
        cos_pooled.append(cp)
        print(f"{m:>4} {ct:8.5f} {cs:8.5f} {a:8.3f} {cstar:8.5f} {cp:10.5f}")

    print(f"\n逐月 α 最优：median={np.median(alphas):.3f} std={alphas.std():.3f} "
          f"范围 [{alphas.min():.3f}, {alphas.max():.3f}]")
    print("α 稳定性判读：std < ~0.3 且全为正 → pooled α 可信；否则等权更稳（v3 教训）")
    print(f"\ncos 汇总（12 月 mean）:")
    print(f"  tabm     = {np.mean(cos_base_t):.5f}")
    print(f"  snap     = {np.mean(cos_base_s):.5f}")
    print(f"  集成逐月最优 α = {np.mean(cos_star):.5f}（上界参考）")
    print(f"  集成 pooled α={pooled_a:.3f} = {np.mean(cos_pooled):.5f}")
    np.save(os.path.join(OOF_DIR, "blend_alpha.npy"),
            np.array([pooled_a, alphas.std()]))


if __name__ == "__main__":
    main()
