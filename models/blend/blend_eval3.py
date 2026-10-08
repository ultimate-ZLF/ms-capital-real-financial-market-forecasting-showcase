"""blend_eval3.py — 三方 OOF 集成评估：TabM × snap-CNN × LGBM(58 特征)。

沿用 blend_eval.py 协议（v7 惯例）：pred = p_tabm + α·p_snap + β·p_lgbm
- 2D 网格（α,β ∈ [0,2] 步 0.25）逐月最优 → 稳定性检验（v3 教训：std 大 = 权重不可信）
- pooled (α,β) = 各月最优的中位数
- 输出三方相关矩阵 + 逐月表 + 集成 cos
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
    lgbm = load_oof("lgb_newfeat")
    assert tabm.height == snap.height == lgbm.height, \
        f"OOF 行数不一致: {tabm.height}/{snap.height}/{lgbm.height}"
    merged = (tabm.rename({"pred": "p_tabm"})
              .join(snap.rename({"pred": "p_snap"}), on=["sample_id", "month"])
              .join(lgbm.rename({"pred": "p_lgbm"}), on=["sample_id", "month"]))
    print(f"OOF 样本数 {merged.height}；月份 {sorted(merged['month'].unique().to_list())}")

    # 相关矩阵
    P = merged.select(["p_tabm", "p_snap", "p_lgbm"]).to_numpy()
    c = np.corrcoef(P.T)
    print(f"\nOOF 相关矩阵:\n  tabm x snap = {c[0,1]:.4f}\n  tabm x lgbm = {c[0,2]:.4f}"
          f"\n  snap x lgbm = {c[1,2]:.4f}")

    # —— 逐月：基准 cos + 2D 网格最优 (α,β) ——
    agrid = np.arange(0.0, 2.01, 0.25)
    bgrid = np.arange(0.0, 6.01, 0.5)   # β 上界 6（首轮发现 β 顶在 2.0 边界）
    vm = sorted(merged["month"].unique().to_list())
    print(f"\n{'月':>4} {'tabm':>8} {'snap':>8} {'lgbm':>8} {'α*':>6} {'β*':>6} {'cos(α*,β*)':>10} {'cos(pooled)':>11}")
    rows = {m: None for m in vm}
    best_ab = []
    cos_base = {"tabm": [], "snap": [], "lgbm": []}
    for m in vm:
        mm = merged.filter(pl.col("month") == m)
        t = mm["p_tabm"].to_numpy()
        s = mm["p_snap"].to_numpy()
        l_ = mm["p_lgbm"].to_numpy()
        mask = marr == m
        ym = y[mask]
        ct, cs, cl = cos_score(t, ym), cos_score(s, ym), cos_score(l_, ym)
        cos_base["tabm"].append(ct)
        cos_base["snap"].append(cs)
        cos_base["lgbm"].append(cl)
        best = (0.0, 0.0, -1.0)
        for a in agrid:
            for b in bgrid:
                cc = cos_score(t + a * s + b * l_, ym)
                if cc > best[2]:
                    best = (a, b, cc)
        best_ab.append(best)
        rows[m] = (ct, cs, cl, best)
    ab = np.array([(b[0], b[1]) for b in best_ab])
    pa, pb = float(np.median(ab[:, 0])), float(np.median(ab[:, 1]))
    print(f"逐月最优 α: median={np.median(ab[:,0]):.3f} std={ab[:,0].std():.3f} "
          f"范围 [{ab[:,0].min():.3f},{ab[:,0].max():.3f}]")
    print(f"逐月最优 β: median={np.median(ab[:,1]):.3f} std={ab[:,1].std():.3f} "
          f"范围 [{ab[:,1].min():.3f},{ab[:,1].max():.3f}]")
    print("稳定性判读（v3 教训）：std < ~0.3 且符号稳定 → pooled 可信；否则退等权")

    cos_pooled = []
    cos_star = []
    for m in vm:
        ct, cs, cl, best = rows[m]
        mm = merged.filter(pl.col("month") == m)
        t, s, l_ = (mm["p_tabm"].to_numpy(), mm["p_snap"].to_numpy(),
                    mm["p_lgbm"].to_numpy())
        mask = marr == m
        ym = y[mask]
        cos_star.append(best[2])
        cos_pooled.append(cos_score(t + pa * s + pb * l_, ym))
        print(f"{m:>4} {ct:8.5f} {cs:8.5f} {cl:8.5f} {best[0]:6.3f} {best[1]:6.3f} "
              f"{best[2]:10.5f} {cos_pooled[-1]:11.5f}")

    print(f"\ncos 汇总（{len(vm)} 月 mean）:")
    print(f"  tabm      = {np.mean(cos_base['tabm']):.5f}")
    print(f"  snap      = {np.mean(cos_base['snap']):.5f}")
    print(f"  lgbm      = {np.mean(cos_base['lgbm']):.5f}")
    print(f"  三方逐月最优 = {np.mean(cos_star):.5f}（上界参考）")
    print(f"  三方 pooled α={pa:.3f} β={pb:.3f} = {np.mean(cos_pooled):.5f}")
    print(f"  参考：v7 二方 pooled 0.14272")
    np.save(os.path.join(OOF_DIR, "blend3_alpha.npy"), np.array([pa, pb]))


if __name__ == "__main__":
    main()
