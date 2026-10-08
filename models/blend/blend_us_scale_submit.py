"""blend_us_scale_submit.py — 把被 tick 单位化除掉的 **U_s 尺度**按幂律加回提交（2026-09-23）。

背景（完整推导见 **WALKTHROUGH10.md**；假设登记 HYPOTHESES **C-18**）
--------------------------------------------------------------------
tick 单位化把 42 个 tick 列 ÷U_s —— 它修好了迁移（TabM 衰减 0.797→0.849），代价是
**U_s 本身在输入里彻底消失**（15 条粗段通道逐条审计过：`mid_dev`/`dmid`/`spread_n`/
`micro_off`/`dpx` 全是分母，其余是比值/相对量/标记，没有一条携带 U_s）。

而全局 cos 的每个样本权重 ∝|y|，要求模型把 Σp² 按样本尺度分配 —— 尺度信息没了，
模型的 |p| 几乎不随 |y| 变（1.9× vs 26×）。

实测（双塔CNN OOF，1,257,637 样本）：

    p · U_s^w      w* = +0.245  →  cos 0.12028 → 0.12316   (+0.00288)
    月交错 6 折留出（每折单独选 w）              +0.00286     ← 过拟合风险 ≈ 0
    2 参数版（加 log²U 项）→ w2 = 0 ⇒ 是纯幂律
    跨 7 个模型 w* = 0.175~0.550（**全为正**）→ 是数据/指标的性质，不是单个模型的偶然

两个变体（差别是**可判读的**，见下）：

    all  : pred = blend_v8 × U_s^w              （w 默认 0.20）
    cnn  : pred = unit(tabm2emb) + 0.5·unit(cnn × U_s^0.245) + 0.25·unit(lgb_v5)

⚠️ **可判读性**：`lgb_v5` 的输入是**原始因子**（`spread0`/`spread_mean60` 未归一化）→ 它
   本来就有尺度信息；而 `tabm2emb` 的 spread 列在 `TABM2 TICK_FEATS` 里
   （`models/tabm/tabm2_common.py:53`）→ 与 CNN 同病。于是：

    · `cnn` 有效、`all` 不明显 → 缺尺度的是**序列主干**
    · `all` 明显更好           → 缺尺度的还包括 **tabm2emb**
    · 两者都无效               → 这个机制在 test 上不成立（"train 拟合结构又输一次"）

用法
----
    python models/blend/blend_us_scale_submit.py                  # 出两个变体
    python models/blend/blend_us_scale_submit.py --w-all 0.15     # 改 all 变体的指数
    python models/blend/blend_us_scale_submit.py --train-check    # 先在 train OOF 上复算证据
    python models/blend/blend_us_scale_submit.py --only all       # 只出一个

⚠️ 与 `blend3_submit.py` 一样：本脚本**只写新文件名**，不覆盖任何历史提交。
"""
import os
import sys
import argparse

import numpy as np
import polars as pl

# 跨目录复用（models/tabm）——目录结构见 README
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tabm"))

from tabm_common import BASE

SUB_DIR = os.path.join(BASE, "submissions")
MEMBERS = {
    "tabm2emb": "submission_tabm2emb_v2.parquet",
    "cnn": "submission_twotower_cnn.parquet",
    "lgb": "submission_lgb_v5.parquet",
}
ALPHA, BETA = 0.5, 0.25                      # v21/v8 配方（与 blend3_submit.py 一致）
BASE_SUB = "submission_blend_v8.parquet"
TICK_FALLBACK_TEST = 0.0012257               # 与 tabm2_common.TICK_FALLBACK["test"] 同值
W_CNN = 0.245                                # 双塔CNN OOF 上拟合出来的指数


def unit(x: np.ndarray) -> np.ndarray:
    """去均值 + L2 归一（yangq369 的 unit()，与 blend3_submit.py 逐字一致）。"""
    xc = x - x.mean()
    n = np.sqrt((xc ** 2).sum())
    return xc / n if n > 0 else np.zeros_like(xc)


def load_sub(name: str) -> np.ndarray:
    return pl.read_parquet(os.path.join(SUB_DIR, name)).sort("sample_id")["prediction"].to_numpy()


def load_test_us(sids: np.ndarray) -> np.ndarray:
    """test 每样本的 U_s（per-sample 正 spread 中位数 ≈ 1 tick）。test 时**可得**、零拟合。"""
    p = os.path.join(BASE, "snap_cache/test_snap_index.parquet")
    idx = pl.read_parquet(p).select("sample_id", "U_s").sort("sample_id")
    assert idx.height == sids.size, f"test index 行数 {idx.height} != 提交行数 {sids.size}"
    assert (idx["sample_id"].to_numpy() == sids).all(), "test index 与提交文件的 sample_id 行序不一致"
    us = idx["U_s"].fill_null(TICK_FALLBACK_TEST).to_numpy().astype(np.float64)
    n_null = int(idx["U_s"].is_null().sum())
    if n_null:
        print(f"⚠️ test U_s 有 {n_null} 个 null（{100 * n_null / idx.height:.3f}%）→ 用 tick 常数兜底")
    assert np.isfinite(us).all() and (us > 0).all(), "U_s 有非正/非有限值"
    return us


def report(tag: str, pred: np.ndarray) -> None:
    print(f"  {tag:<34} std={pred.std():.5f} min={pred.min():+.5f} max={pred.max():+.5f}")


def train_check(w_cnn: float) -> None:
    """在 train 上复算 U_s 幂律的证据（需要 train index + 双塔CNN OOF + label）。"""
    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).select(
        "month", "sample_id", "target").sort("sample_id")
    idx = pl.read_parquet(os.path.join(BASE, "snap_cache/train_snap_index.parquet")).select(
        "sample_id", "U_s").sort("sample_id")
    oof = pl.read_parquet(os.path.join(BASE, "snap_cache/oof_parts_twotower/oof_all.parquet")).select(
        "sample_id", "pred").sort("sample_id")
    assert lab.height == idx.height == oof.height, "train 三份数据行数不一致"
    assert (lab["sample_id"].to_numpy() == idx["sample_id"].to_numpy()).all()
    assert (lab["sample_id"].to_numpy() == oof["sample_id"].to_numpy()).all()

    y = lab["target"].to_numpy().astype(np.float64)
    mo = lab["month"].to_numpy()
    p = oof["pred"].to_numpy().astype(np.float64)
    lu = np.log(idx["U_s"].fill_null(0.0015037).to_numpy().astype(np.float64))
    lu -= lu.mean()
    Y2 = (y ** 2).sum()

    def cos_w(w):
        q = p * np.exp(w * lu)
        return (q * y).sum() / np.sqrt((q ** 2).sum() * Y2)

    c0 = cos_w(0.0)
    grid = np.linspace(-1.0, 1.0, 401)
    c = np.array([cos_w(w) for w in grid])
    print(f"双塔CNN OOF： w*={grid[c.argmax()]:+.3f}  cos {c0:.5f} → {c.max():.5f} "
          f"({c.max() - c0:+.5f})")

    # 月交错 6 折留出（每折单独选 w、在留出折上评）——这才是有说服力的那个数
    oof_pred = np.zeros_like(p)
    ws = []
    for k in range(6):
        tr, te = (mo % 6 != k), (mo % 6 == k)
        cs = np.array([((p[tr] * np.exp(w * lu[tr])) * y[tr]).sum()
                       / np.sqrt(((p[tr] * np.exp(w * lu[tr])) ** 2).sum() * (y[tr] ** 2).sum())
                       for w in grid])
        w_ = grid[cs.argmax()]
        ws.append(w_)
        oof_pred[te] = p[te] * np.exp(w_ * lu[te])
    c_oof = (oof_pred * y).sum() / np.sqrt((oof_pred ** 2).sum() * Y2)
    print(f"  月交错 6 折留出（逐折选 w，{['%+.3f' % w for w in ws]}）： "
          f"cos {c_oof:.5f}  ({c_oof - c0:+.5f})")
    print(f"  本脚本用的 w_cnn = {w_cnn}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-all", type=float, default=0.20, help="all 变体的指数（默认 0.20）")
    ap.add_argument("--w-cnn", type=float, default=W_CNN, help=f"cnn 变体的指数（默认 {W_CNN}）")
    ap.add_argument("--only", choices=["all", "cnn", "both"], default="both")
    ap.add_argument("--train-check", action="store_true", help="先在 train OOF 上复算证据")
    args = ap.parse_args()

    if args.train_check:
        print("=== train 侧证据复算 ===")
        train_check(args.w_cnn)
        print()

    pt = load_sub(MEMBERS["tabm2emb"])
    ps = load_sub(MEMBERS["cnn"])
    pl_ = load_sub(MEMBERS["lgb"])
    ref = load_sub(BASE_SUB)
    assert pt.shape == ps.shape == pl_.shape == ref.shape, "test 行数不一致"
    sids = pl.read_parquet(os.path.join(SUB_DIR, BASE_SUB)).sort("sample_id")["sample_id"].to_numpy()

    # —— 1. 先验证 v8 配方（不验证就改动 = 拿错参照，CLAUDE.md #7 的纪律）——
    v8 = unit(pt) + ALPHA * unit(ps) + BETA * unit(pl_)
    d = np.abs(unit(v8) - unit(ref)).max()
    print(f"配方校验：unit(tabm2emb) + {ALPHA}·unit(cnn) + {BETA}·unit(lgb) vs {BASE_SUB}")
    print(f"  unit 后 max|Δ| = {d:.3e}   {'✅ 逐位一致' if d < 1e-12 else '❌ 配方不符，停止'}")
    if d >= 1e-12:
        raise SystemExit("配方校验失败——不要基于它做变体")
    c = np.corrcoef(np.vstack([pt, ps, pl_]))
    print(f"  成员相关: tabm×cnn={c[0,1]:.4f} tabm×lgb={c[0,2]:.4f} cnn×lgb={c[1,2]:.4f}")

    # —— 2. U_s ——
    us = load_test_us(sids)
    q = np.quantile(us, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"\ntest U_s: n={us.size:,} 均值={us.mean():.6g} "
          f"q05/25/50/75/95 = {' / '.join(f'{v:.3g}' for v in q)}")
    print(f"  （train 中位 0.000902 → test 中位 {q[2]:.6g}，比值 {q[2] / 0.000902:.3f}）")

    print("\n=== 变体 ===")
    if args.only in ("all", "both"):
        pred = v8 * us ** args.w_all
        report(f"all  (blend × U_s^{args.w_all:.3f})", pred)
        out = os.path.join(SUB_DIR, f"submission_blend_v8_usall{args.w_all:.2f}.parquet")
        pl.DataFrame({"sample_id": sids, "prediction": pred}).write_parquet(out)
        print(f"    → {out}")
    if args.only in ("cnn", "both"):
        cnn_scaled = unit(ps * us ** args.w_cnn)
        pred = unit(pt) + ALPHA * cnn_scaled + BETA * unit(pl_)
        report(f"cnn  (仅双塔CNN × U_s^{args.w_cnn:.3f})", pred)
        out = os.path.join(SUB_DIR, f"submission_blend_v8_uscnn{args.w_cnn:.3f}.parquet")
        pl.DataFrame({"sample_id": sids, "prediction": pred}).write_parquet(out)
        print(f"    → {out}")

    # —— 3. 与现状的相关（按 HYPOTHESES A-11 的流程：相关 <0.99 才有希望看到 LB 变化）——
    print("\n与 v8 的 test 预测相关（<0.99 才有希望看到 LB 变化，A-11 的流程）：")
    if args.only in ("all", "both"):
        print(f"  all  vs v8: {np.corrcoef(unit(v8 * us ** args.w_all), unit(ref))[0, 1]:.5f}")
    if args.only in ("cnn", "both"):
        pp = unit(pt) + ALPHA * unit(ps * us ** args.w_cnn) + BETA * unit(pl_)
        print(f"  cnn  vs v8: {np.corrcoef(unit(pp), unit(ref))[0, 1]:.5f}")


if __name__ == "__main__":
    main()
