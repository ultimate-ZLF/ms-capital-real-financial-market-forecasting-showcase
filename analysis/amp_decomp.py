"""amp_decomp.py — 预测的**三层幅度分解** + **用归档数字认领 OOF 目录**（不训练，几分钟）。

## 为什么有它（2026-09-28 的一次真错）

当天量"细段派生通道有没有用"时，我把基线认成了 `snap_cache/oof_parts_tcn`
（**它其实是 v22 MSE 臂**），而当前最强的 cos 臂在 **`oof_parts_tcn_cos`**。
于是把 **−0.00149 读成 +0.00541**，并据此编出一整套"跨 |y| 带幅度重分配"的机制解释——
三条判据互相印证，看起来毫无破绽。

根因：**目录名 `oof_parts_tcn` 看起来就像"那个 TCN 臂"**，只有 `_cos` 后缀才是最强臂。
（同族：CLAUDE.md #7 的"静默认错对象"。）

⇒ 本脚本把对策做成标准动作：**不认名字，用 MEMORY 里已归档的汇总数字反查**。
本项目记的是"**成员口径 = 逐折均值**"（不是把 oof_all 拼起来算一个 cos，两者差 ~0.002）。

## 三层分解（恒等式，不是比喻）

按任意切片 s（默认 |y| 十分位）有

    cos(p, y) = Σ_s u_s · v_s · cos_s

其中 `u_s = ‖p_s‖/‖p‖`（**模型投放的跨带幅度轮廓**，模型自己选的）、
`v_s = ‖y_s‖/‖y‖`（数据固定）、`cos_s` = 带内方向准确度。
逐层消融即可回答"两个臂的差由谁承载"：

| 消融臂 | 保留 | 抹掉 |
|---|---|---|
| `sign(p)` | 只有符号 | 全部幅度 |
| `sign(p) × 带内 RMS` | 符号 + **跨带轮廓** | 带内幅度 |
| `p / 带内 RMS` | 符号 + **带内幅度** | 跨带轮廓 |

**实测用途**（2026-09-28）：MSE 臂 → cos 臂（该臂 LB 0.122→0.128、迁移比 +1.003）的 **+0.0069** 里，
**符号层贡献 −0.0006（等于没有）、跨带幅度轮廓贡献 +0.004~+0.006**。
⚠️ 这是**一个配对**的分解，推不出"幅度是被低估的金矿"：跨 8 条 OOF 臂看，
`u_s` 集中度与 cos **没有单调关系**（`twotower` 轮廓与 target 最像而 cos 最低）。
三条已测边界（oracle 用了 y / 外部因子只给 +0.0012 / 理想目标依赖 X 没有的信息）
见 `CLAUDE.md #15` 第 5 条与 `MEMORY.md`「幅度轴」。

⚠️ **定位：这是归因工具，不是过滤器。** 它回答"这个改动到底动了什么"，
**不能**单独用来否定一个改动——否定要"归因 + 该机制在你手上确实失效过的证据"。

用法（服务器）：
    python analysis/amp_decomp.py --arms snap_cache/oof_parts_tcn,snap_cache/oof_parts_tcn_cos
    python analysis/amp_decomp.py --a oof_parts_mlp_crop128 --b oof_parts_mlp_yw99 --fold 0
"""
import argparse
import os
import sys

import numpy as np
import polars as pl

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)

# MEMORY「LB 分数历史」记的**逐折均值（成员口径）**——用来认领目录。加一条就自动多认一个。
ARCHIVED_CV = {
    "oof_parts_tcn": ("v22 双塔TCN-MSE", 0.13072),
    "oof_parts_tcn_cos": ("v24 双塔TCN-cos", 0.13670),
    "oof_parts_twotower": ("双塔CNN", 0.12022),
}


def cos(p, y):
    d = np.sqrt((p * p).sum() * (y * y).sum())
    return float((p * y).sum() / d) if d > 0 else float("nan")


def load_fold(d, fold):
    p = d if os.path.isabs(d) else os.path.join(BASE, d)
    f = os.path.join(p, f"f{fold}.parquet")
    if not os.path.exists(f):
        raise FileNotFoundError(f)
    return pl.read_parquet(f).select(["sample_id", "pred"])


def identify(dirs):
    """**不认名字，认数字**：逐折均值对上归档值才认领，对不上就明确说"未认领"。"""
    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).select("sample_id", "target")
    print("=== 认领目录（逐折均值 vs MEMORY 归档值）===")
    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        vals = []
        for fi in range(6):
            try:
                j = load_fold(d, fi).join(lab, on="sample_id")
            except FileNotFoundError:
                continue
            vals.append(cos(j["pred"].to_numpy(), j["target"].to_numpy()))
        if not vals:
            print(f"  {name:26s} 无 f*.parquet")
            continue
        m = float(np.mean(vals))
        tag = ARCHIVED_CV.get(name)
        if tag is None:
            verdict = "（未登记，无法认领）"
        elif abs(m - tag[1]) < 5e-5:
            verdict = f"✅ = {tag[0]}（归档 {tag[1]:.5f}）"
        else:
            verdict = f"❌ 对不上任何归档值（最接近 {tag[0]} {tag[1]:.5f}）"
        print(f"  {name:26s} 逐折均值={m:.5f}  {len(vals)} 折  {verdict}")
    print()


def layer_ablation(p, y, bands):
    """→ (sign, sign×带内RMS, p/带内RMS)。后两者分别只保留"跨带轮廓"与"带内幅度"。"""
    s = np.sign(p)
    rms = np.array([np.sqrt((p[m] ** 2).mean()) for m in bands])
    rb = np.empty_like(p)
    for i, m in enumerate(bands):
        rb[m] = rms[i]
    return s, s * rb, p / np.where(rb == 0, 1, rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None, help="逗号分隔的 OOF 目录（≥1 个）；会用归档数字认领")
    ap.add_argument("--a", default=None, help="--a/--b：直接指定两个目录做分解对照")
    ap.add_argument("--b", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--bins", type=int, default=10, help="按 |y| 等频分几层")
    args = ap.parse_args()

    dirs = [x.strip() for x in (args.arms or "").split(",") if x.strip()]
    if args.a and args.b:
        dirs = [args.a, args.b]
    if not dirs:
        ap.error("给 --arms 或 --a/--b")
    identify(dirs)

    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).select(
        "sample_id", "month", "target")
    df = load_fold(dirs[0], args.fold).rename({"pred": "p0"})
    for i, d in enumerate(dirs[1:], start=1):
        df = df.join(load_fold(d, args.fold).rename({"pred": f"p{i}"}), on="sample_id")
    df = df.join(lab, on="sample_id").sort("sample_id")
    y = df["target"].to_numpy().astype(np.float64)
    q = np.quantile(np.abs(y), np.linspace(0, 1, args.bins + 1))
    q[-1] += 1.0
    k = np.clip(np.searchsorted(q[1:-1], np.abs(y)), 0, args.bins - 1)
    bands = [k == i for i in range(args.bins)]

    print(f"fold {args.fold}  n={len(y):,}  分层 = |y| {args.bins} 等频\n")
    names = ["cos(p,y)", "符号+带轮廓", "符号+带内幅度", "只有符号"]
    hdr = f"{'臂':<26}" + "".join(f"{h:>14}" for h in names)
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for i, d in enumerate(dirs):
        p = df[f"p{i}"].to_numpy().astype(np.float64)
        s, A, B = layer_ablation(p, y, bands)
        rows.append([cos(p, y), cos(A, y), cos(B, y), cos(s, y)])
        print(f"{os.path.basename(d.rstrip('/')):<26}" + "".join(f"{v:>14.5f}" for v in rows[-1]))
    if len(rows) == 2:
        d_ = [b - a for a, b in zip(rows[0], rows[1])]
        print(f"{'Δ (第2 − 第1)':<26}" + "".join(f"{v:>+14.5f}" for v in d_))
        print(f"\n→ 符号层贡献 {d_[3]:+.5f}｜跨带轮廓贡献 {d_[1]:+.5f}｜带内幅度贡献 {d_[2]:+.5f}"
              f"（总 {d_[0]:+.5f}，差额是交互项）")
    # 轮廓本身
    print()
    for i, d in enumerate(dirs):
        p = df[f"p{i}"].to_numpy().astype(np.float64)
        u = np.array([np.sqrt((p[m] ** 2).sum()) for m in bands])
        u = u / u.sum()
        print(f"  u_s({os.path.basename(d.rstrip('/'))}) 熵={-np.sum(u * np.log(u)):.4f}"
              f"（上限 {np.log(args.bins):.4f}）  " + " ".join(f"{v:.3f}" for v in u))
    v = np.array([np.sqrt((y[m] ** 2).sum()) for m in bands])
    v = v / v.sum()
    print(f"  v_s(target)           " + " ".join(f"{v[i]:.3f}" for i in range(len(v))))
    print("\n  ⚠️ u 越接近均匀、而 v 越集中 ⇒ 幅度轮廓离最优越远（见 CLAUDE.md #15 第 5 条）")


if __name__ == "__main__":
    sys.exit(main())
