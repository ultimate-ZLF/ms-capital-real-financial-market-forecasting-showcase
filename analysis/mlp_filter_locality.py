r"""mlp_filter_locality.py — 量 MLP 时间滤波器的**局部性**与**共享性**（不训练、纯 CPU）。

为什么量这个
------------
`mlp_deadaxis_diag.py` + `mlp_deadaxis_functional.py` 已证明输出位置轴死掉，于是塔的时间块
**功能上就是**：

    out_d = Σ_h c[h]·gelu( Σ_t W1[h,t]·z_d[t] ),   c = colmean(W2)
          = 对每一条特征通道施加**同一个** 2 层 MLP（T→128→1）

⇒ **"逐位置稠密 vs 沿时间共享"这个轴功能上不存在**（没有第二个位置可共享）。
剩下的第一条可真判的差异是 **局部性**：这 128 条滤波是"整窗模板"还是"局部核"？

两个量（每个单元 h 一条 T 维滤波 W1[h,:]）
----------------------------------------
1. **局部性**：质心 `t̄_h` 与带宽 `Σ_t |t−t̄_h|·|w|`（按 |W1[h]| 归一）。
   参照尺度：铺满整窗 ≈ T/3；宽 5 的局部核 ≈ 1.3。
2. **共享性**：把每条滤波**按质心对齐**后，跨单元的两两余弦 / PC1。
   = 1 → 128 条是**同一个核的平移族**（= 它自己学出了"共享+局部"）
   ≈ 0 → 各单元的核互不相关

判读表（2×2，直接把用户假说切成四格）
------------------------------------
|              | 对齐后相似度高 | 相似度低 |
|--------------|---------------|---------|
| **带宽小（局部）** | 共享局部核（= 已学出卷积）| **128 个不同位置的局部检测器 = "固定位置"假说成立** |
| **带宽大（全局）** | 共享的大核 | 128 个独立的整窗模板 |

参照系（合成）：单核随机平移族 / 128 条独立随机 / 128 条独立局部核（分散锚点）。

用法：OMP_NUM_THREADS=4 python analysis/mlp_filter_locality.py [--figs analysis/figs]
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "models", "mlp"))
BASE = os.environ.get("MSC_BASE") or "/root/msdata"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import torch  # noqa: E402

CKPT = os.path.join(BASE, "snap_cache", "ckpt_mlp_crop128")
TOWERS = [("A(粗段)", "tower_a", 128), ("B(细段)", "tower_b", 60)]


def locality(W):
    """(质心数组, 带宽数组, ±5质量数组, top5占比数组)。W: (H, T)。"""
    A = np.abs(W)
    w = A / (A.sum(axis=1, keepdims=True) + 1e-30)
    t = np.arange(W.shape[1])[None, :]
    c = (w * t).sum(1)
    bw = (w * np.abs(t - c[:, None])).sum(1)
    m5 = (w * (np.abs(t - c[:, None]) <= 5)).sum(1)
    top5 = np.sort(w, axis=1)[:, -5:].sum(1)
    return c, bw, m5, top5


def max_cross_corr(W):
    """**共享性的有效判据**：每对滤波扫全部位移取相关系数最大值，再对全部对取均值。

    判读：真平移族 → 1.000；独立的同形状白噪声 → 0.23（T=128）/ 0.30（T=60），即零假设地板。

    ⚠️ **不要用"质心对齐后再算余弦"代替它**（本脚本第一版就是那么写的、结论差点反掉）：
    把一条**白噪声**核平移一下得到的就是另一条独立白噪声 → 对齐恢复不出对应关系；
    而宽带滤波的质心本身就是噪声估计 → 连真平移族都被打成 0.004。
    2026-09-24 实测四组参照：平滑核平移族 0.773 / 白噪声核平移族 **0.004** / 独立 0.001 /
    局部核平移族 0.894 —— 而同一批的 max_cross_corr 是 1.000 / 1.000 / 0.232 / 1.000。
    """
    A = W - W.mean(axis=1, keepdims=True)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-30)
    H, T = A.shape
    M = -np.ones((H, H))
    for s in range(T):
        M = np.maximum(M, A @ np.roll(A, s, axis=1).T)
    iu = np.triu_indices(H, 1)
    return M[iu]


def aligned_similarity(W):
    """按质心对齐后，跨单元的两两余弦均值 + 行空间 PC1。0 填充（不环绕）。"""
    H, T = W.shape
    c, _, _, _ = locality(W)
    sh = (T // 2) - np.round(c).astype(int)
    A = np.zeros_like(W)
    for h in range(H):
        s = sh[h]
        src = slice(max(0, -s), T - max(0, s))
        dst = slice(max(0, s), T - max(0, -s))
        A[h, dst] = W[h, src]
    R = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-30)
    iu = np.triu_indices(H, 1)
    sv = np.linalg.svd(R, compute_uv=False)
    return float((R @ R.T)[iu].mean()), float(sv[0] ** 2 / (sv ** 2).sum()), A


def refs(H, T, seed=0):
    """参照系。返回 {名字: (带宽中位, 对齐余弦, PC1, 最大互相关)}。

    ⚠️ 合成平移族时 **`np.roll` 必须滚同一个 `k`**——写成生成器里现取随机数会得到
    "128 条各不相同的随机向量"，参照系自己就是错的（2026-09-24 踩过）。
    """
    rng = np.random.default_rng(seed)
    out = {}
    k = rng.standard_normal(T)                               # 同一个核
    out["① 真平移族(白噪声核)"] = np.stack([np.roll(k, int(s))
                                            for s in rng.integers(0, T, H)])
    out["② 独立随机(全局)"] = rng.standard_normal((H, T))
    # ③ 128 条独立局部核、锚点分散 = "固定位置检测器"假说的形态
    W = np.zeros((H, T))
    for h in range(H):
        a = int(rng.integers(0, T))
        lo, hi = max(0, a - 4), min(T, a + 5)
        W[h, lo:hi] = rng.standard_normal(hi - lo)
    out["③ 独立局部核(分散锚点)"] = W
    out["④ 全窗均匀"] = np.ones((H, T))
    res = {}
    for name, W in out.items():
        _, bw, _, _ = locality(W)
        ac, pc, _ = aligned_similarity(W)
        res[name] = (float(np.median(bw)), ac, pc, float(max_cross_corr(W).mean()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--figs", default=os.path.join(_HERE, "figs"))
    args = ap.parse_args()

    H = 128
    for tname, tkey, T in TOWERS:
        print("=" * 96)
        print(f"塔 {tname}  (H=128 条滤波, T={T})")
        R = refs(H, T)
        print("\n  参照系（合成，同一套指标）：")
        print(f"    {'形态':<26} {'带宽中位':>8} {'对齐余弦':>9} {'PC1':>7} {'最大互相关':>10}")
        for name, (bw, ac, pc, mc) in R.items():
            print(f"    {name:<26} {bw:8.2f} {ac:9.3f} {pc:7.3f} {mc:10.3f}")

        BWs, ACs, PCs, CENTS, M5s, MCs = [], [], [], [], [], []
        figdata = None
        for fi in range(args.folds):
            p = os.path.join(CKPT, f"f{fi}.best.pt")
            if not os.path.exists(p):
                continue
            W1 = torch.load(p, map_location="cpu", weights_only=False)[
                "state_dict"][f"{tkey}.time.0.weight"].float().numpy()
            c, bw, m5, top5 = locality(W1)
            ac, pc, A = aligned_similarity(W1)
            BWs.append(bw); ACs.append(ac); PCs.append(pc); CENTS.append(c); M5s.append(m5)
            MCs.append(max_cross_corr(W1).mean())
            if fi == 0:
                figdata = (W1, A, c, bw)
                print("\n  fold 0 逐单元细节（前 5 条最强滤波）：")
                for h in np.argsort(-np.linalg.norm(W1, axis=1))[:5]:
                    print(f"    unit {h:3d}: ||W1||={np.linalg.norm(W1[h]):6.2f}  "
                          f"质心={c[h]:6.1f}  带宽={bw[h]:5.2f}  ±5质量={m5[h]:.3f}  "
                          f"top5占比={top5[h]:.3f}")
        BWs = np.array(BWs); CENTS = np.array(CENTS)
        bw_all = BWs.ravel()
        m_all = np.concatenate(M5s)
        print(f"\n  实测（{len(BWs)} 折 × 128 单元）：")
        print(f"    带宽：中位={np.median(bw_all):.2f}  均值={bw_all.mean():.2f}  "
              f"p5={np.percentile(bw_all, 5):.2f}  p95={np.percentile(bw_all, 95):.2f}"
              f"   （T/3 = {T/3:.1f}）")
        print(f"    带宽 < 5 的单元占比 = {(bw_all < 5).mean():.3f}   "
              f"< 10 的占比 = {(bw_all < 10).mean():.3f}")
        print(f"    ±5 质量：中位={np.median(m_all):.3f}")
        # ⚠️ 质心的参照别搞错：随机 T 维滤波的质心会被 CLT 收紧到 std≈sqrt(T/12)，
        # **不是**"均匀铺开时"的 T/sqrt(12)。真参照 = 默认初始化的实测值（见 folds 打印）。
        print(f"    质心：std={CENTS.std():.2f}（T={T}；随机滤波的 CLT 预期 = "
              f"{np.sqrt(T / 12):.2f}，默认初始化实测 ≈1.85/1.35）")
        print(f"    **最大互相关 = {np.mean(MCs):.4f}**（独立零假设 {R['② 独立随机(全局)'][3]:.4f}，"
              f"真平移族 {R['① 真平移族(白噪声核)'][3]:.4f}）→ "
              + ("**贴着零假设 = 不是平移族、各条互不相关**"
                 if np.mean(MCs) < R['② 独立随机(全局)'][3] + 0.02 else "**有共享结构**"))
        print(f"    （失效判据留档：对齐后余弦 = {np.mean(ACs):.3f}，PC1 = {np.mean(PCs):.3f}"
              f" —— 别用它下结论，理由见 max_cross_corr 的 docstring）")

        if args.figs and figdata is not None:
            os.makedirs(args.figs, exist_ok=True)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            W1, A, c, bw = figdata
            order = np.argsort(-c)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            v = np.abs(W1).max() * 0.7
            axes[0].imshow(W1[order], aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
            axes[0].set_title("W1 (128 filters), rows sorted by centroid\n"
                              "x = absolute tap t (0 = oldest)")
            axes[0].set_xlabel("tap t"); axes[0].set_ylabel("filter (sorted)")
            Aord = A[order]
            axes[1].imshow(Aord, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
            axes[1].set_title(f"same filters, ALIGNED by centroid\n"
                              f"aligned cross-unit cosine = {np.mean(ACs):.3f}"
                              f"  (1.0 = all shifts of ONE kernel)")
            axes[1].set_xlabel("lag around centroid")
            # ⚠️ 标题用 ASCII：服务器没配 CJK 字体，中文会渲染成豆腐块
            fig.suptitle(f"{tkey}: are the 128 temporal filters one shared kernel,"
                         f" or 128 independent templates?   median bandwidth = "
                         f"{np.median(bw):.2f}   (T={T})", fontsize=12)
            fig.tight_layout()
            out = os.path.join(args.figs, f"mlp_filters_{tkey}.png")
            fig.savefig(out, dpi=115)
            plt.close(fig)
            print(f"    图 → {out}")


if __name__ == "__main__":
    main()
