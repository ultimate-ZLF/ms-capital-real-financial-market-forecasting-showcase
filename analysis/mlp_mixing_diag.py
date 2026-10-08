r"""mlp_mixing_diag.py — MLP「有效混合算子」结构诊断（不训练、纯 CPU）。

要回答的问题
------------
双塔MLP 的时间维主干是 `Linear(T→H) + GELU + Linear(H→T)`，权重**按位置索引**：
`out[j] = Σ_t W[j,t]·x[t]`（跨通道共享同一套 W）。双塔CNN 的卷积核则**沿时间共享**。
本模型的 LB 衰减 0.716（CNN 是 0.973），登记的头号推断是「逐位置稠密权重 vs
权重共享」的归纳偏置差异（CLAUDE.md #13 ① / models/mlp/README.md「推断部分①」），
但从未被分离。

本脚本不训练，只把**已经学到的算子**量出来，看它符不符合那个推断的**可测签名**：

  签名 A（用户假说）：MLP 的 `K_j[·] = ∂out[j]/∂in[·]` 在 j 之间**互不相似**
                     —— 每个输出位置一个专属的"形变检测器"。
  签名 B（反面）：K_j 高度相似 / 近似 Toeplitz
                 —— 模型自己学出了权重共享，则该机制**当场被证伪**。

三个量
------
1. **Toeplitz 残差** `||K − T(K)||_F / ||K||_F`，`T(K)` = 沿对角线取均值。
   卷积的算子是**结构上** Toeplitz（残差 ≈ 0）；随机矩阵 ≈ 1。
2. **跨位置相似度**：行归一化后两两余弦的均值，以及行空间的 PC1 方差占比。
   CNN 结构上 = 1.0 / 1.0（只有一个核，逐位置性只在读出 `w_j` 上）。
3. **有效带宽**：`|j−t|` 在 `|K|` 下的加权均值（+ 落在 ±5 / ±20 的质量）。
   用来把「共享 vs 逐位置」和「局部 vs 全局」两个轴分开。

算子用**两条路**各算一次，互为对照：
  - `M = W2 @ W1`            —— 线性路径（若 GELU 是线性的算子），无数据依赖
  - `J = W2 @ diag(ḡ') @ W1` —— 真实 Jacobian，ḡ' 用真实激活上的 GELU' 均值（`--real-data`）

用法
----
    OMP_NUM_THREADS=1 python analysis/mlp_mixing_diag.py                       # 只用 M（秒级）
    OMP_NUM_THREADS=1 python analysis/mlp_mixing_diag.py --real-data           # 另算 J（需粗段缓存）

⚠️ **必须钉线程**：服务器 cgroup 只给 0.5 核（`cpu.max=50000 100000`），而 numpy/torch
按 `nproc`(=80) 开线程 → 超订后同一矩阵乘慢 557×（2026-09-24 实测）。
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "models", "mlp"))

BASE = os.environ.get("MSC_BASE") or "/root/msdata"

# 先钉线程再 import torch（torch 在 import 时读线程数环境变量）
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch  # noqa: E402

torch.set_num_threads(1)

TOWERS = [("A(粗段)", "tower_a", 128), ("B(细段)", "tower_b", 60)]


# ----------------------------------------------------------------- 指标

def toeplitz_resid(K):
    """K 离「只通过 (j−t) 依赖」有多远。0 = 严格 Toeplitz，1 = 与 Toeplitz 无关。"""
    T = K.shape[0]
    diags = np.arange(T)[None, :] - np.arange(T)[:, None]      # (j,t) -> j−t
    Kt = np.zeros_like(K)
    for d in np.unique(diags):
        m = diags == d
        Kt[m] = K[m].mean()
    return float(np.linalg.norm(K - Kt) / (np.linalg.norm(K) + 1e-30))


def row_similarity(K):
    """行归一化后两两余弦：均值 + 行空间 PC1 方差占比。"""
    R = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-30)
    S = R @ R.T
    iu = np.triu_indices(len(K), 1)
    cos = S[iu]
    sv = np.linalg.svd(R, compute_uv=False)
    pc1 = float(sv[0] ** 2 / (sv ** 2).sum())
    return float(cos.mean()), float(np.percentile(cos, 5)), float(np.percentile(cos, 95)), pc1


def bandwidth(K):
    """|K| 权重下 |j−t| 的加权均值中位数，以及落在 ±5 / ±20 的质量。"""
    T = K.shape[0]
    d = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
    A = np.abs(K)
    w = A / (A.sum(axis=1, keepdims=True) + 1e-30)
    mean_abs = (w * d).sum(axis=1)
    return (float(np.median(mean_abs)),
            float(np.median((w * (d <= 5)).sum(axis=1))),
            float(np.median((w * (d <= 20)).sum(axis=1))))


def nulls(T, n=5, seed=0):
    """参照系：随机矩阵 / rank-1 外积 / 理想 Toeplitz(k=45 方波)。"""
    rng = np.random.default_rng(seed)
    out = {}
    acc = [[] for _ in range(7)]
    for _ in range(n):
        G = rng.standard_normal((T, T))
        u, v = rng.standard_normal(T), rng.standard_normal(T)
        R1 = np.outer(u, v)
        d = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
        Tz = (d <= 22).astype(float)
        for K, i in ((G, 0), (R1, 1), (Tz, 2)):
            a, b, c, p = row_similarity(K)
            acc[i * 2].append((toeplitz_resid(K), a, p))
            acc[i * 2 + 1].append(bandwidth(K))
    for i, name in enumerate(["随机 iid", "rank-1 外积", "理想 Toeplitz(k=45)"]):
        arr = np.array(acc[i * 2])
        bw = np.array(acc[i * 2 + 1])
        out[name] = dict(toep=arr[:, 0].mean(), rowcos=arr[:, 1].mean(), pc1=arr[:, 2].mean(),
                         bw=bw[:, 0].mean(), m5=bw[:, 1].mean(), m20=bw[:, 2].mean())
    return out


# ----------------------------------------------------------------- 主流程

def fmt(name, toe, cos, c5, c95, pc1, bw, m5, m20):
    return (f"    {name:<8} Toeplitz残差={toe:.3f}  行余弦={cos:+.3f}[{c5:+.2f},{c95:+.2f}] "
            f"PC1={pc1:.3f}  带宽={bw:5.1f}  ±5={m5:.2f} ±20={m20:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=os.path.join(BASE, "snap_cache", "ckpt_mlp_crop128"))
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--real-data", action="store_true",
                    help="另算真实 Jacobian J（要读粗段缓存；细段现建到内存）")
    ap.add_argument("--n-samples", type=int, default=512, help="算 J 用的真实样本数")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print(f"检查点: {args.ckpt_dir}  折数={args.folds}  取 {args.which}")
    recs = {("A(粗段)", "M"): [], ("A(粗段)", "J"): [],
            ("B(细段)", "M"): [], ("B(细段)", "J"): []}
    readout, posinfo = {t: [] for t, _, _ in TOWERS}, {t: [] for t, _, _ in TOWERS}
    Mats = {}

    for fi in range(args.folds):
        path = os.path.join(args.ckpt_dir, f"f{fi}.{args.which}.pt")
        if not os.path.exists(path):
            print(f"  [跳过] 缺 {path}")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        for tname, tkey, T in TOWERS:
            W1 = sd[f"{tkey}.time.0.weight"].float().numpy()      # (H, T)
            W2 = sd[f"{tkey}.time.3.weight"].float().numpy()      # (T, H)
            M = W2 @ W1                                          # (T, T) 线性路径
            recs[(tname, "M")].append((toeplitz_resid(M), *row_similarity(M), *bandwidth(M)))
            Mats[(fi, tname, "M")] = M

            # 读出权重剖面（四分位质量，旧→新）+ 最新 4 个位置占比
            rw = torch.softmax(sd[f"{tkey}.read_w"].float(), dim=0).numpy()
            q = np.array_split(rw, 4)
            readout[tname].append(([float(x.sum()) for x in q], float(rw[-4:].sum()),
                                   int(rw.argmax()), float(sd[f"{tkey}.scale"]) ))

            # 位置嵌入：沿位置的模长剖面
            pos = sd[f"{tkey}.pos"].float().numpy()                # (T, dim)
            pn = np.linalg.norm(pos, axis=1)
            pq = np.array_split(pn, 4)
            posinfo[tname].append(([float(x.mean()) for x in pq], float(pn.std())))

            if args.real_data:
                J = None   # 由下面统一算（需要数据）
                Mats[(fi, tname, "W1")] = W1
                Mats[(fi, tname, "W2")] = W2

    print("\n" + "=" * 96)
    print("参照系（解析/随机，T=128）")
    for name, d in nulls(128).items():
        print(fmt(name, d["toep"], d["rowcos"], 0, 0, d["pc1"], d["bw"], d["m5"], d["m20"]))
    print("\n  结构参照：双塔CNN 的卷积核沿时间**共享** → 算子结构上就是 Toeplitz")
    print("           （Toeplitz残差≈0、行余弦=1.0、PC1=1.0）；它的逐位置性只体现在读出 w_j 上")

    for tname, _, _ in TOWERS:
        print("\n" + "=" * 96)
        print(f"塔 {tname}   —— M = W2@W1（线性路径，6 折逐折）")
        arr = np.array(recs[(tname, "M")])
        for fi, a in enumerate(arr):
            print(fmt(f"fold {fi}", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]))
        m = arr.mean(axis=0)
        print(fmt("均值", m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]))
        if args.real_data and recs[(tname, "J")]:
            print(f"\n塔 {tname}   —— J（真实 Jacobian，GELU' 按真实激活加权）")
            aj = np.array(recs[(tname, "J")])
            mj = aj.mean(axis=0)
            print(fmt("均值", mj[0], mj[1], mj[2], mj[3], mj[4], mj[5], mj[6], mj[7]))

        print(f"\n  {tname} 读出 softmax 四分位质量（旧→新）与最新 4 位占比，逐折：")
        for fi, (q, newest4, am, sc) in enumerate(readout[tname]):
            print(f"    fold {fi}: {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f} / {q[3]:.3f}"
                  f"   最新4位={newest4:.4f}  argmax=#{am:3d}  scale={sc:.3f}")
        rq = np.array([r[0] for r in readout[tname]]).mean(axis=0)
        print(f"    均值   : {rq[0]:.3f} / {rq[1]:.3f} / {rq[2]:.3f} / {rq[3]:.3f}"
              f"   最新4位={np.mean([r[1] for r in readout[tname]]):.4f}")

        print(f"\n  {tname} 位置嵌入 ||pos_j|| 四分位（旧→新）与沿位置的 std：")
        for fi, (pq, sdv) in enumerate(posinfo[tname]):
            print(f"    fold {fi}: {pq[0]:.3f} / {pq[1]:.3f} / {pq[2]:.3f} / {pq[3]:.3f}"
                  f"   std={sdv:.3f}")
        pqa = np.array([p[0] for p in posinfo[tname]]).mean(axis=0)
        print(f"    均值   : {pqa[0]:.3f} / {pqa[1]:.3f} / {pqa[2]:.3f} / {pqa[3]:.3f}")

    if args.out:
        np.savez_compressed(args.out, **{f"{k[0]}_{k[1]}_{k[2]}": v for k, v in Mats.items()})
        print(f"\n算子矩阵已存 {args.out}")


if __name__ == "__main__":
    main()
