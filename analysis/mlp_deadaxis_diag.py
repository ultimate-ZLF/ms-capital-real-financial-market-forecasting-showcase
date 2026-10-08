r"""mlp_deadaxis_diag.py — 验证「输出位置轴是死轴」这个假说（承接上面两个诊断）。

疑点
----
`mlp_mixing_diag.py` 报 `M = W2@W1` 强烈非 Toeplitz（0.965 ≈ 随机矩阵的 0.993），
据此本可读成「模型学出了逐位置的检测器」。但 `mlp_readout_diag.py` 发现
`softmax(read_w)` **几乎均匀**（原始 std 仅 0.037）。

若读出均匀，则 `∂L/∂W2[j,k] = w_j · G[k]` —— **W2 的每一行拿到（近似）相同的梯度**，
行与行之间在优化中**永不分化**，W2 的行结构会一直停在**初始化噪声**上。于是：

  假设 H：W2 的行间差异 ≈ 初始化噪声，不是"学到的逐位置结构"；
          模型真正学到的输出维只有 W2 的**列均值**（= 均匀读出的唯一存活方向）；
          于是「时间混合」在**学到的那部分**上是 rank-1（位置盲）。

三条判据
--------
1. `W2 − colmean(W2)` 的尺度 vs PyTorch 默认初始化尺度 1/√H。
2. **决定性**：`M_learned = colmean(W2) @ W1`（学到的那部分）的行余弦 / PC1。
   = 1.0/1.0 → 位置盲，假设 H 成立。与 `M_raw = W2@W1` 对照即知第一个诊断的数字
   里有多少是初始化噪声。
3. `M_true` 用**真实** `softmax(read_w)` 算 —— 与 M_learned 之差 = 读出那 ±15% 调制
   究竟贡献了多少位置结构。

用法：OMP_NUM_THREADS=1 python analysis/mlp_deadaxis_diag.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "models", "mlp"))
BASE = os.environ.get("MSC_BASE") or "/root/msdata"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import torch  # noqa: E402
torch.set_num_threads(1)

CKPT = os.path.join(BASE, "snap_cache", "ckpt_mlp_crop128")
TOWERS = [("A(粗段)", "tower_a", 128), ("B(细段)", "tower_b", 60)]


def row_struct(K):
    """行归一化后两两余弦均值 + 行空间 PC1 方差占比。1.0/1.0 = 各行只差标量倍 = 位置盲。"""
    R = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-30)
    iu = np.triu_indices(len(K), 1)
    sv = np.linalg.svd(R, compute_uv=False)
    return float((R @ R.T)[iu].mean()), float(sv[0] ** 2 / (sv ** 2).sum())


def toeplitz_resid(K):
    T = K.shape[0]
    d = np.arange(T)[None, :] - np.arange(T)[:, None]
    Kt = np.zeros_like(K)
    for u in np.unique(d):
        m = d == u
        Kt[m] = K[m].mean()
    return float(np.linalg.norm(K - Kt) / (np.linalg.norm(K) + 1e-30))


for tname, tkey, T in TOWERS:
    print("=" * 94)
    print(f"塔 {tname}  (T={T}, 默认 init 界 1/√H = {1 / np.sqrt(128):.5f})")
    R = {k: [] for k in ("resid", "w2std", "ml_rc", "ml_pc", "mt_rc", "mt_pc",
                         "mr_rc", "mr_pc", "ml_tz", "mr_tz", "rd")}

    for fi in range(6):
        p = os.path.join(CKPT, f"f{fi}.best.pt")
        if not os.path.exists(p):
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        W1 = sd[f"{tkey}.time.0.weight"].float().numpy()          # (H, T)
        W2 = sd[f"{tkey}.time.3.weight"].float().numpy()          # (T, H)
        w = torch.softmax(sd[f"{tkey}.read_w"].float(), 0).numpy()  # (T,)

        cm = W2.mean(axis=0)                                       # (H,) 均匀读出的存活方向
        R["resid"].append(float((W2 - cm[None, :]).std()))
        R["w2std"].append(float(W2.std()))

        # 三个算子（都是 (T,T)，但内容不同）：
        #   M_raw     = W2 @ W1                  —— 含未学的行噪声（第一个诊断报的）
        #   M_learned = colmean(W2) @ W1         —— 只用"学到的那部分"（均匀读出的极限）
        #   M_true    = (wᵀW2) @ W1              —— 用真实读出（含 ±15% 调制）
        M_raw = W2 @ W1
        M_learned = np.tile((cm @ W1)[None, :], (T, 1))
        M_true = np.tile(((w @ W2) @ W1)[None, :], (T, 1))

        for M, tag in ((M_learned, "ml"), (M_true, "mt"), (M_raw, "mr")):
            rc, pc = row_struct(M)
            R[f"{tag}_rc"].append(rc)
            R[f"{tag}_pc"].append(pc)
        R["ml_tz"].append(toeplitz_resid(M_learned))
        R["mr_tz"].append(toeplitz_resid(M_raw))
        R["rd"].append(float((w.max() * T - 1)))   # 读出相对均匀的最大增益（0=完全均匀）

    print(f"\n  ① W2 行间残差 std = {np.mean(R['resid']):.5f}   "
          f"默认 init 界 1/√H = {1 / np.sqrt(128):.5f}   "
          f"比值 = {np.mean(R['resid']) * np.sqrt(128):.3f}")
    print(f"     W2 整体 std = {np.mean(R['w2std']):.5f}（均匀分布 U(−1/√H,1/√H) 的 std = "
          f"{1 / np.sqrt(128) / np.sqrt(3):.5f}）")
    print(f"     读出相对均匀的最大增益 = {np.mean(R['rd']):.4f}")

    print(f"\n  ② 行空间相似度（行余弦 / PC1）；1.00 / 1.00 = 各行只差标量倍 = 位置盲")
    print(f"     M_learned = colmean(W2)@W1（学到的那部分）: {np.mean(R['ml_rc']):+.4f} / "
          f"{np.mean(R['ml_pc']):.4f}      Toeplitz残差={np.mean(R['ml_tz']):.4f}")
    print(f"     M_true    = (wᵀW2)@W1    （含 ±15% 调制）: {np.mean(R['mt_rc']):+.4f} / "
          f"{np.mean(R['mt_pc']):.4f}")
    print(f"     M_raw     = W2@W1        （含行噪声）     : {np.mean(R['mr_rc']):+.4f} / "
          f"{np.mean(R['mr_pc']):.4f}      Toeplitz残差={np.mean(R['mr_tz']):.4f}")
