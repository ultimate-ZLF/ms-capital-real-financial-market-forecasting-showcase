r"""twotower_kernel_view.py — 把**双塔CNN 学到的卷积核**画出来（不训练、不需要数据）。

为什么现在才能做
----------------
`twotower_baseline.py` 此前**不存检查点**（CLAUDE.md #12 记的：存权重的纪律是
`models/mlp/` 在 2026-09-21 才首次引入的）→ 双塔CNN 的卷积权重从来没被留下过，
`snap_cache/` 下只有 `submit_parts_twotower/*.npy`（预测值，不是权重）。
2026-09-24 给 `twotower_baseline.py` 加了 `--ckpt-dir`，跑一折（GPU ≈14 分钟）即可。

本脚本做两件事
--------------
1. **解出每个塔的"线性化复合核"** `K_f[c, t]`：把 `tower.conv` 里的 GELU 临时换成
   `Identity`（BN 留在 eval 的仿射态），此时三层卷积**整体是线性**的，
   于是用**单位脉冲**在全部 (c,t) 上各打一发、减去零输入响应，
   得到的就是**精确的复合卷积核**（等价于逐层卷积核做 stride/dilation 复合，
   但不必手推那些容易错的 pad/stride 公式）。
   → 这回答"塔到底在找什么形状"：`K_f` 就是它施加在 `(C 通道 × T 步)` 上的模板。
   ⚠️ **诚实说明**：GELU 是逐层的非线性门，被去掉后的复合核是
     **"线性化视图"**（ERF 分析的通行做法），不是塔在真实输入上的精确算子。

2. **画三层卷积核本身** + **块读出权重**（逐位置剖面，与 MLP 的均匀读出形成对照）。

⚠️ 图里标签一律用英文——服务器上没配 CJK 字体，中文会渲染成方框。

用法
----
    OMP_NUM_THREADS=4 python analysis/twotower_kernel_view.py --ckpt <f0.best.pt> --out-dir figs
"""
import argparse
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "models", "cnn"))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "seq"))   # 序列数据管道

BASE = os.environ.get("MSC_BASE") or "/root/msdata"
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import seq_common  # noqa: E402
import twotower_baseline as W  # noqa: E402


def eff_kernel(tower, C, T):
    """线性化复合核 K[c, t, f]（含块读出与尺度；零输入的仿射偏置已减掉）。

    做法：把 `tower.conv` 里的 GELU 全换成 `Identity`（BN 保持 eval → 逐通道仿射），
    此时 conv 堆叠**整体线性**；用单位脉冲打满全部 (c,t)，一次批量前向得到响应矩阵。
    含块读出权重 `softmax(block_w)` 与 `scale`，所以 `K` 直接对应**塔的输出**，
    不是中间层。
    """
    import torch.nn as nn

    orig = [(i, m) for i, m in enumerate(tower.conv) if isinstance(m, nn.GELU)]
    for i, _ in orig:
        tower.conv[i] = nn.Identity()
    try:
        tower.eval()
        x = torch.zeros(C * T, C, T)
        for i, (c, t) in enumerate([(c, t) for c in range(C) for t in range(T)]):
            x[i, c, t] = 1.0
        with torch.no_grad():
            a = torch.softmax(tower.block_w, dim=0)
            out = (tower.conv(x) * a[None, None, :]).sum(-1) * tower.scale
            base = (tower.conv(torch.zeros(1, C, T)) * a[None, None, :]).sum(-1) * tower.scale
    finally:
        for i, m in orig:
            tower.conv[i] = m
    return (out - base).numpy().reshape(C, T, -1), base.numpy()[0]


def fig_effective(K, C, T, names, title, out_png, n_show=6):
    nf = K.shape[2]
    norm = np.linalg.norm(K.reshape(-1, nf), axis=0)
    top = np.argsort(-norm)[:n_show]
    tprof = np.abs(K).sum(axis=(0, 2))
    tprof = tprof / tprof.sum()

    fig = plt.figure(figsize=(15, 2.4 * (n_show + 1)))
    gs = fig.add_gridspec(n_show + 1, 1, hspace=0.55)
    vmax = float(np.abs(K[:, :, top]).max()) * 0.6
    for r, f in enumerate(top):
        ax = fig.add_subplot(gs[r])
        im = ax.imshow(K[:, :, f].T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[0, T, C - 0.5, -0.5])
        ax.set_yticks(range(C)); ax.set_yticklabels(names, fontsize=6)
        ax.set_title(f"feature #{f}  (||K||={norm[f]:.3g})", fontsize=9)
        if r == 0:
            cb = fig.colorbar(im, ax=ax, pad=0.01); cb.ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[n_show])
    ax.plot(np.arange(T), tprof, lw=1.2, color="k")
    ax.set_xlabel("input step t  (0 = oldest, T-1 = NEWEST)")
    ax.set_ylabel("mass")
    ax.set_title("time profile  of the tower's effective filter  sum_c |K|, averaged over features",
                 fontsize=9)
    ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.savefig(out_png, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return top, norm


def fig_kernels(tower, C, names, title, out_png):
    ks = [m.conv.weight.detach().cpu().numpy()
          for m in tower.conv if isinstance(m, W.CausalConv)]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    k1 = ks[0]                                    # (64, C, 5)
    ax = axes[0, 0]
    im = ax.imshow(k1.reshape(k1.shape[0], -1), aspect="auto", cmap="RdBu_r")
    ax.set_xlabel(f"input channel x tap   (tap = 0..{k1.shape[2] - 1})")
    ax.set_ylabel("conv1 out-channel (64)")
    ax.set_title("conv1 kernels  (64 x C x k1)", fontsize=10)
    fig.colorbar(im, ax=ax, pad=0.01)

    ax = axes[0, 1]
    ax.imshow(k1.sum(axis=1), aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("tap"); ax.set_ylabel("conv1 out-channel (64)")
    ax.set_title("conv1 tap profile (summed over input channels)", fontsize=10)
    ax.set_xticks(range(k1.shape[2]))
    ax.set_xticklabels([f"t-{k1.shape[2] - 1 - i}" for i in range(k1.shape[2])], fontsize=9)

    ax = axes[1, 0]
    e = np.linalg.norm(k1, axis=(1, 2))
    ax.bar(np.arange(len(e)), e, color="steelblue")
    ax.set_xlabel("conv1 out-channel"); ax.set_ylabel("||kernel||")
    ax.set_title("which conv1 channels are strong", fontsize=10)

    ax = axes[1, 1]
    mass = np.abs(k1).sum(axis=0)                 # (C, 5)
    mass = mass / mass.sum(axis=0, keepdims=True)
    ax.imshow(mass, aspect="auto", cmap="viridis")
    ax.set_yticks(range(C)); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("tap"); ax.set_title("per-channel tap mass (|w| normalized per channel)",
                                       fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=115)
    plt.close(fig)


def fig_readout(ck, out_png):
    fig, axes = plt.subplots(2, 1, figsize=(13, 6))
    for ax, (tkey, lab, T, span) in zip(axes, [
            ("tower_a", "tower A (coarse: snapshots, 224 steps -> 56 blocks)", W.COARSE_T, 4),
            ("tower_b", "tower B (fine: 1s bins, 60 steps -> 15 blocks)", W.FINE_T, 4)]):
        bw = torch.softmax(ck[f"{tkey}.block_w"].float(), 0).numpy()
        nb = len(bw)
        xs = np.arange(nb) * span + (span - 1) / 2.0
        ax.bar(xs, bw, width=span * 0.8, color="darkorange",
               label="learned block weights")
        ax.axhline(1.0 / nb, ls="--", c="k", lw=1, label=f"uniform = {1.0/nb:.4f}")
        ax.set_xlabel("input step (0 = oldest -> newest)")
        ax.set_ylabel("softmax weight")
        ax.set_title(f"{lab}   scale={float(ck[tkey + '.scale']):.3f}", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("block readout weights (per-position, SHARED across features)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=115)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(BASE, "snap_cache", "ckpt_twotower",
                                                   "f0.best.pt"))
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--head-drop", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "figs"))
    args = ap.parse_args()

    # 指纹断言（CLAUDE.md #12：读侧必须验，不静默用错配置的权重）
    ns = types.SimpleNamespace(drop=args.drop, head_drop=args.head_drop, seed=args.seed,
                               lr=1e-3, wd=3e-4, batch=1024)
    payload = W.load_ckpt(args.ckpt, ns)      # 指纹不符会直接抛
    sd = payload["state_dict"]
    print(f"检查点 {args.ckpt}\n  meta={payload.get('meta')}\n  指纹={payload['arch_hash']}")

    model = W.TwoTowerCNN(drop=args.drop, head_drop=args.head_drop)
    model.load_state_dict(sd)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    names_c, names_f = seq_common.FEAT_NAMES, None
    import flow_feats
    names_f = flow_feats.FEAT_NAMES

    for tkey, C, T, names, lab in (("tower_a", W.COARSE_C, W.COARSE_T, names_c, "coarse"),
                                   ("tower_b", W.FINE_C, W.FINE_T, names_f, "fine")):
        tower = getattr(model, tkey)
        K, base = eff_kernel(tower, C, T)
        f1 = os.path.join(args.out_dir, f"twotower_eff_{lab}.png")
        top, norm = fig_effective(K, C, T, names,
                                  f"Two-tower CNN, tower {tkey} — effective receptive-field kernel"
                                  f"  ({lab}: {C}ch x {T} steps)", f1)
        f2 = os.path.join(args.out_dir, f"twotower_conv1_{lab}.png")
        fig_kernels(tower, C, names,
                    f"Two-tower CNN, tower {tkey} ({lab}) — conv layers", f2)
        print(f"[{lab}] 有效核 {K.shape} → {f1}\n[{lab}] 卷积核 → {f2}")
        print(f"[{lab}] top features {top.tolist()}  norms "
              f"{[round(float(x), 3) for x in norm[top]]}")

    f3 = os.path.join(args.out_dir, "twotower_readout.png")
    fig_readout(sd, f3)
    print(f"读出权重 → {f3}")


if __name__ == "__main__":
    main()
