r"""mlp_readout_diag.py — 追查「读出权重退化成均匀」的连锁后果（承接 mlp_mixing_diag.py）。

发现（2026-09-24，`mlp_mixing_diag.py`）
---------------------------------------
双塔MLP 的 `softmax(read_w)` 在**全部 6 折、两个塔**上都停在**均匀**：
`0.250/0.250/0.250/0.250`，最新 4 位占比 = 4/T（粗段 0.0313、细段 0.0656）。
`read_w` 初始化是 θ=0 → 均匀，且它在 `model.parameters()` 里、确实被 AdamW 优化。

为什么这件事要紧：`read_w` 均匀时
    out_d = (1/T) Σ_j h_d[j] = (1/T) Σ_j Σ_t W[j,t]·z_d[t] = Σ_t c[t]·z_d[t],  c = colmean_j(W)
即**W 的行结构（按输出位置的索引）在读出处被平均掉了**——只有列均值 c 存活。
所以「学到了逐位置检测器」这件事，有多大比例**真的到达了输出**，必须单独量。

本脚本量四件事（都不训练）
------------------------
1. `read_w` 原始值（确认"均匀"不是 softmax 把 1e-3 抹平了）
2. `W2` 与 `W1` 的**位置结构占比**：`||X − 沿被池化维的均值|| / ||X||`
   → W2 的这部分**被读出平均掉**，W1 的不被
3. **有效输出滤波器** `e[t] = Σ_k c[k]·W1[k,t]`（线性路径）：它长什么样、集中在哪、
   带宽多少 —— 这是塔**真正**施加在时间轴上的东西（所有 256 维共用同一条）
4. `--real-data` 时另测：`pos` 相对激活的尺度，以及端到端 `pos` 影响力
   （`1 − cos(out, out|_{pos=0})`）

用法
----
    OMP_NUM_THREADS=1 python analysis/mlp_readout_diag.py
    OMP_NUM_THREADS=1 python analysis/mlp_readout_diag.py --real-data
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "models", "mlp"))

BASE = os.environ.get("MSC_BASE") or "/root/msdata"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch  # noqa: E402
torch.set_num_threads(1)

TOWERS = [("A(粗段)", "tower_a", 128), ("B(细段)", "tower_b", 60)]


def rel_removed(X, axis):
    """沿 axis 求均值后，原矩阵里"被这个均值带走"的比例。0=全是均值，1=与均值无关。"""
    m = X.mean(axis=axis, keepdims=True)
    return float(np.linalg.norm(X - m) / (np.linalg.norm(X) + 1e-30))


def profile(a, name, quarters=4):
    q = np.array_split(a, quarters)
    tot = a.sum()
    s = " / ".join(f"{x.sum() / tot:.3f}" for x in q)
    return (f"    {name:<10} 四分位质量(旧→新)={s}  argmax=#{int(a.argmax()):3d}  "
            f"max/均匀={a.max() * len(a):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=os.path.join(BASE, "snap_cache", "ckpt_mlp_crop128"))
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--real-data", action="store_true")
    ap.add_argument("--n-samples", type=int, default=256)
    args = ap.parse_args()

    ck = None
    for tname, tkey, T in TOWERS:
        print("\n" + "=" * 92)
        print(f"塔 {tname}  (T={T})")
        rows_rw, rows_w2, rows_w1, rows_e = [], [], [], []
        for fi in range(args.folds):
            p = os.path.join(args.ckpt_dir, f"f{fi}.{args.which}.pt")
            if not os.path.exists(p):
                continue
            ck = torch.load(p, map_location="cpu", weights_only=False)
            sd = ck["state_dict"]
            rw = sd[f"{tkey}.read_w"].float().numpy()
            w = np.exp(rw - rw.max()); w /= w.sum()
            rows_rw.append((rw, w))

            W1 = sd[f"{tkey}.time.0.weight"].float().numpy()      # (H, T) 每个隐单元一条时间滤波
            W2 = sd[f"{tkey}.time.3.weight"].float().numpy()      # (T, H) 每个输出位置一条
            rows_w2.append(rel_removed(W2, axis=0))   # 沿"输出位置"求均值 → 被读出池化掉的
            rows_w1.append(rel_removed(W1, axis=1))   # 沿"时间"求均值 → W1 的时间结构（不被池化）
            c = W2.mean(axis=0)                        # (H,) 均匀读出下的唯一存活方向
            e = c @ W1                                 # (T,) 有效输出滤波器
            rows_e.append(e)

        print("  ① read_w 原始值（若已均匀，说明它压根没离开初始化 θ=0）")
        raw = np.concatenate([r[0] for r in rows_rw])
        print(f"     raw：min={raw.min():+.5f} max={raw.max():+.5f} std={raw.std():.5f} "
              f"|max|={np.abs(raw).max():.5f}")
        dev = max(np.abs(r[1] - 1.0 / len(r[1])).max() for r in rows_rw)
        print(f"     softmax 离均匀的最大偏差 = {dev:.6f}")
        for fi, (_, w) in enumerate(rows_rw):
            print(profile(w, f"fold {fi}"))

        print("\n  ② 位置结构占比（1.0 = 与沿该维的均值完全无关；0 = 全是均值）")
        a2, a1 = np.array(rows_w2), np.array(rows_w1)
        print(f"     W2 沿【输出位置】对均值求残差: {a2.mean():.3f} ± {a2.std():.3f}"
              f"   ← 这部分**被均匀读出平均掉**")
        print(f"     W1 沿【时间】对均值求残差    : {a1.mean():.3f} ± {a1.std():.3f}"
              f"   ← 不被池化，完整保留")

        print("\n  ③ 有效输出滤波器 e = colmean(W2) @ W1 —— 塔真正施加在时间轴上的东西")
        E = np.array(rows_e)
        for fi, e in enumerate(E):
            print(profile(np.abs(e) / np.abs(e).sum(), f"fold {fi} (|e|归一)"))
        ea = np.abs(E).mean(axis=0); ea /= ea.sum()
        print(profile(ea, "均值 |e|"))
        # 它是否接近"只取最新"或"只取最旧"？
        T = E.shape[1]
        half = T // 2
        print(f"     |e| 质量：旧半={ea[:half].sum():.3f}  新半={ea[half:].sum():.3f}")
        # 与"逐位置读出"的对照：read_w 若可学本可表达同样的东西
        print("     ⚠️ 对照：read_w 本可以（且初始就可以）表达任意逐位置剖面，"
              "但它停在均匀")

    if args.real_data:
        print("\n" + "=" * 92)
        print("④ 真实数据上的端到端检验")
        import mlp_baseline as M
        Xc, nv, Xf, y, marr, months, sids = M.load_train()
        rows = np.arange(0, len(y), max(1, len(y) // args.n_samples))[:args.n_samples]
        xa = torch.from_numpy(np.stack([M.crop_newest(np.asarray(Xc[r]).T, nv[r], 128)
                                        for r in rows]).astype(np.float32))
        xb = torch.from_numpy(np.asarray(Xf[rows], dtype=np.float32).transpose(0, 2, 1))

        for tname, tkey, T in TOWERS:
            p = os.path.join(args.ckpt_dir, f"f0.{args.which}.pt")
            ck = torch.load(p, map_location="cpu", weights_only=False)
            sd = ck["state_dict"]
            pos = sd[f"{tkey}.pos"].float()
            ln_in_w, ln_in_b = sd[f"{tkey}.ln_in.weight"], sd[f"{tkey}.ln_in.bias"]
            ch_w, ch_b = sd[f"{tkey}.channel.0.weight"], sd[f"{tkey}.channel.0.bias"]
            ln_mid_w, ln_mid_b = sd[f"{tkey}.ln_mid.weight"], sd[f"{tkey}.ln_mid.bias"]
            x = (xa if tkey == "tower_a" else xb)
            # 复刻 forward 到 time 之前，拿 z（= time 的输入）
            h = torch.nn.functional.linear(x.transpose(1, 2), ln_in_w, ln_in_b)
            h = torch.nn.functional.gelu(torch.nn.functional.linear(h, ch_w, ch_b))
            h = h + pos
            z = torch.nn.functional.layer_norm(h, (h.shape[-1],), ln_mid_w, ln_mid_b)
            act = h - pos                      # channel 输出（未加 pos）
            print(f"\n  塔 {tname}：")
            print(f"     ||channel(x)|| 逐位置均值 = {act.norm(dim=-1).mean():.4f}   "
                  f"||pos|| 逐位置均值 = {pos.norm(dim=-1).mean():.4f}   "
                  f"比值 = {pos.norm(dim=-1).mean() / act.norm(dim=-1).mean():.4f}")
            print(f"     ||z|| 逐位置均值 = {z.norm(dim=-1).mean():.4f}")

            # 端到端：pos 清零后的输出变化
            tw = {k[len(tkey) + 1:]: v for k, v in sd.items() if k.startswith(tkey + ".")}
            tower = M.ChannelTimeTower(15 if tkey == "tower_a" else 16, T)
            tower.load_state_dict(tw); tower.eval()
            with torch.no_grad():
                o1 = tower(x)
                tower.pos.zero_()
                o2 = tower(x)
                tower.pos.copy_(pos)
                o3 = tower(torch.zeros_like(x))     # 输入全零的参照尺度
            c1 = torch.nn.functional.cosine_similarity(o1, o2, dim=1).mean().item()
            c2 = torch.nn.functional.cosine_similarity(o1, o3, dim=1).mean().item()
            print(f"     pos 清零后 1-cos = {1 - c1:.4f}   "
                  f"（对照：输入全零的 1-cos = {1 - c2:.4f}）")


if __name__ == "__main__":
    main()
