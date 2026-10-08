r"""mlp_deadaxis_functional.py — 「输出位置轴是死轴」的功能级验证（决定性）。

背景（`mlp_deadaxis_diag.py`，2026-09-24）
---------------------------------------
- `W2 = tower.time.3.weight` (T,H)：行间残差 std 0.0525 ≈ 默认初始化 U(−1/√H,1/√H) 的 std 0.0510
  → **行结构从未被训练过，就是初始化噪声**（因为 `∂L/∂W2[j,k] = w_j·G[k]`，w 近似均匀
    → 所有行拿到相同梯度 → 行间永不分化）。
- `M_learned = colmean(W2)@W1` 是 rank-1（行全同）→ 模型**学到**的时间混合是**位置盲**的。

推断需要功能级确认：**把 W2 的行结构换掉，预测应该几乎不变。**
本脚本在 fold 0 的验证集上直接测这一点（对比 cos 与预测向量的相关性）。

四个干预
--------
  as-is    原样
  colmean  W2 每行 := 列均值（抹掉全部行结构，保留"学到的那部分"）
  reinit   W2 每行 := 全新的默认初始化（换成**另一组**随机噪声，量级相同）
  uni       read_w := 0（读出严格均匀）

判读
----
  若 colmean / reinit 的 cos ≈ as-is，且预测间相关 ≈ 1 → **死轴成立**：
  输出位置维上的一切（87% 的 W2 参数）对预测没有贡献。
  若 reinit 大幅掉分而 colmean 不掉 → 行结构在起作用，只是内容是噪声（另一种结论）。

⚠️ reinit 与 colmean 必须**一起**看：colmean 使模型更像"理论上该有的样子"，
   reinit 保持噪声量级只换内容。两者都不掉分，才是"这个维度无关紧要"。

用法：
    OMP_NUM_THREADS=2 python analysis/mlp_deadaxis_functional.py --ckpt <f0.best.pt>
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "models", "mlp"))

BASE = os.environ.get("MSC_BASE") or "/root/msdata"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import torch  # noqa: E402
torch.set_num_threads(2)

import mlp_baseline as M  # noqa: E402


def cosm(a, b):
    a = a - a.mean(); b = b - b.mean()          # 预测/目标的 cos 不中心化；这里用中心化
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def uncentered(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(BASE, "snap_cache", "ckpt_mlp_crop128",
                                                  "f0.best.pt"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    Xc, nv, Xf, y, marr, months, sids = M.load_train()
    folds = M.make_folds(months, 6)
    va_rows = np.where(np.isin(marr, folds[args.fold]))[0]
    step = max(1, len(va_rows) // args.n)
    sub = va_rows[::step][:args.n]
    yv = y[sub]
    print(f"fold {args.fold}: 验证行 {len(va_rows):,} → 抽样 {len(sub):,}")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = M.TwoTowerMLP(dim=256, tok_hidden=128, drop=0.2, head_drop=0.3, coarse_t=128)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    def forward():
        return M.eval_forward(model, Xc, Xf, nv, sub, args.chunk, "cpu", K=128)

    base = forward()
    print(f"\n  as-is              cos={M.cos_score(base, yv):.5f}  (中心化 {cosm(base, yv):.5f})")

    keep = {}
    for tkey in ("tower_a", "tower_b"):
        keep[tkey] = (model.get_parameter(f"{tkey}.time.3.weight").detach().clone(),
                      model.get_parameter(f"{tkey}.read_w").detach().clone())

    def restore():
        for tkey, (w2, rw) in keep.items():
            with torch.no_grad():
                model.get_parameter(f"{tkey}.time.3.weight").copy_(w2)
                model.get_parameter(f"{tkey}.read_w").copy_(rw)

    arms = {}
    # ① colmean：抹掉行结构
    with torch.no_grad():
        for tkey in ("tower_a", "tower_b"):
            P = model.get_parameter(f"{tkey}.time.3.weight")
            P.copy_(P.mean(dim=0, keepdim=True).expand_as(P))
    arms["colmean"] = forward(); restore()
    # ② reinit：换一组同量级随机噪声
    with torch.no_grad():
        for tkey in ("tower_a", "tower_b"):
            P = model.get_parameter(f"{tkey}.time.3.weight")
            b = 1.0 / np.sqrt(P.shape[1])
            P.copy_(torch.empty_like(P).uniform_(-b, b))
    arms["reinit"] = forward(); restore()
    # ③ uni：读出严格均匀
    with torch.no_grad():
        for tkey in ("tower_a", "tower_b"):
            model.get_parameter(f"{tkey}.read_w").zero_()
    arms["uni"] = forward(); restore()

    print(f"\n  {'臂':<10} {'cos':>9} {'中心化cos':>10} {'与 as-is 的预测相关':>20}  {'|Δ|/σ_y'}")
    sd = base.std()
    for k, p in arms.items():
        print(f"  {k:<10} {M.cos_score(p, yv):9.5f} {cosm(p, yv):10.5f} "
              f"{uncentered(p, base):20.6f}  {np.abs(p - base).mean() / sd:.4f}")
    print(f"\n  （as-is 的预测 std={sd:.5f}；y std={yv.std():.5f}）")


if __name__ == "__main__":
    main()
