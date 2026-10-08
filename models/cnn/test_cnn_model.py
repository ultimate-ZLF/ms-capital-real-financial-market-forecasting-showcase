"""test_cnn_model.py — 一代旧主干 `SnapCNN` 的 τ 池化回归测试。

**本机即可跑，不需要数据**：`python models/cnn/test_cnn_model.py`

守的是什么（两条都是"看着像 bug、其实是收益"的机制）：
- **`legacy` 与 `mask` 必须行为不同**——否则隔离实验无效，"τ 无影响"会是个假结论。
- **`legacy` 的脆弱性**：它依赖 pad 全零，无效区取非零值会改变池化。这条要留着——
  它正是 CLAUDE.md #10 说的三个耦合条件之一，改数据布局/卷积结构时会被它拦住。
- **τ 仍在强调近端**（没有加权错端）。
- **满格样本上 legacy == mask**（无无效块时两口径应等价）。

⚠️ 数据管道（`batch_to_tensor` / `build_snap_one` / Loader / 折式）的测试在
`seq/test_seq_common.py`——两层已分家，别再混回来。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                     # cnn_model
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "seq"))                         # seq_common

from seq_common import F, T  # noqa: E402
from cnn_model import SnapCNN  # noqa: E402


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    return cond


def _tau_probe(mode, nv, base, mask):
    torch.manual_seed(0)
    m = SnapCNN(pool_tau=64.0, tau_mode=mode).eval()
    with torch.no_grad():
        h0 = m.encode(torch.from_numpy(base.copy()), mask)
        x2 = base.copy()
        x2[0, :, nv:] = 999.0
        d_pad = float((h0 - m.encode(torch.from_numpy(x2), mask)).abs().max())
    return h0, d_pad


def test_tau_modes(ok):
    """legacy 与 mask 必须行为不同——否则隔离实验无效。

    跑对照的前提是：两种模式下**只有 τ 池化这一步不同**，其他一切逐位相同。
    如果某个开关没接对（比如代码里没生效），两个数字会一样，实验会得出"τ 无影响"的错误结论。
    """
    print("\n[τ 池化两口径：legacy（默认） vs mask（对照）]")
    nv = 189
    mask = torch.from_numpy(np.arange(T)[None, :] < np.array([nv])[:, None])
    rng = np.random.default_rng(0)
    base = np.zeros((1, F, T), dtype=np.float32)
    base[0, :, :nv] = rng.normal(size=(F, nv))

    outs = {}
    for mode, expect_leak in (("legacy", True), ("mask", False)):
        outs[mode], d_pad = _tau_probe(mode, nv, base, mask)
        ok &= check(f"{mode:6s} 无效区{'影响' if expect_leak else '不影响'}池化",
                    (d_pad > 0) == expect_leak, f"max|Δh|={d_pad:.4f}")
    ok &= check("legacy 确实不同于 mask（两种口径都生效）",
                not torch.allclose(outs["legacy"], outs["mask"]),
                f"max|Δ|={float((outs['legacy'] - outs['mask']).abs().max()):.4f}")

    # —— 曾经提过的第三种口径 "align"（权重从最新【有效】块起算）已删除 ——
    # 它被证明与 "mask" **恒等**，理由是数学的：
    #     mask : w_b = exp(-(nb-1-b)/τ')
    #     align: w_b = exp(-(last-b)/τ')
    # 两者只差常数因子 exp(-(nb-1-last)/τ')，在归一化加权平均里抵消。
    # 结论：**「τ 衰减从哪一块开始」在归一化下不可观测**——任何"把权重对齐一下"的
    # 改动都是空操作。真正能改变结果的只有"分子里包含哪些块"。
    # 把这条断言留在测试里，是为了让下一个想"对齐 τ"的人先撞到它。
    m_full = torch.ones(1, T, dtype=torch.bool)
    full = np.zeros((1, F, T), dtype=np.float32)
    full[0] = rng.normal(size=(F, T))
    h_l, _ = _tau_probe("legacy", T, full, m_full)
    h_m, _ = _tau_probe("mask", T, full, m_full)
    ok &= check("满格样本（无 pad）上 legacy == mask（无无效块时两口径应等价）",
                torch.allclose(h_l, h_m),
                f"max|Δ|={float((h_l - h_m).abs().max()):.6f}")

    # 非满格样本上 legacy 的池化**会**受无效区取值影响——这正是它的脆弱点，
    # 也是 WALKTHROUGH8 记录的"依赖 pad 必须是零"那条耦合。
    ok &= check("legacy 的脆弱性：无效区取非零值会改变池化",
                _tau_probe("legacy", nv, base, mask)[1] > 0)
    return ok


def test_tau_pooling(ok):
    """τ 池化的两条性质：**新鲜度加权仍在**，以及两种口径对无效区的敏感度不同。

    - pool_tau=0（等权）走 masked_pool，**不碰**无效区 ✓
    - pool_tau>0 + tau_mode="mask"：分子分母同口径，不碰无效区 ✓
    - pool_tau>0 + tau_mode="legacy"（默认）：**会**碰——分子含无效块。这是它已知的
      脆弱点（依赖 pad 是全零），但实测比 mask 高 0.044，所以仍是默认。见 WALKTHROUGH8.md。

    另外确认 τ **没有**加权错端：实测近端敏感度高于远端（pad 在最新那端，而因果卷积
    在 pad 位的输出会向前整合最近的真数据，所以 τ 依然在强调近端）。这条要守住——
    否则 τ 就退化成噪声。
    """
    print("\n[τ 池化：新鲜度加权 + 无效区敏感度]")
    nv = 189
    mask = torch.from_numpy(np.arange(T)[None, :] < np.array([nv])[:, None])
    rng = np.random.default_rng(0)
    base = np.zeros((1, F, T), dtype=np.float32)
    base[0, :, :nv] = rng.normal(size=(F, nv))

    cases = [(64.0, "legacy", True), (64.0, "mask", False), (0.0, "legacy", False)]
    for tau, mode, expect_leak in cases:
        torch.manual_seed(0)
        m = SnapCNN(pool_tau=tau, tau_mode=mode).eval()
        with torch.no_grad():
            h0 = m.encode(torch.from_numpy(base.copy()), mask)
            x2 = base.copy()
            x2[0, :, nv:] = 999.0                        # 只动无效区（pad）
            d_pad = float((h0 - m.encode(torch.from_numpy(x2), mask)).abs().max())
        ok &= check(f"pool_tau={tau:5.1f} mode={mode:6s} 无效区"
                    f"{'影响' if expect_leak else '不影响'}池化",
                    (d_pad > 0) == expect_leak, f"max|Δh|={d_pad:.6f}")

    for tau in (64.0, 32.0):
        torch.manual_seed(0)
        m = SnapCNN(pool_tau=tau).eval()
        with torch.no_grad():
            h0 = m.encode(torch.from_numpy(base.copy()), mask)
            xo = base.copy()
            xo[0, :, :20] += 5.0                         # 最旧 20 行
            xn = base.copy()
            xn[0, :, nv - 20:nv] += 5.0                  # 最新 20 行
            d_old = float((h0 - m.encode(torch.from_numpy(xo), mask)).abs().max())
            d_new = float((h0 - m.encode(torch.from_numpy(xn), mask)).abs().max())
        ok &= check(f"pool_tau={tau:5.1f} 仍强调近端（新/旧敏感度 > 1）", d_new > d_old,
                    f"新={d_new:.4f} 旧={d_old:.4f} 比={d_new / max(d_old, 1e-9):.2f}")
    return ok


def main():
    print("=" * 68)
    print("SnapCNN τ 池化回归测试（合成数组，本机可跑）")
    print("=" * 68)
    ok = True
    ok = test_tau_pooling(ok)
    ok = test_tau_modes(ok)
    print(f"\n{'全部通过' if ok else '存在失败项'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
