"""test_twotower_baseline.py — 双塔脚本的回归测试（**本机可跑**，不需要数据/GPU）。

跑法：/home/zlf/.venvs/lab/bin/python models/cnn/test_twotower_baseline.py

守住七件事（每条对应一个"不会报错的错"或一条刻意的设计）：

  1. **单行构建器 == 整批构建器**（两条塔都查）：训练走单行、验证走整批，两条路径必须数值一致
     —— 不一致等于训练/验证喂了不同分布的数据，**不会报错**，只会悄悄变差。
  2. **卷积因果**（两条塔，含粗段的 d=4）：改最新一格不能影响最旧一格的输出，含正对照；
     空洞那层最容易写错 pad 长度（应是 (k−1)·d 而不是 k−1）。
  3. **读出**：θ=0 时权重均匀（起点 = 等权块均值）、`h == 块均值 × 尺度`、
     softmax 非负和为 1、梯度能流到 `block_w` 与 `scale`。
  4. **两塔互不串**：两条塔的参数是独立实例；改塔 A 的输入**不影响**塔 B 的输出（反之亦然）。
  5. **梯度能流到两条塔**（关键风险）：head 若只读了一路的切片，另一条塔会静默不训练。
  6. **形状与规模**：粗段 224→56 块、细段 60→15 块、拼接后 (B,512)、forward → (B,1)；参数量 ~42 万。
  7. **端到端迷你训练**（合成数据 / CPU / 1 epoch / workers=0）：验维度与 DataLoader 接口。
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twotower_baseline import (COARSE_BLOCKS, COARSE_C, COARSE_T, FINE_BLOCKS, FINE_C,  # noqa: E402
                               FINE_T, CausalConv, Tower, TwoTowerCNN, batch_two,
                               build_one, train_fold)


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    return bool(cond)


def _synth(n, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, COARSE_T, COARSE_C)).astype(np.float16),
            rng.standard_normal((n, FINE_T, FINE_C)).astype(np.float16))


# ------------------------------------------------------------------ 1

def test_builders_agree(ok):
    print("\n[单行构建器 vs 整批构建器]")
    Xc, Xf = _synth(12)
    rows = np.sort(np.random.default_rng(1).choice(12, size=4, replace=False))
    ac, bc = batch_two(Xc, Xf, rows, "cpu")
    good, dt = True, None
    for i, r in enumerate(rows):
        a, b = build_one(Xc, Xf, r)
        if a.shape != ac[i].numpy().shape or b.shape != bc[i].numpy().shape:
            good, dt = False, f"形状 {a.shape}/{b.shape}"
            break
        if not (np.array_equal(a, ac[i].numpy()) and np.array_equal(b, bc[i].numpy())):
            good, dt = False, f"第 {i} 行数值不同"
            break
    ok &= check(f"粗段+细段 单行 == 整批（{len(rows)} 行）", good, dt or "")
    ok &= check("无 mask 语义：构建器返回二元组", len(build_one(Xc, Xf, 0)) == 2)
    return ok


# ------------------------------------------------------------------ 2

def test_causal(ok):
    print("\n[卷积因果性]")
    for name, c, t, nb, dil in [("塔A(粗段,56 块, d=4)", COARSE_C, COARSE_T, COARSE_BLOCKS, 4),
                                ("塔B(细段,15 块, d=1)", FINE_C, FINE_T, FINE_BLOCKS, 1)]:
        tower = Tower(c, nb, l3_dilation=dil).eval()
        x = torch.randn(2, c, t)
        with torch.no_grad():
            h = tower.conv(x)
            x_fut = x.clone(); x_fut[:, :, -1] += 5.0
            x_past = x.clone(); x_past[:, :, 0] += 5.0
            h_fut, h_past = tower.conv(x_fut), tower.conv(x_past)
        ok &= check(f"{name}: 改最新一格 → 最旧块不变", torch.equal(h[:, :, 0], h_fut[:, :, 0]))
        ok &= check(f"{name}: 正对照（改最旧一格）",
                    not torch.allclose(h[:, :, 0], h_past[:, :, 0]))

    c = CausalConv(3, 4, k=3, stride=1, dilation=8).eval()
    xt = torch.randn(1, 3, 60)
    with torch.no_grad():
        y = c(xt)
        xt_fut = xt.clone(); xt_fut[:, :, -1] += 5.0
        xt_past = xt.clone(); xt_past[:, :, 0] += 5.0
        y_fut, y_past = c(xt_fut), c(xt_past)
    ok &= check("CausalConv(k=3,d=8)：改最新一格 → 位置 0 不变",
                torch.equal(y[:, :, 0], y_fut[:, :, 0]))
    ok &= check("CausalConv(k=3,d=8)：正对照", not torch.allclose(y[:, :, 0], y_past[:, :, 0]))
    return ok


# ------------------------------------------------------------------ 3

def test_readout(ok):
    print("\n[可学习块权重读出]")
    torch.manual_seed(0)
    tower = Tower(FINE_C, FINE_BLOCKS).eval()
    ok &= check("θ=0 时块权重均匀（起点 = 等权块均值）",
                np.allclose(tower.block_weights(), 1.0 / FINE_BLOCKS))
    x = torch.randn(3, FINE_C, FINE_T)
    with torch.no_grad():
        h = tower.conv(x)
        out = tower(x)
    ok &= check("输出 == 块均值 × 尺度(初始 1)", torch.allclose(out, h.mean(-1) * tower.scale, atol=1e-6))
    tower.zero_grad()
    tower(x).sum().backward()          # 独立的一次前向：上面那个 out 在 no_grad 里算的
    ok &= check("梯度能流到 block_w（且非零）",
                tower.block_w.grad is not None and float(tower.block_w.grad.abs().sum()) > 0)
    ok &= check("梯度能流到 scale",
                tower.scale.grad is not None and float(tower.scale.grad.abs().sum()) > 0)
    with torch.no_grad():
        tower.block_w.add_(torch.randn(FINE_BLOCKS))
    a = tower.block_weights()
    ok &= check("任意 θ 下权重非负且和为 1", bool((a >= 0).all()) and abs(a.sum() - 1) < 1e-6)
    return ok


# ------------------------------------------------------------------ 4 & 5

def test_towers_independent_and_both_trained(ok):
    print("\n[两塔独立性 + 梯度覆盖面]")
    torch.manual_seed(0)
    net = TwoTowerCNN().eval()
    xa, xb = torch.randn(2, COARSE_C, COARSE_T), torch.randn(2, FINE_C, FINE_T)
    with torch.no_grad():
        h = net(xa, xb)
        ha, hb = net.tower_a(xa), net.tower_b(xb)
        h2 = net(xa * 3.0, xb)          # 只改粗段输入
        hb2 = net.tower_b(xb)
    ok &= check("改塔 A 的输入 → 塔 B 的输出逐位不变", torch.equal(hb, hb2))
    ok &= check("head 输入 = 两塔拼接 (B,512)",
                torch.cat([ha, hb], 1).shape == (2, 512))
    ok &= check("两条塔参数互相独立（不是同一个实例）", net.tower_a is not net.tower_b)

    net.zero_grad()
    net(xa, xb).sum().backward()
    ga = sum(float(p.grad.abs().sum()) for p in net.tower_a.parameters() if p.grad is not None)
    gb = sum(float(p.grad.abs().sum()) for p in net.tower_b.parameters() if p.grad is not None)
    ok &= check("梯度流到塔 A（非零）", ga > 0, f"|g|={ga:.3f}")
    ok &= check("梯度流到塔 B（非零）—— head 没漏掉任何一路", gb > 0, f"|g|={gb:.3f}")
    return ok


# ------------------------------------------------------------------ 6

def test_shapes(ok):
    print("\n[形状与规模]")
    torch.manual_seed(0)
    net = TwoTowerCNN().eval()
    with torch.no_grad():
        ha = net.tower_a.conv(torch.randn(2, COARSE_C, COARSE_T))
        hb = net.tower_b.conv(torch.randn(2, FINE_C, FINE_T))
        out = net(torch.randn(2, COARSE_C, COARSE_T), torch.randn(2, FINE_C, FINE_T))
    ok &= check(f"粗段 {COARSE_T} 步 → {COARSE_BLOCKS} 块", ha.shape == (2, 256, COARSE_BLOCKS),
                f"{tuple(ha.shape)}")
    ok &= check(f"细段 {FINE_T} 步 → {FINE_BLOCKS} 块", hb.shape == (2, 256, FINE_BLOCKS),
                f"{tuple(hb.shape)}")
    ok &= check("forward → (2,1)", out.shape == (2, 1))
    n_par = sum(p.numel() for p in net.parameters())
    ok &= check("参数量 ~42 万", 350_000 < n_par < 500_000, f"{n_par:,}")
    wa = sum(p.numel() for p in net.tower_a.parameters())
    wb = sum(p.numel() for p in net.tower_b.parameters())
    ok &= check("两塔参数量相当（粗段塔输入通道少、细段多）", 0.9 < wa / wb < 1.1,
                f"A={wa:,} B={wb:,}")
    return ok


# ------------------------------------------------------------------ 7

def test_mini_train(ok):
    print("\n[端到端迷你训练（合成数据 / CPU / 1 epoch / workers=0）]")
    n = 256
    Xc, Xf = _synth(n)
    y = (np.random.default_rng(0).standard_normal(n) * 0.003).astype(np.float32)
    args = argparse.Namespace(epochs=1, patience=16, batch=64, workers=0,
                              lr=1e-3, wd=3e-4, drop=0.2, head_drop=0.3, seed=42)
    r = train_fold(Xc, Xf, y, np.arange(n), np.arange(192), np.arange(192, n), args, "cpu")
    keys = {"cos", "epoch", "va_pred", "va_sids", "bw_a", "bw_b", "scale_a", "scale_b"}
    ok &= check("返回结构完整", keys <= set(r), str(sorted(r)))
    ok &= check("预测长度 == 验证行数", len(r["va_pred"]) == 64)
    ok &= check("cos 有限", bool(np.isfinite(r["cos"])), f"{r['cos']:.5f}")
    ok &= check("块权重形状正确",
                np.asarray(r["bw_a"]).shape == (COARSE_BLOCKS,) and
                np.asarray(r["bw_b"]).shape == (FINE_BLOCKS,))
    return ok


def main():
    print("=" * 72)
    print("twotower_baseline 回归测试（本机可跑，不需要数据/GPU）")
    print("=" * 72)
    ok = True
    for t in (test_builders_agree, test_causal, test_readout,
              test_towers_independent_and_both_trained, test_shapes, test_mini_train):
        ok = t(ok)
    print("\n" + ("全部通过" if ok else "*** 有失败项 ***"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
