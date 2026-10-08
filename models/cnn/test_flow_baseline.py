"""test_flow_baseline.py — flow_baseline（新主干）的回归测试（**本机可跑**，不需要数据/GPU）。

跑法：/home/zlf/.venvs/lab/bin/python models/cnn/test_flow_baseline.py

守住八件事（每一条都对应一个"不会报错的错"或一条刻意的设计）：

  1. **单行构建器 == 整批构建器**（细段三源 + 粗段）：训练走单行、验证走整批，两条路径
     必须数值一致——不一致等于训练和验证喂了不同分布的数据，**不会报错**，只会悄悄变差。
  2. **细段无 pad**：60 步原样返回，不补零不截断。
  3. **粗段掩码语义**：mask = 位置 < n_valid，且 pad 在**最新端**（有效位在开头）。
  4. **卷积因果**：改最新一格不能影响最旧一格（未来不许泄漏到过去），含正对照；
     空洞那层单独再守一遍（`(k−1)·d` 最容易写错）。
  5. **读出**：θ=0 时权重均匀、`encode` == 块均值 × 尺度、梯度能流到 `block_w`/`scale`、
     softmax 和为 1。
  6. **掩码两口径**：`hard` 在**全有效**掩码下与 `none` **逐位相同**（细段三臂不受该开关影响）；
     `hard` 下无效块的权重**恒为 0**。
  7. **形状与规模**：60 步 → 15 块、224 步 → 56 块；C=7/16/23/15 都能过。
  8. **C-12 读数**：`pad_weight_mass` 的手算对照（这是粗段 `none` 口径的核心产出）。
  9. **端到端**：细段与粗段各跑一个迷你训练（合成数据、CPU、1 epoch、workers=0）。
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flow_baseline import (COARSE_T, FINE_T, SRC, LearnPoolCNN,  # noqa: E402
                           batch_to_tensor, build_input_one, pad_weight_mass, train_fold)

NB_FINE = SRC["flow"]["blocks"]        # 15
NB_COARSE = SRC["coarse"]["blocks"]    # 56
C_COARSE = SRC["coarse"]["c"]          # 15


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    return bool(cond)


def _srs_args(source, **kw):
    d = dict(source=source, epochs=1, patience=16, batch=64, workers=0,
             lr=1e-3, wd=3e-4, drop=0.2, head_drop=0.3, seed=42, block_mask="hard")
    d.update(kw)
    return argparse.Namespace(**d)


# ------------------------------------------------------------------ 1 & 2

def test_builders_agree(ok):
    print("\n[单行构建器 vs 整批构建器]")
    rng = np.random.default_rng(0)
    n = 12
    # ⚠️ `mkt` / `both` 两臂已于 2026-09-26 随 tri60 代码移除（`SRC` 里已无这两项）
    for source, ks, t in [("flow", [16], FINE_T), ("coarse", [C_COARSE], COARSE_T)]:
        arrays = [rng.standard_normal((n, t, k)).astype(np.float16) for k in ks]
        nv = None if source != "coarse" else rng.integers(1, t + 1, size=n).astype(np.int32)
        rows = np.sort(rng.choice(n, size=4, replace=False))
        bx, bm = batch_to_tensor(arrays, nv, rows, "cpu")
        good, dt = True, None
        for i, r in enumerate(rows):
            sx, sm = build_input_one(arrays, nv, r)
            if sx.shape != bx[i].numpy().shape:
                good, dt = False, f"形状 {sx.shape} vs {tuple(bx[i].shape)}"
                break
            if not np.array_equal(sx, bx[i].numpy()):
                good, dt = False, f"第 {i} 行数值不同（max|Δ|={np.abs(sx - bx[i].numpy()).max():.3e}）"
                break
            if not np.array_equal(sm, bm[i].numpy()):
                good, dt = False, "mask 不同"
                break
        ok &= check(f"{source}: 单行 == 整批（{len(rows)} 行，C={sum(ks)}，T={t}）", good, dt or "")
    return ok


def test_no_pad(ok):
    print("\n[细段输入无 pad]")
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, FINE_T, 16)).astype(np.float16)
    x, m = build_input_one([a], None, 1)
    ok &= check("形状 = (C,60)", x.shape == (16, FINE_T), f"{x.shape}")
    ok &= check("数值与源逐位相同（无补零/截断/重排）",
                np.array_equal(x, a[1].T.astype(np.float32)))
    ok &= check("mask 全 True（60 格全部有效）", bool(m.all()) and m.shape == (FINE_T,))
    b = rng.standard_normal((3, FINE_T, 7)).astype(np.float16)
    x2, _ = build_input_one([a, b], None, 2)
    ok &= check("both：通道顺序 flow 在前、market 在后",
                np.array_equal(x2[:16], a[2].T.astype(np.float32)) and
                np.array_equal(x2[16:], b[2].T.astype(np.float32)))
    return ok


# ------------------------------------------------------------------ 3

def test_coarse_mask(ok):
    print("\n[粗段掩码语义]")
    rng = np.random.default_rng(2)
    nv = np.array([17, 100, 224, 1], dtype=np.int32)
    a = rng.standard_normal((4, COARSE_T, C_COARSE)).astype(np.float16)
    for i, n in enumerate(nv):
        x, m = build_input_one([a], nv, i)
        exp = np.arange(COARSE_T) < n
        ok &= check(f"n_valid={n:3d}: mask = 位置 < n_valid", np.array_equal(m, exp))
        if i == 1:
            ok &= check("真数据原样返回（pad 在最新端）",
                        np.array_equal(x, a[1].T.astype(np.float32)))
    xv, mv = batch_to_tensor([a], nv, np.arange(4), "cpu")
    ok &= check("整批 mask 行数 = n_valid", (mv.numpy().sum(1) == nv).all())
    return ok


# ------------------------------------------------------------------ 4

def test_causal(ok):
    print("\n[卷积因果性：未来不泄漏到过去]")
    for name, c, t, nb, dil in [("细段(15 块, d=1)", 16, FINE_T, NB_FINE, 1),
                                ("粗段(56 块, d=4)", C_COARSE, COARSE_T, NB_COARSE, 4)]:
        net = LearnPoolCNN(c, n_blocks=nb, l3_dilation=dil).eval()
        x = torch.randn(2, c, t)
        with torch.no_grad():
            h = net.conv(x)
            x_fut = x.clone(); x_fut[:, :, -1] += 5.0
            h_fut = net.conv(x_fut)
            x_past = x.clone(); x_past[:, :, 0] += 5.0
            h_past = net.conv(x_past)
        ok &= check(f"{name}: 改最新一格 → 最旧块不变", torch.equal(h[:, :, 0], h_fut[:, :, 0]))
        ok &= check(f"{name}: 正对照（改最旧一格）",
                    not torch.allclose(h[:, :, 0], h_past[:, :, 0]))

    from flow_baseline import CausalConv
    c = CausalConv(3, 4, k=3, stride=1, dilation=8).eval()
    xt = torch.randn(1, 3, 60)
    with torch.no_grad():
        y = c(xt)
        x_fut = xt.clone(); x_fut[:, :, -1] += 5.0
        x_past = xt.clone(); x_past[:, :, 0] += 5.0
        y_fut, y_past = c(x_fut), c(x_past)
    ok &= check("CausalConv(k=3,d=8)：改最新一格 → 位置 0 不变",
                torch.equal(y[:, :, 0], y_fut[:, :, 0]))
    ok &= check("CausalConv(k=3,d=8)：正对照", not torch.allclose(y[:, :, 0], y_past[:, :, 0]))
    return ok


# ------------------------------------------------------------------ 5

def test_readout(ok):
    print("\n[可学习块权重读出]")
    torch.manual_seed(0)
    net = LearnPoolCNN(16, n_blocks=NB_FINE, block_mask="none").eval()
    ok &= check("θ=0 时块权重是均匀分布（起点 = 等权块均值）",
                np.allclose(net.block_weights(), 1.0 / NB_FINE))
    x = torch.randn(3, 16, FINE_T)
    with torch.no_grad():
        h = net.conv(x)
        enc = net.encode(x)
    ok &= check("encode == 块均值 × 尺度(初始 1)",
                torch.allclose(enc, h.mean(-1) * net.scale, atol=1e-6))
    net.zero_grad()
    net(x).sum().backward()
    ok &= check("梯度能流到 block_w（且非零）",
                net.block_w.grad is not None and float(net.block_w.grad.abs().sum()) > 0)
    ok &= check("梯度能流到 scale（尺度自由度没被掐掉）",
                net.scale.grad is not None and float(net.scale.grad.abs().sum()) > 0)
    with torch.no_grad():
        net.block_w.add_(torch.randn(NB_FINE))
    a = net.block_weights()
    ok &= check("任意 θ 下 softmax 权重非负且和为 1",
                bool((a >= 0).all()) and abs(a.sum() - 1.0) < 1e-6)
    return ok


# ------------------------------------------------------------------ 6

def test_block_mask_modes(ok):
    print("\n[掩码两口径 hard vs none]")
    torch.manual_seed(0)
    torch.manual_seed(0)
    m_none = LearnPoolCNN(16, n_blocks=NB_FINE, block_mask="none").eval()
    torch.manual_seed(0)
    m_hard = LearnPoolCNN(16, n_blocks=NB_FINE, block_mask="hard").eval()
    x = torch.randn(2, 16, FINE_T)
    allv = torch.ones(2, FINE_T, dtype=torch.bool)
    with torch.no_grad():
        a, b = m_none.encode(x, allv), m_hard.encode(x, allv)
    ok &= check("全有效掩码下 hard 与 none **逐位相同**（细段三臂不受该开关影响）",
                torch.equal(a, b), f"max|Δ|={float((a-b).abs().max()):.3e}")
    # 前 40 位无效 → 前 10 个块（每块 4 位）全无效
    mask = torch.ones(1, FINE_T, dtype=torch.bool)
    mask[0, :40] = False
    w = m_hard.block_weights(mask)          # (1, nb)
    ok &= check("hard：无效块权重恒为 0", bool((w[0, :10] == 0).all()),
                f"前 10 块和={float(w[0, :10].sum()):.3e}")
    ok &= check("hard：逐样本权重和为 1", abs(float(w.sum()) - 1.0) < 1e-6)
    bv = LearnPoolCNN.block_valid(mask)
    ok &= check("block_valid：块 9 无效、块 10 有效（边界对齐 4 位）",
                (not bool(bv[0, 9])) and bool(bv[0, 10]))
    w_none = m_none.block_weights(mask)
    ok &= check("none：掩码不影响权重（一维）", np.asarray(w_none).ndim == 1)
    return ok


# ------------------------------------------------------------------ 7

def test_shapes(ok):
    print("\n[形状与规模]")
    torch.manual_seed(0)
    for source, c, t, nb in [("flow", 16, FINE_T, NB_FINE),
                             ("coarse", C_COARSE, COARSE_T, NB_COARSE)]:
        net = LearnPoolCNN(c, n_blocks=nb, l3_dilation=SRC[source]["l3_dil"]).eval()
        xx = torch.randn(2, c, t)
        with torch.no_grad():
            h = net.conv(xx)
            out = net(xx)
        ok &= check(f"{source:6s}: {t} 步 → {nb} 块", h.shape == (2, 256, nb), f"{tuple(h.shape)}")
        ok &= check(f"{source:6s}: forward → (2,1)", out.shape == (2, 1))
    n_fine = sum(p.numel() for p in LearnPoolCNN(16, n_blocks=NB_FINE).parameters())
    n_coarse = sum(p.numel() for p in LearnPoolCNN(C_COARSE, n_blocks=NB_COARSE).parameters())
    ok &= check("细段参数量 ~21 万", 150_000 < n_fine < 300_000, f"{n_fine:,}")
    ok &= check("粗段参数量 ~21 万", 150_000 < n_coarse < 300_000, f"{n_coarse:,}")
    return ok


# ------------------------------------------------------------------ 8

def test_pad_weight_mass(ok):
    print("\n[C-12 读数：pad 块权重质量]")
    # 2 条样本：n_valid = 4 与 224（T=224）
    mask = torch.zeros(2, COARSE_T, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, :] = True
    a = np.full(NB_COARSE, 1.0 / NB_COARSE)        # 均匀权重
    # 块 0 在两条样本里都有效；块 1..55 只在第 2 条有效 → 平均无效比例 0.5
    expect = 55 * (1.0 / NB_COARSE) * 0.5
    got = pad_weight_mass(a, mask)
    ok &= check(f"手算对照（期望 {expect:.5f}）", abs(got - expect) < 1e-9, f"实得 {got:.5f}")
    # 权重全压在最后一个块上 → pad 质量应 ≈ 0.5
    a2 = np.zeros(NB_COARSE); a2[-1] = 1.0
    ok &= check("权重集中在末块时 pad 质量 ≈ 0.5", abs(pad_weight_mass(a2, mask) - 0.5) < 1e-9)
    return ok


# ------------------------------------------------------------------ 9

def test_mini_train(ok):
    print("\n[端到端迷你训练（合成数据 / CPU / 1 epoch / workers=0）]")
    rng = np.random.default_rng(0)
    for source, c, t, nb in [("flow", 16, FINE_T, NB_FINE), ("coarse", C_COARSE, COARSE_T, NB_COARSE)]:
        n = 256
        arrays = [rng.standard_normal((n, t, c)).astype(np.float16)]
        y = (rng.standard_normal(n) * 0.003).astype(np.float32)
        nv = None if source == "flow" else rng.integers(t // 2, t + 1, size=n).astype(np.int32)
        args = _srs_args(source, block_mask="none" if source == "coarse" else "hard")
        r = train_fold(arrays, nv, y, np.arange(n), np.arange(192), np.arange(192, n),
                       args, "cpu")
        keys = {"cos", "epoch", "va_pred", "va_sids", "block_w", "pad_mass"}
        ok &= check(f"{source}: 返回结构完整", keys <= set(r), str(sorted(r)))
        ok &= check(f"{source}: 预测长度 == 验证行数", len(r["va_pred"]) == 64)
        ok &= check(f"{source}: cos 有限", bool(np.isfinite(r["cos"])), f"{r['cos']:.5f}")
        ok &= check(f"{source}: 块权重 {nb} 个", np.asarray(r["block_w"]).shape == (nb,))
        if source == "coarse":
            ok &= check("粗段 none 口径：pad_mass ∈ [0,1]",
                        0.0 <= r["pad_mass"] <= 1.0, f"{r['pad_mass']:.4f}")
    return ok


def main():
    print("=" * 72)
    print("flow_baseline（新主干：空洞卷积 + 可学习池化）回归测试")
    print("=" * 72)
    ok = True
    for t in (test_builders_agree, test_no_pad, test_coarse_mask, test_causal,
              test_readout, test_block_mask_modes, test_shapes, test_pad_weight_mass,
              test_mini_train):
        ok = t(ok)
    print("\n" + ("全部通过" if ok else "*** 有失败项 ***"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
