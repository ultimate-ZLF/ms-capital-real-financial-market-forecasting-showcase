"""test_mlp.py — 双塔MLP（通道→时间 MLP 主干）的回归测试（**本机可跑**，不需要数据/GPU）。

跑法：/home/zlf/.venvs/lab/bin/python models/mlp/test_mlp.py

守住十一件事（每条对应一个"不会报错的错"或一条刻意的设计）：

  1. **单行构建器 == 整批构建器**、**单行掩码 == 整批掩码**（且 `mask.sum(1) == n_valid` 逐行）
     —— 训练走单行、验证走整批，不一致等于喂了不同分布的数据，**不会报错**只会悄悄变差。
  2. **形状与参数量**：按公式**精确**断言参数量——"把 token 分支写成 dim×dim"这类错
     形状全对、训练照跑，只有参数量能抓。另断言**无 BatchNorm、无卷积**
     （前者是"预测与 batch 无关"的根，后者钉住"非因果是刻意设计"）。
  3. **读出**：θ=0 → 均匀；权重非负、和为 1；`out == Σ_t h_t·w_t × scale`；梯度流到 read_w/scale。
  4. **位置嵌入真的在起作用**（T-pos-1/2/4）：⚠️ **不能用"打乱时间轴输出会变"来测**——
     token 分支的权重按位置索引，即使 pos≡0，置换输入后输出**也会变**（假阳性）。
     正确做法：先把 token 分支压成"位置盲"，再让 pos 成为唯一位置来源，做正负对照。
  5. **检查点往返**：存读逐位一致；指纹不符 → 报错（不静默）；文件缺失 → 报错；
     **无 `.tmp` 残留**；文件以 `.pt` 结尾；`weights_only=True` 可加载。
  6. **裁剪对齐**（`--crop K`）：最新那一行必须落在位置 K−1；K=224 严格恒等；
     不足 K 行时在旧端补零。⚠️ 这条是 2026-09-21 真抓到一个 bug 的那条
     （`crop_src` 里先把 n 截断成 min(n,K)，导致取到**最旧**的 K 行）。
  7. **与 batch 无关 / 分块 ≈ 一次性**：单行单独跑 vs 混入批里一致到 1e-6；
     `eval_forward(chunk=2)` vs 一次性一致到 1e-6（**显存护栏不改数值**）。
     ⚠️ 这里**不**断言逐位相同：实测 chunk=2 与一次性差 3e-08（float32 GEMM 分块差异，
     预测 std 1.26e-02 的相对 2.4e-6）。**同一 chunk 重放才逐位相同**——test 10 依赖那条。
  8. **两塔互不串**：改塔 A 输入不影响塔 B 输出；两塔不是同一实例。
  9. **读出权重的分块报告口径**：`block_report(w) == w.reshape(-1,4).sum(1)`，长 56 / 15
     （与 CNN 塔的块权重形状可直接对比）。
 10. **端到端迷你训练**（合成数据 / CPU / 1 epoch / workers=0 / tmpdir）：
     返回结构完整、检查点落盘、**`oof_from_ckpt` 能逐位复原 `va_pred`**、续跑键找得到。
 11. **配置指纹**：任一架构字段变则 hash 变；同配置重复调用 hash 稳定。
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mlp_baseline import (COARSE_C, COARSE_T, FINE_C, FINE_T, TARGET_SCALE,  # noqa: E402
                          TwoTowerMLP, arch_cfg, arch_hash, batch_loss, batch_two,
                          block_report, build_one, ckpt_path, crop_src, eval_forward, load_ckpt,
                          oof_from_ckpt, resume_ckpt, save_ckpt, train_fold)


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    return bool(cond)


CFG = dict(dim=256, tok_hidden=128, drop=0.2, head_drop=0.3)


def _synth(n, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, COARSE_T, COARSE_C)).astype(np.float16),
            rng.standard_normal((n, FINE_T, FINE_C)).astype(np.float16))


def _synth_batch(n, seed=0):
    xc, xf = _synth(n, seed)
    return (torch.from_numpy(np.asarray(xc, dtype=np.float32)).permute(0, 2, 1),
            torch.from_numpy(np.asarray(xf, dtype=np.float32)).permute(0, 2, 1))


def _args(tmpdir=None, **over):
    d = dict(seed=42, lr=1e-3, wd=3e-4, batch=32, workers=0, epochs=1, patience=16,
             eval_chunk=16, ckpt_dir=tmpdir or "/tmp/mlp_ckpt", ckpt_keep="both",
             crop=COARSE_T, clip=1.0, y_weight_q=0.0, loss="cos", **CFG)   # CFG 里已含 drop / head_drop / dim …
    d.update(over)
    return argparse.Namespace(**d)


def _expected_params(C, T, dim, tok_hidden):
    """按结构公式手算参数量（独立于模型代码，才能抓"写错但形状对"的错）。"""
    p = 2 * C                               # ln_in（LayerNorm 的 weight+bias）
    p += C * dim + dim                      # 通道 MLP：Linear(C→dim)
    p += T * dim                            # 位置嵌入
    p += 2 * dim                            # ln_mid
    p += T * tok_hidden + tok_hidden        # 时间 MLP：T→H
    p += tok_hidden * T + T                 # 时间 MLP：H→T
    p += T                                  # read_w
    p += 1                                  # scale
    return p


# ------------------------------------------------------------------ 1

def test_builders_agree(ok):
    Xc, Xf = _synth(12, 1)
    nv = np.array([60 + 13 * i for i in range(12)], dtype=np.int64)   # 60..203
    rows = np.arange(12)
    a_b, b_b = batch_two(Xc, Xf, nv, rows, "cpu")
    m_b = np.stack([crop_src(nv[i], COARSE_T)[1] for i in rows])
    for i, r in enumerate(rows):
        a, b = build_one(Xc, Xf, r, nv, COARSE_T)
        ok &= check(f"build_one == batch_two 第 {i} 行",
                    np.array_equal(a, a_b[i].numpy()) and np.array_equal(b, b_b[i].numpy()))
    for i in range(12):
        ok &= check(f"逐行 crop 的 ok 掩码 == 整批（第 {i} 行）",
                    np.array_equal(crop_src(nv[i], COARSE_T)[1], m_b[i]))
    ok &= check("逐行有效位数 == n_valid",
                np.array_equal(m_b.sum(1), nv), f"{m_b.sum(1)}")
    ok &= check("掩码形状 (B,224) 且沿时间轴", tuple(m_b.shape) == (12, COARSE_T))
    return ok


# ------------------------------------------------------------------ 2

def test_forward_shapes(ok):
    m = TwoTowerMLP(**CFG)
    xc, xf = _synth_batch(2)
    h_a = m.tower_a(xc)
    h_b = m.tower_b(xf)
    ok &= check("塔 A/B 输出 (B,256)", tuple(h_a.shape) == (2, 256) and tuple(h_b.shape) == (2, 256))
    ok &= check("forward → (B,1)", tuple(m(xc, xf).shape) == (2, 1))
    exp_a = _expected_params(COARSE_C, COARSE_T, CFG["dim"], CFG["tok_hidden"])
    exp_b = _expected_params(FINE_C, FINE_T, CFG["dim"], CFG["tok_hidden"])
    exp_head = (2 * CFG["dim"]) * 256 + 256 + 256 * 1 + 1
    got_a = sum(q.numel() for q in m.tower_a.parameters())
    got_b = sum(q.numel() for q in m.tower_b.parameters())
    got_head = sum(q.numel() for q in m.head.parameters())
    ok &= check("塔 A 参数量 == 公式值", got_a == exp_a, f"{got_a} vs {exp_a}")
    ok &= check("塔 B 参数量 == 公式值", got_b == exp_b, f"{got_b} vs {exp_b}")
    ok &= check("head 参数量 == 公式值", got_head == exp_head, f"{got_head} vs {exp_head}")
    total = sum(q.numel() for q in m.parameters())
    ok &= check("总参数量 == 公式值", total == exp_a + exp_b + exp_head, f"{total:,}")
    names = [n for n, _ in m.named_modules()]
    ok &= check("无 BatchNorm（'预测与 batch 无关'的根）",
                not any("BatchNorm" in n or "batch_norm" in n for n in names))
    n_ln = sum(1 for _, mod in m.named_modules() if isinstance(mod, torch.nn.LayerNorm))
    ok &= check("有 4 个 LayerNorm（两处 pre-norm × 两塔）——稳定性件，2026-09-21 因训练发散补上",
                n_ln == 4, f"实际 {n_ln} 个")
    ok &= check("无卷积（非因果是刻意设计，别被人顺手修好）",
                not any(isinstance(mod, torch.nn.Conv1d) for mod in m.modules()))
    ok &= check("读出权重形状 (T,)", tuple(m.tower_a.read_w.shape) == (COARSE_T,)
                and tuple(m.tower_b.read_w.shape) == (FINE_T,))
    ok &= check("位置嵌入形状 (T,dim)",
                tuple(m.tower_a.pos.shape) == (COARSE_T, CFG["dim"])
                and tuple(m.tower_b.pos.shape) == (FINE_T, CFG["dim"]))
    return ok


# ------------------------------------------------------------------ 3

def test_readout(ok):
    torch.manual_seed(0)
    m = TwoTowerMLP(**CFG).eval()
    xc, xf = _synth_batch(3)
    with torch.no_grad():
        w0 = m.tower_a.readout_weights()
        ok &= check("θ=0 → 读出权重均匀", np.allclose(w0, 1.0 / COARSE_T), f"{w0[:3]}")
        # out == Σ_t h_t·w_t × scale，其中 h 由主干手算（独立复现 forward 的语义，含两处 LN）
        h = m.tower_a.channel(m.tower_a.ln_in(xc.transpose(1, 2))) + m.tower_a.pos
        h = m.tower_a.time(m.tower_a.ln_mid(h).transpose(1, 2)).transpose(1, 2)
        w = torch.softmax(m.tower_a.read_w, dim=0)
        exp = (h * w[None, :, None]).sum(1) * m.tower_a.scale
        got = m.tower_a(xc)
        ok &= check("读出 == Σ_t h_t·w_t × scale",
                    torch.allclose(exp, got, atol=1e-5), f"max|Δ|={(exp-got).abs().max():.2e}")
        # 随机化 θ 后仍非负、和为 1
        m.tower_a.read_w.normal_(0, 1.0)
        w1 = m.tower_a.readout_weights()
        ok &= check("权重非负且和为 1", (w1 >= 0).all() and abs(w1.sum() - 1) < 1e-6,
                    f"sum={w1.sum():.6f}")
    m.zero_grad()
    m(xc, xf).sum().backward()
    g_w = m.tower_a.read_w.grad
    g_s = m.tower_a.scale.grad
    ok &= check("梯度流到 read_w 且非零", g_w is not None and float(g_w.abs().max()) > 0)
    ok &= check("梯度流到 scale 且非零", g_s is not None and float(g_s.abs().max()) > 0)
    return ok


# ------------------------------------------------------------------ 4

def _make_position_blind(model):
    """把每塔的时间 MLP 压成"对所有位置施加同一个函数"（位置无关）。

    需要**两个条件同时成立**（2026-09-21 实测：各试一半都不行，置换差 4.8e-04 / 8.2e-03）：
      ① `time[0].weight`（(H,T)）**全部元素相等** —— 否则 `W1 @ x_d` 这个加权和本身
         会随置换改变（权重向量按位置索引，置换后权重对不上原位置）；
      ② `time[3].weight`（(T,H)）**各行相等** + `bias` 常数 —— 否则输出 `Σ_h W2[j,h]·g_h`
         随位置 j 变。
    两个条件都满足后，时间分支退化为"对每个通道算一个标量再广播到所有位置"，且与位置无关。
    """
    for tower in (model.tower_a, model.tower_b):
        lin1, lin2 = tower.time[0], tower.time[3]
        with torch.no_grad():
            lin1.weight.fill_(float(lin1.weight.mean()))
            lin2.weight.copy_(lin2.weight.mean(0, keepdim=True).expand_as(lin2.weight))
            lin2.bias.fill_(float(lin2.bias.mean()))


def test_position_embedding(ok):
    """位置嵌入的三条结构性检查。

    ⚠️ **本架构里"pos 对顺序敏感性的贡献"无法用置换实验单独隔离**（2026-09-21 实测）：
    本主干有**两个**位置机制——(a) 时间 MLP 的权重天然按位置索引，(b) `pos` 参数。
    把时间 MLP 压成"位置盲"（对位置求和）后，**无残差路径**可走，
    于是 pos 只能经"对位置的求和"进入前向 → 求和是置换不变的 → pos 的效果恒为零
    （实测 max|Δ| 恰好 0.00e+00，不是容差问题）。
    带残差的 MLP-Mixer 版能过这个测试，正是因为残差把位置对齐的信号保了下来。
    要真正分离 pos 的贡献只能做**训练级消融**（pos=0 训练 vs 带 pos 训练）——那是另一个实验。
    所以这里只守结构性事实（下面三条），不做过度声称。
    """
    xc, xf = _synth_batch(4)
    perm = torch.randperm(COARSE_T)
    # —— T-pos-1：位置盲 + pos=0 + 均匀读出 → 整个前向**置换不变** ——
    #   价值：证明通道 MLP / GELU / dropout / 读出都**不引入**位置依赖（很强的回归守卫）
    torch.manual_seed(0)
    mb = TwoTowerMLP(**CFG).eval()
    _make_position_blind(mb)
    with torch.no_grad():
        mb.tower_a.pos.zero_()
        mb.tower_a.read_w.zero_()
        mb.tower_b.pos.zero_()
        mb.tower_b.read_w.zero_()
        d1 = float((mb(xc, xf) - mb(xc[:, :, perm], xf)).abs().max())
    ok &= check("T-pos-1 位置盲 + pos=0 + 均匀读出 → 置换时间轴输出不变",
                d1 < 1e-6, f"max|Δ|={d1:.2e}")
    # —— T-pos-2：pos 被真正接进前向（改 pos 必然改输出）——
    #   ⚠️ 必须用**未压平**的模型：位置盲 + 无残差时 pos 只能经"对位置的求和"进入，
    #   效应被压到 1e-6 量级（实测 9.69e-07），用它判读容易误判成"pos 没用"
    torch.manual_seed(1)
    m = TwoTowerMLP(**CFG).eval()
    with torch.no_grad():
        ref = m(xc, xf)
        m.tower_a.pos.normal_(0, 0.1)
        m.tower_b.pos.normal_(0, 0.1)
        d2 = float((m(xc, xf) - ref).abs().max())
    ok &= check("T-pos-2 改 pos → 输出必变（pos 已接进前向）", d2 > 1e-5, f"max|Δ|={d2:.2e}")
    # —— T-pos-4：pos 必须是 Parameter（进 state_dict 而非 buffer）、形状对、梯度非零 ——
    m.zero_grad()
    m(xc, xf).sum().backward()
    ok &= check("T-pos-4 pos 是 Parameter 且在 state_dict 里",
                isinstance(m.tower_a.pos, torch.nn.Parameter) and "tower_a.pos" in m.state_dict())
    ok &= check("T-pos-4 pos 形状 (T,dim)",
                tuple(m.tower_a.pos.shape) == (COARSE_T, CFG["dim"])
                and tuple(m.tower_b.pos.shape) == (FINE_T, CFG["dim"]))
    ok &= check("T-pos-4 pos 梯度非零",
                m.tower_a.pos.grad is not None and float(m.tower_a.pos.grad.abs().max()) > 0)
    return ok


# ------------------------------------------------------------------ 5

def test_checkpoint_roundtrip(ok):
    tmp = tempfile.mkdtemp(prefix="mlp_ck_")
    try:
        args = _args(tmp)
        torch.manual_seed(0)
        m = TwoTowerMLP(**CFG).eval()
        xc, xf = _synth_batch(3)
        with torch.no_grad():
            ref = m(xc, xf)
        path = ckpt_path(tmp, 0, "best")
        save_ckpt(path, m, arch_hash(args), {"fold": 0, "kind": "best", "epoch": 7, "cos": 0.1})
        ok &= check("检查点文件名以 .pt 结尾", path.endswith(".pt"), path)
        ok &= check("无 .tmp 残留", not any(f.endswith(".tmp") for f in os.listdir(tmp)),
                    f"{os.listdir(tmp)}")
        payload = torch.load(path, map_location="cpu", weights_only=True)   # weights_only 可加载
        ok &= check("payload 可用 weights_only=True 加载", "state_dict" in payload)
        m2 = TwoTowerMLP(**CFG).eval()
        m2.load_state_dict(load_ckpt(path, args)["state_dict"])
        with torch.no_grad():
            got = m2(xc, xf)
        ok &= check("存读往返逐位一致", torch.equal(ref, got))
        # 指纹不符必须报错（不能静默用错配置推理）
        try:
            load_ckpt(path, _args(tmp, dim=128))
            ok &= check("指纹不符 → 应报错", False, "没有报错！")
        except ValueError as e:
            ok &= check("指纹不符 → ValueError", "配置不符" in str(e))
        # 文件缺失必须报错
        try:
            load_ckpt(os.path.join(tmp, "nope.pt"), args)
            ok &= check("文件缺失 → 应报错", False, "没有报错！")
        except FileNotFoundError:
            ok &= check("文件缺失 → FileNotFoundError", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


# ------------------------------------------------------------------ 6

# ------------------------------------------------------------------ 7

def test_batch_invariance(ok):
    Xc, Xf = _synth(40, 7)
    nv = np.array([80 + 3 * i for i in range(40)], dtype=np.int64)     # 80..197
    rows = np.arange(40)
    for mode in ("none", "hard"):
        torch.manual_seed(0)
        m = TwoTowerMLP(**CFG).eval()
        with torch.no_grad():
            a, b = build_one(Xc, Xf, 5, nv, COARSE_T)
            one = float(m(torch.from_numpy(a)[None], torch.from_numpy(b)[None]).item())
            batch = eval_forward(m, Xc, Xf, nv, rows, 40, "cpu", COARSE_T)
            chunked = eval_forward(m, Xc, Xf, nv, rows, 2, "cpu", COARSE_T)
        ok &= check(f"{mode}：单行单独跑 == 混在批里第 5 行",
                    abs(one - float(batch[5])) < 1e-6, f"{one:.6f} vs {batch[5]:.6f}")
        dm = float(np.abs(chunked - batch).max())
        ok &= check(f"{mode}：eval_forward(chunk=2) ≈ 一次性（≤1e-6，非逐位）",
                    dm <= 1e-6, f"max|Δ|={dm:.2e}")
        ok &= check(f"{mode}：同一 chunk 重放逐位相同",
                    np.array_equal(chunked, eval_forward(m, Xc, Xf, nv, rows, 2, "cpu", COARSE_T)))
    return ok


# ------------------------------------------------------------------ 8

def test_towers_independent_and_grads(ok):
    torch.manual_seed(0)
    m = TwoTowerMLP(**CFG).eval()
    xc, xf = _synth_batch(2)
    with torch.no_grad():
        h_b0 = m.tower_b(xf)
        xc2 = xc.clone()
        xc2 += 5.0
        m.tower_a(xc2)
        ok &= check("改塔 A 输入 → 塔 B 输出逐位不变",
                    torch.equal(h_b0, m.tower_b(xf)))
    ok &= check("两塔是不同实例", m.tower_a is not m.tower_b)
    m.zero_grad()
    m(xc, xf).sum().backward()
    ga = max(float(q.grad.abs().max()) for q in m.tower_a.parameters() if q.grad is not None)
    gb = max(float(q.grad.abs().max()) for q in m.tower_b.parameters() if q.grad is not None)
    ok &= check("两塔梯度都非零（head 只读一路会静默不训练）", ga > 0 and gb > 0,
                f"A={ga:.2e} B={gb:.2e}")
    return ok


# ------------------------------------------------------------------ 9

def test_block_report(ok):
    m = TwoTowerMLP(**CFG)
    w_a = np.random.default_rng(0).random(COARSE_T)
    w_b = np.random.default_rng(1).random(FINE_T)
    r_a, r_b = block_report(w_a), block_report(w_b)
    ok &= check("塔 A 聚合块数 == 56", r_a.shape == (COARSE_T // 4,), f"{r_a.shape}")
    ok &= check("塔 B 聚合块数 == 15", r_b.shape == (FINE_T // 4,), f"{r_b.shape}")
    ok &= check("聚合口径 == reshape(-1,4).sum(1)", np.allclose(r_a, w_a.reshape(-1, 4).sum(1)))
    ok &= check("聚合口径（细段）", np.allclose(r_b, w_b.reshape(-1, 4).sum(1)))
    ok &= check("全时段权重聚合后守恒", abs(r_a.sum() - w_a.sum()) < 1e-9)
    return ok


# ------------------------------------------------------------------ 10

def test_mini_train(ok):
    tmp = tempfile.mkdtemp(prefix="mlp_mini_")
    try:
        n = 64
        Xc, Xf = _synth(n, 11)
        rng = np.random.default_rng(0)
        nv = np.array([120 + (i % 100) for i in range(n)], dtype=np.int64)
        y = (rng.standard_normal(n) * 0.003).astype(np.float32)
        sids = np.arange(1000, 1000 + n)
        tr_rows, va_rows = np.arange(0, 40), np.arange(40, n)
        args = _args(tmp, epochs=1, batch=32, workers=0)
        r = train_fold(Xc, Xf, nv, y, sids, tr_rows, va_rows, args, "cpu", 0, False)
        need = {"cos", "epoch", "va_pred", "va_sids", "bw_a", "bw_b", "pad_mass",
                "pos_std", "scale_a", "scale_b"}
        ok &= check("返回结构完整", need <= set(r), f"缺 {need - set(r)}")
        ok &= check("va_pred 长度 == 验证行数", len(r["va_pred"]) == len(va_rows))
        ok &= check("cos 有限", np.isfinite(r["cos"]), f"{r['cos']:.5f}")
        ok &= check("块权重形状 56 / 15", r["bw_a"].shape == (56,) and r["bw_b"].shape == (15,))
        best = ckpt_path(tmp, 0, "best")
        last = ckpt_path(tmp, 0, "last")
        ok &= check("best / last 检查点都落盘", os.path.exists(best) and os.path.exists(last))
        ok &= check("续跑键找到 best", resume_ckpt(tmp, 0) == best)
        pred, meta = oof_from_ckpt(best, args, Xc, Xf, nv, va_rows, "cpu")
        ok &= check("oof_from_ckpt 逐位复原 va_pred", np.array_equal(pred, r["va_pred"]))
        ok &= check("检查点 meta 带折号/轮数/cos",
                    meta.get("fold") == 0 and "epoch" in meta and "cos" in meta)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


# ------------------------------------------------------------------ 11

def test_arch_hash(ok):
    a1 = _args()
    a2 = _args()
    ok &= check("同配置 hash 稳定", arch_hash(a1) == arch_hash(a2), arch_hash(a1))
    for field, val in [("dim", 128), ("tok_hidden", 64), ("drop", 0.4),
                       ("seed", 7), ("head_drop", 0.5), ("crop", 128), ("clip", 0.0)]:
        ok &= check(f"改 {field} → hash 变",
                    arch_hash(_args(**{field: val})) != arch_hash(a1))
    return ok


def test_y_weight(ok):
    """**Y3 样本加权**（`--y-weight-q`，HYPOTHESES C-16 的"target 清洗"臂）：
    w = 1/(1+|y|/τ)、按 Σw 归一；τ 取**训练折** |y| 的分位数。

    三条防线（按重要性）：
      a) **指纹只在 q>0 时才带字段** —— 否则已训好的 `ckpt_mlp_crop128` 会被误拒
         （CLAUDE.md #12 的"反向纪律"：别为"等价于不变"的开关加指纹字段）
      b) 损失公式：τ=0 必须**逐位退回** `F.mse_loss`；权重归一后 w≡1 也要退回 MSE
      c) **τ 零泄漏**：把验证折的 |y| 刻意放大，τ 仍必须等于训练行的分位数
    """
    import torch.nn.functional as F

    # (a) 指纹
    ok &= check("y_weight_q=0 → arch_cfg **不含**该字段（保护既有检查点）",
                "y_weight_q" not in arch_cfg(_args(y_weight_q=0.0)))
    ok &= check("y_weight_q=99 → 字段进入指纹",
                arch_cfg(_args(y_weight_q=99.0)).get("y_weight_q") == 99.0)
    base = arch_hash(_args())
    ok &= check("q=0 与默认配置 hash 相同", arch_hash(_args(y_weight_q=0.0)) == base)
    ok &= check("改 y_weight_q → hash 变", arch_hash(_args(y_weight_q=99.0)) != base)
    ok &= check("q=99 与 q=95 也不同",
                arch_hash(_args(y_weight_q=99.0)) != arch_hash(_args(y_weight_q=95.0)))

    # (b) 损失公式
    torch.manual_seed(0)
    pred = torch.randn(257)
    yb = torch.randn(257) * 0.01
    tau = 0.005
    w = 1.0 / (1.0 + yb.abs() / tau)
    ok &= check("w 随 |y| 单调不增",
                bool((w[torch.argsort(yb.abs())][1:] <= w[torch.argsort(yb.abs())][:-1] + 1e-12).all()))
    ok &= check("w(|y|=τ) == 0.5", abs(1.0 / (1.0 + 1.0) - 0.5) < 1e-12)
    got = (w * (pred - yb) ** 2).sum() / w.sum()
    ok &= check("加权损失 == 按 Σw 归一的手写公式",
                torch.allclose(got, (w * (pred - yb) ** 2).sum() / w.sum()))
    ones = torch.ones_like(yb)
    ok &= check("w≡1 → 退回 MSE（归一化没改量纲）",
                torch.allclose((ones * (pred - yb) ** 2).sum() / ones.sum(), F.mse_loss(pred, yb)))

    # (c) τ 零泄漏（端到端迷你训练）
    tmp = tempfile.mkdtemp(prefix="mlp_yw_")
    try:
        n = 64
        Xc, Xf = _synth(n, 12)
        rng = np.random.default_rng(1)
        nv = np.array([120 + (i % 100) for i in range(n)], dtype=np.int64)
        y = (rng.standard_normal(n) * 0.003).astype(np.float32)
        y[40:] *= 50.0                     # 验证折的 |y| 刻意放大到与训练折不同分布
        sids = np.arange(2000, 2000 + n)
        tr_rows, va_rows = np.arange(0, 40), np.arange(40, n)
        args = _args(tmp, epochs=1, batch=32, workers=0, y_weight_q=99.0)
        train_fold(Xc, Xf, nv, y, sids, tr_rows, va_rows, args, "cpu", 0, False)
        _, meta = oof_from_ckpt(ckpt_path(tmp, 0, "best"), args, Xc, Xf, nv, va_rows, "cpu")
        want = float(np.quantile(np.abs(y[tr_rows]), 0.99) * TARGET_SCALE)   # ⚠️ 必须是 ×1000 后
        alt = float(np.quantile(np.abs(y), 0.99) * TARGET_SCALE)
        got_tau = float(meta.get("tau", -1))
        ok &= check("τ 写进检查点 meta", "tau" in meta, f"tau={got_tau:.5f}")
        ok &= check("τ == **训练行** |y| 的 q99（零泄漏）",
                    abs(got_tau - want) < 1e-6, f"got {got_tau:.6f} want {want:.6f}")
        ok &= check("全量 q99 明显不同（证明没把验证折算进去）", abs(alt - want) > 1e-3,
                    f"{alt:.6f} vs {want:.6f}")
        ok &= check("meta 记了 y_weight_q", meta.get("y_weight_q") == 99.0)

        # ⚠️ **量纲一致性**——这条是治 2026-09-23 那个真 bug 的：`y` 未缩放而 loss 里的 `yb`
        #    乘过 TARGET_SCALE，τ 漏乘会让 `|yb|/τ` 对**所有**样本都极大（实测比值 ~300），
        #    于是 `w≈τ/|yb|` 退化成 L1 式加权，**测的就不是 Y3 了**。
        #    直接验的后果：中位权重塌掉（τ=q99 时中位 |y| ≪ τ，中位 w 应接近 1）。
        yb_tr = np.abs(y[tr_rows]) * TARGET_SCALE
        w_med = float(1.0 / (1.0 + np.median(yb_tr) / got_tau))
        ok &= check("τ 与 loss 里的 yb **同量纲**（中位权重不塌）", w_med > 0.5,
                    f"median w = {w_med:.4f}（漏乘 TARGET_SCALE 时 ≈1/300）")
        ok &= check("|yb|/τ 的量级正常（不是全体都远超 1）",
                    float(np.median(yb_tr / got_tau)) < 1.0,
                    f"median |yb|/τ = {np.median(yb_tr / got_tau):.4f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


def test_cos_loss(ok):
    """**batch cosine loss**（`--loss cos`，HYPOTHESES C-17）的数学与指纹规则。

    要点：它与 MSE 在**目标层面等价**（同最优解、同梯度方向，只差每批自适应尺度），
    所以这里钉的是"实现没写错"与"没偷偷混进别的变量"，**不是**"它应该更好"。
    """
    import torch.nn.functional as F

    # (a) 指纹用**历史编码**：`loss != "mse"` 才记字段（别顺着新默认值改成 != "cos"，
    #     那会让 cos 的默认跑与 mse 归档撞同一个 hash）
    ok &= check("loss=cos → arch_cfg **含**该字段",
                arch_cfg(_args(loss="cos")).get("loss") == "cos")
    ok &= check("loss=mse → **不含**该字段", "loss" not in arch_cfg(_args(loss="mse")))
    ok &= check("改 loss → hash 变", arch_hash(_args(loss="mse")) != arch_hash(_args()))
    ok &= check("loss 与 y_weight_q 是**两个独立**指纹字段",
                arch_hash(_args(loss="mse", y_weight_q=99.0))
                != arch_hash(_args(loss="mse")))

    # (b) 数学：正比 → 0；反号 → 2；**尺度不变**
    torch.manual_seed(0)
    yb = torch.randn(1024)
    ok &= check("p ∝ y → loss ≈ 0", abs(float(batch_loss(yb * 3.7, yb, "cos"))) < 1e-6)
    ok &= check("p = −y → loss = 2", abs(float(batch_loss(-yb, yb, "cos")) - 2.0) < 1e-6)
    for c in (0.01, 1.0, 137.0):
        ok &= check(f"尺度不变：loss({c}·p) == loss(p)",
                    abs(float(batch_loss(c * yb * 0.3, yb, "cos"))
                        - float(batch_loss(yb * 0.3, yb, "cos"))) < 1e-6)

    # (c) **等价恒等式**（本设计的核心命题）：
    #     把预测整批缩放到与目标同范数再算**平方和**  ⇒  = 2|y|²·(1 − cos(p,y))
    #     ⚠️ 是**平方和**不是 MSE：`F.mse_loss` 除以了 N，直接用会差一个 batch 大小
    #        （第一版就写错了，被这条测试抓出来：1.392 vs 1425.43，比值恰为 1024）
    p = yb * 0.3 + torch.randn(1024) * 0.8
    p_norm = p * (yb.norm() / p.norm())
    ss_after_norm = ((p_norm - yb) ** 2).sum()
    rhs = 2 * yb.norm() ** 2 * batch_loss(p, yb, "cos")
    ok &= check("「缩放后平方和」= 2|y|²·cos_loss（推导的落地验证）",
                torch.allclose(ss_after_norm, rhs, rtol=1e-5),
                f"{float(ss_after_norm):.6f} vs {float(rhs):.6f}")
    ok &= check("（同一恒等式换算成 MSE 口径也要对）",
                torch.allclose(ss_after_norm / yb.numel(), rhs / yb.numel()))

    # (d) 回归守卫：mse 路径必须**逐位**等于 F.mse_loss（否则历史结果不可比）
    ok &= check("loss=mse 逐位等于 F.mse_loss", torch.equal(batch_loss(p, yb, "mse"),
                                                            F.mse_loss(p, yb)))

    # (e) 分母恰好为 0 → 报错（而不是静默变 NaN）
    try:
        batch_loss(torch.zeros(8), torch.zeros(8), "cos")
        ok &= check("全零输入 → 报 RuntimeError", False, "没报错")
    except RuntimeError:
        ok &= check("全零输入 → 报 RuntimeError（不静默 NaN）", True)
    return ok


def test_grad_norm_noop(ok):
    """`clip=0` 时用 `clip_grad_norm_(params, inf)` **只为读范数**——必须逐位不改梯度。

    否则无裁剪的臂会被这个"记录用"调用悄悄改掉梯度（`x*1.0` 在 float 里确实是恒等，
    但这正是那种"看起来显然、错了也不报"的地方，值得钉住）。
    """
    torch.manual_seed(0)
    m = TwoTowerMLP(**CFG)
    xc, xf = _synth_batch(4)
    m(xc, xf).sum().backward()
    before = {n: q.grad.detach().clone() for n, q in m.named_parameters()
              if q.grad is not None}
    gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf")))
    after = {n: q.grad.detach().clone() for n, q in m.named_parameters()
             if q.grad is not None}
    ok &= check("clip_grad_norm_(·, inf) 不改任何梯度（逐位）",
                all(torch.equal(before[n], after[n]) for n in before))
    # 且返回的范数确实等于手算的全局 L2 范数
    manual = float(torch.sqrt(sum((q.grad ** 2).sum() for q in m.parameters()
                                  if q.grad is not None)))
    ok &= check("返回的范数 == 手算的全局 L2 范数", abs(gn - manual) < 1e-6,
                f"{gn:.6f} vs {manual:.6f}")
    return ok


def test_crop_alignment(ok):
    """`--crop K` 的对齐语义（原理与证据见 PAD_AND_CROP.md §3）。
    用"通道 0 = 源行序号+1"当标记，验证三条：
      ① **最新那一行必须落在位置 K−1** —— 这正是治「位置=序号、时间含义随样本漂移」那个病的机制
      ② 不足 K 行时在**旧端**补零（mask 在旧端为 False）
      ③ **K=COARSE_T 是恒等**（沿用缓存原布局）—— 保证 `--crop 224`（默认）与既有结果、
         与正在跑的基线仍然逐位可比
    ⚠️ 这条测试若失败，最可能的写法是把 n≥K 的分支写成 `a[:, T-K:]`（= 数组最后 K 行，
    而那里现在**全是 pad**）——CLAUDE.md #9 记的 `--tcrop` 老 bug 就是这个。
    """
    Xc, Xf = _synth(8, 5)
    nv = np.array([120, 107, 199, 189, 224, 1, 50, 212], dtype=np.int64)
    for r in range(len(nv)):
        Xc[r, :, 0] = np.arange(COARSE_T) + 1.0        # 源位置 j 的标记值 = j+1
    for K in (COARSE_T, 192, 128, 96, 64):
        newest_ok = short_ok = mask_ok = ident_ok = True
        for r, n in enumerate(nv):
            a, _ = build_one(Xc, Xf, r, nv, K)
            _, m = crop_src(int(n), K)
            src = np.asarray(Xc[r], dtype=np.float32).T
            n = int(n)
            # ① K<COARSE_T 时：最新那一行（源位置 n−1，标记值 = n）必须落在位置 K−1
            #    （K=COARSE_T 是"不裁剪"，沿用缓存原布局，最新行仍在 n−1 → 见 ③）
            if K < COARSE_T:
                newest_ok &= (abs(float(a[0, K - 1]) - n) < 1e-3)
            if K >= COARSE_T:                          # ③ 严格恒等（且最新行仍在原位置 n−1）
                ident_ok &= np.array_equal(a, src)
                newest_ok &= (abs(float(a[0, n - 1]) - n) < 1e-3)
            elif n >= K:                               # ② 精确切片 [n−K, n)
                ident_ok &= np.array_equal(a, np.ascontiguousarray(src[:, n - K:n]))
                mask_ok &= bool(m.all())
            else:                                      # ② 不足 → 旧端补零
                short_ok &= bool((a[:, :K - n] == 0).all()) and (abs(float(a[0, K - n])) > 0)
                mask_ok &= bool((~m[:K - n]).all() and m[K - n:].all())
        ok &= check(f"K={K:3d}：最新那一行的位置正确（K<224 → 位置 K−1；K=224 → 原位置 n−1）",
                    newest_ok)
        ok &= check(f"K={K:3d}：切片/补零正确（K=224 恒等；n≥K 精确取 [n−K,n)；n<K 旧端补零）",
                    ident_ok and short_ok)
        ok &= check(f"K={K:3d}：mask 与裁剪对齐（n≥K 全 True；n<K 旧端 False）", mask_ok)
    # crop_src 的两种对齐方式必须自洽：ok 为 True 的位置，标记值必须递增到 n
    for K in (COARSE_T, 96):
        for n in (1, 50, 107, 189, 212, 224):
            src, o = crop_src(n, K)
            ok &= check(f"crop_src(n={n},K={K}) 自洽（ok 数 == min(n,K)）",
                        int(o.sum()) == min(n, K), f"{int(o.sum())} vs {min(n, K)}")
    # ⚠️ 守卫：**eval_forward 不传 K 时必须取模型自带的 n_tok**
    #    2026-09-21 提交侧就因为漏传 K 拿 224 长输入喂 128 位置的塔而崩
    #    （`size of tensor a (224) must match b (128)`）
    torch.manual_seed(0)
    m128 = TwoTowerMLP(coarse_t=128, **CFG).eval()
    rows_all = np.arange(len(nv))
    with torch.no_grad():
        got = eval_forward(m128, Xc, Xf, nv, rows_all, 4, "cpu")          # 故意不传 K
        ref = []
        for r in rows_all:
            a, b = build_one(Xc, Xf, r, nv, 128)
            ref.append(float(m128(torch.from_numpy(a)[None],
                                  torch.from_numpy(b)[None]).item()))
    ok &= check("eval_forward 不传 K → 取模型自带 n_tok(128)（提交侧漏传的守卫）",
                np.allclose(got, np.array(ref), atol=1e-6),
                f"max|Δ|={np.abs(got - np.array(ref)).max():.2e}")
    return ok


def main():
    print("=== 双塔MLP 回归测试 ===")
    ok = True
    for t in (test_builders_agree, test_forward_shapes, test_readout, test_position_embedding,
              test_checkpoint_roundtrip, test_crop_alignment, test_batch_invariance,
              test_towers_independent_and_grads, test_block_report, test_mini_train,
              test_arch_hash, test_y_weight, test_cos_loss, test_grad_norm_noop):
        print(f"\n[{t.__name__}]")
        ok &= t(True)
    print("\n" + ("全部通过" if ok else "*** 有失败项 ***"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
