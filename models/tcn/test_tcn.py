"""test_tcn.py — 双塔TCN 的本机回归测试（**CPU、合成数组、不需要数据/GPU**）。

跑法：
    /home/zlf/.venvs/lab/bin/python models/tcn/test_tcn.py

不用 pytest（全仓库零 `import pytest`），沿用 `models/mlp/test_mlp.py` 的自写框架：
`check(name, cond, detail)` 打印 PASS/FAIL，`def test_xxx(ok)` 内逐条 `ok &= check(...)`，
`main()` 汇总并给出退出码。

**守住十三件事**（每条都对应一个真实的错法，不是形式）：

 1. `test_causal_and_rf`      —— 因果性 + **实测感受野 == 理论公式**（脉冲法）
 2. `test_last_step_visible`  —— ⚠️ **输入最后一格必须影响读出**（WALKTHROUGH11 §8.1 的回归守卫）
 3. `test_pad_invisible`      —— 位置 > n_valid−1 的内容**逐位不影响**预测（消掉 mask/τ 的数学依据）
 4. `test_last_readout_gather`—— 末步读出 == 直接索引 `h[:, :, k]`（gather 下标不能写错）
 5. `test_receptive_field`    —— 公式自检 + 级数下界断言（`check_rf` 与 `TCNTower` 两处都要挡）
 6. `test_towers_both_used`   —— 两塔都被 head 读到、都有梯度（防 head 只切一路）
 7. `test_builders_agree`     —— 单行构建器 == 整批构建器（不一致会**静默**喂两种分布）
 8. `test_ckpt_roundtrip`     —— 检查点往返、指纹拒绝、冒烟名不碰正式名
 9. `test_arch_cfg_discipline`—— 该进指纹的进、不该进的**不进**（CLAUDE.md #12 两条纪律）
10. `test_no_stride_param`    —— `CausalConv` 签名里**没有 stride**（§8.1 那类错误从架构上不可能）
11. `test_chunked_eval`       —— 同 chunk 重放**逐位相同**；不同 chunk ≤1e-6
12. `test_mini_train`         —— 端到端迷你训练：检查点落盘、OOF 复算逐位复原
13. `test_dataset_returns_nv` —— Dataset 必须把 n_valid 带出来（漏了就退化成读 pad）
14. `test_cos_loss`           —— `--loss cos` 的语义（尺度不变 / 等价式 / 分母为 0 报错 / mse 逐位不变）

⚠️ **所有涉及"不变性"的测试都必须在 `model.eval()` 下做**：train() 时 BatchNorm 用批统计量，
pad 会经由统计量影响所有位置（含读出点）。见 README「已知风险」。
"""
import argparse
import inspect
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(MODELS, "cnn"))
sys.path.insert(0, os.path.join(MODELS, "tabm"))

from tcn_baseline import (COARSE_C, COARSE_T, FINE_C, FINE_T, K,  # noqa: E402
                          OUT_DIM, STEM_K,
                          CausalConv, TCNBlock, TCNTower, TwoTowerTCNDataset,
                          TwoTowerTCN, arch_cfg, arch_hash, batch_loss, batch_two,
                          build_model, build_one, check_rf, ckpt_path, eval_forward,
                          fine_c, fine_spec, load_ckpt, oof_from_ckpt, receptive_field,
                          resume_ckpt, save_ckpt, train_fold)


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    return bool(cond)


# ---------------------------------------------------------------- 测试用合成数据

def _synth(n, seed=0):
    """合成 (n,224,15) / (n,60,16) float16 数组（与真实缓存的 dtype 一致）。"""
    rng = np.random.default_rng(seed)
    Xc = (rng.standard_normal((n, COARSE_T, COARSE_C)) * 0.1).astype(np.float16)
    Xf = (rng.standard_normal((n, FINE_T, FINE_C)) * 0.1).astype(np.float16)
    return Xc, Xf


def _nv(n, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(1, COARSE_T + 1, size=n).astype(np.int64)


def _args(tmpdir="", **over):
    """手造 Namespace——**不走 main()、不解析命令行**（与 test_mlp.py 同法）。

    字段必须覆盖 `arch_cfg` 会读的全部属性，否则 `arch_cfg` 直接 AttributeError。
    """
    d = dict(width=8, levels_coarse=6, levels_fine=4, drop=0.0, head_drop=0.0,
             seed=42, lr=1e-3, wd=3e-4, batch=16, epochs=1, patience=2, loss="cos",
             eval_chunk=8, workers=0, ckpt_dir=tmpdir, ckpt_keep="both", oof_dir=tmpdir)
    d.update(over)
    return argparse.Namespace(**d)


def _stack(c_in, width, levels):
    """手搭「干 + L 个残差块」的裸卷积堆，**绕过 `TCNTower` 的 RF≥T 断言**
    （感受野的边界测试需要 RF < T）。用的是同一批组件，不是复制品。"""
    stem = torch.nn.Sequential(CausalConv(c_in, width, STEM_K), torch.nn.BatchNorm1d(width),
                               torch.nn.GELU(), torch.nn.Dropout(0.0))
    blocks = torch.nn.Sequential(*[TCNBlock(width, K, 2 ** i, 0.0) for i in range(levels)])
    return torch.nn.Sequential(stem, blocks).eval()


# ---------------------------------------------------------------- 1

def test_causal_and_rf(ok):
    """因果性 + 感受野，用**单位脉冲精确解出**（WALKTHROUGH11 §8.1 的同一手法）。

    期望：脉冲落在位置 j → 只有 `[j, j+RF−1]` 受影响，且**精确为 0 之外**。
    """
    torch.manual_seed(0)
    T, L, j = 64, 3, 20
    rf = receptive_field(L)                      # 3 + 4*(2^3−1) = 31
    net = _stack(3, 8, L)
    rng = np.random.default_rng(0)
    x = torch.from_numpy((rng.standard_normal((1, 3, T)) * 0.1).astype(np.float32))
    with torch.no_grad():
        base = net(x)
        x2 = x.clone()
        x2[0, 0, j] += 10.0
        got = net(x2)
    # ⚠️ 必须**对通道维取 max**：`(got-base).abs()[0]` 是 (width,T)，直接 where 拿到的是
    #    展平下标（我第一次就写错了，报出"受影响区间 [0,7]"这种反因果的假象）
    d = (got - base).abs()[0].amax(dim=0).numpy()          # (T,) 逐位置的最大变化
    hit = np.where(d > 0.0)[0]        # 区间外是**结构上精确的 0**，所以 >0 就是完美判别

    ok &= check(f"实测感受野 == 理论值 {rf}", len(hit) and int(hit.max()) - j + 1 == rf,
                f"受影响区间 [{hit.min() if len(hit) else -1}, {hit.max() if len(hit) else -1}]"
                f"（实测 RF {int(hit.max()) - j + 1 if len(hit) else -1}）")
    ok &= check("受影响区间起点 == 脉冲位置（输出 j 看得到输入 j）", len(hit) and int(hit.min()) == j)
    ok &= check("受影响区间连续（无空洞）", len(hit) > 0 and bool((np.diff(hit) == 1).all()))
    ok &= check(f"j+RF={j + rf} 及其后**精确为 0**（不只是小）",
                float(d[j + rf:].max(initial=0.0)) == 0.0)
    ok &= check("脉冲之前**精确为 0**（因果性）", float(d[:j].max(initial=0.0)) == 0.0)

    # 因果性单独再钉一次（换一个位置，避免"只有 j=20 成立"）
    with torch.no_grad():
        x3 = x.clone()
        x3[0, :, 40] += 10.0
        d3 = (net(x3) - base).abs()[0].amax(dim=0).numpy()
    ok &= check("改位置 40 → 输出 0..39 精确为 0", float(d3[:40].max(initial=0.0)) == 0.0)
    ok &= check("改位置 40 → 输出 40 onward 有响应", float(d3[40:].max()) > 0.0)
    return ok


# ---------------------------------------------------------------- 2

def test_last_step_visible(ok):
    """⚠️ `WALKTHROUGH11 §8.1` 的回归守卫：**输入的最后一格必须影响读出**。

    双塔CNN 在这一条上会**失败**：它的 `CausalConv` 左边 pad 写死 `(k−1)·dilation`，
    这在 stride=2 时把输出对齐错了，导致输入末端 3 步对输出完全不存在
    （细段 60 格是真实 60 秒、无 pad → 双塔CNN 看不到最近 3 秒的成交流）。
    本模型的 `CausalConv` 没有 stride 参数（只有 stride=1），这类错误不可能发生。
    """
    torch.manual_seed(1)
    T = 60
    t = TCNTower(3, T, width=8, levels=4, drop=0.0, out_dim=5).eval()      # RF 63 ≥ 60
    x = torch.randn(1, 3, T)
    with torch.no_grad():
        base = t(x)
        x_last = x.clone()
        x_last[0, :, -1] += 10.0
        d_last = float((t(x_last) - base).abs().max())
        x_prev = x.clone()
        x_prev[0, :, -2] += 10.0
        d_prev = float((t(x_prev) - base).abs().max())
        x_first = x.clone()
        x_first[0, :, 0] += 10.0
        d_first = float((t(x_first) - base).abs().max())
    ok &= check("改**最后一格** → 末步读出必变（双塔CNN 在这条上会失败）", d_last > 1e-4,
                f"max|Δ|={d_last:.3e}")
    ok &= check("改倒数第二格 → 也变（确认不是数值巧合）", d_prev > 1e-4, f"max|Δ|={d_prev:.3e}")
    # 第一格到末步要走满 8 层空洞卷积，衰减很大（实测 ~6e-6），所以阈值只能取 `> 0`：
    # 若 RF 盖不住全窗，这条通路**根本不存在** → 差值会是**结构上精确的 0**。
    # 用 >0 判别是充分的，而 1e-4 那种阈值会把这个真实通路误判成"没有"。
    ok &= check("改第一格 → 仍能传到末步（RF 必须覆盖全窗；不存在则精确为 0）",
                d_first > 0.0, f"max|Δ|={d_first:.3e}")
    return ok


# ---------------------------------------------------------------- 3

def test_pad_invisible(ok):
    """末步读出的核心保证：位置 `> n_valid−1` 的内容**逐位不影响**预测。

    这是"池化 / mask / τ 一整类机制从架构上消失"的数学依据：卷积因果 + pad 在时间轴末尾
    ⇒ 取 `n_valid−1` 时末尾 pad 进不来。手法同 WALKTHROUGH8（pad 区灌 999）。
    ⚠️ 只在 `eval()` 下成立（train() 时 BN 批统计量会把 pad 混进来）。
    """
    torch.manual_seed(2)
    m = TwoTowerTCN(width=8, drop=0.0, head_drop=0.0).eval()
    xa = torch.randn(3, COARSE_C, COARSE_T)
    xb = torch.randn(3, FINE_C, FINE_T)
    nv = torch.tensor([50, 189, COARSE_T], dtype=torch.long)
    xa2 = xa.clone()
    for i, n in enumerate(nv.tolist()):
        if n < COARSE_T:
            xa2[i, :, n:] = 999.0
    with torch.no_grad():
        o1 = m(xa, nv, xb)
        o2 = m(xa2, nv, xb)
    ok &= check("pad 区灌 999 → 预测**逐位不变**", torch.equal(o1, o2),
                f"max|Δ|={float((o1-o2).abs().max()):.2e}")
    ok &= check("前 n_valid 格是真的被读到（改它就变）",
                float((m(xa + 5.0, nv, xb) - o1).abs().max()) > 1e-4)
    # n_valid 本身必须起作用：同一份输入、不同 nv → 不同读出位置 → 不同预测
    with torch.no_grad():
        o3 = m(xa, torch.tensor([COARSE_T, COARSE_T, COARSE_T], dtype=torch.long), xb)
    ok &= check("改 n_valid → 预测变（证明读出位置真的由 nv 决定）",
                not torch.equal(o1, o3), f"max|Δ|={float((o1-o3).abs().max()):.2e}")
    return ok


# ---------------------------------------------------------------- 4

def test_last_readout_gather(ok):
    """末步读出 == 直接索引 `h[:, :, k]`（gather 的下标/维度最容易写错且不报错）。"""
    torch.manual_seed(3)
    T = 60
    t = TCNTower(3, T, width=8, levels=4, drop=0.0, out_dim=5).eval()
    x = torch.randn(4, 3, T)
    with torch.no_grad():
        h = t.blocks(t.stem(x))                       # (B,width,T)
        auto = t(x, None)                             # last=None → 位置 T−1
        idx_T = t(x, torch.full((4,), T - 1, dtype=torch.long))
        manual = t.proj(h[:, :, T - 1])
    ok &= check("last=None 等价于 last=T−1", torch.equal(auto, idx_T))
    ok &= check("last=T−1 等价于直接索引 h[:,:,T−1]", torch.equal(auto, manual))
    for k in (0, 1, 17, 42):
        with torch.no_grad():
            got = t(x, torch.full((4,), k, dtype=torch.long))
            want = t.proj(h[:, :, k])
        ok &= check(f"last={k:2d} 读的就是位置 {k:2d}", torch.equal(got, want),
                    f"max|Δ|={float((got-want).abs().max()):.2e}")
    # 逐样本不同位置：第 i 行读它自己的位置。
    # ⚠️ 这里只能要 ≤1e-6 而不能要逐位：手工参照是 `Linear` 吃 (width,) 向量、
    #    实现是吃 (B,width) 矩阵，float32 GEMM 的路径不同（同 test_chunked_eval 的道理）。
    #    判别力靠**配对**:同一批数据换一个位置差远了（下面的对照）。
    pos = torch.tensor([0, 13, 40, T - 1], dtype=torch.long)
    with torch.no_grad():
        got = t(x, pos)
        want = torch.stack([t.proj(h[i, :, int(pos[i])]) for i in range(4)])
        wrong = t(x, torch.tensor([1, 14, 41, T - 2], dtype=torch.long))
    ok &= check("逐样本不同位置（gather 的 batch 维正确）",
                float((got - want).abs().max()) <= 1e-6,
                f"max|Δ|={float((got - want).abs().max()):.2e}")
    # **真阳性对照**（CLAUDE.md #15：判据要先验证它有判别力，否则"通过"什么都说明不了）
    ok &= check("对照：位置整体挪 1 → 输出明显不同（判别力）",
                float((got - wrong).abs().max()) > 1e-3,
                f"max|Δ|={float((got - wrong).abs().max()):.2e}")
    return ok


# ---------------------------------------------------------------- 5

def test_receptive_field(ok):
    """RF 公式自检 + **两处**级数下界断言（`check_rf` 与 `TCNTower` 都要挡）。"""
    for L in range(1, 9):
        want = 1 + (STEM_K - 1) + 2 * (K - 1) * (2 ** L - 1)
        ok &= check(f"receptive_field({L}) == 1+(stem_k−1)+2(k−1)(2^L−1) == {want}",
                    receptive_field(L) == want)
    rf_a, rf_b = check_rf(6, 4)
    ok &= check(f"默认级数覆盖两塔窗口（RF {rf_a} ≥ {COARSE_T}、{rf_b} ≥ {FINE_T}）",
                rf_a >= COARSE_T and rf_b >= FINE_T)
    raised = False
    try:
        check_rf(5, 4)
    except AssertionError:
        raised = True
    ok &= check(f"粗段 5 级（RF {receptive_field(5)} < {COARSE_T}）必须被 check_rf 拦住", raised)
    try:
        check_rf(6, 3)
        ok &= check("细段级数不够也必须拦（漏检一塔等于没检）", False)
    except AssertionError:
        ok &= check(f"细段 3 级（RF {receptive_field(3)} < {FINE_T}）必须被 check_rf 拦住", True)
    raised = False
    try:
        TCNTower(3, COARSE_T, width=8, levels=5)
    except AssertionError:
        raised = True
    ok &= check("TCNTower 自己也断言（不依赖调用方检查）", raised)
    m = TwoTowerTCN(width=8)
    ok &= check(f"默认 TwoTowerTCN：粗段 RF {m.tower_a.rf}、细段 RF {m.tower_b.rf}",
                m.tower_a.rf >= COARSE_T and m.tower_b.rf >= FINE_T)
    return ok


# ---------------------------------------------------------------- 6

def test_towers_both_used(ok):
    """两塔都必须被 head 读到、都必须有梯度。

    错法：head 若只切了 concat 的一半，另一半**静默不训练**、指标照样在动——
    这正是 `test_twotower_baseline.py::test_towers_independent_and_both_trained` 守的东西。
    """
    torch.manual_seed(4)
    m = TwoTowerTCN(width=8, drop=0.0, head_drop=0.0).eval()
    xa = torch.randn(2, COARSE_C, COARSE_T)
    xb = torch.randn(2, FINE_C, FINE_T)
    nv = torch.tensor([100, 200], dtype=torch.long)
    with torch.no_grad():
        o0 = m(xa, nv, xb)
        o_a = m(xa + 3.0, nv, xb)
        o_b = m(xa, nv, xb + 3.0)
        ha0 = m.tower_a(xa, nv - 1)
        _ = m.tower_b(xb + 3.0)
        ha1 = m.tower_a(xa, nv - 1)
    ok &= check("改塔A输入 → 模型输出变（head 确实读了 A 那半）",
                float((o_a - o0).abs().max()) > 1e-5)
    ok &= check("改塔B输入 → 模型输出变（head 确实读了 B 那半）",
                float((o_b - o0).abs().max()) > 1e-5)
    ok &= check("改塔B输入 → 塔A输出**逐位不变**（两塔真独立）", torch.equal(ha0, ha1))

    m.train()
    m(xa, nv, xb).sum().backward()
    ga = sum(float(p.grad.abs().sum()) for p in m.tower_a.parameters() if p.grad is not None)
    gb = sum(float(p.grad.abs().sum()) for p in m.tower_b.parameters() if p.grad is not None)
    ok &= check("两塔梯度都非零", ga > 0 and gb > 0, f"|g_A|={ga:.3e}  |g_B|={gb:.3e}")
    return ok


# ---------------------------------------------------------------- 7

def test_builders_agree(ok):
    """`build_one`（DataLoader 用）必须与 `batch_two`（验证用）数值一致。

    不一致 = 训练/验证喂了不同分布的数据，**不会报任何错**——只有这条测试能发现。
    """
    Xc, Xf = _synth(8, 5)
    rows = [0, 3, 7]
    a0, b0 = build_one(Xc, Xf, 0)
    a, b = batch_two(Xc, Xf, rows, "cpu")
    a1, _ = build_one(Xc, Xf, 3)
    _, b2 = build_one(Xc, Xf, 7)
    ok &= check("单行形状 (C,T)、整批形状 (B,C,T)",
                a0.shape == (COARSE_C, COARSE_T) and b0.shape == (FINE_C, FINE_T)
                and tuple(a.shape) == (3, COARSE_C, COARSE_T)
                and tuple(b.shape) == (3, FINE_C, FINE_T),
                f"{a0.shape} / {tuple(a.shape)}")
    ok &= check("单行 == 整批（粗段，行 0）", np.array_equal(a0, a[0].numpy()))
    ok &= check("单行 == 整批（粗段，行 3）", np.array_equal(a1, a[1].numpy()))
    ok &= check("单行 == 整批（细段，行 7）", np.array_equal(b2, b[2].numpy()))
    ok &= check("f16 → f32 且无 NaN", a.dtype == torch.float32 and bool(torch.isfinite(a).all()))
    return ok


# ---------------------------------------------------------------- 8

def test_ckpt_roundtrip(ok):
    """检查点往返 + **指纹拒绝** + **冒烟名绝不碰正式名**（CLAUDE.md #12 四条纪律）。"""
    tmp = tempfile.mkdtemp(prefix="tcn_ckpt_")
    try:
        args = _args(tmp)
        m = build_model(args, "cpu")
        h = arch_hash(args)
        meta = {"fold": 0, "kind": "best", "epoch": 7, "cos": 0.12345}
        p = ckpt_path(tmp, 0, "best")
        save_ckpt(p, m, h, meta)
        ok &= check("检查点落盘且**没有 .tmp 残留**",
                    os.path.exists(p) and not os.path.exists(p + ".tmp"))
        ok &= check("原子写的临时名是 `path + '.tmp'`（torch.save 不追加扩展名）",
                    not os.path.exists(p + ".tmp.pt"))
        payload = load_ckpt(p, args)
        ok &= check("指纹读回一致", payload["arch_hash"] == h)
        ok &= check("meta 完整", payload["meta"]["cos"] == 0.12345)

        m2 = build_model(args, "cpu")
        m2.load_state_dict(payload["state_dict"])
        for (k1, v1), (k2, v2) in zip(m.state_dict().items(), m2.state_dict().items()):
            if not torch.equal(v1, v2):
                ok &= check(f"往返逐位一致（{k1}）", False)
                break
        else:
            ok &= check("state_dict 往返**逐位一致**", True)

        bad = _args(tmp, width=16)
        raised = False
        try:
            load_ckpt(p, bad)
        except ValueError:
            raised = True
        ok &= check("换 width → 指纹不符必须 ValueError（不静默加载）", raised)

        raised = False
        try:
            load_ckpt(os.path.join(tmp, "nonexistent.best.pt"), args)
        except FileNotFoundError:
            raised = True
        ok &= check("缺文件 → FileNotFoundError", raised)

        # 冒烟写 `_smoke` 名——**正式名必须逐字节不变**（否则 1 epoch 的冒烟会被正式跑当成
        # "已完成"而跳过；mlp / twotower 都是这个坑的受害者名单成员）
        before = open(p, "rb").read()
        ps = ckpt_path(tmp, 0, "best", smoke=True)
        save_ckpt(ps, m, h, {**meta, "smoke": True})
        ok &= check("冒烟名是 `f0_smoke.best.pt`", os.path.basename(ps) == "f0_smoke.best.pt")
        ok &= check("冒烟跑**没有改动正式检查点**（逐字节）", open(p, "rb").read() == before)
        ok &= check("冒烟检查点自己也在", os.path.exists(ps))
        ok &= check("resume_ckpt 优先 best", resume_ckpt(tmp, 0) == p)
        ok &= check("resume_ckpt(smoke=True) 只看冒烟名",
                    resume_ckpt(tmp, 0, smoke=True) == ps)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


# ---------------------------------------------------------------- 9

def test_arch_cfg_discipline(ok):
    """CLAUDE.md #12 的两条**反向**纪律。

    ① 会改变"学出什么权重"的配置**必须**进指纹（架构、输入形状、seed/lr/wd/batch）；
    ② 不改权重的工程件**不要**进指纹——否则"等价于不变"的重跑会被误判成配置不符而拒用。
    另外本轮没有读出开关，所以**不该有 readout 字段**（那就是 ② 的另一种形态）。
    """
    base = _args()
    h0 = arch_hash(base)
    for field, val in (("width", 16), ("levels_coarse", 7), ("levels_fine", 5),
                       ("drop", 0.3), ("head_drop", 0.1), ("seed", 7),
                       ("lr", 2e-3), ("wd", 1e-4), ("batch", 512)):
        ok &= check(f"改 {field} → 指纹必须变", arch_hash(_args(**{field: val})) != h0)
    for field, val in (("epochs", 5), ("patience", 99), ("workers", 2),
                       ("eval_chunk", 4096), ("ckpt_keep", "last"),
                       ("oof_dir", "/tmp/x"), ("ckpt_dir", "/tmp/y")):
        ok &= check(f"改 {field} → 指纹**不该**变（工程件）",
                    arch_hash(_args(**{field: val})) == h0)
    cfg = arch_cfg(base)
    ok &= check("没有 readout 字段（本轮无读出开关，别加永不变化的字段）",
                "readout" not in cfg)
    # 损失指纹用的是**历史编码**：`loss != "mse"` 才记字段（默认值曾是 mse，
    # v22/v24 两条归档都是按这套编的）。⚠️ **不要顺着新默认值改成 `!= "cos"`**——
    # 那样 cos 的默认跑会算出与 v22（MSE 训的）**相同**的 hash，静默加载错协议。
    # 保持历史编码 ⇒ cos 有字段（= v24 的指纹）、mse 无字段（= v22 的指纹），两条都能载。
    ok &= check("--loss cos → **进**指纹（= v24 的编码）",
                arch_cfg(_args(loss="cos")).get("loss") == "cos")
    ok &= check("--loss mse → **不进**指纹（= v22 的编码）",
                "loss" not in arch_cfg(_args(loss="mse")))
    ok &= check("改 loss → 指纹必须变", arch_hash(_args(loss="mse")) != h0)
    ok &= check("指纹含架构关键项",
                {"width", "levels", "k", "stem_k", "out_dim", "head_dim"} <= set(cfg),
                f"{sorted(cfg)}")
    ok &= check("指纹含输入形状", cfg["coarse"] == [COARSE_T, COARSE_C]
                and cfg["fine"] == [FINE_T, FINE_C])
    return ok


# ---------------------------------------------------------------- 10

def test_no_stride_param(ok):
    """`CausalConv` 签名里**没有 stride**——这是刻意的（§8.1 那类错误从架构上不可能发生）。

    对照：`models/cnn/twotower_baseline.py::CausalConv` 有 `stride` 参数，
    而 `left=(k−1)·dilation` 在 stride=2 时是错的 → 输入末端 3 步消失。
    """
    params = list(inspect.signature(CausalConv.__init__).parameters)
    ok &= check(f"CausalConv 参数只有 (cin, cout, k, dilation)，实际 {params}",
                params == ["self", "cin", "cout", "k", "dilation"])
    c = CausalConv(3, 4, k=3, dilation=8)
    ok &= check("空洞左 pad = (k−1)·dilation = 16", c.left == 16, f"left={c.left}")
    ok &= check("底层 conv 的 stride == 1（不可配）", c.conv.stride == (1,))
    # 输出长度必须与输入相同（stride=1、左 pad 补足）——这是末步读出能按位置索引的前提
    with torch.no_grad():
        y = c(torch.randn(2, 3, 37))
    ok &= check("输出长度 == 输入长度（逐位置对齐）", y.shape[2] == 37, f"{tuple(y.shape)}")
    return ok


# ---------------------------------------------------------------- 11

def test_chunked_eval(ok):
    """分块前向：**同 chunk 重放逐位相同**（检查点复算 OOF 依赖这条）；
    不同 chunk 有 float32 GEMM 分块差异 → 断言 ≤1e-6（**不写"逐位相同"**）。"""
    torch.manual_seed(5)
    Xc, Xf = _synth(24, 7)
    nv = _nv(24)
    args = _args(width=8, eval_chunk=8)
    m = build_model(args, "cpu").eval()
    rows = np.arange(24)
    a1 = eval_forward(m, Xc, Xf, nv, rows, 8, "cpu")
    a2 = eval_forward(m, Xc, Xf, nv, rows, 8, "cpu")
    a3 = eval_forward(m, Xc, Xf, nv, rows, 64, "cpu")     # 一次性（不分块）
    a4 = eval_forward(m, Xc, Xf, nv, rows, 5, "cpu")      # 非整除切法
    ok &= check("同 chunk 重放**逐位相同**", np.array_equal(a1, a2))
    ok &= check("不同 chunk 差异 ≤1e-6（浮点分块，非逐位）",
                float(np.abs(a1 - a3).max()) <= 1e-6, f"max|Δ|={float(np.abs(a1-a3).max()):.2e}")
    ok &= check("非整除 chunk 差异 ≤1e-6", float(np.abs(a1 - a4).max()) <= 1e-6,
                f"max|Δ|={float(np.abs(a1-a4).max()):.2e}")
    ok &= check("输出长度与行数一致且有限", a1.shape == (24,) and bool(np.isfinite(a1).all()))
    return ok


# ---------------------------------------------------------------- 12

def test_mini_train(ok):
    """端到端迷你训练（CPU、合成数据）：检查点落盘 + **OOF 用检查点逐位复原**。

    这条把"检查点复算 OOF（免重训）"的整条链路钉住——它依赖
    `train_fold` 选 best 与 `oof_from_ckpt` 走**同一个 `eval_forward`、同一个 chunk**。
    """
    tmp = tempfile.mkdtemp(prefix="tcn_mini_")
    try:
        n = 48
        Xc, Xf = _synth(n, 11)
        nv = _nv(n, 12)
        rng = np.random.default_rng(0)
        y = (rng.standard_normal(n) * 0.003).astype(np.float32)
        sids = np.arange(1000, 1000 + n)
        args = _args(tmp, epochs=1, batch=16, workers=0, eval_chunk=8)
        tr_rows, va_rows = np.arange(0, 32), np.arange(32, n)
        r = train_fold(Xc, Xf, nv, y, sids, tr_rows, va_rows, args, "cpu",
                       tmp, arch_hash(args), 0, False)
        need = {"cos", "epoch", "va_pred", "va_sids", "fold_secs"}
        ok &= check("返回结构完整", need <= set(r), f"缺 {need - set(r)}")
        ok &= check("va_pred 长度 == 验证行数", len(r["va_pred"]) == len(va_rows))
        # 合成数据 + 1 epoch → cos **可以且通常是负的**（没有真实信号可学），
        # 所以这里只断言"是个合法的相关系数"，不断言正负。
        ok &= check("cos 有限且落在 [−1,1]", np.isfinite(r["cos"]) and -1.0 <= r["cos"] <= 1.0,
                    f"cos={r['cos']:.5f}")
        best, last = ckpt_path(tmp, 0, "best"), ckpt_path(tmp, 0, "last")
        ok &= check("best / last 检查点都落盘", os.path.exists(best) and os.path.exists(last))
        ok &= check("续跑键找到 best", resume_ckpt(tmp, 0) == best)
        pred, meta = oof_from_ckpt(best, args, Xc, Xf, nv, va_rows, "cpu")
        ok &= check("oof_from_ckpt **逐位复原** va_pred", np.array_equal(pred, r["va_pred"]),
                    f"max|Δ|={float(np.abs(pred-r['va_pred']).max()):.2e}")
        ok &= check("检查点 meta 记了折号与轮数",
                    meta.get("fold") == 0 and "epoch" in meta, f"{meta}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


# ---------------------------------------------------------------- 13

def test_dataset_returns_nv(ok):
    """Dataset 必须把 `n_valid` 带出来。

    错法：漏了它，训练侧就会退化成"读数组末位"（粗段那一段是 pad）——
    **不报错、指标照样动**，只是模型在学 pad。这条把接口钉住。
    """
    Xc, Xf = _synth(6, 3)
    nv = np.array([10, 60, 120, 150, 189, COARSE_T], dtype=np.int64)
    y = np.arange(6, dtype=np.float32)
    ds = TwoTowerTCNDataset(np.arange(6), y, Xc, Xf, nv)
    got = ds[2]
    ok &= check("Dataset 返回 4 元组 (a, b, nv, y)", len(got) == 4)
    a, b, nvb, yb = got
    ok &= check("粗段 (C,T) / 细段 (C,T)", a.shape == (COARSE_C, COARSE_T)
                and b.shape == (FINE_C, FINE_T))
    ok &= check("nv 是 int64 标量、值正确", nvb.dtype == torch.long and int(nvb) == 120)
    ok &= check("y 已乘 TARGET_SCALE", float(yb) == 2.0 * 1000.0, f"{float(yb)}")
    return ok


# ---------------------------------------------------------------- 14

def test_cos_loss(ok):
    """`--loss cos` 的语义（HYPOTHESES C-17）。**直接测 `batch_loss` 本身**，不测复制品。

    钉住四件事：与 `1−cos(p,y)` 的定义一致；**对预测整体尺度不变**（这正是它不同于 MSE 的地方，
    也是"尺度塌缩"这个预期失败模式的来源）；MSE 分支与原协议**逐位一致**；分母为 0 时报错而非静默 NaN。
    """
    torch.manual_seed(9)
    p = torch.randn(64) * 3.0 + 0.5
    y = torch.randn(64) * 0.002
    want = 1.0 - float((p * y).sum() / (p.norm() * y.norm()))
    got = float(batch_loss(p, y, "cos"))
    ok &= check("cos 损失 == 1 − cos(p, y)", abs(got - want) < 1e-6, f"{got:.6f} vs {want:.6f}")

    for s in (0.01, 1.0, 100.0):
        v = float(batch_loss(p * s, y, "cos"))
        ok &= check(f"预测整体 ×{s:<6g} → cos 损失不变（尺度不变性）", abs(v - got) < 1e-5,
                    f"{v:.6f}")
    # ⚠️ 符号检查必须用**与 p 相关**的目标：上面那组 p/y 是独立随机的，
    #    cos(p,y) ≈ 0 → 取反后 cos 只是换个符号，损失反而可能更小（我第一版就栽在这）。
    y_corr = p * 0.001 + torch.randn(64) * 0.0002      # 与 p 正相关，cos ≈ 0.9
    v_pos = float(batch_loss(p, y_corr, "cos"))
    v_neg = float(batch_loss(-p, y_corr, "cos"))
    ok &= check("预测与目标正相关时损失小", v_pos < 0.2, f"{v_pos:.6f}")
    ok &= check("同一目标下预测取反 → 损失变大", v_neg > v_pos + 1.0,
                f"{v_neg:.6f} vs {v_pos:.6f}")

    # MLP README 记的等价式：把预测缩放到与目标同范数再算 MSE ⇒ Σ(p′−y)² = 2|y|²(1−cos)
    p2 = p * (y.norm() / p.norm())
    lhs = float(((p2 - y) ** 2).sum())
    rhs = float(2 * (y.norm() ** 2) * (1 - (p * y).sum() / (p.norm() * y.norm())))
    ok &= check("等价式 Σ(p′−y)² == 2|y|²(1−cos)（缩放到同范数后）",
                abs(lhs - rhs) / max(abs(rhs), 1e-12) < 1e-5, f"{lhs:.6e} vs {rhs:.6e}")

    ok &= check("mse 分支 == F.mse_loss（逐位）",
                torch.equal(batch_loss(p, y, "mse"), torch.nn.functional.mse_loss(p, y)))
    ok &= check("默认（不传 loss_kind）就是 mse",
                torch.equal(batch_loss(p, y), torch.nn.functional.mse_loss(p, y)))
    raised = False
    try:
        batch_loss(torch.zeros(8), torch.zeros(8), "cos")
    except RuntimeError:
        raised = True
    ok &= check("分母恰为 0 → RuntimeError（不静默变 NaN）", raised)
    return ok


def test_fine_derive_wiring(ok):
    """细段派生通道的接线：`fine_c` / 指纹条件字段 / 变通道数建模型。

    这一组守着三件**不会报错**的事：
    ① 空 `--derive` 必须与今天**逐位相同**（指纹不加字段，否则 v22/v24/v25 会被误拒）；
    ② 两个**通道数相同**的 spec 必须给出**不同**的指纹（否则静默复用另一套通道的检查点）；
    ③ 细段塔的输入通道数必须跟着 `--derive` 走（否则形状断言炸在训练中途）。
    """
    ok &= check("--derive 缺省/none → 空", fine_spec(_args()) == []
                and fine_spec(_args(derive="none")) == [])
    ok &= check("--derive 解析按**顺序**保留（顺序即通道顺序）",
                fine_spec(_args(derive="t_imb_run,t_big_imb")) == ["t_imb_run", "t_big_imb"])
    ok &= check("空 derive → fine_c = 基 16", fine_c(_args()) == FINE_C)
    ok &= check("2 条 derive → fine_c = 18", fine_c(_args(derive="t_imb_run,t_big_imb")) == 18)
    # ① 空 derive 不许加字段 —— 否则归档检查点被误拒
    ok &= check("空 derive → **不加** derive_fine 字段（= 归档的编码）",
                "derive_fine" not in arch_cfg(_args()))
    ok &= check("空 derive → 指纹与不带该属性时**相同**",
                arch_hash(_args()) == arch_hash(_args(derive="none")))
    # ② 同 K 不同 spec 必须可区分（这正是"只看 fine:[60,K] 不够"的那个坑）
    h_a = arch_hash(_args(derive="t_imb_run,t_big_imb"))
    h_b = arch_hash(_args(derive="t_big_imb,t_imb_run"))
    h_c = arch_hash(_args(derive="o_cancel_imb,vol_px_corr"))
    ok &= check("两条 derive → 指纹必变", h_a != arch_hash(_args()))
    ok &= check("**同通道数、不同组合** → 指纹必须不同", h_a != h_c)
    ok &= check("**同组合、不同顺序** → 指纹必须不同（顺序即排布）", h_a != h_b)
    ok &= check("指纹里记的是名字列表",
                arch_cfg(_args(derive="t_imb_run,t_big_imb")).get("derive_fine")
                == ["t_imb_run", "t_big_imb"])
    # ③ 细段塔真的按 fine_c 建
    m = TwoTowerTCN(width=8, levels_coarse=6, levels_fine=4, in_fine=18).eval()
    xf = torch.randn(2, 18, FINE_T)
    xc = torch.randn(2, COARSE_C, COARSE_T)
    with torch.no_grad():
        y = m(xc, torch.full((2,), COARSE_T, dtype=torch.long), xf)
    ok &= check("in_fine=18 能前向", tuple(y.shape) == (2, 1))
    ok &= check("默认 in_fine 仍是基 16",
                TwoTowerTCN(width=8).tower_b.stem[0].conv.in_channels == FINE_C)
    ok &= check("build_model 跟着 --derive 走",
                build_model(_args(derive="t_imb_run,t_big_imb"), "cpu")
                .tower_b.stem[0].conv.in_channels == 18)
    return ok


TESTS = (test_causal_and_rf, test_last_step_visible, test_pad_invisible,
         test_last_readout_gather, test_receptive_field, test_towers_both_used,
         test_builders_agree, test_ckpt_roundtrip, test_arch_cfg_discipline,
         test_no_stride_param, test_chunked_eval, test_mini_train,
         test_dataset_returns_nv, test_cos_loss, test_fine_derive_wiring)


def main():
    print("双塔TCN 回归测试（CPU / 合成数组 / 不需要数据）")
    ok = True
    for t in TESTS:
        print(f"\n[{t.__name__}]")
        ok &= t(True)
    print("\n" + ("全部通过" if ok else "*** 有失败项 ***"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
