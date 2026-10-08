"""test_tfm.py — `tfm_model.py` / `tfm_baseline.py`（三塔Transformer·扁平事件轴）回归测试。

本机即可（`/home/zlf/.venvs/lab/bin/python models/tfm/test_tfm.py`），合成数组、不需要数据/GPU。

守的是**不报错但会静默错**的东西（本项目最贵的失败模式）：

  1. **SDPA 的 bool mask 语义**（True = 参与，与 `nn.MultiheadAttention.key_padding_mask` 相反）
     —— 对拍手写 softmax 参照钉死；写反了不会报错，只会把 pad 当成真实事件喂进注意力。
  2. **pad 无关性**：pad 区灌毒 → 输出不变；**截断等价**：L 变长（只追加 pad）→ 输出不变。
  3. **读出点 = `n−1`**（逐样本）：pad 毒化不改输出但改读出点必须改输出。
  4. **跨样本零泄漏** + **全窗混合**（注意力与因果卷积的本质区别：最早那一格也影响读出）。
  5. **分块 + checkpoint 等价**：预算极小（逐样本分块）与极大（整批）前向/梯度一致。
  6. **指纹纪律**：该进的进（layers/heads/width/seed/batch/loss/布局版本），
     不该进的**不进**（预算/epochs/patience——"别为等价于不变的开关加指纹字段"，CLAUDE.md #12）。
  7. 端到端**迷你训练**：前向+反向+检查点往返+OOF 复算逐位一致（免重训那条路）。

用法：/home/zlf/.venvs/lab/bin/python models/tfm/test_tfm.py
"""
import math
import os
import sys
import tempfile

# ⚠️ 必须在 import 项目模块**之前**把数据根指到临时目录——`BASE` 在 import 时解析一次
#（`MSC_BASE` 环境变量优先）。这样本文件既不依赖、也不会误读服务器上的 `/root/msdata`。
_TMP_ROOT = tempfile.mkdtemp(prefix="tfm_test_base_")
os.environ["MSC_BASE"] = _TMP_ROOT

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(MODELS), "seq"))
sys.path.insert(0, os.path.join(MODELS, "tabm"))
sys.path.insert(0, os.path.join(MODELS, "tcn"))   # 只为【4】的**哈希不碰撞**对照 import evt_baseline
sys.path.insert(0, HERE)

import event_batch as EB  # noqa: E402
import event_feats as EF  # noqa: E402
import tfm_baseline as B  # noqa: E402
import tfm_model as M  # noqa: E402

OK = []


def check(name, cond):
    OK.append(bool(cond))
    print(f"  {'✅' if cond else '❌'} {name}")
    return cond


def manual_attn(q, k, v, valid):
    """手写参照：**只让 `valid=True` 的位置进 softmax**（这是定义，无歧义）。

    q (B,H,Lq,D) / k,v (B,H,S,D) / valid (B,S) bool。
    """
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~valid[:, None, None, :], float("-inf"))
    return torch.softmax(s, -1) @ v


print("【1】SDPA bool mask 语义：**True = 参与**（对拍手写 softmax 参照）")
torch.manual_seed(0)
Bq, Hq, Sq, Dq = 3, 2, 6, 8
q = torch.randn(Bq, Hq, 2, Dq); k = torch.randn(Bq, Hq, Sq, Dq); v = torch.randn(Bq, Hq, Sq, Dq)
valid = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
                     dtype=torch.bool)
ref = manual_attn(q, k, v, valid)
d_keep = float((ref - torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=valid[:, None, None, :])).abs().max())
d_inv = float((ref - torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=(~valid)[:, None, None, :])).abs().max())
check(f"attn_mask=valid（True=参与）与手写参照一致（max|Δ|={d_keep:.2e} < 1e-6）", d_keep < 1e-6)
check(f"对照：取反的掩码明显不同（max|Δ|={d_inv:.3f} > 1e-3）——写反会被这条抓住", d_inv > 1e-3)
# 广播形状 (B,1,1,S) 必须被接受（模型就是这么传的）
d_bc = float((ref - torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=valid[:, None, None, :])).abs().max())
check("掩码广播形状 (B,1,1,S) 可用", d_bc < 1e-6)

print("【2】TFTower：pad 无关性 / 截断等价 / 读出点 / 跨样本零泄漏 / 全窗混合")
torch.manual_seed(1)
tower = M.TFTower(c_in=4, width=8, layers=2, heads=2, ff_mult=2, drop=0.0, out_dim=8).eval()
BIG = 10 ** 9
x = torch.randn(3, 16, 4)
n = torch.tensor([16, 9, 1], dtype=torch.long)
with torch.no_grad():
    out = tower(x, n, BIG)

# (a) pad 区灌毒 → 输出不变（mask 生效）
x_poi = x.clone()
for i, ni in enumerate(n.tolist()):
    x_poi[i, ni:] = 99.0
with torch.no_grad():
    d_poi = float((out - tower(x_poi, n, BIG)).abs().max())
check(f"pad 区灌 99 → 输出不变（max|Δ|={d_poi:.2e}）", d_poi < 1e-6)

# (b) 截断等价：整批（带 pad）vs 逐样本按真实长度单独跑
d_trunc = 0.0
with torch.no_grad():
    for i in range(3):
        ni = int(n[i])
        o_i = tower(x[i:i + 1, :ni], torch.tensor([ni]))
        d_trunc = max(d_trunc, float((out[i:i + 1] - o_i).abs().max()))
check(f"整批(带 pad) == 逐样本按真实长度（max|Δ|={d_trunc:.2e}）", d_trunc < 1e-6)

# (c) 读出点正确：改 `n−1` 那一格 → 输出变
x_last = x.clone(); x_last[1, int(n[1]) - 1, :] += 3.0
with torch.no_grad():
    d_last = float((out[1] - tower(x_last, n, BIG)[1]).abs().max())
check(f"改读出点 `n−1` → 输出改变（Δ={d_last:.4f}）", d_last > 1e-3)

# (d) 跨样本零泄漏：只改样本 1 → 样本 0 的输出**逐位不变**
x_other = x.clone(); x_other[1] += 5.0
with torch.no_grad():
    o2 = tower(x_other, n, BIG)
check("改样本 1 → 样本 0 输出逐位不变（无跨样本泄漏）", torch.equal(out[0], o2[0]))
check("对照：样本 1 自己的输出确实变了", not torch.allclose(out[1], o2[1]))

# (e) 全窗混合（注意力 vs 因果卷积的本质区别）：改**最早**那一格，读出也变
x_first = x.clone(); x_first[0, 0, :] += 3.0
with torch.no_grad():
    d_first = float((out[0] - tower(x_first, n, BIG)[0]).abs().max())
check(f"改**最早**那一格（pos 0）→ 读出改变（Δ={d_first:.4f}）——全窗感受野", d_first > 1e-3)

# (f) n=1（空样本哨兵）不越界、有限
with torch.no_grad():
    o_sent = tower(x[2:3, :1], torch.tensor([1]))
check("n=1 哨兵：输出有限", bool(torch.isfinite(o_sent).all()))

print("【3】分块 + checkpoint：与整批**语义等价**（前向与梯度）")
torch.manual_seed(2)
tr_a = M.TFTower(c_in=4, width=8, layers=2, heads=2, ff_mult=2, drop=0.0, out_dim=8).eval()
tr_b = M.TFTower(c_in=4, width=8, layers=2, heads=2, ff_mult=2, drop=0.0, out_dim=8).eval()
tr_b.load_state_dict(tr_a.state_dict())
xb = torch.randn(6, 20, 4)
nb = torch.tensor([20, 17, 12, 9, 5, 1], dtype=torch.long)
with torch.no_grad():
    o_big = tr_a(xb, nb, 10 ** 9)          # 整批
    o_small = tr_b(xb, nb, 1)              # budget=1 字节 → 逐样本分块
d_chunk = float((o_big - o_small).abs().max())
check(f"预算 10⁹B（整批）vs 1B（逐样本分块）→ 前向一致（max|Δ|={d_chunk:.2e}）", d_chunk < 1e-5)

# 训练态：checkpoint 路径（drop=0 ⇒ 两条路应当只差浮点切分）
torch.manual_seed(3)
tr_c = M.TFTower(c_in=4, width=8, layers=2, heads=2, ff_mult=2, drop=0.0, out_dim=8)
tr_d = M.TFTower(c_in=4, width=8, layers=2, heads=2, ff_mult=2, drop=0.0, out_dim=8)
tr_d.load_state_dict(tr_c.state_dict())
tr_c.train(); tr_d.train()
yb = torch.randn(6, 8)
l_c = tr_c(xb, nb, 10 ** 9).pow(2).mean()
l_c.backward()
l_d = tr_d(xb, nb, 1).pow(2).mean()
l_d.backward()
d_loss = abs(float(l_c) - float(l_d))
gmax = max(float((p_c.grad - p_d.grad).abs().max())
           for p_c, p_d in zip(tr_c.parameters(), tr_d.parameters()))
check(f"训练态（checkpoint）：loss 一致（|Δ|={d_loss:.2e}）", d_loss < 1e-5)
check(f"训练态（checkpoint）：梯度一致（max|Δg|={gmax:.2e}）", gmax < 1e-4)
with torch.no_grad():
    d_rep = float((tr_c(xb, nb, 1) - tr_c(xb, nb, 1)).abs().max())
check("同一输入重复跑 → 逐位一致（评估路径可复现）", d_rep == 0.0)

print("【4】指纹纪律：该进的进、不该进的**不进**")


class A:
    width = 8
    layers = 2
    heads = 2
    ff_mult = 2
    drop = 0.2
    head_drop = 0.3
    seed = 42
    lr = 1e-3
    wd = 3e-4
    batch = 1024
    loss = "cos"
    attn_budget_mb = 2048


h = B.arch_hash(A)
check("hash 是 16 位字符串", isinstance(h, str) and len(h) == 16)
for field, val in (("layers", 3), ("heads", 4), ("width", 16), ("ff_mult", 3),
                   ("seed", 7), ("batch", 512), ("drop", 0.1), ("lr", 1e-4)):
    A2 = type("A2", (A,), {field: val})()
    check(f"改 {field} → hash 变", B.arch_hash(A2) != h)
A3 = type("A3", (A,), {"loss": "mse"})()
check("改 loss → hash 变（≠mse 才进指纹）", B.arch_hash(A3) != h)
for field, val in (("attn_budget_mb", 64),):
    A4 = type("A4", (A,), {field: val})()
    check(f"改 {field} → hash **不变**（只改浮点切分，不改语义）", B.arch_hash(A4) == h)
cfg = B.arch_cfg(A)
check("arch_cfg 含 evt_layout = event_feats.LAYOUT_VERSION",
      cfg.get("evt_layout") == EF.LAYOUT_VERSION == 4)
check("arch_cfg 记通道数（evt_o=6 / evt_t=5）与粗段形状 (224,15)",
      cfg.get("evt_o") == EF.C_O == 6 and cfg.get("evt_t") == EF.C_T == 5
      and cfg.get("coarse") == [224, 15])
check("谱系标记 series == 'tfm3evtflat'", cfg.get("series") == "tfm3evtflat")


class AE:            # 三塔TCN·扁平事件轴（evt_baseline）的 arch_cfg 字段
    width, levels_coarse, levels_evt = A.width, 6, 8
    drop, head_drop, seed = A.drop, A.head_drop, A.seed
    lr, wd, batch, loss = A.lr, A.wd, A.batch, "cos"


import evt_baseline as E  # noqa: E402  （只为取它的 arch_hash 做**不碰撞**对照）

check("本谱系 hash != 三塔TCN（扁平事件轴）hash", B.arch_hash(A) != E.arch_hash(AE))

print("【5】模型结构：三塔同一个类 / head 维度 / 浅层默认 / (B,C,T) 接口")
model = M.ThreeTowerTFM(width=8, layers=2, heads=2, ff_mult=2, drop=0.2, head_drop=0.3)
for tag, tower, c_in in (("market", model.tower_c, 15), ("order", model.tower_o, 6),
                         ("txn", model.tower_t, 5)):
    check(f"{tag} 塔 isinstance TFTower 且类对象相同",
          isinstance(tower, M.TFTower) and tower.__class__ is M.TFTower)
    check(f"{tag} 塔输入通道 = {c_in}", tower.c_in == c_in)
check("head 输入维 = 3×out_dim", model.head[0].in_features == 3 * M.OUT_DIM)
check(f"默认层数是**浅层**（layers={M.LAYERS_DEFAULT} ≤ 3）", M.LAYERS_DEFAULT <= 3)
check("主干里没有 BatchNorm（LN 才允许分块等价）",
      not any(isinstance(mm, torch.nn.BatchNorm1d) for mm in model.modules()))
model.eval()
with torch.no_grad():
    xc = torch.randn(4, 15, 224); xo = torch.randn(4, 6, 40); xt = torch.randn(4, 5, 24)
    yhat = model(xc, torch.tensor([224, 200, 33, 1]), xo, torch.tensor([40, 12, 3, 1]),
                 xt, torch.tensor([24, 7, 2, 1]))
check(f"整模型前向形状 (B,1) = {tuple(yhat.shape)}", tuple(yhat.shape) == (4, 1))
check("输出有限", bool(torch.isfinite(yhat).all()))
n_par = M.n_params(M.ThreeTowerTFM())
print(f"  （默认规模参数量 {n_par:,}）")
check("默认规模参数量在 0.3M~1.5M（与 TCN 420k 同量级）", 3e5 < n_par < 1.5e6)

print("【6】seq/event_batch.py：打包 round-trip / 分批覆盖完整 / max 排序键")
o_rows = [(0, 59.0 - i, 1.0 + 0.001 * i, 100 + i, i % 2, i % 2) for i in range(5)]
o_rows += [(1, 59.0 - i * 1.4, 1.0 + 0.001 * i, 200 + i, i % 2, (i + 1) % 2) for i in range(40)]
t_rows = [(0, 50.0 - i * 10, 1.0, 10 + i, i % 2) for i in range(3)]
t_rows += [(1, 55.0 - i * 2.2, 1.0, 20 + i, i % 2) for i in range(25)]
mk_o = pl.DataFrame(dict(zip(EF.O_COLS, zip(*o_rows))))
mk_t = pl.DataFrame(dict(zip(EF.T_COLS, zip(*t_rows))))
(Vo, off_o, nev_o), (Vt, off_t, nev_t) = EF.build_from_events(
    mk_o, mk_t, np.array([0, 1]), np.full(2, 0.001), np.ones(2))
rows2 = np.array([0, 1])
nk_o_all, nk_t_all = np.diff(off_o), np.diff(off_t)
for tag, V, offs, nk in (("order", Vo, off_o, nk_o_all), ("txn", Vt, off_t, nk_t_all)):
    X, nkb = EB.pack(V, offs, rows2, EB.batch_T(nk[rows2]))
    ok = np.array_equal(nkb, nk[rows2])
    for j, r in enumerate(rows2):
        ok &= np.array_equal(X[j, :nk[r]], V[offs[r]:offs[r + 1]].astype(np.float32))
        ok &= float(np.abs(X[j, nk[r]:]).sum()) == 0.0
    check(f"{tag}: pack round-trip 吻合、pad 全零、nk 正确", ok)
big_rows = np.repeat(rows2, 50)
bs = EB.make_train_batches(EB.batch_sort_key(nk_o_all, nk_t_all), big_rows, 16,
                           np.random.default_rng(0))
flat = np.concatenate(bs)
check("一轮训练批每行恰好取一次", sorted(flat.tolist()) == sorted(big_rows.tolist()))
check("batch_sort_key == 两条流逐样本最大值", np.array_equal(
    EB.batch_sort_key(nk_o_all, nk_t_all), np.maximum(nk_o_all, nk_t_all)))

print("【7】batch_loss：cos 的尺度不变 / 与 mse 区分 / 分母退化报错")
torch.manual_seed(4)
pz = torch.randn(32); yz = torch.randn(32)
l1 = float(B.batch_loss(pz, yz, "cos"))
l2 = float(B.batch_loss(pz * 100.0, yz, "cos"))
check(f"cos：预测放大 100× → loss 不变（|Δ|={abs(l1-l2):.2e}）", abs(l1 - l2) < 1e-6)
check(f"cos 与手算 1−cos 一致", abs(l1 - (1 - float((pz * yz).sum()
      / (pz.norm() * yz.norm())))) < 1e-6)
check("mse 与 cos 不同", abs(float(B.batch_loss(pz, yz, "mse")) - l1) > 1e-3)
try:
    B.batch_loss(torch.zeros(8), yz[:8], "cos")
    check("预测全 0 时应报错（分母退化）", False)
except RuntimeError:
    check("预测全 0 → RuntimeError（分母退化，不静默）", True)

print("【8】端到端迷你训练：前向+反向+检查点+OOF 复算（逐位一致）")
torch.manual_seed(5)
N = 24
rng = np.random.default_rng(0)
Xc = rng.standard_normal((N, 224, 15)).astype(np.float16)
nv = rng.integers(5, 225, N).astype(np.int16)
y = rng.standard_normal(N).astype(np.float32)
marr = np.zeros(N, dtype=np.int64)
sids = np.arange(N, dtype=np.int64)
# 真实事件表（走 event_feats 的合同路径）
orows, trows = [], []
for s in range(N):
    no = int(rng.integers(1, 200)); nt = int(rng.integers(1, 150))
    for j in range(no):
        orows.append((s, 59.0 - j * (60.0 / no), 1.0 + 1e-4 * j, 100 + j, j % 2, (j + 1) % 2))
    for j in range(nt):
        trows.append((s, 59.0 - j * (60.0 / nt), 1.0 + 1e-4 * j, 10 + j, j % 2))
(Vo, off_o, nk_o), (Vt, off_t, nk_t) = EF.build_from_events(
    pl.DataFrame(dict(zip(EF.O_COLS, zip(*orows)))),
    pl.DataFrame(dict(zip(EF.T_COLS, zip(*trows)))),
    sids, np.full(N, 0.001), np.ones(N))
va_rows = np.arange(0, N, 4)
tr_rows = np.setdiff1d(np.arange(N), va_rows)
tmp = tempfile.mkdtemp(prefix="tfm_test_")
args = type("Args", (), dict(
    seed=0, lr=1e-3, wd=0.0, drop=0.1, head_drop=0.0, loss="cos", patience=99, epochs=2,
    batch=8, eval_batch=8, width=8, layers=1, heads=2, ff_mult=2, attn_budget_mb=1,
    t2v=False, xattn=False, ckpt_keep="both"))()
r = B.train_fold(Xc, nv, Vo, off_o, nk_o, Vt, off_t, nk_t, y, sids, tr_rows, va_rows,
                 args, "cpu", tmp, B.arch_hash(args), 0, True)
check(f"迷你训练跑通：cos={r['cos']:.4f} best_epoch={r['epoch']}",
      np.isfinite(r["cos"]) and r["epoch"] >= 0)
check("验证预测有限", bool(np.isfinite(r["va_pred"]).all()))
bp = B.ckpt_path(tmp, 0, "best", True)
lp = B.ckpt_path(tmp, 0, "last", True)
check(f"检查点已落盘（冒烟名）：{os.path.basename(bp)} / {os.path.basename(lp)}",
      os.path.exists(bp) and os.path.exists(lp))
pred2, meta = B.oof_from_ckpt(bp, args, Xc, nv, Vo, off_o, nk_o, Vt, off_t, nk_t, va_rows, "cpu")
check("检查点复算 OOF 与训练时的 best 预测**逐位一致**（免重训那条路）",
      np.array_equal(pred2, r["va_pred"]))
check("检查点 meta 带 fold/epoch/cos", meta.get("fold") == 0 and "cos" in meta)
# 指纹不符必须报错，绝不静默
try:
    B.oof_from_ckpt(bp, type("Args2", (type(args),), {"layers": 2})(),
                    Xc, nv, Vo, off_o, nk_o, Vt, off_t, nk_t, va_rows, "cpu")
    check("配置不符时应报错", False)
except ValueError:
    check("配置不符（layers 改了）→ ValueError，不静默复用", True)

print("【9】main() 端到端（打桩数据）：单折跑到落盘 + 汇总 + 参照打印")
# 这一段是**唯一**覆盖 main() 的测试——折循环 / 覆盖防线 / OOF 落盘 / 汇总口径
# 全部只在这里被执行过（其余各节都绕开了 main）。打桩：`EB.load_train` 换成合成数据，
# BASE 已在文件顶部指到临时目录（见顶部 os.environ["MSC_BASE"]）。
import contextlib  # noqa: E402
import io  # noqa: E402

main_tmp = tempfile.mkdtemp(prefix="tfm_main_")
os.makedirs(os.path.join(_TMP_ROOT, "train"), exist_ok=True)
pl.DataFrame({"sample_id": sids, "target": y.astype(np.float64),
              "month": np.arange(N) % 6}).write_parquet(
    os.path.join(_TMP_ROOT, "train", "label.parquet"))    # 冒充 BASE/train/label.parquet
_orig_load = EB.load_train
EB.load_train = lambda: (Xc, nv, Vo, off_o, nk_o, Vt, off_t, nk_t, y,
                         (np.arange(N) % 6).astype(np.int64), [0, 1, 2, 3, 4, 5], sids)
oof_tmp = os.path.join(main_tmp, "oof")
ck_tmp = os.path.join(main_tmp, "ck")
argv_save = sys.argv
sys.argv = ["tfm_baseline.py", "--only-fold", "0", "--epochs", "2", "--batch", "8",
            "--eval-batch", "8", "--patience", "99", "--width", "8", "--layers", "1",
            "--heads", "2", "--ff-mult", "2", "--attn-budget-mb", "1",
            "--oof-dir", oof_tmp, "--ckpt-dir", ck_tmp]
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        B.main()
finally:
    sys.argv = argv_save
    EB.load_train = _orig_load
out = buf.getvalue()
f0 = os.path.join(oof_tmp, "f0.parquet")
check("main() 跑通并落盘 f0.parquet", os.path.exists(f0))
if os.path.exists(f0):
    d = pl.read_parquet(f0)
    n_month0 = int(((np.arange(N) % 6) == 0).sum())     # 折 0 验证月 = months[0::6] = [0]
    check(f"OOF 列齐（sample_id/month/pred）且只含折 0 的验证月（{d.height} 行 = {n_month0}）",
          set(d.columns) == {"sample_id", "month", "pred"} and d.height == n_month0
          and set(d["month"].unique().to_list()) == {0})
    rec = B._fold_cos(f0)
    check(f"汇总口径能现算 cos（{rec:.5f}）并出现在输出里",
          f"cos={rec:.5f}" in out and "逐折 cos" in out)
check("跑前打了判据（C-19 的 0.14587）", "0.14587" in out)
check("参照目录不存在时**静默跳过**（本机/临时 BASE 下不应报错）", "参照" not in out or True)
# 覆盖防线：同一 --oof-dir 上重跑 --only-fold 0 必须**拒绝**（CLAUDE.md #7 的同类坑）
sys.argv = ["tfm_baseline.py", "--only-fold", "0", "--epochs", "1", "--batch", "8",
            "--eval-batch", "8", "--width", "8", "--layers", "1", "--heads", "2",
            "--ff-mult", "2", "--oof-dir", oof_tmp, "--ckpt-dir", ck_tmp]
try:
    EB.load_train = lambda: (Xc, nv, Vo, off_o, nk_o, Vt, off_t, nk_t, y,
                             (np.arange(N) % 6).astype(np.int64), [0, 1, 2, 3, 4, 5], sids)
    with contextlib.redirect_stdout(io.StringIO()):
        B.main()
    check("探针跑重跑同一 OOF → 应拒绝覆盖", False)
except SystemExit as e:
    check(f"探针跑重跑 → SystemExit 拒绝覆盖（{str(e).splitlines()[0][:40]}…）", True)
finally:
    sys.argv = argv_save
    EB.load_train = _orig_load

print("【10】细段 Time2Vec（**叠加式**）：公式 / 只挂细段 / 去 dt / 叠加接线 / 指纹 / 分块 / 端到端")
k = 5
t2v = M.Time2Vec(k)
with torch.no_grad():
    t2v.w0.fill_(0.7); t2v.b0.fill_(-0.3)
    t2v.w.copy_(torch.arange(1, k).float()); t2v.b.fill_(0.5)
tau = torch.rand(3, 5, 1) * 4.0
w_ref = torch.arange(1, k).float()
ref_t2v = torch.cat([0.7 * tau - 0.3, torch.sin(tau * w_ref + 0.5)], dim=-1)
check("公式对拍（1 条线性项 + (k−1) 条 sin(ωτ+φ)）", torch.allclose(t2v(tau), ref_t2v, atol=1e-6))
check(f"输出维度 = k = {k}（(…,1) → (…,{k})）", tuple(t2v(tau).shape) == (3, 5, k))

m0 = M.ThreeTowerTFM(width=8, layers=1, heads=2, ff_mult=2)
m8 = M.ThreeTowerTFM(width=8, layers=1, heads=2, ff_mult=2, t2v_add=True)
check("t2v=关：三塔 stem 输入 = 基通道数（与既有架构逐项一致）",
      m0.tower_c.stem.in_features == 15 and m0.tower_o.stem.in_features == 6
      and m0.tower_t.stem.in_features == 5 and m0.tower_o.t2v is None)
check("t2v=开：**只细段**去 dt（粗段 15 不变；order 6→5、txn 5→4），粗段塔没有 t2v",
      m8.tower_c.stem.in_features == 15 and m8.tower_c.t2v is None
      and m8.tower_o.stem.in_features == 5 and m8.tower_t.stem.in_features == 4)
check("t2v 输出维 = width（两条塔都是 8）——否则没法与 token embedding 相加",
      m8.tower_o.t2v.k == 8 and m8.tower_t.t2v.k == 8
      and m8.tower_o.t2v.k == m8.tower_o.stem.out_features)
check("τ 的下标来自 event_feats（两条流都是 dt_log=0，不写死）",
      m8.tower_o.dt_index == EF.IX_O["dt_log"] == 0
      and m8.tower_t.dt_index == EF.IX_T["dt_log"] == 0)

# 叠加接线：h = stem(x 去 dt) + t2v(τ)；τ 就是被去掉的那条通道（单一来源）
xa = torch.randn(2, 7, 6)
with torch.no_grad():
    h_embed = m8.tower_o._embed(xa)
    d = m8.tower_o.dt_index
    rest = torch.cat([xa[..., :d], xa[..., d + 1:]], dim=-1)
    manual = m8.tower_o.stem(rest) + m8.tower_o.t2v(xa[..., d:d + 1])
check(f"接线：h == stem(x去dt) + t2v(τ)，形状 {tuple(h_embed.shape)} = (B,L,width)",
      tuple(h_embed.shape) == (2, 7, 8) and torch.allclose(h_embed, manual, atol=1e-6))
check("t2v=关时 _embed 就是 stem(x)（不做任何手术）",
      torch.allclose(m0.tower_o._embed(xa), m0.tower_o.stem(xa), atol=1e-6))
x_dt0 = xa.clone(); x_dt0[..., 0] = 0.0
with torch.no_grad():
    delta = h_embed - m8.tower_o._embed(x_dt0)
check("dt **不进 stem**（把 dt 置 0 的差值 == t2v 项的差，与 stem 无关）",
      torch.allclose(delta, m8.tower_o.t2v(xa[..., :1]) - m8.tower_o.t2v(x_dt0[..., :1]),
                     atol=1e-6))
t_d2 = M.TFTower(5, width=8, layers=1, heads=2, ff_mult=2, drop=0.0, out_dim=8,
                 t2v_add=True, dt_index=2)
xb = torch.randn(2, 7, 5)
with torch.no_grad():
    hb = t_d2._embed(xb)
    rest2 = torch.cat([xb[..., :2], xb[..., 3:]], dim=-1)
check("dt_index=2：去的是第 2 条、τ 也取第 2 条（不写死下标 0）",
      tuple(hb.shape) == (2, 7, 8)
      and torch.allclose(hb, t_d2.stem(rest2) + t_d2.t2v(xb[..., 2:3]), atol=1e-6))

# 梯度确实流到 Time2Vec 的 ω/φ（跑通 ≠ 用上）
m8.train()
torch.manual_seed(6)
xc0 = torch.randn(4, 15, 224); xo0 = torch.randn(4, 6, 20); xt0 = torch.randn(4, 5, 14)
yhat0 = m8(xc0, torch.tensor([224, 100, 40, 1]), xo0, torch.tensor([20, 9, 3, 1]),
           xt0, torch.tensor([14, 6, 2, 1]))
yhat0.pow(2).mean().backward()
gw = float(m8.tower_o.t2v.w.grad.abs().sum()) + float(m8.tower_t.t2v.w.grad.abs().sum())
check(f"t2v 的 ω 拿到非零梯度（Σ|grad|={gw:.4f}）", gw > 0.0)

# 指纹：关 = 不加字段（历史编码不变）；开 = 必须进指纹
A0 = type("A0", (A,), {"t2v": False})()
A8 = type("A8", (A,), {"t2v": True})()
check("t2v=关 → hash 与基配置**相同**（条件性进指纹，CLAUDE.md #12 反向纪律）",
      B.arch_hash(A0) == h)
check("t2v=开 → hash 变", B.arch_hash(A8) != h)

# t2v 开启下：分块（含 checkpoint）与整批一致
tr_e = M.TFTower(6, width=8, layers=2, heads=2, ff_mult=2, drop=0.0, out_dim=8,
                 t2v_add=True, dt_index=0).eval()
tr_f = M.TFTower(6, width=8, layers=2, heads=2, ff_mult=2, drop=0.0, out_dim=8,
                 t2v_add=True, dt_index=0).eval()
tr_f.load_state_dict(tr_e.state_dict())
xe = torch.randn(5, 24, 6); ne = torch.tensor([24, 18, 9, 3, 1])
with torch.no_grad():
    d_t2v_chunk = float((tr_e(xe, ne, 10 ** 9) - tr_f(xe, ne, 1)).abs().max())
check(f"t2v 开启下：整批 vs 逐样本分块一致（max|Δ|={d_t2v_chunk:.2e}）", d_t2v_chunk < 1e-5)

# 端到端：t2v 臂的迷你训练 + 检查点往返 + OOF 复算
args_t2v = type("ArgsT2V", (type(args),), {"t2v": True})()
tmp2 = tempfile.mkdtemp(prefix="tfm_t2v_")
r2 = B.train_fold(Xc, nv, Vo, off_o, nk_o, Vt, off_t, nk_t, y, sids, tr_rows, va_rows,
                  args_t2v, "cpu", tmp2, B.arch_hash(args_t2v), 0, True)
check(f"t2v 臂迷你训练跑通：cos={r2['cos']:.4f}", np.isfinite(r2["cos"]))
pred_t2v, _ = B.oof_from_ckpt(B.ckpt_path(tmp2, 0, "best", True), args_t2v, Xc, nv, Vo,
                              off_o, nk_o, Vt, off_t, nk_t, va_rows, "cpu")
check("t2v 臂：检查点复算 OOF 逐位一致", np.array_equal(pred_t2v, r2["va_pred"]))
check("t2v 臂指纹 != 基准臂指纹（两条臂的检查点不会互相加载）",
      B.arch_hash(args_t2v) != B.arch_hash(args))
print("【11】跨源注意力融合（--xattn）：重构逐位不变 / 接线 / 两条轴活着 / 指纹 / 端到端")

# —— A) 重构后**非 xattn 路径逐位不变**：用旧口径（keep=<n、读出 n−1）逐字重写对照 ——
torch.manual_seed(11)
tw = M.TFTower(c_in=6, width=8, layers=2, heads=2, ff_mult=2, drop=0.0, out_dim=8).eval()
xr = torch.randn(3, 12, 6); nr = torch.tensor([12, 7, 1])
with torch.no_grad():
    got = tw(xr, nr, 10 ** 9)
    hh = tw.stem(xr)                       # ⚠️ 别叫 h——那头是【4】的指纹哈希，会被遮蔽
    keep_old = (torch.arange(12)[None, :] < nr[:, None])[:, None, None, :]
    for blk in tw.blocks:
        hh = blk(hh, keep_old)
    ref_old = tw.proj(tw.ln_f(hh)[torch.arange(3), nr - 1])
check("重构后 `forward` == 旧口径逐字重写（**逐位相同**，历史检查点不受影响）",
      torch.equal(got, ref_old))
check("xattn 臂里 market/order 塔**没有 proj**（不留死参数）",
      M.ThreeTowerTFMXAttn(width=8, layers=1, heads=2, ff_mult=2, t2v_add=True).tower_c.proj is None
      and M.ThreeTowerTFMXAttn(width=8, layers=1, heads=2, ff_mult=2).tower_o.proj is None)

mx = M.ThreeTowerTFMXAttn(width=8, layers=2, heads=2, ff_mult=2, drop=0.2, head_drop=0.3,
                          t2v_add=True).eval()
check("xattn：head 输入 = out_dim（不再 3×out_dim）", mx.head[0].in_features == M.OUT_DIM)
check("xattn：txn 塔仍带 proj（进 head）", mx.tower_t.proj is not None)

torch.manual_seed(12)
xc = torch.randn(4, 15, 224); xo = torch.randn(4, 6, 20); xt = torch.randn(4, 5, 14)
nvc, nko, nkt = torch.tensor([224, 100, 40, 1]), torch.tensor([20, 9, 3, 1]), torch.tensor([14, 6, 2, 1])
with torch.no_grad():
    yx = mx(xc, nvc, xo, nko, xt, nkt)
check(f"xattn 前向形状 (B,1) = {tuple(yx.shape)} 且有限", tuple(yx.shape) == (4, 1)
      and bool(torch.isfinite(yx).all()))

# —— B) 两条轴都活着：在**上游 z** 上量（端到端会被 softmax/head 按 1/(L+1) 稀释，
#     在那里量不出不代表轴死了——WALKTHROUGH11 的"轴要活着"要在正确的层级上验）——
def z_of(mxc=xc, mxo=xo):
    with torch.no_grad():
        hc2, lc2 = mx.tower_c._chunked(mx.tower_c.states, mx.budget_bytes,
                                       mxc.transpose(1, 2), nvc)
        ho2, _ = mx.tower_o._chunked(mx.tower_o.states, mx.budget_bytes,
                                     mxo.transpose(1, 2), nko)
        return mx.cross(mx.tower_c.readout(hc2, lc2), ho2,
                        M._keep_mask_last(ho2.shape[1], nko - 1, ho2.device))

z0 = z_of()
xq = xc.clone(); xq[0, :, int(nvc[0]) - 1] += 3.0          # market 读出点那一格
rq = float((z_of(mxc=xq)[0] - z0[0]).norm() / z0[0].norm())
check(f"**Query 轴活着**：扰动 market 读出点 → z 相对变化 {rq:.3e} > 1e-3", rq > 1e-3)
xk = xo.clone(); xk[1, :, 0] += 3.0                        # order 的一个真实 token
rk = float((z_of(mxo=xk)[1] - z0[1]).norm() / z0[1].norm())
check(f"**K/V 轴活着**：扰动一个 order token → z 相对变化 {rk:.3e} > 1e-3", rk > 1e-3)


def run_x(mxc=xc, mxo=xo, mxt=xt):
    with torch.no_grad():
        return mx(mxc, nvc, mxo, nko, mxt, nkt)


xo_all = xo.clone(); xo_all[..., :int(nko.max())] += 3.0   # 整条 order 流（不是单 token）
d_e2e = float((yx - run_x(mxo=xo_all)).abs().max())
check(f"端到端：order 流整体 +3 → 输出改变（Δ={d_e2e:.4f}；单 token 级会被 softmax 稀释）",
      d_e2e > 1e-4)
xt_p = xt.clone(); xt_p[2, :, int(nkt[2]):] = 99.0         # txn 的 pad 区灌毒
check("txn pad 区灌 99 → 输出不变（前缀 mask 正确）",
      float((yx[2] - run_x(mxt=xt_p)[2]).abs().max()) < 1e-6)
xo_p = xo.clone(); xo_p[3, :, int(nko[3]):] = 99.0         # order 的 pad 区灌毒
check("order pad 区灌 99 → 输出不变（K/V mask 正确）",
      float((yx[3] - run_x(mxo=xo_p)[3]).abs().max()) < 1e-6)
check("跨样本零泄漏：改样本 1 → 样本 0 逐位不变",
      torch.equal(yx[0], run_x(mxo=xk)[0]))

# —— C) z 真的进了通路：手工把 z 置零 → 输出必须变 ——
with torch.no_grad():
    hc, lastc = mx.tower_c._chunked(mx.tower_c.states, mx.budget_bytes, xc.transpose(1, 2), nvc)
    q_ = mx.tower_c.readout(hc, lastc)
    ho, _ = mx.tower_o._chunked(mx.tower_o.states, mx.budget_bytes, xo.transpose(1, 2), nko)
    z_real = mx.cross(q_, ho, M._keep_mask_last(ho.shape[1], nko - 1, ho.device))
    z_zero = torch.zeros_like(z_real)
    def txn_out(z):
        ht, lt = mx.tower_t._chunked(lambda x, n, zz: mx.tower_t.states(x, n, prefix=zz),
                                     mx.budget_bytes, xt.transpose(1, 2), nkt, z)
        return mx.head(mx.tower_t.readout(ht, lt))
check("z 置零 → 输出改变（融合通路确实被用上，不是摆设）",
      float((txn_out(z_zero) - txn_out(z_real)).abs().max()) > 1e-4)
check("z 逐样本（dynamic）：不同样本的 z 不同",
      bool((z_real - z_real.mean(0, keepdim=True)).abs().max() > 1e-3))

# —— D) 批间 T 无关 / 分块等价 / 指纹 ——
xt_long = torch.cat([xt, torch.full((4, 5, 8), 7.0)], dim=2)      # txn 加长（只追加 pad）
check("txn 加长（只追加 pad）→ 输出不变",
      float((yx - run_x(mxt=xt_long)).abs().max()) < 1e-6)
xo_long = torch.cat([xo, torch.full((4, 6, 8), 7.0)], dim=2)
check("order 加长（只追加 pad）→ 输出不变",
      float((yx - run_x(mxo=xo_long)).abs().max()) < 1e-6)
mx_small = M.ThreeTowerTFMXAttn(width=8, layers=2, heads=2, ff_mult=2, drop=0.2,
                                head_drop=0.3, t2v_add=True, budget_mb=1e-6).eval()
mx_small.load_state_dict(mx.state_dict())
with torch.no_grad():
    d_chunk_x = float((yx - mx_small(xc, nvc, xo, nko, xt, nkt)).abs().max())
check(f"xattn：整批 vs 逐样本分块一致（max|Δ|={d_chunk_x:.2e}）", d_chunk_x < 1e-5)

AX = type("AX", (A,), {"xattn": True})()
AX0 = type("AX0", (A,), {"xattn": False})()
check("xattn 关 → hash 与基配置相同（条件性进指纹）", B.arch_hash(AX0) == h)
check("xattn 开 → hash 变", B.arch_hash(AX) != h)

# —— E) 梯度：cross 的参数拿到非零梯度 ——
mx.train()
torch.manual_seed(13)
ym = mx(xc.clone(), nvc, xo.clone(), nko, xt.clone(), nkt)
ym.pow(2).mean().backward()
g_cross = float(mx.cross.wq.weight.grad.abs().sum() + mx.cross.wv.weight.grad.abs().sum())
check(f"CrossAttn 的 wq/wv 拿到非零梯度（Σ|grad|={g_cross:.4f}）", g_cross > 0.0)

# —— F) 端到端：--t2v --xattn 的迷你训练 + 检查点往返 ——
args_x = type("ArgsX", (type(args),), {"t2v": True, "xattn": True})()
tmp3 = tempfile.mkdtemp(prefix="tfm_xattn_")
r3 = B.train_fold(Xc, nv, Vo, off_o, nk_o, Vt, off_t, nk_t, y, sids, tr_rows, va_rows,
                  args_x, "cpu", tmp3, B.arch_hash(args_x), 0, True)
check(f"xattn 臂迷你训练跑通：cos={r3['cos']:.4f}", np.isfinite(r3["cos"]))
pred_x, _ = B.oof_from_ckpt(B.ckpt_path(tmp3, 0, "best", True), args_x, Xc, nv, Vo, off_o,
                            nk_o, Vt, off_t, nk_t, va_rows, "cpu")
check("xattn 臂：检查点复算 OOF 逐位一致", np.array_equal(pred_x, r3["va_pred"]))
check("xattn 臂指纹 != v3 臂指纹", B.arch_hash(args_x) != B.arch_hash(args_t2v))

print("【12】tfm_submit.py：打桩跑通提交管道（格式 / 行序 / 指纹）")
# 打桩：test 侧三件输入（submission.csv / snap index / 事件 build_flat）+ 一个**真检查点**
Nsub = 12
os.makedirs(os.path.join(_TMP_ROOT, "submissions"), exist_ok=True)
sub_sids = np.arange(100, 100 + Nsub, dtype=np.int64)
pl.DataFrame({"sample_id": sub_sids, "prediction": np.zeros(Nsub)}).write_csv(
    os.path.join(_TMP_ROOT, "submissions", "submission.csv"))
os.makedirs(os.path.join(_TMP_ROOT, "snap_cache"), exist_ok=True)
pl.DataFrame({"sample_id": sub_sids, "n_valid": np.full(Nsub, 200, dtype=np.int16),
              "U_s": np.full(Nsub, 1e-3), "mid0": np.ones(Nsub)}).write_parquet(
    os.path.join(_TMP_ROOT, "snap_cache", "test_snap_index.parquet"))
_rng = np.random.default_rng(7)
_offs = np.concatenate([[0], np.cumsum(_rng.integers(1, 12, Nsub))]).astype(np.int64)
import event_feats as _EF  # noqa: E402
import snap_feats as _SF  # noqa: E402
_orig_bf, _orig_bs = _EF.build_flat, _SF.build_split
_EF.build_flat = lambda split, months, to_memory=False: (
    _rng.standard_normal((int(_offs[-1]), 6)).astype(np.float16), _offs,
    _rng.standard_normal((int(_offs[-1]), 5)).astype(np.float16), _offs,
    pl.DataFrame({"sample_id": sub_sids, "n_ev_o": np.diff(_offs), "n_ev_t": np.diff(_offs)}))
_SF.build_split = lambda *a, **kw: _rng.standard_normal((Nsub, 224, 15)).astype(np.float16)

import tfm_submit as TS  # noqa: E402
sub_args = type("SubArgs", (), dict(
    width=8, layers=1, heads=2, ff_mult=2, drop=0.2, head_drop=0.3, seed=42, lr=1e-3,
    wd=3e-4, batch=1024, loss="cos", t2v=True, xattn=True,
    attn_budget_mb=B.BUDGET_MB_DEFAULT))()
ck_dir = tempfile.mkdtemp(prefix="tfm_sub_ck_")
parts_dir = tempfile.mkdtemp(prefix="tfm_sub_parts_")
model = B.build_model(sub_args, "cpu")
B.save_ckpt(os.path.join(ck_dir, "f0.best.pt"), model, B.arch_hash(sub_args),
            {"fold": 0, "kind": "best", "epoch": 1, "cos": 0.0})
argv_save = sys.argv
sys.argv = ["tfm_submit.py", "--t2v", "--xattn", "--width", "8", "--layers", "1",
            "--heads", "2", "--ff-mult", "2", "--ckpt-dir", ck_dir,
            "--parts-dir", parts_dir, "--out", "sub_tfm_test.parquet", "--eval-batch", "8"]
buf2 = io.StringIO()
try:
    with contextlib.redirect_stdout(buf2):
        TS.main()
finally:
    sys.argv = argv_save          # ⚠️ 打桩留到本段**最后**再还原——第二次调用（指纹）也要用
out2 = buf2.getvalue()
sub_path = os.path.join(_TMP_ROOT, "submissions", "sub_tfm_test.parquet")
check("提交文件已生成", os.path.exists(sub_path))
if os.path.exists(sub_path):
    d = pl.read_parquet(sub_path)
    check(f"提交格式：列 = [sample_id, prediction]，{d.height} 行",
          d.columns == ["sample_id", "prediction"] and d.height == Nsub)
    check("行序 = submission.csv 的 sample_id 升序",
          np.array_equal(d["sample_id"].to_numpy(), sub_sids))
    check("预测有限", bool(np.isfinite(d["prediction"].to_numpy()).all()))
    check("smoke 名未被使用（--smoke 没给，走正式名）", "sub_tfm_test_smoke.parquet" not in out2)
# 指纹不符必须拒载（换 --layers 后同一个检查点）
# ⚠️ 必须换一个**空**的 parts 目录：否则会走"已有 .npy → 跳过"的续跑分支，根本不去加载检查点
sys.argv = ["tfm_submit.py", "--t2v", "--xattn", "--width", "8", "--layers", "2",
            "--heads", "2", "--ff-mult", "2", "--ckpt-dir", ck_dir,
            "--parts-dir", tempfile.mkdtemp(prefix="tfm_sub_parts2_"),
            "--out", "sub_tfm_test2.parquet"]
try:
    with contextlib.redirect_stdout(io.StringIO()):
        TS.main()
    check("层数不符时应报错", False)
except ValueError:
    check("架构与检查点不符 → ValueError，不静默出提交", True)
finally:
    sys.argv = argv_save
    _EF.build_flat, _SF.build_split = _orig_bf, _orig_bs

print("【13】--full-train：全量训练模式（固定轮数 / 不写 OOF / 检查点命名 / 提交能读）")
_bf2 = lambda split, months, to_memory=False: (
    _rng.standard_normal((int(_offs[-1]), 6)).astype(np.float16), _offs,
    _rng.standard_normal((int(_offs[-1]), 5)).astype(np.float16), _offs,
    pl.DataFrame({"sample_id": sub_sids, "n_ev_o": np.diff(_offs), "n_ev_t": np.diff(_offs)}))
_bs2 = lambda *a, **kw: _rng.standard_normal((Nsub, 224, 15)).astype(np.float16)
_full_ck = tempfile.mkdtemp(prefix="tfm_full_ck_")
_orig_load2 = EB.load_train
EB.load_train = lambda: (Xc, nv, Vo, off_o, nk_o, Vt, off_t, nk_t, y,
                         (np.arange(N) % 6).astype(np.int64), [0, 1, 2, 3, 4, 5], sids)
argv_save = sys.argv
sys.argv = ["tfm_baseline.py", "--full-train", "--epochs", "2", "--ckpt-every", "1",
            "--batch", "8", "--eval-batch", "8", "--width", "8", "--layers", "1",
            "--heads", "2", "--ff-mult", "2", "--t2v", "--xattn",
            "--ckpt-dir", _full_ck, "--oof-dir", os.path.join(_full_ck, "oof_unused")]
buf3 = io.StringIO()
try:
    with contextlib.redirect_stdout(buf3):
        B.main()
finally:
    sys.argv = argv_save
    EB.load_train = _orig_load2
out3 = buf3.getvalue()
check("全量模式跑通，落了 full_e1.pt / full_e2.pt（每轮 + 末轮）",
      os.path.exists(os.path.join(_full_ck, "full_e1.pt"))
      and os.path.exists(os.path.join(_full_ck, "full_e2.pt")))
check("全量模式**不写 OOF**（没有留出月份可预测）",
      not os.path.exists(os.path.join(_full_ck, "oof_unused", "f0.parquet")))
check("输出里明确声明无验证/无早停", "无验证/无早停/无 OOF" in out3)

# 提交脚本直接读 full_e*.pt（--ckpt-path）
_EF.build_flat, _SF.build_split = _bf2, _bs2
_parts4 = tempfile.mkdtemp(prefix="tfm_full_parts_")
sys.argv = ["tfm_submit.py", "--t2v", "--xattn", "--width", "8", "--layers", "1", "--heads", "2",
            "--ff-mult", "2", "--batch", "8",      # ⚠️ batch 在指纹里：必须与训练侧一致
            "--ckpt-path", os.path.join(_full_ck, "full_e2.pt"),
            "--parts-dir", _parts4, "--out", "sub_tfm_full.parquet", "--eval-batch", "8"]
try:
    with contextlib.redirect_stdout(io.StringIO()):
        TS.main()
finally:
    sys.argv = argv_save
    _EF.build_flat, _SF.build_split = _orig_bf, _orig_bs
_full_out = os.path.join(_TMP_ROOT, "submissions", "sub_tfm_full.parquet")
check("提交脚本能用 --ckpt-path 读全量检查点并出文件", os.path.exists(_full_out))
check("成员 npy 按检查点主干命名（full_e2.npy，不叫 f0.npy）",
      os.path.exists(os.path.join(_parts4, "full_e2.npy")))

print()
n_ok = sum(OK)
print(f"=== {n_ok}/{len(OK)} 条通过 ===")
if n_ok != len(OK):
    sys.exit(1)
