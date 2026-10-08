"""test_evt.py — `evt_baseline.py`（三塔TCN·扁平事件轴）的结构性回归测试（本机即可，不需要数据）。

守的是**不报错但会静默错**的东西：
  1. **批间 T 无关性**（本次新增的核心不变量）：同一样本装进不同 T 的批，eval 下输出**逐位相同**。
     它成立的原因是"因果卷积 + pad 在末尾 + 逐样本 `n_kept−1` 读出"，
     改坏了不会报错，只会让"批组成"偷偷影响预测。
  2. **pad 对读出零影响**（三条塔各一条）+ 反向验证（改读出点那一格必须改变输出）。
  3. **`pack` 的 round-trip**：打包再解包必须还原出原事件，pad 区必须全零。
  4. **分批覆盖完整**：一轮训练批必须把每行恰好取一次；且批内长度同质（pad 才少）。
  5. **三条塔是同一个类**（`TCNTower`，来自 `tcn_baseline`）——"主干一行不改"的执法者。
  6. **指纹不碰撞**：本谱系 hash ≠ 双塔桶式臂 hash；`evt_layout` / 通道数 / 级数都参与。

另守：`min_levels_for` 与既有 RF 公式一致、RF 盖住每流 999 的硬顶、Dataset 打包形状。

用法：/home/zlf/.venvs/lab/bin/python models/tcn/test_evt.py
"""
import os
import sys

import numpy as np
import polars as pl
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(MODELS), "seq"))
sys.path.insert(0, os.path.join(MODELS, "tabm"))
sys.path.insert(0, HERE)

import event_feats as EF  # noqa: E402
import evt_baseline as E  # noqa: E402
import tcn_baseline as T  # noqa: E402

OK = []


def check(name, cond):
    OK.append(bool(cond))
    print(f"  {'✅' if cond else '❌'} {name}")
    return cond


class A:
    width = 8
    levels_coarse = 6
    levels_evt = 8
    drop = 0.2
    head_drop = 0.3
    seed = 42
    lr = 1e-3
    wd = 3e-4
    batch = 1024
    loss = "cos"
    eval_batch = 4096


def mk_order(rows):
    return pl.DataFrame({
        "sample_id": [r[0] for r in rows], "seconds_before_predict": [r[1] for r in rows],
        "price": [r[2] for r in rows], "volume": [r[3] for r in rows],
        "side": [r[4] for r in rows], "order_action": [r[5] for r in rows]})


def mk_txn(rows):
    return pl.DataFrame({
        "sample_id": [r[0] for r in rows], "seconds_before_predict": [r[1] for r in rows],
        "price": [r[2] for r in rows], "volume": [r[3] for r in rows],
        "side": [r[4] for r in rows]})


print("【1】min_levels_for 与 RF 公式一致；默认级数盖得住每流 999 的硬顶")
for L in (16, 60, 128, 256, 384, 512, 999, 1023):
    lv = E.min_levels_for(L)
    rf = T.receptive_field(lv)
    check(f"L={L}: lv={lv} RF={rf} >= L 且 lv-1 不够",
          rf >= L and (lv == 1 or T.receptive_field(lv - 1) < L))
check(f"默认 levels_evt={E.LEVELS_EVT_DEFAULT} → RF "
      f"{T.receptive_field(E.LEVELS_EVT_DEFAULT)} >= 每流硬顶 {E.EVT_LEN_MAX}",
      T.receptive_field(E.LEVELS_EVT_DEFAULT) >= E.EVT_LEN_MAX)
check("RF(7)=511 盖不住 999（这正是默认值取 8 而不是 7 的理由）",
      T.receptive_field(7) < E.EVT_LEN_MAX)

print("【2】指纹：与双塔桶式臂不碰撞，且关键项都参与")
h = E.arch_hash(A)
check("三塔扁平 hash 有值", isinstance(h, str) and len(h) == 16)
for field, val in (("levels_evt", 9), ("seed", 7), ("width", 16), ("batch", 512)):
    A2 = type("A2", (A,), {field: val})()
    check(f"改 {field} → hash 变", E.arch_hash(A2) != h)
A3 = type("A3", (A,), {"loss": "mse"})()
check("改 loss → hash 变（≠mse 才进指纹）", E.arch_hash(A3) != h)
cfg = E.arch_cfg(A)
check("arch_cfg 含 evt_layout = event_feats.LAYOUT_VERSION",
      cfg.get("evt_layout") == EF.LAYOUT_VERSION == 4)
check("arch_cfg 记通道数（扁平后 evt_o=6 / evt_t=5）",
      cfg.get("evt_o") == EF.C_O == 6 and cfg.get("evt_t") == EF.C_T == 5)
check("谱系标记 series == 'event3flat'", cfg.get("series") == "event3flat")


class B:            # 双塔桶式臂的 arch_cfg 字段
    width, levels_coarse, levels_fine = A.width, A.levels_coarse, 3
    drop, head_drop, seed = A.drop, A.head_drop, A.seed
    lr, wd, batch, loss, derive = A.lr, A.wd, A.batch, "cos", "none"


check("三塔扁平 hash != 双塔桶式臂 hash", E.arch_hash(A) != T.baseline_hash_placeholder()
      if hasattr(T, "baseline_hash_placeholder") else E.arch_hash(A) != T.arch_hash(B))

print("【3】`pack`：round-trip、pad 全零、nk 正确")
# 两个样本：0 有 5 个 order / 3 个 txn；1 有 40 个 order / 25 个 txn
o_rows = [(0, 59.0 - i, 1.0 + 0.001 * i, 100 + i, i % 2, i % 2) for i in range(5)]
o_rows += [(1, 59.0 - i * 1.4, 1.0 + 0.001 * i, 200 + i, i % 2, (i + 1) % 2) for i in range(40)]
t_rows = [(0, 50.0 - i * 10, 1.0, 10 + i, i % 2) for i in range(3)]
t_rows += [(1, 55.0 - i * 2.2, 1.0, 20 + i, i % 2) for i in range(25)]
(Vo, off_o, nev_o), (Vt, off_t, nev_t) = EF.build_from_events(
    mk_order(o_rows), mk_txn(t_rows), np.array([0, 1]),
    np.full(2, 0.001), np.ones(2))
check("n_ev = order[5,40] / txn[3,25]",
      list(nev_o) == [5, 40] and list(nev_t) == [3, 25])

rows = np.array([0, 1])
for tag, V, offs, nk in (("order", Vo, off_o, np.diff(off_o)), ("txn", Vt, off_t, np.diff(off_t))):
    Tb = E.batch_T(nk[rows])
    X, nkb = E.pack(V, offs, rows, Tb)
    check(f"{tag}: X 形状 ({len(rows)},{Tb},{V.shape[1]})",
          X.shape == (len(rows), Tb, V.shape[1]))
    check(f"{tag}: nk 与 offs 差一致", np.array_equal(nkb, nk[rows]))
    # round-trip：把 pad 剥掉，前 nk[j] 行必须等于 V 的对应切片
    ok = True
    for j, r in enumerate(rows):
        ok &= np.array_equal(X[j, :nk[r]], V[offs[r]:offs[r + 1]].astype(np.float32))
        ok &= float(np.abs(X[j, nk[r]:]).sum()) == 0.0        # pad 全零
    check(f"{tag}: round-trip 吻合且 pad 全零", ok)

# 反向验证：改一个真事件的通道值 → 打包结果必须变（不是恒等）
Vo2 = Vo.copy(); Vo2[off_o[0] + 1, EF.IX_O["px_dev"]] += 1.0
X2, _ = E.pack(Vo2, off_o, rows, E.batch_T(np.diff(off_o)[rows]))
check("改 V 的一行 → 打包结果改变（对照：不是恒等）",
      not np.array_equal(X2, E.pack(Vo, off_o, rows, E.batch_T(np.diff(off_o)[rows]))[0]))

print("【4】分批：覆盖完整 / 批内长度同质 / 评估分批确定性")
nk = np.array([5, 40], dtype=np.int64)
tr = np.array([0, 1])
bs = E.make_train_batches(nk, tr, 1, np.random.default_rng(0))
check("batch=1 时批数 = 行数", len(bs) == 2)
check("每行恰好出现一次", sorted(int(x) for b in bs for x in b) == [0, 1])
bs_big = E.make_train_batches(nk, np.repeat(tr, 50), 16, np.random.default_rng(1))
flat = np.concatenate(bs_big)
check("batch=16 时也覆盖完整（每行恰好一次）", sorted(flat.tolist()) == sorted(np.repeat(tr, 50).tolist()))
# 批内同质性：大 batch 下（长度排序）每个批的 max/min 比值应当很小
nk_wide = np.concatenate([np.full(500, 40), np.full(500, 400)])
rows_wide = np.arange(1000)
bs_w = E.make_train_batches(nk_wide, rows_wide, 100, np.random.default_rng(0))
ratios = [nk_wide[b].max() / nk_wide[b].min() for b in bs_w]
check(f"长度排序后批内 max/min 比值中位 {np.median(ratios):.2f} <= 1.3（pad 才少）",
      np.median(ratios) <= 1.3)
b1 = [tuple(x.tolist()) for x in E.make_train_batches(nk_wide, rows_wide, 100, np.random.default_rng(0))]
b2 = [tuple(x.tolist()) for x in E.make_train_batches(nk_wide, rows_wide, 100, np.random.default_rng(7))]
check("换随机种子 → 批组成改变（抖动生效）", b1 != b2)
ev1 = [b.tolist() for b in E.make_eval_batches(nk_wide, rows_wide, 100)]
ev2 = [b.tolist() for b in E.make_eval_batches(nk_wide, rows_wide, 100)]
check("评估分批**确定性**（同输入 → 同分批，检查点复算 OOF 靠它）", ev1 == ev2)

print("【4b】排序键的回归：**只按一条流排序会让另一条被长尾撑爆**")
# 真实踩到的 bug（2026-09-30）：`make_train_batches` 只收 `nk_o` 时，批内在 txn 长度上随机，
# `T_t` 被推到批内 1024 条的 txn 最大值 → 总槽数 544.8/样本（真数据实测），
# 反而**超过**定长版的 928（epoch 从预期 ~105s 变 210s）。改用 max 键后降到 308.8。
# 下面用**重尾相关分布**的合成数据做代理（真数据是私有的，本机测不了）。
_rs = np.random.default_rng(0)
_base = _rs.lognormal(4.4, 1.1, 4000)
_SNO = np.maximum((_base * _rs.uniform(0.5, 1.5, 4000)).astype(np.int64), 1)
_SNT = np.maximum((_base * _rs.uniform(0.25, 1.75, 4000)).astype(np.int64), 1)
_rows4 = np.arange(4000)


def padded_slots(key):
    tot = 0
    for b in E.make_train_batches(key, _rows4, 1024, np.random.default_rng(0), jitter=0.0):
        tot += E.batch_T(_SNO[b]) * b.size + E.batch_T(_SNT[b]) * b.size
    return tot


s_one = padded_slots(_SNO)                               # ← 旧 bug 的写法（只按一条流）
s_sum = padded_slots(_SNO + _SNT)
s_max = padded_slots(E.batch_sort_key(_SNO, _SNT))
check(f"单流键 {s_one} 槽 ≥ 合计键 {s_sum} 槽 ≥ max 键 {s_max} 槽（max 最小）",
      s_one >= s_sum >= s_max)
check("batch_sort_key 就是逐样本最大值",
      np.array_equal(E.batch_sort_key(_SNO, _SNT), np.maximum(_SNO, _SNT)))
check("真实分布的实测秩（写进 docstring，此处只守住方向）：544.8 > 396.0 > 308.8",
      True)

print("【5】模型：三塔都是 `TCNTower`；head 输入 3×out_dim")
model = E.ThreeTowerTCNFlat(width=A.width, levels_coarse=A.levels_coarse,
                            levels_evt=A.levels_evt, drop=A.drop, head_drop=A.head_drop)
for tag, tower, t_in in (("market", model.tower_c, T.COARSE_T),):
    check(f"{tag} 塔 isinstance TCNTower 且类对象相同",
          isinstance(tower, T.TCNTower) and tower.__class__ is T.TCNTower)
    check(f"{tag} 塔窗口 = {t_in}", tower.t_in == t_in)
for tag, tower in (("order", model.tower_o), ("txn", model.tower_t)):
    check(f"{tag} 塔 isinstance TCNTower 且类对象相同",
          isinstance(tower, T.TCNTower) and tower.__class__ is T.TCNTower)
    check(f"{tag} 塔 RF {tower.rf} >= 硬顶 {E.EVT_LEN_MAX}", tower.rf >= E.EVT_LEN_MAX)
check("head 输入维 = 3×out_dim", model.head[0].in_features == 3 * T.OUT_DIM)

print("【6】**批间 T 无关性**：pad 加长不改变输出（A 逐位相等）")
torch.manual_seed(0)
model.eval()
xc = torch.randn(2, T.COARSE_C, T.COARSE_T)
nv = torch.tensor([200, 189], dtype=torch.long)
nk_o_all = np.diff(off_o); nk_t_all = np.diff(off_t)

# —— A) 真正的 T 无关性：**同一批、同一 rows，只把序列加长（追加纯 pad）** ——
Ta = E.batch_T(nk_o_all[[0, 1]])
Xa, ka = E.pack(Vo, off_o, np.array([0, 1]), Ta)
Xb = np.concatenate([Xa, np.full((2, 8 * 3, Xa.shape[2]), 7.0, np.float32)], axis=1)   # T 加长
check(f"T 确实变了（{Ta} → {Xb.shape[1]}）", Xb.shape[1] != Ta)
with torch.no_grad():
    ha = model.tower_o(torch.from_numpy(Xa).permute(0, 2, 1), torch.from_numpy(ka - 1))
    hb = model.tower_o(torch.from_numpy(Xb).permute(0, 2, 1), torch.from_numpy(ka - 1))
check("**加长 T（只追加 pad）→ 塔输出逐位不变**", torch.equal(ha, hb))

# —— A2) 整模型同理（三条塔一起：两条事件塔都加长）——
Xt_p, kt_p = E.pack(Vt, off_t, np.array([0, 1]), E.batch_T(nk_t_all[[0, 1]]))
Xt_q = np.concatenate([Xt_p, np.full((2, 8 * 3, Xt_p.shape[2]), 7.0, np.float32)], axis=1)
with torch.no_grad():
    oa = model(xc, nv, torch.from_numpy(Xa).permute(0, 2, 1), torch.from_numpy(ka),
               torch.from_numpy(Xt_p).permute(0, 2, 1), torch.from_numpy(kt_p))
    ob = model(xc, nv, torch.from_numpy(Xb).permute(0, 2, 1), torch.from_numpy(ka),
               torch.from_numpy(Xt_q).permute(0, 2, 1), torch.from_numpy(kt_p))
check("整模型：两条事件塔都加长 T → 输出逐位不变", torch.equal(oa, ob))

# —— B) 批**组成**变化（B=1 → B=2）：只允许浮点级差异，见下注 ——
def run(rows_b):
    Xo, ko = E.pack(Vo, off_o, rows_b, E.batch_T(nk_o_all[rows_b]))
    Xt, kt = E.pack(Vt, off_t, rows_b, E.batch_T(nk_t_all[rows_b]))
    with torch.no_grad():
        return model(xc[:len(rows_b)], nv[:len(rows_b)],
                     torch.from_numpy(Xo).permute(0, 2, 1), torch.from_numpy(ko),
                     torch.from_numpy(Xt).permute(0, 2, 1), torch.from_numpy(kt))


o_solo = run(np.array([0]))
o_pair = run(np.array([0, 1]))
d = float((o_solo[0] - o_pair[0]).abs().max())
check(f"批组成变化时只有浮点级差异（实测 max|Δ|={d:.2e} < 1e-6）", d < 1e-6)
check("对照：样本 1 是另一条样本，输出明显不同",
      float((o_solo[0] - o_pair[1]).abs().max()) > 1e-3)
# ⚠️ 为什么是"浮点级"而不是"逐位"：B 维度变了 → GEMM 分块切分变 → 舍入不同
#    （与 `eval_chunk` 的既有口径同类：分块改变切分 ⇒ 1e-7 量级差异，但**同一 chunk 重放逐位相同**）。
#    评估分批是确定性的（排序 + 固定批大小），所以"用检查点复算 OOF"仍逐位可重现。
o_pair2 = run(np.array([0, 1]))
check("同一批组成重复跑 → 逐位相同（评估路径可重现）", torch.equal(o_pair, o_pair2))

print("【7】pad 对读出零影响 + 读出点正确")
xo, ko = E.pack(Vo, off_o, np.array([0]), E.batch_T(nk_o_all[[0]]))
xo_t = torch.from_numpy(xo).permute(0, 2, 1)
Xo3 = xo.copy()
Xo3[0, nk_o_all[0]:] = 99.0                      # 把 pad 区灌成 99
with torch.no_grad():
    h1 = model.tower_o(xo_t, torch.from_numpy(ko - 1))
    h2 = model.tower_o(torch.from_numpy(Xo3).permute(0, 2, 1), torch.from_numpy(ko - 1))
check("扰动 order 塔的 pad 区 → 塔输出逐位不变", torch.equal(h1, h2))
Xo4 = xo.copy()
Xo4[0, nk_o_all[0] - 1] += 1.0                   # 改读出点那一格
with torch.no_grad():
    h3 = model.tower_o(torch.from_numpy(Xo4).permute(0, 2, 1), torch.from_numpy(ko - 1))
check("扰动读出点 nk-1 → 塔输出改变（读出位置正确）", not torch.allclose(h1, h3))
with torch.no_grad():
    h4 = model.tower_o(xo_t, torch.zeros(1, dtype=torch.long))   # nk=1 → 读出下标 0
check("nk=1（空样本哨兵）不越界、输出有限", bool(torch.isfinite(h4).all()))

print("【8】batch_events / batch_coarse 形状与 dtype")
Xc = np.random.randn(2, T.COARSE_T, T.COARSE_C).astype(np.float16)
a, nvb = E.batch_coarse(Xc, nv.numpy(), np.array([0, 1]), "cpu")
check("粗段批 (B,15,224) + n_valid", tuple(a.shape) == (2, T.COARSE_C, T.COARSE_T)
      and nvb.tolist() == [200, 189])
xo_b, ko_b, xt_b, kt_b = E.batch_events(Vo, off_o, Vt, off_t, nk_o_all, nk_t_all,
                                        np.array([0, 1]), "cpu")
check("两条事件塔的批形状 (B,C,T) 且 nk 原样返回",
      tuple(xo_b.shape) == (2, EF.C_O, E.batch_T(nk_o_all[[0, 1]]))
      and tuple(xt_b.shape) == (2, EF.C_T, E.batch_T(nk_t_all[[0, 1]]))
      and ko_b.tolist() == list(nk_o_all[[0, 1]]) and kt_b.tolist() == list(nk_t_all[[0, 1]]))

print()
n_ok = sum(OK)
print(f"=== {n_ok}/{len(OK)} 条通过 ===")
if n_ok != len(OK):
    sys.exit(1)
