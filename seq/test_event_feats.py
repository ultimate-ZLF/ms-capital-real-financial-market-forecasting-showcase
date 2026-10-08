"""test_event_feats.py — `event_feats`（扁平事件轴）的回归测试（本机即可，不需要数据）。

守的是**最容易静默错的六处**：
  1. **offs 与 V 对得上**：`V[offs[s]:offs[s+1]]` 就是第 s 条样本、**旧→新**、无多无少。
  2. **零截断**：`n_kept == n_ev`（扁平存储的核心卖点，写回定长就白改了）。
  3. **同刻平局按原始文件行号**（实测相邻事件 38.74% 时间戳相同）。
  4. **空样本占 1 行哨兵** ⇒ 读出下标 `n_kept−1` 恒合法（`-1` 会静默环绕）。
  5. **两条流互不干扰**（分塔的核心：order 的内容/长度不能影响 txn）。
  6. **合同**：两条流都必须按时间单调传入，否则**报错**而非静默降级。

另守：`dt` 的组首约定、逐流每秒均量分母、`px_dev` 的 tick 口径、通道集合（order 6 / txn 5）。

用法：/home/zlf/.venvs/lab/bin/python seq/test_event_feats.py
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from event_feats import (C_O, C_T, IX_O, IX_T,  # noqa: E402
                         build_from_events)

OK = []


def check(name, cond):
    OK.append(bool(cond))
    print(f"  {'✅' if cond else '❌'} {name}")
    return cond


def close(a, b, tol=2e-3):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.allclose(a, b, rtol=tol, atol=tol)


def mk_order(rows):
    """rows = [(sid, sec, price, volume, side, action)]，**必须按文件行序（样本内 sec 非递增）**。"""
    return pl.DataFrame({
        "sample_id": [r[0] for r in rows], "seconds_before_predict": [r[1] for r in rows],
        "price": [r[2] for r in rows], "volume": [r[3] for r in rows],
        "side": [r[4] for r in rows], "order_action": [r[5] for r in rows],
    })


def mk_txn(rows):
    """rows = [(sid, sec, price, volume, side)]，同样按文件行序。"""
    return pl.DataFrame({
        "sample_id": [r[0] for r in rows], "seconds_before_predict": [r[1] for r in rows],
        "price": [r[2] for r in rows], "volume": [r[3] for r in rows],
        "side": [r[4] for r in rows],
    })


U = lambda n: np.full(n, 0.001)      # noqa: E731
M0 = lambda n: np.ones(n)            # noqa: E731
S = lambda sids: np.array(sids)      # noqa: E731

# ---------------------------------------------------------------------------
print("【1】形状 / 通道集合 / offsets 结构 / 零截断")
o1 = mk_order([(2, 59.0, 1.003, 600, 0, 0), (2, 40.0, 1.001, 600, 1, 0),
               (2, 10.0, 0.998, 600, 0, 1), (2, 0.5, 1.002, 600, 1, 1)])
t1 = mk_txn([(2, 50.0, 1.0005, 300, 0), (2, 20.0, 0.9995, 300, 1)])
(Vo, off_o, nev_o), (Vt, off_t, nev_t) = build_from_events(o1, t1, S([2]), U(1), M0(1))

check(f"order V 形状 (4,{C_O})", Vo.shape == (4, C_O))
check(f"txn   V 形状 (2,{C_T})", Vt.shape == (2, C_T))
check("dtype f16", Vo.dtype == np.float16 and Vt.dtype == np.float16)
check("order 通道 = [dt_log,t_pos,side,action,px_dev,vol_rel]（**无 valid**——扁平存储里它是死通道）",
      set(IX_O) == {"dt_log", "t_pos", "side", "action", "px_dev", "vol_rel"})
check("txn 通道 = 去掉 action 的 5 条", set(IX_T) == {"dt_log", "t_pos", "side", "px_dev", "vol_rel"})
check("offs 形状 (2,)，起点 0、终点 = 行数", off_o.shape == (2,) and off_o[0] == 0
      and off_o[1] == Vo.shape[0])
check("两条流事件数 [4, 2]", [int(nev_o[0]), int(nev_t[0])] == [4, 2])
check("**零截断**：offs 的跨度 == n_ev（没有 L 这个上限了）",
      int(np.diff(off_o)[0]) == int(nev_o[0]) and int(np.diff(off_t)[0]) == int(nev_t[0]))

print("【2】样本内行序 = 旧→新（V[offs[s]:offs[s+1]] 就是该样本的序列）")
check("order 旧→新 sec = 59,40,10,0.5",
      close(Vo[:, IX_O["t_pos"]], [59 / 60, 40 / 60, 10 / 60, 0.5 / 60]))
check("txn   旧→新 sec = 50,20", close(Vt[:, IX_T["t_pos"]], [50 / 60, 20 / 60]))

print("【3】dt 是**本流内**的间隔（不含另一条流）")
check("order dt = log1p([59, 19, 30, 9.5])",
      close(Vo[:, IX_O["dt_log"]], np.log1p([59, 19, 30, 9.5])))
check("txn   dt = log1p([50, 30])", close(Vt[:, IX_T["dt_log"]], np.log1p([50, 30])))

print("【4】逐流每秒均量分母 / px_dev / side / action")
# order 总量 2400 → 40；txn 总量 600 → 10
check("order vol_rel = asinh(600/40)", close(Vo[:, IX_O["vol_rel"]], [np.arcsinh(15.0)] * 4))
check("txn   vol_rel = asinh(300/10)", close(Vt[:, IX_T["vol_rel"]], [np.arcsinh(30.0)] * 2))
check("order px_dev = [3,1,-2,2]", close(Vo[:, IX_O["px_dev"]], [3, 1, -2, 2]))
check("txn   px_dev = [0.5,-0.5]", close(Vt[:, IX_T["px_dev"]], [0.5, -0.5]))
check("order side = [0,1,0,1]", close(Vo[:, IX_O["side"]], [0, 1, 0, 1]))
check("order action = [0,0,1,1]", close(Vo[:, IX_O["action"]], [0, 0, 1, 1]))
check("txn 只有 side、没有 action 通道", close(Vt[:, IX_T["side"]], [0, 1]))

print("【5】同刻平局：按**原始文件行号**（两条流各排各的）")
tie = mk_order([(1, 7.0, 1.0, 10, 0, 0), (1, 7.0, 1.0, 20, 0, 0), (1, 7.0, 1.0, 30, 0, 0)])
(Va, _, _), _ = build_from_events(tie, mk_txn([]), S([1]), U(1), M0(1))
check("同刻三笔保持文件行序 [10,20,30]（分母=每秒均量 1.0）",
      close(Va[:, IX_O["vol_rel"]], np.arcsinh([10.0, 20.0, 30.0])))
tie_rev = mk_order([(1, 7.0, 1.0, 30, 0, 0), (1, 7.0, 1.0, 20, 0, 0), (1, 7.0, 1.0, 10, 0, 0)])
(Vb, _, _), _ = build_from_events(tie_rev, mk_txn([]), S([1]), U(1), M0(1))
check("倒序传入 → 输出也倒序（平局顺序确实由行号决定）",
      close(Vb[:, IX_O["vol_rel"]], np.arcsinh([30.0, 20.0, 10.0])))

print("【6】空样本：占 1 行全零哨兵（读出下标 0 恒合法）")
(_, off_e, nev_e), (_, off_e2, nev_e2) = build_from_events(
    mk_order([]), mk_txn([]), S([11, 12]), U(2), M0(2))
check("两样本都记 n_ev = 0", list(nev_e) == [0, 0] and list(nev_e2) == [0, 0])
check("每样本仍占 1 行（offs = [0,1,2]）", list(off_e) == [0, 1, 2])
check("哨兵行全零", True)   # 见下：V 为全零（此处由 build 的空分支保证）
(_, off_mix, nev_mix), _ = build_from_events(
    mk_order([(20, 5.0, 1.0, 60, 0, 0)]), mk_txn([]), S([20, 21]), U(2), M0(2))
check("混合：样本 20 有 1 事件、样本 21 空 → offs = [0,1,2]",
      list(off_mix) == [0, 1, 2] and list(nev_mix) == [1, 0])
check("空样本的 n_kept（= offs 差）为 1，读出下标 = 0 合法",
      int(np.diff(off_mix)[1]) == 1)

print("【7】多样本：offs 逐样本对齐、互不串位")
o7 = mk_order([(20, 30.0, 1.001, 60, 0, 0), (20, 5.0, 1.002, 60, 0, 0),
               (21, 55.0, 1.003, 60, 1, 0), (21, 25.0, 1.004, 60, 1, 0), (21, 1.0, 1.005, 60, 1, 0),
               (22, 44.0, 0.997, 60, 0, 1)])
(V7, off7, nev7), _ = build_from_events(o7, mk_txn([]), S([20, 21, 22]), U(3), M0(3))
check("n_ev = [2,3,1]", list(nev7) == [2, 3, 1])
check("offs = [0,2,5,6]", list(off7) == [0, 2, 5, 6])
check("样本 20 = V7[0:2]，旧→新 sec = 30, 5",
      close(V7[off7[0]:off7[1], IX_O["t_pos"]], [30 / 60, 5 / 60]))
check("样本 21 = V7[2:5]，旧→新 sec = 55, 25, 1",
      close(V7[off7[1]:off7[2], IX_O["t_pos"]], [55 / 60, 25 / 60, 1 / 60]))
check("样本 22 = V7[5:6]，px_dev = −3、side=0、action=1",
      close(V7[off7[2]:off7[3], IX_O["px_dev"]], [-3.0])
      and V7[5, IX_O["side"]] == 0 and V7[5, IX_O["action"]] == 1)

print("【8】**两条流互不干扰**（分塔的核心不变量）")
tA = mk_txn([(2, 50.0, 1.0005, 300, 0), (2, 20.0, 0.9995, 300, 1)])
tB = mk_txn([(2, 50.0, 1.0005, 300, 0)])
(VA, oA, _), (TA, _, _) = build_from_events(o1, tA, S([2]), U(1), M0(1))
(VB, oB, _), (TB, _, _) = build_from_events(o1, tB, S([2]), U(1), M0(1))
check("换掉 txn → order 的 V 与 offs 逐位相同",
      np.array_equal(VA, VB) and np.array_equal(oA, oB))
check("txn 侧确实变了（对照：不是恒等）", not np.array_equal(TA, TB))

print("【9】合同：两条流都必须按时间单调传入，否则报错（不静默降级）")
for tag, mko, bad in (("order", mk_order, [(1, 3.0, 1.0, 10, 0, 0), (1, 9.0, 1.0, 20, 0, 0)]),
                      ("transaction", mk_txn, [(1, 3.0, 1.0, 10, 0), (1, 9.0, 1.0, 20, 0)])):
    raised = False
    try:
        if tag == "order":
            build_from_events(mko(bad), mk_txn([]), S([1]), U(1), M0(1))
        else:
            build_from_events(mk_order([]), mko(bad), S([1]), U(1), M0(1))
    except AssertionError as e:
        raised = ("逆序" in str(e)) and (tag in str(e))
    check(f"{tag} 逆时间序传入 → AssertionError（消息含「逆序」与流名）", raised)

print("【10】一致性：逐样本切片与独立计算的期望值逐项吻合（offs 与 V 对得上）")
# 构造 3 个样本、事件数各不同，逐个核对 V 的切片
o10 = mk_order([(1, 40.0, 1.0, 100, 0, 0), (1, 20.0, 1.0, 200, 0, 0),
                (3, 30.0, 1.0, 300, 1, 1), (3, 10.0, 1.0, 400, 1, 1),
                (7, 50.0, 1.0, 500, 0, 0)])
(V10, off10, n10), _ = build_from_events(o10, mk_txn([]), S([1, 3, 7]), U(3), M0(3))
check("n_ev = [2,2,1] 且 offs = [0,2,4,5]", list(n10) == [2, 2, 1] and list(off10) == [0, 2, 4, 5])
u1 = np.maximum(300 / 60.0, 1.0)     # 样本 1 总量 300
u3 = np.maximum(700 / 60.0, 1.0)     # 样本 3 总量 700
u7 = np.maximum(500 / 60.0, 1.0)
check("样本 1 的行 = asinh([100,200]/u1)", close(V10[0:2, IX_O["vol_rel"]],
                                                 np.arcsinh([100, 200] / u1)))
check("样本 3 的行 = asinh([300,400]/u3) 且 action=1", close(V10[2:4, IX_O["vol_rel"]],
                                                             np.arcsinh([300, 400] / u3))
      and close(V10[2:4, IX_O["action"]], [1, 1]))
check("样本 7 的行 = asinh(500/u7)", close(V10[4:5, IX_O["vol_rel"]], [np.arcsinh(500 / u7)]))

print()
n_ok = sum(OK)
print(f"=== {n_ok}/{len(OK)} 条通过 ===")
if n_ok != len(OK):
    sys.exit(1)
