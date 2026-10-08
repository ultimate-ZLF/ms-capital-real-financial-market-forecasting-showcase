"""test_fine_derive.py — `fine_derive` 的回归测试（本机 venv 即可跑，不需要数据）。

    /home/zlf/.venvs/lab/bin/python seq/test_fine_derive.py

覆盖：四类算子的手算用例、`derive_all` 的端到端数值、以及**判据本身的分辨力**
（CLAUDE.md #15：新判据先在合成样本上跑一组"应该高 / 应该低"的对照，看它分不分得开）。

⚠️ 这些用例**不依赖真实数据**——`derive_all` 收的是原始聚合 dict，不是 parquet。
所以本机能跑全部数值逻辑；只有 binning（polars 那段）要等服务器。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fine_derive as fd  # noqa: E402
import screen_fine_channels as screen  # noqa: E402  只借它的 cos / orth_cos

PASS = 0
FAIL = []


def check(cond, msg):
    global PASS
    if cond:
        PASS += 1
    else:
        FAIL.append(msg)
        print(f"  FAIL  {msg}")


def close(a, b, tol=1e-6):
    """`derive_all` 输出 f32，断言容差按 f32 给（1e-9 会到处假失败）。"""
    return np.allclose(np.asarray(a, dtype=np.float64),
                       np.asarray(b, dtype=np.float64), atol=tol, rtol=0)


def make_raw(n=2, T=fd.FINE_T):
    return {c: np.zeros((n, T), dtype=np.float64) for c in fd.ORDER_COLS + fd.TX_COLS}


def get(dev_names, raw, name):
    """跑一遍 derive_all 并取出指定通道。"""
    d = fd.derive_all(raw, np.ones(raw["nb"].shape[0]), dev_names)
    return d[:, :, dev_names.index(name)]


def cos(a, b):
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


# --------------------------------------------------------------------------- #

def test_spec():
    print("test_spec")
    check(fd.parse_spec(None) == [], "None → 空")
    check(fd.parse_spec("none") == [], "'none' → 空")
    check(fd.parse_spec("") == [], "空串 → 空")
    check(fd.parse_spec("t_imb, t_imb_run") == ["t_imb", "t_imb_run"], "逗号切分 + 去空格")
    try:
        fd.parse_spec("nope")
        check(False, "未知通道应报错")
    except SystemExit:
        check(True, "")
    # 候选池与参照必须无重叠，否则筛网的"排除参照"逻辑会把候选也排掉
    check(not (set(fd.POOL) & set(fd.REFS)), "POOL 与 REFS 不重叠")
    check(len(fd.POOL) == 11, f"候选池应为 11 条，实际 {len(fd.POOL)}")
    check(len(set(fd.ALL_NAMES)) == len(fd.ALL_NAMES), "名字不重复")


def test_ratio():
    print("test_ratio")
    num = np.array([3.0, -2.0, 5.0, 1.0])
    den = np.array([4.0, 2.0, 0.0, -1.0])
    r = fd._ratio(num, den)
    check(close(r, [0.75, -1.0, 0.0, 0.0]), f"den<=0 一律 0，实际 {r}")
    # 形状广播
    check(fd._ratio(np.ones((3, 1)), np.array([2.0, 4.0])).shape == (3, 2), "广播形状")


def test_run_signed():
    print("test_run_signed")
    v = np.zeros((1, 8))
    v[0, :5] = [1.0, 1.0, -1.0, 1.0, 0.0]
    out = fd._run_signed(v, 10.0)
    check(close(out[0, :5], [0.1, 0.2, -0.1, 0.1, 0.0]),
          f"连续同向累加，实际 {out[0, :5]}")
    check(close(out[0, 5:], 0.0), "0 值之后保持 0")
    # 截断
    w = np.ones((1, 30)) * 0.5
    o2 = fd._run_signed(w, 10.0)
    check(close(o2[0, -1], 1.0) and close(o2[0, 11], 1.0),
          f"run 截断在 cap=10，实际 {o2[0, -1]}")


def test_since():
    print("test_since")
    cond = np.zeros((1, 60), dtype=bool)
    cond[0, 20] = True
    out = fd._since(cond, 30.0)
    check(close(out[0, 20], 0.0), "命中当格 = 0")
    check(close(out[0, 21], 1 / 30), "之后逐格递增")
    check(close(out[0, 19], 30 / 30), "命中之前用 cap 截断（前向累进，无未来信息）")
    check(close(out[0, 59], 1.0), "超过 cap 饱和到 1")
    empty = fd._since(np.zeros((1, 60), dtype=bool), 30.0)
    check(close(empty, 1.0), "从未发生 → 全 1.0")


def test_roll():
    print("test_roll")
    v = np.arange(12, dtype=np.float64)[None, :]
    check(close(fd._roll_sum(v, 3)[0], [0, 1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]),
          f"滑动和应为最近 3 格，实际 {fd._roll_sum(v,3)[0]}")
    check(close(fd._roll_max(v, 3)[0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
          f"滑动最大，实际 {fd._roll_max(v,3)[0]}")
    w = np.array([[5.0, 1.0, 9.0, 2.0]])
    check(close(fd._roll_max(w, 2)[0], [5, 5, 9, 9]), f"滑动最大含当前格，实际 {fd._roll_max(w,2)[0]}")


def test_peak():
    print("test_peak")
    v = np.zeros((2, 5))
    v[0] = [1.0, -3.0, 2.0, 0.0, 0.0]      # |max| 在 idx1
    v[1] = [1.0, 2.0, 0.5, 0.0, 0.0]       # |max| 在 idx1，符号相反
    out = fd._peak_signed(v)
    check(close(out[0], -3.0), f"取 |v| 最大那格的**带符号**值，实际 {out[0]}")
    check(close(out[1], 2.0), f"同上，实际 {out[1]}")
    check(out.shape == v.shape, "广播到全窗")


def test_roll_corr():
    print("test_roll_corr")
    x = np.arange(40, dtype=np.float64)[None, :] + 1.0
    check(close(fd._roll_corr(x, 2 * x, 20)[0, 39], 1.0, tol=1e-6), "完全正相关 → 1")
    check(close(fd._roll_corr(x, -3 * x, 20)[0, 39], -1.0, tol=1e-6), "完全负相关 → -1")
    const = np.ones((1, 40))
    check(close(fd._roll_corr(const, x, 20)[0, 39], 0.0), "常数序列 → 0（分母守卫）")
    check(close(fd._roll_corr(x, x, 20)[0, 18], 0.0), "窗口未满 → 0")
    # 单个 20 格窗口的 r 本身有 σ≈1/√19≈0.23 的抖动，不能拿一格当判据——
    # 对全部满窗取均值，才算"这条判据在噪声上不给假阳性"（CLAUDE.md #15）。
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((1, 400)), rng.standard_normal((1, 400))
    r_ind = fd._roll_corr(a, b, 20)[0, 19:]
    check(abs(r_ind.mean()) < 0.1 and np.abs(r_ind).max() < 0.8,
          f"独立噪声：381 个窗均值 {r_ind.mean():+.4f}（max|r| {np.abs(r_ind).max():.3f}）")
    check(np.abs(fd._roll_corr(a, a, 20)[0, 19:]).min() > 0.999,
          "同一序列自相关 → 全为 1（判据在真阳性上必亮）")


def test_derive_base():
    """基通道复现：逐条对照 `flow_feats.build_chunk` 的算式（数值口径必须一致）。"""
    print("test_derive_base")
    raw = make_raw(n=1)
    raw["nb"][0, :] = 2.0                     # 每秒 2 单位新挂买（全 60 格）
    raw["ovol"][0, :], raw["opxv"][0, :] = 2.0, 2.0 * 0.003
    d = fd.derive_base(raw, np.array([0.001]))
    check(close(d["o_new_buy"][0, 0], np.arcsinh(1.0)),
          f"asinh(2/每秒均量2)，实际 {d['o_new_buy'][0,0]}")
    check(close(d["o_new_imb"][0, 0], 1.0), "全买 → +1")
    check(close(d["o_px_dev"][0, 0], 0.0), "每格 VWAP == 全窗 VWAP → 0")
    check(close(d["flow_cnt"][0, 0], 0.0) and close(d["has_event"][0, 0], 0.0), "无事件")
    check(close(d["x_flow_gap"][0], np.minimum(np.arange(60) + 1, 10) / 10.0),
          f"从未有事件 → 逐格递增后截断，实际 {d['x_flow_gap'][0,:4]}")
    # 分母下限 1：稀疏样本不被放大
    raw2 = make_raw(n=1)
    raw2["nb"][0, 0] = 0.5
    check(close(fd.derive_base(raw2, np.array([0.001]))["o_new_buy"][0, 0], np.arcsinh(0.5)),
          "o_unit 下限 1（0.5/60=0.008 → 抬到 1）")
    # t_size_rel = 本格均量/全窗均量 − 1
    raw3 = make_raw(n=1)
    raw3["tb"][0, 0], raw3["tvol"][0, 0], raw3["tcnt"][0, 0] = 3.0, 3.0, 1.0
    raw3["tb"][0, 1], raw3["tvol"][0, 1], raw3["tcnt"][0, 1] = 1.0, 1.0, 3.0
    d3 = fd.derive_base(raw3, np.array([0.001]))
    # ⚠️ 分母带 EPS（`t_size_smp + EPS`，与 flow_feats 逐字一致）→ 期望值必须一起带
    check(close(d3["t_size_rel"][0, 0], 3.0 / (1.0 + fd.EPS) - 1.0),
          f"3/(1+EPS)−1，实际 {d3['t_size_rel'][0,0]}")
    check(close(d3["t_size_rel"][0, 1], (1 / 3) / (1.0 + fd.EPS) - 1.0),
          f"(1/3)/(1+EPS)−1，实际 {d3['t_size_rel'][0,1]}")
    check(close(d3["t_imb"][0, 0], 1.0), "bin0 全买 → +1")
    # 基通道与候选必须无重名（否则 derive_all 的 update 会静默覆盖）
    check(not (set(fd.BASE_NAMES) & set(fd.POOL)), "基通道与候选池不重名")
    check(set(fd.REFS) <= set(fd.BASE_NAMES), "参照通道 ⊂ 基通道")
    check(len(fd.BASE_NAMES) == 16, f"基通道应 16 条，实际 {len(fd.BASE_NAMES)}")


def test_derive_end_to_end():
    print("test_derive_end_to_end")
    names = fd.ALL_NAMES
    raw = make_raw(n=2)
    # 撤单失衡（bin 5）
    raw["cb"][0, 5], raw["cs"][0, 5] = 3.0, 1.0
    raw["cb"][1, 5], raw["cs"][1, 5] = 0.0, 2.0
    # 撤/新挂比（bin 6）
    raw["cb"][0, 6], raw["nb"][0, 6] = 1.0, 3.0       # → 1/4
    raw["cs"][1, 6], raw["ns"][1, 6] = 6.0, 2.0       # → 6/8
    # 失衡符号序列（bin 0..4）：+, +, −, +, 0（每笔量 1、笔数 1，保持 tvol == tb+ts == tcnt）
    for k, (b, s) in enumerate([(1, 0), (1, 0), (0, 1), (1, 0), (0, 0)]):
        raw["tb"][0, k], raw["ts"][0, k] = float(b), float(s)
        raw["tvol"][0, k], raw["tcnt"][0, k] = float(b + s), float(b + s)
    # 挂价离散（bin 9）：std=0.004 / U=0.002 → 2 tick
    raw["opx_std"][0, 9] = 0.004
    # --- 以下三组刻意让 `tvol == tb + ts == tcnt`，使 per-sample 单笔均量 = 1（阈值 2 受控）---
    # 大单方向（bin 7）：tvol=16，(8−2)/16 = 0.375
    raw["tb"][0, 7], raw["ts"][0, 7] = 8.0, 8.0
    raw["tvol"][0, 7], raw["tcnt"][0, 7] = 16.0, 16.0
    raw["big_buy"][0, 7], raw["big_sell"][0, 7] = 8.0, 2.0
    # 组内相邻成交间隔（bin 8）：3 笔，跨度 0.7s → 0.35
    raw["tb"][0, 8], raw["ts"][0, 8] = 1.5, 1.5
    raw["tvol"][0, 8], raw["tcnt"][0, 8] = 3.0, 3.0
    raw["tsec_max"][0, 8], raw["tsec_min"][0, 8] = 10.9, 10.2
    # 大单距离（bin 20）：t_maxv=5 > 2×均量=2 → 触发
    raw["tb"][0, 20], raw["ts"][0, 20] = 1.5, 1.5
    raw["tvol"][0, 20], raw["tcnt"][0, 20], raw["t_maxv"][0, 20] = 3.0, 3.0, 5.0
    # 突发（bin 30..32 事件数 1,5,2）
    raw["ocnt"][0, 30] = 1.0
    raw["tb"][0, 31], raw["ts"][0, 31] = 1.0, 1.0
    raw["tvol"][0, 31], raw["tcnt"][0, 31] = 2.0, 2.0
    raw["ocnt"][0, 31] = 3.0
    raw["ocnt"][0, 32] = 2.0
    # 自洽性自检：tvol 必须等于 tb+ts（真实数据里每条成交非买即卖）
    check(close(raw["tvol"][0], raw["tb"][0] + raw["ts"][0]), "测试夹具自洽：tvol = tb+ts")
    check(close(raw["tvol"][0].sum(), raw["tcnt"][0].sum()), "测试夹具自洽：单笔均量 = 1")

    u = np.array([0.002, 0.002])
    d = fd.derive_all(raw, u, names)
    check(d.shape == (2, fd.FINE_T, len(names)), f"形状 {d.shape}")
    check(np.isfinite(d).all(), "无 NaN/inf")

    g = lambda nm: d[:, :, names.index(nm)]           # noqa: E731

    check(close(g("o_cancel_imb")[0, 5], 0.5), f"撤单失衡 +0.5，实际 {g('o_cancel_imb')[0,5]}")
    check(close(g("o_cancel_imb")[1, 5], -1.0), f"撤单失衡 -1.0，实际 {g('o_cancel_imb')[1,5]}")
    check(close(g("o_cancel_ratio_buy")[0, 6], 0.25), "买侧撤/新挂 = 1/4")
    check(close(g("o_cancel_ratio_sell")[1, 6], 0.75), "卖侧撤/新挂 = 6/8")
    check(close(g("t_big_imb")[0, 7], 0.375), f"大单净方向 6/16，实际 {g('t_big_imb')[0,7]}")
    check(close(g("t_intragap")[0, 8], 0.35), f"(0.9-0.2)/(3-1)=0.35，实际 {g('t_intragap')[0,8]}")
    check(close(g("t_intragap")[0, 9], 0.0), "只 1 笔 → 0（谈不上间隔）")
    check(close(g("o_px_disp")[0, 9], 2.0), f"0.004/0.002 = 2 tick，实际 {g('o_px_disp')[0,9]}")
    check(close(g("t_imb_run")[0, :5], [0.1, 0.2, -0.1, 0.1, 0.0]),
          f"带符号连续长度，实际 {g('t_imb_run')[0,:5]}")
    check(close(g("since_big_txn")[0, 20], 0.0), "大单当格 = 0")
    check(close(g("since_big_txn")[0, 24], 4 / 30), "大单后 4 格 = 4/30")
    check(close(g("since_big_txn")[0, 19], 1.0), "大单前 = cap（因果，无未来信息）")
    # 突发：bin31 处最近 10 格最大事件数 = 5 → arcsinh(5)
    check(close(g("txn_burst_10s")[0, 31], np.arcsinh(5.0)),
          f"arcsinh(前 10 格最大事件数)，实际 {g('txn_burst_10s')[0,31]}")
    check(close(g("txn_burst_10s")[0, 30], np.arcsinh(1.0)), "窗口更早 → 最大值更小")
    # 峰值：整窗 |imb| 最大的 bin
    check(close(g("t_imb_peak")[0], 1.0), "整窗 |imb| 最大 = +1")
    # 参照自洽：t_imb 与手算一致
    check(close(g("t_imb")[0, 0], 1.0) and close(g("t_imb")[0, 2], -1.0), "参照 t_imb")
    check(close(g("t_imb")[0, 59], 0.0), "无成交 → 0（与 flow_feats 同约定）")


def test_refs_and_criterion():
    """CLAUDE.md #15：判据先在合成样本上验证分辨力，再拿它下结论。"""
    print("test_refs_and_criterion")
    rng = np.random.default_rng(7)
    n = 4000
    y = rng.standard_normal(n)
    planted = y * 1.0 + rng.standard_normal(n) * 3.0          # 真阳性：应测出中等 cos
    noise = rng.standard_normal(n)                            # 真阴性：应测出 ≈0
    c_pos, c_neg = cos(planted, y), cos(noise, y)
    check(c_pos > 0.2, f"真阳性应被检出，实际 cos={c_pos:.4f}")
    check(abs(c_neg) < 0.05, f"真阴性应判为 ≈0，实际 cos={c_neg:.4f}")
    check(c_pos > 5 * abs(c_neg), f"分辨力：{c_pos:.4f} vs {c_neg:.4f}")
    # ⚠️ 配套小陷阱（CLAUDE.md #15）：参照里别把 `rng.standard_normal(...)` 写进生成器表达式
    # ——那会让"应当相同"的行实际上每次都是新随机向量，"平移族"参照系自己就是错的。
    # 这里钉住两种写法的差别：
    wrong = np.array([rng.standard_normal(50) for _ in range(3)])   # 每行都是新的随机向量
    base = rng.standard_normal(50)
    right = np.stack([base + i * 0.0 for i in range(3)])            # 三行完全相同
    check(not np.allclose(wrong[0], wrong[1]), "生成器表达式里的 rng → 行间不同")
    check(np.allclose(right[0], right[1]) and np.allclose(right[1], right[2]),
          "先生成再 stack → 行间可相同（可用作平移族参照）")


def test_orth_cos():
    """CLAUDE.md #15：新判据（正交化 cos）先在合成样本上验证分辨力，再拿它读筛网表。

    场景刻意造成 `t_imb_run` 那种形态：一个量**大部分是已有通道的重新编码**，
    边际 cos 很高、增量却接近 0。判据必须能把这两种情形分开。
    """
    print("test_orth_cos")
    rng = np.random.default_rng(11)
    n = 200_000
    y = rng.standard_normal(n)
    z = rng.standard_normal(n)                       # 与 y 独立的成分
    ctrl = (0.7 * y + rng.standard_normal(n))[:, None]   # 模型已看到：与 y 相关 0.50

    # 1) 完全共线 → 残差恒 0 → nan（**不能**读成"有信号"）
    check(np.isnan(screen.orth_cos(ctrl[:, 0].copy(), ctrl, y)), "完全共线 → nan")

    # 2) sig = ctrl + 纯独立成分 z → 边际 cos 高，增量 ≈ 0（这正是要抓的形态）
    sig2 = ctrl[:, 0] + z
    c_marg2 = screen.cos(sig2, y)
    c_orth2 = screen.orth_cos(sig2, ctrl, y)
    check(c_marg2 > 0.4, f"边际 cos 应高（≈0.443），实际 {c_marg2:.4f}")
    check(abs(c_orth2) < 0.02, f"增量应≈0，实际 {c_orth2:.5f}")
    check(c_marg2 > 20 * abs(c_orth2), f"分辨力：{c_marg2:.4f} vs {c_orth2:.5f}")

    # 3) sig 自带独立信号 y+3z → 增量 cos 应为正值（≈0.216），且低于边际（≈0.316）
    sig3 = y + 3.0 * z
    c_marg3, c_orth3 = screen.cos(sig3, y), screen.orth_cos(sig3, ctrl, y)
    check(0.15 < c_orth3 < 0.28, f"增量 cos ≈0.216，实际 {c_orth3:.4f}")
    check(c_orth3 < c_marg3, f"去掉已有成分后应更低：{c_orth3:.4f} < {c_marg3:.4f}")

    # 4) 与 y 无关的纯噪声 → 两个口径都 ≈0
    noise = rng.standard_normal(n)
    check(abs(screen.orth_cos(noise, ctrl, y)) < 0.02, "纯噪声 → ≈0")


def main():
    for fn in [test_spec, test_ratio, test_run_signed, test_since, test_roll,
               test_peak, test_roll_corr, test_derive_base, test_derive_end_to_end,
               test_refs_and_criterion, test_orth_cos]:
        fn()
    print(f"\n{PASS} 条断言通过", end="")
    if FAIL:
        print(f"，{len(FAIL)} 条**失败**：")
        for m in FAIL:
            print(f"  - {m}")
        return 1
    print("，全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
