"""screen_fine_channels.py — 细段候选通道的**前置筛选**（不训练，纯 numpy）。

## 为什么需要它

时序侧一次试验 = 重建（细段现建 ≈1.5min）+ 重训全部序列模型（TCN 6 折 1.73h）≈ 2 小时，
**不可能一条一条试**。所以先用一个不需要 GPU、几分钟跑完的闸门把明显死的捞掉。

## 它做什么

对每条候选通道，把 `(N, 60)` 的序列**降成 per-sample 标量**再算 cos——**两个口径都算**，
因为它们对齐的是不同的东西：

| 口径 | 测的是 | 对齐谁 |
|---|---|---|
| **末 2 格均值**（bin 58,59）| "预测时刻这个量到了什么水平" | **模型的末步读出**（细段塔读 bin 59）|
| **全窗均值**（60 bin）| "过去一分钟的累积" | **表格因子的定义**（`t_imb_2s` 这类窗口聚合）|

再按 `analysis/factor_eval.py` 的口径算 **71 月符号稳定性 + 逐月 cos std**
（避雷 #3：单月检验会骗人，标准是符号稳定性，不是单月 corr）。

## ⚠️ 闸门 −1：两套 binning 逐位对照

`fine_derive` 与 `flow_feats` **各自**把 `order`/`transaction` 聚合成每秒一格。两套实现
一旦漂移（bin 差一格、空 bin 约定不同），筛网的数字会**看着正常但全是错的**。
所以先取一个月块，两边各算一遍 `t_imb` / `x_to_buy_diff`，要求逐位一致（容差按 f16 量化
步长给）。**不过就整表作废。**

## ⚠️ 闸门 0：判据必须先过**真阳性验证**（CLAUDE.md #15）

`t_imb` / `x_to_buy_diff` 是两条**已知有用**的通道（FACTORS.md 因子表：0.0733 / 0.0214），
本脚本把它们**混在候选里一起跑**。若它俩的 cos 没到参照值的 0.6 倍，说明是
**binning / 行序 / target 对齐出了问题**，此时**表里其他任何数字都不可信**——
直接判"判据无效"，不要拿它选特征。（`check_flow_signal.py` 的"缓存坏了会被读成 flow 冗余"
是同一类陷阱。）

## ⚠️ 定位：只捞明显死的，不选活的

**别拿它当选择器**。依据是避雷 #17：表格侧同一套闸门把 LGBM 后来证明**有用**的因子
判死了（`t_has_data_15`、`rel_spread_2_0` 在树里有实权）。它测不出两件事：
1. **模型自己能不能学出来**——一条通道的 cos 可能很高，但卷积本来就算得出；
2. **通道之间的冗余**——两条 cos 都高的通道可能完全共线。
   粗段实证（2026-09-27）：22 条派生通道整批进模型反而 **−0.0033**，且筛网**测不出**
   模型会不会把 45% 的第一层权重分给筛网判为噪声的通道。

所以阈值要松：只淘汰"cos≈0 **且**符号不稳"的。**其余整批进模型**。

用法（服务器）：
    python seq/screen_fine_channels.py --months 0,12,24,36,48,60,70   # 跨 regime 先小跑
    python seq/screen_fine_channels.py                                # 全部 71 月
    python seq/screen_fine_channels.py --pool o_cancel_imb,t_imb_run   # 只筛几条
"""
import argparse
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fine_derive as fd  # noqa: E402
import flow_feats  # noqa: E402
from seq_common import BASE  # noqa: E402

# 真阳性参照（FACTORS.md 71 月因子表；缓存版的 `t_imb` 在空 bin 上写 0，应**略低**）
REF_EXPECT = {"t_imb": ("t_imb_2s", 0.0733), "x_to_buy_diff": ("x_to_buy_diff", 0.0214)}
REF_TOL = 0.6          # 与 check_flow_signal.py 同口径

_W: dict = {}


def cos(a, b):
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def orth_cos(sig, ctrl, y):
    """`sig` 对 `ctrl`（可多条，含截距）最小二乘投影后，**残差**与 `y` 的 cos。

    度量的是**增量信号**——`ctrl` 已经解释掉的那部分不算。本筛网的主体（`cos` 列）
    量的是**边际**信号，两条通道完全共线时两个 cos 都会很高；这一列才是"除了
    `ctrl` 之外还剩下什么"。`ctrl` 能完全解释 `sig` 时返回 nan。

    ⚠️ 本列**只对 `ctrl` 去冗余**，不是"对模型的 16 条基通道去冗余"的替代品。
    """
    if ctrl.size == 0:
        return cos(sig, y)
    A = np.column_stack([np.ones(sig.shape[0]), ctrl])
    coef, *_ = np.linalg.lstsq(A, sig, rcond=None)
    r = sig - A @ coef
    if np.sqrt((r * r).sum()) < 1e-9 * np.sqrt((sig * sig).sum() + 1e-30):
        return float("nan")
    return cos(r, y)


def _init_worker(sids, u_s, names):
    """spawn 的子进程入口：把只读数据一次性送进去。"""
    _W.update(sids=sids, u_s=u_s, names=names)


def _worker(job):
    """子进程任务：算一个月块的 (full, last2)。"""
    cid, lo, hi, s0, cnt, out_at = job
    f, l = fd.reduce_chunk("train", lo, hi,
                           _W["sids"][s0:s0 + cnt], _W["u_s"][s0:s0 + cnt], _W["names"])
    return cid, out_at, f, l


def _max_workers(workers=None):
    """`--workers` 缺省 = 1（串行）。见 `_pool` 的 fork 警告——默认并行太危险。"""
    return max(1, int(workers if workers is not None else 1))


def verify_binning(sids, u_all, months_arg):
    """把本模块的 binning 与 `flow_feats` **逐位对照**——防两套实现静默漂移。

    取第一个月块，两边各算一遍**全部 16 条基通道**：
    `flow_feats.build_chunk` 产 f16，本模块产 f32，所以容差按 f16 的量化步长给
    （`|v|≤1` 时 f16 步长 4.88e-4；`o_new_buy` 这类 asinh 通道值域更宽，故用 1e-3）。
    **bin 定义或空 bin 约定一旦漂了，差值会是 0.1 量级**——比容差大两个数量级，分得开。
    判不过就整表作废。
    """
    cid, lo, hi = next(iter(flow_feats.iter_chunks("train", months_arg)))
    sel = (sids >= lo) & (sids <= hi)
    sc, uc = sids[sel], u_all[sel]
    X16, _ = flow_feats.build_chunk("train", lo, hi, sc, uc)
    d = fd.derive_all(fd.build_raw_chunk("train", lo, hi, sc), uc, list(fd.BASE_NAMES))
    # ⚠️ 容差必须按 **f16 的相对精度**给，不能用固定 atol：`o_px_dev` 值域到 ±50，
    # 那里的 f16 量化步长是 0.031，固定 atol=1e-3 会把它误判成"不一致"——
    # 而真正的 binning 漂移在**相对**口径下仍有 ~100× 的分离度（相对误差 0.1 量级）。
    worst, wname, wsample = 0.0, "", None
    for i, nm in enumerate(fd.BASE_NAMES):
        a = X16[:, :, flow_feats.IDX[nm]].astype(np.float64)
        b = d[:, :, i].astype(np.float64)
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1.0)
        j = int(np.argmax(rel))
        if rel.flat[j] > worst:
            worst, wname = float(rel.flat[j]), nm
            wsample = (float(a.flat[j]), float(b.flat[j]))
    return cid, int(sc.size), worst, wname, wsample


def gather(names, months_arg, workers):
    """跑一遍数据，返回 `(full (n,K), last2 (n,K), y (n,), marr (n,))`。

    只处理 `months_arg` 指定的月——`flow_feats.iter_chunks` 按月切块，
    所以 `--months` **真的**减少计算量（粗段筛网第一版在这里静默忽略过，见其 docstring）。
    """
    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).sort("sample_id")
    y_all = lab["target"].to_numpy().astype(np.float64)
    marr = lab["month"].to_numpy()
    sids = lab["sample_id"].to_numpy()
    counts = lab.group_by("month").len().sort("month")["len"].to_numpy()
    u_all = flow_feats.load_u_s("train", sids)

    chunks = list(flow_feats.iter_chunks("train", months_arg))
    jobs, pos = [], 0
    for cid, lo, hi in chunks:
        m = int(cid[1:])
        s0 = int(counts[:m].sum())
        cnt = int(counts[m])
        assert cnt > 0, f"月 {m} 没有样本"
        seg = sids[s0:s0 + cnt]
        assert seg[0] == lo and seg[-1] == hi, \
            f"月 {m} 的 sample_id 不连续（[{seg[0]},{seg[-1]}] vs [{lo},{hi}]）——" \
            "行号映射不可用，不要用本脚本"
        jobs.append((cid, lo, hi, s0, cnt, pos))
        pos += cnt

    n = pos
    K = len(names)
    full = np.zeros((n, K), dtype=np.float64)
    last2 = np.zeros((n, K), dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    m_out = np.empty(n, dtype=marr.dtype)
    print(f"  月份 {len(chunks)} 个 / 共 {n:,} 行（全量 {y_all.size:,}）", flush=True)

    t0 = time.time()
    nw = _max_workers(workers)
    if nw <= 1 or len(jobs) <= 1:
        _init_worker(sids, u_all, names)
        for i, job in enumerate(jobs):
            _, out_at, f, l = _worker(job)
            k = f.shape[0]
            full[out_at:out_at + k], last2[out_at:out_at + k] = f, l
            print(f"  {i+1}/{len(jobs)} 块（{time.time()-t0:.0f}s）", flush=True)
    else:
        import multiprocessing as mp
        # ⚠️ **绝不能用 `fork`**（2026-09-27 实测，见 CLAUDE.md #17）：父进程只要 import 过
        # polars，fork 出的子进程再调 polars（`scan_parquet` 和 `read_parquet` 都一样）
        # **无条件挂死**——4 个 worker 存活 7 分钟、CPU 时间 0:00:00、不报错、不退出。
        # `spawn` / `forkserver` 正常（子进程是全新解释器，运行时不被继承）。
        done = 0
        with mp.get_context("spawn").Pool(nw, initializer=_init_worker,
                                          initargs=(sids, u_all, names)) as pool:
            for cid, out_at, f, l in pool.imap_unordered(_worker, jobs):
                k = f.shape[0]
                full[out_at:out_at + k], last2[out_at:out_at + k] = f, l
                done += 1
                print(f"  {done}/{len(jobs)} 块完成（{nw} 进程，{time.time()-t0:.0f}s）",
                      flush=True)
    # 行号 → target/month：块按 (月, 行号) 升序拼接，与 lab 的行序一致
    idx = []
    for _, _, _, s0, cnt, _ in jobs:
        idx.append(np.arange(s0, s0 + cnt))
    idx = np.concatenate(idx)
    y[:] = y_all[idx]
    m_out[:] = marr[idx]
    return full, last2, y, m_out


def monthly_stats(sig, y, marr):
    """逐月 cos → (cos_mean, cos_std, pos_frac, 负月数)。口径照 `analysis/factor_eval.py`。"""
    cs = []
    for mm in np.unique(marr):
        m = marr == mm
        a, b = sig[m], y[m]
        if np.sqrt((a ** 2).sum()) > 0 and np.sqrt((b ** 2).sum()) > 0:
            cs.append(cos(a, b))
    if not cs:
        return float("nan"), float("nan"), float("nan"), 0
    cs = np.array(cs)
    return float(np.nanmean(cs)), float(np.nanstd(cs)), \
        float(np.mean(cs > 0)), int((cs < 0).sum())


def structural_check(sig):
    """结构硬检查（照 `check_flow_signal.py`）：死列 = 全常数 / 几乎全零。"""
    if not np.isfinite(sig).all():
        return "❌ 含 NaN/inf"
    if sig.std() == 0.0:
        return "❌ 死列（std=0）"
    if float((sig != 0).mean()) < 0.01:
        return f"❌ 几乎全零（非零率 {(sig != 0).mean():.3f}）"
    return ""


def main():
    p = argparse.ArgumentParser(description="细段候选通道前置筛选（不训练）")
    p.add_argument("--months", default=None,
                   help="逗号分隔月号；默认全部 71 月。**会真的只跑这些月**——"
                        "先小跑验管线用（建议 0,12,24,36,48,60,70 跨 regime）")
    p.add_argument("--pool", default=None,
                   help="只筛这些候选（逗号分隔）；默认 11 条候选 + 参照")
    p.add_argument("--workers", type=int, default=1,
                   help="并行进程数（**spawn**，绝不 fork——见 gather 里的警告）。默认 1 = 串行")
    p.add_argument("--ctrl", default="all",
                   help="正交化控制组：'all'（默认）= 全部 16 条基通道，也就是**模型实际看到的**"
                        "细段输入；'none' = 关闭该列；或逗号分隔的通道名")
    p.add_argument("--dead-cos", type=float, default=0.002,
                   help="|cos| 低于此且符号不稳 → 判死")
    args = p.parse_args()

    report = fd.parse_spec(args.pool) if args.pool else list(fd.POOL) + list(fd.REFS)
    if str(args.ctrl).lower() == "none":
        ctrl_names = []
    elif str(args.ctrl).lower() == "all":
        ctrl_names = list(fd.BASE_NAMES)
    else:
        ctrl_names = fd.parse_spec(args.ctrl)
    names = list(dict.fromkeys(report + ctrl_names))      # 控制组也要真算出来

    print(f"筛选 {len(report)} 条通道（口径：末2格均值 + 全窗均值；逐月 cos + 符号稳定性）")
    print(f"⚠️ 其中 {fd.REFS} 是**真阳性参照**，用来验证本判据有没有分辨力——"
          f"它们不过就整表作废。")
    print(f"   另有 {len(ctrl_names)} 条基通道参与计算，只作正交化控制组，不进报告表。\n")

    lab_v = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).sort("sample_id")
    sids_v = lab_v["sample_id"].to_numpy()
    cid, nv, worst, wname, wsample = verify_binning(
        sids_v, flow_feats.load_u_s("train", sids_v), args.months)
    ok_bin = worst < 1e-3
    print(f"闸门 −1（binning 与 flow_feats 逐位对照）：{cid} {nv:,} 样本，"
          f"{len(fd.BASE_NAMES)} 条基通道**最大相对差** {worst:.2e}（最差：{wname}"
          f"，缓存 {wsample[0]:.6g} vs 本模块 {wsample[1]:.6g}）  "
          f"{'PASS' if ok_bin else 'FAIL（容差 1e-3）'}\n")
    if not ok_bin:
        print("### 结论：**两套 binning 不一致**——筛网所有数字都不可信，先修管线。")
        return 1

    t0 = time.time()
    full, last2, y, marr = gather(names, args.months, args.workers)
    print(f"取数 + 派生完成：{full.shape}（{time.time()-t0:.0f}s）\n")

    # 正交化控制组 = 模型**已经看到**的 16 条基通道。残差 cos 才是"除它们之外还剩什么"。
    ctrl = np.column_stack([full[:, names.index(n)] for n in ctrl_names]) \
        if ctrl_names else np.zeros((full.shape[0], 0))
    if ctrl_names:
        print(f"正交化控制组 = {len(ctrl_names)} 条基通道（模型实际看到的细段输入）"
              f"—— '正交cos' 列是去掉它们之后的**增量信号**\n")

    rows = []
    for nm in report:
        k = names.index(nm)
        c_last, c_full = cos(last2[:, k], y), cos(full[:, k], y)
        c_orth = float("nan") if nm in ctrl_names else orth_cos(full[:, k], ctrl, y)
        m_mean, m_std, pos, neg = monthly_stats(full[:, k], y, marr)
        flag = structural_check(full[:, k])
        if not flag:
            strong = max(abs(c_last), abs(c_full))
            if strong < args.dead_cos and (np.isnan(pos) or pos < 0.55):
                flag = "❌ 判死（cos≈0 且符号不稳）"
            elif strong < args.dead_cos * 2.5:
                flag = "⚠️ 弱"
            else:
                flag = "✅ 保留"
        rows.append([nm, float((full[:, k] != 0).mean()), float(full[:, k].std()),
                     c_last, c_full, c_orth, m_mean, m_std, pos, neg, flag])

    # ---------------- 闸门 0：真阳性验证 ----------------
    gate_ok, gate_msg = True, []
    for r in rows:
        if r[0] in REF_EXPECT:
            src, ref = REF_EXPECT[r[0]]
            got = max(abs(r[3]), abs(r[4]))
            good = got >= REF_TOL * ref
            gate_ok &= good
            gate_msg.append(f"    {'PASS' if good else 'FAIL'}  {r[0]:<16} "
                            f"cos={got:.5f}  参照 {src} {ref:.4f}（阈值 {REF_TOL}×）")

    print(f"{'通道':<20}{'非零率':>8}{'std':>10}{'末2格cos':>10}{'全窗cos':>9}{'正交cos':>9}"
          f"{'月cos_μ':>9}{'月cos_σ':>9}{'符号稳':>8}{'负月':>5}  判定")
    print("-" * 118)
    for r in sorted(rows, key=lambda z: (z[0] not in REF_EXPECT, -max(abs(z[3]), abs(z[4])))):
        mark = "★" if r[0] in REF_EXPECT else " "
        print(f"{mark}{r[0]:<19}{r[1]:>8.3f}{r[2]:>10.4f}{r[3]:>10.5f}{r[4]:>9.5f}"
              f"{r[5]:>9.5f}{r[6]:>9.5f}{r[7]:>9.5f}{r[8]:>8.3f}{r[9]:>5d}  {r[10]}")
    print("-" * 118)

    print("\n闸门 0（真阳性验证，★ 行）：")
    print("\n".join(gate_msg))
    if not gate_ok:
        print("\n### 结论：**判据无效**——已知有用的参照通道都没测出来，说明本次运行的 "
              "binning / 行序 / target 对齐有问题。**表里其他数字一律不可信**，先修管线。")
        return 1

    kept = [r[0] for r in rows if not r[10].startswith("❌")]
    dropped = [r[0] for r in rows if r[10].startswith("❌")]
    cand = [x for x in kept if x not in REF_EXPECT]
    print(f"\n闸门 0 PASS。保留 {len(cand)}/{len(rows) - len(REF_EXPECT)} 条候选；"
          f"判死 {len(dropped)}：{','.join(dropped) or '（无）'}")
    print("\n⚠️ 本闸门只捞\"明显死的\"，**不选活的**（避雷 #17：它把 LGBM 有用的因子判死过）。")
    print("   筛完应把保留的**整批**给模型，不要据此做逐通道取舍。")
    if ctrl_names:
        print(f"   '正交cos' 一列专门用来识别**改头换面**的通道：`末2格cos`/`全窗cos` 量的是"
              f"**边际**信号，\n   与 {ctrl_names} 共线的通道两列都会高；只有 `正交cos` 高"
              f"才说明它有独立信息。")
    if cand:
        print(f"\n下一条命令（在 tcn_baseline 接上 --derive 之后）：\n"
              f"  python models/tcn/tcn_baseline.py --folds 6 --derive {','.join(cand)}\n"
              f"  --ckpt-dir snap_cache/ckpt_tcn_fd --oof-dir snap_cache/oof_parts_tcn_fd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
