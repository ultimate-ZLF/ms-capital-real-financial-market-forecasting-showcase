r"""diag_zero_target.py — `y==0` 样本的**输入侧成因**诊断（HYPOTHESES C-16 的前置；不训练、几分钟）。

背景（2026-09-23 本机实测，双塔CNN OOF，全局 cos 口径）
------------------------------------------------------
- `y==0` 的样本占 **5.54%**，吃掉 **7.01% 的 Σp²**，对 Σpy 贡献**恰好 0**
  → 理想化收益（把它们的预测置 0）**+0.00445**
- **反常**：这些样本上 |p| 均值 **0.322 > 全体 0.259**，且 `y==0` 比例随 |p| 十分位
  从 **4.00% 单调升到 8.60%**
- 用 5 个臂的 |p| 线性打分预测 `y==0`，留出奇月 **AUC 0.646**

**但是事后收缩只能兑现 +0.0005**（top-2% 里真 `y==0` 的只有 7.5%，基准率 5.5%）——
识别精度不够。所以真正要回答的是：**这个反常来自输入侧的什么**（HYPOTHESES 想法 4）。

四个候选成因，各自给出**可区分**的预测，本脚本一次分开它们：

| 假设 | 机制 | 可区分的预测 |
|---|---|---|
| H1 `U_s` 偏小 | tick 单位化拿 per-sample spread 中位数当分母 → 42 列被**放大** | `U_s` 小 / `rel_tick` 小的样本 `y==0` 更多、\|p\| 更大 |
| H2 pad 占比高 | C-12 已知 pad 块拿 27% 读出权重 → pad 主导的样本被推到共同偏移 | `pad_frac` 高的样本 `y==0` 更多、各臂更一致 |
| H3 静默/无成交 | `y==0` ⟺ 未来 240s 无成交；活动量自相关 | `o_n`/`t_n`/`last_gap` 与 `y==0` 强相关 |
| H4 尺度/regime | 月度波动率 | 与 H3 相关但落在**月份**维度（corr(月度 y==0 率, 月度 σ) = −0.70 已实测）|

⚠️ 本脚本**只用输入侧可得量做分组**（`n_valid` / `U_s` / `o_n` / `t_n` / 因子列），
**绝不拿 |y| 分组**——那是 oracle，只作为**上界对照**打印在 C 节。

用法
----
    python analysis/diag_zero_target.py                       # 默认双塔CNN OOF
    python analysis/diag_zero_target.py --oof-dir snap_cache/oof_parts_twotower
    python analysis/diag_zero_target.py --no-factors          # 跳过因子表（省内存/时间）
    # 带上 E1 各臂，跑「各臂一致性」一节：
    python analysis/diag_zero_target.py --extra-oof \
        flow_cache/oof_parts_flow,flow_cache/oof_parts_both,flow_cache/oof_parts_mkt,\
snap_cache/oof_parts_learnpool_none

读：`$MSC_BASE/{train/label.parquet, snap_cache/*_snap_index.parquet, flow_cache/*_flow_index.parquet,
    factors/factors_m*.parquet}` + OOF 目录。缺哪个就跳过哪一节（打印提示，不报错）。

本机等价物（无需服务器，2026-09-23 已验证可跑通）：
    MSC_BASE=<合成 fixture> python analysis/diag_zero_target.py
    ——本机 `~/kaggle/runs/` 有 `tau_2026-09-15/label.parquet` 与
    `e1_2026-09-16/tt_oof/oof_all.parquet`，配上合成的 index 即可复算 A 节全部数字。
"""
import os
import glob
import argparse
import numpy as np
import polars as pl

# —— BASE 解析（与全项目 42 个脚本同格式；改平台默认值即可整机搬迁）——
BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
T_COARSE = 224          # 粗段窗口步数（snap_feats 的 T）——用于算 pad 占比

# 因子表里值得看的列（缺失就跳过；`o_n`/`t_n` 优先用 flow index 的，口径更直接）
FACTOR_COLS = ["last_gap", "mid_vol60", "mid_vol600", "spread0", "spread_mean60",
               "burst5", "accel", "vol_ratio", "sig_txn", "t_vol", "o_vol",
               "t_imb_2s", "range_pos", "patience_diff"]


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------
def rank_avg(x: np.ndarray) -> np.ndarray:
    """平均秩（并列取平均）——AUC 必须用平均秩，否则大量并列会系统性偏移。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = xs[1:] != xs[:-1]
    starts = np.where(is_new)[0].tolist()
    bidx = np.cumsum(is_new) - 1
    ends_excl = np.empty(len(starts), dtype=np.int64)
    ends_excl[:-1] = starts[1:]
    ends_excl[-1] = n
    blk_start = np.asarray(starts, dtype=np.float64)[bidx]
    blk_end = (ends_excl.astype(np.float64) - 1.0)[bidx]
    r = np.empty(n, dtype=np.float64)
    r[order] = (blk_start + blk_end) / 2.0 + 1.0
    return r


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """AUC(score 判 label==1)。>0.5 表示 score 越大越像 label==1。"""
    label = np.asarray(label, dtype=bool)
    n1 = int(label.sum())
    n0 = label.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rank_avg(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rank_avg(a), rank_avg(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def cos_of(p: np.ndarray, y: np.ndarray) -> float:
    """全局 cos（与 models/tabm/tabm_common.py::cos_score 同口径）。"""
    return float((p * y).sum() / np.sqrt((p ** 2).sum() * (y ** 2).sum()))


def rescaled_cos(p: np.ndarray, y: np.ndarray, key: np.ndarray) -> tuple[float, np.ndarray, float]:
    """按 key 分组、每组乘一个最优标量倍数后的 cos（**不改变组内相对幅度与方向**）。

    最大化 (Σ_s g_s A_s)/√(Σ_s g_s² B_s) 的解是 g_s ∝ A_s/B_s，代入得
        cos_max = √(Σ_s A_s²/B_s) / √(Σ y²)。
    返回 (cos, 每组倍数 g, 原始 cos)。A_s/B_s 用组内和算，避免逐样本除零。
    """
    A = np.zeros(int(key.max()) + 1)
    B = np.zeros(int(key.max()) + 1)
    np.add.at(A, key, p * y)
    np.add.at(B, key, p * p)
    ok = B > 0
    g = np.zeros_like(A)
    g[ok] = A[ok] / B[ok]
    num = float((A[ok] ** 2 / B[ok]).sum())
    c = float(np.sqrt(num) / np.sqrt((y ** 2).sum()))
    return c, g, cos_of(p, y)


def qbin(x: np.ndarray, n: int) -> np.ndarray:
    """等频分箱（0..n-1）。用于分层表与分组标定。"""
    q = np.quantile(x, np.linspace(0, 1, n + 1))
    q[-1] += 1.0
    return np.clip(np.searchsorted(q[1:-1], x), 0, n - 1).astype(np.int32)


def hdr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def read_oof(oof_dir: str) -> pl.DataFrame:
    """读 OOF：优先 oof_all.parquet，否则拼 f*.parquet。"""
    allp = os.path.join(oof_dir, "oof_all.parquet")
    if os.path.exists(allp):
        return pl.read_parquet(allp).select("sample_id", "pred")
    parts = sorted(glob.glob(os.path.join(oof_dir, "f*.parquet")))
    if not parts:
        raise FileNotFoundError(f"{oof_dir} 下没有 oof_all.parquet 也没有 f*.parquet")
    return pl.concat([pl.read_parquet(f).select("sample_id", "pred") for f in parts])


# ----------------------------------------------------------------------------
# 自检（本机即可跑，不需要任何数据）：python analysis/diag_zero_target.py --selftest
# ----------------------------------------------------------------------------
def selftest() -> int:
    rng = np.random.default_rng(0)
    bad = 0

    def chk(cond: bool, msg: str) -> None:
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        bad += 0 if cond else 1

    print("1. rank_avg：与朴素双重循环逐位一致（含大量并列）")
    ok = True
    for _ in range(200):
        n = int(rng.integers(2, 40))
        x = rng.integers(0, 5, n).astype(np.float64)          # 大量并列
        ref = np.array([1.0 + sum(v < xi for v in x) + 0.5 * sum(v == xi for v in x) - 0.5
                        for xi in x], dtype=np.float64)
        ok &= np.allclose(rank_avg(x), ref)
    chk(ok, "200 组随机并列数组（值域 0..4）全部一致")

    print("2. AUC：与朴素成对比较一致（并列算 0.5）")
    ok = True
    for _ in range(100):
        n = int(rng.integers(4, 60))
        s = rng.integers(0, 4, n).astype(np.float64)
        lb = rng.random(n) < 0.5
        if lb.all() or (~lb).all():
            continue
        pos, neg = s[lb], s[~lb]
        ref = ((pos[:, None] > neg[None, :]).sum()
               + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (pos.size * neg.size)
        ok &= abs(auc(s, lb) - ref) < 1e-12
    chk(ok, "100 组随机数组全部一致")

    print("3. rescaled_cos：返回的 g 确实取到返回的 cos，且任何随机 g 都不超过它")
    ok = True
    for _ in range(200):
        n = int(rng.integers(30, 300))
        yv = rng.standard_normal(n)
        pv = 0.3 * yv + rng.standard_normal(n)               # 有一点信号
        key = rng.integers(0, 6, n).astype(np.int32)
        c, g, c0 = rescaled_cos(pv, yv, key)
        ok &= abs(cos_of(pv * g[key], yv) - c) < 1e-12        # 自洽
        gr = rng.random((50, int(key.max()) + 1)) + 1e-6      # 随机正倍数
        ok &= max(cos_of(pv * gr[i][key], yv) for i in range(50)) <= c + 1e-12
    chk(ok, "200 组随机数组：自洽 + 随机 g 不越界")

    print("4. rescaled_cos：退化为「一组」时等于原 cos")
    yv = rng.standard_normal(500)
    pv = 0.4 * yv + rng.standard_normal(500)
    c, _, c0 = rescaled_cos(pv, yv, np.zeros(500, dtype=np.int32))
    chk(abs(c - c0) < 1e-12, "单组 → 与原 cos 恒等（尺度自由度对 cos 无效）")

    print("5. qbin：等频性（每箱样本数相差 ≤1）")
    v = rng.standard_normal(10000)
    cnts = np.bincount(qbin(v, 10), minlength=10)
    chk(cnts.max() - cnts.min() <= 1, f"箱计数 {cnts.min()}~{cnts.max()}")

    print(f"\n{'全部通过' if bad == 0 else f'{bad} 条失败'}（共 5 组检查）")
    return bad


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default="snap_cache/oof_parts_twotower",
                    help="逐折 OOF 目录（相对 BASE 或绝对路径）")
    ap.add_argument("--extra-oof", default="",
                    help="逗号分隔的额外 OOF 目录，用于「各臂一致性」一节的对照（可留空）")
    ap.add_argument("--no-factors", action="store_true", help="跳过因子表（省内存/时间）")
    ap.add_argument("--bins", type=int, default=10, help="分层表的箱数（默认 10）")
    ap.add_argument("--selftest", action="store_true",
                    help="只跑数学自检（本机可跑，不需要任何数据）")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    oof_dir = args.oof_dir if os.path.isabs(args.oof_dir) else os.path.join(BASE, args.oof_dir)
    print(f"BASE     = {BASE}\nOOF      = {oof_dir}")

    # ---- 载入 ----------------------------------------------------------------
    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).select(
        "month", "sample_id", "target")
    df = lab

    snap_i = os.path.join(BASE, "snap_cache/train_snap_index.parquet")
    if os.path.exists(snap_i):
        df = df.join(pl.read_parquet(snap_i).select("sample_id", "n_valid", "U_s", "mid0"),
                     on="sample_id", how="left")
    else:
        print(f"⚠️ 缺 {snap_i} → 跳过 pad / U_s 相关分析")

    flow_i = os.path.join(BASE, "flow_cache/train_flow_index.parquet")
    if os.path.exists(flow_i):
        df = df.join(pl.read_parquet(flow_i).select("sample_id", "o_n", "t_n"),
                     on="sample_id", how="left")
        df = df.with_columns([pl.col("o_n").cast(pl.Float64), pl.col("t_n").cast(pl.Float64)])
    else:
        print(f"⚠️ 缺 {flow_i} → 跳过事件数相关分析")

    factor_cols_found: list[str] = []
    if not args.no_factors:
        fs = sorted(glob.glob(os.path.join(BASE, "factors/factors_m*.parquet")))
        if fs:
            # 用 `columns=` 做投影下推：71 个月文件 × ~50 列全读进来会白吃几 GB（容器限额 72 GiB）
            fdf = pl.concat([pl.read_parquet(
                f, columns=["sample_id"] + [c for c in FACTOR_COLS
                                            if c in pl.read_parquet_schema(f)])
                for f in fs])
            factor_cols_found = [c for c in fdf.columns if c != "sample_id"]
            if factor_cols_found:
                # NaN 计数（只看找到的那几列，够用且便宜）
                fdf = fdf.with_columns(
                    pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int16)
                                       for c in factor_cols_found]).alias("nan_cnt"))
            df = df.join(fdf, on="sample_id", how="left")
            print(f"因子表：并入 {len(factor_cols_found)} 列 + nan_cnt（{len(fs)} 个月文件）")
        else:
            print(f"⚠️ {BASE}/factors 下没有 factors_m*.parquet → 跳过因子列")

    df = df.join(read_oof(oof_dir), on="sample_id", how="inner").sort("sample_id")
    print(f"合并后 {df.height:,} 行 × {df.width} 列")

    # ⚠️ 行没对齐时 join 会**静默产生 null**（index 行序/子集不一致），下游会算出一片 nan。
    #    这里显式报出来——「结果全是 nan」不该靠人肉猜。
    if df.height != lab.height:
        print(f"⚠️ 合并后行数 {df.height:,} != label 行数 {lab.height:,}"
              f"（OOF 少 {lab.height - df.height:,} 行）——A/B 节的比例会偏高，注意")
    for c in ("pred", "n_valid", "U_s", "o_n", "t_n"):
        if c in df.columns:
            nn = df[c].null_count()
            if nn:
                print(f"⚠️ `{c}` 有 {nn:,} 个 null（{100 * nn / df.height:.2f}%）"
                      f"——多半是对不齐；该列参与的分层可能是错的")
    if df["pred"].null_count() or df["target"].null_count():
        raise SystemExit("pred/target 有 null，先查 OOF 与 label 的对齐，别急着读结论")

    y = df["target"].to_numpy().astype(np.float64)
    p = df["pred"].to_numpy().astype(np.float64)
    mo = df["month"].to_numpy().astype(np.int64)
    z = (y == 0)
    P2 = float((p ** 2).sum())
    Y2 = float((y ** 2).sum())

    # ---- A. 复现反常 ---------------------------------------------------------
    hdr("A. 复现反常（校验 OOF 口径 + |y| 分层账）")
    c0 = cos_of(p, y)
    print(f"全局 cos（全量池化） = {c0:.5f}   ← 与文档核对（双塔CNN 记 0.12022）")
    per = [cos_of(p[(mo - k) % 6 == 0], y[(mo - k) % 6 == 0]) for k in range(6)]
    print(f"逐折 cos = {[f'{c:.5f}' for c in per]}   均值 {np.mean(per):.5f}"
          f"   ← 文档记 0.12280")

    print(f"\n{'分层':<16}{'占比':>8}{'Σpy 贡献':>11}{'Σp² 成本':>11}{'|p| 均值':>10}{'命中率':>9}")
    edges = [(0.0, 1e-4), (1e-4, 1e-3), (1e-3, 3e-3), (3e-3, 1e-2), (1e-2, np.inf)]
    for lo, hi in edges:
        s = (np.abs(y) >= lo) & (np.abs(y) < hi)
        if not s.any():
            continue
        sv = s & (y != 0)                       # 命中率只在 y!=0 上算
        hit = (np.sign(p[sv] * y[sv]) > 0).mean() if sv.any() else float("nan")
        tag = f"|y|∈[{lo:g},{'inf' if np.isinf(hi) else f'{hi:g}'})"
        print(f"{tag:<16}"
              f"{100 * s.mean():>7.2f}%{100 * (p[s] * y[s]).sum() / (p * y).sum():>10.2f}%"
              f"{100 * (p[s] ** 2).sum() / P2:>10.2f}%{np.abs(p[s]).mean():>10.3e}{hit:>9.4f}")
    print(f"{'y==0':<16}{100 * z.mean():>7.2f}%{'0.00%':>10}{100 * (p[z] ** 2).sum() / P2:>10.2f}%"
          f"{np.abs(p[z]).mean():>10.3e}{'—':>9}")
    print(f"  理想化：把 y==0 的预测置 0 → cos "
          f"{cos_of(np.where(z, 0.0, p), y):.5f}（{cos_of(np.where(z, 0.0, p), y) - c0:+.5f}）")

    print(f"\n|p| 均值：y==0 {np.abs(p[z]).mean():.4f} | y!=0 {np.abs(p[~z]).mean():.4f}"
          f"  → 比值 {np.abs(p[z]).mean() / np.abs(p[~z]).mean():.3f}（>1 = 反常）")
    print(f"\n按 |p| 十分位看 y==0 比例（基准 {100 * z.mean():.2f}%）：")
    kp = qbin(np.abs(p), 10)
    for i in range(10):
        s = kp == i
        print(f"  十分位{i}  |p| 均值 {np.abs(p[s]).mean():.4f}   y==0 比例 {100 * z[s].mean():5.2f}%")

    extra = [d for d in args.extra_oof.split(",") if d.strip()]
    if extra:
        print("\n各臂两两相关：y==0 上 vs y!=0 上（显著更低 = 输出更像噪声；更高 = 共同偏差）")
        arms = {os.path.basename(oof_dir.rstrip("/")): p}
        for d in extra:
            dd = d.strip()
            dd = dd if os.path.isabs(dd) else os.path.join(BASE, dd)
            try:
                a = read_oof(dd).sort("sample_id")
                if a.height == df.height:
                    arms[os.path.basename(dd.rstrip("/"))] = a["pred"].to_numpy().astype(np.float64)
                else:
                    print(f"  ⚠️ {dd} 行数 {a.height} != {df.height}，跳过")
            except Exception as e:
                print(f"  ⚠️ {dd}: {e}")
        names = list(arms)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = arms[names[i]], arms[names[j]]
                c_z = np.corrcoef(a[z], b[z])[0, 1]
                c_n = np.corrcoef(a[~z], b[~z])[0, 1]
                print(f"  {names[i]:<22}-{names[j]:<22} {c_z:+.4f} / {c_n:+.4f}  差 {c_z - c_n:+.4f}")

    # ---- B. 候选成因 ---------------------------------------------------------
    hdr("B. 候选成因（每个量各给出可区分的预测）")
    cand: dict[str, np.ndarray] = {}
    if "n_valid" in df.columns:
        nv = df["n_valid"].to_numpy().astype(np.float64)
        cand["pad_frac"] = 1.0 - nv / T_COARSE          # H2
        cand["n_valid"] = nv
    if "U_s" in df.columns:
        cand["U_s"] = df["U_s"].to_numpy().astype(np.float64)           # H1
        if "mid0" in df.columns:
            cand["rel_tick"] = cand["U_s"] / df["mid0"].to_numpy().astype(np.float64)
    for c in ("o_n", "t_n"):
        if c in df.columns:
            cand[c] = np.log1p(df[c].to_numpy().astype(np.float64))     # H3（对数化）
    if "nan_cnt" in df.columns:
        cand["nan_cnt"] = df["nan_cnt"].to_numpy().astype(np.float64)
    for c in factor_cols_found:
        v = df[c].to_numpy().astype(np.float64)
        if np.isfinite(v).sum() > 0:
            cand[c] = np.nan_to_num(v, nan=np.nanmedian(v))

    print(f"{'量':<16}{'AUC(y==0)':>11}{'ρ(,|y|)':>10}{'ρ(,|p|)':>10}   y==0 比例随十分位（低→高）")
    for nm, v in cand.items():
        ks = qbin(v, args.bins)
        # ⚠️ 整数/大量并列的量（如 nan_cnt）会让等频分箱出现**空格**——打印 `-` 而不是 nan
        cells = []
        for i in range(args.bins):
            s = ks == i
            cells.append(f"{100 * z[s].mean():5.1f}" if s.any() else "    -")
        print(f"{nm:<16}{auc(v, z):>11.4f}{spearman(v, np.abs(y)):>10.4f}"
              f"{spearman(v, np.abs(p)):>10.4f}   {' '.join(cells)}")

    # 多量组合（月交错 6 折做诚实评估：按 month%6 切，避免 regime 泄漏）
    feat_names = [nm for nm in cand if nm not in ("nan_cnt",)]
    if len(feat_names) >= 2:
        X = np.column_stack([cand[nm] for nm in feat_names])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1.0, X.std(0))
        X = np.column_stack([X, np.ones(len(X))])
        oof_score = np.zeros(len(X))
        for k in range(6):
            tr, te = (mo % 6 != k), (mo % 6 == k)
            w, *_ = np.linalg.lstsq(X[tr], z[tr].astype(np.float64), rcond=None)
            oof_score[te] = X[te] @ w
        print(f"\n多量线性打分（{len(feat_names)} 列，月交错 6 折诚实口径）"
              f"AUC(y==0) = {auc(oof_score, z):.4f}")
        print("  最强单量 AUC 见上表；若多量显著更高 → 成因是**多个量共同**的，不是一个")
    else:
        oof_score = None

    # ---- C. 可兑现增益（本脚本的核心）----------------------------------------
    hdr("C. 可兑现增益：用**合法量**分组做 cos 最优重标定")
    print("（对照：按 |y| 分组是 oracle，只作上界；|y| 在 test 上不可得）\n")
    oracle_key = qbin(np.abs(y), 20)
    oc, _, _ = rescaled_cos(p, y, oracle_key)
    print(f"  {'oracle: |y| 20 组（上界）':<38} cos {oc:.5f}  ({oc - c0:+.5f})")
    mc = rescaled_cos(p, y, mo.astype(np.int32))[0]
    print(f"  {'按月份 71 组（regime 上界）':<38} cos {mc:.5f}  ({mc - c0:+.5f})")
    zp = rescaled_cos(p, y, z.astype(np.int32))[0]
    print(f"  {'oracle: 仅 y==0 与否（2 组）':<38} cos {zp:.5f}  ({zp - c0:+.5f})\n")
    print("  合法量（test 上真的可得）：")
    best_legit = (c0, "—")
    for nm, v in cand.items():
        c, g, _ = rescaled_cos(p, y, qbin(v, args.bins))
        flag = " ⭐" if c - c0 > 0.001 else ""
        print(f"    {nm:<34} cos {c:.5f}  ({c - c0:+.5f}){flag}")
        if c > best_legit[0]:
            best_legit = (c, nm)
    if oof_score is not None:
        c, _, _ = rescaled_cos(p, y, qbin(oof_score, args.bins))
        print(f"    {'多量组合打分':<34} cos {c:.5f}  ({c - c0:+.5f})"
              f"{' ⭐' if c - c0 > 0.001 else ''}")
        if c > best_legit[0]:
            best_legit = (c, "多量组合")
    print(f"\n  最强合法量：{best_legit[1]}  →  {best_legit[0] - c0:+.5f}")
    print("  ⚠️ 这是**组内最优倍数**（用 |y| 定的组内最优，合法量只负责分组）→ 仍是上界；")
    print("     真实可兑现还要乘一个「分组精度」折扣（本机已实测：|p| 打分收缩只兑现 +0.0005）")

    # ---- D. train vs test 的分布（跨 regime 风险）----------------------------
    hdr("D. train vs test 的分布对照（跨 regime 风险）")
    for split in ("train", "test"):
        si = os.path.join(BASE, f"snap_cache/{split}_snap_index.parquet")
        fi = os.path.join(BASE, f"flow_cache/{split}_flow_index.parquet")
        if not os.path.exists(si):
            print(f"⚠️ 缺 {si}，跳过 {split}")
            continue
        t = pl.read_parquet(si).select("sample_id", "n_valid", "U_s", "mid0")
        if os.path.exists(fi):
            t = t.join(pl.read_parquet(fi).select("sample_id", "o_n", "t_n"),
                       on="sample_id", how="left")
        t = t.with_columns((1.0 - pl.col("n_valid") / T_COARSE).alias("pad_frac"))
        print(f"\n[{split}]  n={t.height:,}")
        for c in ("pad_frac", "n_valid", "U_s", "o_n", "t_n"):
            if c not in t.columns:
                continue
            v = t[c].to_numpy().astype(np.float64)
            v = v[np.isfinite(v)]
            qs = np.quantile(v, [0.05, 0.25, 0.50, 0.75, 0.95])
            print(f"  {c:<10} 均值 {v.mean():>10.4g} | q05/25/50/75/95 = "
                  f"{qs[0]:.4g} / {qs[1]:.4g} / {qs[2]:.4g} / {qs[3]:.4g} / {qs[4]:.4g}")

    print("\n  ↳ 若 test 的 pad_frac / 事件数分布明显偏离 train，则 `y==0` 的真实比例也会偏；")
    print("    在 train 上按 5.5% 学到的行为到 test 上会系统性错位（**需要一次 LB 验证**）")

    hdr("完成")
    print("判读指南：")
    print("  · 哪个量的 AUC(y==0) 最高 → 那就是反常的直接来源")
    print("  · C 节里哪个合法量的增益最大 → 那就是「清洗」该先动的地方")
    print("  · 若 C 节全部 ≈ 0 → 清洗无处可洗，C-16 该转向输入侧的其它靶子")
    print("  · 无论结果如何：**判据必须含一次 LB 验证**（CLAUDE.md #13 实践含义①）")


if __name__ == "__main__":
    main()
