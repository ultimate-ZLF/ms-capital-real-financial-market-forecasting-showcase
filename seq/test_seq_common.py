"""test_seq_common.py — batch_to_tensor 的 tcrop 语义回归测试（2026-09-15）。

背景：粗段缓存的 pad 在数组最末（最新那一端），真数据在位置 0..n_valid−1
（snap_feats.py 的 `_pos` 按秒数降序编号，写到 n_valid−1 为止）。
旧实现用 `[:, -tcrop:, :]` 取"数组最后 tcrop 行"，会把 pad 当成最新快照喂进池化，
而 mask 还写 `nb = min(n_valid, tcrop)` 标成"全有效"——pad 静默污染池化表示。
中位样本 n_valid=189，只要 n_valid > T−tcrop 就中招（tcrop=64 时是绝大多数样本）。

本测试用"行号编码"的合成缓存（真数据 = 行号+1，pad = 0）验证：
  1. 取出来的必须是位置 [n_valid−tcrop, n_valid) 的真数据，一格不多一格不少；
  2. 有效快照不足 tcrop 的样本在旧端补零，且补的零被 mask 挡掉；
  3. tcrop=None 路径不受影响；
  4. 复现旧实现，证明它在 n_valid > T−tcrop 时确实取到 pad（回归证据）。

用法：python seq/test_seq_common.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seq_common import F, T, batch_to_tensor  # noqa: E402

DEV = "cpu"


def make_cache(n_valids):
    """合成缓存：第 i 条样本的真数据在位置 0..n_valid-1，值 = 位置+1；pad 区为 0。

    值 = 位置+1 是为了让"0"唯一地表示 pad——真实特征里 0 是合法值（如 has_txn），
    直接比对数值无法区分"取到 pad"和"取到值为 0 的真数据"。
    """
    X = np.zeros((len(n_valids), T, F), dtype=np.float16)
    for i, nv in enumerate(n_valids):
        X[i, :nv, :] = (np.arange(nv) + 1)[:, None]
    return X


def old_impl(Xm, n_valid, rows, tcrop):
    """旧实现（有 bug），仅用于回归对照。"""
    xb = torch.from_numpy(np.asarray(Xm[rows][:, -tcrop:, :]).astype(np.float32))
    xb = xb.permute(0, 2, 1)
    nb = np.minimum(n_valid[rows], tcrop)
    mb = torch.arange(xb.shape[2])[None, :] < torch.from_numpy(nb)[:, None]
    return xb, mb


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    return cond


_DL_X = None      # 模块级：DataLoader 的 worker 进程要能 pickle 到（不能用闭包）


def _dl_build(row):
    """测试用的 build：从模块级数组取一行，切成 (C,T) 与 mask。"""
    x = np.asarray(_DL_X[row], dtype=np.float32)      # (T, C)
    return np.ascontiguousarray(x.T), np.ones(x.shape[0], dtype=bool)


def test_dataloader_matches_manual(ok):
    """DataLoader 路径必须与手写循环**逐批次位级相同**。

    这是换 DataLoader 的前提：不同实验臂用不同的训练循环，
    只要批次内容或顺序差一点，"差异来自输入还是来自代码"就说不清了。
    num_workers 只该改变"谁去读"，不该改变"读到什么"。
    """
    global _DL_X
    from seq_common import SeqDataset, block_perm, make_loader

    n, T, C, batch = 5000, 64, 23, 256
    rng = np.random.default_rng(0)
    _DL_X = rng.normal(size=(n + 5, T, C)).astype(np.float16)
    y = rng.normal(size=n + 5).astype(np.float32)
    tr_rows = np.arange(3, n + 3, dtype=np.int64)      # 故意不从 0 开始，防下标混淆

    print("\n[DataLoader 与手写循环的批次一致性]")
    gen = np.random.default_rng(42)
    perm = block_perm(len(tr_rows), block=1024, gen=gen)

    # —— 手写循环（现状）——
    manual = []
    for i in range(0, len(tr_rows), batch):
        r = tr_rows[perm[i:i + batch]]
        xs = np.stack([_dl_build(int(v))[0] for v in r])          # (B,C,T)
        ms = np.stack([_dl_build(int(v))[1] for v in r])
        manual.append((xs, ms, y[r] * 1000.0))
    print(f"  手写循环：{len(manual)} 个 batch，最后一批 {manual[-1][0].shape[0]} 条")

    ds = SeqDataset(tr_rows, y, _dl_build)
    for workers in (0, 2, 4):
        loader, sampler = make_loader(ds, len(tr_rows), batch, workers)
        sampler.set_perm(perm)
        got = []
        for xb, mb, yb in loader:
            got.append((xb.numpy(), mb.numpy(), yb.numpy()))
        same = len(got) == len(manual) and all(
            np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
            and np.array_equal(a[2], b[2]) for a, b in zip(got, manual))
        ok &= check(f"num_workers={workers} 逐批次位级相同", same,
                    f"{len(got)} batch")
    return ok


def test_loader_no_fd_leak(ok):
    """反复换排列、复用**同一个** DataLoader，fd 不应增长。

    这是 2026-09-15 那次服务器卡死的回归测试。当时的写法是每 epoch 新建一个
    DataLoader（因为 sampler 在构造时绑定），配合 `persistent_workers=True`：
    每次都拉起新一批 worker、旧的不回收，worker 又各自持有 memmap 句柄 →
    fd 单调增长，跑到第 5 折时 `OSError: [Errno 24] Too many open files`。

    后果特别隐蔽：**不是崩溃，是挂住**——主进程 121% CPU、GPU 0%、日志不再增长，
    不看进程状态会以为还在正常跑。所以这条要有测试守着。
    """
    global _DL_X
    from seq_common import SeqDataset, block_perm, make_loader

    print("\n[DataLoader 复用不泄漏 fd]")
    n, T, C, batch, workers, epochs = 4000, 64, 23, 256, 2, 12
    if _DL_X is None or _DL_X.shape[0] < n:
        _DL_X = np.random.default_rng(1).normal(size=(n, T, C)).astype(np.float16)
    y = np.zeros(n, dtype=np.float32)
    tr_rows = np.arange(n, dtype=np.int64)

    def n_fd():
        return len(os.listdir("/proc/self/fd"))       # Linux；本机与服务器都是

    ds = SeqDataset(tr_rows, y, _dl_build)
    loader, sampler = make_loader(ds, n, batch, workers)
    counts = []
    for ep in range(epochs):
        sampler.set_perm(block_perm(n, block=1024, gen=np.random.default_rng(ep)))
        for _ in loader:
            pass
        counts.append(n_fd())
    # 前 2 轮可能还在建立连接，从第 3 轮起应稳定
    base, final = counts[2], counts[-1]
    ok &= check(f"{epochs} 轮复用后 fd 稳定（{counts[2]} → {counts[-1]}）",
                final <= base + 2, f"全程 {counts}")
    return ok



def test_builders_agree(ok):
    """单行构建器（训练/DataLoader）与整批构建器（验证）必须数值一致。

    换 DataLoader 后，训练走 `build_*_one`（逐行）、验证走 `batch_to_tensor*`（整批）。
    两条路径如果对同一行给出不同张量，就等于训练和验证喂了不同分布的数据——
    这种错不会报错，只会让指标莫名偏低。
    """
    print("\n[单行构建器 vs 整批构建器]")
    from seq_common import batch_to_tensor, build_snap_one

    n = 40
    rng = np.random.default_rng(0)
    rows = np.arange(n)

    def agree(one_fn, batch_x, batch_m):
        """one_fn(row) → (x, mask)，逐行与整批结果比对。"""
        for i, r in enumerate(rows):
            o_x, o_m = one_fn(int(r))
            if not (np.array_equal(o_x, batch_x[i].numpy())
                    and np.array_equal(o_m, batch_m[i].numpy())):
                return False, i
        return True, -1

    # —— 粗段路径（T=224, F=15）；tcrop 是刚修过的分支，必须一起比 ——
    X = rng.normal(size=(n, T, F)).astype(np.float16)
    n_valid = rng.integers(1, T + 1, size=n).astype(np.int64)
    for tcrop in (None, 32, 64):
        b_x, b_m = batch_to_tensor(X, n_valid, rows, "cpu", None, tcrop)
        good, bad_i = agree(lambda r: build_snap_one(X, n_valid, r, None, tcrop),
                            b_x, b_m)
        ok &= check(f"粗段单行 == 整批（tcrop={tcrop}）", good,
                    "" if good else f"第 {bad_i} 行不符")

    # ⚠️ tri60 路径的构建器一致性检查已于 2026-09-26 移除（`tri60_baseline.py` 已删）——
    #    它测的是被删掉的代码，没有别的覆盖价值。
    return ok


def main():
    rng = np.random.default_rng(0)
    # 覆盖四类边界：远大于 tcrop / 恰好等于 / 小于 / 接近满格(224)
    n_valids = np.array([189, 194, 60, 64, 32, 1, 224, 223, 20], dtype=np.int64)
    X = make_cache(n_valids)
    rows = np.arange(len(n_valids))
    ok = True

    for tcrop in (64, 32, 16):
        print(f"\n[tcrop={tcrop}]")
        xb, mb = batch_to_tensor(X, n_valids, rows, DEV, tcrop=tcrop)
        assert xb.shape == (len(rows), F, tcrop), xb.shape
        assert mb.shape == (len(rows), tcrop), mb.shape

        # 逐样本对照期望：掩码右对齐，有效位是 [n_valid−k, n_valid)，值 = 位置+1
        for i, nv in enumerate(n_valids):
            k = min(int(nv), tcrop)
            exp_mask = np.zeros(tcrop, dtype=bool)
            exp_mask[tcrop - k:] = True
            got_mask = mb[i].numpy()
            exp_vals = np.zeros((F, tcrop), dtype=np.float32)
            exp_vals[:, tcrop - k:] = (np.arange(nv - k, nv) + 1)[None, :].astype(np.float32)
            if not np.array_equal(got_mask, exp_mask):
                ok &= check(f"n_valid={nv} mask", False,
                            f"期望 {exp_mask.sum()} 个有效位，实得 {got_mask.sum()}")
                continue
            if not np.array_equal(xb[i].numpy(), exp_vals):
                bad = int((xb[i].numpy() != exp_vals).sum())
                ok &= check(f"n_valid={nv} 取值", False, f"{bad} 个元素不符（pad 混入？）")
                continue
            if not np.isfinite(xb[i].numpy()).all():
                ok &= check(f"n_valid={nv} 有限性", False)
                continue
        ok &= check(f"tcrop={tcrop} 全部 {len(n_valids)} 个边界样本（mask+取值+有限性）", True,
                    f"（n_valid 覆盖 {n_valids.min()}..{n_valids.max()}，含 <tcrop / =tcrop / >T-tcrop）")

    # —— tcrop=None 路径不受影响 ——
    print("\n[tcrop=None]")
    xb, mb = batch_to_tensor(X, n_valids, rows, DEV)
    assert xb.shape == (len(rows), F, T)
    exp_mask = (np.arange(T)[None, :] < n_valids[:, None])
    ok &= check("mask 等于 arange(T) < n_valid", np.array_equal(mb.numpy(), exp_mask))
    bad = []
    for i, nv in enumerate(n_valids):
        a = xb[i].numpy()                                    # (F, T)
        # 注意 np.array_equal 不做广播：(F,nv) vs (1,nv) 会被判不等，必须显式扩到 (F,nv)
        exp_head = np.broadcast_to((np.arange(nv) + 1)[None, :].astype(np.float32), (F, nv))
        if not np.array_equal(a[:, :nv], exp_head) or not (a[:, nv:] == 0).all():
            bad.append(int(nv))
    ok &= check("取值正确（真数据 1..n_valid，pad 区为 0）", not bad, f"异常样本 n_valid={bad}" if bad else "")

    # —— 回归证据：旧实现确实把 pad 当数据 ——
    print("\n[旧实现回归对照]")
    tcrop = 64
    xb_o, mb_o = old_impl(X, n_valids, rows, tcrop)
    nv = int(n_valids[0])                       # 189：中位水平的样本
    # 旧实现取数组位置 [T−tcrop, T)，落在 pad 区 [n_valid, T) 里的格子数
    n_pad_in_crop = min(int(X.shape[1]) - nv, tcrop)
    leaked = (xb_o[0, 0, :].numpy() == 0).sum()  # 通道 0 上值为 0 的格子
    print(f"  n_valid={nv}, tcrop={tcrop}：旧实现通道 0 里出现 {leaked}/{tcrop} 个 0 值")
    ok &= check("旧实现确实取到 pad（回归证据）", leaked == n_pad_in_crop,
                f"预期 {n_pad_in_crop} 个 pad 位")
    ok &= check("旧 mask 错误地把 pad 标为有效", bool(mb_o[0].all()))

    ok = test_dataloader_matches_manual(ok)
    ok = test_builders_agree(ok)
    ok = test_loader_no_fd_leak(ok)

    print(f"\n{'全部通过' if ok else '存在失败项'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
