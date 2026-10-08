"""event_batch.py — **扁平事件轴**缓存的消费端：打包 / 分批 / 上卡。多模型族共用。

2026-10-01 从 `models/tcn/evt_baseline.py` **逐字移出**（按 CLAUDE.md 的目录分层：
"会被两个以上模型族 import 的代码，不属于任何一个模型目录"）。
`evt_baseline.py`（三塔TCN·扁平事件轴，已归档）现在从本模块 import，行为不变——
由 `models/tcn/test_evt.py` 的 55 条断言守着。

存储（`seq/event_feats.py::build_flat` 产出，`LAYOUT_VERSION = 4`）：

    Vo (M_o, 6) f16 + offs_o (N+1,) int64     order 事件，**样本内旧→新**
    Vt (M_t, 5) f16 + offs_t (N+1,) int64     txn   事件
    `offs[s]..offs[s+1]` = 第 s 条样本的事件行；**空样本占 1 行全零哨兵** ⇒ 每样本 `nk ≥ 1`
    （读出下标 `nk−1` 恒合法；`nk=0` 会让 `−1` 静默环绕到末尾）。

本模块只管**布局**：变长扁平数组 → `(B, T, C)` 批（批内定长、批间变长，pad 一律在**末尾**）。
模型侧的读出约定：逐样本取 `nk−1`（事件塔）/ `n_valid−1`（粗段塔）——pad 在末尾 + 因果卷积
（TCN）或 pad-mask（注意力）⇒ 补多长都不改变读出。

⚠️ 四条会**静默出错**、由测试守着的约定：

1. **分批排序键必须覆盖所有会被 pad 的维度**：用 `batch_sort_key`（两条流的逐样本最大值）。
   只按一条流排序时，另一条塔的 `T` 会被推到批内 1024 条的最大值（≈p99.9，700~900）
   ⇒ 实测总槽数 **544.8/样本，反超定长版的 928**（2026-09-30 踩到：epoch 从预期 ~105s 变 210s）。
2. **pad 全零、且只在末尾**——模型侧靠因果（TCN）或 mask（注意力）把它排除。TCN 的
   因果卷积对读出天然免疫；**注意力必须显式 mask**，写反不报错（见 `models/tfm/tfm_model.py`）。
3. 批内 `T` 对齐到 `ALIGN` 的倍数：控制形状种类（cuDNN/内核计划按形状缓存；
   `CLAUDE.md #18` 实测变长批的形状多样性本身吃显存）。
4. `pack` 是**纯向量化**的（~3ms/1024 行），不要改成逐样本 Python 循环。

上卡接口（与"粗段 + 两条事件塔"的三塔模型对齐）：
`batch_coarse` / `batch_events` 返回 **(B, C, T)**（卷积习惯）；注意力主干内部自己转置。
"""
from __future__ import annotations

import numpy as np
import torch

import event_feats as EF
from seq_common import T as COARSE_T, F as COARSE_C  # 粗段形状的**单一来源**

ALIGN = 8                 # 批内 T 对齐粒度（控制形状种类）
EVT_LEN_MAX = 999         # 每流事件数的**数据硬顶**（实测两表 max 都恰好 999，见 DATA.md）


def align_up(t):
    return ((t + ALIGN - 1) // ALIGN) * ALIGN


def batch_T(nk_b):
    """批内 T = max(nk) 对齐到 ALIGN（控制 cuDNN 形状种类）。"""
    return max(align_up(int(nk_b.max())), ALIGN)


def batch_sort_key(nk_o, nk_t):
    """分批排序键 = **两条流长度的逐样本最大值**。

    实测（2026-09-30，train 真实分布，batch=1024）padded 槽/样本：
      只按 `nk_o` = **544.8**（踩过的坑：txn 塔被长尾撑爆，epoch 从预期 ~105s 变 210s）
      `nk_o + nk_t` = 396.0    `hypot` = 341.8    **`max` = 308.8**（采用，pad 31.0%）
    参照：定长版 928 槽（pad 76.5%）。`corr(nk_o, nk_t) = 0.845`。
    """
    return np.maximum(nk_o, nk_t)


def make_train_batches(nk, rows, batch, gen, jitter=0.10):
    """按**排序键**切批 + ±jitter 抖动（每轮重排，避免"永远同几条样本同批"）。

    ⚠️ `nk` 必须是 `batch_sort_key(nk_o, nk_t)`（两条流的逐样本最大值），不能只用一条——
    只按 `nk_o` 排序时批内在 txn 长度上完全随机，txn 塔的 `T_t` 会被推到批内 1024 条的
    txn 最大值（≈ p99.9，700~900）而不是应有的 ~90，**总槽数反而超过定长版**。
    """
    if rows.size == 0:
        return []
    k = nk[rows].astype(np.float64)
    key = k * (1.0 + jitter * gen.uniform(-1.0, 1.0, size=k.size))
    perm = rows[np.argsort(key, kind="stable")]
    return [perm[i:i + batch] for i in range(0, perm.size, batch)]


def make_eval_batches(nk, rows, batch):
    """评估/推理**确定性**分批：按排序键排序后切、无抖动——
    同一批行永远给同一批组成，这样"用检查点复算 OOF"才能逐位重现。"""
    if rows.size == 0:
        return []
    perm = rows[np.argsort(nk[rows], kind="stable")]
    return [perm[i:i + batch] for i in range(0, perm.size, batch)]


def pack(V, offs, rows, Tb):
    """把 `rows` 的事件装进 `(B, Tb, C)` f32（**末尾补零**）→ `(X, nk)`。

    `nk` 是每条样本的真实行数（= `offs[r+1] − offs[r]`）。pad 落在每条样本自己的末尾，
    因果卷积下**对读出零影响**；注意力主干侧由 pad-mask 排除。纯内存向量化，1024 行约 3ms。
    """
    B = rows.size
    starts = offs[rows]
    nk = (offs[rows + 1] - starts).astype(np.int64)
    assert Tb >= int(nk.max()), f"Tb={Tb} < 批内最长 {int(nk.max())}"
    total = int(nk.sum())
    excl = np.repeat(np.cumsum(nk) - nk, nk)          # 每行在打包下标里的起点
    within = np.arange(total, dtype=np.int64) - excl
    src = np.repeat(starts, nk) + within
    dst = np.repeat(np.arange(B, dtype=np.int64) * Tb, nk) + within
    X = np.zeros((B, Tb, V.shape[1]), dtype=np.float32)
    X.reshape(-1, V.shape[1])[dst] = V[src]
    return X, nk.astype(np.int64)


# ---------------------------------------------------------------- 上卡

def batch_coarse(Xc, nv, rows, device):
    """粗段 memmap 行切片 → `(x (B,C,T) f32, n_valid (B,))`（两侧都在 device 上）。"""
    a = torch.from_numpy(np.asarray(Xc[rows], dtype=np.float32)).permute(0, 2, 1).to(device)
    nvb = torch.from_numpy(nv[rows].astype(np.int64)).to(device)
    return a, nvb


def batch_events(Vo, offs_o, Vt, offs_t, nk_o, nk_t, rows, device):
    """→ `(xo, nko, xt, nkt)`，两条流各自按**批内最长**补零，形状 (B,C,T)。"""
    ko, kt = nk_o[rows], nk_t[rows]
    Xo, ko = pack(Vo, offs_o, rows, batch_T(ko))
    Xt, kt = pack(Vt, offs_t, rows, batch_T(kt))
    xo = torch.from_numpy(Xo).permute(0, 2, 1).to(device)
    xt = torch.from_numpy(Xt).permute(0, 2, 1).to(device)
    return (xo, torch.from_numpy(ko).to(device),
            xt, torch.from_numpy(kt).to(device))


def load_train():
    """把"粗段 + 两条事件塔"训练所需的全部数组一次读齐：

    → `(Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, y, month, months, sids)`

    - 粗段走 `seq_common.load_train_snap()`（memmap + 与 label 的行序断言）
    - 事件走 `event_feats.build_flat("train", None, to_memory=True)`——
      ⚠️ **现建到内存**（order ~2.4GB + txn ~1.3GB，构建峰值 ~7GB）⇒
      **只能在有卡模式跑**（无卡模式 memory.max=2GiB 会被静默 OOM，见 CLAUDE.md #16）
    - 三份输入的**行序必须一致**（都按 sample_id 升序），这里逐条断言；并验**零截断**：
      扁平存储下 `offs` 跨度 == `n_ev`（空样本除外，它占 1 行哨兵）
    """
    from seq_common import load_train_snap

    Xc, nv, y, marr, months, sids = load_train_snap()
    assert Xc.shape[1:] == (COARSE_T, COARSE_C), f"粗段形状 {Xc.shape}"

    Vo, offs_o, Vt, offs_t, idx = EF.build_flat("train", None, to_memory=True)
    assert Vo.shape[1] == EF.C_O and Vt.shape[1] == EF.C_T, f"通道数 {Vo.shape} / {Vt.shape}"
    N = Xc.shape[0]
    assert offs_o.shape == (N + 1,) and offs_t.shape == (N + 1,), "offs 行数与粗段不一致"
    assert idx.height == N and (idx["sample_id"].to_numpy() == sids).all(), \
        "index 与 label 行序不一致"
    nk_o = np.diff(offs_o); nk_t = np.diff(offs_t)
    assert (nk_o >= 1).all() and (nk_t >= 1).all(), "有空样本占了 0 行（哨兵没写进去）"
    for tag, nk, nev in (("order", nk_o, idx["n_ev_o"].to_numpy()),
                         ("txn", nk_t, idx["n_ev_t"].to_numpy())):
        assert (nk == np.maximum(nev, 1)).all(), f"{tag}: offs 跨度与 n_ev 不一致"
    assert (nv >= 1).all() and (nv <= COARSE_T).all(), f"n_valid 越界：[{nv.min()}, {nv.max()}]"
    return Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, y, marr, months, sids
