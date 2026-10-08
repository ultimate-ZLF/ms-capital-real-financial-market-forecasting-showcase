"""seq_common.py — **序列数据管道**：粗段/细段缓存的加载、折式、Loader。多模型共用。

cnn / tcn / mlp 三个模型族都从这里取数据。本模块**不含任何模型结构**——
CNN 的 `SnapCNN` / `_CausalConv` / `masked_pool` 已拆到 `models/cnn/cnn_model.py`。

快照序列缓存布局（由 `snap_feats.py` 构建）：
- `snap_cache/train_snap_X.dat`  float16 memmap (1,257,637, 224, 17)，行序 = label 按 sample_id 升序
- `snap_cache/train_snap_index.parquet`  sample_id/month/n_valid int16/U_s/mid0
- `snap_cache/test_snap_index.parquet`   sample_id/n_valid/U_s/mid0
（**test 的 `.dat` 已按设计删除**——提交时 `build_split("test", …, to_memory=True)` 现算）

坑（新会话避雷）：
1. **行 0 = 最旧**，行 `n_valid−1` = 最新；pad 在数组**末尾**（`_pos` 按秒数降序编号）。
   做任何时间窗运算都靠这条。
2. pad 区特征全 0，CNN 侧池化要用 mask 剔除（T/16 = 14 严格整除）。
3. 特征已 per-sample tick 单位化——训练时**不做 z-score**。
4. `dt` 通道**截断在 30s**（真实缺口最大 310s）→ 重建时间轴要用 `rel_t`，别用 `dt`。
"""
import os

import numpy as np
import polars as pl
import torch

# `BASE` 按项目约定各自解析一份（43 个 .py 同格式拷贝：MSC_BASE 环境变量优先）。
# ⚠️ 本模块**不**从 `models/tabm` 借 BASE——数据管道不应依赖某个模型目录。
BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)

CACHE = os.path.join(BASE, "snap_cache")
T = 224          # 16×14：4 层 stride-2 卷积后时间维 = 14，masked pooling 严格对齐
F = 15           # 14 个盘口/成交通道 + dt（2026-09-14 并入，见 DESIGN_CNN_V2.md §4.1）
                 # 2026-09-27：曾短暂扩到 17（+txn_cnt/+size_rel）做派生特征实验，
                 # 实验无正面证据后**已回滚**，见 MEMORY 的「粗段特征工程」一条。

FEAT_NAMES = [
    "mid_dev",      # (mid−mid0)/U_s，clip ±256 —— 600s 价格路径（tick 单位）
    "dmid",         # diff(mid)/U_s —— 快照微跳（整数 tick）
    "spread_n",     # (ask1−bid1)/U_s —— 价差 tick 数
    "book_imb",     # (bv1−av1)/(bv1+av1)
    "micro_off",    # (micro−mid)/U_s —— Stoikov 微价偏移
    "d_imb2",       # 二档深度失衡
    "depth_rel",    # (depth−med_depth)/med_depth —— per-sample 相对深度
    "t_svol",       # asinh(sign(txn_px−mid)·txn_vol) —— 有向成交流
    "t_vol_rel",    # txn_vol/(mean_vol+0.5) —— 相对活动强度
    "dpx",          # (txn_px−mid)/U_s —— 吃单方向（无成交=0）
    "has_txn",      # 0/1
    "tvol_dep",     # asinh(txn_vol/(av1+bv1+1)) —— 队列相对消耗
    "eside",        # 空档行标记 0/1
    "rel_t",        # (600−sec)/600 时间位置（0=最旧 1=最新），pad 区 0
    "dt",           # min(距上次观测秒数, 30)/30（每样本首行 = 1.0；pad 区 = 0，被 mask 排除）
                    # 快照间隔不规则（中位 3.0s、q99 12s、最长 310s），差分/波动率类通道
                    # 需要它才能把"跨了多久"与"动了多少"分开；per-sample、零拟合
]


def _load_split(split):
    Xm = np.load(os.path.join(CACHE, f"{split}_snap_X.dat"), mmap_mode="r")
    idx = pl.read_parquet(os.path.join(CACHE, f"{split}_snap_index.parquet"))
    assert Xm.shape[0] == idx.height, "memmap 行数与 index 不一致"
    assert Xm.shape[1:] == (T, F)
    return Xm, idx


def load_train_snap():
    """返回 (Xm f16 memmap(N,T,F), n_valid int16(N), y f32(N), month int32(N), months, sids)。"""
    Xm, idx = _load_split("train")
    lab = pl.read_parquet(os.path.join(BASE, r"train/label.parquet"))
    assert lab.height == idx.height and (lab["sample_id"] == idx["sample_id"]).all(), \
        "label 与 snap index 行序不一致"
    n_valid = idx["n_valid"].to_numpy()
    y = lab["target"].to_numpy().astype(np.float32)
    marr = lab["month"].to_numpy()
    months = sorted(lab["month"].unique().to_list())
    sids = lab["sample_id"].to_numpy()
    return Xm, n_valid, y, marr, months, sids


def batch_to_tensor(Xm, n_valid, rows, device, drop_channels=None, tcrop=None):
    """memmap 行切片 → (B, F, T) float32 tensor + (B, T) bool mask（放 device 上）。

    drop_channels: 要置 0 的通道下标列表（--no-mask-channels 消融）。
    tcrop: 只保留**最新的 tcrop 个有效快照**（逐样本取位置 [n_valid−tcrop, n_valid)）；
           None = 全 224 步。必须是 16 的倍数（主干 4 层 stride-2 → tcrop/16 个池化块）。

    ⚠️ tcrop 的语义（2026-09-15 修正）：**不是数组的最后 tcrop 行**。
    粗段的 pad 在数组最末（最新那一端），真数据在位置 0..n_valid−1
    （snap_feats.py 的 `_pos` 按秒数降序编号，写到 n_valid−1 为止）。所以 `[:, -tcrop:, :]`
    取到的是"真数据的尾巴 + 一整段零"，再配 `nb = min(n_valid, tcrop)` 那个"全有效"mask，
    等于把 pad 当成最新快照喂进池化：

        数组位置   [0 ......... n_valid-1][n_valid ....... 223]
                    └── 真数据（旧→新）──┘└──── 零填充 ────┘
        tcrop=64   -64: 从这里开始 ────────┘└── 全落在 pad 里 ──┘

    中位样本 n_valid=189，只要 n_valid > 224−tcrop 就中招（tcrop=64 时是绝大多数样本）。

    修正后：有效快照不足 tcrop 的样本在**旧端补零**（右对齐），补的零由 mask 挡掉。
    实现上先整行读入再在内存里切——与 tcrop=None 走同一条 memmap 读路径，代价是多读
    (224/tcrop) 倍的行（tcrop 是探查用旋钮，不在主训练路径上）。
    """
    raw = Xm[rows]                                     # (B, T, F) f16 ndarray
    nb = np.asarray(n_valid[rows], dtype=np.int64)
    if tcrop is None:
        xb = raw.astype(np.float32)
        mb = np.arange(raw.shape[1])[None, :] < nb[:, None]
    else:
        assert tcrop % 16 == 0, f"tcrop={tcrop} 必须是 16 的倍数（池化对齐）"
        T = raw.shape[1]
        # 逐样本取最新的 tcrop 个有效快照：位置 nv-tcrop .. nv-1（不足则索引为负 → 补零）
        idx = nb[:, None] - tcrop + np.arange(tcrop)[None, :]      # (B, tcrop)
        mb = idx >= 0                                             # 有效位右对齐
        xb = raw[np.arange(len(rows))[:, None],
                 np.clip(idx, 0, T - 1), :].astype(np.float32)
        # 无效位必须清零：masked_pool 会乘 mask 所以安全，但 encode() 的 pool_tau>0
        # 分支只对分子乘 w、分母才乘 mask（见 seq_common.encode），留垃圾值就会漏进池化
        xb[~mb] = 0.0
    if drop_channels:
        xb[:, :, drop_channels] = 0.0
    xb = torch.from_numpy(xb).permute(0, 2, 1).to(device)  # (B, F, T)
    mb = torch.from_numpy(mb).to(device)
    return xb, mb



def build_snap_one(Xm, n_valid, row, drop_channels=None, tcrop=None):
    """单行版 `batch_to_tensor` → (x (F,T) f32, mask (T,) bool)。

    训练走这个（DataLoader 按行取），验证走 `batch_to_tensor`（整批）。
    **两者必须数值一致**——不一致等于训练/验证喂了不同分布的数据，而且不会报错。
    改其中一个必须同步另一个；`seq/test_seq_common.py` 有逐行等价性测试。
    """
    raw = np.asarray(Xm[row], dtype=np.float32)          # (T,F)
    nb = int(n_valid[row])
    if tcrop is None:
        x = raw
        mb = np.arange(raw.shape[0]) < nb
    else:
        T = raw.shape[0]
        idx = nb - tcrop + np.arange(tcrop)              # 最新的 tcrop 个有效快照
        mb = idx >= 0                                    # 不足则旧端补零、右对齐
        x = raw[np.clip(idx, 0, T - 1)]
        x[~mb] = 0.0
    if drop_channels:
        x[:, drop_channels] = 0.0
    return np.ascontiguousarray(x.T), mb


def avail_mem_bytes():
    """**当前容器内**还能用的内存字节数。

    ⚠️ 不要直接用 `psutil.virtual_memory().available`——AutoDL 容器是 cgroup 限额的
    （2026-09-15 实测 `memory.max` = 72GB），但 `psutil` 和 `free` 报的是**宿主机**的数
    （629GB）。拿宿主机数字做内存守卫**永远不会触发**：容器真被吃满时，
    没有任何预警，cgroup OOM 直接杀进程——而日志里只会显示"空闲内存 586GB"。

    回退链：cgroup v2 → psutil → /proc/meminfo → 无穷大（无从判断时不阻塞）。
    """
    try:                                      # 1) cgroup v2（容器真正的限额）
        with open("/sys/fs/cgroup/memory.max") as f:
            mx = f.read().strip()
        if mx != "max":                       # "max" = 未限额
            with open("/sys/fs/cgroup/memory.current") as f:
                return int(mx) - int(f.read().strip())
    except Exception:
        pass
    try:                                      # 2) psutil（宿主机的数）
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        pass
    try:                                      # 3) /proc/meminfo（无依赖兜底）
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return float("inf")


TARGET_SCALE = 1000.0


class SeqDataset(torch.utils.data.Dataset):
    """按行取样 → (x, mask, y)，喂给 DataLoader 用多进程预取。

    ⚠️ **训练数学必须与单线程手写循环逐位相同**——num_workers 只改变"谁去读"，
    不改变"读到什么"。为此：

    - 批次顺序由外部 sampler 决定（调用方传 `block_perm` 的结果作为 sampler），
      DataLoader 用 `shuffle=False` → 批次内容与顺序和 `tr_rows[perm[i:i+B]]` 完全一致；
    - `drop_last=False` → 最后一个不满的 batch 也照样发（与手写循环一致）；
    - `build(row)` 返回与手写路径同一函数产出的 (x, mask) 数组，不在这里做任何额外变换。

    这与 CLAUDE.md 工程教训 #3（不要擅自调大 batch 改变梯度噪声）不冲突：
    预取不动 batch 组成、顺序、种子。
    """

    def __init__(self, rows, y, build, scale=TARGET_SCALE):
        self.rows = rows          # (n,) int64，训练行号（未打乱）
        self.y = y
        self.build = build        # build(row_idx) -> (x f32 (C,T), mask bool (T,))
        self.scale = scale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        x, m = self.build(r)
        return (torch.from_numpy(x), torch.from_numpy(m),
                torch.tensor(self.y[r] * self.scale, dtype=torch.float32))


class EpochPermSampler(torch.utils.data.Sampler):
    """可变排列 sampler：持有当前 epoch 的排列，由调用方每轮开始时替换。

    为什么需要它：`DataLoader` 的 `sampler` 在**构造时绑定**，而我们的批次顺序每
    epoch 都要换（block_perm）。直觉做法是每 epoch 新建一个 DataLoader——但配合
    `persistent_workers=True`，每次新建都会拉起新一批 worker 进程，旧的不回收，
    **每个 worker 又各自持有 memmap 句柄**，于是 fd 单调增长：

        2026-09-15 实测：跑到第 5 折时 OSError: [Errno 24] Too many open files，
        主进程 121% CPU 但 GPU 0%，作业静默卡死（不是崩，是挂住）。

    改成"建一次 + 换排列"后 worker 常驻复用，fd 恒定。
    `models/cnn/test_seq_common.py::test_loader_no_fd_leak` 守着这条。
    """

    def __init__(self, n):
        self.n = n
        self.perm = np.arange(n, dtype=np.int64)

    def set_perm(self, perm):
        perm = np.asarray(perm)
        assert len(perm) == self.n, f"排列长度 {len(perm)} != {self.n}"
        self.perm = perm

    def __iter__(self):
        return iter(self.perm.tolist())

    def __len__(self):
        return self.n


def make_loader(ds, n, batch, workers):
    """建**一个** DataLoader（worker 常驻复用）→ (loader, sampler)。

    每轮开始时调 `sampler.set_perm(block_perm(...))` 换批次顺序。
    批次内容与顺序和手写循环逐位相同（`test_dataloader_matches_manual` 守着）。
    workers=0 等价于单进程（调试用）。
    """
    from torch.utils.data import DataLoader
    sampler = EpochPermSampler(n)
    kw = dict(batch_size=batch, shuffle=False, drop_last=False,
              sampler=sampler, num_workers=workers)
    if workers > 0:
        kw.update(pin_memory=True, persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, **kw), sampler


FOLD_STEP = 6      # 月交错折：折 k 的验证月 = months[k::FOLD_STEP]


def make_folds(months, n_folds=FOLD_STEP):
    """月交错折划分：折 k 验证 months[k::FOLD_STEP]，训练 = 其余月份。

    n_folds 个折的验证月互不重叠；取满 FOLD_STEP 个即覆盖全部月份。
    为什么不用"按样本随机分折"：样本虽是独立模拟片段（DATA.md 独立性检验），
    但**月份是模拟器参数**（target std 逐月 2.58e-3~5.0e-3），随机分折会让同月样本
    同时进训练和验证 → regime 泄漏、CV 偏乐观。见 CLAUDE.md #8。
    """
    assert 1 <= n_folds <= FOLD_STEP, \
        f"n_folds 必须在 1..{FOLD_STEP}（超过 FOLD_STEP 会让验证月重复）"
    return [months[k::FOLD_STEP] for k in range(n_folds)]


def block_perm(n, block=4096, gen=None):
    """块洗牌：4096 行一块，块间 shuffle + 块内 shuffle → memmap 半顺序 IO。"""
    rng = np.random.default_rng() if gen is None else gen
    nb = (n + block - 1) // block
    block_order = rng.permutation(nb)
    perm = []
    for b in block_order:
        s = b * block
        chunk = np.arange(s, min(s + block, n))
        perm.append(rng.permutation(chunk))
    return np.concatenate(perm)

