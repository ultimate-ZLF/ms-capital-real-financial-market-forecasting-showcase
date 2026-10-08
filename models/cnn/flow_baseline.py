"""flow_baseline.py — **新主干**（空洞因果卷积 + 可学习池化权重）在各输入源上的 CV 评估。

**独立脚本**（不共用脚本、不共用 OOF 目录）。原本的理由之一是"tri60 是已记录结果 0.07511 的
参照实现，改它等于动历史证据的生成器"——**该参照已于 2026-09-26 随 tri60 代码移除**。
  2. "一个 flag 语义只在一个臂里成立" 正是 `cab2d1b` 清空那批融合实验的病根。
见 HYPOTHESES C-01 的记录纪律与 CLAUDE.md 工程教训 #7。

设计的公共部分（依据 HYPOTHESES C-01 / C-10 / C-11 / C-12 / DESIGN_CNN_V2 §4.2）：

  主干   因果卷积 3 层：k=5,s=2 → T/2；k=5,s=2 → T/4；k=3,s=1,**dilation 按源** → T/4
  读出   **可学习的 per-block 权重** a_b = softmax(θ)（T/4 个参数）+ 可学习尺度 s：
             h = s · Σ_b a_b·h_b
         取代原来固定的 τ 指数先验。依据：
         - C-10：让网络自己学权重，表达力**严格包含**任何固定位置权重；
         - C-11：60 秒窗口上新鲜度加权接近空操作（WALKTHROUGH4 §3 的截短表），
           flow 侧没有 τ 可丢，所以"换成可学习"不是取舍、是纯粹的解放；
         - C-12：粗段侧 τ 的效应量是 0.05 量级（等权 0.062 → τ=64 0.111），
           **只有在这里才测得出来**，所以粗段是本脚本的主战场。
         θ 零初始化 → 训练起点就是等权块均值，之后由数据决定给谁多少权重。
         ⚠️ 尺度 s 是**故意保留的自由度**：C-10 的注意项——若把 legacy 的价值来源误判成
         "权重分配"，只给权重自由度会复现不出来。
   head   Linear(256→256) + GELU + Dropout(head_drop) + Linear(256→1)

各源规格（`SRC`）：

| source | 输入 | 缓存 | 时间轴 | 块数 | 第三层 dilation | 读出对 pad |
|---|---|---|---|---|---|---|
| `flow` | 16 通道 | flow（现建到内存）| 60 | 15 | 1 | 无 pad |
| `coarse` | 15 通道 | snap_cache `.dat` + `n_valid` | **224** | **56** | **4** | `--block-mask` |

- **细段三臂无 pad、无 mask**：60 步不凑 16 的倍数（pad 是旧主干的对齐垫片），60 格全有效；
- **粗段有 pad**（在**最新端**：真数据在 0..n_valid−1）。第三层给 `dilation=4` 是因为照搬
  flow 的 d=1 会让 RF 只有 21 步 ≈ 63 秒（占 672 秒窗口 9%，比旧主干的 17% 还差）；
  d=4 时 RF = 45 步 ≈ 135 秒 = 20%。
- **`--block-mask hard|none` 就是 C-12 的对照**：
  - `hard`：块内只要有任一有效位就算有效，无效块的 softmax 概率**恒为 0**（θ 上填 −inf）；
  - `none`：不加掩码，让可学习权重自己决定给 pad 块多少——**学出来的 pad 权重质量就是
    C-12 的读数**（`legacy` 手写的是约 40%）。脚本每折会打印这个量。
  细段两臂两者恒等（全有效），所以这个开关不影响它们。

训练协议与 E1 各臂**逐字一致**（lr 1e-3 / wd 3e-4 / batch 1024 / drop 0.2 / head_drop 0.3 /
patience 16 / epochs 300 / seed 42 / MSE(target×1000) / val cos 早停 / 月交错 6 折）——
折式没变，所以**逐折配对差仍然可比**：变的是模型，不是数据划分。

细段输入默认**运行期现建到内存**（不落盘，见 check_memory_build.py 的 docstring）；
粗段直接读 `snap_cache/train_snap_X.dat`（7.9GB memmap，那份还在）。

本脚本**只做 CV/OOF**，不生成提交（等某条臂被证明值得提交再单独写，避免过早铺代码）。

用法：
    python flow_baseline.py --source flow --folds 1 --smoke-fold    # 管线冒烟
    python flow_baseline.py --source flow --folds 6                 # 细段臂
    python flow_baseline.py --source coarse --folds 6 --block-mask hard
    python flow_baseline.py --source coarse --folds 6 --block-mask none   # C-12 对照
"""
import argparse
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "seq"))   # 序列数据管道（顶层 seq/）

from seq_common import (SeqDataset, block_perm, make_folds, make_loader)  # noqa: E402

# 跨目录复用（models/tabm）——目录结构见 README
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tabm"))

from tabm_common import BASE, cos_score  # noqa: E402

SPAN = 4                  # 每个输出位置覆盖的输入步数（两层 stride-2）
FINE_T = 60               # 细段窗口：60 秒 × 1 秒
COARSE_T = 224            # 粗段窗口：224 个快照 × ~3 秒 ≈ 672 秒
FLOW_CACHE = os.path.join(BASE, "flow_cache")
SNAP_CACHE = os.path.join(BASE, "snap_cache")
TARGET_SCALE = 1000.0

# 每个 source 的规格。`kind` 决定输入从哪来、要不要 n_valid。
SRC = {
    "flow":   dict(c=16, t=FINE_T,   blocks=FINE_T // SPAN,     l3_dil=1, kind="fine"),
    "coarse": dict(c=15, t=COARSE_T, blocks=COARSE_T // SPAN,   l3_dil=4, kind="coarse"),
}
N_FEATS = {k: v["c"] for k, v in SRC.items()}


# ---------------------------------------------------------------- 输入

def _source_list(source):
    """细段的源列表。

    ⚠️ **`mkt` / `both` 两条臂已于 2026-09-26 随 tri60 代码一并移除**（用户指令）——
    它们读的是 `tri60_feats` 构建的 market 派生通道，而那个文件已删。
    **代价（已知并接受）**：`mkt`（0.07322）与 `both`（0.10626）两臂**永久不可复现**，
    而 `both` 正是 `HYPOTHESES C-01`（"丢历史导致短窗模型输"不成立，+0.03115）的对照臂。
    已记录的数字仍在 `WALKTHROUGH9` / `MEMORY`，只是生成器没了。
    """
    return ["flow"]


def _build_to_memory(split, which):
    """现建到内存、不落盘（进程结束即消失）。"""
    assert which == "flow", f"只剩 `flow` 一条细段源（mkt/both 已随 tri60 移除）：{which}"
    import flow_feats
    return flow_feats.build_split(split, None, False, to_memory=True)


def _load_from_disk(split, which):
    assert which == "flow", f"只剩 `flow` 一条细段源（mkt/both 已随 tri60 移除）：{which}"
    p = os.path.join(FLOW_CACHE, f"{split}_flow_X.dat")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"缺 {p}。用内存构建（去掉 --from-disk），或先跑构建脚本重建")
    return np.load(p, mmap_mode="r")


def _check_index_rows(source, split, sids):
    """磁盘路径专用：核对缓存 index 的行序与基准一致。

    内存路径**不需要**这一步——`*_feats.build_split` 内部就用 label（或 submission.csv）
    按 sample_id 升序取行，行序由构建过程本身保证；磁盘路径则要防"缓存是另一份数据建的"。
    """
    for _w in _source_list(source):
        p = os.path.join(FLOW_CACHE, f"{split}_flow_index.parquet")
        if not os.path.exists(p):
            print(f"[warn] 缺 {p}，跳过行序核对", flush=True)
            continue
        idx = pl.read_parquet(p).sort("sample_id")
        assert idx.height == len(sids) and (idx["sample_id"].to_numpy() == sids).all(), \
            f"{p} 的 sample_id 与基准（label）行序不一致——缓存可能不是这份数据建的"
        print(f"[check] {os.path.basename(p)} 行序与基准一致（{idx.height} 行）", flush=True)


def load_source(split, source, from_disk=False):
    """细段：返回 (数组列表, 行数)。"""
    arrs = [(_load_from_disk if from_disk else _build_to_memory)(split, w)
            for w in _source_list(source)]
    for a in arrs:
        assert a.shape[1] == SRC[source]["t"], f"时间轴 {a.shape[1]} 与 {source} 不符（{a.shape}）"
    ns = {a.shape[0] for a in arrs}
    assert len(ns) == 1, f"各源行数不一致：{ns}"
    c = sum(a.shape[2] for a in arrs)
    assert c == N_FEATS[source], f"通道数 {c} != {N_FEATS[source]}"
    return arrs, ns.pop()


def load_train(source, from_disk=False):
    """返回 (数组列表, n_valid 或 None, y f32, month, months, sids)。

    行序基准：所有构建路径都按 **sample_id 升序**写行，所以这里用 label 的
    sample_id 升序作基准核对（粗段走 seq_common.load_train_snap，它自带同样的断言）。
    """
    if SRC[source]["kind"] == "coarse":
        from seq_common import load_train_snap
        Xm, nv, y, marr, months, sids = load_train_snap()
        return [Xm], nv, y, marr, months, sids

    arrs, n = load_source("train", source, from_disk)
    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).sort("sample_id")
    assert lab.height == n, f"label {lab.height} 行 != 输入 {n} 行"
    sids = lab["sample_id"].to_numpy()
    if from_disk:
        _check_index_rows(source, "train", sids)
    return (arrs, None, lab["target"].to_numpy().astype(np.float32),
            lab["month"].to_numpy(), sorted(lab["month"].unique().to_list()), sids)


# ---------------------------------------------------------------- 构建器
#
# 单行构建器（训练，DataLoader 按行取）与整批构建器（验证，一次性）**必须数值一致**——
# 不一致等于训练和验证喂了不同分布的数据，而且**不会报错**。
# 改任何一个都要同步另一个；`test_flow_baseline.py::test_builders_agree` 守着这条。

def build_input_one(arrays, nv, row):
    """单行 → (x (C,T) f32, mask (T,) bool)。

    nv=None（细段）→ mask 全 True（60 格全有效，主干不消费它）。
    nv 给定（粗段）→ mask = 位置 < n_valid；pad 在**最新端**，所以有效位在开头。
    """
    x = np.concatenate([np.asarray(a[row], dtype=np.float32) for a in arrays], axis=1)
    if nv is None:
        m = np.ones(x.shape[0], dtype=bool)
    else:
        m = np.arange(x.shape[0]) < int(nv[row])
    return np.ascontiguousarray(x.T), m


def batch_to_tensor(arrays, nv, rows, device):
    """整批版（验证集用）→ (x (B,C,T), mask (B,T) bool)。"""
    x = np.concatenate([np.asarray(a[rows], dtype=np.float32) for a in arrays], axis=2)
    xb = torch.from_numpy(x).permute(0, 2, 1).to(device)
    if nv is None:
        mb = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=device)
    else:
        nb = np.asarray(nv[rows], dtype=np.int64)
        mb = torch.from_numpy(np.arange(x.shape[1])[None, :] < nb[:, None]).to(device)
    return xb, mb


class SourceBuilder:
    """DataLoader worker 用的单行构建器。写成类而不是 lambda——lambda 不可 pickle，多进程下会报错。"""

    def __init__(self, arrays, nv=None):
        self.arrays = arrays
        self.nv = nv

    def __call__(self, row):
        return build_input_one(self.arrays, self.nv, row)


# ---------------------------------------------------------------- 模型

class CausalConv(torch.nn.Module):
    """因果 Conv1d：左侧 pad (k−1)·dilation，输出位置 i 只依赖输入 ≤ i。

    与 `seq_common._CausalConv` 同一手法（torch 的 padding 只支持对称，用 F.pad 实现），
    多支持 dilation——本脚本要用空洞卷积在**不额外降采样**的前提下把 RF 拉长。
    """

    def __init__(self, cin, cout, k, stride=1, dilation=1):
        super().__init__()
        self.left = (k - 1) * dilation
        self.conv = torch.nn.Conv1d(cin, cout, k, stride=stride, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left, 0)))


class LearnPoolCNN(torch.nn.Module):
    """新主干：因果卷积 + **可学习块权重读出** + MLP head（无 τ、无 mask 池化）。

    与 SnapCNN 的差别只有两处（都是刻意的）：
      1. **没有 pool_tau**：读出权重 `block_w` 是可学习参数（C-10），不是固定指数先验；
      2. **没有 masked_pool**：读出是块权重加权和；粗段的 pad 由 `block_mask` 决定怎么处理
         （`hard`：无效块概率置 0；`none`：交给网络自己学 —— 即 C-12 的对照）。
    读出后为什么还留一个可学习尺度 `scale`：见模块 docstring（C-10 的注意项）。
    """

    def __init__(self, n_feats, n_blocks=15, l3_dilation=1, block_mask="hard",
                 drop=0.2, head_drop=0.3, out=1, head_dim=256):
        super().__init__()
        assert block_mask in ("hard", "none"), block_mask
        self.block_mask = block_mask
        self.conv = torch.nn.Sequential(
            CausalConv(n_feats, 64, 5, stride=2), torch.nn.BatchNorm1d(64),
            torch.nn.GELU(), torch.nn.Dropout(drop),
            CausalConv(64, 128, 5, stride=2), torch.nn.BatchNorm1d(128),
            torch.nn.GELU(), torch.nn.Dropout(drop),
            CausalConv(128, 256, 3, stride=1, dilation=l3_dilation),
            torch.nn.BatchNorm1d(256), torch.nn.GELU(), torch.nn.Dropout(drop),
        )
        self.block_w = torch.nn.Parameter(torch.zeros(n_blocks))   # softmax → 块权重
        self.scale = torch.nn.Parameter(torch.ones(1))             # 尺度自由度（C-10）
        self.head = torch.nn.Sequential(
            torch.nn.Linear(256, head_dim), torch.nn.GELU(),
            torch.nn.Dropout(head_drop), torch.nn.Linear(head_dim, out),
        )

    @staticmethod
    def block_valid(mask):
        """(B,T) 时间掩码 → (B,nb) 块有效标志：块内**有任一**有效位即算有效。

        与 seq_common 的 `max_pool1d(mask,16,16)` 同一约定（块粒度对齐主干的下采样倍数）。
        """
        return F.max_pool1d(mask.float().unsqueeze(1), SPAN, SPAN).squeeze(1) > 0

    def block_weights(self, mask=None):
        """当前块权重（numpy，供诊断）。

        mask 给定时返回逐样本权重 (B,nb)（`hard` 下无效块为 0）；否则返回 (nb,)。
        """
        with torch.no_grad():
            if self.block_mask == "hard" and mask is not None:
                theta = self.block_w[None, :].masked_fill(~self.block_valid(mask), float("-inf"))
                return torch.softmax(theta, dim=-1).cpu().numpy()
            return torch.softmax(self.block_w, dim=0).cpu().numpy()

    def encode(self, x, mask=None):
        """x:(B,C,T) f32, mask:(B,T) bool 或 None → (B,256)：块权重加权和 × 尺度。"""
        h = self.conv(x)                                       # (B,256,nb)
        if self.block_mask == "hard" and mask is not None:
            theta = self.block_w[None, :].masked_fill(~self.block_valid(mask), float("-inf"))
            a = torch.softmax(theta, dim=-1)                   # (B,nb)，无效块恒为 0
            h = (h * a[:, None, :]).sum(-1)
        else:                                                  # none：权重与样本无关
            a = torch.softmax(self.block_w, dim=0)             # (nb,)
            h = (h * a[None, None, :]).sum(-1)
        return h * self.scale

    def forward(self, x, mask=None):
        return self.head(self.encode(x, mask))


# ---------------------------------------------------------------- 训练

def pad_weight_mass(a, mask):
    """C-12 的读数：块权重里落到 **pad 块** 的总质量。

    a:    (nb,) 与样本无关的权重（`none` 口径才有意义）
    mask: (B,T) 时间掩码
    对每个块算"它在多少比例的样本里是无效的"，再按权重加权求和。
    """
    if a.ndim != 1:
        return float("nan")
    invalid = ~LearnPoolCNN.block_valid(mask).cpu().numpy()    # (B,nb)
    return float((a * invalid.mean(axis=0)).sum())


def train_fold(arrays, nv, y, sids, tr_rows, va_rows, args, device):
    spec = SRC[args.source]
    xv, mv = batch_to_tensor(arrays, nv, va_rows, device)
    yv = (y[va_rows] * TARGET_SCALE).astype(np.float32)

    torch.manual_seed(args.seed)
    model = LearnPoolCNN(n_feats=spec["c"], n_blocks=spec["blocks"],
                         l3_dilation=spec["l3_dil"], block_mask=args.block_mask,
                         drop=args.drop, head_drop=args.head_drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = np.random.default_rng(args.seed)

    n_tr = len(tr_rows)
    # DataLoader 多进程预取：读盘 + f16→f32 + 通道拼接挪到 worker，与 GPU 计算重叠。
    # 批次顺序仍由 block_perm 决定、shuffle=False → 与手写循环逐位相同（有测试守着）。
    ds = SeqDataset(tr_rows, y, SourceBuilder(arrays, nv))
    loader, sampler = make_loader(ds, n_tr, args.batch, args.workers)

    best_cos, best_epoch, best_va_pred, stall = -1.0, -1, None, 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        sampler.set_perm(block_perm(n_tr, block=4096, gen=gen))
        for xb, mb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = F.mse_loss(model(xb, mb).squeeze(-1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(xv, mv).squeeze(-1).cpu().numpy()
        cos = cos_score(pred_va, yv)
        if cos > best_cos:
            best_cos, best_epoch, best_va_pred, stall = cos, ep, pred_va, 0
        else:
            stall += 1
            if stall >= args.patience:
                print(f"    ep={ep:3d} loss={loss.item():.4f} val_cos={cos:.5f} "
                      f"t={time.time()-t0:5.1f}s  早停", flush=True)
                break
        if ep % 5 == 0:
            print(f"    ep={ep:3d} loss={loss.item():.4f} val_cos={cos:.5f} "
                  f"t={time.time()-t0:5.1f}s", flush=True)

    a = model.block_weights(mv)
    return {"cos": best_cos, "epoch": best_epoch, "va_pred": best_va_pred,
            "va_sids": sids[va_rows], "block_w": model.block_weights(),
            "pad_mass": pad_weight_mass(a, mv) if args.block_mask == "none" else 0.0}


def _fmt_block_w(a):
    """块权重摘要。短向量逐个打印；长向量按四段（旧→新）给质量占比 + 最大块。"""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1:
        return "(逐样本权重，见 pad_mass)"
    if a.size <= 20:
        return " ".join(f"{v:.3f}" for v in a) + f"   argmax=#{int(a.argmax())}"
    q = np.array_split(a, 4)
    return ("四段质量(旧→新)=" + " / ".join(f"{x.sum():.3f}" for x in q) +
            f"   argmax=#{int(a.argmax())}({a.max():.3f})")


def main():
    p = argparse.ArgumentParser(description="新主干（空洞卷积 + 可学习池化）的 CV 评估")
    p.add_argument("--source", default="flow", choices=list(SRC),
                   help="flow=60s 细段；coarse=224 步粗段（snap_cache）。mkt/both 已于 2026-09-26 随 tri60 移除")
    p.add_argument("--block-mask", default="hard", choices=["hard", "none"],
                   help="粗段读出对 pad 块的处理：hard=无效块概率置 0；"
                        "none=交给网络自己学（C-12 的对照）。细段两侧恒等")
    p.add_argument("--folds", type=int, default=6,
                   help="月交错折数（1..6）：折 k 验证 months[k::6]")
    p.add_argument("--start-fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=16)
    p.add_argument("--batch", type=int, default=1024,
                   help="与 E1 各臂/粗段一致，**不要改**（见 CLAUDE.md 工程教训 #3）")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader 预取进程数。不改训练数学（批次内容/顺序逐位相同，有测试）")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--head-drop", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--from-disk", action="store_true",
                   help="细段读 *.dat 缓存（2026-09-16 起该缓存已删——此路只剩报错）；"
                        "⚠️ 粗段**静默忽略**本开关（load_train 在传参前就 return 了），"
                        "日志里的 input=disk 对 --source coarse 是假的")
    p.add_argument("--oof-dir", default=None,
                   help="细段默认 flow_cache/oof_parts_{source}；"
                        "粗段默认 snap_cache/oof_parts_learnpool_{block_mask}（两个掩码口径各一目录）")
    p.add_argument("--smoke-fold", action="store_true",
                   help="冒烟：每折只跑 1 epoch、不汇总（配 --folds 1 用，验管线）")
    args = p.parse_args()

    spec = SRC[args.source]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.oof_dir:
        oof_dir = args.oof_dir if os.path.isabs(args.oof_dir) \
            else os.path.join(BASE, args.oof_dir)
    elif spec["kind"] == "coarse":
        oof_dir = os.path.join(SNAP_CACHE, f"oof_parts_learnpool_{args.block_mask}")
    else:
        oof_dir = os.path.join(FLOW_CACHE, f"oof_parts_{args.source}")
    os.makedirs(oof_dir, exist_ok=True)

    print(f"device={device} source={args.source}(C={spec['c']},T={spec['t']}) "
          f"blocks={spec['blocks']} l3_dil={spec['l3_dil']} block_mask={args.block_mask} "
          f"batch={args.batch} lr={args.lr} wd={args.wd} drop={args.drop}/{args.head_drop} "
          f"seed={args.seed} workers={args.workers} "
          f"input={'disk' if args.from_disk or spec['kind'] == 'coarse' else 'memory'}")
    print(f"读出 = 可学习块权重 + 可学习尺度（**无 τ**）"
          + ("；细段无 pad 无 mask" if spec["kind"] == "fine" else "；粗段 pad 在最新端"))
    print(f"OOF → {oof_dir}")

    t_load = time.time()
    arrays, nv, y, marr, months, sids = load_train(args.source, args.from_disk)
    print(f"loaded: {[a.shape for a in arrays]} "
          f"{'f16' if isinstance(arrays[0], np.memmap) else 'memory'}"
          f"{'' if nv is None else f' + n_valid(中位 {int(np.median(nv))})'}"
          f"（{time.time()-t_load:.1f}s）", flush=True)

    folds = make_folds(months, args.folds)
    results = []
    for fi, va_months in enumerate(folds):
        if fi < args.start_fold:
            continue
        part_path = os.path.join(oof_dir, f"f{fi}.parquet")
        va_mask = np.isin(marr, va_months)
        tr_rows = np.where(~va_mask)[0]
        va_rows = np.where(va_mask)[0]
        print(f"[fold {fi + 1}/{len(folds)}] 验证月={va_months} "
              f"训练={len(tr_rows)} 验证={len(va_rows)}", flush=True)
        t0 = time.time()
        epochs = 1 if args.smoke_fold else args.epochs
        r = train_fold(arrays, nv, y, sids, tr_rows, va_rows,
                       argparse.Namespace(**{**vars(args), "epochs": epochs}), device)
        r["fold_secs"] = time.time() - t0
        results.append(r)
        pl.DataFrame({"sample_id": r["va_sids"], "month": marr[va_rows],
                      "pred": r["va_pred"].astype(np.float64)}).write_parquet(part_path)
        print(f"  → cos={r['cos']:.5f} best_epoch={r['epoch']} "
              f"耗时 {r['fold_secs']:.0f}s（已落盘 {part_path}）", flush=True)
        print(f"    块权重: {_fmt_block_w(r['block_w'])}", flush=True)
        if args.block_mask == "none" and spec["kind"] == "coarse":
            print(f"    pad 块权重质量 = {r['pad_mass']:.3f}（C-12 读数：legacy 手写约 0.40，"
                  f"mask 口径 = 0）", flush=True)

    if args.smoke_fold or len(results) <= 0:
        print("\n（冒烟模式：不汇总）")
        return

    cos_arr = np.array([r["cos"] for r in results])
    ep_arr = np.array([r["epoch"] for r in results])
    print(f"\n### {args.source}（block_mask={args.block_mask}）逐折 cos（{len(results)} 折，月交错）###")
    print(f"mean={cos_arr.mean():.5f} std={cos_arr.std():.5f} "
          f"min={cos_arr.min():.5f} neg={int((cos_arr < 0).sum())}")
    print(f"best_epoch: median={int(np.median(ep_arr))} 各折={ep_arr.tolist()}")
    if len(results) > 1:
        print(f"块权重均值: {_fmt_block_w(np.mean([r['block_w'] for r in results], axis=0))}")
    if args.block_mask == "none" and spec["kind"] == "coarse":
        pm = np.array([r["pad_mass"] for r in results])
        print(f"pad 块权重质量: mean={pm.mean():.3f} 各折={np.round(pm, 3).tolist()}")
    if spec["kind"] == "coarse":
        print("对照（同折式、逐折配对）：粗段 legacy + τ 池化 0.10336；"
              "（flow 0.10088* 是 60s 口径，仅作参照）")
    else:
        print("对照（同折式、可逐折配对）：粗段 legacy 0.10336 / "
              "单因子 t_imb_2s 0.0733")

    if len(results) > 1:
        parts = [pl.read_parquet(os.path.join(oof_dir, f))
                 for f in sorted(os.listdir(oof_dir)) if f.endswith(".parquet")]
        oof_path = os.path.join(oof_dir, "oof_all.parquet")
        pl.concat(parts).write_parquet(oof_path)
        print(f"\nOOF 已存 {oof_path}（{sum(p.height for p in parts)} 行）")


if __name__ == "__main__":
    main()
