"""mlp_baseline.py — **双塔MLP（通道→时间 顺序 MLP 主干）**：粗段 672 秒 + 细段 60 秒联合训练。

> 命名（2026-09-21）：本模型是"双塔×"家族第二个成员（第一个是 `models/cnn/twotower_baseline.py`
> 的**双塔CNN**）。本目录自包含：代码与文档都在 `models/mlp/` 下，**设计/超参/闸门/结果见
> 本目录 `README.md`**，共享文档只留一行指针。
> 本模型**最初**按 MLP-Mixer 的"交替块 + 残差"实现，2026-09-21 当天按用户要求简化为
> **通道 MLP → 时间 MLP 一趟顺序**（算量降到 1/5.7）。演进对比见本目录 README「架构演进」。

**独立脚本**——不 import `twotower_baseline.py`（每个训练脚本是一支独立实验谱系，改一个不应
影响另一个已记录的结果，记录纪律见 CLAUDE.md #7）。只复用**共享基础设施**：
`seq_common`（loader / 折式 / 块洗牌）、`flow_feats` / `snap_feats`（缓存构建）、
`tabm_common`（BASE / cos_score）。

设计（唯一变量是**主干**——输入、读出维度、head、训练协议都与双塔CNN 逐字相同，否则 CV 不可比）：

```
塔 A（粗段） snap_cache train_snap_X.dat (N,224,15) f16 + n_valid
             → **取最新 K 行**（`--crop K` 必填；右对齐、不足者旧端补零）→ (N,K,15)
             → LN(C) → channel_mlp：逐位置 Linear(15→256) + GELU + Drop（跨通道混合）
             → + 位置嵌入 (K,256)   ← 正好在时间 MLP 之前
             → LN(dim) → time_mlp：Linear(K→128) + GELU + Drop + Linear(128→K) + Drop
                          （跨时间混合，权重跨通道共享）
             → 逐步读出 softmax(read_w)(K) + 可学习尺度 → h_A (B,256)
塔 B（细段） flow_cache 现建到内存 (N,60,16) f16  —— 同构，T=60
合           concat[h_A ; h_B] (512) → Linear(512→256) + GELU + Drop(head_drop) → Linear(256→1)
```

无残差、无多块重复（就是一趟）；**两处 pre-norm LayerNorm**——参数 28.7 万，
算量 ≈19.7M MACs/样本（双塔CNN 0.42M / 10.0M 的 0.68× / 2.0×）。

> **LN 是 2026-09-21 补上的，原因是一条实测**：无 LN 版在 fold 0 上 **val_cos 于 epoch 0 见顶
> （0.11723）后单调变差**（ep10 → 0.07667、ep15 → 0.06434），训练 loss 出现 10.67 的尖峰
> （正常量级 ~4）——是**训练不稳定**而不是过拟合。lr 1e-3 是给带 BatchNorm 的 CNN 调的，
> 而这版主干当时既无 BN 也无 LN、还无残差。**加 LN 只动架构、不动训练协议**
> （lr/batch/drop/折式全不变），所以与双塔CNN 的 CV 仍同口径可比。

## ⚠️ 三条必须在动手前知道的事

1. **本主干是"非因果"的**（窗口内双向混合），且 **无 BatchNorm、无卷积**。
   因此 **CLAUDE.md #10 的 τ/legacy 三条耦合条件不成立**，
   **"pad 参与读出 +0.024"（C-12）也不能照搬**——CNN 那条的机理是因果卷积把 pad 压成
   近常数偏置；这里 pad 区是逐位置独立的位置嵌入 + 独立读出权重，自由度大得多。
   → **已实测裁决（2026-09-21）**：`hard`（屏蔽 pad）只比 `none` 高 0.0029（噪声带内），
   而 `--crop 128` 高 **+0.0113** → **掩码逻辑已删除**，pad 的处理完全交给 `--crop`。
   原理/证据/判据见 `PAD_AND_CROP.md`；假设登记见 HYPOTHESES C-15。
2. **验证/推理必须分块**：整折 21.2 万行一次性前向时，中间张量是
   `212139 × 224 × 256 × 4B ≈ 48 GB` → **必 OOM**（不是"慢"）。故有 `--eval-chunk`（默认 8192）。
   因为无 BN、逐样本，分块与一次性**数值一致**——但**不是逐位相同**：
   实测 `chunk=2` vs 一次性差 **3e-08**（float32 GEMM 分块不同，预测 std 1.26e-02 的相对 2.4e-6）。
   **同一 chunk 重放是逐位相同的**——检查点复算 OOF 依赖的正是这一条（`test 10` 钉住）。
3. **检查点机制是全项目首次引入**（四条纪律见 CLAUDE.md #12）。它与
   `twotower_baseline.py:37` 的"从头训练、不加载任何已有模型的权重"**不冲突**：
   检查点是**本脚本自己逐折产出的**，只用于「同一次实验续跑」与「提交侧推理」，
   不跨谱系 / 模型 / 种子流通。

训练协议与双塔CNN / E1 各臂**逐字一致**（lr 1e-3 / wd 3e-4 / batch 1024 / drop 0.2 /
head_drop 0.3 / patience 16 / epochs 300 / seed 42 / MSE(target×1000) / val cos 早停 / 月交错 6 折）——
折式没变，所以**逐折配对差仍然可比**。

**预注册判据（跑之前定死）**：

| 结果（6 折均值） | 含义 | 后续 |
|---|---|---|
| **≥ 0.12280** | 主干替换成功（追平/超过双塔CNN） | 继续挖（位置嵌入消融 / 通道 MLP 加宽 / 加塔）|
| 0.115 ~ 0.1228 | 单塔之上但不敌双塔CNN | 保留作**集成多样性**候选（A-11：CNN 侧是低杠杆成员，价值在多样性）|
| **< 0.115** | 主干方向重估 | 先分清欠拟合/过拟合（train loss 滑动均值 + val cos），`--tok-hidden` 调宽排除容量因素 |

用法（**`--crop` 必填**，用户 2026-09-21 指定）：
    python mlp_baseline.py --crop 128 --folds 1 --smoke-fold          # G1 管线冒烟
    python mlp_baseline.py --crop 128 --folds 1 --epochs 2 --time-smoke   # G2 时间/显存
    python mlp_baseline.py --crop 128 --folds 6 \
           --ckpt-dir snap_cache/ckpt_mlp_crop128 --oof-dir snap_cache/oof_parts_mlp_crop128
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(MODELS), "seq"))   # 序列数据管道（顶层 seq/）
sys.path.insert(0, os.path.join(MODELS, "tabm"))    # tabm_common（BASE / cos_score）

from seq_common import F as COARSE_C_BASE  # noqa: E402  基通道数：单一来源
from seq_common import block_perm, make_folds, make_loader  # noqa: E402
from tabm_common import BASE, cos_score  # noqa: E402


COARSE_T, COARSE_C = 224, COARSE_C_BASE
# ⚠️ `COARSE_C_BASE` **取自 `seq_common.F`（单一来源）**，不再手写数字——
#    原先两处独立声明，基通道一变就静默不一致（形状断言会炸，但炸得晚）。
#    派生通道**一律追加在它之后，绝不重排**（cnn_baseline.py 的
#    MASK_CHANNELS 是裸下标 10/12）。
FINE_T, FINE_C = 60, 16
READ_SPAN = 4                 # 读出权重报告时每 4 步聚合成一块（224→56 / 60→15，与 CNN 塔可比）
TARGET_SCALE = 1000.0
SNAP_CACHE = os.path.join(BASE, "snap_cache")
DEFAULT_CKPT_DIR = "snap_cache/ckpt_mlp"
DEFAULT_OOF_DIR = "snap_cache/oof_parts_mlp"

# 判据用参照（双塔CNN 与 E1 各臂，月交错 6 折逐折均值）
REF = {"双塔CNN": 0.12280, "cNone": 0.10776, "coarse": 0.10336,
       "晚融合(coarse+flow)": 0.11514}


# ---------------------------------------------------------------- 数据

def load_train():
    """返回 (粗段 memmap, n_valid, 细段内存数组, y, month, months, sids)。

    - 粗段走 `seq_common.load_train_snap()`（自带与 label 的行序断言）
    - 细段**现建到内存**（`.dat` 已按 2026-09-16 的决定删除）
    - 两份输入必须行序一致（都按 sample_id 升序），这里显式断言
    """
    from seq_common import load_train_snap

    Xc, nv, y, marr, months, sids = load_train_snap()
    assert Xc.shape[1:] == (COARSE_T, COARSE_C), f"粗段形状 {Xc.shape}"

    import flow_feats
    Xf = flow_feats.build_split("train", None, False, to_memory=True)
    assert Xf.shape[1:] == (FINE_T, FINE_C), f"细段形状 {Xf.shape}"
    assert Xf.shape[0] == Xc.shape[0], \
        f"两份输入行数不一致：粗段 {Xc.shape[0]} vs 细段 {Xf.shape[0]}"
    return Xc, nv, Xf, y, marr, months, sids


def crop_src(n, K):
    """裁剪的单一样式来源 → （源位置 src (K,) int，有效位 ok (K,) bool）。

    - **K >= COARSE_T（=224）：不裁剪**，沿用缓存原布局（真数据 `0..n−1`，pad 在尾部）。
      这样 `--crop 224`（默认）与引入本开关之前的行为**逐位一致**，已有结果与正在跑的
      基线都仍然可比。
    - **K < COARSE_T：右对齐裁剪**——取最近的 K 个观测 `[n−K, n)`；不足 K 时在**旧端**补零。
      于是**最新那一行恒落在位置 K−1**（这正是治「位置=序号」那个病的机制）。

    ⚠️ 两种 K 的**对齐方式不同**（224 = 沿用缓存左对齐；<224 = 右对齐），这是刻意的：
    保证 K=224 是恒等。别把它当成 bug。
    ⚠️ K<224 时必须取 `[n−K, n)`，**不能取"数组最后 K 行"**——数组最后 K 行现在全是 pad
    （CLAUDE.md #9：当年 `--tcrop` 就栽在这上面，中位样本 n=189 > 224−64，绝大多数中招）。
    """
    assert 0 < K <= COARSE_T, f"K 必须在 1..{COARSE_T}"
    n = int(n)
    if K >= COARSE_T:
        src = np.arange(K)
        return src, src < n
    # ⚠️ 这里**不能**先把 n 截断成 min(n,K) 再去算源位置：那样 n=199/K=192 会给出 0..191
    #    （= 最旧的 192 行）而不是 [7,199)（= 最近的 192 行）。而且**批量版若犯同一个错，
    #    "单行==整批"的一致性测试照样通过**——只有 test_mlp.py::test_crop_alignment 抓得住。
    k = min(n, K)
    src = np.zeros(K, dtype=np.int64)
    src[K - k:] = np.arange(n - k, n)          # 尾部 k 个 → 最近 k 个观测
    ok = np.zeros(K, dtype=bool)
    ok[K - k:] = True
    return src, ok


def crop_newest(a, n, K):
    """「取最新 K 行」的单行实现：a (C,T) → (C,K)。定义见 `crop_src` 与 PAD_AND_CROP.md §3。

    K >= COARSE_T 时**严格恒等**（连非零的 pad 区也原样返回）——保证 `--crop 224`（默认）
    与本次改动之前的行为**逐位相同**，既有结果与正在跑的基线都仍然可比。
    """
    if K >= COARSE_T:
        return a
    src, ok = crop_src(n, K)
    return np.where(ok[None, :], a[:, src], 0)


def build_one(Xc, Xf, row, n_valid, K=COARSE_T):
    """单行构建器 → (粗段 (15,K) f32, 细段 (16,60) f32)。与形状无关的掩码另给。"""
    a = np.ascontiguousarray(crop_newest(np.asarray(Xc[row]).T, n_valid[row], K), dtype=np.float32)
    b = np.ascontiguousarray(np.asarray(Xf[row], dtype=np.float32).T)
    return a, b


def batch_two(Xc, Xf, n_valid, rows, device, K=COARSE_T):
    """整批版（验证/推理用）→ (粗段 (B,15,K), 细段 (B,16,60))。

    必须与 `build_one` 数值一致——不一致等于训练/验证喂了不同分布的数据，**不会报错**。
    `test_mlp.py::test_builders_agree` 守着。

    裁剪用**向量化 gather**（先把整行搬进内存再按索引取），避免逐行 Python 循环；
    源位置按 `crop_src` 的同一套算式给出。
    """
    full = np.asarray(Xc[rows], dtype=np.float32)                     # (B,T,15)
    if K >= COARSE_T:
        a = full
    else:
        nv = np.asarray(n_valid[rows], dtype=np.int64)                # ⚠️ 不要先 min(nv,K)，见 crop_src
        src = nv[:, None] - K + np.arange(K)[None, :]                 # (B,K) 源位置
        ok = (src >= 0) & (src < nv[:, None])
        a = np.where(ok[:, :, None],
                     full[np.arange(len(rows))[:, None], np.clip(src, 0, COARSE_T - 1)], 0.0)
    b = np.asarray(Xf[rows], dtype=np.float32)
    return (torch.from_numpy(a).permute(0, 2, 1).to(device),
            torch.from_numpy(b).permute(0, 2, 1).to(device))


class MLPDataset(torch.utils.data.Dataset):
    """按行取样 → (x_coarse, x_fine, mask, y)。写成类是为了能被 DataLoader 的 worker pickle。"""

    def __init__(self, rows, y, Xc, Xf, n_valid, K=COARSE_T, scale=TARGET_SCALE):
        self.rows, self.y, self.Xc, self.Xf = rows, y, Xc, Xf
        self.n_valid, self.K, self.scale = n_valid, K, scale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        a, b = build_one(self.Xc, self.Xf, r, self.n_valid, self.K)
        return (torch.from_numpy(a), torch.from_numpy(b),
                torch.tensor(self.y[r] * self.scale, dtype=torch.float32))


# ---------------------------------------------------------------- 模型

class ChannelTimeTower(torch.nn.Module):
    """单塔：**通道维 MLP → 位置嵌入 → 时间维 MLP → 逐步读出 + 可学习尺度** → (B,dim)。

    顺序（用户 2026-09-21 指定，比 MLP-Mixer 的"交替块 + 残差"简单得多）：
        h = channel_mlp(x)   # (B,C,T) → (B,D,T)：逐位置 Linear(C→D)+GELU（跨通道混合）
        h = h + pos          # 位置嵌入 (T,D) —— 正好在时间 MLP 之前，它需要位置信息
        h = time_mlp(h)      # (B,D,T) → (B,D,T)：Linear(T→H)+GELU+Linear(H→T)，权重跨通道共享
    无残差、无多块重复；两处 pre-norm LayerNorm（稳定性件，见 README「训练不稳定」节）。

    ⚠️ 时间维 MLP 的权重矩阵**按位置索引**（`out[j] = Σ_t W[j,t]·x[t]`），
    所以它本身就带位置信息——「位置嵌入是否起作用」不能用"置换输入输出会变"来测
    （那是假阳性，见 `test_mlp.py::T-pos-*` 的注释与正确构造）。

    ⚠️ **本塔没有掩码机制**（原 `--pad-mode none|hard` 的对照已裁决、逻辑已删，2026-09-21）：
    残留的"旧端补零"由**数据侧**的 `--crop K` 负责——它把每个样本裁成"最近 K 个观测，
    不足者在旧端补零"，而 pad 位就是普通的零输入。理由与证据见 `PAD_AND_CROP.md`。
    """

    def __init__(self, n_feats, n_tok, dim=256, tok_hidden=128, drop=0.2):
        super().__init__()
        self.n_tok, self.dim = n_tok, dim
        # 通道维 MLP（逐位置：跨通道混合）。**pre-norm** + 单层 C→D + GELU
        self.ln_in = torch.nn.LayerNorm(n_feats)
        self.channel = torch.nn.Sequential(
            torch.nn.Linear(n_feats, dim), torch.nn.GELU(), torch.nn.Dropout(drop))
        self.pos = torch.nn.Parameter(torch.zeros(n_tok, dim))     # 可学习位置嵌入
        # 时间维 MLP（(B,D,T) 的最后一维是 T：跨时间混合，权重跨通道共享）。**pre-norm**
        self.ln_mid = torch.nn.LayerNorm(dim)
        self.time = torch.nn.Sequential(
            torch.nn.Linear(n_tok, tok_hidden), torch.nn.GELU(), torch.nn.Dropout(drop),
            torch.nn.Linear(tok_hidden, n_tok), torch.nn.Dropout(drop))
        self.read_w = torch.nn.Parameter(torch.zeros(n_tok))       # 逐步读出权重（θ=0 → 均匀）
        self.scale = torch.nn.Parameter(torch.ones(1))             # 尺度自由度（C-10 的注意项）

    def readout_weights(self):
        """(T,) 读出权重（softmax）——报告"学到的形状"用。"""
        with torch.no_grad():
            return torch.softmax(self.read_w, dim=0).cpu().numpy()

    def forward(self, x):                      # x: (B,C,T)
        h = self.channel(self.ln_in(x.transpose(1, 2)))             # (B,T,dim)
        h = h + self.pos
        h = self.time(self.ln_mid(h).transpose(1, 2)).transpose(1, 2)
        w = torch.softmax(self.read_w, dim=0)                       # (T,)
        return (h * w[None, :, None]).sum(1) * self.scale           # (B,dim)


class TwoTowerMLP(torch.nn.Module):
    """粗段塔 + 细段塔（都是"通道→时间"顺序 MLP 主干），各自读出后在 head 处拼接。"""

    def __init__(self, dim=256, tok_hidden=128, drop=0.2, head_drop=0.3,
                 head_dim=256, coarse_t=COARSE_T):
        super().__init__()
        self.tower_a = ChannelTimeTower(COARSE_C, coarse_t, dim, tok_hidden, drop)
        self.tower_b = ChannelTimeTower(FINE_C, FINE_T, dim, tok_hidden, drop)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, head_dim), torch.nn.GELU(),
            torch.nn.Dropout(head_drop), torch.nn.Linear(head_dim, 1))

    def forward(self, x_coarse, x_fine):
        h = torch.cat([self.tower_a(x_coarse), self.tower_b(x_fine)], dim=1)
        return self.head(h)


def build_model(args, device):
    """`coarse_t=args.crop`：粗段塔的时间轴长度完全由 **必填的 `--crop`** 决定。"""
    return TwoTowerMLP(dim=args.dim, tok_hidden=args.tok_hidden, drop=args.drop,
                       head_drop=args.head_drop, coarse_t=args.crop).to(device)


def block_report(w, span=READ_SPAN):
    """逐步读出权重 → 每 span 步聚合的块权重（与 CNN 塔的块权重形状可直接对比）。"""
    w = np.asarray(w, dtype=np.float64)
    assert w.size % span == 0, f"T={w.size} 不是 {span} 的倍数"
    return w.reshape(-1, span).sum(1)


def oof_summary(oof_dir):
    """**以 oof_dir 里全部折的 parquet 重算逐折 cos** → (folds, cos 数组)。

    ⚠️ 为什么不用内存里的 `results`：断点续跑**跳过的折不会进 results**，
    用它做汇总会**漏折**——2026-09-21 实测踩过：折 0 被复用后，6 折被报成 5 折
    （0.12579 而非 0.12712），差 0.0013。汇总一律以**盘上产物**为准。
    """
    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).select(["sample_id", "target"])
    files = sorted(f for f in os.listdir(oof_dir)
                   if f.startswith("f") and f.endswith(".parquet"))
    vals = []
    for f in files:
        p = pl.read_parquet(os.path.join(oof_dir, f)).join(lab, on="sample_id", how="inner")
        pr, tg = p["pred"].to_numpy(), p["target"].to_numpy()
        vals.append(float((pr * tg).sum() / np.sqrt((pr ** 2).sum() * (tg ** 2).sum())))
    return files, np.array(vals)


# ---------------------------------------------------------------- 检查点（全项目首次引入，纪律见 CLAUDE.md #12）



def arch_cfg(args):
    """参与指纹的配置（架构 + 输入形状 + 跑批配置）。**换配置跑同一目录会报错而不是静默覆盖。**"""
    cfg = {"dim": args.dim, "tok_hidden": args.tok_hidden,
           "drop": args.drop, "head_drop": args.head_drop,
           # ⚠️ pad_mode 是**冻结的常量**：掩码逻辑已于 2026-09-21 删除（A/B 裁决完了），
           #    而"删掉掩码"在行为上正是原本的 `none`。保留这个字面量是为了让**删除前训出的
           #    检查点仍能通过指纹校验**（否则 6 折 crop128 的检查点会全部作废）。
           "pad_mode": "none", "seed": args.seed, "norm": "ln",
           # ⚠️ 指纹反映**会影响学出什么权重的全部配置**（不只是架构）：
           #    - 粗段塔的时间轴长度就是 args.crop（**不要**另加 "crop" 字段——那会让
           #      crop=224（= 不裁剪、架构与改动前完全相同）与旧检查点指纹不符而误拒）；
           #    - clip（梯度裁剪）改变训练协议 → 属于配置，必须进指纹，否则换 clip 后会
           #      **静默复用**另一套协议产出的检查点。
           "clip": args.clip, "coarse": [args.crop, COARSE_C], "fine": [FINE_T, FINE_C]}
    # ⚠️ y_weight_q **只在真的生效时才进指纹**（CLAUDE.md #12 的反向纪律：别为"等价于不变"的
    #    开关加字段）。`--y-weight-q 0`（默认）与"没有这个开关"是**同一套训练协议**；无条件加字段
    #    会让已训好的 `ckpt_mlp_crop128/*.pt` 指纹不符而被**误拒**（那是 6 折 ×1h 的产物）。
    #    与"`--crop 224` 不另加 crop 字段"是同一条道理。
    if getattr(args, "y_weight_q", 0.0) > 0:
        cfg["y_weight_q"] = args.y_weight_q
    # 同理：`--loss mse`（默认）就是原协议，不进指纹；换损失才进。
    # ⚠️ 判据是 **`!= "mse"`**，不是 `!= "cos"`——这是**历史编码规则**，别"顺着默认值改"。
    #    存盘的指纹一律按"mse = 无字段"编码（默认值曾是 mse）。改成 `!= "cos"` 会让
    #    **cos 的默认跑算出与 v22 的 MSE 检查点相同的 hash** → 静默加载另一套协议的权重
    #    （实测踩到：cos 默认给 15cf0b8e13614e4b，正是 v22 存盘的那一串）。
    #    保持 `!= "mse"` 则两条归档都能载：cos → 有字段（= v24）、mse → 无字段（= v22）。
    if getattr(args, "loss", "cos") != "mse":
        cfg["loss"] = args.loss
    return cfg


def arch_hash(args):
    blob = json.dumps(arch_cfg(args), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def ckpt_path(ckpt_dir, fold, kind="best", smoke=False):
    """kind ∈ {best, last}；冒烟跑写 `f{fi}_smoke.*`，**绝不碰正式名**
    （否则 1 epoch 的冒烟会被正式跑当成"已完成"而跳过）。"""
    tag = f"f{fold}_smoke" if smoke else f"f{fold}"
    return os.path.join(ckpt_dir, f"{tag}.{kind}.pt")


def save_ckpt(path, model, cfg_hash, meta):
    """原子写：先写 `path + ".tmp"` 再 os.replace。

    ⚠️ `torch.save` **不追加扩展名**（与 `np.save` 相反，见 CLAUDE.md #1）——
    所以 `path + ".tmp"` 是安全的。**不要"顺手改成 `.tmp.pt`"**，那会再造一个同类坑。
    payload 只含 CPU tensor 与基本类型 → 读侧可用 `weights_only=True`。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "arch_hash": cfg_hash, "meta": meta}
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_ckpt(path, args):
    """读检查点。**指纹不符/文件缺失一律报错，绝不静默继续。**"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺检查点 {path}——先跑 mlp_baseline.py 对应折")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    got, want = payload.get("arch_hash"), arch_hash(args)
    if got != want:
        raise ValueError(f"检查点配置不符：{path}\n  文件={got}  当前={want}\n"
                         f"  当前配置 {json.dumps(arch_cfg(args), sort_keys=True)}")
    return payload


def load_model_from_ckpt(path, args, device):
    """读检查点 → 建模型 → 装载权重（不回传 optimizer：提交侧只推理）。"""
    payload = load_ckpt(path, args)
    model = build_model(args, device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("meta", {})


def resume_ckpt(ckpt_dir, fold, smoke=False):
    """断点续跑的键：优先 `best`，其次 `last`；都没有 → None。

    用"哪个存在用哪个"而不是固定 best，是为了 `--ckpt-keep` 改成 last 后仍能接上。
    """
    for kind in ("best", "last"):
        p = ckpt_path(ckpt_dir, fold, kind, smoke)
        if os.path.exists(p):
            return p
    return None


def oof_from_ckpt(path, args, Xc, Xf, nv, va_rows, device):
    """用检查点复算该折 OOF（`.pt` 在而 parquet 不在时用，**免重训**）→ (pred, meta)。"""
    model, meta = load_model_from_ckpt(path, args, device)
    return eval_forward(model, Xc, Xf, nv, va_rows, args.eval_chunk, device, args.crop), meta


# ---------------------------------------------------------------- 评估

def eval_forward(model, Xc, Xf, n_valid, rows, chunk, device, K=None):
    """**分块**前向 → (n,) f32 预测。

    分块是**显存护栏**（整折 21 万行一次性前向的中间张量约 48GB → 必 OOM）。
    无 BN、LayerNorm 逐样本 → 与一次性前向数值一致（实测差 3e-08 量级，非逐位；
    同 chunk 重放则逐位相同——test 7 守着）。

    `K=None` 时**取模型自带的 `tower_a.n_tok`**——这样调用方忘了传也不会错。
    （2026-09-21 的教训：`mlp_submit.py` 就是因为没传 K，拿 K=224 的输入去喂
    只有 128 个位置嵌入的塔，报 `size of tensor a (224) must match b (128)`。）
    """
    if K is None:
        K = int(model.tower_a.n_tok)
    model.eval()
    out = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, len(rows), chunk):
            sl = rows[lo:lo + chunk]
            xa, xb = batch_two(Xc, Xf, n_valid, sl, device, K)
            out[lo:lo + chunk] = model(xa, xb).squeeze(-1).cpu().numpy()
    return out


# ---------------------------------------------------------------- 损失（训练协议件）

def batch_loss(pred, yb, loss_kind="mse", tau=0.0):
    """训练损失。`pred` / `yb` 都已在该折的 target×TARGET_SCALE 单位上。

    - **`mse`（默认）**：走 `F.mse_loss`，与原协议**逐位一致** → 历史检查点与结果口径不变。
    - **`cos`**：`1 − cos(p_batch, y_batch)`，**不中心化**（HYPOTHESES C-17）。
      为什么它与 MSE 的差别**只可能**落在优化动力学上：全局 cos 的梯度
      `∂cos/∂pᵢ ∝ yᵢ − c·pᵢ`（`c = Σpy/Σp²`）**正是** MSE 的负梯度，逐 batch 等价于
      "目标缩放 c 倍"，而 c 是**每批自适应**的尺度（Adam 只做逐参数自适应，做不了这个）。
      另有直观等价（测试 13 钉住）：**把预测整批缩放到与目标同范数再算 MSE**
      ⇒ `Σ(p′−y)² = 2|y|²·(1 − cos(p,y))`。
      **不中心化**的理由：指标是未中心化的 cos，上面那条推导里也没有中心化项；公开方案
      在 batch 内中心化（那优化的是 Pearson 相关，与本指标不是同一个量）。附带好处——
      未中心化的 cos 会**自己**把预测均值压向 0（常数偏移只抬 Σp²、不抬 Σpy）。
      ⚠️ 用户 2026-09-23 明确**不加分母下限 ε**，所以这里没有保护。风险明确：cos 对整体尺度
      不变，`wd` 会把 `‖p‖` 往下推，而梯度 ∝ `1/‖p‖` → **输出尺度塌缩**是这条臂的预期失败
      模式；故 `train_fold` 每轮记录 `|p|` 均值。分母**恰好为 0** 时直接报错（这是真异常，
      不是可以静默变 NaN 的数值噪声）。
    - **`tau > 0`**：按 `Σw` 归一的加权 MSE（`w = 1/(1+|y|/τ)`，HYPOTHESES C-16 的 Y3 臂）。

    逻辑**只此一处**——`train_fold` 调它，测试也直接测它（避免"测了个复制品"）。
    """
    if loss_kind == "cos":
        den = pred.norm() * yb.norm()
        if not torch.isfinite(den) or float(den) == 0.0:
            raise RuntimeError(
                f"[loss=cos] 分母退化：‖p‖={float(pred.norm()):.3g} ‖y‖={float(yb.norm()):.3g}")
        return 1.0 - (pred * yb).sum() / den
    if tau > 0:
        # **按 Σw 归一**（否则 loss 量纲随 τ 漂，等价于偷偷改了有效学习率）
        wgt = 1.0 / (1.0 + yb.abs() / tau)
        err = pred - yb
        return (wgt * err * err).sum() / wgt.sum()
    return F.mse_loss(pred, yb)


# ---------------------------------------------------------------- 训练

def train_fold(Xc, Xf, nv, y, sids, tr_rows, va_rows, args, device, fold, smoke):
    torch.manual_seed(args.seed)
    model = build_model(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = np.random.default_rng(args.seed)
    cfg_hash = arch_hash(args)
    keep_last = args.ckpt_keep in ("last", "both")
    keep_best = args.ckpt_keep in ("best", "both")

    n_tr = len(tr_rows)

    # —— Y3 样本加权（HYPOTHESES C-16 的"target 清洗"臂）：w = 1/(1+|y|/τ) ——
    # 动机（实测）：target 峰度 20.2、max 32σ；放大 1000 倍后**单个 |y|=0.05 的样本平方误差
    # 2500+，而整批 1024 个的均值才 ~4**。crop128 是在 `--clip 1.0` **开着**的情况下仍在
    # ep5–7 塌陷的——梯度裁剪只缩放步长，**不改变梯度方向仍被那一个样本主导**。
    # τ 用**训练折**的 |y| 分位数（只用 tr_rows 算，验证折零泄漏）；w 是 y 的确定性函数，
    # 所以不需要动 Dataset，在 loss 里现算即可。
    # ⚠️ 单位：这里的 `y` 是**未缩放**的 target，而 loss 里的 `yb` 是 `MLPDataset.__getitem__`
    #    乘过 TARGET_SCALE 的。τ **必须换算到同一单位**——否则 `w=1/(1+|yb|/τ)` 的比值对**所有**
    #    样本都极大，退化成 `w≈τ/|yb|`（L1 式加权），那就不是 Y3 了。
    #    （2026-09-23 G1 冒烟实测：漏乘 TARGET_SCALE 时 τ=0.0088 而 |yb|~2.6，比值 ~300。）
    tau = 0.0
    if args.y_weight_q > 0:
        tau = float(np.quantile(np.abs(y[tr_rows]), args.y_weight_q / 100.0) * TARGET_SCALE)
        print(f"    y-weight q{args.y_weight_q:g}: τ={tau:.5f}（= target×{TARGET_SCALE:.0f} 单位，"
              f"原始 {tau / TARGET_SCALE:.3e}）| w(|y|=τ)=0.500  w(|y|=4τ)=0.200  "
              f"w(|y|=0.1τ)=0.909", flush=True)
    # DataLoader 多进程预取：只建一次、每轮换排列（CLAUDE.md 工程教训 #11：每 epoch 新建会泄漏 fd）
    ds = MLPDataset(tr_rows, y, Xc, Xf, nv, K=args.crop)
    loader, sampler = make_loader(ds, n_tr, args.batch, args.workers)

    best_cos, best_epoch, best_va_pred, stall = -1.0, -1, None, 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        sampler.set_perm(block_perm(n_tr, block=4096, gen=gen))
        loss_sum, n_batch, gmax = 0.0, 0, 0.0   # 累计**整轮均值**（末批单点噪声太大，见下）
        pnorm_sum = 0.0                          # `|p|` 均值：cos 臂的**尺度塌缩**监控（loss=cos 时才有意义）
        for xa, xb, yb in loader:
            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xa, xb).squeeze(-1)
            loss = batch_loss(pred, yb, args.loss, tau)
            if args.loss == "cos":
                pnorm_sum += float(pred.abs().mean())
            opt.zero_grad()
            loss.backward()
            # 梯度裁剪：target 峰度 20.2（max |y| = 32σ）、放大 1000 倍后**单个极端样本**的
            # 平方误差可达 2500+（整批均值才 ~4）→ 一步就能把无 BN 的网络打歪且回不来。
            # 双塔CNN 靠 BN 吸收这种冲击，故其协议里没有这一步；本模型的协议因此在**这一处**偏离
            # （lr/batch/drop/折式全不变）——理由与实测见 README「训练不稳定」节。
            # ⚠️ `clip=0` 时改用 `inf` 调用：**不裁剪**（scale=min(inf/norm,1)=1，逐位无操作）
            #    但**照样返回梯度范数** → 无裁剪的臂也有 `gmax` 可见度。
            #    这条不是锦上添花：无裁剪时"输出尺度塌缩 → 梯度 ∝1/‖p‖ 爆炸"是预期失败模式，
            #    看不见就只能靠猜（2026-09-23 C-17 的 cos 臂就是这种情形）。
            gn = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip if args.clip > 0 else float("inf")))
            gmax = max(gmax, gn)             # 记录**裁剪前**的梯度范数峰值（诊断用）
            opt.step()
            loss_sum += float(loss.item())
            n_batch += 1
        loss_mean = loss_sum / max(n_batch, 1)
        pnorm = pnorm_sum / max(n_batch, 1)      # 每批 |p| 的均值再整轮平均
        pnorm_s = f" |p|={pnorm:.3f}" if args.loss == "cos" else ""

        pred_va = eval_forward(model, Xc, Xf, nv, va_rows, args.eval_chunk, device, args.crop)
        cos = cos_score(pred_va, (y[va_rows] * TARGET_SCALE).astype(np.float32))
        if cos > best_cos:
            best_cos, best_epoch, best_va_pred, stall = cos, ep, pred_va, 0
            if keep_best:
                save_ckpt(ckpt_path(args.ckpt_dir, fold, "best", smoke), model, cfg_hash,
                          {"fold": fold, "kind": "best", "epoch": ep, "cos": float(cos),
                           "y_weight_q": args.y_weight_q, "tau": tau,
                           "smoke": bool(smoke)})
        else:
            stall += 1
            if stall >= args.patience:
                print(f"    ep={ep:3d} loss={loss_mean:.4f} val_cos={cos:.5f} "
                      f"gmax={gmax:.1f}{pnorm_s} t={time.time()-t0:5.1f}s  早停", flush=True)
                break
        if ep % 5 == 0:
            print(f"    ep={ep:3d} loss={loss_mean:.4f} val_cos={cos:.5f} "
                  f"gmax={gmax:.1f}{pnorm_s} t={time.time()-t0:5.1f}s", flush=True)

    if keep_last:
        save_ckpt(ckpt_path(args.ckpt_dir, fold, "last", smoke), model, cfg_hash,
                  {"fold": fold, "kind": "last", "epoch": ep, "cos": float(cos),
                   "y_weight_q": args.y_weight_q, "tau": tau,
                   "smoke": bool(smoke)})

    # pad 质量：用**该 K 的对齐方式**取 pad 位（K=224 时 pad 在尾部；K<224 时在旧端）
    nv_med = int(np.median(nv[va_rows]))
    _, _ok = crop_src(nv_med, args.crop)
    w_a = model.tower_a.readout_weights()
    w_b = model.tower_b.readout_weights()
    return {"cos": best_cos, "epoch": best_epoch, "va_pred": best_va_pred,
            "va_sids": sids[va_rows], "fold_secs": time.time() - t0,
            "bw_a": block_report(w_a), "bw_b": block_report(w_b),
            "w_a": w_a, "w_b": w_b,
            "pad_mass": float(w_a[~_ok].sum()),
            "pos_std": (float(model.tower_a.pos.detach().std()),
                        float(model.tower_b.pos.detach().std())),
            "scale_a": float(model.tower_a.scale.detach()),
            "scale_b": float(model.tower_b.scale.detach())}


def _fmt_bw(a, name):
    """块权重摘要：短向量逐个打印；长向量按四段（旧→新）给质量占比。"""
    a = np.asarray(a, dtype=np.float64)
    if a.size <= 20:
        return f"{name}: " + " ".join(f"{v:.3f}" for v in a) + f"  argmax=#{int(a.argmax())}"
    q = np.array_split(a, 4)
    return (f"{name}: 四段质量(旧→新)=" + " / ".join(f"{x.sum():.3f}" for x in q) +
            f"  argmax=#{int(a.argmax())}({a.max():.3f})")


def main():
    p = argparse.ArgumentParser(description="双塔MLP（通道→时间 MLP 主干）联合训练")
    p.add_argument("--folds", type=int, default=6, help="月交错折数（1..6）")
    p.add_argument("--start-fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=16)
    p.add_argument("--batch", type=int, default=1024,
                   help="与双塔CNN / E1 各臂一致，**不要改**（见 CLAUDE.md 工程教训 #3）")
    p.add_argument("--eval-chunk", type=int, default=8192,
                   help="验证前向的分块行数（**显存护栏**，不改数值）。整折一次性前向约 97GB → 必 OOM")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--head-drop", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss", choices=["mse", "cos"], default="cos",
                   help="训练损失。`cos` = `1−cos(p_batch, y_batch)`（**不中心化**，与 C-17 同口径）。"
                        "**默认 cos**（用户 2026-09-26 指令：之后所有模型都用 cos；2026-09-27 落实为默认值）。"
                        "⚠️ 只在 ≠cos 时才进配置指纹")
    p.add_argument("--y-weight-q", type=float, default=0.0,
                   help="样本加权 w=1/(1+|y|/τ) 的分位点（0=关）。τ 取**训练折** |y| 的该分位数"
                        "（验证折零泄漏）。⚠️ 仅在 >0 时进配置指纹 → 换值会另起谱系，"
                        "且必须换 --ckpt-dir/--oof-dir")
    p.add_argument("--clip", type=float, default=1.0,
                   help="梯度裁剪阈值（0 = 关）。target 峰度 20.2、单样本平方误差可达 2500+，无 BN 的网络一步就被打歪；双塔CNN 靠 BN 吸收故其协议无此步，本模型的协议在**这一处**偏离")
    # —— 主干（通道→时间 顺序 MLP；默认 dim 256 / 时间 MLP 隐层 128 ≈ 28.6 万参数）——
    p.add_argument("--tok-hidden", type=int, default=128,
                   help="时间维 MLP 隐层宽（T→H→T 的 H）")
    p.add_argument("--dim", type=int, default=256, help="通道 MLP 输出维（= 两塔读出维）")
    p.add_argument("--crop", type=int, required=True,
                   help=f"**必填**（用户 2026-09-21 指定）：粗段只取**最新的 K 行**，右对齐、"
                        f"最新一行恒在位置 K−1；不足 K 行者在旧端补零。"
                        f"必须是 {READ_SPAN} 的倍数且 ≤{COARSE_T}。实测 128 优于 224（+0.0113，"
                        f"fold 0）——原理/证据/判据见 PAD_AND_CROP.md。"
                        f"（原 --pad-mode none|hard 的掩码对照已裁决、逻辑已删）")
    # —— 检查点 ——
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--ckpt-keep", choices=["best", "last", "both"], default="both",
                   help="best＝val cos 最好那轮；last＝最后一轮（早停时即停在那轮）")
    p.add_argument("--oof-dir", default=DEFAULT_OOF_DIR)
    p.add_argument("--smoke-fold", action="store_true", help="冒烟：每折 1 epoch、不汇总、写 _smoke 名")
    p.add_argument("--time-smoke", action="store_true",
                   help="G2：跑 2 epoch 并报告 s/epoch 与显存峰值（判断 6 折总时长的硬闸门）")
    args = p.parse_args()

    assert 0 < args.crop <= COARSE_T and args.crop % READ_SPAN == 0, \
        f"--crop 必须在 1..{COARSE_T} 且为 {READ_SPAN} 的倍数（逐步读出按 {READ_SPAN} 步聚合报告）"


    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.ckpt_dir = args.ckpt_dir if os.path.isabs(args.ckpt_dir) \
        else os.path.join(BASE, args.ckpt_dir)
    oof_dir = args.oof_dir if os.path.isabs(args.oof_dir) else os.path.join(BASE, args.oof_dir)
    os.makedirs(oof_dir, exist_ok=True)

    print(f"device={device} batch={args.batch} lr={args.lr} wd={args.wd} "
          f"drop={args.drop}/{args.head_drop} seed={args.seed} workers={args.workers} "
          f"clip={args.clip} y_weight_q={args.y_weight_q:g} loss={args.loss}")
    # ⚠️ 两条线**互斥**（项目纪律：一次只改一个东西）。cos 已经是尺度不变的批次统计量，
    #    再叠样本加权就不是"换损失"而是第三种东西了，且没人为它定过判据。
    if args.loss != "mse" and args.y_weight_q > 0:
        raise SystemExit("--loss cos 与 --y-weight-q 不要同时用（一次只改一个变量）；"
                         "C-17 的 cos 臂按用户 2026-09-23 指定：**数据用原本的**")
    print(f"双塔MLP：粗段 ({COARSE_C},{args.crop}) + 细段 ({FINE_C},{FINE_T}) → 各 {args.dim} 维拼接 → head；"
          f"主干 = LN → 通道MLP(逐位置 C→{args.dim}) → 位置嵌入 → LN → 时间MLP(T→{args.tok_hidden}→T) → 逐步读出")
    paper = "不裁剪（沿用缓存原布局，真数据在 0..n_valid−1）" if args.crop >= COARSE_T \
        else f"取最新 {args.crop} 行（右对齐；最新一行恒在位置 {args.crop - 1}）"
    print(f"粗段输入：{paper}   [PAD_AND_CROP.md]")
    n_par = sum(q.numel() for q in build_model(args, "cpu").parameters())
    print(f"参数量 {n_par:,}（双塔CNN 423k 的 {n_par/423000:.2f}×）；"
          f"eval_chunk={args.eval_chunk}；cfg 指纹={arch_hash(args)}")
    print(f"OOF → {oof_dir}；检查点 → {args.ckpt_dir}")
    print("判据（预注册）：" + "  ".join(f"{k}={v:.5f}" for k, v in REF.items()))

    t0 = time.time()
    Xc, nv, Xf, y, marr, months, sids = load_train()
    print(f"loaded: 粗段 {Xc.shape} {'f16' if isinstance(Xc, np.memmap) else 'mem'} + "
          f"细段 {Xf.shape}（n_valid 中位 {int(np.median(nv))}；{time.time()-t0:.1f}s）", flush=True)

    folds = make_folds(months, args.folds)
    results = []
    for fi, va_months in enumerate(folds):
        if fi < args.start_fold:
            continue
        part_path = os.path.join(oof_dir, f"f{fi}.parquet")
        smoke_naming = args.smoke_fold or args.time_smoke
        resume_path = resume_ckpt(args.ckpt_dir, fi, smoke_naming)
        va_mask = np.isin(marr, va_months)
        tr_rows = np.where(~va_mask)[0]
        va_rows = np.where(va_mask)[0]

        # 断点续跑：检查点存在且指纹匹配 → 免重训；parquet 缺了就用检查点补出来
        if resume_path is not None and not smoke_naming:
            try:
                pred, meta = oof_from_ckpt(resume_path, args, Xc, Xf, nv, va_rows, device)
            except ValueError as e:
                raise SystemExit(f"[fold {fi}] 检查点存在但配置不符，拒绝覆盖：\n{e}")
            if not os.path.exists(part_path):
                pl.DataFrame({"sample_id": sids[va_rows], "month": marr[va_rows],
                              "pred": pred.astype(np.float64)}).write_parquet(part_path)
                print(f"[fold {fi+1}/{len(folds)}] 复用检查点补出 OOF（{part_path}）", flush=True)
            else:
                print(f"[fold {fi+1}/{len(folds)}] 检查点与 OOF 都在，跳过"
                      f"（cos={meta.get('cos', float('nan')):.5f}）", flush=True)
            continue

        print(f"[fold {fi + 1}/{len(folds)}] 验证月={va_months} "
              f"训练={len(tr_rows)} 验证={len(va_rows)}", flush=True)
        epochs = 1 if args.smoke_fold else args.epochs
        if args.time_smoke:
            epochs = 2
        t1 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        r = train_fold(Xc, Xf, nv, y, sids, tr_rows, va_rows,
                       argparse.Namespace(**{**vars(args), "epochs": epochs}), device, fi,
                       smoke_naming)
        r["fold_secs"] = time.time() - t1
        results.append(r)

        if args.time_smoke:
            per_ep = r["fold_secs"] / epochs
            peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
            print(f"\n### G2 实测 ###\n  {per_ep:.1f} s/epoch（{epochs} epoch 共 {r['fold_secs']:.0f}s）"
                  f"\n  显存峰值 {peak:.1f} GiB\n"
                  f"  投影 6 折 ≈ {per_ep * 27 * 6 / 3600:.2f} h"
                  f"（按每折 27 epoch = best 10 + patience 16 估）", flush=True)
            print("\n（G2 模式：不写正式 OOF/检查点，只报时间与显存）")
            return

        if args.smoke_fold:
            print(f"  → cos={r['cos']:.5f} best_epoch={r['epoch']} 耗时 {r['fold_secs']:.0f}s"
                  f"（冒烟：写 _smoke 名，不落 OOF）", flush=True)
            continue

        pl.DataFrame({"sample_id": r["va_sids"], "month": marr[va_rows],
                      "pred": r["va_pred"].astype(np.float64)}).write_parquet(part_path)
        print(f"  → cos={r['cos']:.5f} best_epoch={r['epoch']} 耗时 {r['fold_secs']:.0f}s"
              f"（已落盘 {part_path}）", flush=True)
        print(f"    pad 质量={r['pad_mass']:.3f}  pos_std A/B={r['pos_std'][0]:.4f}/"
              f"{r['pos_std'][1]:.4f}  尺度 A={r['scale_a']:.3f} B={r['scale_b']:.3f}", flush=True)
        print(f"    {_fmt_bw(r['bw_a'], '塔A(粗段)')}", flush=True)
        print(f"    {_fmt_bw(r['bw_b'], '塔B(细段)')}", flush=True)

    if args.smoke_fold or len(results) <= 0:
        print("\n（冒烟模式：不汇总）")
        return

    # 逐折 cos：**以盘上 OOF 为准**（续跑跳过的折不进 results，用它会漏折——见 oof_summary）
    files, cos_arr = oof_summary(oof_dir)
    ep_arr = np.array([r["epoch"] for r in results])
    n_done = len(cos_arr)
    print(f"\n### 双塔MLP 逐折 cos（{n_done} 折，月交错，crop={args.crop}，clip={args.clip}）###")
    for fn, c in zip(files, cos_arr):
        print(f"  {fn}: cos={c:.5f}")
    print(f"mean={cos_arr.mean():.5f} std={cos_arr.std():.5f} "
          f"min={cos_arr.min():.5f} neg={int((cos_arr < 0).sum())}")
    if len(results) < n_done:
        print(f"（本次训练 {len(results)} 折；其余 {n_done - len(results)} 折取自盘上的检查点/OOF）")
    if len(ep_arr):
        print(f"best_epoch: median={int(np.median(ep_arr))} 本次各折={ep_arr.tolist()}")
    if len(results) > 1:
        print(_fmt_bw(np.mean([r["bw_a"] for r in results], axis=0), "塔A(粗段) 均值"))
        print(_fmt_bw(np.mean([r["bw_b"] for r in results], axis=0), "塔B(细段) 均值"))
    print("\n判据对照（预注册）：")
    for k, v in REF.items():
        print(f"  {'✅ 已超过' if cos_arr.mean() >= v else '❌ 未达到'}  {k} = {v:.5f}")
    print(f"  本模型 = {cos_arr.mean():.5f}")

    if len(results) > 1:
        parts = [pl.read_parquet(os.path.join(oof_dir, f))
                 for f in sorted(os.listdir(oof_dir)) if f.endswith(".parquet")]
        oof_path = os.path.join(oof_dir, "oof_all.parquet")
        pl.concat(parts).write_parquet(oof_path)
        print(f"\nOOF 已存 {oof_path}（{sum(p.height for p in parts)} 行）")


if __name__ == "__main__":
    main()
