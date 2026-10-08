"""twotower_baseline.py — **双塔CNN**（粗段 672 秒 + 细段 60 秒）联合训练：C-13 的「先分再合」。

> 命名（2026-09-17）：本模型叫 **双塔CNN**——它是"双塔×"家族的第一个成员，
> 将来还会有别的双塔模型（双塔×TabM 之类）。提交侧见 `twotower_cnn_submit.py`。

**独立脚本**——不 import `flow_baseline.py`，也不 import `mlp_baseline.py` / `tcn_baseline.py`：
每个脚本是一支独立的实验谱系，改一个不应影响另一个已记录的结果（记录纪律，见 CLAUDE.md #7）。
只复用**共享基础设施**：`seq_common`（SeqDataset 之外的 loader / 折式 / 块洗牌）与
`tabm_common`（BASE / cos_score）。

为什么必须双塔：**粗段 224 步 × 3 秒与细段 60 步 × 1 秒时间轴长度不同，塞不进同一个张量**
（除非改数据布局另拼一张表，那是 D-04）。而这一对恰好是已知最互补的：
`corr(coarse, flow) = 0.517`、模型外融合增益 **+0.01178**（→ 0.11514）。
对照 flow+mkt 那一对：融合增益只有 +0.0006，天花板太低，不值得为它建双塔。

设计（依据 HYPOTHESES C-13、MEMORY「新主干（E1）」、WALKTHROUGH9）：

```
塔 A（粗段） snap_cache train_snap_X.dat (N,224,17) f16 + n_valid
             → 3 层因果卷积(k5s2, k5s2, k3s1 d=4) → 56 块(每块 4 步=12s)
             → **pad 自由**的可学习块权重读出 → h_A (B,256)
塔 B（细段） flow_cache 现建到内存 (N,60,16) f16
             → 3 层因果卷积(k5s2, k5s2, k3s1) → 15 块(每块 4 步=4s)
             → 可学习块权重读出 → h_B (B,256)
合           concat[h_A ; h_B] (512) → Linear(512→256) + GELU + Drop(head_drop) → Linear(256→1)
```

两个**刻意的**选择（都有实测依据，不是默认）：

1. **两条塔都用"pad 自由"的读出（不加任何掩码）**：E1 的 6 折对照里，粗段加掩码（`hard`，无效块
   概率置 0）只有 **0.08398**，不加掩码（`none`）是 **0.10776**——**差 0.024**。
   所以这里直接把掩码去掉：全零的 pad 区经因果卷积后是近常数向量，让它以一定权重进池化
   本身是正收益（详见 WALKTHROUGH9 §4.4）。**本脚本因此完全没有 mask 语义**，`n_valid` 只用于校验。
2. **可学习尺度**：每塔读出后乘一个可学习 scalar（C-10 的注意项——两塔表示的尺度不同，
   硬拼 concat 会先被尺度差异支配）。

**从头训练**，不加载任何已有模型的权重（用户 2026-09-17 明确）。

训练协议与 E1 各臂**逐字一致**（lr 1e-3 / wd 3e-4 / batch 1024 / drop 0.2 / head_drop 0.3 /
patience 16 / epochs 300 / seed 42 / MSE(target×1000) / val cos 早停 / 月交错 6 折）——
折式没变，所以**逐折配对差仍然可比**。

**判据（预注册，跑之前定死）**：

| 结果 | 含义 | 后续 |
|---|---|---|
| **≥ 0.11514** | 联合训练**没输**给"分开训练 + 晚融合"（粗段+flow 的诚实值），且多一条交互通路 | 继续加塔（both / cNone），冲 0.1222 |
| 0.108 ~ 0.115 | 拿到单模型但没超过晚融合——联合训练又一次输给晚融合 | 与 C-13 一致：本项目走"独立训练 + 晚融合"，双塔封存 |
| < 0.108（单流 `both` 的水平）| 连"合"这一步都没学会 | 架构方向重估 |

⚠️ 已知风险：本项目目前**所有**"联合 vs 分开+晚融合"的对照，方向都是**联合输**
（`both` 0.10626 < flow+mkt 0.10921，差 −0.0029）。双塔要翻掉这个规律再去吃交互。

用法：
    python twotower_baseline.py --folds 2                   # 2 折快筛（闸门 1）
    python twotower_baseline.py --folds 6                   # 正式
    # 探针（审权重/画核/冒烟）——**必须另给 --oof-dir**，否则会被下面的防线拦住：
    python twotower_baseline.py --only-fold 0 --oof-dir /root/msdata/snap_cache/oof_probe_twotower

⚠️ **探针跑有两道覆盖防线**（2026-09-24 加）：`--only-fold` 或 `--smoke-fold` 时，若目标
`f{fi}.parquet` 已存在就**直接报错退出**、不静默覆盖 —— 因为默认的 `oof_parts_twotower/`
里躺的是 **v21（LB 0.133）的归档 OOF**（CLAUDE.md #7 的同类坑）。正式 6 折跑不拦。
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "seq"))   # 序列数据管道（顶层 seq/）

from seq_common import F as COARSE_C_BASE  # noqa: E402  基通道数：单一来源
from seq_common import block_perm, make_folds, make_loader  # noqa: E402

# 跨目录复用（models/tabm）——目录结构见 README
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tabm"))

from tabm_common import BASE, cos_score  # noqa: E402


SPAN = 4                     # 每个输出位置覆盖的输入步数（两层 stride-2）
COARSE_T, COARSE_C = 224, COARSE_C_BASE
# ⚠️ `COARSE_C_BASE` **取自 `seq_common.F`（单一来源）**，不再手写数字——
#    原先两处独立声明，基通道一变就静默不一致（形状断言会炸，但炸得晚）。
#    派生通道**一律追加在它之后，绝不重排**（cnn_baseline.py 的
#    MASK_CHANNELS 是裸下标 10/12）。
FINE_T, FINE_C = 60, 16
COARSE_BLOCKS = COARSE_T // SPAN      # 56
FINE_BLOCKS = FINE_T // SPAN          # 15
FLOW_CACHE = os.path.join(BASE, "flow_cache")
SNAP_CACHE = os.path.join(BASE, "snap_cache")
TARGET_SCALE = 1000.0

# 判据用参照（E1 六条臂，月交错 6 折逐折均值）
REF = {"coarse": 0.10336, "flow": 0.10055, "cNone": 0.10776,
       "晚融合(coarse+flow)": 0.11514, "四模型集成": 0.12218}

DEFAULT_CKPT_DIR = os.path.join(SNAP_CACHE, "ckpt_twotower")


# ---------------------------------------------------------------- 检查点
# 2026-09-24 引入，纪律见 CLAUDE.md #12（本模型此前**不存权重**，所以卷积核从来没被留下过）。
# 与 mlp_baseline.py 的同名函数是同一套：只存 CPU state_dict + 配置指纹，读侧**必须**断言指纹。



def arch_cfg(args):
    """参与指纹的配置。**换配置跑同一目录会报错而不是静默覆盖。**"""
    cfg = {
        # 架构：这些都改变"学出什么权重"。`out_dim`/`head_dim`/`dil` 目前是写死在
        # `TwoTowerCNN.__init__` 里的常量（不是 CLI 参数）——照实记下来，将来若被参数化，
        # 指纹会自然跟着变。
        "out_dim": 256, "head_dim": 256, "dil": [4, 1],
        "drop": args.drop, "head_drop": args.head_drop,
        "coarse": [COARSE_T, COARSE_C], "fine": [FINE_T, FINE_C],
        # 协议里**会改变学出权重**的项
        "seed": args.seed, "lr": args.lr, "wd": args.wd, "batch": args.batch,
    }
    return cfg


def arch_hash(args):
    blob = json.dumps(arch_cfg(args), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def ckpt_path(ckpt_dir, fold, kind="best", smoke=False):
    """kind ∈ {best, last}；冒烟跑写 `f{fi}_smoke.*`，**绝不碰正式名**
    （否则 1 epoch 的冒烟会被正式跑当成"已完成"而跳过）。"""
    tag = f"f{fold}_smoke" if smoke else f"f{fold}"
    return os.path.join(ckpt_dir, f"{tag}.{kind}.pt")


def save_ckpt(path, state_dict, cfg_hash, meta):
    """原子写：先写 `path + ".tmp"` 再 `os.replace`。

    ⚠️ `torch.save` **不追加扩展名**（与 `np.save` 相反，见 CLAUDE.md #1）——
    所以 `path + ".tmp"` 是安全的。**不要"顺手改成 `.tmp.pt`"**，那会再造一个同类坑。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
               "arch_hash": cfg_hash, "meta": meta}
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_ckpt(path, args):
    """读检查点。**指纹不符/文件缺失一律报错，绝不静默继续。**"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺检查点 {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    got, want = payload.get("arch_hash"), arch_hash(args)
    if got != want:
        raise ValueError(f"检查点配置不符：{path}\n  文件={got}  当前={want}\n"
                         f"  当前配置 {json.dumps(arch_cfg(args), sort_keys=True)}")
    return payload


# ---------------------------------------------------------------- 数据

def load_train():
    """返回 (粗段 memmap, n_valid, 细段内存数组, y, month, months, sids)。

    - 粗段走 `seq_common.load_train_snap()`（自带与 label 的行序断言）
    - 细段**现建到内存**（`.dat` 已按 2026-09-16 的决定删除）
    - 两份输入必须**行序一致**（都按 sample_id 升序），这里显式断言
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


def build_one(Xc, Xf, row):
    """单行构建器 → (粗段 (C,224) f32, 细段 (C,60) f32)。**无 mask**（见模块 docstring 选择 1）。"""
    a = np.ascontiguousarray(np.asarray(Xc[row], dtype=np.float32).T)   # (15,224)
    b = np.ascontiguousarray(np.asarray(Xf[row], dtype=np.float32).T)   # (16,60)
    return a, b


def batch_two(Xc, Xf, rows, device):
    """整批版（验证集用）→ (粗段 (B,15,224), 细段 (B,16,60))。

    必须与 `build_one` 数值一致——不一致等于训练/验证喂了不同分布的数据，**不会报错**。
    `test_twotower_baseline.py::test_builders_agree` 守着。
    """
    a = torch.from_numpy(np.asarray(Xc[rows], dtype=np.float32)).permute(0, 2, 1).to(device)
    b = torch.from_numpy(np.asarray(Xf[rows], dtype=np.float32)).permute(0, 2, 1).to(device)
    return a, b


class TwoTowerDataset(torch.utils.data.Dataset):
    """按行取样 → (x_coarse, x_fine, y)。写成类是为了能被 DataLoader 的 worker pickle。"""

    def __init__(self, rows, y, Xc, Xf, scale=TARGET_SCALE):
        self.rows, self.y, self.Xc, self.Xf, self.scale = rows, y, Xc, Xf, scale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        a, b = build_one(self.Xc, self.Xf, r)
        return (torch.from_numpy(a), torch.from_numpy(b),
                torch.tensor(self.y[r] * self.scale, dtype=torch.float32))


# ---------------------------------------------------------------- 模型

class CausalConv(torch.nn.Module):
    """因果 Conv1d：左侧 pad (k−1)·dilation（torch 的 padding 只支持对称，用 F.pad 实现）。"""

    def __init__(self, cin, cout, k, stride=1, dilation=1):
        super().__init__()
        self.left = (k - 1) * dilation
        self.conv = torch.nn.Conv1d(cin, cout, k, stride=stride, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left, 0)))


class Tower(torch.nn.Module):
    """单塔：3 层因果卷积 + 可学习块权重读出 + 可学习尺度 → (B, 256)。"""

    def __init__(self, n_feats, n_blocks, l3_dilation=1, drop=0.2, out_dim=256):
        super().__init__()
        self.conv = torch.nn.Sequential(
            CausalConv(n_feats, 64, 5, stride=2), torch.nn.BatchNorm1d(64),
            torch.nn.GELU(), torch.nn.Dropout(drop),
            CausalConv(64, 128, 5, stride=2), torch.nn.BatchNorm1d(128),
            torch.nn.GELU(), torch.nn.Dropout(drop),
            CausalConv(128, out_dim, 3, stride=1, dilation=l3_dilation),
            torch.nn.BatchNorm1d(out_dim), torch.nn.GELU(), torch.nn.Dropout(drop),
        )
        self.block_w = torch.nn.Parameter(torch.zeros(n_blocks))   # softmax → 块权重
        self.scale = torch.nn.Parameter(torch.ones(1))             # 尺度自由度

    def block_weights(self):
        with torch.no_grad():
            return torch.softmax(self.block_w, dim=0).cpu().numpy()

    def forward(self, x):
        h = self.conv(x)                                   # (B,256,nb)
        a = torch.softmax(self.block_w, dim=0)             # (nb,)
        h = (h * a[None, None, :]).sum(-1)                 # (B,256)
        return h * self.scale


class TwoTowerCNN(torch.nn.Module):
    """粗段塔 + 细段塔，各自读出后在 head 处拼接。

    `tower_a`（粗段）用 `l3_dilation=4`：224 步上照搬 d=1 的 RF 只有 21 步 ≈63 秒
    （占 672 秒窗口 9%，比旧主干 17% 还差）；d=4 → RF 45 步 ≈135 秒 = 20%。
    """

    def __init__(self, drop=0.2, head_drop=0.3, out=1, head_dim=256, out_dim=256):
        super().__init__()
        self.tower_a = Tower(COARSE_C, COARSE_BLOCKS, l3_dilation=4, drop=drop, out_dim=out_dim)
        self.tower_b = Tower(FINE_C, FINE_BLOCKS, l3_dilation=1, drop=drop, out_dim=out_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * out_dim, head_dim), torch.nn.GELU(),
            torch.nn.Dropout(head_drop), torch.nn.Linear(head_dim, out),
        )

    def forward(self, x_coarse, x_fine):
        h = torch.cat([self.tower_a(x_coarse), self.tower_b(x_fine)], dim=1)   # (B,512)
        return self.head(h)


# ---------------------------------------------------------------- 训练

def train_fold(Xc, Xf, y, sids, tr_rows, va_rows, args, device,
               ckpt_dir="", cfg_hash=None, fold=0, smoke=False):
    xav, xbv = batch_two(Xc, Xf, va_rows, device)
    yv = (y[va_rows] * TARGET_SCALE).astype(np.float32)

    torch.manual_seed(args.seed)
    model = TwoTowerCNN(drop=args.drop, head_drop=args.head_drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = np.random.default_rng(args.seed)

    n_tr = len(tr_rows)
    # DataLoader 多进程预取：读盘/读数组 + f16→f32 + 转置挪到 worker，与 GPU 计算重叠。
    # 批次顺序由 block_perm 决定、shuffle=False → 与手写循环逐位相同。
    ds = TwoTowerDataset(tr_rows, y, Xc, Xf)
    loader, sampler = make_loader(ds, n_tr, args.batch, args.workers)

    best_cos, best_epoch, best_va_pred, stall = -1.0, -1, None, 0
    best_sd = None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        sampler.set_perm(block_perm(n_tr, block=4096, gen=gen))
        for xa, xb, yb in loader:
            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = F.mse_loss(model(xa, xb).squeeze(-1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(xav, xbv).squeeze(-1).cpu().numpy()
        cos = cos_score(pred_va, yv)
        if cos > best_cos:
            best_cos, best_epoch, best_va_pred, stall = cos, ep, pred_va, 0
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= args.patience:
                print(f"    ep={ep:3d} loss={loss.item():.4f} val_cos={cos:.5f} "
                      f"t={time.time()-t0:5.1f}s  早停", flush=True)
                break
        if ep % 5 == 0:
            print(f"    ep={ep:3d} loss={loss.item():.4f} val_cos={cos:.5f} "
                  f"t={time.time()-t0:5.1f}s", flush=True)

    # 落盘检查点（best 用于复用/审权重，last 用于断点续跑；冒烟跑走 `_smoke` 名）
    if ckpt_dir and best_sd is not None:
        meta = {"fold": fold, "epoch": best_epoch, "cos": best_cos, "seed": args.seed}
        save_ckpt(ckpt_path(ckpt_dir, fold, "best", smoke), best_sd, cfg_hash, meta)
        save_ckpt(ckpt_path(ckpt_dir, fold, "last", smoke), model.state_dict(),
                  cfg_hash, {**meta, "epoch": ep, "cos": float(cos)})

    return {"cos": best_cos, "epoch": best_epoch, "va_pred": best_va_pred,
            "va_sids": sids[va_rows],
            "bw_a": model.tower_a.block_weights(), "bw_b": model.tower_b.block_weights(),
            "scale_a": float(model.tower_a.scale.detach()), "scale_b": float(model.tower_b.scale.detach())}


def _fmt_bw(a, name):
    """块权重摘要：短向量逐个打印；长向量按四段（旧→新）给质量占比。"""
    a = np.asarray(a, dtype=np.float64)
    if a.size <= 20:
        return f"{name}: " + " ".join(f"{v:.3f}" for v in a) + f"  argmax=#{int(a.argmax())}"
    q = np.array_split(a, 4)
    return (f"{name}: 四段质量(旧→新)=" + " / ".join(f"{x.sum():.3f}" for x in q) +
            f"  argmax=#{int(a.argmax())}({a.max():.3f})")


def main():
    p = argparse.ArgumentParser(description="双塔（粗段 672s + 细段 60s）联合训练")
    p.add_argument("--folds", type=int, default=6, help="月交错折数（1..6）")
    p.add_argument("--start-fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=16)
    p.add_argument("--batch", type=int, default=1024,
                   help="与 E1 各臂一致，**不要改**（见 CLAUDE.md 工程教训 #3）")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--head-drop", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--oof-dir", default="snap_cache/oof_parts_twotower")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR,
                   help="逐折检查点目录（best/last，纪律见 CLAUDE.md #12）；空串 = 不存")
    p.add_argument("--only-fold", type=int, default=-1,
                   help="只跑这一折（-1 = 按 --start-fold/--folds 全跑）；审权重/画核用")
    p.add_argument("--smoke-fold", action="store_true",
                   help="冒烟：每折只跑 1 epoch、不汇总（配 --folds 1 用，验管线）")
    args = p.parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    oof_dir = args.oof_dir if os.path.isabs(args.oof_dir) else os.path.join(BASE, args.oof_dir)
    os.makedirs(oof_dir, exist_ok=True)
    ckpt_dir = args.ckpt_dir
    if ckpt_dir and not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(BASE, ckpt_dir)
    cfg_hash = arch_hash(args) if ckpt_dir else None

    print(f"device={device} batch={args.batch} lr={args.lr} wd={args.wd} "
          f"drop={args.drop}/{args.head_drop} seed={args.seed} workers={args.workers}")
    print(f"双塔：粗段 ({COARSE_C},{COARSE_T})→{COARSE_BLOCKS} 块(d=4) + 细段 ({FINE_C},{FINE_T})"
          f"→{FINE_BLOCKS} 块  →  各 256 维拼接 512  →  head")
    print(f"**从头训练**、不加载已有权重；两塔读出**都不加掩码**（E1 的 none 口径，差 0.024）")
    print(f"OOF → {oof_dir}")
    print(f"检查点 → {ckpt_dir or '（不存）'}   cfg 指纹={cfg_hash or '-'}"
          + (f"   **只跑 fold {args.only_fold}**" if args.only_fold >= 0 else ""))
    print("判据（预注册）：" + "  ".join(f"{k}={v:.5f}" for k, v in REF.items()))

    t0 = time.time()
    Xc, nv, Xf, y, marr, months, sids = load_train()
    print(f"loaded: 粗段 {Xc.shape} {'f16' if isinstance(Xc, np.memmap) else 'mem'} + "
          f"细段 {Xf.shape}（n_valid 中位 {int(np.median(nv))}，仅用于校验；"
          f"{time.time()-t0:.1f}s）", flush=True)

    folds = make_folds(months, args.folds)
    results = []
    for fi, va_months in enumerate(folds):
        if fi < args.start_fold:
            continue
        if args.only_fold >= 0 and fi != args.only_fold:
            continue
        part_path = os.path.join(oof_dir, f"f{fi}.parquet")
        # **探针跑**（`--only-fold` 审权重/画核、`--smoke-fold` 验管线）默认 oof_dir 里躺着
        # **正式归档**的逐折 OOF（`oof_parts_twotower/` 是 v21 LB 0.133 的组成）
        # → 拒绝静默覆盖（CLAUDE.md #7）。正式 6 折跑不拦（那是刻意的重跑）。
        if (args.only_fold >= 0 or args.smoke_fold) and os.path.exists(part_path):
            raise SystemExit(f"[探针跑] 目标 OOF 已存在，拒绝覆盖：{part_path}\n"
                             f"  请另给 --oof-dir，例如 snap_cache/oof_parts_twotower_probe")
        va_mask = np.isin(marr, va_months)
        tr_rows = np.where(~va_mask)[0]
        va_rows = np.where(va_mask)[0]
        print(f"[fold {fi + 1}/{len(folds)}] 验证月={va_months} "
              f"训练={len(tr_rows)} 验证={len(va_rows)}", flush=True)
        t1 = time.time()
        epochs = 1 if args.smoke_fold else args.epochs
        r = train_fold(Xc, Xf, y, sids, tr_rows, va_rows,
                       argparse.Namespace(**{**vars(args), "epochs": epochs}), device,
                       ckpt_dir=ckpt_dir, cfg_hash=cfg_hash, fold=fi, smoke=args.smoke_fold)
        r["fold_secs"] = time.time() - t1
        results.append(r)
        pl.DataFrame({"sample_id": r["va_sids"], "month": marr[va_rows],
                      "pred": r["va_pred"].astype(np.float64)}).write_parquet(part_path)
        print(f"  → cos={r['cos']:.5f} best_epoch={r['epoch']} "
              f"耗时 {r['fold_secs']:.0f}s（已落盘 {part_path}）", flush=True)
        print(f"    {_fmt_bw(r['bw_a'], '塔A(粗段)')}", flush=True)
        print(f"    {_fmt_bw(r['bw_b'], '塔B(细段)')}  尺度 A={r['scale_a']:.3f} B={r['scale_b']:.3f}",
              flush=True)

    if args.smoke_fold or len(results) <= 0:
        print("\n（冒烟模式：不汇总）")
        return

    cos_arr = np.array([r["cos"] for r in results])
    ep_arr = np.array([r["epoch"] for r in results])
    print(f"\n### 双塔 逐折 cos（{len(results)} 折，月交错）###")
    print(f"mean={cos_arr.mean():.5f} std={cos_arr.std():.5f} "
          f"min={cos_arr.min():.5f} neg={int((cos_arr < 0).sum())}")
    print(f"best_epoch: median={int(np.median(ep_arr))} 各折={ep_arr.tolist()}")
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
