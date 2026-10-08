"""evt_baseline.py — **三塔TCN·扁平事件轴**：market | order 事件 | transaction 事件（2026-09-30）。

## 与前两版事件臂的区别

| 版本 | 细段存储 | pad | 截断 | 每样本算力 | 内存 |
|---|---|---|---|---|---|
| 桶式（基线 `tcn_baseline`）| 定长 60 格 | 0 | 0 | 60 | — |
| 事件·合并（layout 1/2）| 定长 384 | 43% | 15.5% / **25.0%** | 384 | 7.2GB |
| 事件·分塔（layout 3）| 定长 384+320 | 65%/74% | 7.1% / 13.5% | 704 | 10.8GB |
| **本版（layout 4）** | **扁平 + 批内 pad** | **≈10%** | **0** | **≈250** | **3.6GB** |

为什么值得改：定长版的 `L` **由长尾决定、浪费由中位付**（order 中位 88 / p99 716），
两个压力方向相反，任何固定 L 都不划算。扁平存储把两个病一起消掉。
（桶式臂天生没这两个病——它的格号"距现在第 k 秒"跨样本对齐，空着就是真实的零活动。）

## 批内定长、批间变长——模型为什么能接受

- **批内**所有样本 pad 到该批最长 → 每次前向拿到的都是规整 `(B, C, T)`
- **批间** T 不同：主干里**没有任何一处依赖 T**（Conv1d 权重只取决于通道数与核长；
  BN 逐通道；head 在时间维已消失）
- **最关键**：因果卷积 + 末尾 pad ⇒ 位置 `i` 的输出只依赖 `≤ i` 的输入；
  读出取逐样本的 `n_kept−1` ⇒ **补多长都不改变读出点的值**，且 eval 模式下 BN 用
  running stats ⇒ **同一样本放进不同 T 的批，输出逐位相同**（`test_evt.py` 钉住）

采样：**按事件数排序后切批 + ±10% 的排序抖动**（每轮重排，避免"永远同几条样本同批"）。
pad 时把 T **对齐到 8 的倍数**——不然 ~1000 种形状会让 cuDNN 反复选 kernel。

## 主干一行不改

`CausalConv` / `TCNBlock` / `TCNTower` / `batch_loss` **全部 import 自 `tcn_baseline`**，
本文件里没有任何一行卷积/残差/BN 的实现。三条塔都是**同一个类** `TCNTower` 的实例。

⚠️ **归因须知**：与归档的双塔TCN-cos 比，本臂同时改了"表示"（桶→事件）、"分塔数"（2→3）、
"存储方式"（定长→扁平）。**不是纯单变量对照**——先用它判"值不值得继续"。

用法：
    python models/tcn/evt_baseline.py --stats            # 看两条流事件数分布
    python models/tcn/evt_baseline.py --only-fold 0      # 单折探针
    python models/tcn/evt_baseline.py --folds 6          # 正式 6 折

⚠️ **内存**：扁平缓存现建到内存（order 2.4GB + txn 1.3GB ≈ **3.6GB**，构建峰值约 7GB）
⇒ 只能在**有卡模式**跑（无卡模式 memory.max = 2GiB 会被静默 OOM，见 CLAUDE.md #16）。
⚠️ **本脚本不再用 DataLoader**：打包是纯内存向量化（~3ms/批），不需要多进程——
顺带免掉 fork/polars 与 fd 泄漏两类坑。
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

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(MODELS), "seq"))   # 序列数据管道（顶层 seq/）
sys.path.insert(0, os.path.join(MODELS, "tabm"))                   # tabm_common（BASE / cos_score）

import event_feats as EF  # noqa: E402
import tcn_baseline as T  # noqa: E402  ★ 主干与协议件**全部来自这里**（唯一来源）
# 打包 / 分批 / 上卡：**2026-10-01 移到顶层 `seq/event_batch.py`**（会被 tfm 等其他模型族
# import，按 CLAUDE.md 的目录分层不该住在模型目录里）。函数逐字未改，行为由本目录
# `test_evt.py` 的 55 条断言守着。保留 `EB` 别名只为在下面标注出处。
import event_batch as EB  # noqa: E402,F401
from event_batch import (ALIGN, EVT_LEN_MAX, align_up, batch_T, batch_coarse,  # noqa: E402
                         batch_events, batch_sort_key, load_train, make_eval_batches,
                         make_train_batches, pack)
from seq_common import make_folds  # noqa: E402
from tabm_common import BASE, cos_score  # noqa: E402


COARSE_T, COARSE_C = T.COARSE_T, T.COARSE_C
C_O, C_T = EF.C_O, EF.C_T
OUT_DIM, HEAD_DIM = T.OUT_DIM, T.HEAD_DIM
WIDTH_DEFAULT = T.WIDTH_DEFAULT
LEVELS_COARSE_DEFAULT = T.LEVELS_COARSE_DEFAULT
LEVELS_EVT_DEFAULT = 8        # RF 1023 ≥ 每流硬顶 999（RF(7)=511 盖不住长尾——那是"有一段永远看不到"）
K, STEM_K = T.K, T.STEM_K
TARGET_SCALE = T.TARGET_SCALE
SNAP_CACHE = T.SNAP_CACHE

# `ALIGN`（批内 T 对齐粒度）与 `EVT_LEN_MAX`（每流 999 的数据硬顶）已随打包层
# 移到 `seq/event_batch.py`，此处从那里 import（单一来源）。
DEFAULT_CKPT_DIR = os.path.join(SNAP_CACHE, "ckpt_tcn_evt4")
DEFAULT_OOF_DIR = "snap_cache/oof_parts_tcn_evt4"
# 归档参照：桶式细段 + cos 损失（v24，fold-0 cos = 0.14587）。**不是本次判据。**
REF_BUCKET_OOF_DIR = "snap_cache/oof_parts_tcn_cos"


def min_levels_for(t_in):
    """满足 RF ≥ t_in 的最小级数（与 `tcn_baseline.check_rf` 同一条 RF 公式）。"""
    lv = 1
    while T.receptive_field(lv) < t_in:
        lv += 1
        assert lv < 24, f"t_in={t_in} 超出 RF 公式的可达范围"
    return lv


# `align_up` / `pack` / `batch_T` / `batch_sort_key` / `make_train_batches` /
# `make_eval_batches` 六个打包件已随顶层 `seq/event_batch.py` 移出（见顶部 import）。


# ---------------------------------------------------------------- 检查点
# ⚠️ **不能复用 `tcn_baseline.load_ckpt`**——它拿 `tcn_baseline.arch_hash` 比对。

def arch_cfg(args):
    """参与指纹的配置。通道数、级数、**数据布局版本**都必须在内。"""
    cfg = {
        "series": "event3flat",
        "width": args.width, "levels": [args.levels_coarse, args.levels_evt],
        "k": K, "stem_k": STEM_K, "out_dim": OUT_DIM, "head_dim": HEAD_DIM,
        "drop": args.drop, "head_drop": args.head_drop,
        "coarse": [COARSE_T, COARSE_C], "evt_o": C_O, "evt_t": C_T,
        # 数据布局版本：**形状全变或存储方式变**都必须靠它进指纹，否则换版本会静默复用。
        "evt_layout": EF.LAYOUT_VERSION, "align": ALIGN,
        "seed": args.seed, "lr": args.lr, "wd": args.wd, "batch": args.batch,
    }
    # 判据是 `!= "mse"`（**历史编码规则**，别顺着默认值改）——见 tcn_baseline.arch_cfg 的长注释。
    if getattr(args, "loss", "cos") != "mse":
        cfg["loss"] = args.loss
    return cfg


def arch_hash(args):
    return hashlib.sha256(json.dumps(arch_cfg(args), sort_keys=True).encode()).hexdigest()[:16]


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


def resume_ckpt(ckpt_dir, fold, smoke=False):
    for kind in ("best", "last"):
        p = T.ckpt_path(ckpt_dir, fold, kind, smoke)
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------- 打包 / 分批
# ⚠️ 六个打包件（pack / batch_T / batch_sort_key / make_train_batches / make_eval_batches
#    + align_up）已于 2026-10-01 移到顶层 `seq/event_batch.py`（会被 tfm 等模型族 import）。
#    本谱系从那里 import（见顶部），函数逐字未改，行为由 `test_evt.py` 守着。


# ---------------------------------------------------------------- 数据
# `load_train` / `batch_coarse` / `batch_events` 已随打包层移到 `seq/event_batch.py`
#（同样从顶部 import）。三份输入的**行序一致性**与**零截断**断言在那边逐条保留。


# ---------------------------------------------------------------- 模型（容器）

class ThreeTowerTCNFlat(torch.nn.Module):
    """market | order 事件 | transaction 事件，三塔各自**末步读出**后在 head 处 concat。

    三条塔都是 `tcn_baseline.TCNTower`（**同一个类**，不是复制品）；
    事件塔的窗口**不是常量**（批内定长、批间变长），所以 `TCNTower` 的 `t_in` 传一个下界即可
    ——它只用来做 RF 断言，而 RF 只要盖住最长可能序列（两流硬顶 999，级数 7 → RF 511 覆盖 p99+）。
    """

    def __init__(self, width=WIDTH_DEFAULT, levels_coarse=LEVELS_COARSE_DEFAULT,
                 levels_evt=LEVELS_EVT_DEFAULT, rf_evt_window=EVT_LEN_MAX,
                 drop=0.2, head_drop=0.3, out=1, head_dim=HEAD_DIM, out_dim=OUT_DIM,
                 in_coarse=COARSE_C, in_o=C_O, in_t=C_T):
        super().__init__()
        self.tower_c = T.TCNTower(in_coarse, COARSE_T, width, levels_coarse, K, drop, out_dim)
        self.tower_o = T.TCNTower(in_o, rf_evt_window, width, levels_evt, K, drop, out_dim)
        self.tower_t = T.TCNTower(in_t, rf_evt_window, width, levels_evt, K, drop, out_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(3 * out_dim, head_dim), torch.nn.GELU(),
            torch.nn.Dropout(head_drop), torch.nn.Linear(head_dim, out),
        )

    def forward(self, x_c, nv, x_o, nko, x_t, nkt):
        """三塔都走**逐样本末步读出**；两条事件塔的 pad 在各自序列末尾（因果 ⇒ 对读出零影响）。"""
        h = torch.cat([self.tower_c(x_c, nv - 1),
                       self.tower_o(x_o, (nko - 1).clamp_min(0)),
                       self.tower_t(x_t, (nkt - 1).clamp_min(0))], dim=1)
        return self.head(h)


def build_model(args, device):
    return ThreeTowerTCNFlat(width=args.width, levels_coarse=args.levels_coarse,
                             levels_evt=args.levels_evt,
                             drop=args.drop, head_drop=args.head_drop).to(device)


# ---------------------------------------------------------------- 评估 / 训练（协议与 tcn_baseline 同）

def eval_forward(model, Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, rows, batch, device):
    """确定性分批前向 → (n,) f32（第 i 位对应 `rows[i]`）。

    分批是**显存护栏**（事件轴不降采样，长批的中间张量很大），**不改口径**——
    读出取逐样本 `n_kept−1`、eval 下 BN 用 running stats ⇒ 输出与批内 T 无关。
    `rows` 必须**升序**（写回位置靠 searchsorted）。
    """
    model.eval()
    assert np.all(np.diff(rows) > 0), "eval_forward 要求 rows 严格升序"
    out = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for sl in make_eval_batches(batch_sort_key(nk_o, nk_t), rows, batch):
            a, nvb = batch_coarse(Xc, nv, sl, device)
            xo, kob, xt, ktb = batch_events(Vo, offs_o, Vt, offs_t, nk_o, nk_t, sl, device)
            pred = model(a, nvb, xo, kob, xt, ktb).squeeze(-1).cpu().numpy()
            out[np.searchsorted(rows, sl)] = pred
    return out


def train_fold(Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, y, sids, tr_rows, va_rows,
               args, device, ckpt_dir, cfg_hash, fold, smoke):
    torch.manual_seed(args.seed)
    model = build_model(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = np.random.default_rng(args.seed)      # 与 torch.manual_seed 是两条独立随机流
    keep_last = args.ckpt_keep in ("last", "both")
    keep_best = args.ckpt_keep in ("best", "both")

    best_cos, best_epoch, best_va_pred, stall = -1.0, -1, None, 0
    ep = -1
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        batches = make_train_batches(batch_sort_key(nk_o, nk_t), tr_rows, args.batch, gen)
        loss_sum, n_batch, pnorm_sum = 0.0, 0, 0.0
        nb_tok = 0        # 真实事件槽（两条塔合计）
        nb_slot = 0       # **实际 padded 槽**（含批内 pad）——名义算力有没有兑现看它
        for sl in batches:
            a, nvb = batch_coarse(Xc, nv, sl, device)
            xo, kob, xt, ktb = batch_events(Vo, offs_o, Vt, offs_t, nk_o, nk_t, sl, device)
            yb = torch.from_numpy((y[sl] * TARGET_SCALE).astype(np.float32)).to(device)
            pred = model(a, nvb, xo, kob, xt, ktb).squeeze(-1)
            loss = T.batch_loss(pred, yb, args.loss)
            if args.loss == "cos":
                pnorm_sum += float(pred.abs().mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            n_batch += 1
            nb_tok += int(kob.sum() + ktb.sum())
            nb_slot += int(xo.shape[0] * xo.shape[2] + xt.shape[0] * xt.shape[2])   # B×T
        loss_mean = loss_sum / max(n_batch, 1)
        pnorm = pnorm_sum / max(n_batch, 1)
        pnorm_s = f" |p|={pnorm:.3f}" if args.loss == "cos" else ""

        pred_va = eval_forward(model, Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t,
                               va_rows, args.eval_batch, device)
        cos = cos_score(pred_va, (y[va_rows] * TARGET_SCALE).astype(np.float32))
        if cos > best_cos:
            best_cos, best_epoch, best_va_pred, stall = cos, ep, pred_va, 0
            if keep_best:
                T.save_ckpt(T.ckpt_path(ckpt_dir, fold, "best", smoke), model, cfg_hash,
                            {"fold": fold, "kind": "best", "epoch": ep, "cos": float(cos),
                             "smoke": bool(smoke)})
        else:
            stall += 1
            if stall >= args.patience:
                print(f"    ep={ep:3d} loss={loss_mean:.4f} val_cos={cos:.5f}"
                      f"{pnorm_s} t={time.time()-t0:5.1f}s  早停", flush=True)
                break
        if ep % 5 == 0:
            print(f"    ep={ep:3d} loss={loss_mean:.4f} val_cos={cos:.5f}{pnorm_s}"
                  f" 槽/批 真实{nb_tok/max(n_batch,1):.0f}+pad"
                  f"{nb_slot/max(n_batch,1) - nb_tok/max(n_batch,1):.0f}"
                  f"={nb_slot/max(n_batch,1):.0f} t={time.time()-t0:5.1f}s", flush=True)

    if keep_last:
        T.save_ckpt(T.ckpt_path(ckpt_dir, fold, "last", smoke), model, cfg_hash,
                    {"fold": fold, "kind": "last", "epoch": ep, "cos": float(cos),
                     "smoke": bool(smoke)})
    return {"cos": best_cos, "epoch": best_epoch, "va_pred": best_va_pred,
            "va_sids": sids[va_rows], "fold_secs": time.time() - t0}


def oof_from_ckpt(path, args, Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, va_rows, device):
    """`.pt` 在而 parquet 不在 → 复算 OOF（免重训）。与训练里选 best 用**同一个 eval_forward**。"""
    payload = load_ckpt(path, args)
    model = build_model(args, device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return eval_forward(model, Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, va_rows,
                        args.eval_batch, device), payload.get("meta", {})


def _fold_cos(path):
    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).select(["sample_id", "target"])
    d = pl.read_parquet(path).join(lab, on="sample_id", how="inner")
    pr, tg = d["pred"].to_numpy(), d["target"].to_numpy()
    return float((pr * tg).sum() / np.sqrt((pr ** 2).sum() * (tg ** 2).sum()))


def ref_fold_cos(fold):
    """归档的**桶式**细段 fold cos（v24 cos 臂）——只作参照，**不是本次判据**。"""
    p = os.path.join(BASE, REF_BUCKET_OOF_DIR, f"f{fold}.parquet")
    return _fold_cos(p) if os.path.exists(p) else None


def oof_summary(oof_dir):
    files = sorted(f for f in os.listdir(oof_dir) if f.startswith("f") and f.endswith(".parquet"))
    return files, np.array([_fold_cos(os.path.join(oof_dir, f)) for f in files])


def main():
    p = argparse.ArgumentParser(description="三塔TCN·扁平事件轴（market | order | transaction）")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--start-fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=10,
                   help="与 tcn_baseline 一致（用户 2026-09-25 指定降到 10）。不进指纹")
    p.add_argument("--batch", type=int, default=1024,
                   help="与既有各臂一致，**不要改**（CLAUDE.md #3）")
    p.add_argument("--eval-batch", type=int, default=4096,
                   help="评估/推理的分批行数（**显存护栏**，不改口径：读出与 T 无关）")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--head-drop", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss", choices=["mse", "cos"], default="cos")
    p.add_argument("--width", type=int, default=WIDTH_DEFAULT)
    p.add_argument("--levels-coarse", type=int, default=LEVELS_COARSE_DEFAULT)
    p.add_argument("--levels-evt", type=int, default=LEVELS_EVT_DEFAULT,
                   help=f"两条事件塔的空洞级数（默认 {LEVELS_EVT_DEFAULT} → RF "
                        f"{T.receptive_field(LEVELS_EVT_DEFAULT)}）")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--ckpt-keep", choices=["best", "last", "both"], default="both")
    p.add_argument("--oof-dir", default=DEFAULT_OOF_DIR)
    p.add_argument("--only-fold", type=int, default=-1)
    p.add_argument("--smoke-fold", action="store_true")
    p.add_argument("--time-smoke", action="store_true")
    p.add_argument("--stats", action="store_true", help="只打印两条流事件数分布并退出")
    args = p.parse_args()

    if args.stats:
        (o, t), _ = EF.event_counts("train")
        for tag, v in (("order", o), ("txn", t)):
            print(f"[train] {tag}: mean={v.mean():.1f} 中位={int(np.median(v))}", end="")
            for q in (90, 99):
                print(f" p{q}={int(np.percentile(v, q))}", end="")
            print(f" max={int(v.max())}")
        for q in (50, 90, 99):
            print(f"  pad 期望（中位口径）: q{q} = {int(np.percentile(o, q))}/{int(np.percentile(t, q))}")
        return

    rf_c = T.receptive_field(args.levels_coarse)
    rf_e = T.receptive_field(args.levels_evt)
    assert rf_c >= COARSE_T, f"粗段级数 {args.levels_coarse} → RF {rf_c} < {COARSE_T}"
    assert rf_e >= 1000, \
        f"事件塔级数 {args.levels_evt} → RF {rf_e} < 1000（盖不住每流 999 的硬顶）"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.ckpt_dir = args.ckpt_dir if os.path.isabs(args.ckpt_dir) \
        else (os.path.join(BASE, args.ckpt_dir) if args.ckpt_dir else "")
    oof_dir = args.oof_dir if os.path.isabs(args.oof_dir) else os.path.join(BASE, args.oof_dir)
    os.makedirs(oof_dir, exist_ok=True)
    cfg_hash = arch_hash(args) if args.ckpt_dir else None

    print(f"device={device} batch={args.batch} eval_batch={args.eval_batch} lr={args.lr} "
          f"wd={args.wd} drop={args.drop}/{args.head_drop} seed={args.seed} loss={args.loss} "
          f"patience={args.patience}")
    print(f"三塔TCN·扁平事件轴：market ({COARSE_C},{COARSE_T}) → {args.levels_coarse} 级（RF {rf_c}）；"
          f"order ({C_O},变长) / txn ({C_T},变长) → {args.levels_evt} 级（RF {rf_e}）")
    print(f"读出：三塔各自逐样本末步（`n_valid−1` / `n_kept−1`）→ {3*OUT_DIM} → head")
    print("主干：CausalConv / TCNBlock / TCNTower / batch_loss **全部 import 自 tcn_baseline**（未改一行）")
    print(f"分批：按事件数排序 + ±10% 抖动；批内 pad 到 max(nk) 并对齐 {ALIGN} 的倍数")
    n_par = sum(q.numel() for q in build_model(args, "cpu").parameters())
    print(f"参数量 {n_par:,}；cfg 指纹={cfg_hash or '-'}；布局版本={EF.LAYOUT_VERSION}")
    print(f"OOF → {oof_dir}；检查点 → {args.ckpt_dir or '（不存）'}"
          + (f"   **只跑 fold {args.only_fold}**" if args.only_fold >= 0 else ""))

    t0 = time.time()
    (Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, y, marr, months, sids) = load_train()
    print(f"loaded: 粗段 {Xc.shape} + order {Vo.shape}（offs {offs_o.shape}，中位 "
          f"{int(np.median(nk_o))}）+ txn {Vt.shape}（中位 {int(np.median(nk_t))}）"
          f"  {time.time()-t0:.1f}s", flush=True)

    folds = make_folds(months, args.folds)
    results = []
    for fi, va_months in enumerate(folds):
        if fi < args.start_fold:
            continue
        if args.only_fold >= 0 and fi != args.only_fold:
            continue
        part_path = os.path.join(oof_dir, f"f{fi}.parquet")
        if (args.only_fold >= 0 or args.smoke_fold) and os.path.exists(part_path):
            raise SystemExit(f"[探针跑] 目标 OOF 已存在，拒绝覆盖：{part_path}\n  请另给 --oof-dir")

        smoke_naming = args.smoke_fold or args.time_smoke
        resume_path = resume_ckpt(args.ckpt_dir, fi, smoke_naming) if args.ckpt_dir else None
        va_mask = np.isin(marr, va_months)
        tr_rows = np.where(~va_mask)[0]
        va_rows = np.where(va_mask)[0]

        if resume_path is not None and not smoke_naming:
            try:
                pred, meta = oof_from_ckpt(resume_path, args, Xc, nv, Vo, offs_o, nk_o,
                                           Vt, offs_t, nk_t, va_rows, device)
            except ValueError as e:
                raise SystemExit(f"[fold {fi}] 检查点存在但配置不符，拒绝覆盖：\n{e}")
            if not os.path.exists(part_path):
                pl.DataFrame({"sample_id": sids[va_rows], "month": marr[va_rows],
                              "pred": pred.astype(np.float64)}).write_parquet(part_path)
                print(f"[fold {fi}] 复用检查点补出 OOF（{part_path}）", flush=True)
            else:
                print(f"[fold {fi}] 检查点与 OOF 都在，跳过", flush=True)
            continue

        print(f"[fold {fi + 1}/{len(folds)}] 验证月={va_months} "
              f"训练={len(tr_rows)} 验证={len(va_rows)}", flush=True)
        epochs = 1 if args.smoke_fold else args.epochs
        if args.time_smoke:
            epochs = 2
        t1 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        r = train_fold(Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, y, sids, tr_rows, va_rows,
                       argparse.Namespace(**{**vars(args), "epochs": epochs}), device,
                       args.ckpt_dir, cfg_hash, fi, smoke_naming)
        r["fold_secs"] = time.time() - t1
        results.append(r)

        if args.time_smoke:
            per_ep = r["fold_secs"] / epochs
            peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
            est_ep = 10 + args.patience
            print(f"\n### 时间冒烟 ###\n  {per_ep:.1f} s/epoch（{epochs} epoch 共 {r['fold_secs']:.0f}s）"
                  f"\n  显存峰值 {peak:.1f} GiB\n"
                  f"  投影 6 折 ≈ {per_ep * est_ep * 6 / 3600:.2f} h（按每折 {est_ep} epoch 估）")
            return

        if args.smoke_fold:
            print(f"  → cos={r['cos']:.5f} best_epoch={r['epoch']} 耗时 {r['fold_secs']:.0f}s"
                  f"（冒烟：写 _smoke 名，不落 OOF）", flush=True)
            continue

        pl.DataFrame({"sample_id": r["va_sids"], "month": marr[va_rows],
                      "pred": r["va_pred"].astype(np.float64)}).write_parquet(part_path)
        rc = ref_fold_cos(fi)
        ref_s = f"   归档桶式折叠 cos={rc:.5f}  Δ={r['cos']-rc:+.5f}" if rc is not None else ""
        print(f"  → cos={r['cos']:.5f} best_epoch={r['epoch']} 耗时 {r['fold_secs']:.0f}s"
              f"（已落盘 {part_path}）{ref_s}", flush=True)

    if args.smoke_fold or len(results) <= 0:
        print("\n（冒烟模式：不汇总）")
        return

    files, cos_arr = oof_summary(oof_dir)
    print(f"\n### 三塔TCN·扁平事件轴 逐折 cos（{len(cos_arr)} 折，月交错）###")
    for fn, c in zip(files, cos_arr):
        print(f"  {fn}: cos={c:.5f}")
    print(f"mean={cos_arr.mean():.5f} std={cos_arr.std():.5f} min={cos_arr.min():.5f}")
    for fi in range(len(cos_arr)):
        rc = ref_fold_cos(fi)
        if rc is not None:
            print(f"  归档桶式 fold {fi}: {rc:.5f} → Δ={cos_arr[fi]-rc:+.5f}")
    print("\n结构类改动必须预留一次 LB 验证（CLAUDE.md #13）——本轮不预注册阈值。")

    if len(results) > 1:
        parts = [pl.read_parquet(os.path.join(oof_dir, f))
                 for f in sorted(os.listdir(oof_dir)) if f.endswith(".parquet")]
        pl.concat(parts).write_parquet(os.path.join(oof_dir, "oof_all.parquet"))
        print(f"\nOOF 已存 {os.path.join(oof_dir, 'oof_all.parquet')}")


if __name__ == "__main__":
    main()
