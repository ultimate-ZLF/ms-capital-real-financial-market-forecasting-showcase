"""tfm_baseline.py — **三塔Transformer·扁平事件轴**：market | order 事件 | transaction 事件。

C-19 的 **B 路（注意力 + 事件序列）**：主干从因果卷积换成**浅层 Transformer**
（`tfm_model.TFTower`），其余**逐字沿用** `models/tcn/evt_baseline.py` 的三塔数据口径与训练协议。

## 与三塔TCN·扁平事件轴（B-06 的最后一轮）的差别

| | 三塔TCN·扁平（归档）| **本臂** |
|---|---|---|
| 时间混合 | 空洞因果卷积（RF 1023，只向前看）| **多头自注意力（全窗双向 + pad-mask）** |
| 时间对齐 | 无位置编码，靠 `dt`/`t_pos` 内容 | 同（**刻意不加位置编码**，见 `tfm_model` 模块头）|
| 数据 / 分塔 / 读出 / 损失 / 折式 / 超参 | —— **完全相同** —— | |
| 批内 pad | 因果卷积天然免疫 | **显式 mask**（写反不报错，测试钉死）|

主干的推理在 `tfm_model.py`；本文件只管数据、协议与产物。

## 产物（全部在 `$MSC_BASE` 下，**新名字、不覆盖任何既有文件**）

| 产物 | 路径 |
|---|---|
| 检查点 | `snap_cache/ckpt_tfm_evt/f{fi}.{best,last}.pt`（冒烟 → `f{fi}_smoke.*`）|
| 逐折 OOF | `snap_cache/oof_parts_tfm_evt/f{fi}.parquet` |

## 判据

`C-19` 登记的判据：**fold-0 对 0.14587**（桶式 TCN-cos 的 fold-0，`snap_cache/oof_parts_tcn_cos`）。
同形态参照（**不是判据**）：三塔TCN·扁平事件轴 fold-0 = 0.12946（`oof_parts_tcn_evt4`）。
两个参照的 fold-0 都会在跑完后**从盘上现算并打印**（认目录靠数字、不靠目录名——
2026-09-28 认错过一次基线，见 MEMORY「细段特征工程」）。
⚠️ `CLAUDE.md #13`：结构类改动 CV 常选不出来，**要预留一次 LB 验证**。

用法：
    python models/tfm/tfm_baseline.py --stats              # 看两条流事件数分布
    python models/tfm/tfm_baseline.py --only-fold 0        # 单折探针（本轮的用法）
    python models/tfm/tfm_baseline.py --folds 6            # 正式 6 折

⚠️ **内存**：扁平事件缓存现建到内存（order 2.4GB + txn 1.3GB ≈ **3.6GB**，构建峰值约 7GB）
⇒ 只能在**有卡模式**跑（无卡模式 memory.max = 2GiB 会被静默 OOM，见 CLAUDE.md #16）。
⚠️ 本脚本不用 DataLoader：打包是纯内存向量化（~3ms/批），顺带免掉 fork/polars 与 fd 泄漏两类坑。
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
sys.path.insert(0, os.path.join(MODELS, "tabm"))                   # tabm_common（BASE / cos_score）

import event_batch as EB  # noqa: E402  扁平事件轴的打包/分批（多模型族共用）
import event_feats as EF  # noqa: E402
import tfm_model as M  # noqa: E402      ★ 主干（本目录自包含）
from seq_common import F as COARSE_C_BASE  # noqa: E402
from seq_common import T as COARSE_T_BASE  # noqa: E402
from seq_common import make_folds  # noqa: E402
from tabm_common import BASE, cos_score  # noqa: E402


COARSE_T, COARSE_C = COARSE_T_BASE, COARSE_C_BASE
C_O, C_T = EF.C_O, EF.C_T
OUT_DIM, HEAD_DIM = M.OUT_DIM, M.HEAD_DIM
WIDTH_DEFAULT = M.WIDTH_DEFAULT
LAYERS_DEFAULT = M.LAYERS_DEFAULT
HEADS_DEFAULT = M.HEADS_DEFAULT
FF_MULT_DEFAULT = M.FF_MULT_DEFAULT
BUDGET_MB_DEFAULT = M.BUDGET_MB_DEFAULT
TARGET_SCALE = 1000.0
SNAP_CACHE = os.path.join(BASE, "snap_cache")

DEFAULT_CKPT_DIR = os.path.join(SNAP_CACHE, "ckpt_tfm_evt")
DEFAULT_OOF_DIR = "snap_cache/oof_parts_tfm_evt"

# 归档参照（**只作参照，不是判据**；fold-0 会从盘上现算，认数字不认目录名）
REF_DIRS = [
    ("桶式TCN-cos（v24；C-19 判据 = 0.14587）", "snap_cache/oof_parts_tcn_cos"),
    ("三塔TCN·扁平事件轴（B-06 末轮 = 0.12946）", "snap_cache/oof_parts_tcn_evt4"),
]


# ---------------------------------------------------------------- 检查点
# 与 tcn_baseline / mlp_baseline / evt_baseline 的同名函数是**同一套**（照抄一份，不互相 import）：
# 只存 CPU state_dict + 配置指纹，读侧**必须**断言指纹。四条纪律见 CLAUDE.md #12。

def arch_cfg(args):
    """参与指纹的配置。通道数、层数、**数据布局版本**都必须在内。"""
    cfg = {
        "series": "tfm3evtflat",
        "width": args.width, "layers": args.layers, "heads": args.heads,
        "ff_mult": args.ff_mult,
        "out_dim": OUT_DIM, "head_dim": HEAD_DIM,
        "drop": args.drop, "head_drop": args.head_drop,
        "coarse": [COARSE_T, COARSE_C], "evt_o": C_O, "evt_t": C_T,
        # 数据布局版本：**形状全变或存储方式变**都必须靠它进指纹，否则换版本会静默复用。
        "evt_layout": EF.LAYOUT_VERSION, "align": EB.ALIGN,
        "seed": args.seed, "lr": args.lr, "wd": args.wd, "batch": args.batch,
    }
    # 细段 Time2Vec（叠加式）：**条件性**进指纹——关（默认）时完全不加字段，否则已归档的
    # `ckpt_tfm_evt/*.pt`（fold-0 = 0.14959）会被**误拒**（CLAUDE.md #12 的反向纪律）。
    # 开时改变输入通道数（去 dt）与学出的权重 ⇒ 必须进指纹。
    if getattr(args, "t2v", False):
        cfg["t2v"] = "add"
    # 跨源注意力融合：**条件性**进指纹（关时不加字段，历史检查点不受影响）。
    # 开时换模型类（`ThreeTowerTFMXAttn`）、换 head（3×out → out）⇒ 必须进指纹。
    if getattr(args, "xattn", False):
        cfg["xattn"] = "prefix1"     # 语义：z 前缀 token 法；改设计要换这个标记 + 换目录
    # 判据是 `!= "mse"`（**历史编码规则**，别顺着默认值改）——见 tcn_baseline.arch_cfg 的长注释。
    if getattr(args, "loss", "cos") != "mse":
        cfg["loss"] = args.loss
    # ⚠️ 刻意**不进指纹**：`budget_mb`（只改浮点切分，不改语义——与 eval_chunk 同类）、
    #    epochs / patience（只改跑多久）、oof_dir / ckpt_dir（工程件）。
    return cfg


def arch_hash(args):
    return hashlib.sha256(json.dumps(arch_cfg(args), sort_keys=True).encode()).hexdigest()[:16]


def ckpt_path(ckpt_dir, fold, kind="best", smoke=False):
    """kind ∈ {best, last}；冒烟跑写 `f{fi}_smoke.*`，**绝不碰正式名**。"""
    tag = f"f{fold}_smoke" if smoke else f"f{fold}"
    return os.path.join(ckpt_dir, f"{tag}.{kind}.pt")


def save_ckpt(path, model, cfg_hash, meta):
    """原子写：先写 `path + ".tmp"` 再 os.replace（`torch.save` **不追加扩展名**，CLAUDE.md #1）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
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


def resume_ckpt(ckpt_dir, fold, smoke=False):
    for kind in ("best", "last"):
        p = ckpt_path(ckpt_dir, fold, kind, smoke)
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------- 模型 / 损失

def build_model(args, device):
    """按 args 建模型并搬到 device。训练/推理/OOF 复算**共用这一处**（口径只有一份）。

    `--t2v` 时给**两条事件塔**（细段）各挂一个 Time2Vec（输出维 = width，**叠加**到 token
    embedding 上），τ 取各自流的 `dt_log`（下标来自 `event_feats.IX_O/IX_T`，单一来源——
    不写死 0），**该通道同时从输入里去掉**。
    `--xattn` 时改用 `ThreeTowerTFMXAttn`（跨源注意力融合、head 只走融合通路）。
    """
    kw = dict(width=args.width, layers=args.layers, heads=args.heads,
              ff_mult=args.ff_mult, drop=args.drop, head_drop=args.head_drop,
              budget_mb=args.attn_budget_mb, t2v_add=args.t2v,
              dt_index_o=EF.IX_O["dt_log"], dt_index_t=EF.IX_T["dt_log"])
    cls = M.ThreeTowerTFMXAttn if args.xattn else M.ThreeTowerTFM
    return cls(**kw).to(device)


def batch_loss(pred, yb, loss_kind="mse"):
    """训练损失（**照抄自 `tcn_baseline.batch_loss`**，同一口径，不互相 import）。

    - **`cos`（本谱系默认）**：`1 − cos(p_batch, y_batch)`，**不中心化**（HYPOTHESES C-17 的口径）。
      全局 cos 的梯度 `∂cos/∂pᵢ ∝ yᵢ − c·pᵢ`（`c = Σpy/Σp²`）**正是** MSE 的负梯度、
      逐 batch 等价于"目标缩放 c 倍"——c 是**每批自适应**的尺度（Adam 做不了这个）。
      不中心化：指标是未中心化的 cos；附带好处是本损失会自己把预测均值压向 0。
      ⚠️ 用户 2026-09-23 明确**不加分母下限 ε**；分母恰好为 0 时报错（真异常）。
    - **`mse`**：`F.mse_loss`，与原协议逐位一致。
    """
    if loss_kind == "cos":
        den = pred.norm() * yb.norm()
        if not torch.isfinite(den) or float(den) == 0.0:
            raise RuntimeError(
                f"[loss=cos] 分母退化：‖p‖={float(pred.norm()):.3g} ‖y‖={float(yb.norm()):.3g}")
        return 1.0 - (pred * yb).sum() / den
    return F.mse_loss(pred, yb)


# ---------------------------------------------------------------- 评估 / 训练

def eval_forward(model, Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, rows, batch, device):
    """确定性分批前向 → (n,) f32（第 i 位对应 `rows[i]`）。

    分批是**显存护栏**（事件轴不降采样，长批的张量很大），**不改口径**——读出取逐样本
    `n−1`、eval 下 Dropout 关闭、LayerNorm 逐样本 ⇒ 输出与批内组成无关（浮点级）。
    `rows` 必须**升序**（写回位置靠 searchsorted）。
    """
    model.eval()
    assert np.all(np.diff(rows) > 0), "eval_forward 要求 rows 严格升序"
    out = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for sl in EB.make_eval_batches(EB.batch_sort_key(nk_o, nk_t), rows, batch):
            a, nvb = EB.batch_coarse(Xc, nv, sl, device)
            xo, kob, xt, ktb = EB.batch_events(Vo, offs_o, Vt, offs_t, nk_o, nk_t, sl, device)
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
        batches = EB.make_train_batches(EB.batch_sort_key(nk_o, nk_t), tr_rows, args.batch, gen)
        loss_sum, n_batch, pnorm_sum = 0.0, 0, 0.0
        nb_tok = 0        # 真实事件槽（两条塔合计）
        nb_slot = 0       # **实际 padded 槽**（含批内 pad）——名义算力有没有兑现看它
        for sl in batches:
            a, nvb = EB.batch_coarse(Xc, nv, sl, device)
            xo, kob, xt, ktb = EB.batch_events(Vo, offs_o, Vt, offs_t, nk_o, nk_t, sl, device)
            yb = torch.from_numpy((y[sl] * TARGET_SCALE).astype(np.float32)).to(device)
            pred = model(a, nvb, xo, kob, xt, ktb).squeeze(-1)
            loss = batch_loss(pred, yb, args.loss)
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
            if keep_best and ckpt_dir:      # ckpt_dir 空串 = 不存（不再往 CWD 里写 f0.best.pt）
                save_ckpt(ckpt_path(ckpt_dir, fold, "best", smoke), model, cfg_hash,
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

    if keep_last and ckpt_dir:
        save_ckpt(ckpt_path(ckpt_dir, fold, "last", smoke), model, cfg_hash,
                  {"fold": fold, "kind": "last", "epoch": ep, "cos": float(cos),
                   "smoke": bool(smoke)})
    return {"cos": best_cos, "epoch": best_epoch, "va_pred": best_va_pred,
            "va_sids": sids[va_rows], "fold_secs": time.time() - t0}


def train_full(Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, y, args, device, ckpt_dir,
               cfg_hash, ckpt_every):
    """**全量训练**（用户 2026-10-02 定）：71 个月全上，固定 `args.epochs` 轮。

    ⚠️ 与折式训练的三处**结构性差别**（都是有意的）：
      1. **没有验证集** ⇒ 没有早停、没有"best epoch"选择——轮数必须**事先定死**；
         依据是 fold-0 的观测峰值（ep36 ⇒ 看过 37.6M 样本）+ 全量多 20% 数据/轮的换算，
         取"步数对齐(30) 与轮数对齐(36) 之间偏后"的 35（见 README「全量训练」节）。
      2. **不写 OOF**（没有留出月份可预测）。
      3. 检查和**每 `ckpt_every` 轮 + 末轮**落盘（`full_e{e}.pt`），命名不含折号——
         选哪个进提交是**事先声明的**（用哪一轮读哪个文件），不是事后挑。
      训练损失口径、batch、种子、优化器与折式臂**逐字相同**（可比）。
    """
    torch.manual_seed(args.seed)
    model = build_model(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = np.random.default_rng(args.seed)
    tr_rows = np.arange(len(y))          # **全部**样本
    t0 = time.time()
    losses = []
    for ep in range(args.epochs):
        model.train()
        batches = EB.make_train_batches(EB.batch_sort_key(nk_o, nk_t), tr_rows, args.batch, gen)
        loss_sum, n_batch, pnorm_sum = 0.0, 0, 0.0
        for sl in batches:
            a, nvb = EB.batch_coarse(Xc, nv, sl, device)
            xo, kob, xt, ktb = EB.batch_events(Vo, offs_o, Vt, offs_t, nk_o, nk_t, sl, device)
            yb = torch.from_numpy((y[sl] * TARGET_SCALE).astype(np.float32)).to(device)
            pred = model(a, nvb, xo, kob, xt, ktb).squeeze(-1)
            loss = batch_loss(pred, yb, args.loss)
            if args.loss == "cos":
                pnorm_sum += float(pred.abs().mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            n_batch += 1
        loss_mean = loss_sum / max(n_batch, 1)
        losses.append(loss_mean)
        pnorm_s = f" |p|={pnorm_sum/max(n_batch,1):.3f}" if args.loss == "cos" else ""
        last = (ep == args.epochs - 1)
        if (ep + 1) % ckpt_every == 0 or last:
            save_ckpt(os.path.join(ckpt_dir, f"full_e{ep + 1}.pt"), model, cfg_hash,
                      {"full_train": True, "epoch": ep, "loss": loss_mean,
                       "n_rows": int(len(y)), "seed": args.seed})
            print(f"    ep={ep:3d} loss={loss_mean:.4f}{pnorm_s} t={time.time()-t0:5.1f}s"
                  f"  → full_e{ep + 1}.pt", flush=True)
        elif ep % 5 == 0:
            print(f"    ep={ep:3d} loss={loss_mean:.4f}{pnorm_s} t={time.time()-t0:5.1f}s", flush=True)
    return {"loss_first": losses[0], "loss_last": losses[-1], "secs": time.time() - t0,
            "n_rows": int(len(y)), "epochs": args.epochs}


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
    """归档参照的 fold cos——**从盘上现算**（认数字不认目录名，2026-09-28 的教训）。"""
    out = []
    for tag, rel in REF_DIRS:
        p = os.path.join(BASE, rel, f"f{fold}.parquet")
        out.append((tag, _fold_cos(p) if os.path.exists(p) else None))
    return out


def oof_summary(oof_dir):
    files = sorted(f for f in os.listdir(oof_dir) if f.startswith("f") and f.endswith(".parquet"))
    return files, np.array([_fold_cos(os.path.join(oof_dir, f)) for f in files])


def main():
    p = argparse.ArgumentParser(
        description="三塔Transformer·扁平事件轴（market | order | txn，浅层注意力主干）")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--start-fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=10,
                   help="与 tcn_baseline / evt_baseline 一致（用户 2026-09-25 指定降到 10）。不进指纹")
    p.add_argument("--batch", type=int, default=1024,
                   help="与既有各臂一致，**不要改**（CLAUDE.md #3）")
    p.add_argument("--eval-batch", type=int, default=2048,
                   help="评估/推理的分批行数（**显存护栏**，不改口径：读出与 T 无关）")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--head-drop", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss", choices=["mse", "cos"], default="cos",
                   help="训练损失。**默认 cos**（用户 2026-09-26 指令：之后所有模型都用 cos）。"
                        "⚠️ 只在 ≠mse 时才进配置指纹 → 换损失必须另给 --ckpt-dir / --oof-dir")
    # —— 主干（浅层 Transformer）——
    p.add_argument("--width", type=int, default=WIDTH_DEFAULT, help="d_model")
    p.add_argument("--layers", type=int, default=LAYERS_DEFAULT,
                   help=f"每个塔的编码块数（**浅层**，默认 {LAYERS_DEFAULT}）。"
                        f"注意力没有 RF 限制——全窗在第 1 层就可达，层数只加非线性深度")
    p.add_argument("--heads", type=int, default=HEADS_DEFAULT)
    p.add_argument("--ff-mult", type=int, default=FF_MULT_DEFAULT, help="MLP 隐层 = ff_mult×width")
    p.add_argument("--attn-budget-mb", type=int, default=BUDGET_MB_DEFAULT,
                   help="每个子批的注意力矩阵预算（MiB）→ 子批大小 = budget/(H·L²·4)。"
                        "**只改浮点切分、不改语义**（LayerNorm/注意力/MLP 全逐样本），不进指纹")
    p.add_argument("--xattn", action="store_true",
                   help="**跨源注意力融合**臂（用户 2026-10-02 定）：阶段1 用 market 塔读出态当 Query、"
                        "order 全部 token 当 K/V → z（dynamic latent，只在读出点做一次）；"
                        "阶段2 把 z 作**前缀 token** 喂进 txn 塔；**head 只走融合通路**"
                        "（三塔读出不再 concat，market/order 的 proj 不建 ⇒ 参数更少）。"
                        "⚠️ 换模型类 ⇒ **进指纹**，必须另给 --ckpt-dir / --oof-dir")
    p.add_argument("--t2v", action="store_true",
                   help="**细段**（order/txn 两条事件塔）开 Time2Vec：τ = 该流的 `dt_log`，"
                        "编码**叠加**到 token embedding 上（`h = stem(x) + t2v(τ)`，输出维 = width，"
                        "用户 2026-10-02 定），同时**去掉原 dt 通道**"
                        "（用户 2026-10-01：\"有 time2vec 就不要 dt\"）。"
                        "ω/φ 可学习 ⇒ 只能住在模型里、不能预先算进缓存。"
                        "⚠️ 改变输入通道数与学出的权重 ⇒ **进指纹**，必须另给 --ckpt-dir / --oof-dir")
    # —— 检查点 / 产物 ——
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--ckpt-keep", choices=["best", "last", "both"], default="both")
    p.add_argument("--oof-dir", default=DEFAULT_OOF_DIR)
    p.add_argument("--full-train", action="store_true",
                   help="**全量训练**（用户 2026-10-02）：71 个月全上、固定 --epochs 轮、"
                        "**无验证集 ⇒ 无早停/无 best 选择/不写 OOF**；每 --ckpt-every 轮 + 末轮落"
                        "`full_e{e}.pt`。⚠️ 轮数必须事先定死（见 README「全量训练」节）")
    p.add_argument("--ckpt-every", type=int, default=5,
                   help="全量训练时每隔几轮落一个检查点（末轮必落）")
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
        return

    if args.heads and args.width % args.heads != 0:
        raise SystemExit(f"--width {args.width} 不能被 --heads {args.heads} 整除")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.ckpt_dir = args.ckpt_dir if os.path.isabs(args.ckpt_dir) \
        else (os.path.join(BASE, args.ckpt_dir) if args.ckpt_dir else "")
    oof_dir = args.oof_dir if os.path.isabs(args.oof_dir) else os.path.join(BASE, args.oof_dir)
    os.makedirs(oof_dir, exist_ok=True)
    cfg_hash = arch_hash(args) if args.ckpt_dir else None

    print(f"device={device} batch={args.batch} eval_batch={args.eval_batch} lr={args.lr} "
          f"wd={args.wd} drop={args.drop}/{args.head_drop} seed={args.seed} loss={args.loss} "
          f"patience={args.patience}")
    print(f"三塔Transformer：market ({COARSE_C},{COARSE_T}) / order ({C_O},变长) / "
          f"txn ({C_T},变长) —— 每塔 {args.layers} 块 × width {args.width} × {args.heads} 头"
          f"（ff×{args.ff_mult}）")
    if args.t2v:
        print(f"细段 Time2Vec（**叠加式**）：τ=dt_log → {args.width} 维编码加到 token embedding 上"
              f"（`h = stem(x) + t2v(τ)`）；输入去掉 dt 通道：order {C_O}→{C_O-1}、"
              f"txn {C_T}→{C_T-1}；粗段塔不加")
    if args.xattn:
        print("跨源注意力融合（**前缀 token 法**）：阶段1 z = CrossAttn(q=market读出态@nv−1, "
              "K/V=order 块输出，单次)；阶段2 输入 = [z] 前置 + txn token；"
              "head 只收 txn 读出（market/order 的 proj 不建）")
    print(f"读出：三塔各自逐样本末步（`n_valid−1` / `nk−1`）→ {3*OUT_DIM} → head；"
          f"pad 由 **attn mask** 排除（True=参与）")
    print(f"注意（与 TCN 的区别）：注意力**全窗双向**、**无位置编码**（时间信息全在内容通道里）；"
          f"子批预算 {args.attn_budget_mb} MiB（训练时子批走 checkpoint）")
    n_par = M.n_params(build_model(args, "cpu"))
    print(f"参数量 {n_par:,}；cfg 指纹={cfg_hash or '-'}；布局版本={EF.LAYOUT_VERSION}")
    print(f"OOF → {oof_dir}；检查点 → {args.ckpt_dir or '（不存）'}"
          + (f"   **只跑 fold {args.only_fold}**" if args.only_fold >= 0 else ""))
    print("判据（C-19）：fold-0 对 **0.14587**（桶式 TCN-cos）。结构类改动要预留一次 LB 验证"
          "（CLAUDE.md #13）")

    t0 = time.time()
    (Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, y, marr, months, sids) = EB.load_train()
    print(f"loaded: 粗段 {Xc.shape} + order {Vo.shape}（offs {offs_o.shape}，中位 "
          f"{int(np.median(nk_o))}）+ txn {Vt.shape}（中位 {int(np.median(nk_t))}）"
          f"  {time.time()-t0:.1f}s", flush=True)

    if args.full_train:
        if not args.ckpt_dir:
            raise SystemExit("--full-train 需要 --ckpt-dir（没有验证集 ⇒ 检查点是唯一产物）")
        print(f"\n### 全量训练 ###\n  全部 {len(y)} 行（71 月）、固定 {args.epochs} 轮、"
              f"无验证/无早停/无 OOF；每 {args.ckpt_every} 轮落检查点 → {args.ckpt_dir}/full_e*.pt")
        print("  轮数依据：fold-0 峰值 ep36（× 1.045M = 37.6M 样本）⇒ 全量步数对齐 30 / "
              "轮数对齐 36 ⇒ 取 35", flush=True)
        r = train_full(Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, y, args, device,
                       args.ckpt_dir, cfg_hash, args.ckpt_every)
        print(f"\n全量训练完成：{r['epochs']} 轮、{r['n_rows']} 行、{r['secs']/60:.1f} min；"
              f"loss {r['loss_first']:.4f} → {r['loss_last']:.4f}")
        print("⚠️ 无验证集 ⇒ 本模式**没有 CV 数字**；检查点选择靠事先声明的轮数。")
        return

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
                  f"  投影 6 折 ≈ {per_ep * est_ep * 6 / 3600:.2f} h（按每折 {est_ep} epoch 估）\n"
                  f"  ⚠️ 峰值远低于 B×H×L²×4 的理论值时，说明 SDPA 走了 mem-efficient 内核，"
                  f"可把 --attn-budget-mb 调大换速度", flush=True)
            return

        if args.smoke_fold:
            print(f"  → cos={r['cos']:.5f} best_epoch={r['epoch']} 耗时 {r['fold_secs']:.0f}s"
                  f"（冒烟：写 _smoke 名，不落 OOF）", flush=True)
            continue

        pl.DataFrame({"sample_id": r["va_sids"], "month": marr[va_rows],
                      "pred": r["va_pred"].astype(np.float64)}).write_parquet(part_path)
        refs = ref_fold_cos(fi)
        ref_s = "".join(f"\n    {tag}: {c:.5f}  Δ={r['cos']-c:+.5f}"
                        for tag, c in refs if c is not None)
        print(f"  → cos={r['cos']:.5f} best_epoch={r['epoch']} 耗时 {r['fold_secs']:.0f}s"
              f"（已落盘 {part_path}）{ref_s}", flush=True)

    if args.smoke_fold or len(results) <= 0:
        print("\n（冒烟模式：不汇总）")
        return

    files, cos_arr = oof_summary(oof_dir)
    print(f"\n### 三塔Transformer·扁平事件轴 逐折 cos（{len(cos_arr)} 折，月交错）###")
    for fn, c in zip(files, cos_arr):
        print(f"  {fn}: cos={c:.5f}")
    print(f"mean={cos_arr.mean():.5f} std={cos_arr.std():.5f} min={cos_arr.min():.5f}")
    for fi in range(len(cos_arr)):
        for tag, c in ref_fold_cos(fi):
            if c is not None:
                print(f"  参照 fold {fi}｜{tag}: {c:.5f} → Δ={cos_arr[fi]-c:+.5f}")
    print("\n结构类改动必须预留一次 LB 验证（CLAUDE.md #13）——本轮不预注册阈值。")

    parts = [pl.read_parquet(os.path.join(oof_dir, f)) for f in files]   # 只收 f*.parquet
    if len(parts) > 1:
        oof_path = os.path.join(oof_dir, "oof_all.parquet")
        pl.concat(parts).write_parquet(oof_path)
        print(f"\nOOF 已存 {oof_path}（{sum(p.height for p in parts)} 行）")


if __name__ == "__main__":
    main()
