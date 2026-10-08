"""snap-CNN：纯盘口快照序列（15 通道 × T=224）回归，**月交错 6 折 CV**。

折划分（2026-09-15 起，替代旧的"12 折各留一个月"）：
    折 k 的验证月 = months[k::6]，训练 = 其余月份
    6 折合计覆盖全部 71 个月，各折验证月互不重叠 → 每折约 21.2 万验证样本
    （旧的 months[2::6] 12 折是同一 blocking 的 k=2 余数类，每折只有 1.8 万样本）
    理由：折数减半、验证样本 ×12、CV 均值精度提升约 2.4 倍，且保住"验证月完全没参与训练"
    （随机按样本分折会让同月样本混入训练 → 月份 regime 泄漏，CV 会偏乐观）。

坑清单（新会话避雷）：
1. 排序方向：缓存 t=0 最旧（snap_feats 有断言）；pad 在时间轴末尾，特征全 0
2. masked pool：max_pool1d(16,16) 对齐 T/16=14；忘记 mask = pad 泄漏
3. 不做 z-score：特征已 per-sample tick 单位化（z-score 会把 tick 单位映射成
   train 世界标准差单位，破坏跨 regime 不变性——TabM 衰减教训）
4. memmap 块洗牌：全随机洗牌会把顺序 IO 打成随机 IO（HDD 慢一个量级）
5. **默认参数陷阱**：`--pool-tau` 默认 0.0 = 等权池化（那版 cos 只有 0.062），
   而 `--drop/--head-drop` 默认 0.1/0.25 与 `cnn_submit.py` 的 0.2/0.3 不一致。
   复现历史结果（LB 0.107 / 衰减 0.946）必须显式传
   `--pool-tau 64 --drop 0.2 --head-drop 0.3`。
6. **折式变更后历史 OOF 不可比**：旧 12 折口径的 OOF 是"模型见过同折其它验证月"训练的，
   与新 6 折口径（每折留出约 12 个月）不是同一件事。旧存档 `snap_cache/oof_parts_v1_14ch/`
   只作历史记录，**不能当作新折式的参照**——新参照要用本脚本在新折式下重跑。
7. 逐折 OOF 默认写 `snap_cache/oof_parts_k6/`（**不与旧 `oof_parts/` 同目录**，
   否则 f2.parquet 这类文件名会跨折式撞车、静默覆盖）。换模型版本时用 `--oof-dir` 再换目录。
8. **不要动 `factors/snap_oof.parquet`**：它是旧 12 折口径的产物，被
   `models/blend/blend_eval3_unit.py` 读取（当前 LB 0.132 的 blend 依赖它）。
   本脚本把汇总 OOF 写在 `--oof-dir` 内部，不碰这个文件。

闸门：
1. 0 负月 + mean ≥ 0.0868（公式 v2 基线）起评
2. 决策指标：与 factors/tabm_oof.parquet 逐月相关——均值 < 0.9 = 新信息源成立；
   > 0.95 且 mean 低于 TabM 0.1330 → 序列无增量，停
   （注意：tabm_oof 是旧 12 折口径，只能取交集算相关，作定性判据）
"""
import argparse
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

# 序列数据管道在顶层 seq/（多模型共用）；cnn_model 在同目录
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "seq"))

from seq_common import (FOLD_STEP, SeqDataset, batch_to_tensor, block_perm,  # noqa: E402
                        build_snap_one, load_train_snap, make_folds, make_loader)
from cnn_model import SnapCNN  # noqa: E402


class SnapBuilder:
    """DataLoader worker 用的单行构建器。写成类而不是闭包——多进程下要可 pickle。"""

    def __init__(self, Xm, n_valid, drop_ch, tcrop):
        self.Xm, self.n_valid = Xm, n_valid
        self.drop_ch, self.tcrop = drop_ch, tcrop

    def __call__(self, row):
        return build_snap_one(self.Xm, self.n_valid, row, self.drop_ch, self.tcrop)

# 跨目录复用（models/tabm）——目录结构见 README
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tabm"))

from tabm_common import BASE, cos_score

TARGET_SCALE = 1000.0
MASK_CHANNELS = {"has_txn": 10, "eside": 12}  # regime 风险通道（--no-mask-channels 消融）


def parse_args():
    p = argparse.ArgumentParser(description="snap-CNN CV 训练评估")
    p.add_argument("--folds", type=int, default=6,
                   help=f"月交错折数（1..{FOLD_STEP}）：折 k 验证 months[k::{FOLD_STEP}]。"
                        f"{FOLD_STEP} 折覆盖全部月份")
    p.add_argument("--start-fold", type=int, default=0,
                   help="从第 N 折开始（前面折的 OOF 已落盘，供被内存击杀后续跑）")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=16)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--workers", type=int, default=6,
                   help="DataLoader 预取进程数。**不改变训练数学**（批次内容/顺序与单线程"
                        "逐位相同，有测试），只把数据准备与 GPU 计算重叠。0=单进程")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--head-drop", type=float, default=0.25)
    p.add_argument("--tcrop", type=int, default=None,
                   help="只用**最新的 tcrop 个有效快照**（逐样本取，pad 区不参与）；"
                        "必须是 16 的倍数；None=全 224 步。"
                        "⚠️ 2026-09-15 前它取的是数组最后 tcrop 行——pad 在数组最末，"
                        "旧实现会把一整段零当最新快照，配 mask 全有效一起喂进池化")
    p.add_argument("--pool-tau", type=float, default=0.0,
                   help="池化时间衰减权重 τ（步）；0=等权平均")
    p.add_argument("--tau-mode", default="legacy", choices=["legacy", "mask"],
                   help="τ 池化的口径。legacy（默认，分子含无效块，实测高 0.044，"
                        "但依赖 pad 位置/零填充/因果性三个耦合，机理见 WALKTHROUGH8.md）；"
                        "mask=分子分母同口径的对照版本")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mask-channels", action="store_true",
                   help="把 has_txn/eside 通道置 0（regime 风险消融）")
    p.add_argument("--drop-channels", default=None,
                   help="额外置 0 的通道下标（逗号分隔）。消融用——"
                        "`14` 就是 dt（FEAT_NAMES 最后一位），置 0 等价于回到 14 通道版输入。"
                        "见 HYPOTHESES.md：验证 dt 是否让新缓存变差")
    p.add_argument("--oof-dir", default="snap_cache/oof_parts_k6",
                   help="逐折 OOF 落盘目录（相对 BASE 或绝对路径）+ 汇总 oof_all.parquet。"
                        "默认与旧 12 折口径的 oof_parts/ 分开；换模型版本时再换目录。"
                        "⚠️ 绝不可指向 snap_cache/oof_parts_v1_14ch/（受保护归档，文件名同构会静默覆盖；"
                        "且 check_cache.py 的逐折排序键会因 oof_all.parquet 无数字而 ValueError）")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def train_fold(Xm, n_valid, y, sids, tr_rows, va_rows, fold_k, args, device):
    """单折训练：块洗牌 memmap 分批 → 因果 CNN → val cos 早停。"""
    drop_ch = list(MASK_CHANNELS.values()) if args.no_mask_channels else []
    if args.drop_channels:
        drop_ch += [int(x) for x in args.drop_channels.split(",")]
    drop_ch = sorted(set(drop_ch)) or None
    xv, mv = batch_to_tensor(Xm, n_valid, va_rows, device, drop_ch, args.tcrop)
    yv = (y[va_rows] * TARGET_SCALE).astype(np.float32)

    torch.manual_seed(args.seed)
    model = SnapCNN(drop=args.drop, head_drop=args.head_drop,
                    pool_tau=args.pool_tau,
                    tau_mode=args.tau_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = np.random.default_rng(args.seed)

    n_tr = len(tr_rows)
    # DataLoader 多进程预取：把 memmap 随机读 + f16→f32 挪到 worker，与 GPU 计算重叠。
    # 批次顺序仍由 block_perm 决定、shuffle=False → 训练数学与手写循环逐位相同
    # （seq/test_seq_common.py 的 test_dataloader_matches_manual / test_builders_agree 守着）。
    ds = SeqDataset(tr_rows, y, SnapBuilder(Xm, n_valid, drop_ch, args.tcrop))
    # DataLoader 只建一次（worker 常驻复用）；每轮用 sampler.set_perm 换批次顺序。
    # 每 epoch 新建 DataLoader 会累积 worker 与 fd，跑到中途 "Too many open files" 卡死。
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
            pred = model(xb, mb)
            loss = F.mse_loss(pred.squeeze(-1), yb)
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
        if ep % 5 == 0 or cos > 0.11:
            print(f"    ep={ep:3d} loss={loss.item():.4f} val_cos={cos:.5f} "
                  f"t={time.time()-t0:5.1f}s", flush=True)

    return {"fold": fold_k, "cos": best_cos, "epoch": best_epoch,
            "va_pred": best_va_pred, "va_sids": sids[va_rows]}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} batch={args.batch} lr={args.lr} wd={args.wd} "
          f"drop={args.drop}/{args.head_drop} tcrop={args.tcrop} "
          f"pool_tau={args.pool_tau} seed={args.seed} "
          f"no_mask_ch={args.no_mask_channels}")

    Xm, n_valid, y, marr, months, sids = load_train_snap()
    print(f"loaded: {Xm.shape} f16 序列, {len(months)} 个月")

    if args.smoke:
        folds = [[months[0]]]          # 最小验证集
        tr_allow = set(months[1:3])    # 只训 2 个月
        n_epochs = 2
    else:
        folds = make_folds(months, args.folds)   # 折 k 验证 months[k::FOLD_STEP]
        tr_allow = None
        n_epochs = args.epochs

    # 逐折落盘：被内存击杀后可用 --start-fold 续跑，不丢已完成的折
    oof_dir = args.oof_dir if os.path.isabs(args.oof_dir) else os.path.join(BASE, args.oof_dir)
    os.makedirs(oof_dir, exist_ok=True)
    results = []
    for fi, va_months in enumerate(folds):
        if fi < args.start_fold:
            continue
        part_path = os.path.join(oof_dir, f"f{fi}.parquet")
        va_mask = np.isin(marr, va_months)
        tr_mask = (~va_mask) if tr_allow is None else np.isin(marr, list(tr_allow))
        tr_rows = np.where(tr_mask)[0]
        va_rows = np.where(va_mask)[0]
        print(f"[fold {fi + 1}/{len(folds)}] 验证月={va_months} "
              f"训练={len(tr_rows)} 验证={len(va_rows)}", flush=True)
        t0 = time.time()
        r = train_fold(Xm, n_valid, y, sids, tr_rows, va_rows, fi,
                       argparse.Namespace(**{**vars(args), "epochs": n_epochs}), device)
        r["fold_secs"] = time.time() - t0
        results.append(r)
        pl.DataFrame({"sample_id": r["va_sids"], "month": marr[va_rows],
                      "pred": r["va_pred"].astype(np.float64)}
                     ).write_parquet(part_path)  # 立即落盘
        print(f"  → cos={r['cos']:.5f} best_epoch={r['epoch']} "
              f"耗时 {r['fold_secs']:.0f}s（已落盘 {part_path}）", flush=True)

    cos_arr = np.array([r["cos"] for r in results])
    ep_arr = np.array([r["epoch"] for r in results])
    print(f"\n### snap-CNN 逐折 cos（{len(results)} 折，月交错 months[k::{FOLD_STEP}]）###")
    print(f"mean={cos_arr.mean():.5f} std={cos_arr.std():.5f} "
          f"min={cos_arr.min():.5f} neg={int((cos_arr < 0).sum())}")
    print(f"best_epoch: median={int(np.median(ep_arr))} 各折={ep_arr.tolist()}")
    print("注：旧归档 OOF（oof_parts_v1_14ch/，12 折单月口径）与新折式**不可比**——"
          "新折每折留出约 12 个月、训练集约 59 个月，旧折训练集 70 个月，"
          "同一折的验证月在新旧口径下训练集不同。新参照须用本脚本在新折式下重跑。")

    if len(results) > 1:
        parts = [pl.read_parquet(os.path.join(oof_dir, f))
                 for f in sorted(os.listdir(oof_dir)) if f.endswith(".parquet")]
        oof = pl.concat(parts)
        # 写进 oof_dir 内部，**不碰 factors/snap_oof.parquet**——那是旧 12 折口径的产物，
        # 被 models/blend/blend_eval3_unit.py 读取（当前 LB 0.132 的 blend 依赖它）
        oof_path = os.path.join(oof_dir, "oof_all.parquet")
        oof.write_parquet(oof_path)
        print(f"\nOOF 已存 {oof_path}（{oof.height} 行；未触碰 factors/snap_oof.parquet）")

        # —— 决策指标：与 TabM OOF 的逐月相关 ——
        tabm_path = os.path.join(BASE, "factors", "tabm_oof.parquet")
        if os.path.exists(tabm_path):
            tabm_oof = pl.read_parquet(tabm_path)
            merged = oof.join(tabm_oof.select(["sample_id", "pred"]),
                              on="sample_id", suffix="_tabm")
            cors = []
            for m in sorted(merged["month"].unique().to_list()):
                mm = merged.filter(pl.col("month") == m)
                c = np.corrcoef(mm["pred"], mm["pred_tabm"])[0, 1]
                cors.append(c)
            cors = np.array(cors)
            print(f"\n### 与 TabM OOF 逐月相关（新信息源判别）###")
            print(f"mean={cors.mean():.4f} min={cors.min():.4f} "
                  f"各月={np.round(cors, 3).tolist()}")
            print("判读：mean < 0.9 → 序列有增量；> 0.95 且 mean 低于 TabM → 无增量")


if __name__ == "__main__":
    main()
