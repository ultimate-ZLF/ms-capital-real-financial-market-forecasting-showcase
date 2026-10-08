"""tabm2_baseline.py — TabM 重做 CV：top-150 tick 单位化（臂 A）+ 可选 TabM† 分箱（臂 B）。

相对 tabm_baseline.py（v1，47 特征 z-score）的改动：
1. 输入：tabm2_common 的 150 特征 + tick 单位化预处理（替代 z-score+clip±5）
2. 臂 B：--embeddings → TabM†（PiecewiseLinearEmbeddings, quantile bins n_bins=48,
   d_embedding=24, activation=False, version="B"；官方 make() 带 embeddings 时
   默认 n_blocks=2，本脚本自动切换）
   常数列防护：compute_bins 要求每列 ≥2 个不同值且 ≥2 边箱——折内常数列
   （全 NaN → 中位数填充后成常数）用 [x0-1, x0+1] 占位箱绕过
3. 断点续跑：--start-fold N 跳过前 N 折；逐折 OOF 即时落盘
   factors/tabm2_oof_folds/{tag}_fold{fi:02d}.parquet（避雷 #15：本机无页面文件，
   长任务必须可续）
4. 每折结果追加 factors/tabm2_cv_results.jsonl

输出：factors/tabm2_oof.parquet（臂 A）/ factors/tabm2_oof_emb.parquet（臂 B）。
对照基准：v1（47 特征 z-score 同协议）CV mean 0.13301（seed 42）。

沿用 v1 的坑守则：mean-loss-over-k、val cos 早停、独立洗牌 Generator、
eval() 关 Dropout、target ×1000 仅数值卫生。
"""
import argparse
import json
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tabm import TabM

from tabm2_common import (OUT, apply_preprocess, cos_score, fit_preprocess,
                          load_train)

VAL_STEP = 6          # 与 v1/LGBM 完全一致：months[2::6] → 2, 8, 14, ..., 68 共 12 折
VAL_START = 2
TARGET_SCALE = 1000.0  # target ×1000 数值卫生（cos 尺度无关）

N_BINS = 48            # TabM† 论文配置
D_EMBEDDING = 24


def parse_args():
    p = argparse.ArgumentParser(description="TabM 重做 CV（150 特征 tick 单位化）")
    p.add_argument("--embeddings", action="store_true",
                   help="臂 B：加 TabM† PiecewiseLinearEmbeddings 分箱")
    p.add_argument("--folds", type=int, default=12, help="只跑前 N 折（快速迭代）")
    p.add_argument("--start-fold", type=int, default=0,
                   help="断点续跑：跳过前 N 折（需同 seed 同参数）")
    p.add_argument("--epochs", type=int, default=300, help="每折 epoch 硬上限")
    p.add_argument("--patience", type=int, default=16, help="早停耐心")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n_blocks", type=int, default=None,
                   help="默认：无 embeddings=3 / 有 embeddings=2（官方配置）")
    p.add_argument("--d_block", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="冒烟：训练前 2 个月/验证第 3 月、2 epochs")
    return p.parse_args()


def make_bins(X_tr_np: np.ndarray) -> list:
    """臂 B 的 quantile bins（CPU 上从本折训练矩阵算）。

    compute_bins 要求每列 ≥2 个不同值且 ≥2 边箱——折内常数列
    （全 NaN → 中位数填充后成常数）用 [x0-1, x0+1] 占位箱绕过。
    """
    from rtdl_num_embeddings import compute_bins
    Xt = torch.from_numpy(X_tr_np)
    const = (Xt == Xt[0]).all(dim=0)
    nconst = (~const).nonzero().flatten()
    bins = [None] * X_tr_np.shape[1]
    if len(nconst):
        sub_bins = compute_bins(Xt[:, nconst], n_bins=N_BINS)
        for j, b in zip(nconst.tolist(), sub_bins):
            bins[j] = b
    for j in const.nonzero().flatten().tolist():
        x0 = float(Xt[0, j])
        bins[j] = torch.tensor([x0 - 1.0, x0 + 1.0])
    return bins


def make_model(n_features: int, args, device, bins=None):
    """臂 A/B 模型构建。臂 B 的 bins 由 make_bins() 预先算好
    （每折算一次、多 seed 复用——避免整折保留 X_tr_np 的 0.74GB CPU 副本，避雷 #15）。"""
    num_embeddings = None
    if args.embeddings:
        from rtdl_num_embeddings import PiecewiseLinearEmbeddings
        assert bins is not None
        num_embeddings = PiecewiseLinearEmbeddings(
            bins, d_embedding=D_EMBEDDING, activation=False, version="B")
    n_blocks = args.n_blocks
    if n_blocks is None:
        n_blocks = 2 if args.embeddings else 3
    return TabM.make(
        n_num_features=n_features, d_out=1,
        n_blocks=n_blocks, k=args.k, d_block=args.d_block,
        num_embeddings=num_embeddings,
    ).to(device)


def train_fold(X, y, sids, us, tr_mask, va_mask, vm, args, device):
    """单折训练：per-fold 中位数填充 + per-sample 单位变换 → mean-loss-over-k →
    val cos 早停。返回 dict(vm, cos, epoch, va_pred, va_sids)。"""
    # 1) 预处理：中位数来自训练折；单位变换是 per-sample 的（无需拟合统计）
    #    布尔切片是独立副本、变换严格就地——峰值内存 ~1.5GB（避雷 #15）
    X_tr_raw = X[tr_mask]
    us_tr = us[tr_mask]
    med = fit_preprocess(X_tr_raw)
    apply_preprocess(X_tr_raw, med, us_tr)  # 就地改写 X_tr_raw
    X_va_raw = X[va_mask]
    apply_preprocess(X_va_raw, med, us[va_mask])
    X_tr = torch.from_numpy(X_tr_raw).to(device)
    X_va = torch.from_numpy(X_va_raw).to(device)
    y_tr = torch.from_numpy(y[tr_mask].astype(np.float32) * TARGET_SCALE).to(device)
    y_va = (y[va_mask] * TARGET_SCALE).astype(np.float32)  # cos 用，留 numpy

    # 2) 模型与优化器
    torch.manual_seed(args.seed)
    bins = make_bins(X_tr_raw) if args.embeddings else None
    n_f = X_tr_raw.shape[1]
    del X_tr_raw, X_va_raw, us_tr  # 训练/验证都走 GPU tensor；numpy 副本释放
    import gc
    gc.collect()
    model = make_model(n_f, args, device, bins)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = torch.Generator().manual_seed(args.seed)  # 独立洗牌种子（多种子名存实亡防护）

    # 3) 训练循环（与 v1 完全一致）
    n_tr = X_tr.shape[0]
    best_cos, best_epoch, best_va_pred, stall = -1.0, -1, None, 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        idx = torch.randperm(n_tr, generator=gen)
        for i in range(0, n_tr, args.batch):
            xb = X_tr[idx[i:i + args.batch]]
            yb = y_tr[idx[i:i + args.batch]]
            y_pred = model(xb)                      # (B, k, 1)
            # mean-loss-over-k：对全部 B×k 元素平均（禁止先 mean(dim=1) 再算 loss）
            loss = F.mse_loss(
                y_pred.squeeze(-1),
                yb.unsqueeze(1).expand(-1, args.k),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(X_va).squeeze(-1).mean(dim=1).cpu().numpy()  # k 平均
        cos = cos_score(pred_va, y_va)
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

    return {
        "vm": vm,
        "cos": best_cos,
        "epoch": best_epoch,
        "va_pred": best_va_pred,
        "va_sids": sids[va_mask],
    }


def main():
    args = parse_args()
    tag = "tabm2emb" if args.embeddings else "tabm2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{tag}] device={device} embeddings={args.embeddings} k={args.k} "
          f"n_blocks={args.n_blocks} d_block={args.d_block} lr={args.lr} "
          f"wd={args.wd} batch={args.batch} seed={args.seed} start_fold={args.start_fold}")

    if args.smoke:
        X, y, marr, months, sids, us = load_train()  # 冒烟走 parquet 路径
    else:
        from tabm2_common import cache_ready, load_train_cached
        X, y, marr, months, sids, us = (load_train_cached() if cache_ready()
                                        else load_train())
    import gc
    gc.collect()  # 释放加载阶段 polars 中间帧（避雷 #15）
    print(f"loaded: {X.shape} 样本, {len(months)} 个月, NaN 占比 "
          f"{float(np.isnan(X).mean()):.4f}, U_s 中位数 {np.median(us):.6f}")

    if args.smoke:
        val_months = [months[VAL_START]]
        tr_allow = set(months[:VAL_START])
        n_epochs = 2
        print(f"[smoke] 训练月 {sorted(tr_allow)} → 验证月 {val_months[0]}，epochs={n_epochs}")
    else:
        val_months = months[VAL_START::VAL_STEP][:args.folds]
        tr_allow = None  # 训练 = 其余全部 70 个月
        n_epochs = args.epochs

    oof_dir = os.path.join(OUT, "tabm2_oof_folds")
    os.makedirs(oof_dir, exist_ok=True)
    results_path = os.path.join(OUT, "tabm2_cv_results.jsonl")

    results = []
    for fi, vm in enumerate(val_months):
        if fi < args.start_fold:
            print(f"[fold {fi + 1}/{len(val_months)}] 跳过（--start-fold={args.start_fold}）")
            continue
        tr_mask = (marr != vm) if tr_allow is None else \
            np.isin(marr, list(tr_allow))
        va_mask = marr == vm
        print(f"[fold {fi + 1}/{len(val_months)}] 验证月={vm} "
              f"训练={int(tr_mask.sum())} 验证={int(va_mask.sum())}", flush=True)
        t0 = time.time()
        r = train_fold(X, y, sids, us, tr_mask, va_mask, vm,
                       argparse.Namespace(**{**vars(args), "epochs": n_epochs}), device)
        r["fold_secs"] = time.time() - t0
        results.append(r)

        # 逐折即时落盘（断点续跑 + 崩溃保护，避雷 #15）；tmp+replace 原子写防杀在写盘中途
        part = pl.DataFrame({
            "sample_id": r["va_sids"],
            "month": vm,
            "pred": r["va_pred"].astype(np.float64),
        })
        part_path = os.path.join(oof_dir, f"{tag}_fold{fi:02d}.parquet")
        part.write_parquet(part_path + ".tmp")
        os.replace(part_path + ".tmp", part_path)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tag": tag, "fold": fi, "vm": int(vm),
                                "cos": float(r["cos"]),
                                "best_epoch": int(r["epoch"]),
                                "secs": int(r["fold_secs"])}) + "\n")
        print(f"  → cos={r['cos']:.5f} best_epoch={r['epoch']} "
              f"耗时 {r['fold_secs']:.0f}s（已存 {os.path.basename(part_path)}）", flush=True)

    cos_arr = np.array([r["cos"] for r in results])
    ep_arr = np.array([r["epoch"] for r in results])
    print(f"\n### [{tag}] 逐月 cos（{len(results)} 折）###")
    print(f"mean={cos_arr.mean():.5f} std={cos_arr.std():.5f} "
          f"min={cos_arr.min():.5f} neg={int((cos_arr < 0).sum())}")
    print(f"best_epoch: median={int(np.median(ep_arr))} 各折={ep_arr.tolist()}")
    print(f"对标：v1（47 特征 z-score）CV mean=0.13301；LGBM v5 0.14097")

    if not args.smoke:
        parts = [pl.read_parquet(os.path.join(oof_dir, f"{tag}_fold{fi:02d}.parquet"))
                 for fi in range(len(val_months))
                 if os.path.exists(os.path.join(oof_dir, f"{tag}_fold{fi:02d}.parquet"))]
        if len(parts) == len(val_months):
            oof = pl.concat(parts)
            oof_path = os.path.join(OUT, f"{tag}_oof.parquet")
            oof.write_parquet(oof_path)
            print(f"OOF 已存 {oof_path}（{oof.height} 行，12 折齐全，供 blend 集成）")
        else:
            print(f"警告：仅 {len(parts)}/{len(val_months)} 折已落盘，未合成 OOF（可续跑补齐）")


if __name__ == "__main__":
    main()
