"""TabM 基线 v1：47 因子（一维）回归，按月分组 CV（每 6 月取 1 验证月，12 折）。

坑清单（新会话别重踩）：
1. NaN 漏处理：Int 可空列 to_numpy() 后 null→NaN，预处理必须中位数填充（tabm_common.apply_preprocess）
2. mean-loss-over-k 写反：loss 必须对 (B,k) 全元素平均，禁止先 mean(dim=1) 再算 loss
   （那是优化"平均预测"，违背论文核心）
3. 推理忘 k 平均：val/submit 预测必须 squeeze(-1).mean(dim=1) 得到 (B,)
4. Windows DataLoader：num_workers>0 需 spawn + __main__ 保护，本脚本用手动索引循环绕过
5. 早停用 val cos 不用 val MSE：各月波动率差近一倍，MSE 跨折不可比
6. 种子三件套：torch.manual_seed + shuffle 用独立 torch.Generator，否则多种子名存实亡
7. train()/eval() 切换：网络含 Dropout(0.1)，eval 时才可算验证预测

闸门（对标 LGBM CV 0.1271 / 公式 v2 0.0868）：
1. 逐月 cos 均值 ≥ 0.1271（不达标则按计划调 d_block/lr 重跑）
2. 逐月 cos 全为正
通过闸门才生成 test 提交。
"""
import argparse
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tabm import TabM

from tabm_common import (BASE, apply_preprocess, cos_score, fit_preprocess,
                         load_train)

VAL_STEP = 6          # 每 6 月取 1 验证月（与 LGBM 完全一致：months[2::6]）
VAL_START = 2         # → 验证月 2, 8, 14, ..., 68 共 12 折（lgb 注释写 3,9,... 与实际不符）
TARGET_SCALE = 1000.0  # target ×1000 数值卫生（cos 尺度无关）


def parse_args():
    p = argparse.ArgumentParser(description="TabM CV 训练评估")
    p.add_argument("--folds", type=int, default=12, help="只跑前 N 折（快速迭代）")
    p.add_argument("--epochs", type=int, default=300, help="每折 epoch 硬上限")
    p.add_argument("--patience", type=int, default=16, help="早停耐心（镜像官方 example）")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n_blocks", type=int, default=3)
    p.add_argument("--d_block", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="冒烟：训练前 2 个月/验证第 3 月、2 epochs")
    return p.parse_args()


def train_fold(X, y, sids, tr_mask, va_mask, vm, args, device):
    """单折训练：per-fold 预处理 → mean-loss-over-k 训练 → val cos 早停。

    返回 dict(vm, cos, epoch, va_pred, va_sids)——va_pred 为 best epoch 的 k 平均预测。
    """
    # 1) 预处理（只用训练折统计）
    med, mu, sd = fit_preprocess(X[tr_mask])
    X_tr = torch.from_numpy(apply_preprocess(X[tr_mask], med, mu, sd)).to(device)
    X_va = torch.from_numpy(apply_preprocess(X[va_mask], med, mu, sd)).to(device)
    y_tr = torch.from_numpy(y[tr_mask].astype(np.float32) * TARGET_SCALE).to(device)
    y_va = (y[va_mask] * TARGET_SCALE).astype(np.float32)  # cos 用，留 numpy

    # 2) 模型与优化器（论文默认 AdamW lr=2e-3 wd=3e-4）
    torch.manual_seed(args.seed)
    model = TabM.make(
        n_num_features=X_tr.shape[1], d_out=1,
        n_blocks=args.n_blocks, k=args.k, d_block=args.d_block,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = torch.Generator().manual_seed(args.seed)  # 独立洗牌种子（与模型初始化分开）

    # 3) 训练循环
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
            # mean-loss-over-k：对全部 B×k 元素平均（禁止先 mean 再算 loss）
            loss = F.mse_loss(
                y_pred.squeeze(-1),
                yb.unsqueeze(1).expand(-1, args.k),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        # 验证（eval 关 Dropout）
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} k={args.k} n_blocks={args.n_blocks} d_block={args.d_block} "
          f"lr={args.lr} wd={args.wd} batch={args.batch} seed={args.seed}")

    X, y, marr, months, sids = load_train()
    print(f"loaded: {X.shape} 样本, {len(months)} 个月, NaN 占比 "
          f"{float(np.isnan(X).mean()):.4f}")

    if args.smoke:
        # 冒烟：训练前 2 个月、验证第 3 个月（VAL_MONTHS[0]）、2 epochs
        val_months = [months[VAL_START]]
        tr_allow = set(months[:VAL_START])
        n_epochs = 2
        print(f"[smoke] 训练月 {sorted(tr_allow)} → 验证月 {val_months[0]}，epochs={n_epochs}")
    else:
        val_months = months[VAL_START::VAL_STEP][:args.folds]
        tr_allow = None  # 训练 = 其余全部 70 个月
        n_epochs = args.epochs

    results = []
    oof_parts = []
    for fi, vm in enumerate(val_months):
        tr_mask = (marr != vm) if tr_allow is None else \
            np.isin(marr, list(tr_allow))
        va_mask = marr == vm
        print(f"[fold {fi + 1}/{len(val_months)}] 验证月={vm} "
              f"训练={int(tr_mask.sum())} 验证={int(va_mask.sum())}", flush=True)
        t0 = time.time()
        r = train_fold(X, y, sids, tr_mask, va_mask, vm,
                       argparse.Namespace(**{**vars(args), "epochs": n_epochs}), device)
        r["fold_secs"] = time.time() - t0
        results.append(r)
        oof_parts.append(pl.DataFrame({
            "sample_id": r["va_sids"],
            "month": vm,
            "pred": r["va_pred"].astype(np.float64),
        }))
        print(f"  → cos={r['cos']:.5f} best_epoch={r['epoch']} "
              f"耗时 {r['fold_secs']:.0f}s", flush=True)

    cos_arr = np.array([r["cos"] for r in results])
    ep_arr = np.array([r["epoch"] for r in results])
    print(f"\n### TabM 逐月 cos（{len(results)} 折）###")
    print(f"mean={cos_arr.mean():.5f} std={cos_arr.std():.5f} "
          f"min={cos_arr.min():.5f} neg={int((cos_arr < 0).sum())}")
    print(f"best_epoch: median={int(np.median(ep_arr))} 各折={ep_arr.tolist()}")
    print(f"对标：LGBM CV mean=0.1271 / 公式 v2=0.0868")

    if not args.smoke and len(results) > 1:
        oof = pl.concat(oof_parts)
        oof_path = os.path.join(BASE, "factors", "tabm_oof.parquet")
        oof.write_parquet(oof_path)
        print(f"OOF 已存 {oof_path}（{oof.height} 行，供后续 LGBM×TabM 集成）")


if __name__ == "__main__":
    main()
