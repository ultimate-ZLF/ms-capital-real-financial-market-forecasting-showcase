"""tabm2_submit.py — TabM 重做提交：12 折 × n_seeds 个模型（各训 70 个月、固定 N_EPOCHS）对 test 预测平均。

镜像 tabm_submit.py（v1）协议：无早停、固定轮数 = 该臂 CV 12 折 best_epoch 中位数。
与 OOF 分两个循环产出（第一轮教训：OOF 用逐折早停预测，提交用固定轮数模型）。

改动点：tabm2_common 的 150 特征 + tick 单位化；--embeddings 走 TabM† 分箱臂。

断点续跑：每个 (vm, seed) 的 test 预测存 factors/tabm2_submit_parts/{vm}_s{seed}.npy，
已存在则跳过（3 小时级任务，本机无页面文件易被系统击杀——避雷 #15）。

内存纪律（同上避雷）：bins 每折算一次多 seed 复用；X_tr_np/Xt_np CPU 副本
即用即删，臂 A 峰值 ~2.3GB。

输出 submissions/submission_tabm_v2.parquet：
- sample_id(Int32) + prediction(Float64)，647,896 行，顺序同 factors_test.parquet
- 顺带输出 cos(tabm2, lgbm_v5) 诊断（互补性参考）
"""
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from tabm2_common import (BASE, apply_preprocess, cos_score, fit_preprocess,
                          load_test, load_train)
from tabm2_baseline import make_bins, make_model  # import 安全（main() 有保护）

VAL_STEP = 6   # 与 v1/lgb 完全一致：months[2::6] → 2, 8, 14, ..., 68
VAL_START = 2
TARGET_SCALE = 1000.0
TEST_BATCH = 8192

SEEDS = [42, 7]           # 2 seed 起步；时间允许加 123 变 3 seed
N_EPOCHS = None           # = 该臂 CV 12 折 best_epoch 中位数（跑完 CV 回填）


def part_dir(tag):
    return os.path.join(BASE, "factors", f"tabm2_submit_parts_{tag}")


def part_path(tag, vm, seed):
    return os.path.join(part_dir(tag), f"{vm}_s{seed}.npy")


def train_fixed(X_tr, y_tr, n_features, args, seed, device, bins=None):
    """70 个月全量训练 N_EPOCHS 个 epoch，返回模型（eval 状态）。"""
    torch.manual_seed(seed)
    model = make_model(n_features, args, device, bins)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = torch.Generator().manual_seed(seed)
    n = X_tr.shape[0]
    for ep in range(args.epochs):
        model.train()
        idx = torch.randperm(n, generator=gen)
        for i in range(0, n, args.batch):
            xb = X_tr[idx[i:i + args.batch]]
            yb = y_tr[idx[i:i + args.batch]]
            y_pred = model(xb)
            loss = F.mse_loss(y_pred.squeeze(-1), yb.unsqueeze(1).expand(-1, args.k))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"      ep={ep:3d} loss={loss.item():.4f}", flush=True)
    model.eval()
    return model


def predict_test(model, Xt, device):
    """test 分批推理，k 平均后返回 (N,) numpy。"""
    preds = []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], TEST_BATCH):
            xb = Xt[i:i + TEST_BATCH].to(device)
            preds.append(model(xb).squeeze(-1).mean(dim=1).cpu().numpy())
    return np.concatenate(preds)


def main():
    import argparse
    import gc
    p = argparse.ArgumentParser(description="TabM 重做提交生成")
    p.add_argument("--epochs", type=int, default=N_EPOCHS, required=True,
                   help="固定训练 epochs（该臂 CV best_epoch 中位数）")
    p.add_argument("--embeddings", action="store_true", help="TabM† 分箱臂")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n_blocks", type=int, default=None,
                   help="默认：无 embeddings=3 / 有 embeddings=2")
    p.add_argument("--d_block", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--folds", type=int, default=12, help="只跑前 N 折（调试）")
    p.add_argument("--out", type=str, default="submission_tabm_v2.parquet")
    p.add_argument("--smoke", action="store_true",
                   help="冒烟：1 折 1 seed 1 epoch，验证 test 预测路径（不写断点缓存）")
    args = p.parse_args()

    tag = "tabm2emb" if args.embeddings else "tabm2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{tag}] device={device} seeds={args.seeds} folds={args.folds} "
          f"n_epochs={args.epochs}")

    if args.smoke:
        X, y, marr, months, _, us = load_train()
        Xt, test_ids, us_test = load_test()
    else:
        from tabm2_common import (cache_ready, load_test_cached,
                                  load_train_cached)
        if cache_ready():
            X, y, marr, months, _, us = load_train_cached()
            Xt, test_ids, us_test = load_test_cached()
        else:
            X, y, marr, months, _, us = load_train()
            Xt, test_ids, us_test = load_test()
    gc.collect()
    print(f"train {X.shape} test {Xt.shape}")

    val_months = months[VAL_START::VAL_STEP][:args.folds]
    if args.smoke:
        val_months = val_months[:1]
        args.seeds = args.seeds[:1]
        args.epochs = 1

    if not args.smoke:
        os.makedirs(part_dir(tag), exist_ok=True)

    t0 = time.time()
    for vm in val_months:
        tr = marr != vm
        X_tr_np = X[tr]  # 布尔切片 = 独立副本，可安全就地改写
        med = fit_preprocess(X_tr_np)
        apply_preprocess(X_tr_np, med, us[tr])
        Xt_np = Xt.copy()  # Xt 跨折共享，必须显式拷贝后再就地变换
        apply_preprocess(Xt_np, med, us_test)  # 本折中位数填充 + 单位变换
        X_tr = torch.from_numpy(X_tr_np).to(device)
        y_tr = torch.from_numpy((y[tr] * TARGET_SCALE).astype(np.float32)).to(device)
        Xt_p = torch.from_numpy(Xt_np)  # 与 Xt_np 共享内存，del 安全
        bins = make_bins(X_tr_np) if args.embeddings else None
        n_f = X_tr_np.shape[1]
        del X_tr_np, Xt_np  # CPU 副本即用即删（bins 已算好，多 seed 复用）
        gc.collect()
        for seed in args.seeds:
            pp = part_path(tag, vm, seed)
            if not args.smoke and os.path.exists(pp):
                print(f"[vm={vm} seed={seed}] 断点已存在，跳过", flush=True)
                continue
            print(f"[vm={vm} seed={seed}] 训练 {X_tr.shape[0]} 样本 "
                  f"× {args.epochs} epochs", flush=True)
            model = train_fixed(X_tr, y_tr, n_f, args, seed, device, bins)
            pt = predict_test(model, Xt_p, device)
            if not args.smoke:
                # np.save 会自动追加 .npy——tmp 名必须已带 .npy 结尾，os.replace 才能对上
                np.save(pp + ".tmp.npy", pt.astype(np.float32))
                os.replace(pp + ".tmp.npy", pp)  # 原子写，防杀在写盘中途产生坏缓存
                print(f"    test pred: mean={pt.mean():.6f} std={pt.std():.6f} "
                      f"(×1000 尺度) 已存 {os.path.basename(pp)}", flush=True)
        del X_tr, y_tr, Xt_p
        gc.collect()
    print(f"总耗时 {time.time() - t0:.0f}s")

    if args.smoke:
        preds = [pt]  # 冒烟路径：循环里最后一个模型的预测
    else:
        preds = [np.load(part_path(tag, vm, s))
                 for vm in val_months for s in args.seeds]
    pred = np.mean(preds, axis=0) / TARGET_SCALE  # 还原 target 尺度

    # —— 与 LGBM v5 的相关性诊断 ——
    lgb_path = os.path.join(BASE, "submissions", "submission_lgb_v5.parquet")
    if os.path.exists(lgb_path):
        lgb = pl.read_parquet(lgb_path)["prediction"].to_numpy()
        print(f"\n### 诊断 ###")
        print(f"cos(tabm2, lgbm_v5) test = {cos_score(pred, lgb):.5f}")
        print(f"tabm2 pred: mean={pred.mean():.6f} std={pred.std():.6f}")
        print(f"lgbm_v5 pred: mean={lgb.mean():.6f} std={lgb.std():.6f}")

    sub = pl.DataFrame({"sample_id": test_ids,
                        "prediction": pred.astype(np.float64)})
    out_path = os.path.join(BASE, "submissions", args.out)
    sub.write_parquet(out_path)
    print(f"\n已保存 {out_path}（{sub.height} 行）")


if __name__ == "__main__":
    main()
