"""TabM 提交：12 折 × n_seeds 个模型（各训 70 个月、固定 N_EPOCHS）对 test 预测平均。

镜像 lgb_submit.py 协议：无早停、固定轮数 = CV 12 折 best_epoch 中位数。
NN 对种子敏感 → 多 seed 平均是免费集成（shuffle 用独立 torch.Generator，
否则多 seed 名存实亡）。

输出 submissions/submission_tabm_v1.parquet：
- sample_id(Int32) + prediction(Float64)，647,896 行，顺序同 factors_test.parquet
- 顺带输出 cos(tabm, lgbm) 诊断（全 test 一个标量）：
  ≈0.99 集成收益小；0.6-0.9 互补性强
"""
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tabm import TabM

from tabm_common import (BASE, apply_preprocess, cos_score, fit_preprocess,
                         load_test, load_train)

VAL_STEP = 6   # 与 lgb_baseline 完全一致：months[2::6] → 2, 8, 14, ..., 68
VAL_START = 2
TARGET_SCALE = 1000.0
TEST_BATCH = 8192

SEEDS = [42, 7]           # 2 seed 起步；时间允许加 2025 变 3 seed
N_EPOCHS = None           # = CV 12 折 best_epoch 中位数（跑完 CV 回填）


def train_fixed(X_tr, y_tr, args, seed, device):
    """70 个月全量训练 N_EPOCHS 个 epoch，返回模型（eval 状态）。"""
    torch.manual_seed(seed)
    model = TabM.make(
        n_num_features=X_tr.shape[1], d_out=1,
        n_blocks=args.n_blocks, k=args.k, d_block=args.d_block,
    ).to(device)
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
    p = argparse.ArgumentParser(description="TabM 提交生成")
    p.add_argument("--epochs", type=int, default=N_EPOCHS, required=True,
                   help="固定训练 epochs（CV best_epoch 中位数）")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n_blocks", type=int, default=3)
    p.add_argument("--d_block", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--folds", type=int, default=12, help="只跑前 N 折（调试）")
    p.add_argument("--smoke", action="store_true",
                   help="冒烟：1 折 1 seed 1 epoch，验证 test 预测路径")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seeds={args.seeds} folds={args.folds} "
          f"n_epochs={args.epochs}")

    X, y, marr, months, _ = load_train()
    Xt, test_ids = load_test()
    print(f"train {X.shape} test {Xt.shape}")

    val_months = months[VAL_START::VAL_STEP][:args.folds]
    if args.smoke:
        val_months = val_months[:1]
        args.seeds = args.seeds[:1]
        args.epochs = 1

    preds = []
    t0 = time.time()
    for vm in val_months:
        tr = marr != vm
        med, mu, sd = fit_preprocess(X[tr])
        X_tr = torch.from_numpy(apply_preprocess(X[tr], med, mu, sd)).to(device)
        y_tr = torch.from_numpy((y[tr] * TARGET_SCALE).astype(np.float32)).to(device)
        Xt_p = torch.from_numpy(apply_preprocess(Xt, med, mu, sd))  # 本折统计变换
        for seed in args.seeds:
            print(f"[vm={vm} seed={seed}] 训练 {X_tr.shape[0]} 样本 "
                  f"× {args.epochs} epochs", flush=True)
            model = train_fixed(X_tr, y_tr, args, seed, device)
            pt = predict_test(model, Xt_p, device)
            preds.append(pt)
            print(f"    test pred: mean={pt.mean():.6f} std={pt.std():.6f} "
                  f"(×1000 尺度)", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s")

    pred = np.mean(preds, axis=0) / TARGET_SCALE  # 还原 target 尺度

    # —— 与 LGBM 的相关性诊断 ——
    lgb_path = os.path.join(BASE, "submissions", "submission_lgb_v1.parquet")
    if os.path.exists(lgb_path):
        lgb = pl.read_parquet(lgb_path)["prediction"].to_numpy()
        print(f"\n### 诊断 ###")
        print(f"cos(tabm, lgbm) test = {cos_score(pred, lgb):.5f}")
        print(f"tabm pred: mean={pred.mean():.6f} std={pred.std():.6f}")
        print(f"lgbm pred: mean={lgb.mean():.6f} std={lgb.std():.6f}")

    sub = pl.DataFrame({"sample_id": test_ids,
                        "prediction": pred.astype(np.float64)})
    out_path = os.path.join(BASE, "submissions", "submission_tabm_v1.parquet")
    sub.write_parquet(out_path)
    print(f"\n已保存 {out_path}（{sub.height} 行）")


if __name__ == "__main__":
    main()
