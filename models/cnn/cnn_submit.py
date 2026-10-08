"""snap-CNN 提交：两种模式。

默认：月交错 6 折 × n_seeds 个模型（各训 59 个月、固定 N_EPOCHS）对 test 平均。
`--full-data`：每个模型训练【全部 71 个月】、不留出验证月，成员数由 `--seeds` 决定。
  提交集成与 CV 折式是两件事——CV 要留出验证月才诚实，提交不需要。
  两条路线各自的理由与未验证点见 HYPOTHESES.md C-06 / C-07。

折式与 cnn_baseline.py 一致（FOLD_STEP=6：折 k 留出 months[k::6]）。
**已认下的取舍**（2026-09-15，为时间成本）：成员数 = 折数 × seeds，从旧 12 折的 24 个
降到 12 个；每个模型的训练集也从 70 个月降到 59 个月。成员数与单模型数据量同时下降，
集成强度会比旧方案弱——若 LB 掉得不划算，回退路径是保持本脚本 6 折不变、
把 `--seeds` 加多，或改为"全 71 个月 × N seeds"（训练集最满）。

镜像 tabm_submit.py 协议：无早停、固定 epochs = CV best_epoch 中位数；
test 序列在构建期已完成 per-sample 归一化，推理只读 memmap + n_valid。

⚠️ 两个"不要覆盖"（旧文件仍是当前 LB 0.132 的组成部分）：
  - 输出默认 `submission_cnn_k6.parquet`，**不覆盖 `submission_cnn_v1.parquet`**
    （后者被 blend/blend3_submit.py 读取）
  - 逐模型缓存默认 `snap_cache/submit_parts_k6/`，**不与旧 `submit_parts/` 同目录**
    （旧文件名 `{月份}_{seed}.npy`、新文件名 `{折号}_{seed}.npy`，同目录会撞车）
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

from seq_common import batch_to_tensor, block_perm, load_train_snap  # noqa: E402
from cnn_model import SnapCNN  # noqa: E402


def load_test_memory():
    """test 特征**现算到内存**，不落盘（2026-09-15 起）。

    test 缓存只在生成提交时用（一天几次），而缓存的价值来自"算一次、复用多次"——
    在 test 侧不成立，白占 4.35GB 磁盘。train 侧仍然走磁盘缓存（每次实验都要读，复用几百次）。
    代价是每次提交多约 3 分钟的特征构建。

    ⚠️ 前提：内存构建与磁盘缓存**逐位一致**——由 `check_memory_build.py` 在 train 上验证过
    （三条管线全过）。如果那条验证失效，这里的特征就会和训练时看到的分布不同，
    而且是**不会报错**的那种错。
    """
    import snap_feats
    X = snap_feats.build_split("test", None, False, False, to_memory=True)
    idx = pl.read_parquet(os.path.join(snap_feats.CACHE, "test_snap_index.parquet"))
    sids = idx["sample_id"].to_numpy()
    assert len(sids) == X.shape[0], "index 行数与构建结果不符"
    assert (np.diff(sids) > 0).all(), "index 的 sample_id 非严格递增——行序异常"
    return X, idx["n_valid"].to_numpy(), sids
# 跨目录复用（models/tabm）——目录结构见 README
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tabm"))

from tabm_common import BASE, cos_score

FOLD_STEP = 6      # 月交错折，与 cnn_baseline.py 一致：折 k 留出 months[k::FOLD_STEP]
TARGET_SCALE = 1000.0
TEST_BATCH = 2048

SEEDS = [42, 7]
N_EPOCHS = None  # = CV 6 折 best_epoch 中位数（跑完 CV 回填）


def _drop_ch(args):
    """解析 --drop-channels。**训练与推理必须共用这一个函数**——两边置零的通道
    不一致的话，test 会看到训练时没见过的通道值，而且不会报错，只是分数悄悄崩掉。"""
    if not args.drop_channels:
        return None
    return [int(x) for x in args.drop_channels.split(",")]


def train_fixed(Xm, n_valid, y, tr_rows, args, seed, device):
    drop_ch = _drop_ch(args)
    torch.manual_seed(seed)
    # 显式传 tau_mode：CV 用的是 legacy，这里也写死，避免哪天默认值变了导致
    # 提交模型与 CV 模型不是同一个东西（这类静默分叉不会报错）
    model = SnapCNN(drop=args.drop, head_drop=args.head_drop,
                    pool_tau=args.pool_tau,
                    tau_mode="legacy").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = np.random.default_rng(seed)
    n_tr = len(tr_rows)
    for ep in range(args.epochs):
        model.train()
        perm = block_perm(n_tr, block=4096, gen=gen)
        for i in range(0, n_tr, args.batch):
            r = tr_rows[perm[i:i + args.batch]]
            xb, mb = batch_to_tensor(Xm, n_valid, r, device, drop_ch)
            yb = torch.from_numpy((y[r] * TARGET_SCALE).astype(np.float32)).to(device)
            pred = model(xb, mb)
            loss = F.mse_loss(pred.squeeze(-1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"      ep={ep:3d} loss={loss.item():.4f}", flush=True)
    model.eval()
    return model


def predict_test(model, Xm, n_valid, device, drop_ch):
    """推理时**必须**与训练用同一组 drop_ch（见 `_drop_ch` 的注释）。"""
    preds = []
    n = Xm.shape[0]
    with torch.no_grad():
        for i in range(0, n, TEST_BATCH):
            r = np.arange(i, min(i + TEST_BATCH, n))
            xb, mb = batch_to_tensor(Xm, n_valid, r, device, drop_ch)
            preds.append(model(xb, mb).squeeze(-1).cpu().numpy())
    return np.concatenate(preds)


def main():
    p = argparse.ArgumentParser(description="snap-CNN 提交生成")
    p.add_argument("--epochs", type=int, default=N_EPOCHS, required=True)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--head-drop", type=float, default=0.3)
    p.add_argument("--pool-tau", type=float, default=64.0)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--folds", type=int, default=6,
                   help=f"月交错折数（1..{FOLD_STEP}）：折 k 留出 months[k::{FOLD_STEP}]")
    p.add_argument("--full-data", action="store_true",
                   help="提交模型用【全部 71 个月】训练、不留出验证月；成员数由 --seeds 决定。"
                        "新折式下每模型只吃 59 个月（旧 12 折是 70），这是补回数据量的手段。"
                        "输出/缓存目录会自动换成 *_full 的名字")
    p.add_argument("--drop-channels", default=None,
                   help="置 0 的通道下标（逗号分隔）。`14` 就是 dt，置零 = 回到 14 通道输入。"
                        "⚠️ 训练与推理共用同一组（_drop_ch），两边不一致会静默崩分")
    p.add_argument("--parts-dir", default=None,
                   help="逐模型 test 预测缓存目录（相对 BASE 或绝对路径）。"
                        "默认按模式派生：submit_parts_{k6|full}[_d14]")
    p.add_argument("--out", default=None,
                   help="提交输出文件名（写入 submissions/）。"
                        "默认按模式派生：submission_cnn_{k6|full}[_d14].parquet。"
                        "都不覆盖 submission_cnn_v1.parquet——那是当前 blend 的成员")
    p.add_argument("--only", default=None,
                   help="只跑一个模型：'k_seed'（k=折号，如 3_42），配合前台逐模型执行")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if not args.full_data:
        assert 1 <= args.folds <= FOLD_STEP, \
            f"--folds 必须在 1..{FOLD_STEP}（月交错折数；超过 {FOLD_STEP} 会让留出月份重复）"
    # 默认输出名/缓存目录随模式变——每种配置的产物必须分开放，否则互相覆盖。
    # tag 例：k6 / full / full_d14（full-data 且置零通道 14）
    tag = "full" if args.full_data else f"k{args.folds}"
    if args.drop_channels:
        tag += "_d" + args.drop_channels.replace(",", "")
    if args.parts_dir is None:
        args.parts_dir = f"snap_cache/submit_parts_{tag}"
    if args.out is None:
        args.out = f"submission_cnn_{tag}.parquet"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seeds={args.seeds} folds={args.folds} "
          f"n_epochs={args.epochs}")

    Xm, n_valid, y, marr, months, _ = load_train_snap()
    Xt, n_valid_t, test_ids = load_test_memory()
    print(f"train {Xm.shape} test {Xt.shape}（test 为内存构建，未落盘）")

    if args.full_data:
        # 全量训练：不留出验证月，每个模型吃全部 71 个月。成员差异只来自 seeds，
        # 所以用 --seeds 控制成员数，`--folds` 在此模式下被忽略。
        # （提交集成与 CV 折式是两件事：CV 需要留出验证月才诚实，提交不需要。）
        folds = [[]]
        print(f"[full-data] 每个模型训练全部 {len(months)} 个月，"
              f"成员数 = {len(args.seeds)}（--folds {args.folds} 已忽略）")
    else:
        # 折 k 留出 months[k::FOLD_STEP]（与 cnn_baseline.py 同一折式）
        folds = [months[k::FOLD_STEP] for k in range(args.folds)]
    if args.only is not None:
        k, seed = args.only.split("_")
        folds, args.seeds = [folds[int(k)]], [int(seed)]
    if args.smoke:
        folds = folds[:1]
        args.seeds = args.seeds[:1]
        args.epochs = 1

    parts_dir = args.parts_dir if os.path.isabs(args.parts_dir) \
        else os.path.join(BASE, args.parts_dir)
    os.makedirs(parts_dir, exist_ok=True)
    preds = []
    t0 = time.time()
    for fi, va_months in enumerate(folds):
        tr_rows = np.where(~np.isin(marr, va_months))[0]
        tag = f"[fold {fi} 留出 {len(va_months)} 月 训练 {len(tr_rows)}]"
        for seed in args.seeds:
            part_path = os.path.join(parts_dir, f"{fi}_{seed}.npy")
            if os.path.exists(part_path):
                pt = np.load(part_path)
                print(f"{tag} seed={seed} 已存在，加载", flush=True)
            else:
                print(f"{tag} seed={seed} 训练 × {args.epochs} epochs", flush=True)
                model = train_fixed(Xm, n_valid, y, tr_rows, args, seed, device)
                pt = predict_test(model, Xt, n_valid_t, device, _drop_ch(args))
                np.save(part_path, pt)  # 立即落盘（实例随时可能被关机，可续跑）
                print(f"    test pred: mean={pt.mean():.6f} std={pt.std():.6f} "
                      f"(×1000 尺度)", flush=True)
            preds.append(pt)
    print(f"总耗时 {time.time() - t0:.0f}s")
    if args.only is not None:
        print(f"[{args.only}] 完成（单模型模式，不组装提交）")
        import sys
        sys.exit(0)

    pred = np.mean(preds, axis=0) / TARGET_SCALE

    print("\n### 诊断 ###")
    for name, fname in [("lgbm", "submission_lgb_v1.parquet"),
                        ("tabm", "submission_tabm_v1.parquet")]:
        path = os.path.join(BASE, "submissions", fname)
        if os.path.exists(path):
            other = pl.read_parquet(path)["prediction"].to_numpy()
            print(f"cos(cnn, {name}) test = {cos_score(pred, other):.5f}")
    print(f"cnn pred: mean={pred.mean():.6f} std={pred.std():.6f}")

    sub = pl.DataFrame({"sample_id": test_ids,
                        "prediction": pred.astype(np.float64)})
    out_path = os.path.join(BASE, "submissions", args.out)
    if os.path.exists(out_path) and args.out == "submission_cnn_v1.parquet":
        print("\n[warn] 正在覆盖 submission_cnn_v1.parquet —— 它是当前 blend 的成员，"
              "blend3_submit.py 会读到新内容，务必确认这是有意的")
    sub.write_parquet(out_path)
    print(f"\n已保存 {out_path}（{sub.height} 行，{len(preds)} 个模型平均）")
    print(f"提示：集成时用 blend3_submit.py 的 alpha/beta，CNN 成员文件换成 {args.out}")


if __name__ == "__main__":
    main()
