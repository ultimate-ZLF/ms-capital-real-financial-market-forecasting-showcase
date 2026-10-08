"""twotower_cnn_submit.py — 双塔CNN 的 test 推理与提交文件生成。

模型（**双塔CNN**，家族里的第一个成员；将来还会有"双塔×其他模型"）定义在
`twotower_baseline.py`，本脚本 import 它，保证**训练/推理用的是同一份模型与构建器**。

协议（镜像 `cnn_submit.py`，与项目的提交惯例一致）：
  - **月交错 6 折 × n_seeds 个模型**，各训 59 个月，对 test 预测取平均；
  - **无早停、固定 epochs**（= CV 的 best_epoch 中位数，六折 10/10/10/12/11/10 → 10）。
    理由同 tabm_submit / cnn_submit：提交侧不需要留验证集，用固定轮数复现"中位折的表现"；
  - test 特征**现算到内存**（粗段 `snap_feats --memory` + 细段 `flow_feats --memory`），
    不在盘上留 test 缓存——缓存的价值是"算一次复用多次"，一天几次的提交侧不成立；
  - 逐模型预测立即落盘 npy（实例随时可能被关机，可 `--only 折_种子` 续跑）。

⚠️ 与训练时的一致性：test 侧的构建路径必须与 train 侧**逐位一致**——
`check_memory_build.py` 已在 train 上证明"内存构建 == 磁盘缓存"（三条管线全过）。
本脚本复用 `twotower_baseline.build_one / batch_two`，不再另写一份构建器。

用法：
    python twotower_cnn_submit.py --epochs 10                 # 6 折 × seed 42
    python twotower_cnn_submit.py --epochs 10 --seeds 42 7    # 6 折 × 2 seeds = 12 个成员
    python twotower_cnn_submit.py --epochs 1 --folds 1 --smoke
    python twotower_cnn_submit.py --epochs 10 --only 3_42     # 只补第 3 折 × seed 42
"""
import argparse
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "seq"))   # 序列数据管道（顶层 seq/）

from seq_common import block_perm, make_folds, make_loader  # noqa: E402

# 跨目录复用（models/tabm）——目录结构见 README
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tabm"))

from tabm_common import BASE  # noqa: E402

import twotower_baseline as TWB  # noqa: E402
from twotower_baseline import (TARGET_SCALE, TwoTowerCNN,  # noqa: E402
                               TwoTowerDataset, batch_two, load_train)


DEFAULT_PARTS = "snap_cache/submit_parts_twotower"
DEFAULT_OUT = "submission_twotower_cnn.parquet"


def load_test_memory():
    """test 输入现算到内存 → (粗段 (N,224,15), 细段 (N,60,16), sids)。

    两条流的行序都以 `submission.csv` 的 sample_id 升序为准；两份 index 都在盘上
    （`*_index.parquet`），这里对它们做交叉断言。
    """
    import flow_feats
    import snap_feats

    xc = snap_feats.build_split("test", None, False, False, to_memory=True)
    xf = flow_feats.build_split("test", None, False, to_memory=True)
    assert xc.shape[0] == xf.shape[0], f"两条流行数不同：{xc.shape[0]} vs {xf.shape[0]}"

    sub = pl.read_csv(os.path.join(BASE, "submissions", "submission.csv"),
                      columns=["sample_id"]).sort("sample_id")
    sids = sub["sample_id"].to_numpy()
    assert len(sids) == xc.shape[0], f"submission.csv {len(sids)} 行 vs 特征 {xc.shape[0]} 行"
    assert (np.diff(sids) > 0).all(), "sample_id 非严格递增——行序异常"
    nv = None
    for name in ("snap_cache/test_snap_index.parquet", "flow_cache/test_flow_index.parquet"):
        p = os.path.join(BASE, name)
        if os.path.exists(p):
            idx = pl.read_parquet(p).sort("sample_id")
            assert idx.height == len(sids) and (idx["sample_id"].to_numpy() == sids).all(), \
                f"{name} 的 sample_id 与 submission.csv 行序不一致"
            if name.startswith("snap_cache"):
                nv = idx["n_valid"].to_numpy().astype(np.int64)

    assert xc.shape[1:] == (TWB.COARSE_T, TWB.COARSE_C), f"粗段形状 {xc.shape}"


def train_one(Xc, Xf, y, tr_rows, va_rows, args, device):
    """固定 epochs 训练一个模型（**不早停**——提交侧不留验证集）。
    验证集只用来打印一个参考 cos，不参与任何选择。"""
    xav, xbv = batch_two(Xc, Xf, va_rows, device)
    yv = (y[va_rows] * TARGET_SCALE).astype(np.float32)

    torch.manual_seed(args.seed)
    model = TwoTowerCNN(drop=args.drop, head_drop=args.head_drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = np.random.default_rng(args.seed)

    n_tr = len(tr_rows)
    ds = TwoTowerDataset(tr_rows, y, Xc, Xf)
    loader, sampler = make_loader(ds, n_tr, args.batch, args.workers)

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
        if ep % 5 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = model(xav, xbv).squeeze(-1).cpu().numpy()
            c = float((pv * yv).sum() / np.sqrt((pv * pv).sum() * (yv * yv).sum()))
            print(f"    ep={ep:3d} loss={loss.item():.4f} val_cos(参考)={c:.5f} "
                  f"t={time.time()-t0:5.1f}s", flush=True)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser(description="双塔CNN 提交生成")
    p.add_argument("--epochs", type=int, required=True,
                   help="**必填**：固定训练轮数（CV 的 best_epoch 中位数 = 10）")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--head-drop", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--parts-dir", default=DEFAULT_PARTS)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--only", default=None, help="只跑指定成员，如 `3_42`（折号_种子），用于续跑")
    p.add_argument("--smoke", action="store_true", help="冒烟：1 折 1 epoch")
    args = p.parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    parts_dir = args.parts_dir if os.path.isabs(args.parts_dir) \
        else os.path.join(BASE, args.parts_dir)
    os.makedirs(parts_dir, exist_ok=True)
    out_path = args.out if os.path.isabs(args.out) \
        else os.path.join(BASE, "submissions", args.out)

    print(f"device={device} epochs={args.epochs} seeds={args.seeds} folds={args.folds} "
          f"batch={args.batch} workers={args.workers}")
    print(f"逐模型缓存 → {parts_dir}\n提交输出 → {out_path}")

    t0 = time.time()
    Xc, nv, Xf, y, marr, months, sids_tr = load_train()
    Xtc, Xtf, sids_te = load_test_memory()
    print(f"数据就绪：train 粗段 {Xc.shape} + 细段 {Xf.shape}；"
          f"test 粗段 {Xtc.shape} + 细段 {Xtf.shape}（{time.time()-t0:.1f}s）", flush=True)

    folds = make_folds(months, args.folds)
    va_of_fold = {fi: np.isin(marr, vm) for fi, vm in enumerate(folds)}

    jobs = []
    for fi in range(len(folds)):
        for seed in args.seeds:
            key = f"{fi}_{seed}"
            if args.only and key != args.only:
                continue
            if args.smoke and fi > 0:
                continue
            jobs.append((fi, seed, key))
    print(f"共 {len(jobs)} 个成员：{jobs}\n", flush=True)

    for fi, seed, key in jobs:
        part_path = os.path.join(parts_dir, f"{key}.npy")
        if os.path.exists(part_path):
            print(f"[{key}] 已存在，跳过", flush=True)
            continue
        va_mask = va_of_fold[fi]
        tr_rows, va_rows = np.where(~va_mask)[0], np.where(va_mask)[0]
        print(f"[{key}] 折{fi} 验证月={folds[fi]} 训练={len(tr_rows)}", flush=True)
        a = argparse.Namespace(**{**vars(args), "seed": seed,
                                  "epochs": 1 if args.smoke else args.epochs})
        model = train_one(Xc, Xf, y, tr_rows, va_rows, a, device)
        # 逐块预测 test（一次全量会顶满显存：212k 的验证集已经占到 38GB）
        preds = []
        with torch.no_grad():
            for lo in range(0, Xtc.shape[0], args.batch * 4):
                hi = min(lo + args.batch * 4, Xtc.shape[0])
                xa, xb = batch_two(Xtc, Xtf, np.arange(lo, hi), device)
                preds.append(model(xa, xb).squeeze(-1).cpu().numpy())
        pt = np.concatenate(preds)
        assert pt.shape[0] == Xtc.shape[0], f"预测长度 {pt.shape[0]} != {Xtc.shape[0]}"
        np.save(part_path, pt)          # 立即落盘（可续跑）
        print(f"  → 已落盘 {part_path}（std={pt.std():.3e}）", flush=True)

    files = sorted(f for f in os.listdir(parts_dir) if f.endswith(".npy"))
    if not files:
        print("没有可汇总的逐模型预测")
        return
    arrs = [np.load(os.path.join(parts_dir, f)) for f in files]
    print(f"\n汇总 {len(arrs)} 个成员：{files}")
    corr = np.corrcoef(np.vstack(arrs))
    if len(arrs) > 1:
        off = corr[np.triu_indices(len(arrs), 1)]
        print(f"成员间相关：min={off.min():.4f} mean={off.mean():.4f} max={off.max():.4f}")
    pred = np.mean(arrs, axis=0) / TARGET_SCALE
    # 除以 TARGET_SCALE：训练时 target 乘过 1000（MSE 的数值尺度），而**提交文件的口径是原始
    # target 尺度**（历史各成员的 std ≈ 3e-4）。指标 cos 与集成用的 unit() 都尺度无关，
    # 所以这一步纯粹是为了和既有提交文件可比、不至于让人误判量级。
    print(f"平均预测（原始尺度）: mean={pred.mean():.5e} std={pred.std():.5e}")

    sub = pl.DataFrame({"sample_id": sids_te, "prediction": pred.astype(np.float64)})
    sub.write_parquet(out_path)
    print(f"saved -> {out_path}（{sub.height} 行）")
    print(sub.head(3))


if __name__ == "__main__":
    main()
