"""mlp_submit.py — 双塔MLP 的 test 推理与提交文件生成（**读检查点，不重训**）。

> 这是本模型**独立**的提交脚本：现有 `models/cnn/twotower_cnn_submit.py` 承载的是
> "提交侧重训固定轮数"协议，其产物 `submission_twotower_cnn.parquet` 是 v21 blend（LB 0.133）
> 的成员——改它等于让那份历史产物无法复现（记录纪律，CLAUDE.md #7）。两条谱系必须分开。

与"重训"协议的区别（本模型的设计选择，用户 2026-09-21 指定）：
  - 成员 = `mlp_baseline.py` 逐折训练时**存下的最佳检查点**（`--ckpt-kind best` 默认），
    不再在 test 侧重新训练 → 省掉一整轮训练，且**推理用的就是 CV 里选出的那组权重**
  - 因此每个成员是"训练过 59 个月 + 早停选轮"的模型；成员数 = 检查点数（6 折 × 1 seed = 6）
  - ⚠️ 口径：与现有 k6 提交（59 个月模型）同源，但**轮数由早停而非固定中位数决定**，
    LB 与 `twotower_cnn` 的 0.117 可比但要记住这一处差异

协议：
  - test 特征**现算到内存**（粗段 `snap_feats` + 细段 `flow_feats`，都不落盘）
  - 推理**分块**（`--eval-chunk`，显存护栏；无 BN 故与一次性推理数值一致，差 ~1e-8 量级）
  - 逐成员预测立即落盘 npy（实例随时可能被关机，可 `--only` 续跑）
  - 最后**算术平均** / `TARGET_SCALE`（与 `twotower_cnn_submit.py` 同口径，不 unit()）

用法（**`--crop` 必填**，与训练时一致）：
    python mlp_submit.py --crop 128 --ckpt-dir snap_cache/ckpt_mlp_crop128   # 出提交
    python mlp_submit.py --crop 128 --ckpt-kind last
    python mlp_submit.py --crop 128 --folds 1 --smoke   # 冒烟：只用折 0，输出到 _smoke 名
    python mlp_submit.py --crop 128 --only 3            # 只补折 3 的成员
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import polars as pl
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(MODELS), "seq"))   # 序列数据管道（顶层 seq/）
sys.path.insert(0, os.path.join(MODELS, "tabm"))

import mlp_baseline as MB  # noqa: E402
from mlp_baseline import (COARSE_T, DEFAULT_CKPT_DIR, TARGET_SCALE,  # noqa: E402
                          arch_hash, eval_forward, load_model_from_ckpt)
from tabm_common import BASE  # noqa: E402


DEFAULT_PARTS = "snap_cache/submit_parts_mlp"
DEFAULT_OUT = "submission_mlp.parquet"


def load_test_memory():
    """test 输入现算到内存 → (粗段 (N,224,15), 细段 (N,60,16), sids)。

    两条流的行序都以 `submission.csv` 的 sample_id 升序为准；两份 index 都在盘上时做交叉断言。

    ⚠️ 两个构建器的**签名不同**（别抄错）：
       snap_feats.build_split(split, months, resume, finalize, to_memory)   ← 5 参
       flow_feats.build_split(split, months, resume, to_memory)             ← 4 参
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

    assert xc.shape[1:] == (COARSE_T, MB.COARSE_C), f"粗段形状 {xc.shape}"


def list_ckpts(ckpt_dir, kind, folds=None):
    """列出可用的检查点 → sorted [(fold, path)]（按折号升序）。"""
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"检查点目录不存在：{ckpt_dir}\n  先跑 mlp_baseline.py --folds 6")
    out = []
    for f in os.listdir(ckpt_dir):
        m = re.fullmatch(r"f(\d+)\.(best|last)\.pt", f)
        if m and m.group(2) == kind:
            out.append((int(m.group(1)), os.path.join(ckpt_dir, f)))
    out.sort()
    if folds is not None:
        out = [(fi, p) for fi, p in out if fi < folds]
    if not out:
        raise SystemExit(f"{ckpt_dir} 下没有 f*.{kind}.pt——先跑 mlp_baseline.py")
    return out


def main():
    p = argparse.ArgumentParser(description="双塔MLP 提交（读检查点推理；--crop 必填）")
    p.add_argument("--folds", type=int, default=6, help="最多用前 N 折的检查点")
    p.add_argument("--eval-chunk", type=int, default=8192, help="推理分块行数（显存护栏）")
    p.add_argument("--parts-dir", default=DEFAULT_PARTS)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--only", type=int, default=None, help="只跑指定折号（续跑用）")
    p.add_argument("--smoke", action="store_true", help="冒烟：只跑折 0，输出到 _smoke 名")
    # —— 必须与 mlp_baseline 的架构参数逐字一致（否则指纹校验会拒绝加载）——
    p.add_argument("--tok-hidden", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--head-drop", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--crop", type=int, required=True,
                   help=f"**必填**（用户 2026-09-21 指定）：必须与训练时一致。"
                        f"crop<{COARSE_T} 时粗段塔的时间轴与位置嵌入形状都不同 → 指纹不符会拒绝加载")
    p.add_argument("--clip", type=float, default=1.0,
                   help="梯度裁剪阈值——**只影响训练**，但进了配置指纹，故推理侧也要给一致的值")
    p.add_argument("--loss", choices=["mse", "cos"], default="cos",
                   help="训练损失——**只影响训练**，但 ≠mse 时进了配置指纹，"
                        "故推理侧也要给一致的值（同 --clip / --y-weight-q 的道理）")
    p.add_argument("--y-weight-q", type=float, default=0.0,
                   help="样本加权分位点——**只影响训练**，但进了配置指纹（仅在 >0 时），"
                        "故推理侧也要给一致的值，否则拒绝加载（同 --clip 的道理）")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--ckpt-kind", choices=["best", "last"], default="best")
    args = p.parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.ckpt_dir = args.ckpt_dir if os.path.isabs(args.ckpt_dir) \
        else os.path.join(BASE, args.ckpt_dir)
    parts_dir = args.parts_dir if os.path.isabs(args.parts_dir) \
        else os.path.join(BASE, args.parts_dir)
    os.makedirs(parts_dir, exist_ok=True)
    out_name = "submission_mlp_smoke.parquet" if args.smoke else args.out
    out_path = os.path.join(BASE, "submissions", out_name)

    ckpts = list_ckpts(args.ckpt_dir, args.ckpt_kind, 1 if args.smoke else args.folds)
    if args.only is not None:
        ckpts = [(fi, q) for fi, q in ckpts if fi == args.only]
        if not ckpts:
            raise SystemExit(f"--only {args.only}：该折没有 f{args.only}.{args.ckpt_kind}.pt")

    print(f"device={device} 指纹={arch_hash(args)} crop={args.crop} "
          f"成员={len(ckpts)}（{args.ckpt_kind}）")
    print(f"检查点 ← {args.ckpt_dir}")
    print(f"逐成员预测 → {parts_dir}；输出 → {out_path}")

    t0 = time.time()
    xc, xf, sids = load_test_memory()
    # 掩码逻辑已删（见 mlp_baseline 的 ChannelTimeTower 注释）：n_valid 只用于 crop 裁剪，
    # 故这里给"全有效"占位即可——crop 会按 crop_src 的同一套算式从 224 行里取最新 K 行。
    nv = np.full(len(sids), COARSE_T, dtype=np.int64)
    print(f"test 现算完成：粗段 {xc.shape} + 细段 {xf.shape}（{time.time()-t0:.0f}s）", flush=True)

    rows = np.arange(len(sids))
    for fi, path in ckpts:
        part_path = os.path.join(parts_dir, f"f{fi}.npy")
        if os.path.exists(part_path):
            print(f"  [折 {fi}] 已有 {part_path}，跳过", flush=True)
            continue
        model, meta = load_model_from_ckpt(path, args, device)
        pred = eval_forward(model, xc, xf, nv, rows, args.eval_chunk, device, args.crop)
        np.save(part_path, pred)          # ⚠️ np.save 会追加 .npy（CLAUDE.md #1）：这里 part_path 已带 .npy
        print(f"  [折 {fi}] 推理完成 cos(meta)={meta.get('cos', float('nan')):.5f} "
              f"epoch={meta.get('epoch')} → {part_path}（{time.time()-t0:.0f}s）", flush=True)

    files = sorted(f for f in os.listdir(parts_dir) if f.endswith(".npy"))
    arrs = [np.load(os.path.join(parts_dir, f)) for f in files]
    print(f"\n成员 {len(arrs)} 个：{' '.join(files)}")
    if len(arrs) > 1:
        cors = [float(np.corrcoef(arrs[i], arrs[j])[0, 1])
                for i in range(len(arrs)) for j in range(i + 1, len(arrs))]
        print(f"成员间相关 min={min(cors):.4f} mean={np.mean(cors):.4f} max={max(cors):.4f}")
    pred = np.mean(arrs, axis=0) / TARGET_SCALE     # cos 与尺度无关；除以 TARGET_SCALE 只为与历史提交同量级
    pl.DataFrame({"sample_id": sids, "prediction": pred.astype(np.float64)}).write_parquet(out_path)
    print(f"\n已写 {out_path}（{len(sids)} 行，pred std={pred.std():.6f}）")


if __name__ == "__main__":
    main()
