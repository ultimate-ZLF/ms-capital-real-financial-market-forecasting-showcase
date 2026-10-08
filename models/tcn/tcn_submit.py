"""tcn_submit.py — 双塔TCN 的 test 推理与提交文件生成（**读检查点，不重训**）。

> 这是本模型**独立**的提交脚本。两条既有谱系都不照搬：
>   - `models/cnn/twotower_cnn_submit.py` 是"**提交侧重训固定轮数**"协议，其产物
>     `submission_twotower_cnn.parquet` 是 v21 blend（LB 0.133）的成员——改它等于让那份历史
>     产物无法复现（记录纪律，CLAUDE.md #7）；
>   - `models/mlp/mlp_submit.py` 是"读检查点"协议，本脚本与它同谱系，
>     但**有一处必须不同**（见下一条）。
>
> ⚠️ **与 `mlp_submit.py` 的关键差别**：双塔MLP 无 pad 语义，`n_valid` 只用来定位 crop，
> 所以它给一个"全有效"占位就能跑。**双塔TCN 的读出位置 = `n_valid−1`，`n_valid` 是必需输入**
> ——占位会让每样本都去读位置 223（那段是 pad）→ 静默产出垃圾预测。所以本脚本**强制**读
> `snap_cache/test_snap_index.parquet` 取真实 `n_valid`，读不到就报错退出。

协议：
  - test 特征**现算到内存**（粗段 `snap_feats` + 细段 `flow_feats`，都不落盘）
  - 推理**分块**（`--eval-chunk`，显存护栏：TCN 全程 T=224，整折一次性前向约 12GB/张）
  - 逐成员预测立即落盘 npy（实例随时可能被关机，可 `--only` 续跑）
  - 最后**算术平均** / `TARGET_SCALE`（与 `twotower_cnn_submit.py` / `mlp_submit.py` 同口径，不 unit()）

用法：
    python tcn_submit.py                          # 出提交（6 折检查点）
    python tcn_submit.py --ckpt-kind last
    python tcn_submit.py --folds 1 --smoke        # 冒烟：只用折 0，输出到 _smoke 名
    python tcn_submit.py --only 3                 # 只补折 3 的成员
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

import tcn_baseline as TB  # noqa: E402
from tcn_baseline import (COARSE_T, DEFAULT_CKPT_DIR, TARGET_SCALE,  # noqa: E402
                          arch_hash, eval_forward, load_model_from_ckpt)
from tabm_common import BASE  # noqa: E402

def _coarse_c():
    return TB.COARSE_C

DEFAULT_PARTS = "snap_cache/submit_parts_tcn"
DEFAULT_OUT = "submission_tcn.parquet"


def load_test_memory():
    """test 输入现算到内存 → (粗段 (N,224,15), 细段 (N,60,16), n_valid (N,), sids)。

    两条流的行序都以 `submission.csv` 的 sample_id 升序为准；三处 sample_id 做交叉断言。

    ⚠️ 两个构建器的**签名不同**（别抄错）：
       snap_feats.build_split(split, months, resume, finalize, to_memory)   ← 5 参
       flow_feats.build_split(split, months, resume, to_memory)             ← 4 参
    """
    import flow_feats
    import snap_feats

    xc = snap_feats.build_split("test", None, False, False, to_memory=True)
    xf = flow_feats.build_split("test", None, False, to_memory=True)
    assert xc.shape[0] == xf.shape[0], f"两条流行数不同：{xc.shape[0]} vs {xf.shape[0]}"
    assert xc.shape[1:] == (COARSE_T, 15) and xf.shape[1:] == (60, 16), \
        f"形状异常（base）：粗段 {xc.shape} 细段 {xf.shape}"

    sub = pl.read_csv(os.path.join(BASE, "submissions", "submission.csv"),
                      columns=["sample_id"]).sort("sample_id")
    sids = sub["sample_id"].to_numpy()
    assert len(sids) == xc.shape[0], f"submission.csv {len(sids)} 行 vs 特征 {xc.shape[0]} 行"
    assert (np.diff(sids) > 0).all(), "sample_id 非严格递增——行序异常"

    # n_valid 是**必需输入**（末步读出取 n_valid−1），不是可选的校验件
    idx_p = os.path.join(BASE, "snap_cache", "test_snap_index.parquet")
    if not os.path.exists(idx_p):
        raise SystemExit(f"缺 {idx_p}——末步读出需要每样本的 n_valid，无法用占位值代替。\n"
                         f"  该文件由 models/cnn/snap_feats.py 构建 test 粗段缓存时产出。")
    idx = pl.read_parquet(idx_p).sort("sample_id")
    assert idx.height == len(sids) and (idx["sample_id"].to_numpy() == sids).all(), \
        f"{idx_p} 的 sample_id 与 submission.csv 行序不一致"
    nv = idx["n_valid"].to_numpy().astype(np.int64)
    assert (nv >= 1).all() and (nv <= COARSE_T).all(), \
        f"test n_valid 越界：[{nv.min()}, {nv.max()}] 必须落在 1..{COARSE_T}"

    for name in ("flow_cache/test_flow_index.parquet",):
        p = os.path.join(BASE, name)
        if os.path.exists(p):
            i2 = pl.read_parquet(p).sort("sample_id")
            assert i2.height == len(sids) and (i2["sample_id"].to_numpy() == sids).all(), \
                f"{name} 的 sample_id 与 submission.csv 行序不一致"
    assert xc.shape[1:] == (COARSE_T, _coarse_c()), f"粗段形状 {xc.shape}"


def list_ckpts(ckpt_dir, kind, folds=None):
    """列出可用的检查点 → sorted [(fold, path)]（按折号升序）。"""
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"检查点目录不存在：{ckpt_dir}\n  先跑 tcn_baseline.py --folds 6")
    out = []
    for f in os.listdir(ckpt_dir):
        m = re.fullmatch(r"f(\d+)\.(best|last)\.pt", f)
        if m and m.group(2) == kind:
            out.append((int(m.group(1)), os.path.join(ckpt_dir, f)))
    out.sort()
    if folds is not None:
        out = [(fi, p) for fi, p in out if fi < folds]
    if not out:
        raise SystemExit(f"{ckpt_dir} 下没有 f*.{kind}.pt——先跑 tcn_baseline.py")
    return out


def main():
    p = argparse.ArgumentParser(description="双塔TCN 提交（读检查点推理）")
    p.add_argument("--folds", type=int, default=6, help="最多用前 N 折的检查点")
    p.add_argument("--eval-chunk", type=int, default=8192, help="推理分块行数（显存护栏）")
    p.add_argument("--parts-dir", default=DEFAULT_PARTS)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--only", type=int, default=None, help="只跑指定折号（续跑用）")
    p.add_argument("--smoke", action="store_true", help="冒烟：只跑折 0，输出到 _smoke 名")
    # —— 必须与 tcn_baseline 的架构参数逐字一致（否则指纹校验会拒绝加载）——
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--levels-coarse", type=int, default=6)
    p.add_argument("--levels-fine", type=int, default=4)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--head-drop", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="**只影响训练**，但进了配置指纹，故推理侧也要给一致的值")
    p.add_argument("--wd", type=float, default=3e-4, help="同上")
    p.add_argument("--batch", type=int, default=1024, help="同上")
    p.add_argument("--loss", choices=["mse", "cos"], default="cos",
                   help="训练损失——**只影响训练**，但 ≠cos 时进了配置指纹；"
                        "推理 **mse** 臂的旧检查点必须显式传 `--loss mse`，否则指纹不符会被拒载。"
                        "默认与训练侧一致 = cos")
    # ⚠️ 刻意**不加** --patience / --epochs：它们不进指纹、也不影响推理 → 加了就是**空转开关**
    #    （CLAUDE.md #12 的反向纪律：别为"等价于不变"的开关加字段/参数）
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--ckpt-kind", choices=["best", "last"], default="best")
    args = p.parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.ckpt_dir = args.ckpt_dir if os.path.isabs(args.ckpt_dir) \
        else os.path.join(BASE, args.ckpt_dir)
    parts_dir = args.parts_dir if os.path.isabs(args.parts_dir) \
        else os.path.join(BASE, args.parts_dir)
    os.makedirs(parts_dir, exist_ok=True)
    out_name = "submission_tcn_smoke.parquet" if args.smoke else args.out
    out_path = os.path.join(BASE, "submissions", out_name)

    ckpts = list_ckpts(args.ckpt_dir, args.ckpt_kind, 1 if args.smoke else args.folds)
    if args.only is not None:
        ckpts = [(fi, q) for fi, q in ckpts if fi == args.only]
        if not ckpts:
            raise SystemExit(f"--only {args.only}：该折没有 f{args.only}.{args.ckpt_kind}.pt")

    print(f"device={device} 指纹={arch_hash(args)} width={args.width} "
          f"levels={args.levels_coarse}/{args.levels_fine} 成员={len(ckpts)}（{args.ckpt_kind}）")
    print(f"检查点 ← {args.ckpt_dir}")
    print(f"逐成员预测 → {parts_dir}；输出 → {out_path}")
    print("⚠️ 读出位置 = 每样本 n_valid−1（取自 test_snap_index.parquet），不是数组末位")

    t0 = time.time()
    xc, xf, nv, sids = load_test_memory()
    print(f"test 现算完成：粗段 {xc.shape} + 细段 {xf.shape}"
          f"（n_valid 中位 {int(np.median(nv))}，{time.time()-t0:.0f}s）", flush=True)

    rows = np.arange(len(sids))
    for fi, path in ckpts:
        part_path = os.path.join(parts_dir, f"f{fi}.npy")
        if os.path.exists(part_path):
            print(f"  [折 {fi}] 已有 {part_path}，跳过", flush=True)
            continue
        model, meta = load_model_from_ckpt(path, args, device)
        pred = eval_forward(model, xc, xf, nv, rows, args.eval_chunk, device)
        np.save(part_path, pred)      # ⚠️ np.save 会追加 .npy（CLAUDE.md #1）：part_path 已带 .npy
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
