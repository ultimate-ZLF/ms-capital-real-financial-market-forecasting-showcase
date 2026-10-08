"""tfm_submit.py — 三塔Transformer 的 **test 推理与提交文件生成**（读检查点，不重训）。

协议（沿用 `models/tcn/tcn_submit.py` 的形状，本谱系独立一份）：
  - test 特征**现算到内存**：粗段 `snap_feats.build_split("test", …, to_memory=True)`
    + 事件 `event_feats.build_flat("test", …, to_memory=True)`（都不落盘）
  - 推理**分块**（`--eval-batch`，显存护栏；模型内部还有 `--attn-budget-mb` 的子批护栏）
  - 逐成员预测立即落盘 npy（实例随时可能被关机，可 `--only` 续跑）
  - 最后**算术平均** / `TARGET_SCALE`（与 tcn/mlp 提交脚本同口径；cos 与尺度无关，
    除 `TARGET_SCALE` 只为与历史提交同量级）

⚠️ **必须给的三个输入**（缺一不可，都会显式断言）：
  1. `n_valid`（粗段每样本有效步数）——读出取 `nv−1`，占位值会静默读成 pad；
  2. 事件的 `offs`（每样本事件切片起点）——读出取 `nk−1`；
  3. 行序：`submission.csv` 的 sample_id 升序（粗段 / 事件 index / n_valid 三处交叉断言）。

⚠️ **内存**：test 现算 ≈ 4.35GB（粗段）+ ~1.6GB（事件）⇒ 需要**有卡模式**（无卡 2GiB 会被静默 OOM）。

用法：
    python models/tfm/tfm_submit.py --t2v --xattn --fold 0          # 单折（折 0 检查点）
    python models/tfm/tfm_submit.py --t2v --xattn --fold 0 --smoke  # 冒烟：输出到 _smoke 名
    python models/tfm/tfm_submit.py --t2v --xattn                   # 全部 f*.best.pt 成员平均
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

import event_feats as EF  # noqa: E402
import tfm_baseline as TB  # noqa: E402
from tabm_common import BASE  # noqa: E402

DEFAULT_CKPT_DIR = "snap_cache/ckpt_tfm_evt_xattn"
DEFAULT_PARTS = "snap_cache/submit_parts_tfm_xattn"
DEFAULT_OUT = "submission_tfm_xattn.parquet"


def load_test_memory():
    """test 输入现算到内存 → `(Xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, sids)`。

    行序以 `submission.csv` 的 sample_id 升序为准；三处 sample_id 交叉断言
    （粗段 index / 事件 index / n_valid 的 index）。
    """
    import snap_feats

    t0 = time.time()
    xc = snap_feats.build_split("test", None, False, False, to_memory=True)   # ⚠️ 5 参
    assert xc.shape[1:] == (TB.COARSE_T, TB.COARSE_C), f"粗段形状异常 {xc.shape}"

    sub = pl.read_csv(os.path.join(BASE, "submissions", "submission.csv"),
                      columns=["sample_id"]).sort("sample_id")
    sids = sub["sample_id"].to_numpy()
    assert len(sids) == xc.shape[0], f"submission.csv {len(sids)} 行 vs 特征 {xc.shape[0]} 行"
    assert (np.diff(sids) > 0).all(), "sample_id 非严格递增——行序异常"

    idx_p = os.path.join(BASE, "snap_cache", "test_snap_index.parquet")
    if not os.path.exists(idx_p):
        raise SystemExit(f"缺 {idx_p}——末步读出需要每样本的 n_valid，不能用占位值代替")
    idx = pl.read_parquet(idx_p).sort("sample_id")
    assert idx.height == len(sids) and (idx["sample_id"].to_numpy() == sids).all(), \
        f"{idx_p} 的 sample_id 与 submission.csv 行序不一致"
    nv = idx["n_valid"].to_numpy().astype(np.int64)
    assert (nv >= 1).all() and (nv <= TB.COARSE_T).all(), f"test n_valid 越界 [{nv.min()},{nv.max()}]"

    Vo, offs_o, Vt, offs_t, ev_idx = EF.build_flat("test", None, to_memory=True)
    assert ev_idx.height == len(sids) and (ev_idx["sample_id"].to_numpy() == sids).all(), \
        "事件 index 的 sample_id 与 submission.csv 行序不一致"
    assert Vo.shape[1] == EF.C_O and Vt.shape[1] == EF.C_T, f"事件通道数 {Vo.shape}/{Vt.shape}"
    nk_o, nk_t = np.diff(offs_o), np.diff(offs_t)
    assert (nk_o >= 1).all() and (nk_t >= 1).all(), "有空样本占 0 行（哨兵没写进去）"
    print(f"test 现算完成：粗段 {xc.shape} + order {Vo.shape}（中位 {int(np.median(nk_o))}）"
          f" + txn {Vt.shape}（中位 {int(np.median(nk_t))}）"
          f"（n_valid 中位 {int(np.median(nv))}，{time.time()-t0:.0f}s）", flush=True)
    return xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, sids


def list_ckpts(ckpt_dir, kind="best", fold=None):
    """→ sorted `[(折号, 路径)]`。`fold` 非 None 时只取该折。"""
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"检查点目录不存在：{ckpt_dir}\n  先跑 tfm_baseline.py --only-fold N")
    out = []
    for f in os.listdir(ckpt_dir):
        m = re.fullmatch(r"f(\d+)\.(best|last)\.pt", f)
        if m and m.group(2) == kind:
            out.append((int(m.group(1)), os.path.join(ckpt_dir, f)))
    out.sort()
    if fold is not None:
        out = [(fi, p) for fi, p in out if fi == fold]
    if not out:
        raise SystemExit(f"{ckpt_dir} 下没有可用的 f*.{kind}.pt"
                         + (f"（--fold {fold}）" if fold is not None else ""))
    return out


def main():
    p = argparse.ArgumentParser(description="三塔Transformer 提交（读检查点推理）")
    # —— 数据 / 产物 ——
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--ckpt-kind", choices=["best", "last"], default="best")
    p.add_argument("--fold", type=int, default=None, help="只用一个折的检查点（默认：全部 f*.best.pt 平均）")
    p.add_argument("--ckpt-path", default=None,
                   help="**直接指定检查点文件**（全量训练的 `full_e*.pt` 不匹配 f*.best.pt 模式）；"
                        "给了它就忽略 --ckpt-dir/--fold/--ckpt-kind/--only")
    p.add_argument("--parts-dir", default=DEFAULT_PARTS)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--only", type=int, default=None, help="只补指定折号的成员（续跑用）")
    p.add_argument("--smoke", action="store_true", help="冒烟：输出到 _smoke 名（不碰正式提交）")
    p.add_argument("--eval-batch", type=int, default=2048, help="推理分批行数（显存护栏）")
    # —— 架构：**必须与训练侧逐字一致**（否则指纹校验拒绝加载）——
    p.add_argument("--width", type=int, default=TB.WIDTH_DEFAULT)
    p.add_argument("--layers", type=int, default=TB.LAYERS_DEFAULT)
    p.add_argument("--heads", type=int, default=TB.HEADS_DEFAULT)
    p.add_argument("--ff-mult", type=int, default=TB.FF_MULT_DEFAULT)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--head-drop", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3, help="**只影响训练**，但进了指纹，推理侧要给一致的值")
    p.add_argument("--wd", type=float, default=3e-4, help="同上")
    p.add_argument("--batch", type=int, default=1024, help="同上")
    p.add_argument("--loss", choices=["mse", "cos"], default="cos", help="同上（默认与训练侧一致）")
    p.add_argument("--t2v", action="store_true", help="细段 Time2Vec——**必须与训练侧一致**")
    p.add_argument("--xattn", action="store_true", help="跨源注意力融合——**必须与训练侧一致**")
    p.add_argument("--attn-budget-mb", type=int, default=TB.BUDGET_MB_DEFAULT,
                   help="注意力子批预算（显存护栏，**不进指纹**、不改语义）")
    # ⚠️ 刻意**不加** --patience / --epochs：不进指纹、也不影响推理（CLAUDE.md #12 反向纪律）
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.ckpt_dir = args.ckpt_dir if os.path.isabs(args.ckpt_dir) else os.path.join(BASE, args.ckpt_dir)
    parts_dir = args.parts_dir if os.path.isabs(args.parts_dir) else os.path.join(BASE, args.parts_dir)
    os.makedirs(parts_dir, exist_ok=True)
    out_name = args.out.replace(".parquet", "_smoke.parquet") if args.smoke else args.out
    out_path = os.path.join(BASE, "submissions", out_name)

    if args.ckpt_path:
        cp = args.ckpt_path if os.path.isabs(args.ckpt_path) else os.path.join(BASE, args.ckpt_path)
        if not os.path.exists(cp):
            raise SystemExit(f"缺检查点 {cp}")
        ckpts = [(os.path.splitext(os.path.basename(cp))[0], cp)]      # tag = 文件名主干
    else:
        ckpts = [(f"f{fi}", q) for fi, q in list_ckpts(args.ckpt_dir, args.ckpt_kind, args.fold)]
        if args.only is not None:
            ckpts = [(t, q) for t, q in ckpts if t == f"f{args.only}"]
            if not ckpts:
                raise SystemExit(f"--only {args.only}：该折没有 f{args.only}.{args.ckpt_kind}.pt")

    print(f"device={device} 指纹={TB.arch_hash(args)} width={args.width} layers={args.layers} "
          f"heads={args.heads} t2v={args.t2v} xattn={args.xattn} 成员={len(ckpts)}（{args.ckpt_kind}）")
    print(f"检查点 ← {ckpts[0][1] if args.ckpt_path else args.ckpt_dir}"
          + (f"（--ckpt-path 指定）" if args.ckpt_path else ""))
    print(f"逐成员预测 → {parts_dir}；输出 → {out_path}")
    print("⚠️ 读出 = 逐样本 n_valid−1 / nk−1（取自 test index），不是数组末位")

    xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t, sids = load_test_memory()
    rows = np.arange(len(sids))
    for tag, path in ckpts:
        part_path = os.path.join(parts_dir, f"{tag}.npy")
        if os.path.exists(part_path):
            print(f"  [{tag}] 已有 {part_path}，跳过", flush=True)
            continue
        payload = TB.load_ckpt(path, args)                 # 指纹不符会报错，绝不静默
        model = TB.build_model(args, device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        pred = TB.eval_forward(model, xc, nv, Vo, offs_o, nk_o, Vt, offs_t, nk_t,
                               rows, args.eval_batch, device)
        np.save(part_path, pred)      # ⚠️ np.save 会追加 .npy（CLAUDE.md #1）：part_path 已带 .npy
        meta = payload.get("meta", {})
        print(f"  [{tag}] 推理完成（meta: cos={meta.get('cos', float('nan')):.5f} "
              f"loss={meta.get('loss', float('nan')):.5f} epoch={meta.get('epoch')}）"
              f"→ {part_path}", flush=True)

    files = sorted(f for f in os.listdir(parts_dir) if f.endswith(".npy"))
    arrs = [np.load(os.path.join(parts_dir, f)) for f in files]
    print(f"\n成员 {len(arrs)} 个：{' '.join(files)}")
    if len(arrs) > 1:
        cors = [float(np.corrcoef(arrs[i], arrs[j])[0, 1])
                for i in range(len(arrs)) for j in range(i + 1, len(arrs))]
        print(f"成员间相关 min={min(cors):.4f} mean={np.mean(cors):.4f} max={max(cors):.4f}")
    pred = np.mean(arrs, axis=0) / TB.TARGET_SCALE
    pl.DataFrame({"sample_id": sids, "prediction": pred.astype(np.float64)}).write_parquet(out_path)
    print(f"\n已写 {out_path}（{len(sids)} 行，pred std={pred.std():.6f}）")


if __name__ == "__main__":
    main()
