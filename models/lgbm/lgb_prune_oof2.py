"""lgb_prune_oof2.py — top-150 剪枝集 OOF（逐折早停协议，与 lgb_newfeat2.py 一致）。

修正：lgb_prune_submit.py 的 OOF 是固定 197 轮无早停版（协议不一致，混合拟合被劣化）。
本脚本只重跑 OOF 部分；test 预测继续用 submission_lgb_v5.parquet。
"""
import glob
import os

import lightgbm as lgb
import numpy as np
import polars as pl

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
OUT = os.path.join(BASE, "factors")

from lgb_prune_submit import (ALL_FEATS, NEW1_FEATS, NEW2_FEATS, NEW3_FEATS,
                              PAIRS, prep)

K = 150
PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.05, num_leaves=63,
    min_child_samples=500, feature_fraction=0.5,
    bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=10.0, verbose=-1, seed=42,
)


def main():
    gains = np.load(os.path.join(OUT, "lgb_prune_gains.npy"))
    order = np.argsort(gains)[::-1]
    keep = [ALL_FEATS[i] for i in order[:K]]
    idx = [ALL_FEATS.index(f) for f in keep]

    files = sorted(glob.glob(os.path.join(OUT, "factors_m*.parquet")))
    train = pl.concat([pl.read_parquet(f) for f in files])
    new1 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new_m*.parquet")))])
    new2 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new2_m*.parquet")))])
    new3 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "new3_m*.parquet")))])
    train = (train.join(new1.select(["sample_id", "month"] + NEW1_FEATS), on=["sample_id", "month"], how="left")
                  .join(new2.select(["sample_id", "month"] + NEW2_FEATS), on=["sample_id", "month"], how="left")
                  .join(new3.select(["sample_id", "month"] + NEW3_FEATS), on=["sample_id", "month"], how="left"))
    train = train.with_columns([
        (pl.col(a) * pl.col(b)).alias(f"xp_{a}__{b}") for a, b in PAIRS
    ])
    X = prep(train)[:, idx]
    y = train["target"].to_numpy()
    marr = train["month"].to_numpy()
    sids = train["sample_id"].to_numpy()
    months = sorted(train["month"].unique().to_list())

    VAL = months[2::6]
    oof_rows = []
    coses = []
    for vm in VAL:
        tr, va = marr != vm, marr == vm
        dtr = lgb.Dataset(X[tr], y[tr])
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(PARAMS, dtr, num_boost_round=800, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = m.predict(X[va], num_iteration=m.best_iteration)
        yv = y[va]
        cos = float((pred * yv).sum() / np.sqrt((pred**2).sum() * (yv**2).sum()))
        coses.append(cos)
        oof_rows.append(pl.DataFrame({
            "sample_id": sids[va], "month": marr[va],
            "pred": pred.astype(np.float32)}))
        print(f"[fold {vm}] cos={cos:.5f} iters={m.best_iteration}", flush=True)

    print(f"\n早停协议 OOF：mean={np.mean(coses):.5f} std={np.std(coses):.5f}")
    oof = pl.concat(oof_rows)
    oof.write_parquet(os.path.join(OUT, "lgb_prune_oof.parquet"))
    print(f"OOF 已覆盖 factors/lgb_prune_oof.parquet（{oof.height} 行）", flush=True)


if __name__ == "__main__":
    main()
