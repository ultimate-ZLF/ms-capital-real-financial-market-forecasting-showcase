"""blend3_submit.py — 三方集成提交（unit() L2 归一混合，LB142 思路自实现）。

用法：
  python blend3_submit.py [lgb_sub_file] [alpha] [beta] [out_name] [tabm_sub_file] [cnn_sub_file]
  pred = unit(tabm) + alpha·unit(cnn) + beta·unit(lgb)

⚠️ **默认参数是老的 v11 配方**（2026-09-21 标注）——无参运行会用 lgb_v3 / tabm_v1 / cnn_v1
写出 `submission_blend_v3.parquet`，**覆盖那份历史提交**。现行配方一律显式传参。

**当前最优 v21（LB 0.133）的实际命令**（配方已用本地提交文件逐位反推验证，max|Δ|=1.7e-18）：

  python blend3_submit.py submission_lgb_v5.parquet 0.5 0.25 submission_blend_v8.parquet \\
         submission_tabm2emb_v2.parquet submission_twotower_cnn.parquet

"""
import os
import sys

import numpy as np
import polars as pl

# 跨目录复用（models/tabm）——目录结构见 README
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tabm"))

from tabm_common import BASE

LGB_FILE = sys.argv[1] if len(sys.argv) > 1 else "submission_lgb_v3.parquet"
A = float(sys.argv[2]) if len(sys.argv) > 2 else 0.875
B = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
OUT_NAME = sys.argv[4] if len(sys.argv) > 4 else "submission_blend_v3.parquet"
TABM_FILE = sys.argv[5] if len(sys.argv) > 5 else "submission_tabm_v1.parquet"
# CNN 成员文件。默认值与历史一致（当前 LB 0.132 用的就是它）；换折式后
# cnn_submit.py 输出叫 submission_cnn_k6.parquet，要用它须显式传第 6 位参数。
SNAP_FILE = sys.argv[6] if len(sys.argv) > 6 else "submission_cnn_v1.parquet"


def unit(x: np.ndarray) -> np.ndarray:
    xc = x - x.mean()
    n = np.sqrt((xc ** 2).sum())
    return xc / n if n > 0 else np.zeros_like(xc)


def load_sub(name):
    df = pl.read_parquet(os.path.join(BASE, "submissions", name))
    return df.sort("sample_id")


def main():
    tabm = load_sub(TABM_FILE)
    snap = load_sub(SNAP_FILE)
    lgbm = load_sub(LGB_FILE)
    assert tabm.height == snap.height == lgbm.height, "test 行数不一致"
    sids = tabm["sample_id"]
    assert (sids.to_numpy() == snap["sample_id"].to_numpy()).all()
    assert (sids.to_numpy() == lgbm["sample_id"].to_numpy()).all()

    pt = tabm["prediction"].to_numpy()
    ps = snap["prediction"].to_numpy()
    pl_ = lgbm["prediction"].to_numpy()
    c = np.corrcoef(np.vstack([pt, ps, pl_]))
    print(f"test 预测相关: tabm×snap={c[0,1]:.4f} tabm×lgbm={c[0,2]:.4f} "
          f"snap×lgbm={c[1,2]:.4f}")

    pred = unit(pt) + A * unit(ps) + B * unit(pl_)
    print(f"blend α={A} β={B}: mean={pred.mean():.5f} std={pred.std():.5f} "
          f"min={pred.min():.5f} max={pred.max():.5f}")

    sub = pl.DataFrame({"sample_id": sids, "prediction": pred})
    out = os.path.join(BASE, "submissions", OUT_NAME)
    sub.write_parquet(out)
    print(f"saved -> {out}")
    print(sub.head(5))


if __name__ == "__main__":
    main()
