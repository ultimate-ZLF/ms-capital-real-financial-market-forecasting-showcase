"""集成提交：pred = p_tabm + α·p_snap（α 来自 blend_eval.py 的 OOF 拟合）。

test 侧两模型预测尺度与 OOF 侧一致（都是 12 折平均），α 可直接套用。
输出 submissions/submission_blend_v1.parquet + 诊断。
"""
import os

import numpy as np
import polars as pl

# 跨目录复用（models/tabm）——目录结构见 README
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tabm"))

from tabm_common import BASE, cos_score

ALPHA = 0.688  # blend_eval.py 拟合的 pooled α（逐月 median）


def main():
    tabm = pl.read_parquet(os.path.join(BASE, "submissions",
                                        "submission_tabm_v1.parquet"))
    snap = pl.read_parquet(os.path.join(BASE, "submissions",
                                        "submission_cnn_v1.parquet"))
    assert (tabm["sample_id"] == snap["sample_id"]).all()
    p = (tabm["prediction"].to_numpy() + ALPHA * snap["prediction"].to_numpy())

    print(f"α={ALPHA}")
    print(f"tabm test pred: mean={tabm['prediction'].mean():.6f} "
          f"std={tabm['prediction'].std():.6f}")
    print(f"snap test pred: mean={snap['prediction'].mean():.6f} "
          f"std={snap['prediction'].std():.6f}")
    print(f"blend test pred: mean={p.mean():.6f} std={p.std():.6f}")
    print(f"cos(tabm, snap) test = {cos_score(tabm['prediction'].to_numpy(), snap['prediction'].to_numpy()):.5f}")

    sub = pl.DataFrame({"sample_id": tabm["sample_id"],
                        "prediction": p.astype(np.float64)})
    out_path = os.path.join(BASE, "submissions", "submission_blend_v1.parquet")
    sub.write_parquet(out_path)
    print(f"已保存 {out_path}（{sub.height} 行）")


if __name__ == "__main__":
    main()
