"""公式基线提交 v3（加权方向 + 幅度调制）：
pred = (0.2·r(t_imb_2s) + 0.3·r(t_imb_w10) + 0.2·r(ofi_w10_norm) + 0.1·r(patience_diff))
       × (1 + 0.5·r(t_vol))
其中 r = 秩归一化到 [-1,1]。

验证（71 月）：
- 全样本 cos 月均 0.1044，最差月 0.0656，71/71 为正
- 时间外验证：前 36 月拟合权重 → 后 35 月 cos 0.0987（等权仅 0.0722）
- 权重结构（w10 > 2s ≈ ofi > patience）在前后两段数据上一致 → 结构稳定，非过拟合
- 预计 LB ≈ 0.104 × 0.83 ≈ 0.087（实测 LB 0.073）

⚠️ **重跑会静默覆盖 `submissions/submission_formula_v3.parquet`**（2026-09-21 标注）——
那是 FACTORS 提交记录表里 v3 的产物本身（LB 0.073 / 衰减 0.699），覆盖后无法再生成同一份历史文件。
公式时代已结束，现行提交一律走 `models/*/*_submit.py`。
"""
import os
import numpy as np
import polars as pl
from scipy.stats import rankdata

BASE = os.environ.get("MSC_BASE") or (
    r"D:\Kaggle\ms-capital-real-financial-market-forecasting"
    if os.name == "nt"
    else "/root/msdata"
)
OUT = os.path.join(BASE, "factors")

df = pl.read_parquet(os.path.join(OUT, "factors_test.parquet"))
print(f"test factors: {df.height} rows")

# (因子, 权重)。权重来自 71 月 OLS 比例四舍五入（2:3:2:1），经时间外验证
DIR_W = [("t_imb_2s", 0.2), ("t_imb_w10", 0.3), ("ofi_w10_norm", 0.2), ("patience_diff", 0.1)]
MOD_FACTOR = "t_vol"
K = 0.5

def ranknorm(x):
    """秩归一化到 [-1, 1]（零均值），NaN 保持 NaN。
    分母必须用 n-1 而非 r.max()-1（极端值并列时 r.max() < n，见 WALKTHROUGH）。"""
    out = np.full(len(x), np.nan)
    m = ~np.isnan(x)
    n = m.sum()
    if n > 1:
        r = rankdata(x[m])
        out[m] = 2 * (r - 1) / (n - 1) - 1
    return out

# 加权方向组合（缺失因子时对剩余权重重归一化）
rks = np.column_stack([ranknorm(df[f].to_numpy()) for f, _ in DIR_W])
wts = np.array([w for _, w in DIR_W])
combo = np.nansum(rks * wts, axis=1) / np.nansum(np.where(~np.isnan(rks), wts, 0), axis=1)
nan_frac = np.isnan(combo).mean()
combo = np.nan_to_num(combo, nan=0.0)

# 幅度调制
sig = ranknorm(df[MOD_FACTOR].to_numpy())
pred = combo * (1 + K * sig)

print(f"direction NaN fraction = {nan_frac:.5f}")
print(f"pred: mean={pred.mean():.4f} std={pred.std():.4f} "
      f"min={pred.min():.3f} max={pred.max():.3f}")

sub = pl.DataFrame({
    "sample_id": df["sample_id"],
    "prediction": pred,
})
# 提交文件一律 parquet，放 submissions/ 目录（目录约定，见 CLAUDE.md）
out_path = os.path.join(BASE, "submissions", "submission_formula_v3.parquet")
sub.write_parquet(out_path)
print(f"saved -> {out_path}")
print(sub.head(5))
