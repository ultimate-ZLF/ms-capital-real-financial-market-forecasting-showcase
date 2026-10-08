# 限价订单簿序列建模 —— Kaggle *MSCapital Real Financial Market Forecasting*

对一个**模拟限价订单簿**数据集做未来价格走势预测。本项目覆盖从原始四张表的解析、
序列缓存管线，到三塔 Transformer 主干与集成的完整实现。

- 比赛：[MSCapital – Real Financial Market Forecasting](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting)
- 指标：`cos(prediction, target)`
- 数据规模：71 个月、约 125.8 万条样本

## 任务与数据

每个 `sample_id` 是一段**独立的模拟交易片段**，在 predict 时刻结束。原始数据为四张表：

| 表 | 窗口 | 粒度 | 内容 |
|---|---|---|---|
| `market` | 600s（约 176 行/片段） | ~3s 快照（时间有 ±0.1s 抖动） | 盘口 1/2 档价格与量、窗口成交均价/量/笔数 |
| `order` | 60s（约 135 条/片段） | 事件级 | 订单 price / volume / side / order_action |
| `transaction` | 60s（约 83 笔/片段） | 事件级 | 成交 price / volume / side |
| `label` | — | 逐样本 | `month`(0–70) / `sample_id` / `target`（未来价格变化） |

关键的数据性质（决定了后面所有设计）：

- **月份是模拟器参数，不是时间戳**。`target` 的逐月标准差在 2.58e-3 ~ 5.0e-3 之间波动，
  即不同月份属于不同 regime。因此**不能按样本随机切分训练/验证**，否则同月样本会同时出现在
  两侧、造成 regime 泄漏使 CV 偏乐观。本项目一律使用**月交错 6 折**（折 k 验证 `months[k::6]`，
  6 折覆盖全部 71 个月）。
- 样本之间相互独立（做过分段连续性检验），但同一 episode 内部的时序结构是有效的建模信号。

## 方法

### 1. 表格因子路线

从四张原始表抽取标量因子（盘口失衡、订单流不平衡、成交强度、价差结构等），
先做**逐月**的 IC / 稳定性检验再入库，然后喂给强正则的 GBDT 与表格神经网络。

- `models/lgbm/` —— LightGBM 基线，因子筛选与消融
- `models/tabm/` —— TabM（表格隐式集成），含数值特征的分箱变体与 tick 单位化预处理

### 2. 序列路线

不依赖手工因子，直接对订单簿序列建模。先由 `seq/` 把原始表构筑成两份对齐好的定长张量缓存：

- **粗段** `(N, 224, 15)` —— 600s 盘口快照 + Δt
- **细段** `(N, 60, 16)` —— 最近 60s 的订单 / 成交事件流

两族缓存都做**逐样本**的有效长度标记（pad 在数组最新端），主干为 `models/tfm/` 的
**三塔 Transformer**：market / order 事件 / txn 事件三座塔各自做浅层 pre-LN 编码
（多头自注意力 pad-masked + MLP 残差块），末步读出后送进 head。

**Time2Vec**：两条事件塔各挂一个 δt 编码，输出维 = `width`，**直接叠加到 token embedding 上**
（`h = stem(x 去掉 dt) + t2v(τ)`，周期项频率与相位全可学习）。粗段塔不加。

另有**跨源注意力融合**变体（`--xattn`）：market 塔读出态作 Query，对 order 塔的块输出做
一次 cross-attention 得到 dynamic latent，再以前缀 token 的形式喂进 txn 塔。

训练脚本支持逐折检查点续跑（`state_dict` + 配置指纹校验，指纹覆盖所有会改变学到的权重的
配置项，读侧强制断言，避免误用其它折/其它配置的权重）。

### 3. 集成

`models/blend/` 对表格与序列路线做加权混合。权重来自 OOF 上的拟合，
但**拟合权重本身要过验证闸门**（历史上出现过 CV 上更优、LB 上更差的情形）。

## 目录结构

```
├── data_prep/    # 原始表转换 + 表格因子生成
├── seq/          # 序列数据管道
│   ├── snap_feats.py      # 粗段缓存构建
│   ├── flow_feats.py      # 细段缓存构建
│   ├── seq_common.py      # 加载器 / 折式 / DataLoader / 通道常量
│   └── check_*.py         # 缓存结构体检与信号体检
├── models/
│   ├── lgbm/  tabm/  cnn/  mlp/  tcn/  tfm/  blend/
└── analysis/     # 探索性分析脚本与诊断图
```

**分层约定**：`data_prep/` 只放原始表转换与表格因子；`seq/` 只放序列管道；
`models/*/` 只放模型本身。判断标准是「会被两个以上模型族 import 的代码，不属于任何一个模型目录」
——所以共享的序列管道被提到了顶层，而不是寄居在某个模型目录下。

### 数据与代码分离

数据目录（`train/ test/ factors/ snap_cache/ flow_cache/ submissions/`）**不在项目目录内**，
统一挂在数据根下，由各脚本顶部的 `BASE` 常量解析（`MSC_BASE` 环境变量优先，否则取平台默认值）。
本仓库**不含数据**——竞赛数据需从 Kaggle 赛题页获取后按上述布局放置。
