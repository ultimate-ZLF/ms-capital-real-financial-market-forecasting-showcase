"""cnn_model.py — 一代旧主干 snap-CNN 的**模型结构**（已归档）。

## 为什么在这里

**模型**代码与**数据管道**分家：
- 数据管道（缓存加载 / 折式 / Loader / 派生通道）在顶层 `seq/`，由 cnn / tcn / mlp
  三族共用（提交时四个提交脚本都在运行时 import 它）。
- 本模块只有 CNN 一族用的结构。`SnapCNN` 现在的使用者是**归档臂**
  `cnn_baseline.py` / `cnn_submit.py` 与其回归测试。

⚠️ 新模型**不要**复用本模块：`twotower_*`（v20+）与 `tcn` / `mlp` 各有自己的塔结构。

## τ 池化的 `legacy` 口径：看着像 bug，实测好 0.044——不要"顺手修"

`SnapCNN.encode` 在 `pool_tau>0` 时分子含**全部**块、分母只含有效块，口径不一致。
改成一致后（`tau_mode="mask"`）粗段两折均值从 0.10198 掉到 0.05763。
**机理已查清**（2026-09-17）：pad 块参与池化**本身是真实收益**，不是 bug。
见 WALKTHROUGH8 / WALKTHROUGH9 / CLAUDE.md #10。它依赖**三个耦合条件**：
**pad 在最新端 + pad 全零 + 卷积因果**——任何一个变了它就变成真 bug。
**改数据布局或卷积结构必须重跑这个对照。**
"""
import torch

from seq_common import F  # noqa: F401  （SnapCNN 的 n_feats 默认值）


def masked_pool(x, mask_down):
    """x:(B,C,14) mask_down:(B,14) → (B,C) masked mean（分母 clamp≥1）。"""
    return (x * mask_down.unsqueeze(1)).sum(-1) / mask_down.sum(-1).clamp(min=1).unsqueeze(1)



class _CausalConv(torch.nn.Module):
    """因果 Conv1d：左 pad k-1（torch 不支持非对称 padding，用 F.pad 实现）。

    输出位置 i 只依赖输入 ≤ 2i 的位置（stride=2 时）。
    """

    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.k = k
        self.conv = torch.nn.Conv1d(cin, cout, k, stride=stride, padding=0)

    def forward(self, x):
        return self.conv(torch.nn.functional.pad(x, (self.k - 1, 0)))


class SnapCNN(torch.nn.Module):
    """纯盘口序列 CNN：因果卷积（感受野 37 快照≈105s）+ masked mean pool + MLP head。

    T=224 → 4 层 stride-2 后时间维 14；mask 用 max_pool1d(16,16) 对齐。
    坑：pad 区特征全 0，池化必须用 mask 剔除（见 masked_pool）。
    """

    def __init__(self, n_feats=F, drop=0.1, head_drop=0.25, out=1, pool_tau=0,
                 tau_mode="legacy"):
        super().__init__()
        # τ 池化的两种口径。**默认 legacy**——它在两折上比 mask 高 0.044，
        # 且与历史结果（旧折式）一脉相承。机理**尚未查清**，完整实验记录见
        # WALKTHROUGH8.md；这里只记结论和使用注意。
        #
        #   "legacy"（默认）分子含【全部】块、分母只含有效块
        #           ⚠️ **已知脆弱**：效果依赖三个耦合条件同时成立——
        #              pad 在最新端 + pad 是全零 + 卷积是因果的。
        #              任何一个变了它都会从"意外有效的机制"变成真的 bug
        #              （2026-09-15 实测：把 pad 挪到最旧端后，同类做法掉 0.02）。
        #              重构数据布局或卷积结构时**必须重测**，不能假定它还在起作用。
        #   "mask"  分子分母同口径（口径上更"正确"），但实测低 0.044。
        #           保留作对照，不要当默认。
        #
        # 注：曾提出过第三种 "align"（权重从最新【有效】块起算），已证明与 "mask"
        # **恒等**——τ 衰减的起点在归一化加权平均里不可观测（常数因子抵消）。
        # 该结论由 test_cnn_model.py 的 test_tau_modes 钉住。
        assert tau_mode in ("legacy", "mask"), tau_mode
        self.tau_mode = tau_mode
        ch = [(n_feats, 64, 5), (64, 128, 5), (128, 256, 3), (256, 512, 3)]
        layers = []
        for cin, cout, k in ch:
            layers += [
                _CausalConv(cin, cout, k, stride=2),
                torch.nn.BatchNorm1d(cout),
                torch.nn.GELU(),
                torch.nn.Dropout(drop),
            ]
        self.conv = torch.nn.Sequential(*layers)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Dropout(head_drop),
            torch.nn.Linear(256, out),
        )
        self.pool_tau = pool_tau  # >0：时间衰减加权池化（τ 以步为单位，越新权重越大）

    def encode(self, x, mask):
        """x:(B,F,T) f32, mask:(B,T) bool → (B,512) 池化后的序列表示（forward = encode + head）。"""
        h = self.conv(x)                                     # (B,512,nb)
        nb = h.shape[-1]
        mask_down = torch.nn.functional.max_pool1d(
            mask.float(), kernel_size=16, stride=16)         # (B,nb)
        if self.pool_tau > 0:
            block_tau = self.pool_tau / 16.0
            pos = torch.arange(nb, device=x.device).float()          # 块号 0=最旧
            w = torch.exp(-(nb - 1 - pos) / block_tau)               # 越靠后权重越大
            if self.tau_mode == "legacy":
                # 分子含全部块（**包括 mask 标为无效的**）、分母只含有效块。口径不一致，
                # 但实测显著更好（两折 +0.044）——机理未查清，见 WALKTHROUGH8.md。
                denom = (mask_down * w[None, :]).sum(-1).clamp(min=1e-6)
                h = (h * w[None, None, :]).sum(-1) / denom.unsqueeze(1)
            else:  # "mask"：分子分母同口径（对照用，实测更低）
                mw = mask_down * w[None, :]
                h = (h * mw.unsqueeze(1)).sum(-1) / \
                    mw.sum(-1).clamp(min=1e-6).unsqueeze(1)
        else:
            h = masked_pool(h, mask_down)                    # (B,512)
        return h

    def forward(self, x, mask):
        """x:(B,F,T) f32, mask:(B,T) bool → (B,1)。"""
        h = self.encode(x, mask)
        return self.head(h)                                  # (B,1)
