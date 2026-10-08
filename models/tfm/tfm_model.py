"""tfm_model.py — **浅层 Transformer 主干**（三塔：market | order 事件 | txn 事件）。

C-19 的 **B 路**：把 order / txn 当**事件序列**（不是 1 秒桶），用注意力做时间混合。
立论（D-11）：卷积 `y[i] = Σ_k w[k]·x[i+k]` 的权重**只依赖下标偏移**，无法按时间差做选择；
注意力 `α[i,j] = softmax(q_i·k_j)` 的分数是**内容的函数**——`dt` / `t_pos` 就在输入通道里
⇒ 分数自带时间感知，**不需要位置编码、也不需要显式相对时间 bias**。

结构（三塔**同一个类**，只有输入通道数与批内长度不同）：

```
塔（×3）  (B, L, C_in)   ← event_batch.batch_* 给的 (B, C, T) 在 forward 里转置
          → 干：Linear(C_in → width)   ← token embedding
          → **（可选，只细段）去掉 `dt_log`，再加 `t2v(τ)`**：`h = stem(x) + t2v(dt_log)`
            （`t2v_add=True` 时；τ 编码**叠加**到 token embedding 上，dim = width）
          → N 个 pre-LN 编码块：LN → 多头自注意力(pad-masked) → +残差；LN → MLP → +残差
          → 末端 LayerNorm
          → **末步读出**：逐样本取位置 `n−1`（最后一个真实事件 / 最后一个真实快照）
          → Linear(width → out_dim) + GELU + Dropout                    → h_t (B, 256)
合        concat[h_market; h_order; h_txn] (768) → Linear(768→256)+GELU+Drop+Linear(256→1)
```

**变体（`--xattn`，2026-10-02 用户指定）**：把"三塔读出在 head 处 concat"换成**层级跨源注意力**
——阶段 1 `z = CrossAttn(q = market 读出态, K/V = order 全部 token)`（z 是逐样本的 dynamic latent）；
阶段 2 把 z 作**前缀 token** 喂进 txn 塔；head 只消费 txn 读出。见 `ThreeTowerTFMXAttn`。

**读出 = `n−1`**（与 TCN 同约定），但语义与 TCN 不同：TCN 的因果窗只向前看有限步
（RF 255 / 1023），这里读出点**聚合的是全窗**（双向 + pad-mask）——正面回应 A-03/C-03
（老 CNN 的粗段 RF 只覆盖 17% 的窗口）。

## 两处刻意的设计

1. **不加位置编码**：位置信息全在内容里（粗段 `rel_t`/`dt`，事件 `t_pos`/`dt_log`）。
   - 事件轴的"第 i 个事件"**跨样本不对齐**（B-06 的机理：事件率跨 20 倍），
     给它加索引位置编码 = 把每条样本各自扭曲的时间轴当成同一条，正是 B-06 判死的做法；
   - 粗段的"第 i 个快照"虽大致对齐，但缺 11.6% 时间轴、最长空洞 310s（`DATA.md`）⇒
     索引 ≠ 时间，`rel_t` 才是时间。
2. **pad 用 mask 排除（不是靠因果）**：`F.scaled_dot_product_attention` 的 bool `attn_mask`
   **True = 允许参与**（⚠️ 与 `nn.MultiheadAttention.key_padding_mask` 的 "True = 忽略"
   **正好相反**，写反了不报错、只会静默把 pad 喂进注意力）——`test_tfm.py` 有一条
   对拍手写 softmax 参照的用例钉死它。

## 显存护栏（分块 + 检查点）

注意力是否物化 `B×H×L²` 取决于 SDPA 选到的后端（f32 在 CUDA 上**可能**落到 math 后端）。
为把峰值显存**按构造**钉住：前向按批切成 `B_sub = budget / (H·L²·4)` 的子批；
训练时子批走 `torch.utils.checkpoint`（非重入版）⇒ 峰值只由**单个子批**决定、
不随子批数累加（否则物化后端下"每个子批各留一份 softmax 概率给 backward"会把省下的全还回去）。

⚠️ 分块**不改语义**：LN / 注意力 / MLP 全部逐样本、无跨样本统计量（这是它和 BatchNorm
的区别——BN 的批统计量会随切批改变，LN 不会），切批只改浮点 GEMM 的切分。
位置参数 `budget_mb` **不进配置指纹**（与 `eval_chunk` 同类：不改"学出什么权重"）。
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(MODELS), "seq"))   # 序列数据管道（顶层 seq/）

from event_feats import C_O, C_T          # noqa: E402  事件两条流的通道数（单一来源）
from seq_common import F as COARSE_C      # noqa: E402  粗段基通道数（单一来源）

OUT_DIM = 256          # 每塔读出维（与 TCN / 双塔CNN 一致：3×256 → head）
HEAD_DIM = 256
WIDTH_DEFAULT = 64     # d_model（与 TCN 的 width 同值，便于对照参数量）
LAYERS_DEFAULT = 2     # **浅层**：注意力没有 RF 限制，全窗在第 1 层就可达
HEADS_DEFAULT = 4
FF_MULT_DEFAULT = 4    # MLP 隐层 = ff_mult × width
BUDGET_MB_DEFAULT = 2048   # 每个子批的注意力矩阵预算（MiB）


class Time2Vec(nn.Module):
    """Time2Vec（Kazemi et al. 2019）：把标量时间 τ 映射成 `k` 维向量

        `[ω₀·τ + φ₀,  sin(ω₁·τ + φ₁), …, sin(ω_{k−1}·τ + φ_{k−1})]`

    用法（用户 2026-10-02 定）：**输出维度 = `d_model`，直接叠加到 token embedding 上**
    （`h = stem(x) + t2v(τ)`，像位置编码那样加），**不是**把 τ 当成新通道 concat 进输入。

    **只用于细段的两条事件塔**（用户 2026-10-01 指定）：τ = 基通道里的 `dt_log`
    （= `log1p(距本流上一个事件的秒数)`）；**并去掉原来的 `dt_log` 通道**
    （"有 time2vec 就不要 dt"——去 dt 在 `TFTower._embed` 里做，信息不丢：
    第 0 条 `ω₀τ+φ₀` 就是 dt 的仿射函数，而事件的绝对时间仍在 `t_pos` 通道）。

    ⚠️ 两条设计依据：
    1. **它必须住在模型里、不能预先算进缓存**——`ω/φ` 是**可学习**参数（这正是它区别于
       手工周期特征的地方），缓存是 f16 静态数组，放不进可学习的东西。
    2. **τ 不反变换回秒**：`dt_log` 在缓存里是 f16，`expm1` 会把量化误差放大
       （dt=60s 时 f16 的 dt_log 分辨率 ≈0.002 ⇒ 反变换后 ≈0.12s）；而周期项的频率
       `ω` 本身可学习，**尺度由它吸收**。要换成秒只需换 τ 的来源，架构不变。
    """

    def __init__(self, k):
        super().__init__()
        assert k >= 2, "Time2Vec 至少要 1 条线性项 + 1 条周期项"
        self.k = k
        self.w0 = nn.Parameter(torch.randn(1))         # 线性项频率
        self.b0 = nn.Parameter(torch.randn(1))         # 线性项偏置
        self.w = nn.Parameter(torch.randn(k - 1))      # 周期项频率
        self.b = nn.Parameter(torch.randn(k - 1))      # 周期项相位

    def forward(self, tau):
        """`tau (..., 1)` → `(..., k)`（第 0 维是线性项，其余是 `sin`）。"""
        return torch.cat([self.w0 * tau + self.b0,
                          torch.sin(tau * self.w + self.b)], dim=-1)


class TFBlock(nn.Module):
    """pre-LN 编码块：`x + drop(proj(attn(ln1(x))))`，`x + drop(mlp(ln2(x)))`。

    ⚠️ **mask 约定**：本块用 `F.scaled_dot_product_attention`，它的 bool `attn_mask` 里
    **True = 允许参与**（torch 2.5 实测；与 `nn.MultiheadAttention.key_padding_mask` 的
    "True = 忽略"相反）。写反不报错——只会把 pad 当成真实事件喂进去。
    """

    def __init__(self, d, heads, ff_mult, drop):
        super().__init__()
        assert d % heads == 0, f"width {d} 不能被 heads {heads} 整除"
        self.d, self.h, self.dh = d, heads, d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                 nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, keep):
        """`x (B,L,d)`；`keep (B,1,1,L)` bool，**True = 该位置参与**。"""
        B, L, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=keep)
        a = a.transpose(1, 2).reshape(B, L, d)
        x = x + self.drop(self.proj(a))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


def _keep_mask_last(L, last, device):
    """`(B,1,1,L)` bool，**True = 参与**（SDPA 约定）；有效位 = `0..last[b]`（pad 恒在末尾）。

    ⚠️ 全项目唯一构造 mask 的地方——读出下标 / 前缀 token 一改，mask 必须跟着改；
    两条塔的测试（pad 灌毒不变、截断等价）都盯着它。
    """
    return (torch.arange(L, device=device)[None, :] <= last[:, None])[:, None, None, :]


class TFTower(nn.Module):
    """单塔：干 → N 个 pre-LN 块 → 末端 LN → **末步读出**（逐样本 `n−1`）→ 投影到 `out_dim`。

    读出靠 `n`（有效长度）定位而不是数组末尾——pad 在末尾，`n−1` 永远是最后一个真实位置。

    对外暴露两个口，供 `ThreeTowerTFMXAttn` 取中间态（`forward` 就是两者的复合，
    **非 xattn 路径逐位不变**）：
      - `states(x, n, prefix=None)` → 块输出 `(B, L(+1), width)` 与末位下标（不读出、不投影）
      - `readout(h, last)`          → 逐样本 gather →（有 proj 时）投影
    """

    def __init__(self, c_in, width=WIDTH_DEFAULT, layers=LAYERS_DEFAULT, heads=HEADS_DEFAULT,
                 ff_mult=FF_MULT_DEFAULT, drop=0.2, out_dim=OUT_DIM,
                 t2v_add=False, dt_index=0, with_proj=True):
        super().__init__()
        self.c_in, self.width, self.layers, self.heads = c_in, width, layers, heads
        # Time2Vec（可选，用户 2026-10-02）：**输出维度 = width，叠加到 token embedding 上**。
        # τ 取输入里第 `dt_index` 条通道（细段 = `dt_log`），**该通道同时从输入里去掉**
        # （用户 2026-10-01："有 time2vec 就不要 dt"）⇒ stem 的输入是 `C_in − 1`。
        self.t2v_add, self.dt_index = t2v_add, dt_index
        self.t2v = Time2Vec(width) if t2v_add else None
        self.stem_in = c_in - 1 if t2v_add else c_in
        self.stem = nn.Linear(self.stem_in, width)
        self.blocks = nn.ModuleList([TFBlock(width, heads, ff_mult, drop)
                                     for _ in range(layers)])
        self.ln_f = nn.LayerNorm(width)
        # `with_proj=False`：xattn 臂里 market/order 的读出态不直接进 head，不需要 256 维投影
        # ⇒ 不留"死参数"（有梯度检查守着：这些塔的 proj 参数一个都没有）。
        self.proj = nn.Sequential(nn.Linear(width, out_dim), nn.GELU(),
                                  nn.Dropout(drop)) if with_proj else None

    def _embed(self, x):
        """→ token embedding `(B,L,width)`：干投影；开 Time2Vec 时**去掉 dt 通道**并加上 `t2v(τ)`。

        `h = stem(x_去dt) + t2v(τ)`——**叠加**，不是把 τ 的编码 concat 进输入
        （用户 2026-10-02 定；这也是 Time2Vec 原论文的用法，与位置编码相加同构）。
        信息不丢：第 0 条 `ω₀·τ+φ₀` 就是 dt 的**仿射函数**（线性项），其余是周期基；
        "事件在窗口内的绝对时间"仍在 `t_pos` 通道里。
        """
        if self.t2v is None:
            return self.stem(x)
        d = self.dt_index
        tau = x[..., d:d + 1]
        rest = torch.cat([x[..., :d], x[..., d + 1:]], dim=-1)   # 去掉 dt，其余顺序不变
        return self.stem(rest) + self.t2v(tau)

    def states(self, x, n, prefix=None):
        """→ `(h (B,L2,width), last (B,))`：干 →（可选前缀）→ 各块 → 块输出 + 末位下标。

        `prefix (B,width)` 作为**第 0 个 token 前置**（xattn 臂的 z）⇒ 有效位变成 `0..n`
        （z 一个 + 真实 token `n` 个），**读出下标顺移为 `n`**、mask 相应右移。
        """
        h = self._embed(x)
        if prefix is not None:
            h = torch.cat([prefix[:, None, :], h], dim=1)
        L2 = h.shape[1]
        last = n if prefix is not None else (n - 1).clamp_min(0)   # n≥1 恒成立（哨兵保证）
        keep = _keep_mask_last(L2, last, x.device)
        for blk in self.blocks:
            h = blk(h, keep)
        return h, last

    def readout(self, h, last):
        """逐样本 gather 末位 →（建了 proj 时）投影到 `out_dim`。"""
        B = h.shape[0]
        x = self.ln_f(h)[torch.arange(B, device=h.device), last]
        return self.proj(x) if self.proj is not None else x

    def _one(self, x, n):
        """单子批前向：`x (B,L,C) f32`，`n (B,) int64`（1 ≤ n ≤ L）。"""
        return self.readout(*self.states(x, n))

    def _chunked(self, fn, budget_bytes, *args):
        """按 `budget_bytes` 切子批调用 `fn(*args)`（第 1 个参数是 `(B,L,…)`，其余按 batch 维同切）。

        子批大小 = `budget / (H·L²·4)`；训练态走**非重入 checkpoint** ⇒ backward 逐子批重算，
        峰值只由单个子批决定（见模块头「显存护栏」）。返回 tuple 时逐项 `cat`。
        """
        B, L = args[0].shape[0], args[0].shape[1]
        sub = int(max(1, min(B, budget_bytes // max(1, self.heads * L * L * 4))))
        if sub >= B:
            return fn(*args)
        outs = []
        for i in range(0, B, sub):
            sl = [a[i:i + sub] for a in args]
            if self.training and torch.is_grad_enabled():
                # 非重入检查点：preserve_rng_state 默认 True ⇒ 重算时 dropout 掩码一致（可复现）。
                outs.append(torch.utils.checkpoint.checkpoint(fn, *sl, use_reentrant=False))
            else:
                outs.append(fn(*sl))
        if isinstance(outs[0], tuple):
            return tuple(torch.cat([o[k] for o in outs], 0) for k in range(len(outs[0])))
        return torch.cat(outs, 0)

    def forward(self, x, n, budget_bytes=BUDGET_MB_DEFAULT * 2 ** 20):
        """`x (B,L,C)` → `(B, out_dim)`。"""
        return self._chunked(self._one, budget_bytes, x, n)


class ThreeTowerTFM(nn.Module):
    """market | order 事件 | transaction 事件，三塔各自**末步读出**后在 head 处 concat。

    输入与 `seq/event_batch.py` 的 `batch_coarse` / `batch_events` **逐字对齐**（都是 (B,C,T)），
    内部转成 (B,T,C)。三塔唯一的差别是输入通道数与批内长度（同一个类，不是复制品）。
    """

    def __init__(self, width=WIDTH_DEFAULT, layers=LAYERS_DEFAULT, heads=HEADS_DEFAULT,
                 ff_mult=FF_MULT_DEFAULT, drop=0.2, head_drop=0.3, out=1,
                 head_dim=HEAD_DIM, out_dim=OUT_DIM, in_coarse=COARSE_C, in_o=C_O, in_t=C_T,
                 budget_mb=BUDGET_MB_DEFAULT, t2v_add=False, dt_index_o=0, dt_index_t=0):
        super().__init__()
        # ⚠️ Time2Vec **只给细段**（两条事件塔；用户 2026-10-01 指定）——粗段塔不加。
        self.tower_c = TFTower(in_coarse, width, layers, heads, ff_mult, drop, out_dim)
        self.tower_o = TFTower(in_o, width, layers, heads, ff_mult, drop, out_dim,
                               t2v_add=t2v_add, dt_index=dt_index_o)
        self.tower_t = TFTower(in_t, width, layers, heads, ff_mult, drop, out_dim,
                               t2v_add=t2v_add, dt_index=dt_index_t)
        self.head = nn.Sequential(
            nn.Linear(3 * out_dim, head_dim), nn.GELU(),
            nn.Dropout(head_drop), nn.Linear(head_dim, out),
        )
        self.budget_bytes = int(budget_mb * 2 ** 20)

    def forward(self, x_c, nv, x_o, nko, x_t, nkt):
        """三塔都走**逐样本末步读出**（`nv−1` / `nko−1` / `nkt−1`）；pad 由 mask 排除。"""
        h = torch.cat([self.tower_c(x_c.transpose(1, 2), nv, self.budget_bytes),
                       self.tower_o(x_o.transpose(1, 2), nko, self.budget_bytes),
                       self.tower_t(x_t.transpose(1, 2), nkt, self.budget_bytes)], dim=1)
        return self.head(h)


class CrossAttn(nn.Module):
    """**市场读出态（q）对 order 全部 token（K/V，pad-masked）做一次多头跨源注意力 → z**。

    只在**读出点**做一次（q 是单向量）：Q 形状 `(B,1,width)`，注意力矩阵 `(B,H,1,L_o)`
    ⇒ 成本 ≈ `L_o`/样本，比"给 market 塔加整层全位置跨注意力"省 224×（用户 2026-10-02 第 4 条）。
    信息上不失一般性：q 已经聚合了 market 全窗（塔内 2 个双向自注意力块），
    见 `TFTower.states` 的读出约定。
    """

    def __init__(self, d, heads, drop):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.ln_q, self.ln_kv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        self.ln_o = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, q, kv, keep_kv):
        """`q (B,d)`；`kv (B,L,d)`；`keep_kv (B,1,1,L)` **True=参与**（SDPA 约定）。"""
        B, L, d = kv.shape
        kk = self.ln_kv(kv)
        k = self.wk(kk).view(B, L, self.h, self.dh).transpose(1, 2)      # (B,h,L,dh)
        v = self.wv(kk).view(B, L, self.h, self.dh).transpose(1, 2)
        qq = self.wq(self.ln_q(q)).view(B, 1, self.h, self.dh).transpose(1, 2)   # (B,h,1,dh)
        a = F.scaled_dot_product_attention(qq, k, v, attn_mask=keep_kv)         # (B,h,1,dh)
        a = a.transpose(1, 2).reshape(B, d)
        return self.ln_o(q + self.drop(self.proj(a)))                    # 残差接 market 读出态


class ThreeTowerTFMXAttn(nn.Module):
    """**跨源注意力融合**臂（用户 2026-10-02 定）：head **只走融合通路**。

        阶段 1  q   = market 塔读出态 @ `nv−1`（已聚合全窗）        (B, width)
                K/V = order 塔的**块输出**（全 token，pad-masked）  (B, L_o, width)
                z   = CrossAttn(q, K/V)                            (B, width)   ← dynamic latent
        阶段 2  txn 塔：输入 = **[z] 前置** + txn token → 块 → 末位读出 → proj(256)
        head    Linear(256→head_dim) + GELU + Drop → Linear(head_dim→1)
                （**不再** concat 三塔读出——市场/order 只能通过 z 影响输出）

    ⚠️ 与 `ThreeTowerTFM` 的差别因此**不止"多一条通路"**：这是一次**融合拓扑**的对照
    （head-concat vs 层级跨源注意力）。市场/order 塔的 `proj` 在本臂里不建（`with_proj=False`），
    所以本臂的参数量比 v3 **更小**——涨分不会来自容量。
    """

    def __init__(self, width=WIDTH_DEFAULT, layers=LAYERS_DEFAULT, heads=HEADS_DEFAULT,
                 ff_mult=FF_MULT_DEFAULT, drop=0.2, head_drop=0.3, out=1,
                 head_dim=HEAD_DIM, out_dim=OUT_DIM, in_coarse=COARSE_C, in_o=C_O, in_t=C_T,
                 budget_mb=BUDGET_MB_DEFAULT, t2v_add=False, dt_index_o=0, dt_index_t=0):
        super().__init__()
        self.tower_c = TFTower(in_coarse, width, layers, heads, ff_mult, drop, out_dim,
                               with_proj=False)
        self.tower_o = TFTower(in_o, width, layers, heads, ff_mult, drop, out_dim,
                               t2v_add=t2v_add, dt_index=dt_index_o, with_proj=False)
        self.tower_t = TFTower(in_t, width, layers, heads, ff_mult, drop, out_dim,
                               t2v_add=t2v_add, dt_index=dt_index_t)     # txn 塔带 proj（进 head）
        self.cross = CrossAttn(width, heads, drop)
        self.head = nn.Sequential(
            nn.Linear(out_dim, head_dim), nn.GELU(),
            nn.Dropout(head_drop), nn.Linear(head_dim, out),
        )
        self.budget_bytes = int(budget_mb * 2 ** 20)

    def forward(self, x_c, nv, x_o, nko, x_t, nkt):
        budget = self.budget_bytes
        # 阶段 1：market 读出态（无 proj）当 q；order 块输出当 K/V
        hc, lastc = self.tower_c._chunked(self.tower_c.states, budget,
                                          x_c.transpose(1, 2), nv)
        q = self.tower_c.readout(hc, lastc)                        # (B, width)
        ho, _ = self.tower_o._chunked(self.tower_o.states, budget,
                                      x_o.transpose(1, 2), nko)
        keep_o = _keep_mask_last(ho.shape[1], nko - 1, ho.device)   # nko≥1（哨兵保证）
        z = self.cross(q, ho, keep_o)                              # (B, width)
        # 阶段 2：z 作为前缀 token 进 txn 塔（读出下标顺移 1，mask 由 states 处理）
        xt = x_t.transpose(1, 2)
        ht, lastt = self.tower_t._chunked(
            lambda x, n, zz: self.tower_t.states(x, n, prefix=zz), budget, xt, nkt, z)
        return self.head(self.tower_t.readout(ht, lastt))


def n_params(model):
    """参数量（打印用）。"""
    return sum(q.numel() for q in model.parameters())
