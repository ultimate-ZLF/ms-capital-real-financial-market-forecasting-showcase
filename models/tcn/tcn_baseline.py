"""tcn_baseline.py — **双塔TCN**（粗段 672 秒 + 细段 60 秒）：空洞因果卷积 + **末步读出**。

> 命名（2026-09-25）：本模型叫 **双塔TCN**——"双塔×"家族的第三个成员
> （第一个是 `models/cnn/twotower_baseline.py` 的**双塔CNN**，第二个是 `models/mlp/` 的**双塔MLP**）。
> **本目录自包含**：代码、测试与本模型的全部文档都在 `models/tcn/` 下（用户 2026-09-21 指定：
> 新模型不混进 `models/cnn/`，只关于本模型的文档也只放本目录）。共享文档只留一行导航指针。

**独立脚本**——不 import `twotower_baseline.py` / `mlp_baseline.py`：每个训练脚本是一支独立的
实验谱系，改一个不应影响另一个已记录的结果（记录纪律，见 CLAUDE.md #7）。只复用**共享基础设施**：
`seq_common`（loader / 折式 / 块洗牌 / 粗段加载）、`flow_feats`（细段现建到内存）、
`tabm_common`（BASE / cos_score）。

本模型要同时对上**三条已登记的事实**（`HYPOTHESES C-03` / `D-01`、`WALKTHROUGH11 §8.1`）：

1. **C-03**：双塔CNN 的粗段主干 RF 只有 **111 秒 / 672 秒窗口 = 17%**（A-03 实测）。
2. **WALKTHROUGH11 §8.1（2026-09-25 实测）**：stride-2 因果卷积把**输入末端 3 步完全丢掉**——
   细段 60 格是真实 60 秒、无 pad，双塔CNN **看不到最近 3 秒的成交流**。
3. **WALKTHROUGH11 §4–§6**：双塔MLP 的时间轴是**死轴**（均匀读出把位置轴平均掉）。

空洞因果卷积一次性对上三条：**不降采样 → 步长恒为 1**（末端不再丢失）、
**空洞指数增长 → RF 覆盖整个窗口**、**末步读出 → 池化 / mask / τ 一整类机制从架构上消失**。

设计（唯一变量是**主干**——输入、两塔读出维度、head、训练协议都与双塔CNN 逐字相同）：

```
塔 A（粗段） snap_cache train_snap_X.dat (N,224,15) f16 + n_valid
             → 干 Conv1d(15→64,k=3,stride=1) + BN + GELU + Drop
             → 6 个残差块，dilation = 1,2,4,8,16,32（k=3，块内 2 个空洞因果卷积）
                 每个块：x → [CausalConv(d)+BN+GELU+Drop] ×2 → + 残差 → GELU
                 （通道恒为 64 → 残差是恒等，不需要 1×1 投影）
             → **末步读出**：取每样本第 n_valid−1 个位置的输出     → (B,64)
             → Linear(64→256) + GELU + Drop                      → h_A (B,256)
塔 B（细段） flow_cache 现建到内存 (N,60,16) f16（**无 pad**）
             → 同构，4 个残差块（dilation 1,2,4,8），末步读出 = 位置 59 → h_B (B,256)
合           concat[h_A ; h_B] (512) → Linear(512→256) + GELU + Drop(head_drop) → Linear(256→1)
```

三处**刻意的**设计（都不是默认写法）：

1. **`CausalConv` 没有 `stride` 参数**，只有 stride=1 这一种。`left = (k−1)·dilation` 这个公式
   **只在 stride=1 时正确**——stride=2 时它会把输出对齐错、让输入末端 3 步完全消失
   （`WALKTHROUGH11 §8.1` 的实测缺陷）。**把 stride 从签名里删掉，这类错误在本模型不可能发生。**
2. **末步读出取 `n_valid−1`**：卷积因果 + pad 在时间轴末尾 ⇒ **位置 `n_valid−1` 的输出只依赖
   位置 ≤ `n_valid−1` 的真实数据，pad 永远到不了读出点**。这是"消掉 mask / τ"的数学保证。
   （与双塔CNN 的"pad 自由参与池化值 +0.024"（C-12）是两回事：那里是池化把 pad 拉进来，
   这里根本没有池化。CLAUDE.md #10 的三条耦合条件在本模型上不成立。）
3. **级数取"刚好覆盖本塔窗口"的最小值**：`RF = 1 + (stem_k−1) + 2(k−1)(2^L − 1)`。
   粗段 L=6 → RF **255 ≥ 224**；细段 L=4 → RF **63 ≥ 60**。既不留没被看过的历史，
   也不为用不到的 RF 付算量。`main()` 与 `test_tcn.py` 用**同一条公式**断言。

**从头训练**，不加载任何已有模型的权重（沿用 `twotower_baseline.py:37` 的用户原始要求）。
检查点是**本脚本自己逐折产出**的，只用于「同一次实验续跑」与「提交侧推理」，
**不跨谱系 / 模型 / 种子流通**（四条纪律见 CLAUDE.md #12）。

训练协议与双塔CNN / E1 各臂**逐字一致**（lr 1e-3 / wd 3e-4 / batch 1024 / drop 0.2 /
head_drop 0.3 / patience 16 / epochs 300 / seed 42 / MSE(target×1000) / val cos 早停 /
月交错 6 折）——折式没变，所以**逐折配对差仍然可比**。

⚠️ **判据（用户 2026-09-25 决定）**：本轮**不预注册阈值**，先看 6 折 CV 再定。
已记录的同形态参照（**不是本次判据**）：双塔CNN 的 k6 口径 6 折均值 **0.12280**（LB 0.117）。

⚠️ 已知风险见本目录 `README.md`「已知风险」节——尤其是**训练时 pad 仍会进 BatchNorm 的批统计量**
（读出通路不受影响，但这条与"完全不碰 pad"有差距）。

用法：
    python tcn_baseline.py --folds 1 --smoke-fold            # G1：验管线
    python tcn_baseline.py --folds 1 --epochs 2 --time-smoke # G2：测 s/epoch 与显存峰值
    python tcn_baseline.py --folds 6                         # 正式 6 折
    # 探针（审权重/验管线）——**必须另给 --oof-dir**，否则会被下面的防线拦住：
    python tcn_baseline.py --only-fold 0 --oof-dir /root/msdata/snap_cache/oof_probe_tcn

⚠️ **探针跑有一道覆盖防线**：`--only-fold` 或 `--smoke-fold` 时，若目标 `f{fi}.parquet`
已存在就**直接报错退出**、不静默覆盖（CLAUDE.md #7 的同类坑）。正式 6 折跑不拦。
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(MODELS), "seq"))   # 序列数据管道（顶层 seq/）
sys.path.insert(0, os.path.join(MODELS, "tabm"))    # tabm_common（BASE / cos_score）

import fine_derive  # noqa: E402  细段派生通道（顶层 seq/，被多模型族共用）
from seq_common import F as COARSE_C_BASE  # noqa: E402  基通道数：单一来源
from seq_common import block_perm, make_folds, make_loader  # noqa: E402
from tabm_common import BASE, cos_score  # noqa: E402


COARSE_T, COARSE_C = 224, COARSE_C_BASE
# ⚠️ `COARSE_C_BASE` **取自 `seq_common.F`（单一来源）**，不再手写数字——
#    原先两处独立声明，基通道一变就静默不一致（形状断言会炸，但炸得晚）。
#    派生通道**一律追加在它之后，绝不重排**（cnn_baseline.py 的
#    MASK_CHANNELS 是裸下标 10/12）。
FINE_T, FINE_C = 60, 16     # `FINE_C` = **基通道数**；派生通道一律**追加在其后，绝不重排**
K = 3                  # 残差块内空洞卷积的核长
STEM_K = 3             # 干那层的核长（也进 RF）
OUT_DIM = 256          # 每塔读出维（对齐双塔CNN 的 256 + 256 → head 512）
HEAD_DIM = 256
WIDTH_DEFAULT = 64
LEVELS_COARSE_DEFAULT = 6     # RF 255 ≥ 224
LEVELS_FINE_DEFAULT = 4       # RF 63 ≥ 60
EST_BEST_EPOCH = 10           # 只用于 G2 的时间投影；出处 = 双塔CNN 实测 best_epoch 中位数
SNAP_CACHE = os.path.join(BASE, "snap_cache")
TARGET_SCALE = 1000.0

# 已记录的同形态参照（**不是本次判据**，用户 2026-09-25 决定不预注册阈值）。
# 出处：MEMORY「新主干（E1）」六条臂表，月交错 6 折逐折均值、成员口径。
REF_TWOTOWER_CNN = 0.12280

DEFAULT_CKPT_DIR = os.path.join(SNAP_CACHE, "ckpt_tcn")
DEFAULT_OOF_DIR = "snap_cache/oof_parts_tcn"


def fine_spec(args):
    """`--derive` → 细段派生通道名列表（空 = 不派生）。"""
    return fine_derive.parse_spec(getattr(args, "derive", None))


def fine_c(args) -> int:
    """细段塔的输入通道数 = 基 16 + 派生条数。顺序即通道顺序。"""
    return FINE_C + len(fine_spec(args))


def receptive_field(levels, k=K, stem_k=STEM_K):
    """理论感受野：`1 + (stem_k−1) + 2(k−1)(2^L − 1)`。

    推导：输入端 RF=1；干那层加 `(stem_k−1)`；之后 L 个残差块，第 i 块的 2 个卷积
    各加 `(k−1)·2^i` → 合计 `2(k−1)·Σ2^i = 2(k−1)(2^L − 1)`。
    残差连接不改变 RF（它只把更小的 RF 并进来）。`test_tcn.py::test_receptive_field`
    用单位脉冲**实测**这个值，两边必须一致。
    """
    return 1 + (stem_k - 1) + 2 * (k - 1) * (2 ** levels - 1)


# ---------------------------------------------------------------- 检查点
# 与 mlp_baseline.py / twotower_baseline.py 的同名函数是**同一套**（照抄一份，不互相 import）：
# 只存 CPU state_dict + 配置指纹，读侧**必须**断言指纹。四条纪律见 CLAUDE.md #12。

def check_rf(levels_coarse, levels_fine):
    """级数下界检查：两塔的 RF 必须各自盖住本塔窗口 → `(rf_coarse, rf_fine)`。

    逻辑**只此一处**：`main()` 调它，`test_tcn.py::test_receptive_field` 也直接调它
    （避免"测了个复制品"）。级数少了不是"差一点"，是**有一段历史永远看不到**——
    C-03 要检验的正是"看到了全部历史会怎样"，所以这条必须硬。
    """
    out = []
    for tag, t_in, lv in (("coarse", COARSE_T, levels_coarse), ("fine", FINE_T, levels_fine)):
        rf = receptive_field(lv)
        assert rf >= t_in, (f"--levels-{tag}={lv} → RF {rf} < {tag} 窗口 {t_in}："
                            f"级数不够，模型无法看到全窗")
        out.append(rf)
    return tuple(out)




def arch_cfg(args):
    """参与指纹的配置。**换配置跑同一目录会报错而不是静默覆盖。**"""
    cfg = {
        # 架构：这些都改变"学出什么权重"。`k`/`stem_k`/`out_dim`/`head_dim` 是写死在
        # 本模块常量里的（不是 CLI 参数）——照实记下来，将来若被参数化指纹会自然跟着变。
        "width": args.width, "levels": [args.levels_coarse, args.levels_fine],
        "k": K, "stem_k": STEM_K, "out_dim": OUT_DIM, "head_dim": HEAD_DIM,
        "drop": args.drop, "head_drop": args.head_drop,
        # 输入形状（改数据布局/通道数必须另起谱系）
        "coarse": [COARSE_T, COARSE_C], "fine": [FINE_T, fine_c(args)],
        # 协议里**会改变学出权重**的项
        "seed": args.seed, "lr": args.lr, "wd": args.wd, "batch": args.batch,
    }
    # ⚠️ `--loss cos`（HYPOTHESES C-17 的损失臂）改变训练协议 → **必须进指纹**，
    #    否则换了损失会**静默复用**另一套协议产出的检查点。
    #    但 `--loss mse`（默认）就是原协议，**不进指纹**——这是 CLAUDE.md #12 的反向纪律
    #    （别为"等价于不变"的开关加字段）：无条件加字段会让已训好的 `ckpt_tcn/*.pt`
    #    （v22 LB 0.122 的成员）指纹不符而被**误拒**。
    #    ⚠️ 这段必须在 `return` **之前**——第一版写在了 return 之后，成了死代码，
    #       是 `test_arch_cfg_discipline` 抓出来的。
    # ⚠️ 判据是 **`!= "mse"`**，不是 `!= "cos"`——这是**历史编码规则**，别"顺着默认值改"。
    #    存盘的指纹一律按"mse = 无字段"编码（默认值曾是 mse）。改成 `!= "cos"` 会让
    #    **cos 的默认跑算出与 v22 的 MSE 检查点相同的 hash** → 静默加载另一套协议的权重
    #    （实测踩到：cos 默认给 15cf0b8e13614e4b，正是 v22 存盘的那一串）。
    #    保持 `!= "mse"` 则两条归档都能载：cos → 有字段（= v24）、mse → 无字段（= v22）。
    if getattr(args, "loss", "cos") != "mse":
        cfg["loss"] = args.loss
    # 细段派生通道：**条件性**进指纹——空 spec（默认）时完全不加字段，否则已归档的
    # `ckpt_tcn/*.pt`（v22/v24/v25）会被**误拒**（CLAUDE.md #12 的反向纪律）。
    # ⚠️ 光靠 `fine:[60,K]` **不够**：两个不同的 spec 可以给出相同的 K
    #    （`[a,b]` vs `[c,d]` 都是 K=2）→ 同 hash → **静默复用另一套通道的检查点**。
    #    所以按**顺序**记名字（顺序即通道顺序，排序会丢掉信息）。
    deriv = fine_spec(args)
    if deriv:
        cfg["derive_fine"] = deriv
    return cfg
    # ⚠️ 刻意**不**写 "readout" 字段：本轮没有读出开关，末步读出**就是**这套架构。
    #    加一个永不变化的字段正是 CLAUDE.md #12 反向纪律要防的（"别为等价于不变的开关加指纹字段"）。
    #    `mlp_baseline.arch_cfg` 里保留的 `pad_mode:"none"` 是**为兼容已存在的旧检查点**而留的
    #    冻结常量，本目录没有历史检查点，不需要这种兼容件。将来真加读出臂时**那时**再进指纹。
    # ⚠️ 同理**刻意不进指纹**（不改"学出什么权重"）：epochs / patience（只改跑多久）、
    #    workers / eval_chunk / oof_dir / ckpt_dir（工程件）。eval_chunk 只带来 1e-7 级
    #    float32 分块差异，与 mlp 侧同口径处理。


def arch_hash(args):
    blob = json.dumps(arch_cfg(args), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def ckpt_path(ckpt_dir, fold, kind="best", smoke=False):
    """kind ∈ {best, last}；冒烟跑写 `f{fi}_smoke.*`，**绝不碰正式名**
    （否则 1 epoch 的冒烟会被正式跑当成"已完成"而跳过）。"""
    tag = f"f{fold}_smoke" if smoke else f"f{fold}"
    return os.path.join(ckpt_dir, f"{tag}.{kind}.pt")


def save_ckpt(path, model, cfg_hash, meta):
    """原子写：先写 `path + ".tmp"` 再 os.replace。

    ⚠️ `torch.save` **不追加扩展名**（与 `np.save` 相反，见 CLAUDE.md #1）——
    所以 `path + ".tmp"` 是安全的。**不要"顺手改成 `.tmp.pt`"**，那会再造一个同类坑。
    payload 只含 CPU tensor 与基本类型 → 读侧可用 `weights_only=True`。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "arch_hash": cfg_hash, "meta": meta}
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_ckpt(path, args):
    """读检查点。**指纹不符/文件缺失一律报错，绝不静默继续。**"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺检查点 {path}——先跑 tcn_baseline.py 对应折")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    got, want = payload.get("arch_hash"), arch_hash(args)
    if got != want:
        raise ValueError(f"检查点配置不符：{path}\n  文件={got}  当前={want}\n"
                         f"  当前配置 {json.dumps(arch_cfg(args), sort_keys=True)}")
    return payload


def load_model_from_ckpt(path, args, device):
    """读检查点 → 建模型 → 装载权重（不回传 optimizer：提交侧只推理）。"""
    payload = load_ckpt(path, args)
    model = build_model(args, device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("meta", {})


def resume_ckpt(ckpt_dir, fold, smoke=False):
    """断点续跑的键：优先 `best`，其次 `last`；都没有 → None。"""
    for kind in ("best", "last"):
        p = ckpt_path(ckpt_dir, fold, kind, smoke)
        if os.path.exists(p):
            return p
    return None


def oof_from_ckpt(path, args, Xc, Xf, nv, va_rows, device):
    """用检查点复算该折 OOF（`.pt` 在而 parquet 不在时用，**免重训**）→ (pred, meta)。

    与训练里选 best 用的是**同一个 `eval_forward`、同一个 `--eval-chunk`**
    → 同 chunk 重放**逐位相同**（`test_tcn.py::test_chunked_eval` 守着）。
    """
    model, meta = load_model_from_ckpt(path, args, device)
    return eval_forward(model, Xc, Xf, nv, va_rows, args.eval_chunk, device), meta


def oof_summary(oof_dir):
    """汇总**盘上**逐折 OOF 的 cos → (文件名列表, cos 数组)。

    ⚠️ 为什么不用内存里的 `results`：断点续跑**跳过的折不会进 results**，
    用它做汇总会**漏折**（mlp 侧实测踩过：6 折被报成 5 折）。汇总一律以**盘上产物**为准。
    """
    lab = pl.read_parquet(os.path.join(BASE, "train/label.parquet")).select(["sample_id", "target"])
    files = sorted(f for f in os.listdir(oof_dir)
                   if f.startswith("f") and f.endswith(".parquet"))
    vals = []
    for f in files:
        p = pl.read_parquet(os.path.join(oof_dir, f)).join(lab, on="sample_id", how="inner")
        pr, tg = p["pred"].to_numpy(), p["target"].to_numpy()
        vals.append(float((pr * tg).sum() / np.sqrt((pr ** 2).sum() * (tg ** 2).sum())))
    return files, np.array(vals)


# ---------------------------------------------------------------- 数据

def load_train(args):
    """返回 (粗段 memmap, n_valid, 细段内存数组, y, month, months, sids)。

    - 粗段走 `seq_common.load_train_snap()`（自带与 label 的行序断言）
    - 细段**现建到内存**（`.dat` 已按 2026-09-16 的决定删除）→ ⚠️ 需要约 2.4GB，
      所以本脚本**只能在有卡模式跑**（无卡模式 memory.max=2GiB，会被静默 OOM 杀掉）
    - `--derive` 非空时，派生通道**追加在 16 条基通道之后**（`Xf` 变 `(N,60,16+K)`）
    - 两份输入必须**行序一致**（都按 sample_id 升序），这里显式断言
    """
    from seq_common import load_train_snap

    Xc, nv, y, marr, months, sids = load_train_snap()
    assert Xc.shape[1:] == (COARSE_T, COARSE_C), f"粗段形状 {Xc.shape}"

    import flow_feats
    Xf = flow_feats.build_split("train", None, False, to_memory=True)   # ⚠️ 4 参（snap_feats 是 5 参）
    assert Xf.shape[1:] == (FINE_T, FINE_C), f"细段基通道形状 {Xf.shape}"
    deriv = fine_spec(args)
    if deriv:
        D = fine_derive.build_full("train", deriv)
        assert D.shape[0] == Xf.shape[0], \
            f"派生行数 {D.shape[0]} != 细段基通道行数 {Xf.shape[0]}（行序映射错了）"
        # ⚠️ 峰值内存 ≈ 2.4 + 1.5 + 3.9 ≈ 7.8GB（三段同时在世）。够用，但别在这里再加东西。
        Xf = np.concatenate([Xf, D], axis=2)
        del D
    assert Xf.shape[1:] == (FINE_T, fine_c(args)), \
        f"细段形状 {Xf.shape} != ({FINE_T}, {fine_c(args)})"
    assert Xf.shape[0] == Xc.shape[0], \
        f"两份输入行数不一致：粗段 {Xc.shape[0]} vs 细段 {Xf.shape[0]}"
    assert (nv >= 1).all() and (nv <= COARSE_T).all(), \
        f"n_valid 越界：[{nv.min()}, {nv.max()}] 必须落在 1..{COARSE_T}（末步读出要取 n_valid−1）"
    return Xc, nv, Xf, y, marr, months, sids


def build_one(Xc, Xf, row):
    """单行构建器 → (粗段 (C,224) f32, 细段 (C,60) f32)。**无 mask**。"""
    a = np.ascontiguousarray(np.asarray(Xc[row], dtype=np.float32).T)   # (15,224)
    b = np.ascontiguousarray(np.asarray(Xf[row], dtype=np.float32).T)   # (16,60)
    return a, b


def batch_two(Xc, Xf, rows, device):
    """整批版（验证集用）→ (粗段 (B,15,224), 细段 (B,16,60))。

    必须与 `build_one` 数值一致——不一致等于训练/验证喂了不同分布的数据，**不会报错**。
    `test_tcn.py::test_builders_agree` 守着。
    """
    a = torch.from_numpy(np.asarray(Xc[rows], dtype=np.float32)).permute(0, 2, 1).to(device)
    b = torch.from_numpy(np.asarray(Xf[rows], dtype=np.float32)).permute(0, 2, 1).to(device)
    return a, b


class TwoTowerTCNDataset(torch.utils.data.Dataset):
    """按行取样 → (x_coarse, x_fine, n_valid, y)。写成类是为了能被 DataLoader 的 worker pickle。

    ⚠️ 与双塔CNN 的 Dataset 只差**多返回一个 `n_valid`**——末步读出要用它定位 `n_valid−1`。
    双塔CNN 的读出是池化、不需要位置，所以那边没有这一项。
    """

    def __init__(self, rows, y, Xc, Xf, nv, scale=TARGET_SCALE):
        self.rows, self.y, self.Xc, self.Xf, self.nv, self.scale = rows, y, Xc, Xf, nv, scale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        a, b = build_one(self.Xc, self.Xf, r)
        return (torch.from_numpy(a), torch.from_numpy(b),
                torch.tensor(int(self.nv[r]), dtype=torch.long),
                torch.tensor(self.y[r] * self.scale, dtype=torch.float32))


# ---------------------------------------------------------------- 模型

class CausalConv(torch.nn.Module):
    """因果 Conv1d：左 pad `(k−1)·dilation`，**只有 stride=1 这一种**。

    ⚠️ **没有 `stride` 参数是刻意的**，不是漏写。`left = (k−1)·dilation` 只在 stride=1 时
    能让输出 j 恰好依赖输入 ≤ j；stride=2 时它会把输出对齐错，**让输入末端 3 步完全消失**
    （`WALKTHROUGH11 §8.1` 在双塔CNN 上实测到的缺陷：细段 60 格是真实 60 秒、无 pad，
    模型看不到最近 3 秒，而细段读出的最高权重恰好给了覆盖那段的块）。
    **把 stride 从签名里删掉，这类错误在本模型不可能发生** —— `test_tcn.py::test_last_step_visible`
    守着这条（双塔CNN 会在那条测试上失败）。
    """

    def __init__(self, cin, cout, k, dilation=1):
        super().__init__()
        self.left = (k - 1) * dilation
        self.conv = torch.nn.Conv1d(cin, cout, k, dilation=dilation)   # stride 默认 1

    def forward(self, x):
        return self.conv(F.pad(x, (self.left, 0)))


class TCNBlock(torch.nn.Module):
    """标准 TCN 残差块（Bai et al. 2018 的形态）：两层空洞因果卷积 + 残差。

    激活/归一化与双塔CNN 的 `Tower` 保持一致（BN + GELU + Dropout）——
    换掉它们就变成两个变量了，本模型只换主干。通道数块内不变 → 残差是**恒等**，
    不需要 1×1 投影（双塔CNN 那三层通道在变，所以那边也没有残差）。
    """

    def __init__(self, ch, k, dilation, drop):
        super().__init__()
        self.conv1 = CausalConv(ch, ch, k, dilation)
        self.bn1 = torch.nn.BatchNorm1d(ch)
        self.conv2 = CausalConv(ch, ch, k, dilation)
        self.bn2 = torch.nn.BatchNorm1d(ch)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        h = self.drop(F.gelu(self.bn1(self.conv1(x))))
        h = self.drop(self.bn2(self.conv2(h)))
        return F.gelu(x + h)          # 残差连接：RF 由空洞分支决定，不受它影响


class TCNTower(torch.nn.Module):
    """单塔：干 → L 个空洞残差块 → **末步读出** → 投影到 `out_dim`。

    末步读出（`forward` 的 `last`）：
      - `last=None` → 取位置 `T−1`（细段用：60 格是真实 60 秒、**无 pad**）
      - `last=nv−1` → 取每样本各自的最后一个有效位置（粗段用：pad 在时间轴末尾）
    ⚠️ 这两个都靠**因果卷积**才有意义：位置 `i` 的输出只依赖输入 ≤ `i`，
    所以取 `n_valid−1` 时末尾的 pad 完全进不来（`test_tcn.py::test_pad_invisible` 钉住）。
    """

    def __init__(self, c_in, t_in, width, levels, k=K, drop=0.2, out_dim=OUT_DIM):
        super().__init__()
        self.t_in, self.width, self.levels = t_in, width, levels
        self.rf = receptive_field(levels, k)
        assert self.rf >= t_in, \
            f"感受野 {self.rf} < 窗口 {t_in}：级数 {levels} 不够（RF 必须覆盖整个窗口才有意义）"
        self.stem = torch.nn.Sequential(
            CausalConv(c_in, width, STEM_K), torch.nn.BatchNorm1d(width),
            torch.nn.GELU(), torch.nn.Dropout(drop),
        )
        self.blocks = torch.nn.Sequential(
            *[TCNBlock(width, k, 2 ** i, drop) for i in range(levels)])
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(width, out_dim), torch.nn.GELU(), torch.nn.Dropout(drop))

    def forward(self, x, last=None):
        h = self.blocks(self.stem(x))                     # (B,width,T)
        if last is None:
            h = h[:, :, -1]
        else:
            # 逐样本 gather：idx (B,1,1) → 展开到 (B,width,1)
            idx = last.view(-1, 1, 1).expand(-1, h.shape[1], 1)
            h = h.gather(2, idx).squeeze(2)
        return self.proj(h)


class TwoTowerTCN(torch.nn.Module):
    """粗段塔 + 细段塔，各自**末步读出**后在 head 处拼接（head 与双塔CNN 逐字相同）。

    两塔唯一的不对称是**级数**（6 vs 4，各自取覆盖本塔窗口的最小值）——
    与双塔CNN 的 `dil=[4,1]` 在架构里的位置相当。
    """

    def __init__(self, width=WIDTH_DEFAULT, levels_coarse=LEVELS_COARSE_DEFAULT,
                 levels_fine=LEVELS_FINE_DEFAULT, drop=0.2, head_drop=0.3,
                 out=1, head_dim=HEAD_DIM, out_dim=OUT_DIM,
                 in_coarse=COARSE_C, in_fine=FINE_C):
        super().__init__()
        self.tower_a = TCNTower(in_coarse, COARSE_T, width, levels_coarse, K, drop, out_dim)
        self.tower_b = TCNTower(in_fine, FINE_T, width, levels_fine, K, drop, out_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * out_dim, head_dim), torch.nn.GELU(),
            torch.nn.Dropout(head_drop), torch.nn.Linear(head_dim, out),
        )

    def forward(self, x_coarse, nv, x_fine):
        """`nv` = 粗段每样本的有效步数 (B,) int64；末步读出取 `nv−1`。细段无 pad，恒定取 `T−1`。"""
        h = torch.cat([self.tower_a(x_coarse, nv - 1), self.tower_b(x_fine)], dim=1)   # (B,512)
        return self.head(h)


def build_model(args, device):
    """按 args 建模型并搬到 device。训练/推理/OOF 复算**共用这一处**（口径只有一份）。"""
    return TwoTowerTCN(width=args.width, levels_coarse=args.levels_coarse,
                       levels_fine=args.levels_fine, drop=args.drop,
                       head_drop=args.head_drop, in_fine=fine_c(args)).to(device)


# ---------------------------------------------------------------- 评估

def eval_forward(model, Xc, Xf, nv, rows, chunk, device):
    """**分块**前向 → (n,) f32 预测。

    分块是**显存护栏**（不是为了快）。双塔CNN 把整折 21.2 万行一次性搬上卡能过，因为它的
    主干降采样了时间轴（最宽的中间张量 256ch @ T=54 ≈ 11.7GB）；**TCN 全程 T=224**，
    `212139 × 64 × 224 × 4B ≈ 12.2 GB / 每个中间张量`，链上同时活着 2–3 个 → 顶满 40GB。

    数值口径（**别照抄"逐位相同"**）：分块改变 float32 GEMM 的切分 → 与一次性前向有
    1e-7 量级的差异；但**同一 chunk 重放逐位相同**——检查点复算 OOF 依赖的正是这一条
    （`test_tcn.py::test_chunked_eval` 两条都钉）。
    """
    model.eval()
    out = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, len(rows), chunk):
            sl = rows[lo:lo + chunk]
            xa, xb = batch_two(Xc, Xf, sl, device)
            nvb = torch.from_numpy(nv[sl].astype(np.int64)).to(device)
            out[lo:lo + chunk] = model(xa, nvb, xb).squeeze(-1).cpu().numpy()
    return out


# ---------------------------------------------------------------- 损失（训练协议件）

def batch_loss(pred, yb, loss_kind="mse"):
    """训练损失。`pred` / `yb` 都已在该折的 `target×TARGET_SCALE` 单位上。

    - **`mse`（默认）**：走 `F.mse_loss`，与原协议**逐位一致** → 历史检查点与结果口径不变。
    - **`cos`**：`1 − cos(p_batch, y_batch)`，**不中心化**（HYPOTHESES C-17 的口径）。
      为什么它与 MSE 的差别**只可能**落在优化动力学上：全局 cos 的梯度
      `∂cos/∂pᵢ ∝ yᵢ − c·pᵢ`（`c = Σpy/Σp²`）**正是** MSE 的负梯度，逐 batch 等价于
      "目标缩放 c 倍"——而 c 是**每批自适应**的尺度（Adam 只做逐参数自适应，做不了这个）。
      （WALKTHROUGH10 §1 有完整推导。）
      **不中心化**的理由：指标是未中心化的 cos；公开方案在 batch 内中心化（那优化的是 Pearson
      相关，与本指标不是同一个量）。附带好处——未中心化的 cos 会**自己**把预测均值压向 0。
      ⚠️ 用户 2026-09-23 明确**不加分母下限 ε**，所以这里没有保护。风险明确：cos 对整体尺度
      不变，`wd` 会把 `‖p‖` 往下推，而梯度 ∝ `1/‖p‖` → **输出尺度塌缩**是这条臂的预期失败模式；
      故 `train_fold` 每轮记录 `|p|` 均值。分母**恰好为 0** 时直接报错（真异常，不是数值噪声）。

    逻辑**只此一处**——`train_fold` 调它，`test_tcn.py` 也直接测它（避免"测了个复制品"）。
    """
    if loss_kind == "cos":
        den = pred.norm() * yb.norm()
        if not torch.isfinite(den) or float(den) == 0.0:
            raise RuntimeError(
                f"[loss=cos] 分母退化：‖p‖={float(pred.norm()):.3g} ‖y‖={float(yb.norm()):.3g}")
        return 1.0 - (pred * yb).sum() / den
    return F.mse_loss(pred, yb)


# ---------------------------------------------------------------- 训练

def train_fold(Xc, Xf, nv, y, sids, tr_rows, va_rows, args, device, ckpt_dir, cfg_hash,
               fold, smoke):
    torch.manual_seed(args.seed)
    model = build_model(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = np.random.default_rng(args.seed)      # 与 torch.manual_seed 是两条独立的随机流
    keep_last = args.ckpt_keep in ("last", "both")
    keep_best = args.ckpt_keep in ("best", "both")

    n_tr = len(tr_rows)
    # DataLoader 多进程预取：**只建一次、每轮换排列**（CLAUDE.md #11：每 epoch 新建会泄漏 fd，
    # 跑到第 5 折会以 "Too many open files" **挂住**——主进程 121% CPU 而 GPU 0%）。
    ds = TwoTowerTCNDataset(tr_rows, y, Xc, Xf, nv)
    loader, sampler = make_loader(ds, n_tr, args.batch, args.workers)

    best_cos, best_epoch, best_va_pred, stall = -1.0, -1, None, 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        sampler.set_perm(block_perm(n_tr, block=4096, gen=gen))
        loss_sum, n_batch = 0.0, 0
        pnorm_sum = 0.0     # `|p|` 均值：cos 臂的**尺度塌缩**监控（只有 loss=cos 时有意义）
        for xa, xb, nvb, yb in loader:
            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            nvb = nvb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xa, nvb, xb).squeeze(-1)
            loss = batch_loss(pred, yb, args.loss)
            if args.loss == "cos":
                pnorm_sum += float(pred.abs().mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            n_batch += 1
        loss_mean = loss_sum / max(n_batch, 1)
        pnorm = pnorm_sum / max(n_batch, 1)      # 每批 |p| 的均值再整轮平均
        pnorm_s = f" |p|={pnorm:.3f}" if args.loss == "cos" else ""

        # 验证走**分块**前向（显存护栏）。与 oof_from_ckpt 共用同一函数/同一 chunk
        # → 检查点复算 OOF 与这里选 best 的 pred 逐位一致。
        pred_va = eval_forward(model, Xc, Xf, nv, va_rows, args.eval_chunk, device)
        cos = cos_score(pred_va, (y[va_rows] * TARGET_SCALE).astype(np.float32))
        if cos > best_cos:
            best_cos, best_epoch, best_va_pred, stall = cos, ep, pred_va, 0
            if keep_best:
                save_ckpt(ckpt_path(ckpt_dir, fold, "best", smoke), model, cfg_hash,
                          {"fold": fold, "kind": "best", "epoch": ep, "cos": float(cos),
                           "smoke": bool(smoke)})
        else:
            stall += 1
            if stall >= args.patience:
                print(f"    ep={ep:3d} loss={loss_mean:.4f} val_cos={cos:.5f}"
                      f"{pnorm_s} t={time.time()-t0:5.1f}s  早停", flush=True)
                break
        if ep % 5 == 0:
            print(f"    ep={ep:3d} loss={loss_mean:.4f} val_cos={cos:.5f}"
                  f"{pnorm_s} t={time.time()-t0:5.1f}s", flush=True)

    if keep_last:
        save_ckpt(ckpt_path(ckpt_dir, fold, "last", smoke), model, cfg_hash,
                  {"fold": fold, "kind": "last", "epoch": ep, "cos": float(cos),
                   "smoke": bool(smoke)})

    return {"cos": best_cos, "epoch": best_epoch, "va_pred": best_va_pred,
            "va_sids": sids[va_rows], "fold_secs": time.time() - t0}


def main():
    p = argparse.ArgumentParser(description="双塔TCN（空洞因果卷积 + 末步读出）联合训练")
    p.add_argument("--folds", type=int, default=6, help="月交错折数（1..6）")
    p.add_argument("--start-fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=10,
                   help="早停耐心（判据是 val cos 不再改善）。⚠️ **本模型与双塔CNN 的唯一协议差异**："
                        "双塔CNN 用的是 16。用户 2026-09-25 指定降到 10"
                        "（6 折从 ~2.5h 降到 ~1.9h）。它**不进配置指纹**（与 epochs 同类，"
                        "不改'学出什么权重'，只改跑多久）——但要知道：耐心太小会**在迟到的高峰之前收手**，"
                        "所以这个值不要在与历史数字做严格对照时随手改")
    p.add_argument("--batch", type=int, default=1024,
                   help="与 E1 各臂 / 双塔CNN / 双塔MLP 一致，**不要改**（见 CLAUDE.md 工程教训 #3）")
    p.add_argument("--eval-chunk", type=int, default=8192,
                   help="验证/推理前向的分块行数（**显存护栏**，不改口径）。整折一次性前向约 12GB/张 → 顶满卡")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--head-drop", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss", choices=["mse", "cos"], default="cos",
                   help="训练损失。`cos` = `1−cos(p_batch, y_batch)`（**不中心化**，与 C-17 同口径）。"
                        "**默认 cos**（用户 2026-09-26 指令：之后所有模型都用 cos；2026-09-27 落实为默认值）。"
                        "⚠️ 只在 ≠cos 时才进配置指纹 → **换损失必须另给 --ckpt-dir / --oof-dir**，"
                        "否则会与另一臂的产物互相拒载/覆盖。"
                        "⚠️ 本模型无梯度裁剪，所以这是**单变量**臂（MLP 那次是 --loss cos --clip 0 两变量）")
    # —— 主干（空洞 TCN；默认 width 64 / 粗段 6 级 / 细段 4 级 = 各自刚好覆盖本塔窗口）——
    p.add_argument("--width", type=int, default=WIDTH_DEFAULT,
                   help="残差块内的通道数（两塔同宽）。算式账见 README「算量账」节")
    p.add_argument("--levels-coarse", type=int, default=LEVELS_COARSE_DEFAULT,
                   help=f"粗段空洞级数，dilation = 1,2,4,...。默认 {LEVELS_COARSE_DEFAULT} → "
                        f"RF {receptive_field(LEVELS_COARSE_DEFAULT)} ≥ {COARSE_T}；少了会被断言拦住")
    p.add_argument("--levels-fine", type=int, default=LEVELS_FINE_DEFAULT,
                   help=f"细段空洞级数。默认 {LEVELS_FINE_DEFAULT} → "
                        f"RF {receptive_field(LEVELS_FINE_DEFAULT)} ≥ {FINE_T}")
    # —— 检查点 ——
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR,
                   help="逐折检查点目录（best/last，纪律见 CLAUDE.md #12）；空串 = 不存")
    p.add_argument("--ckpt-keep", choices=["best", "last", "both"], default="both",
                   help="best＝val cos 最好那轮；last＝最后一轮")
    p.add_argument("--oof-dir", default=DEFAULT_OOF_DIR)
    # —— 细段派生通道（`seq/fine_derive.py`）——
    p.add_argument("--derive", default="none",
                   help="细段派生通道，逗号分隔（候选池见 `seq/fine_derive.py` 的 `POOL`）；"
                        "`none` = 不加。派生通道**追加在 16 条基通道之后，绝不重排**。"
                        "⚠️ 非空时会改变细段塔的输入通道数**并进入配置指纹** → "
                        "**必须另给 --ckpt-dir / --oof-dir**，否则会与不派生的产物互相拒载/覆盖")
    p.add_argument("--only-fold", type=int, default=-1,
                   help="只跑这一折（-1 = 按 --start-fold/--folds 全跑）；审权重/验管线用")
    p.add_argument("--smoke-fold", action="store_true",
                   help="冒烟：每折只跑 1 epoch、不汇总、写 _smoke 名")
    p.add_argument("--time-smoke", action="store_true",
                   help="G2：跑 2 epoch 并报告 s/epoch 与显存峰值（判断 6 折总时长的硬闸门）")
    args = p.parse_args()

    # 级数下界：RF 必须盖住本塔窗口（级数少了不是"差一点"，是有一段历史永远看不到）。
    # ⚠️ **放在任何副作用之前**——配置错就该在读数据/建目录之前报错。放在后面时，
    #    BASE 不可用的机器上会先抛 PermissionError（实测踩到），把真错误盖掉。
    rf_a, rf_b = check_rf(args.levels_coarse, args.levels_fine)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.ckpt_dir = args.ckpt_dir if os.path.isabs(args.ckpt_dir) \
        else (os.path.join(BASE, args.ckpt_dir) if args.ckpt_dir else "")
    oof_dir = args.oof_dir if os.path.isabs(args.oof_dir) else os.path.join(BASE, args.oof_dir)
    os.makedirs(oof_dir, exist_ok=True)
    cfg_hash = arch_hash(args) if args.ckpt_dir else None

    print(f"device={device} batch={args.batch} lr={args.lr} wd={args.wd} "
          f"drop={args.drop}/{args.head_drop} seed={args.seed} workers={args.workers} "
          f"loss={args.loss} patience={args.patience}")
    print(f"双塔TCN：粗段 ({COARSE_C},{COARSE_T}) → {args.levels_coarse} 级空洞残差块"
          f"（RF {rf_a} ≥ {COARSE_T} = 全窗）；"
          f"细段 ({fine_c(args)},{FINE_T}) → {args.levels_fine} 级（RF {rf_b} ≥ {FINE_T}）")
    if fine_spec(args):
        print(f"细段派生通道 {len(fine_spec(args))} 条（追加在 {FINE_C} 条基通道之后）："
              f"{','.join(fine_spec(args))}")
    print(f"读出：**末步**（粗段取每样本 n_valid−1；细段取 59）→ 各 {OUT_DIM} 维拼接 {2*OUT_DIM} → head")
    print(f"**从头训练**、不加载已有权重；无池化 / 无 mask / 无 τ（因果卷积使 pad 进不了读出点）")
    n_par = sum(q.numel() for q in build_model(args, "cpu").parameters())
    print(f"参数量 {n_par:,}（双塔CNN 423k 的 {n_par/423000:.2f}×）；"
          f"eval_chunk={args.eval_chunk}；cfg 指纹={cfg_hash or '-'}")
    print(f"OOF → {oof_dir}；检查点 → {args.ckpt_dir or '（不存）'}"
          + (f"   **只跑 fold {args.only_fold}**" if args.only_fold >= 0 else ""))
    # ⚠️ 这**不是**判据（用户 2026-09-25 决定本轮不预注册阈值），只是把已有的同形态数字摆在旁边
    print(f"同形态参照（已记录，非本次判据）：双塔CNN k6 六折均值 = {REF_TWOTOWER_CNN:.5f}")

    t0 = time.time()
    Xc, nv, Xf, y, marr, months, sids = load_train(args)
    print(f"loaded: 粗段 {Xc.shape} {'f16' if isinstance(Xc, np.memmap) else 'mem'} + "
          f"细段 {Xf.shape}（n_valid 中位 {int(np.median(nv))}；{time.time()-t0:.1f}s）", flush=True)

    folds = make_folds(months, args.folds)
    results = []
    for fi, va_months in enumerate(folds):
        if fi < args.start_fold:
            continue
        if args.only_fold >= 0 and fi != args.only_fold:
            continue
        part_path = os.path.join(oof_dir, f"f{fi}.parquet")
        # **探针跑**（`--only-fold` 审权重、`--smoke-fold` 验管线）时默认 oof_dir 里躺着**正式归档**
        # 的逐折 OOF → 拒绝静默覆盖（CLAUDE.md #7）。正式跑不拦（那是刻意的重跑）。
        if (args.only_fold >= 0 or args.smoke_fold) and os.path.exists(part_path):
            raise SystemExit(f"[探针跑] 目标 OOF 已存在，拒绝覆盖：{part_path}\n"
                             f"  请另给 --oof-dir，例如 snap_cache/oof_parts_tcn_probe")

        smoke_naming = args.smoke_fold or args.time_smoke
        resume_path = resume_ckpt(args.ckpt_dir, fi, smoke_naming) if args.ckpt_dir else None
        va_mask = np.isin(marr, va_months)
        tr_rows = np.where(~va_mask)[0]
        va_rows = np.where(va_mask)[0]

        # 断点续跑：检查点存在且指纹匹配 → 免重训；parquet 缺了就用检查点补出来
        if resume_path is not None and not smoke_naming:
            try:
                pred, meta = oof_from_ckpt(resume_path, args, Xc, Xf, nv, va_rows, device)
            except ValueError as e:
                raise SystemExit(f"[fold {fi}] 检查点存在但配置不符，拒绝覆盖：\n{e}")
            if not os.path.exists(part_path):
                pl.DataFrame({"sample_id": sids[va_rows], "month": marr[va_rows],
                              "pred": pred.astype(np.float64)}).write_parquet(part_path)
                print(f"[fold {fi+1}/{len(folds)}] 复用检查点补出 OOF（{part_path}）", flush=True)
            else:
                print(f"[fold {fi+1}/{len(folds)}] 检查点与 OOF 都在，跳过"
                      f"（cos={meta.get('cos', float('nan')):.5f}）", flush=True)
            continue

        print(f"[fold {fi + 1}/{len(folds)}] 验证月={va_months} "
              f"训练={len(tr_rows)} 验证={len(va_rows)}", flush=True)
        epochs = 1 if args.smoke_fold else args.epochs
        if args.time_smoke:
            epochs = 2
        t1 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        r = train_fold(Xc, Xf, nv, y, sids, tr_rows, va_rows,
                       argparse.Namespace(**{**vars(args), "epochs": epochs}), device,
                       args.ckpt_dir, cfg_hash, fi, smoke_naming)
        r["fold_secs"] = time.time() - t1
        results.append(r)

        if args.time_smoke:
            per_ep = r["fold_secs"] / epochs
            peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
            # 每折 epoch 数 ≈ 峰值轮 + patience。峰值轮取 **10**：那是双塔CNN 的实测
            # `best_epoch` 中位数（本项目唯一一个能用的参照，TCN 自己的还没跑过）。
            est_ep = EST_BEST_EPOCH + args.patience
            print(f"\n### G2 实测 ###\n  {per_ep:.1f} s/epoch（{epochs} epoch 共 {r['fold_secs']:.0f}s）"
                  f"\n  显存峰值 {peak:.1f} GiB\n"
                  f"  投影 6 折 ≈ {per_ep * est_ep * 6 / 3600:.2f} h"
                  f"（按每折 {est_ep} epoch = best ~{EST_BEST_EPOCH} + patience {args.patience} 估；"
                  f"⚠️ 峰值轮是借来的参照，TCN 的还没测）", flush=True)
            print("\n（G2 模式：不写正式 OOF/检查点，只报时间与显存）")
            return

        if args.smoke_fold:
            print(f"  → cos={r['cos']:.5f} best_epoch={r['epoch']} 耗时 {r['fold_secs']:.0f}s"
                  f"（冒烟：写 _smoke 名，不落 OOF）", flush=True)
            continue

        pl.DataFrame({"sample_id": r["va_sids"], "month": marr[va_rows],
                      "pred": r["va_pred"].astype(np.float64)}).write_parquet(part_path)
        print(f"  → cos={r['cos']:.5f} best_epoch={r['epoch']} 耗时 {r['fold_secs']:.0f}s"
              f"（已落盘 {part_path}）", flush=True)

    if args.smoke_fold or len(results) <= 0:
        print("\n（冒烟模式：不汇总）")
        return

    # 逐折 cos：**以盘上 OOF 为准**（续跑跳过的折不进 results，用它会漏折）
    files, cos_arr = oof_summary(oof_dir)
    ep_arr = np.array([r["epoch"] for r in results])
    n_done = len(cos_arr)
    print(f"\n### 双塔TCN 逐折 cos（{n_done} 折，月交错，width={args.width}，"
          f"levels={args.levels_coarse}/{args.levels_fine}）###")
    for fn, c in zip(files, cos_arr):
        print(f"  {fn}: cos={c:.5f}")
    print(f"mean={cos_arr.mean():.5f} std={cos_arr.std():.5f} "
          f"min={cos_arr.min():.5f} neg={int((cos_arr < 0).sum())}")
    if len(results) < n_done:
        print(f"（本次训练 {len(results)} 折；其余 {n_done - len(results)} 折取自盘上的检查点/OOF）")
    if len(ep_arr):
        print(f"best_epoch: median={int(np.median(ep_arr))} 本次各折={ep_arr.tolist()}")
    print(f"\n同形态参照（已记录，非本次判据）：双塔CNN k6 = {REF_TWOTOWER_CNN:.5f}")
    print("⚠️ 结构类改动必须预留一次 LB 验证（CLAUDE.md #13）——本次未预注册阈值，"
          "跑完 CV 后再定是否出提交。")

    if len(results) > 1:
        parts = [pl.read_parquet(os.path.join(oof_dir, f))
                 for f in sorted(os.listdir(oof_dir)) if f.endswith(".parquet")]
        oof_path = os.path.join(oof_dir, "oof_all.parquet")
        pl.concat(parts).write_parquet(oof_path)
        print(f"\nOOF 已存 {oof_path}（{sum(p.height for p in parts)} 行）")


if __name__ == "__main__":
    main()
