# AIC × RGBT-VGNet：AMA 数据适配与 RGB–TIR 融合实施方案

> AS_OF：2026-08-11（Asia/Shanghai）
> 结论依据：RGBT-GroundBench 官方论文/代码/已解压数据的只读全量标注审计，以及 AIC 现有 2,000 图像组的输入画像与 RGB–IR 配准代理审计。
> 重要边界：AIC 没有公开 GT；本文不把 AIC 的天气、真实目标尺寸或单样本改进写成已确认事实。

## 1. 一句话结论

**可行，但不能把 RGBT-VGNet 原样塞进当前 30B 模型，也不应把 RGBT-GroundBench 全量等权用于端到端微调。**

当前平台控制结果仍是 `Qwen3-VL-30B-A3B-Instruct-FP8 = 0.7757`；该提交含 13 条由 8B 模型补框，因此它只能作为当前分数控制点。新的多模态正式路线必须做到同模型合法输出，不再依赖跨模型 fallback。

最稳妥的迁移顺序是：

1. 冻结当前 RGB 主能力；
2. 先训练容量更高的 TIR LoRA / Adapter，使红外分支学会跨域表示；
3. 用零初始化残差把 IR 信息注入 RGB token，初始行为严格等价于 RGB-only；
4. 同时加入 IR 有效区域 mask 和轻量弱对齐鲁棒训练；
5. 在小基座上证明 RGB+IR 稳定优于 RGB-only 后，再移植到当前 30B 基座；
6. Query-conditioned reliability、LAVS/TPF 和 CoDAF 弱对齐留到后续单变量实验。

RGBT-GroundBench 最适合作为 **IR 适配器的监督预训练与机制验证集**，而不是 AIC 的唯一训练域。

## 2. 论文方法中哪些结论已经核实

### 2.1 RGBT-VGNet 的三个模块

- **AMA（Asymmetric Modality Adaptation）**：同一个冻结 CLIP 视觉底座上为 RGB 与 TIR 建立不同 LoRA adapter；TIR 使用更大容量。
- **LAVS（Language-Aware Visual Synergy）**：用语言引导 RGB/TIR 交互。
- **TPF（Tri-Prior Fusion）**：结合模态可靠性等先验进行融合。

官方实现的默认值不是抽象的“高/低 rank”，而是：

```text
RGB LoRA rank = 16
TIR LoRA rank = 48
```

代码位置：

- `D:\AIC赛题一数据集\03_RGBT_GroundBench\source_code\models\mmvg.py:968`
- `D:\AIC赛题一数据集\03_RGBT_GroundBench\source_code\train_val\mmvg_train.py:43`

它并没有复制两套完整视觉编码器，而是在一个冻结视觉底座上切换 `lora_rgb` 与 `lora_ir` adapter。这种实现非常适合迁移到显存受限的 Qwen-VL 原型。

### 2.2 AMA 是合理的第一步，不是凭直觉排序

官方消融中，基线只加入 AMA 后，三个子集测试 ACC@0.5 分别变化为：

| 子集 | 基线 | +AMA | 增益 |
|---|---:|---:|---:|
| RefFLIR | 46.19 | 71.17 | +24.98 pp |
| RefM3FD | 57.57 | 72.53 | +14.96 pp |
| RefMFAD | 54.63 | 65.27 | +10.64 pp |

加入 LAVS/TPF 后仍有进一步提升，但量级通常小于先解决红外域适配。因此本项目先验证 AMA，再研究更复杂融合，顺序有实证支持。

同时，论文的融合消融显示 `Naive Add` 在多个分组上弱于 TPF，说明不能把 RGB 与 TIR token 简单逐元素相加。

官方资料：

- 论文：https://arxiv.org/html/2512.24561
- 代码：https://github.com/crazyxiaoxi/RGBT-GroundBench
- 数据：https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset

## 3. 已解压数据的真实规模与内容

数据位置：

```text
D:\AIC赛题一数据集\03_RGBT_GroundBench\extracted
```

审计结果：

| 指标 | 结果 | 证据类型 |
|---|---:|---|
| RGB–TIR 图像对 | 21,535 | 公开数据事实 |
| grounding 实例 | 38,760 | 公开标注事实 |
| train / val / test | 26,604 / 2,032 / 10,124 | 公开标注事实 |
| Query 平均/中位词数 | 14.30 / 14 | 本地全量标注统计 |
| 真实 bbox 面积中位数 | 0.742% | 外部 GT 事实 |
| 官方 small 标签比例 | 56.78% | 外部 GT 事实 |
| 弱光比例 | 43.15% | 外部条件标签事实 |
| Foggy + Rainy | 23.07% | 外部条件标签事实 |

### 3.1 恶劣天气是否真的存在

**属实，而且是条件标签，不只是图像增强。** 21,535 个唯一图像对按天气统计为：

| 天气 | 图像对 | 比例 |
|---|---:|---:|
| Cloudy | 11,351 | 52.71% |
| Sunny | 5,361 | 24.89% |
| Foggy | 3,346 | 15.54% |
| Rainy | 1,477 | 6.86% |

训练 split 中也保留了 4,148 个 Foggy 实例和 1,912 个 Rainy 实例。它们应保留，因为真实雾雨是红外域适配的一部分；但不能因为它们存在，就在 AIC 训练中人为过采样到 23%。

### 3.2 数据质量不是完美的

全量 Query 自动筛查发现：

- 163 条含否定/目标缺失语言（0.42%）；
- 160 条左右含 bbox 元数据式语言；
- 否定/目标缺失与 bbox 元语言两个高风险规则合并后命中 187 条，其中 train split 为 125 条；
- 632 条含 uncertain / indistinct 等不确定语言（1.63%）。

可见的高风险例子包括：

```text
No person is present within the specified bounding box coordinates.
No motorcycle is present in the provided bounding box; it contains a bicycle.
The bounding box coordinates ... do not correspond ... cover part of the road.
```

因此训练前必须冻结清洗规则并生成 `clean_train_manifest`；当前两个高风险规则应从 train 中剔除 125 条，原始标注保持只读。`uncertain` 条目另设分组，不在规则未核验前一并删除。

## 4. RGBT-GroundBench 与 AIC 的风格差异

### 4.1 强匹配部分

- 输入输出完全相近：`RGB + TIR + Query -> RGB 坐标 bbox`；
- 小目标真实占比较高，真实 bbox 中位面积仅 0.742%；
- 包含低光、遮挡、远距离、雾雨；
- Query 平均 14.30 词，比普通 RefCOCO 更接近长描述 grounding；
- 有明确 train/val/test 和条件标签，适合做分层验证。

### 4.2 关键不匹配部分

| 维度 | RGBT-GroundBench | AIC | 影响 |
|---|---|---|---|
| 目标类别 | 约 95% 是车、人、卡车、公交、两轮车等道路参与者 | 建筑部件、区域、动物、无人机、灯、标牌等更广 | 全量端到端训练会收窄开放词汇能力 |
| 序数 | 0.03% | 24.14% | 无法覆盖 AIC 的 leftmost/second 等核心难点 |
| 区域/结构 | 很少 | 16.69% | passage/awning/building part 覆盖不足 |
| 深度关系 | 6.40% | 15.68% | RGBT 不能替代后续 Depth 训练 |
| 动作 | 52.08% | 19.18% | RGBT 的描述风格偏动作/场景化 |
| Query 长度 | 14.30 词 | 10.36 词 | 文本风格存在偏移 |
| 配准/黑边 | 相对干净 | AIC 黑边和弱对齐更严重 | AMA 单独使用会把错误 IR 位置注入 RGB |

其中 AIC 的 Query 分类是输入文本事实/规则画像，不是 bbox GT；但足以说明任务构成明显不同。

## 5. AIC 是否存在恶劣天气

当前只能给出严格边界内的结论：

- 9,555 条 AIC Query 中，严格天气词命中为 0；
- 只有 8 条 Query 含 puddle / wet 等间接湿润线索；
- 图像低光代理只命中约 0.5% 的图像组；
- AIC 没有公开 weather GT，所以不能证明“完全不存在雾雨”。

因此更合理的判断是：**目前没有证据表明恶劣天气是 AIC 主导分布。**

训练策略不应删除雾雨样本，也不应专门过采样雾雨。将天气标签作为分层变量，采用源/条件平衡采样，并检查正常光、低光、雾、雨四组是否都不退化。

## 6. AIC 的弱对齐风险为何不能忽略

### 6.1 黑边和几何代理对比

| 指标 | RGBT-GroundBench 抽样 | AIC 抽样 | 边界 |
|---|---:|---:|---|
| IR 黑边无效比例中位数 | 0.012% | 9.84% | 图像像素事实/规则统计 |
| IR 黑边无效比例 p90 | 1.02% | 18.03% | 同上 |
| 估计绝对旋转中位数 | 0.45° | 约 1.38° | 跨谱结构配准代理 |
| 估计垂直平移绝对值中位数 | 1.76% | 约 2.26% | 跨谱结构配准代理 |

RGBT-GroundBench 的 600 对抽样全部同尺寸，但论文构建过程会过滤显著错位和极端小目标；AIC 审计中 400 张 IR 有 196 张黑边无效区域达到 10% 以上。两个域的几何难度并不相同。

因此：

- 黑边必须成为显式 `ir_valid_mask`，不能当作冷背景输入模型；
- 最终 bbox 始终定义在 RGB 坐标系；
- IR token 在注入前必须带空间有效性；
- 不能使用 `RGB token + IR token` 的无条件逐点相加；
- 第一阶段至少使用轻量随机仿射错位增强，使 adapter 不依赖像素级完美重合；
- 第二阶段再单独验证 CoDAF / deformable alignment，不能与 AMA 同一轮同时引入。

CoDAF 的 Offset-guided Spatial Alignment 与 Dynamic Adaptive Fusion，以及 IJCAI 2024 CF-Deformable DETR 的跨模态 deformable attention，适合作为第二阶段参考：

- CoDAF：https://www.sciencedirect.com/science/article/pii/S1568494626014766
- CF-Deformable DETR：https://www.ijcai.org/proceedings/2024/84

## 7. 建议的 Qwen RGB–TIR 架构

### 7.1 第一阶段：AMA-only 原型

不复制两套完整 Qwen vision tower。建议：

```text
                  shared frozen vision tower
                 /                          \
RGB image -> RGB LoRA (rank 0/4/8)       IR image -> IR LoRA (rank 16/32/48)
                 \                          /
                   modality-specific projector
```

论文的 `16/48` 是 CLIP-B 上的已验证默认值，不应盲目当成 Qwen 的最优值。Qwen 原型先做 rank 网格：

```text
RGB rank ∈ {0, 4, 8}
IR  rank ∈ {16, 32, 48}
```

RGB rank 为 0 表示彻底冻结 RGB 分支，是保护现有能力最强的设置。

### 7.2 第二阶段：有界 IR 残差注入

建议融合形式：

```text
Z_fused = Z_rgb + sigmoid(g) * Align(IR_valid_mask * Z_ir)
```

其中：

- `g` 初始化为很小的负值，使初始 IR 权重接近 0；
- 初始模型行为等价于 RGB-only，而不是随机融合；
- `Align` 第一版只做 mask-aware token mapping + 错位增强；
- 后续单独替换为 CoDAF/Deformable alignment；
- 视觉 token 继续交给同一个 Qwen grounding/生成头输出 bbox。

这一设计的核心不是让 IR 永远参与，而是保证模型只能在训练证据支持时对 RGB 结果施加有限修正。

### 7.3 本轮明确暂缓的内容

- Query-conditioned IR reliability；
- LAVS 的完整文本引导双向交互；
- TPF 三先验门控；
- Depth 分支；
- 30B 端到端训练。

这些都很有价值，但应在 AMA 与基础融合被独立验证后逐个加入，否则无法判断收益来源。

## 8. 是否使用 RGBT 全量训练

答案分两层：

### 可以使用全部“清洗后的 train pool”训练 IR adapter

保留正常光、低光、晴、阴、雾、雨，因为 IR 域适配需要覆盖完整条件。使用固定官方 val/test 做机制验证，不用测试 split 训练。

### 不应等权端到端训练完整 Qwen

原因：

- MFAD 单源占 train 的 16,000/26,604；
- 类别高度偏向道路参与者；
- 序数和区域任务缺失；
- AIC 没有证据表明天气/低光是主导；
- 直接更新 RGB/语言主干容易重演 RefCOCO Ranker 的域负迁移。

推荐采样策略：

1. 先按 FLIR/M3FD/MFAD 做 source-balanced sampler；
2. 每个 source 内按光照/天气分层，而不是对恶劣天气过采样；
3. IR adapter 阶段只用 RGBT 监督；
4. 开始融合后加入 AIC 风格 RGB grounding replay，保持结构、序数、区域和开放词汇能力；
5. 每批随机执行 modality dropout，确保 IR 无效时同一模型仍能输出合法框，不依赖另一个模型 fallback。

## 9. 严格实验矩阵

先在 8B 或专用 grounding 小基座运行：

| 实验 | 新增变量 | 目的 |
|---|---|---|
| A0 | RGB-only frozen baseline | 控制组 |
| A1 | 对称 RGB/TIR LoRA | 验证“多一模态”本身是否有用 |
| A2 | AMA：低 RGB rank + 高 TIR rank | 验证非对称容量 |
| A3 | A2 + source/condition-balanced sampler | 降低 MFAD/天气偏置 |
| A4 | A3 + IR valid mask + affine misalignment augmentation | 适配 AIC 黑边/弱对齐 |
| A5 | A4 + zero-init residual fusion | 形成可迁移的同模型 RGB+IR 输出 |

每个实验必须报告：

- RGBT val/test 总 ACC@0.5；
- RGB-only / TIR-only / RGB+TIR；
- normal light / low light / fog / rain；
- small / normal；
- aligned-proxy / weak-aligned-proxy；
- RGB replay 上是否退化；
- 无效 bbox、解析失败和同模型重试率；
- 禁止使用外部模型补框。

只有当 A5 同时满足以下条件，才移植到当前 30B：

- RGB+IR 在外部验证中稳定优于同一基座 RGB-only；
- 正常光 RGB 能力无明显下降；
- 黑边与弱对齐分组不产生负增益；
- IR 全黑/缺失时仍能由同一模型输出合法框；
- 无跨模型 fallback。

## 10. 下一步操作顺序

1. 固化清洗规则，生成 `clean_train_manifest` 与剔除清单；
2. 实现 `ir_valid_mask` 和 source/condition-balanced sampler；
3. 在小基座完成 A0–A2，先证明 AMA 值得迁移；
4. 加入 A3–A5，重点验证 AIC 风格的黑边与仿射错位；
5. 冻结方案后再决定移植到 Qwen3-VL-30B-A3B；
6. 后续单独开启 CoDAF/CF-Deformable alignment；
7. 最后才加入 Query-conditioned reliability 与 Depth。

## 11. 产物与复现

只读分析脚本：

```text
D:\12525\Documents\pytorch\baseline_v0\tools\analyze_rgbt_groundbench.py
```

机器结果：

```text
D:\12525\Documents\pytorch\baseline_v0\outputs\rgbt_groundbench_aic_fit_audit_v1\
```

关键文件：

- `rgbt_instance_profile.csv`：38,760 条外部实例画像；
- `rgbt_condition_counts.csv`：split/source/天气/光照计数；
- `rgbt_alignment_sample.csv`：600 对黑边统计与 120 对配准代理；
- `rgbt_summary.json`：总统计；
- `rgbt_aic_comparison.json`：严格区分外部 GT 与 AIC proxy 的比较；
- `sha256_manifest.json`：输出哈希。

## 12. 最终决策

**第一篇论文值得作为主路线，但只先迁移 AMA 的思想和实现模式。**

真正适合 AIC 的第一版并不是“RGBT-VGNet 全部模块 + 30B 全量训练”，而是：

```text
冻结 RGB 能力
+ 高容量 IR Adapter
+ source/condition-balanced RGBT 训练
+ 黑边 valid mask
+ 弱错位增强
+ 零初始化有界残差融合
```

恶劣天气样本应保留但不应主导采样；RGBT-GroundBench 应全量作为清洗后的 IR adapter 训练池使用，但不能成为完整 Qwen 的唯一、等权训练域。CoDAF 弱对齐与 Query-conditioned reliability 都是正确的后续方向，不过必须在 AMA 单变量结论之后再加入。
