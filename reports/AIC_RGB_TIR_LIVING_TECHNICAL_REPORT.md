# AIC RGB–TIR 红外模块持续更新技术报告

> 文档性质：持续更新（Living Document）
> 当前版本：v3.0
> 最近更新：2026-08-23（Asia/Shanghai）
> 当前分支：`agent/rgbtir-module-phase19-sync`
> 当前阶段：D3/G0/G1/G2、Query-control、quality/registration gate 与 12 维 proxy 上限审计均已完成；当前状态 `STOP_CURRENT_12D_QUALITY_PROXY_ROUTE`
> 适用项目：2026 AIC 算法挑战赛赛题一「基于大模型的多模态视觉理解与推理」

## 阅读与更新约定

本报告用于统一记录红外模块从提出、工程化、训练、失败诊断、修复到后续融合的全过程。后续每完成一个阶段，应同步更新：

1. 顶部版本、日期和当前状态；
2. “过程进度”表；
3. 新增实验的配置、输入资产、指标、结论边界；
4. “当前唯一下一步”；
5. 文末更新日志。

报告严格区分四类信息：

- **平台事实**：AIC 官方平台返回的 ACC@0.5；
- **外部验证事实**：RGBT-GroundBench 等带 GT 数据上的机器指标；
- **诊断代理**：质量分数、假负例率、配准风险等启发式或模型代理；
- **待验证假设**：尚未经过训练或对照实验的结构与优化方案。

任何表征指标都不得直接写成 AIC 定位提升，任何无标签 AIC 输入画像都不得写成 GT 统计。

---

# 一、算法概述

本项目面向 AIC 多模态语言引导目标定位任务：模型接收可见光、红外、深度图像和英文 Query，输出目标在原始 RGB 图像坐标系中的归一化边界框。当前红外路线以已验证的 Qwen3-VL RGB-only 能力为锚点，不复制第二套完整视觉大模型，而是在共享 Qwen3-VL 视觉塔上增加 TIR 专用低秩适配器，通过同步几何预处理、红外有效视场 mask、DeepStack 多层特征接口和零初始化残差，为模型提供可拒绝的红外增量证据。算法设计强调“先保护 RGB，再学习红外；先证明表征健康，再验证 Query 相关性，最后才融合”。D2 之后，项目完成了 D3/D3A 层级结构审计、Frozen-8B 四臂和六臂控制、G0 因果接口 trace、G1/G1R 冻结 Query-to-ROI probe、G2 Layer16 residual、Query-control、quality/registration gate 和 12 维 proxy 上限测试。最新证据表明：D3_SP000 冻结特征在 G1R confirmation 中相对 Base TIR 有稳定正信号，但 G2 在 multi-query dev 失败；Query gate 对正确 Query 不优于错误 Query；现有 12 维全图质量 gate 不优于常数 gate，beneficial-vs-harmful AUC 上限仅 `0.59024`。因此当前没有 selected Adapter，也未证明 AIC 平台增益。下一步只允许本地进行候选/ROI 级 Rescue-Harm 可预测性上限测试，不能继续调旧 gate、进入 D4、打开 official-val、迁移 30B 或加入 Depth。

---

# 二、任务背景与控制基线

## 2.1 AIC 任务

输入：

```text
Visible RGB + Infrared/TIR + Depth + English Query
```

输出：

```text
目标在原始 RGB 图像上的 normalized bbox
[x1, y1, x2, y2]
```

当前研究范围限定为：

```text
RGB + TIR + Query → RGB bbox
```

Depth 暂缓，避免同时引入毫米深度、未知 JPG 深度域、第三模态编码和额外坐标映射，使实验无法归因。

## 2.2 RGB-only 控制基线

| 模型 | 输入 | AIC ACC@0.5 | 说明 |
|---|---|---:|---|
| Qwen3-VL-8B-Instruct | RGB + Query | 0.7582 | 红外模块的干净、低成本控制组 |
| Qwen3-VL-30B-A3B-Instruct-FP8 + 8B 补框 | RGB + Query | 0.7757 | 历史最高参考，但不是纯 30B 无补框控制 |

红外模块优先在 8B 开发，原因是：

- 8B 已有稳定、可复现的平台结果；
- 8B 训练和消融成本显著低于 30B；
- 8B/30B 视觉塔均有 27 层、1152 维 pre-merger 特征和 DeepStack 8/16/24 接口；
- 只有 8B 红外融合获得可归因净收益，才值得迁移到 30B；
- 后续正式模型不采用第二模型替代红外失败样本，红外无效时只允许同一模型退化为 RGB-only。

## 2.3 红外模块的安全目标

红外不是强制主模态，而是可拒绝的辅助证据：

```text
IR 无价值 / 质量差 / 有效视场不足 / 配准风险高 / Query 不需要热信息
→ TIR contribution = 0
→ 同一模型严格退化为 RGB-only
```

目标是在低光、热目标、遮挡、远距离和小目标场景产生更多 Rescue，同时将颜色、纹理、OCR 和普通光场景中的 Harm 控制到最低。

---

# 三、方案来源与初始算法设计

## 3.1 主要参考路线

红外模块首先参考 RGBT-GroundBench / RGBT-VGNet 的三类思想：

1. **AMA（Asymmetric Modality Adaptation）**：预训练视觉模型天然偏 RGB，TIR 分支需要更强适配容量；
2. **LAVS（Language-Aware Visual Synergy）**：先用语言筛选 Query 相关视觉区域，再做 RGB/TIR 交互；
3. **TPF（Tri-Prior Fusion）**：融合权重同时考虑照明、局部语义和全局可靠性。

当前工程没有照搬论文网络，而是先迁移最小可验证部分：

```text
RGB rank = 0（完全冻结）
TIR rank = 48
只适配 Qwen vision attention qkv / proj
```

这比论文中 RGB/TIR 双 LoRA 更保守，目的是先保护已经验证的 Qwen RGB 能力。

## 3.2 初始结构

```mermaid
flowchart TD
    RGB["Visible RGB"] --> P["Paired RGB-TIR Processor"]
    TIR["Infrared/TIR"] --> P
    P --> G["共享 resize / crop / pad / grid_thw"]
    P --> M["IR valid-FOV mask"]
    G --> RV["冻结 Qwen RGB Vision"]
    G --> TV["共享 Qwen Vision + TIR rank-48 Adapter"]
    RV --> R["RGB Layer 8/16/24/Final"]
    TV --> T["TIR Layer 8/16/24/Final"]
    R --> F["后续 zero-init masked residual fusion"]
    T --> F
    M --> F
    Q["English Query"] --> F
    F --> LLM["Qwen LLM / bbox generation"]
    LLM --> B["RGB 坐标系 bbox"]
```

## 3.3 关键工程接口

- 不修改安装环境中的 `transformers/modeling_qwen3_vl.py`；
- 通过 wrapper/hook 获取视觉层 8、16、24 和 final 的 1152 维 pre-merger hidden states；
- RGB/TIR 共享完全相同的几何变换和 `grid_thw`；
- 红外黑边只屏蔽与外边界连通的近黑 padding，避免误删内部暗目标；
- 最终 bbox 始终在 RGB 原始坐标系；
- `tir=None`、IR 无效或融合尺度为 0 时，输出与原生 RGB-only 等价；
- 不存在第二模型 fallback 接口。

## 3.4 为什么选择 DeepStack 8/16/24

Qwen3-VL 不只在 final merger 注入视觉信息，DeepStack 的 8/16/24 层也参与后续多模态理解。只修改 final projector 会遗漏大部分层级视觉交互。因此本项目从一开始就将红外适配与诊断覆盖 Layer 8/16/24/final，并把 Final 作为诊断层而非强门禁层。

---

# 四、数据准备、清理与模态审计

## 4.1 RGBT-GroundBench

| 项目 | 数量 |
|---|---:|
| 原始 grounding 实例 | 38,760 |
| 唯一 RGB/TIR 图像对 | 21,535 |
| 官方 train | 26,604 |
| clean train | 26,477 |
| official val | 2,032 |
| official test | 10,124 |

训练清理排除 127 条记录：

- `bbox_meta_language`；
- `negative_or_absent_language`；
- 两个 train/val 重叠图像对对应的 train 记录。

val/test 不因语言质量标志删样本，保持官方评测口径。

## 4.2 数据隔离

Phase 1.6/1.7 使用固定切分：

| 切分 | 记录数 | 唯一图像对 |
|---|---:|---:|
| repair probe train | 4,096 | 4,096 |
| repair dev | 1,024 | 1,024 |
| repair full train | 24,612 | 13,822 |
| official val | 2,032 | 1,115 |

已验证：`probe_dev=0`、`full_dev=0`、`train_official_val=0`。

## 4.3 黑边、有效视场与 Grid

| 数据 | 成功读取 | 检出黑边 | 低信息 IR | 可用 IR | RGB/TIR grid 一致 |
|---|---:|---:|---:|---:|---:|
| RGBT-GroundBench | 21,535/21,535 | 4,056 | 0 | 21,534 | 21,535/21,535 |
| AIC 唯一图像组 | 2,000/2,000 | 1,439 | 0 | 1,998 | 2,000/2,000 |

AIC 统计是输入审计事实，不包含 bbox GT，因此不能推出红外一定提升或下降。

## 4.4 数据域风险

RGBT-GroundBench 与 AIC 任务形式高度接近，但仍存在域差异：

- AIC 中可见黑边、倾斜有效边界和弱配准风险；
- AIC Query 包含大量关系、序数、区域和建筑部件；
- RGBT-GroundBench 的标注与筛选更偏道路参与者和可稳定标注目标；
- official val 通过不等价于 AIC 平台增分。

因此后续只能用 RGBT 数据选择训练方案，最终是否保留红外模块必须由受控 AIC 平台提交确认。

---

# 五、实现方案与训练目标演进

## 5.1 Phase 1 原始目标

Phase 1 只训练 TIR rank-48 Adapter：

```text
paired RGB-TIR ROI alignment
+ same-image background margin
```

RGB 视觉塔、LLM、fusion gate 和 bbox head 全部冻结。

该目标回答的是：

> TIR Adapter 能否学习冻结 RGB Teacher 的目标区域表征？

它不能回答 Query 是否受益，也不能回答 bbox 是否提高。

## 5.2 Phase 1.6 去坍缩候选

| 候选 | 损失结构 | 目的 |
|---|---|---|
| C0 | paired alignment + background margin | 历史控制 |
| C1 | C0 + 256 个跨图 RGB 负例 InfoNCE | 防止不同目标聚集 |
| C2 | C1 + relational distillation | 保留 RGB Teacher 的实例关系结构 |

负样本固定为：

```text
128 同来源 + 64 同条件 + 64 全局随机
```

禁止来自相同图像对，并绑定固定数据切分、随机种子和 Teacher Bank fingerprint。

## 5.3 当前建议的 D1 目标

保留 C2 主体：

```text
0.25 × paired alignment
+ 1.00 × InfoNCE
+ 0.50 × relational distillation
+ 0.10 × background margin
```

新增逐层 Base-relative retention：

\[
L_{retain,l}=\operatorname{ReLU}
\left(\cos_{base,l}-\epsilon_l-\cos_{adapted,l}\right)
\]

建议容差：

```text
epsilon_8  = 0.02
epsilon_16 = 0.02
epsilon_24 = 0.01
Final      = 仅诊断
```

Retention 与再次提高 paired loss 不同：只有 adapted TIR 比 Base TIR 差超过容差时才产生梯度，避免持续强迫 TIR 完全复制 RGB。

---

# 六、过程进度

| 日期 | 阶段 | 状态 | 主要完成内容 | 核心结论 |
|---|---|---|---|---|
| 2026-08-11～12 | 数据与论文审计 | 完成 | RGBT-GroundBench、AIC 黑边/位深/配准风险审计 | 采用 8B-first、RGB 锚点、TIR 可拒绝残差路线 |
| 2026-08-12 | Phase 0 | `PHASE_0_GO` | manifest、processor、mask、grid、坐标、包装器等价性 | 工程链路可训练，gate=0 与 RGB-only 完全等价 |
| 2026-08-12 | Phase 1 smoke/overfit/tracer | GO | 真实 Qwen 视觉权重、10 条 smoke、100 条过拟合、400 条 tracer | 梯度只进入 TIR Adapter，RGB 哈希不变 |
| 2026-08-12 | Phase 1 full warmup | GO | clean train 26,477/26,477 | 配对 alignment 明显改善，但未验证表征区分度 |
| 2026-08-12 | Phase 1.5 full validation | `NO_GO` | official val 2,032 条检索与坍缩检查 | R@1/R@5 提升，但 Layer 8/16/24 严重低秩坍缩 |
| 2026-08-13 | Phase 1.6 probe | `NO_GO` | C0/C1/C2 去坍缩候选 | C2 最佳；有效秩恢复，但 alignment 门禁失败 |
| 2026-08-13 | 云端资产恢复 | 完成 | 重建 Teacher Bank、C1/C2 Adapter、summary、逐记录诊断 | 解决实例释放导致的证据缺失；C0 历史 summary 仍缺失但不阻塞 |
| 2026-08-13 | Phase 1.7A+ | `COMPLETE` | 1,024 条逐层绝对诊断、质量和假负例代理分析 | 确认真实中层漂移，唯一后续为 `D1_RETENTION` |
| 2026-08-14～15 | Phase 1.8 D1 retention | `D1_L050 selected` | retention probe、full train、四 checkpoint、sealed official-val | 检索显著增强且漂移下降，但 official-val 仍因有效秩、alignment 与旧 safety 状态得到 NO-GO |
| 2026-08-15 | Phase 1.8R 云端恢复 | `PHASE_18_FULL_NO_GO` | 5090 sm_120 环境、持久 Stage A、2,032/2,032 official-val、回传校验 | 训练完整；NO-GO 是科研门禁结论，不是运行失败 |
| 2026-08-16 | Phase 1.8R-Audit | `BOTH_REQUIRED` | 三套检索口径、记录/图像对秩、ExcessDrift、安全状态复核 | 历史 safety 失败可修；记录级秩与绝对漂移仍是真实模型问题 |
| 2026-08-16 | Phase 1.9-D2 probe | `PHASE_19_D2_PROBE_GO` | G010/G025/G050 单变量谱几何保持 | G025 在秩恢复与 nonpaired P95 之间折中最佳；G050 因 P95 NO-GO |
| 2026-08-16～17 | Phase 1.9-D2 full | `PHASE_19_D2_FULL_NO_GO` | 23,391 step、四 checkpoint、双 dev 选择、Stage A 回传 | Multi-query 最低秩/Base 最好 84.02% < 85%；未产生 selected Adapter，official-val 保持封存 |
| 2026-08-17 | D2 Layer Audit | `COMPLETE` | 四 checkpoint、双 dev、Layer 8/16/24 逐层谱审计 | 三层轻度压缩，L8 最低；不是单层灾难 |
| 2026-08-18 | Phase 1.9-D3/D3A | `NO_GO / COMPLETE` | SP000/005/010/020、严格无效 ROI 排除、层轨迹与 AIC 同风格诊断 | SP020 绝对过线但材料性不足；收益集中 L24 且伴随 P95 风险 |
| 2026-08-20 | Frozen-8B 四臂 | `FROZEN8B_FOUR_ARM_NO_GO` | RGB/correct/random/misaligned TIR，295 records | Correct TIR 未稳定优于 RGB 或负对照 |
| 2026-08-21 | Frozen-8B 六臂 | `PROTOCOL_CONTROL_NO_GO_REPAIR_INTERFACE` | Native/null/correct 的 A/B 重复控制 | null 双图已改变输出，旧双图比较存在输入结构混杂 |
| 2026-08-21 | G0 Interface Trace | `PHASE19_INTERFACE_TRACE_GO` | Layer16 sidebranch、alpha=0、TIR=None/nonfinite | 因果 sidebranch seam 与 RGB 硬旁路成立 |
| 2026-08-21～22 | G1/G1R | `DEV_GO / CONFIRMATION_GO` | 六历史臂冻结特征、共享 K=8 head、锁定 confirmation | D3_SP000 有可读任务信息，但不是端到端 bbox GO |
| 2026-08-22 | G2 Layer16 | `G2_DEV_NO_GO` | 只训练 Layer16 projector/scale，四 checkpoint，双 dev | semantic 通过、multi-query 失败；静态 residual 泛化不稳 |
| 2026-08-22～23 | Query Gate / QueryControlV2 | `DUAL_DEV_NO_GO` | correct/wrong/shuffled Query 控制 | wrong Query 不更差；停止 Query-conditioned residual |
| 2026-08-23 | Quality/Registration Gate | `SCREEN_NO_GO` | 12 维 query-free gate、constant/full/shuffled/misaligned 控制 | 残差有静态信号，但 learned gate 无自适应价值 |
| 2026-08-23 | 12D Proxy Upper Bound | `STOP_CURRENT_12D_QUALITY_PROXY_ROUTE` | pair-disjoint ExtraTrees/Ridge、置换与 bootstrap | 弱连续相关存在，benefit/harm AUC 近随机，停止旧 proxy |
| 2026-08-23 | GitHub 技术归档补齐 | 本地文档完成 | 总路线报告、README 与 Living Report 更新 | 代码/报告需分批审查后再 commit/push；仍无 AIC 多模态提交 |

---

# 七、关键实验结果

## 7.1 Phase 1：对齐改善

云端环境：RTX 4090 24GB、PyTorch 2.6.0+cu124、Transformers 4.57.6、Qwen3-VL-8B 固定 revision。

| 项目 | 结果 |
|---|---:|
| 训练记录 | 26,477/26,477 |
| skipped/fallback | 0/0 |
| 可训练参数 | 8,957,952（全部 TIR Adapter） |
| official-val-100 loss | 0.152710 → 0.073343 |
| 相对改善 | 51.97% |

分层 paired cosine：

| 层 | Base | Adapted |
|---|---:|---:|
| Layer 8 | 0.71715 | 0.87767 |
| Layer 16 | 0.78759 | 0.90987 |
| Layer 24 | 0.95479 | 0.97669 |
| Final | 0.99539 | 0.99694 |

`51.97%` 是表征 alignment loss 改善，不是 ACC@0.5。

## 7.2 Phase 1.5：低秩坍缩

| 层 | Base effective rank | Adapted effective rank | R@1 Base→Adapted | R@5 Base→Adapted |
|---|---:|---:|---:|---:|
| Layer 8 | 99.45 | 30.39 | 26.08%→50.39% | 44.00%→70.03% |
| Layer 16 | 77.42 | 22.72 | 42.22%→71.06% | 61.07%→85.63% |
| Layer 24 | 59.87 | 22.94 | 33.81%→65.01% | 53.79%→83.61% |
| Final | 约 1.02 | 约 1.02 | 23.97%→46.90% | 40.50%→64.27% |

结论：正确配对变得更近，但大量不同目标被压缩到相似子空间。Phase 1 Adapter 不能直接融合。

## 7.3 Phase 1.6：去坍缩结果

| 指标 | Base TIR | C0 | C1 | C2 |
|---|---:|---:|---:|---:|
| 平均 R@5 | 60.45% | 69.50% | 82.39% | **83.56%** |
| 平均 R@1 | — | 48.93% | 57.65% | **60.03%** |
| 最低有效秩/Base | — | 36.63% | 89.29% | **89.77%** |
| 最低有效秩/RGB Teacher | — | 37.83% | 81.47% | **81.90%** |
| paired-shuffled margin | — | 0.0557 | **0.1800** | 0.1698 |

结论：InfoNCE 是恢复跨实例区分度的关键，relational distillation 对检索和有效秩有小幅正贡献。C2 是最值得保留的修复基座，但它没有通过 alignment 门禁。

## 7.4 Phase 1.7A+：确认真实漂移

| 层 | Base paired cosine | C2 paired cosine | C2 alignment loss - Base loss |
|---|---:|---:|---:|
| Layer 8 | 0.73645 | 0.60752 | +0.12893 |
| Layer 16 | 0.79743 | 0.64481 | +0.15261 |
| Layer 24 | 0.95983 | 0.57607 | **+0.38376** |
| Final | 0.99664 | 0.99674 | -0.00010 |

Layer 24 漂移最大。Final 在 Base/C2/Teacher 中有效秩都接近 1，属于结构性饱和，只用于诊断，不能代表中层表征健康。

## 7.5 RGB Teacher 质量关联

| 信号 | 与 C2 漂移的 Spearman | bootstrap 95% CI |
|---|---:|---:|
| 全图 RGB 亮度 | -0.1486 | [-0.2077, -0.0870] |
| 全图 RGB 模糊度 | -0.1201 | [-0.1821, -0.0594] |
| ROI RGB 对比度 | -0.1370 | [-0.1981, -0.0780] |
| ROI RGB 熵 | -0.1404 | [-0.2031, -0.0782] |
| IR 有效视场比例 | -0.0716 | [-0.1347, -0.0104] |
| RGB Teacher 目标—背景 margin | -0.2464 | [-0.3054, -0.1862] |

方向稳定但效应量较弱，不能把 RGB 质量认定为主要根因，因此暂不进入质量加权训练路线。

## 7.6 假负例代理

| 指标 | 结果 |
|---|---:|
| 固定 256 跨图负例词法潜在假负例率 | 8.50% |
| 高 Query 词集相似率 | 0.54% |
| 模型 top-neighbor 潜在假负例代理率 | 19.01% |
| same-pair 违规 | 0 |

这些结果不是人工真值，不足以证明 InfoNCE 假负例是 C2 漂移主因，因此不优先进入 false-negative-aware 分支。

## 7.7 Phase 1.8R：D1_L050 full 与 sealed official-val

D1 在 C2 基础上加入 Base-relative retention。`lambda_ret={0.10,0.25,0.50}` 的单变量 probe 选择 `D1_L050`，随后从固定起点完成 full train。repair-dev 的 50%/75%/100% checkpoint 均通过，预注册选择器最终选择 100%。sealed official-val 完成 `2,032/2,032` 条记录和 `1,115/1,115` 个图像对，skipped 为 0。

| 层 | R@5 Base | R@5 D1_L050 | D1/Base effective rank | D1/RGB effective rank | paired cosine Base→D1 |
|---|---:|---:|---:|---:|---:|
| Layer 8 | 44.00% | 91.34% | 80.38% | 77.81% | 0.70043→0.61908 |
| Layer 16 | 61.02% | 90.70% | 81.80% | 96.47% | 0.77965→0.68567 |
| Layer 24 | 53.69% | 75.39% | 81.79% | 84.73% | 0.95378→0.70971 |

积极证据：平均 R@5 `85.81%`、最低 paired-shuffled margin `0.18267`、margin bootstrap 95% 下界 `0.17936`，且 RGB 哈希和 gate=0 等价性成立。阻塞证据：三个判别层的 D1/Base 有效秩均低于 `0.85`，总体 alignment loss 从 `0.16077` 上升到 `0.25878`。旧 safety 总门禁还因 Adapter 的 `requires_grad` 标志没有在 official-val 前清除而失败，但没有参数更新、RGB 污染或 fallback 证据。

因此 `PHASE_18_FULL_NO_GO` 是正确的历史结论；它不能被检索增益覆盖，也不能被 safety 合同问题整体推翻。

## 7.8 Phase 1.8R-Audit：验证器问题与模型问题并存

本地审计不训练、不修改 Adapter，只复算三种检索口径、记录/图像对聚合有效秩、ExcessDrift 与安全合同，最终输出 `BOTH_REQUIRED`：

- safety 历史失败属于可修复的推理冻结状态合同问题；
- 记录级 D1/Base 有效秩仍为 `0.8038 / 0.8180 / 0.8179`，是真实模型问题；
- 图像对聚合后提高到 `0.8700 / 0.9090 / 0.8862`，说明重复 Query 会放大记录级秩惩罚，但不能据此取消记录级口径；
- 全局绝对对齐漂移 `0.098007`，显著高于 `0.020000` 门槛；
- FLIR、弱光和部分天气条件存在正 ExcessDrift，需要在未来 task-relevant 分组指标中继续跟踪。

该审计把“验证器需要修”和“模型仍需修”分开，避免两种错误：把全部 NO-GO 归因于代码，或把 safety 状态错误误认为权重污染。

## 7.9 Phase 1.9-D2 probe：谱几何保持

D2 从同一 `D1_L050` 起点出发，只增加 Base-TIR 邻域几何保持：

```text
L_D2 = L_D1_L050 + lambda_geometry × L_base_tir_geometry
lambda_geometry ∈ {0.10, 0.25, 0.50}
```

| 候选 | Multi-query rank 增益 | Semantic rank 增益 | Multi-query R@5 | Semantic R@5 | nonpaired P95 增量 | 结论 |
|---|---:|---:|---:|---:|---|---|
| D2_G010 | +2.23 pp | +3.65 pp | 90.06% | 89.78% | 0.0082 / 0.0078 | GO |
| D2_G025 | **+3.52 pp** | **+5.00 pp** | 90.20% | 89.81% | 0.0150 / 0.0161 | **GO / selected** |
| D2_G050 | +5.38 pp | +6.95 pp | 90.72% | 89.68% | 0.0233 / 0.0266 | NO-GO |

`D2_G050` 的秩恢复最强，但两个 dev 的 nonpaired P95 增量超过预注册 `0.02`；`D2_G025` 是秩恢复与过度重排之间的门禁内折中，因此只允许它进入 full train。

## 7.10 Phase 1.9-D2 Full：局部 probe 未完全泛化

full train 使用 `23,391` 条记录、`13,310` 个图像对，保存四个持久 checkpoint，并在互相隔离的 semantic dev 与 multi-query dev 上选择。结果如下：

| 进度 | step | Multi-query rank/Base | Semantic rank/Base | Multi-query R@5 | Semantic R@5 | 未通过项 |
|---|---:|---:|---:|---:|---:|---|
| 25% | 5,848 | 82.37% | 91.62% | 89.41% | 89.88% | 绝对秩、双 dev 秩增益 |
| 50% | 11,696 | **84.02%** | 90.78% | **91.51%** | 90.43% | 绝对秩 |
| 75% | 17,544 | 83.48% | 91.55% | 91.40% | 90.53% | 绝对秩 |
| 100% | 23,391 | 83.55% | **91.63%** | 91.46% | **90.56%** | 绝对秩 |

50%/75%/100% checkpoint 已通过双 dev 相对秩增益、R@5 保留、nonpaired P95 控制和正 margin；唯一贯穿全部 checkpoint 的阻塞项是 `both_dev_absolute_rank=false`。最好值 `0.840168` 距预注册 `0.85` 还差 `0.009832`，且 50% 以后趋于饱和。由于没有合格 checkpoint：

- 状态为 `PHASE_19_D2_FULL_NO_GO`；
- `selected_checkpoint` 与 `selected_adapter` 为空；
- sealed official-val 没有打开；
- Stage A 的四个 checkpoint、摘要、fingerprint、配置和日志已回传并逐项验证。

这不是 Phase 1 式严重坍缩。更准确的结论是：D2 明显缓解了 multi-query 低秩问题并保留强检索，但单一全局 `geometry_weight=0.25` 对最差层/最难分布的恢复仍不足，继续相同配置延长训练没有证据价值。

---

# 八、遇到的主要问题与解决过程

## 8.1 JPG/PNG 与 Depth 位深差异

问题：不同文件格式和模态的实际 dtype、通道和数值含义不同，不能统一除以 255。

处理：

- RGB/TIR 按真实解码结果归一化；
- PNG uint16 Depth 保留毫米值和 invalid mask；
- JPG uint8 Depth 单独标记为未知相对深度域，不解释为毫米；
- 红外三通道一致性和黑边单独审计。

## 8.2 红外黑边、倾斜边界与弱配准

问题：TIR 有大量黑边和倾斜有效视场，RGB/TIR 仅尺寸一致不代表像素级严格对齐。

处理：

- 边界连通黑区生成 `ir_valid_mask`；
- RGB/TIR 使用相同 resize/grid；
- mask 同步下采样到 patch/merged token；
- 当前不自动仿射校正，弱配准模块必须等待融合 Harm 证据后再进入。

## 8.3 Phase 1 “对齐很好”却发生坍缩

问题：训练只优化正确配对，没有跨图负例，模型可通过压缩所有目标表示降低 paired loss。

处理：Phase 1.5 引入跨样本检索、非配对相似度和 effective rank 硬门禁；Phase 1.6 加入 InfoNCE 和 relational distillation。

## 8.4 C2 去坍缩后又出现中层漂移

问题：强判别约束把不同目标分开，也把 RGB/TIR 共享语义拉开，Layer 24 尤其严重。

处理：Phase 1.7A+ 重新计算绝对 drift，排除“小分母百分比失真”；选择 Base-relative retention，而不是简单重新提高 paired 权重。

## 8.5 云端实例释放导致关键资产丢失

问题：早期只保留报告和 manifest，C1/C2 Adapter、Teacher Bank、逐记录诊断留在临时实例，无法继续 Phase 1.7A+。

处理：重新构建共同 Teacher Bank，按相同初始化和口径重跑 C1/C2，将归档拉回本地并核验 SHA-256。当前已恢复：

- Teacher Bank：5,120 对；
- C1 Adapter SHA-256：`0D56149083B9EFCFA30DDBB30904A70D631EBA5E080EA2DCD63F123BEFAA85EE`；
- C2 Adapter SHA-256：`2832225394CD884D5BA972D9BCD0E19CE8AF8715657DC9675E60AB75B103E786`；
- repair-dev：1,024 条，图像对隔离；
- 回传归档 SHA-256：`3AEAF519809D9FE7BE2818F7C36787667A10444422465FE4A98B8E8CBD37DECE`。

后续所有云端训练必须先定义持久化和回传清单，不能只保存 Markdown 汇总。

## 8.6 报告生成器与机器结果不一致

问题：旧报告在资产已齐全时仍输出 `ASSET_BLOCKED`；C2 诊断在 subgroup 表写出后才连接，造成字段恒为不可用。

处理：修复执行顺序与报告条件，新增回归测试。Phase 1.7A+ 当前为：

```text
ASSETS_READY
1024/1024 completed
8 tests passed
repeat-run SHA-256 byte-stable
```

## 8.7 长任务边界不清导致成果险些丢失

问题：Phase 1.8 首次 full run 把 Teacher Bank、训练、选点和 official-val 放在一个长进程中，official-val 进行到 `569/1115` 时进程被系统终止；训练边界资产尚未独立归档，实例余额耗尽后云端临时文件随实例释放。

处理：把执行合同固定为两个可独立交付的边界：

```text
Stage A = Teacher Bank / full train / 四 checkpoint / dev 选择 / 立即归档回传
Stage B = 独立进程 official-val / cache parts / final archive
```

checkpoint 必须位于云盘真实目录，禁止 `/dev/shm` 和悬空符号链接；所有长任务必须写 PID、阶段日志、failure state、receipt 与 SHA-256 manifest。Phase 1.8R 和 Phase 1.9 已按该合同执行。

## 8.8 Probe GO 不等于 full-train GO

问题：`D2_G025` 在 4,096 条 probe train 上把 multi-query rank/Base 提高到 `0.84197`，通过相对改进门禁；但 full train 的最佳值只有 `0.84017`，仍未达到绝对 `0.85`。

处理：候选筛选与发布门禁严格分层。Probe 只回答“方向是否值得扩大”，full 双 dev 回答“是否有可发布 checkpoint”，sealed official-val 只在双 dev GO 后打开。局部 probe 收益不得直接写成全量泛化或平台收益。

## 8.9 单一聚合最小值不足以定位失效层

问题：Phase 1.9-D2 full 的 `minimum_effective_rank_ratio_vs_base` 是 Layer 8/16/24 的最小值。Stage A 摘要能证明 multi-query 存在绝对秩缺口，但没有保留每层完整谱，因此不能从本轮归档断言具体由哪一层主导。

处理：下一轮先做 checkpoint 逐层审计，固定输出每层 effective rank、participation ratio、奇异值谱、R@1/R@5、margin 和 nonpaired P95，再决定是否使用层选择性谱保持。不得先假设 Layer 8、16 或 24。

---

# 九、算法创新点与工程价值

## 9.1 面向 Qwen3-VL 的非对称红外适配

把 RGBT-VGNet 的 AMA 思想迁移到 Qwen3-VL 视觉塔：RGB 主路完全冻结，TIR 使用 rank-48 Adapter，在不破坏已验证 RGB 能力的前提下学习跨谱表示。

## 9.2 动态网格与黑边感知的成对处理

共享几何变换、动态 `grid_thw` 和边界连通 mask 保证不同分辨率、长宽比和倾斜黑边下的 token 对应关系，避免把黑色 padding 当作热信息。

## 9.3 多层 hook 与 DeepStack 一致性

直接验证 Layer 8/16/24/final，而不是仅观察最终 projector，从而发现“Final 看似正常、中层已经漂移”的隐藏问题。

## 9.4 零初始化、同模型可拒绝残差

所有未来融合采用 `tanh(alpha)` 且 `alpha=0` 初始化。IR 缺失、低质量或无关时，同一模型退化为原生 RGB-only，不依赖第二模型补框。

## 9.5 从单一对齐指标升级为多目标硬门禁

同时检查：

- alignment；
- R@1/R@5；
- effective rank；
- nonpaired cosine P95；
- paired-shuffled margin；
- 分组退化；
- RGB 参数哈希和安全等价性。

该门禁先发现 Phase 1 坍缩，再发现 C2 过度漂移，避免错误进入融合。

## 9.6 Base-relative retention 路线

D1 不要求 TIR 完全复制 RGB，而是将 Base TIR 作为最低共享语义边界，只惩罚越界漂移，目标是在“对齐、判别、红外独有信息”之间建立可控约束。

---

# 十、当前结论边界

## 已验证

- 数据、坐标、mask、grid 和 RGB-only 等价链路可信；
- TIR rank-48 Adapter 可以稳定训练；
- 单纯 paired alignment 会产生低秩坍缩；
- InfoNCE 显著恢复跨实例区分度；
- relational distillation 对 C2 检索和有效秩有小幅正贡献；
- C2 在 Layer 8/16/24 发生真实漂移，Layer 24 最严重；
- D1_L050 retention 显著降低 C2 漂移并保留强检索，但 official-val 仍存在记录级有效秩和绝对 alignment 问题；
- Phase 1.8R 的 safety 历史失败包含可修复的推理冻结状态问题，但该问题不解释真实的秩和漂移缺口；
- D2 Base-TIR 邻域几何保持对有效秩恢复有效，`D2_G025` 的 probe 折中优于 G010/G050；
- D2 full 的 multi-query 绝对有效秩最好为 `0.840168`，未达到预注册 `0.85`，且 50% 以后没有继续改善；
- Phase 1.9-D2 Full 没有合格 selected Adapter，sealed official-val 没有打开；
- D3_SP020 的谱改善主要集中 Layer 24，L8 已平台化，且深层 nonpaired P95 风险同步增大；
- Frozen-8B 四臂未证明 Correct TIR 稳定优于 RGB/random/misaligned；六臂进一步发现旧双图输入协议混杂；
- G0 sidebranch 已证明 alpha=0 和 invalid TIR 可严格旁路；
- G1R 证明 D3_SP000 冻结特征存在共享 head 可读的任务信息，并在锁定 confirmation 复现；
- G2 Layer16 residual、Query gate 和 QueryControlV2 均未通过双 dev／Query-control 门禁；
- 现有 12 维 quality/registration learned gate 不优于 constant gate，且 beneficial-vs-harmful AUC 上限仅 `0.59024`；
- 当前唯一有证据支持的下一步是 candidate/ROI-level Rescue-Harm 可预测性上限测试。

## 尚未验证

- candidate/ROI 局部特征能否可靠预测 residual 的 Rescue/Harm；
- 最小 ROI-local gate 能否同时通过 semantic 与 multi-query dev；
- 新的 image-pair/sequence-disjoint confirmation 能否复现；
- RGB+TIR 是否提高端到端 RGBT grounding ACC；
- AIC 平台 ACC 是否超过 8B RGB-only 0.7582；
- Shared/Complementary 双分支是否必要；
- TIR proposal + RGB ranking 是否比 hidden residual 更可控；
- 红外模块能否安全迁移到 30B；
- Depth 融合收益。

---

# 十一、下一阶段实施计划

## 11.1 Phase 1.9-D2-Layer-Audit：无训练逐层复核

固定使用已回传的四个 checkpoint 和相同 semantic/multi-query dev，不改变权重、不打开 official-val。必须逐 checkpoint、逐层保存：

- Layer 8/16/24 的 absolute effective rank、Base/RGB 比例和 participation ratio；
- 奇异值谱、累计谱能量、协方差迹与条件数；
- R@1/R@5、paired cosine、paired-shuffled margin、nonpaired mean/P95；
- semantic 与 multi-query 的差异；
- 按来源、同图多 Query 数量、小目标、遮挡和黑边风险的分组结果；
- 绝对 alignment loss 与相对 improvement，避免小分母放大。

审计只回答“哪个层、哪类样本、哪个训练阶段形成 0.840168 瓶颈”。如果现有 Stage A 缺少逐层 embedding，允许在固定 dev 上重新编码；结果必须绑定 checkpoint SHA-256、manifest SHA-256 和代码版本。

## 11.2 Phase 1.9-D3：最小层选择性修复 probe

只有 Layer-Audit 定位瓶颈后才定义候选。优先保持单变量：

1. 从同一 `D1_L050` 起点初始化，不从 D2 probe/full checkpoint 续训；
2. 只对失效层提高 geometry/retention，其他层保持 D2_G025；
3. 或在失效层降低 contrastive 权重，二者不能同轮同时变化；
4. 固定数据、负例、Teacher Bank、种子、优化器和步数；
5. 增加同图多目标关系保持，但不得把同一图像对当作跨图负例；
6. 保持 RGB rank=0、无 Query/fusion/Depth/fallback。

Probe 继续使用 semantic 与 multi-query 双 dev，并要求：

- 两个 dev 的最低 rank/Base 均 `>=0.85`；
- 两个 dev 相对 D1 的 rank 增益均 `>=0.02`；
- semantic R@5 下降 `<=0.005`，multi-query R@5 下降 `<=0.010`；
- nonpaired P95 增量 `<=0.02`；
- paired margin 为正；
- 50% 后不出现秩持续回落；
- 所有输出有限，RGB 哈希不变。

不得把本轮门槛事后降到 `0.84` 来追认 D2_G025。

## 11.3 Frozen Query–TIR 诊断

只有新的 representation checkpoint 通过双 dev 才进入正式 Query 诊断。比较 Base TIR、D1_L050 和新候选在以下任务中的表现：

- Query-to-target retrieval；
- target-vs-distractor；
- same-category instance discrimination；
- target-vs-reference；
- target-vs-part；
- small-target、low-light、occlusion 分组。

视觉主干冻结，使用同一轻量语言条件评分头，并设置随机/冻结头控制，避免把评分头能力误认为 Adapter 能力。Query 诊断不得参与回头选择 representation checkpoint。

## 11.4 Phase 2：最小安全融合

严格单变量顺序：

```text
A. RGB-only control
B. TIR-only diagnosis
C. RGB + TIR static tiny zero-init gate
D. C + Query-aware gate
E. D + IR quality gate
F. 仅在 weak-alignment Harm 明显时增加 registration gate
G. modality dropout / degradation guard
```

第一版继续使用 Qwen 原生 bbox 输出，避免同时改变融合和输出协议。

## 11.5 AIC 平台与 30B 迁移

8B RGB–TIR 必须以 `0.7582` 为控制，建议实质提升线：

```text
ACC@0.5 >= 0.7632
```

同时要求非法框为 0、无第二模型补框，并记录 gate 分布。只有 8B 获得稳定净收益，再迁移到 30B pre-merger 1152 维接口；Depth 继续暂缓。

---

# 十二、停止条件

出现任一情况即停止当前支线：

1. Layer-Audit 无法复现 Stage A 摘要或 checkpoint/manifest 指纹不一致；
2. 新的层选择性修复仍无法让两个 dev 的最低 rank/Base 同时达到 `0.85`；
3. 秩恢复以 nonpaired P95、R@5 或 alignment 的显著恶化为代价；
4. Query–TIR probe 不优于 Base TIR；
5. Fusion 的 Harm 接近或超过 Rescue；
6. 普通光、颜色、纹理或 OCR Query 系统性下降；
7. gate 长期趋近全 0 或全 1；
8. weak-alignment 样本未能自动降低 IR 权重；
9. AIC 8B 未达到预注册实质提升线；
10. 出现非法 bbox、TIR 坐标系输出或跨模型 fallback。

回退方式永远是：

```text
关闭同一模型的 TIR gate → 恢复原 RGB-only 路径
```

---

# 十三、团队分工

当前仓库未登记完整参赛成员姓名，因此本报告不虚构人员信息。建议团队补充下表中的姓名：

| 角色 | 当前职责 | 姓名 |
|---|---|---|
| 算法负责人 | RGB–TIR 架构、损失、门禁与实验决策 | 待团队填写 |
| 数据工程 | RGBT/AIC 审计、manifest、mask、split 隔离 | 待团队填写 |
| 训练工程 | 云端环境、checkpoint、Teacher Bank、断点续训 | 待团队填写 |
| 验证分析 | 检索、有效秩、漂移、Rescue/Harm 与报告 | 待团队填写 |
| 文档与提交 | 技术报告、GitHub 索引、平台提交审计 | 待团队填写 |

Codex 作为工程辅助工具参与代码实现、审计、验证和文档整理，不替代参赛团队成员署名。

---

# 十四、复现资产与关键指纹

| 资产 | 标识 |
|---|---|
| Qwen3-VL-8B revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| Phase 1 Adapter SHA-256 | `F8B81349886702497ABC936BCAF1DB4660CF7B3290B2569AE80C303CE362DF7B` |
| C1 Adapter SHA-256 | `0D56149083B9EFCFA30DDBB30904A70D631EBA5E080EA2DCD63F123BEFAA85EE` |
| C2 Adapter SHA-256 | `2832225394CD884D5BA972D9BCD0E19CE8AF8715657DC9675E60AB75B103E786` |
| Teacher Bank fingerprint | `79CF814B64EC640AD3D924342FEEED235764F139445F834B2CDB2DD0B580C05C` |
| repair-dev SHA-256 | `B9DFB8C4E47CAD542326BD25D5B2054EA7CADD64E6AA12BA72E2E7B84CD5D1D7` |
| Phase 1.6 恢复归档 SHA-256 | `3AEAF519809D9FE7BE2818F7C36787667A10444422465FE4A98B8E8CBD37DECE` |
| Phase 1.8R selected Adapter SHA-256 | `03D2C3D595495162DFCF2F0044CF06CE9B05AE2B52C1186B9A1434F64A5CED06` |
| Phase 1.8R Stage A ZIP SHA-256 | `8E147726D4C5ED0263B95D3BBF2F3EA0ADF189E91511FDAD7E0B3793207466FD` |
| Phase 1.8R Final ZIP SHA-256 | `9594FBEC53462A6DBEC28006111432B7E4882FB66FF5DFE6D494DAA60D240E1C` |
| Phase 1.8R/D2 Teacher Bank fingerprint | `D8F01B557D9FAE020767A27DE7DB4602391ACFE5B27141013C3D4D22A2D677E1` |
| D2 probe multi-query dev SHA-256 | `A54490B10BF9FC1BFB455C8052F883422D65038603E72CF650749E41D4DF363D` |
| Phase 1.9-D2 Full Stage A ZIP SHA-256 | `37A1EE5AB2EC52630478269299F32652B9C983750F7D3A5EFD2862A0E133BD26` |

大数据、模型权重、Teacher Bank 和 `outputs/` 不上传 GitHub；GitHub 保存代码、配置模板、指标、哈希和报告。

---

# 十五、解题参考

## 核心资料

1. RGBT-GroundBench / RGBT-VGNet 论文：<https://arxiv.org/abs/2512.24561>
2. 官方代码：<https://github.com/crazyxiaoxi/RGBT-GroundBench>
3. 官方数据集：<https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset>
4. Qwen3-VL 官方模型：<https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>

## 后续按需研究，不在当前轮同时实现

- IV-tuning：低秩适配下的特征空间保护；
- M²D-LIF：Fusion Degradation 与单模态蒸馏；
- UniRGB-IR：冻结 RGB foundation model 的补充特征注入；
- RSDet：先去除无效红外信息再选择融合；
- C²Former / Cascade Alignment-Guided Transformer：弱配准特征校准；
- 多光谱基础模型：长期的专业 TIR 初始化方向。

---

# 十六、当前唯一结论（2026-08-23）

D2 之后的实验已经把矛盾从“最低层有效秩能否过 0.85”进一步定位为：

> **TIR 残差包含局部可读信息，但现有全图质量 proxy 和 Query gate 都不能稳定判断这个残差对某个 Query、某个候选究竟是 Rescue 还是 Harm。**

关键证据链为：

1. D3_SP020 改善 rank，但材料性增益不足且主要集中于 Layer 24；
2. G1R 的 D3_SP000 冻结特征在锁定 confirmation 上相对 Base TIR 获得 R@1 `+0.04981`，证明信息并非为空；
3. G2 静态 Layer16 residual 只通过 semantic dev，multi-query dev 失败；
4. QueryControlV2 虽提高正确 Query 的 R@1，但 wrong Query 略好，不能归因 Query 控制；
5. quality/registration learned gate 与 constant gate 的 R@1 完全相同，且 gate 大小与真实 residual 收益负相关；
6. 更强 ExtraTrees 上限探针 OOF Spearman 为 `0.22666`，但 beneficial-vs-harmful AUC 仅 `0.59024`，不足以做安全选择。

当前唯一下一步是本地、只读的 `candidate/ROI-level Rescue-Harm predictability upper bound`：把控制粒度从整张图像对下沉到 Query×ROI／候选，比较全图 proxy、局部质量、表征冲突和 Query 交互的增量价值。只有该上限通过预注册的 pair/sequence-disjoint 门禁，才允许实现最小 Layer16 local gate；如果仍接近随机，应停止 hidden-residual 路线，转向 TIR proposal + RGB ranking 的可审计 late-rescue 架构。

完整的 D3 至当前证据、数值和下一阶段合同见：

`reports/AIC_RGB_TIR_FULL_ROUTE_OPTIMIZATION_VALIDATION_AND_NEXT_PLAN_2026_08_23.md`

---

# 十七、更新日志

## v3.0 — 2026-08-23

- 补充 D3/D3A 的 same-pair structure、严格无效 ROI 排除和层间响应审计；
- 补充 Frozen-8B 四臂任务价值 NO_GO 与六臂输入协议混杂；
- 补充 G0 sidebranch interface trace 的零残差和非法 TIR 硬旁路 GO；
- 补充 G1/G1R 共享冻结 head 的 dev/confirmation GO，并明确它不是端到端 bbox GO；
- 补充 G2 Layer16-only residual 双-dev NO_GO、BF16 checkpoint 哈希修复和 scale 量化平台诊断；
- 补充 Query gate / QueryControlV2 的 wrong-Query 反证；
- 补充 quality/registration gate、learned-vs-constant 自适应归因和 12 维 proxy 上限测试；
- 将当前路线从继续 D2/D3 谱修复改为 candidate/ROI-level Rescue-Harm 可预测性上限；
- 保持 no selected Adapter、no AIC gain claim、official-val/confirmation 封存与 30B/Depth/D4 阻断边界。

## v2.0 — 2026-08-17

- 补充 Phase 1.8 D1 retention、Phase 1.8R full/official-val 和本地 Audit；
- 补充 Phase 1.9-D2 G010/G025/G050 probe 与 D2_G025 full 双 dev NO-GO；
- 明确 D2 Full 唯一阻塞为 multi-query 绝对有效秩，最好值 `0.840168 < 0.85`；
- 记录 Stage A/Stage B 云端资产边界、持久 checkpoint 与本地哈希回传流程；
- 将下一步从已完成的 D1 retention 更新为 D2 checkpoint 逐层谱审计和最小层选择性修复；
- 保持 official-val、Query、融合、平台 ACC 和 30B/Depth 的结论边界。

## v1.0 — 2026-08-14

- 首次建立持续更新主报告；
- 汇总算法提出、数据审计、Phase 0/1/1.5/1.6/1.7A+ 全过程；
- 记录低秩坍缩、真实逐层漂移、云端资产丢失与报告一致性问题；
- 固化当前唯一后续路线 `D1_RETENTION_PROBE`；
- 明确平台事实、外部验证事实、代理指标和未验证假设的边界。
