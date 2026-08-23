# AIC RGB–TIR 红外模块完整优化与验证路线（截至 2026-08-23）

> 文档用途：提供给 ChatGPT 网页端、项目成员和后续技术报告撰写使用的自包含审计材料。
> 项目：2026 AIC 算法挑战赛赛题一「基于大模型的多模态视觉理解与推理」。
> 当前主干模型：Qwen3-VL-8B-Instruct，固定 revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。
> 当前科研结论：**红外旁路在局部冻结特征检索中存在可用信号，但尚未证明能够稳定提升端到端 AIC grounding；Query gate 和现有 12 维图像对质量 gate 均未学会可靠的逐样本选择。**
> 当前发布状态：没有可发布的 RGB–TIR Adapter；正式安全基线仍是 RGB-only。

---

## 0. 给网页端的最短摘要

这条路线不是简单地“把红外图像作为第二张图输入 Qwen”，而是一直遵循一条保守原则：冻结已经验证的 RGB 主路径，只把 TIR 当作可拒绝的补充残差，在 Layer 8/16/24 少数层注入，并要求缺失、异常或无价值 TIR 时严格退化为同一模型的 RGB-only。

截至 2026-08-23，项目依次解决了五类问题：

1. **工程接口问题**：RGB/TIR 同步几何、有效视场 mask、零初始化等价性、非法 TIR 硬旁路均已建立；
2. **表征坍缩问题**：Phase 1.5 发现强检索表象下的低秩坍缩，Phase 1.6 用跨图 InfoNCE 和关系蒸馏显著修复；
3. **谱结构与层间失衡问题**：D1/D2/D3 逐步提高 rank/Base，但 D3 的收益主要集中在 Layer 24，浅层材料性改善不足且深层 nonpaired 风险升高；
4. **任务可用性问题**：G1R 证明 D3_SP000 冻结特征可被共享 Query-to-ROI head 读取并在 confirmation 复现，但 G2 的静态 Layer16 残差、Query-conditioned gate 和 Frozen-8B bbox 对照都未形成跨协议稳定收益；
5. **选择器问题**：现有 Query gate 对正确 Query 不优于错误 Query；现有 12 维全图质量/粗配准 gate 不优于常数 gate，且与真实 residual 收益负相关。更强的 ExtraTrees 上限探针虽有弱连续相关（Spearman `0.22666`），但 beneficial-vs-harmful AUC 仅 `0.59024`，不能可靠决定何时开红外。

因此，当前最合理的结论不是“红外完全无效”，也不是“再训练更久就会有效”，而是：

> **当前红外残差具有局部可读信息，但全图级、Query级的低容量控制信号都不足以稳定区分 Rescue 与 Harm。下一步必须先验证 ROI／候选级可靠性信息的可预测上限；若上限仍接近随机，应停止本残差路线，而不是继续调 loss、阈值或训练时长。**

---

## 1. 结论边界：已经证明、尚未证明和明确失败

### 1.1 已经证明

- RGB/TIR processor、坐标、`grid_thw`、TIR 有效视场 mask 与 Layer 8/16/24 hook 可以稳定工作；
- `gate=0`、`TIR=None`、非有限值 TIR、空 mask 可以严格回到 RGB 路径；
- TIR rank-48 Adapter 能学习非零表征，InfoNCE 和关系蒸馏能显著缓解低秩坍缩；
- D1 retention、D2 Base-TIR geometry、D3 same-pair structure 都能在不同程度上改善谱结构；
- D3_SP000 冻结特征在 G1R K=8 共享 head 中通过 dev 选择，并在一次锁定 confirmation 上相对 Base TIR 取得 `R@1 +0.04981`，bootstrap 95% CI `[+0.01259,+0.07075]`；
- 固定 D3/G2 Layer16 residual 在部分内部协议上存在正向任务信号；
- 现有 Query gate 和 12 维 quality/registration gate 的自适应性不足，失败是可重复并由负对照支持的。

### 1.2 尚未证明

- 尚未证明 RGB–TIR 在 AIC 赛题一官方平台上超过 RGB-only `0.7582`；
- 尚未证明当前 Adapter 可在端到端 bbox 生成中稳定利用正确 TIR；
- 尚未证明收益能跨数据源、跨 sequence、跨多 Query 复杂度稳定泛化；
- 尚未证明 Query、质量、配准或组合 gate 能可靠选择红外；
- 尚未形成正式 selected Adapter，也没有迁移到 30B；
- AIC Track-2 同风格数据只用于严格 ROI 审计与 Frozen-8B 任务价值诊断，不等价于赛题一搬移性验证，更不是排行榜成绩。

### 1.3 已明确失败或暂停

- 不能用“R@1/R@5 很高”替代有效秩、nonpaired P95、margin 和端到端任务门禁；
- 不能把 D3_SP020 绝对 rank 过线追认为 GO，因为其最差层相对控制增益仅 `+0.00456 < +0.01`；
- 不能继续统一提高同图结构损失：Layer 24 吸收大部分收益并同时放大 nonpaired 风险；
- 不能直接使用双图输入结果归因红外语义：六臂控制发现 RGB+null 本身足以改变 Frozen-8B 输出；
- 不能把 G1R 的冻结特征 retrieval GO 写成端到端 bbox GO；
- 不能把 QueryControlV2 的 R@1 提升归因于 Query 条件，因为错误 Query 反而略好；
- 不能继续优化当前 12 维全图质量 proxy：其 harm/benefit 分类上限接近随机；
- 在新的可预测上限出现前，不进入 D4、不打开 official-val、不做 AIC test、不加入 Depth、不迁移 30B。

---

## 2. 初始架构与不可破坏的安全约束

### 2.1 核心结构

```text
Visible RGB ──> frozen Qwen vision ───────────────────────────> RGB representation
Infrared TIR ─> shared Qwen vision + TIR rank-48 LoRA ───────> TIR residual source
                                      │
                                      └─ Layer 8 / 16 / 24 lightweight projection

F_l = R_l + tanh(alpha_l) * g_l * A_l(M_l * T_l)
```

- `R_l`：冻结 RGB 层特征；
- `T_l`：TIR 层特征；
- `M_l`：从原始红外有效视场下采样得到的 mask；
- `A_l`：轻量 projector；
- `alpha_l`：零初始化 residual scale；
- `g_l`：后续待验证的选择 gate。

采用 `tanh(alpha)` 而不是 `sigmoid(alpha)`，因为 `sigmoid(0)=0.5` 会在初始化时污染 RGB 控制；`tanh(0)=0` 才是真正的 RGB 等价起点。

### 2.2 训练与发布不变量

- Qwen RGB vision、LLM 和原生 bbox 路径冻结；
- TIR Adapter 只作用于视觉 attention `qkv/proj`；
- RGB 与 TIR 共用 resize/crop/pad/坐标映射和 `grid_thw`；
- 近黑边框必须转为显式有效视场 mask；
- 缺失/非有限/空 mask TIR 必须硬旁路；
- 不允许第二模型 fallback 掩盖本模型失败；
- 每次实验只改变一个科研变量，并固定模型 revision、数据切分、seed、门禁和 checkpoint 口径；
- dev 选型、confirmation 复核、official-val 和 AIC test 严格分离；
- NO_GO 是科研结论，不等于运行故障；
- 权重、Teacher Bank、特征 cache 和 outputs 不进 Git，只保存代码、配置模板、指标、哈希和报告。

---

## 3. 数据、切分与证据等级

### 3.1 数据资产

| 数据/口径 | 用途 | 能否支持正式收益声明 |
|---|---|---|
| RGBT-GroundBench head_train | Adapter/head/gate 训练与内部 holdout | 不能单独支持泛化声明 |
| semantic_dev | 单 Query 语义 dev | 与 multiquery_dev 同时过门才可继续 |
| multiquery_dev | 同图多 Query 干扰 dev | 与 semantic_dev 同时过门才可继续 |
| image-pair/sequence-disjoint confirmation | 锁定候选后一次确认 | 可支持外部确认，但仍非 AIC 平台 |
| sealed official-val | 双 dev GO 后一次打开 | 未满足前保持封存 |
| AIC Track-2 同风格数据 | ROI、输入协议和 Frozen-8B 诊断 | 不是赛题一搬移性证明 |
| AIC 赛题一平台 | 最终 bbox ACC@0.5 | 当前未做 RGB–TIR 正式提交 |

### 3.2 统计口径

- 以 image pair 为聚类单位做 bootstrap，避免同一图像的多条 Query 被当作独立样本；
- semantic 和 multi-query 分开报告 R@1、R@5、MRR；
- 主要子群包含 low-light、small、occlusion、normal；
- 负对照包括 shuffled Query、wrong Query、random TIR、misaligned TIR、shuffled quality；
- confirmation 不参与候选选择；
- 任何 exploratory subgroup、数据源差异和特征重要性只能用于定位，不可事后改门禁。

### 3.3 四类证据必须分开

1. **AIC 平台事实**：官方 ACC@0.5；
2. **外部机器指标**：带 GT 数据上的 R@K、MRR、IoU、ACC；
3. **表征诊断**：rank/Base、奇异值谱、cosine、P95、margin；
4. **假设与代理**：质量、配准、Query gate 分数、可预测性上限。

本路线此前多次 NO_GO 的价值，正是阻止第 3/4 类信号被误写为第 1 类结论。

---

## 4. 全路线总表

| 阶段 | 核心问题/唯一变量 | 结果 | 对下一步的真实影响 |
|---|---|---|---|
| Phase 0 | processor、mask、坐标、hook、gate=0 | GO | 工程接口成立，RGB 安全基线可保护 |
| Phase 1 | TIR rank-48 配对对齐 | 完成 | TIR 可学习，但只对齐不足以证明判别性 |
| Phase 1.5 | 全量 retrieval + collapse audit | NO_GO | R@K 提升同时出现 Layer 8/16/24 低秩坍缩 |
| Phase 1.6 | C0/C1/C2：InfoNCE + relational distillation | NO_GO | C2 最佳，去坍缩成功但中深层漂移仍大 |
| Phase 1.7A+ | 逐层漂移、质量与假负例归因 | COMPLETE | 质量/假负例弱相关，不能解释全部问题 |
| Phase 1.8/1.8R | D1 Base-relative retention | FULL NO_GO | 检索强、漂移降，但 official-val 三层 rank/Base 仍不足 |
| Phase 1.9-D2 probe | geometry λ=0.10/0.25/0.50 | G025 selected | G025 是秩修复与 P95 风险的折中 |
| Phase 1.9-D2 Full | G025 full + 双 dev | NO_GO | multi-query 最好 `0.840168 < 0.85`；official-val 封存 |
| D2 Layer Audit | 四 checkpoint 逐层只读审计 | COMPLETE | 三层轻度压缩，Layer 8 最低，不是单层灾难 |
| Phase 1.9-D3 | same-pair λ=0/.05/.10/.20 | NO_GO | SP020 绝对过线但材料性增益不足 |
| D3A Strict Audit | 严格无效 ROI 排除、层轨迹、AIC 同风格审计 | COMPLETE | 收益集中 L24；L8 平台化；统一增权停止 |
| Frozen-8B 四臂 | RGB/correct/random/misaligned TIR bbox | NO_GO | 正确 TIR 未稳定优于 RGB 或负对照 |
| Frozen-8B 六臂 | Native/null/correct + A/B 重复 | NO_GO / repair interface | null 双图已改变输出，旧双图因果比较失效 |
| G0 Interface Trace | Native RGB、alpha=0、TIR=None、nonfinite | GO | 旁路本身可做到严格等价，工程 seam 修复成立 |
| G1/G1R | 六历史臂冻结 Query-to-ROI 共享 head | DEV GO + confirmation GO | D3_SP000 有可读 task signal，但仅是冻结特征 probe |
| G2 Layer16 | 只训练 Layer16 projector + scale | DEV NO_GO | semantic 正、multi-query 不过；静态 residual 泛化不稳 |
| Query Gate | Query-conditioned Layer16 late gate | DEV NO_GO | R@1 可升，但 correct-vs-wrong Query 门禁失败 |
| QueryControlV2 | 更直接的 cosine Query interaction | 双-dev NO_GO | wrong Query 略优；停止 Query-conditioned residual |
| Quality/Registration Gate | 12 维 query-free gate | head_train NO_GO | residual 有静态信号，gate 无自适应价值 |
| Gate Adaptivity Audit | learned vs constant vs full residual | SCALE_ONLY_SIGNAL | learned=constant，gate 与真实收益负相关 |
| 12D Proxy Upper Bound | pair-disjoint ExtraTrees/Ridge | STOP | 弱连续信号存在，benefit/harm 判别近随机 |

---

## 5. Phase 0–1.8R：从工程成立到去坍缩

### 5.1 Phase 0：先保护 RGB

Phase 0 不追求提升，先验证同图配对、同步几何、bbox 映射、black-border mask、Qwen normalization、Layer hooks 和零残差等价性。该阶段 GO 的意义是：后续任何退化都能归因于红外分支或训练，而不是 processor 偷换坐标、图像尺寸或原生 RGB 路径。

### 5.2 Phase 1 与 1.5：为什么“检索强”仍判 NO_GO

初始 paired alignment 使同对 RGB/TIR cosine 和 R@1/R@5 上升，但 Phase 1.5 全量验证发现 Layer 8/16/24 的 adapted effective rank 大幅下降。也就是说，许多 TIR 实例被压进少数方向，表面上容易匹配 RGB Teacher，却失去区分不同目标的能力。

该结论改变了路线：从“继续提高 paired cosine”转向“同时约束跨实例可分性与谱结构”。

### 5.3 Phase 1.6：InfoNCE 是去坍缩关键

| 候选 | 新增约束 | 平均 R@5 | 平均 R@1 | 结论 |
|---|---|---:|---:|---|
| C0 | paired alignment + background margin | 69.50% | 48.93% | 控制 |
| C1 | C0 + 跨图 RGB 负例 InfoNCE | 82.39% | 57.65% | 明显去坍缩 |
| C2 | C1 + relational distillation | **83.56%** | **60.03%** | 最佳基座，仍未过 alignment |

C1 说明跨图负例是恢复实例区分度的主要力量；C2 进一步保留 Teacher 关系结构。但 C2 在中层发生真实漂移，Layer 24 最严重，所以没有直接发布。

### 5.4 Phase 1.7A+：没有把弱代理当根因

项目审计 RGB 质量、遮挡、配准和假负例代理。它们与漂移方向一致但效应弱，无法解释全部失败。因此没有贸然引入复杂质量网络或 false-negative-aware 分支，避免用不可靠代理扩大模型复杂度。

### 5.5 Phase 1.8/1.8R：D1 retention 与持久化闭环

D1 在 C2 上加入 Base-relative retention，单变量 λ 选择 `D1_L050`。full train 和 2,032 条 sealed official-val 均完成；平均 R@5 达 `85.81%`，最低 paired-shuffled margin `0.18267`，但三个判别层 D1/Base rank 均低于 `0.85`，总 alignment loss 从 `0.16077` 上升到 `0.25878`，最终仍为 NO_GO。

该阶段还修复了一次重要工程事故：初次长链路在 official-val 中断，边界资产未独立归档并随实例释放。此后固定为：

```text
Stage A = Teacher Bank + 四 checkpoint + selected decision + 独立归档
Stage B = official-val/cache parts + final archive
```

checkpoint 必须写真实云盘目录，不使用 `/dev/shm` 临时链接；每个阶段保存 PID、failure state、receipt、SHA256 manifest。这一经验后来保护了 D2、D3、G1 和 G2 资产。

---

## 6. D2–D3A：谱几何修复为何没有直接变成任务收益

### 6.1 D2：Base-TIR geometry 有效但 full 泛化差一点

D2 从同一 D1_L050 起点增加 Base-TIR 邻域几何保持。probe 中 G025 的 multi-query/semantic rank 增益约 `+3.52pp/+5.00pp`，且 P95 风险仍在门内；G050 虽更强，但 nonpaired P95 超过 `0.02`，因此只允许 G025 full train。

Full 的 50%/75%/100% checkpoint 均通过相对秩增益、R@5、P95 和正 margin，唯一贯穿失败是 multi-query 绝对 rank。最好 `0.840168`，距离 `0.85` 仍差 `0.009832`，50% 后趋于饱和。没有 selected Adapter，official-val 未打开。

### 6.2 D2 Layer Audit：不是单一坏层

50% checkpoint 的 multi-query rank/Base：

```text
Layer 8  = 0.840170
Layer 16 = 0.840931
Layer 24 = 0.847649
```

三层同时轻度压缩、L8 最低，根因不是某一层彻底崩坏。这推动 D3 测试同图关系保持。

### 6.3 D3：SP020 方向正确但材料性不足

| 候选 | Multi-query L8/L16/L24 rank/Base | 最低增益 vs SP000 | 结论 |
|---|---|---:|---|
| SP005 | 0.84865 / 0.84881 / 0.86775 | +0.00263 | NO_GO |
| SP010 | 0.84980 / 0.85139 / 0.87372 | +0.00378 | NO_GO |
| SP020 | **0.85057 / 0.85486 / 0.88251** | **+0.00456** | NO_GO |

SP020 是唯一三层绝对 rank 都过 0.85 的候选，R@5 没有下降；但预注册要求最差层相对 SP000 至少 `+0.01`，实际只有 `+0.00456`。这是一轮“方向有效、幅度不够”的合法 NO_GO。

### 6.4 D3A：严格 ROI 排除后，深层主导仍存在

D3A 修复无效 IR ROI 混入：AIC 同风格严格集排除 38 条没有任何有效 IR ROI token 的记录，保留 `1,536/1,574`（`97.5858%`）。这只修复数据边界，不改模型结论。

SP020−SP000 在 2048 step 的 multi-query rank 增量：

| Layer 8 | Layer 16 | Layer 24 |
|---:|---:|---:|
| +0.002708 | +0.008846 | **+0.020988** |

L8 从 1536→2048 仅增加 `0.00003159`，已经平台化；L24 的 paired cosine 和 rank 响应最大，同时 AIC 同风格域 nonpaired P95 增加 `+0.029216`、margin 下降。于是“继续延长训练”与“全层统一提高结构权重”都被否决。

---

## 7. Frozen-8B 与 G0：先修因果比较接口

### 7.1 四臂任务价值对照

冻结 8B 对 295 条 AIC-style GT 记录运行 RGB-only、Correct TIR、Random TIR、Misaligned TIR，无训练：

| Arm | ACC@0.5 | mean IoU |
|---|---:|---:|
| RGB-only | **0.72203** | **0.61843** |
| Correct TIR | 0.71864 | 0.61645 |
| Random TIR | 0.71186 | 0.60664 |
| Misaligned TIR | 0.71186 | 0.61036 |

image-pair cluster bootstrap 的 Correct−RGB 点估计 `+0.00942`，CI `[-0.00546,+0.02976]`；Correct−Misaligned 仅 `+0.00268`。正确 TIR 有 3 次 Rescue、4 次 Harm，无法稳定胜过 RGB。因此不能据此开启 D4。

### 7.2 六臂协议控制发现输入结构混杂

104 条、100 pairs、6 arms、624 次冻结生成中，A/B 重复完全一致，但：

- Native RGB ACC `0.701923`；
- RGB+null ACC `0.730769`；
- Correct TIR ACC `0.721154`；
- Native/null ACC flip `3% > 1%` 门禁；
- 双图使 visual tokens 从约 2,040 翻倍到约 4,080。

说明“双图输入结构”本身改变模型输出，Correct TIR 相对 Native RGB 的变化不能归因于红外语义。该结果修正了此前四臂解释，而不是证明红外无效。

### 7.3 G0 interface trace 修复 seam

新的 sidebranch 接口不再把 TIR 作为第二张主输入，而是在固定 Layer16 旁路注入。16 个独立 pairs 的 trace 结果：

- `alpha_zero_all_exact=true`；
- `invalid_tir_hard_bypass_exact=true`；
- active fusion layer 仅 16；
- 无 Adapter 训练；
- 状态 `PHASE19_INTERFACE_TRACE_GO`。

G0 证明工程上可以构造“唯一变化是红外残差”的因果接口，之后才有资格做 G1/G2。

---

## 8. G1/G1R：冻结特征里确实有可读任务信息

G1 云端只提取冻结 Query 与 Layer 8/16/24 ROI 特征；本地用一个共享 K=8 head 比较 RGB Teacher、Base TIR、D1_L050、D2_G025 50%、D3_SP000、D3_SP020。所有 Adapter 与 backbone 冻结，不能为每臂单独调 head。

### 8.1 dev 结果

| Split | Base TIR R@1 | D3_SP000 R@1 | D3_SP020 R@1 | RGB Teacher R@1 |
|---|---:|---:|---:|---:|
| semantic_dev | 0.12760 | **0.16016** | 0.16016 | 0.11068 |
| multiquery_dev | 0.14488 | **0.19554** | 0.19250 | 0.21074 |

`d3_sp000` 与 `d3_sp020` 均合格，预注册选择 `d3_sp000`，状态 `G1R_DEV_GO`。

### 8.2 一次性 confirmation

| Arm | R@1 | R@5 | MRR |
|---|---:|---:|---:|
| Base TIR | 0.17548 | 0.73946 | 0.42187 |
| D3_SP000 | **0.22529** | 0.75019 | **0.44885** |
| RGB Teacher | 0.21379 | **0.75862** | 0.44809 |

D3_SP000 相对 Base TIR：

- R@1 `+0.04981`；
- MRR `+0.02698`；
- cluster bootstrap R@1 CI `[+0.01259,+0.07075]`；
- 最差主要子群仍为 `+0.02335`；
- 单 Query 与 multi-query 方向均为正。

状态为 `G1R_CONFIRMATION_GO`。这是整条路线最强的积极证据，但它只证明“冻结特征可被共享 head 读取”，并不证明注入 Qwen 后的端到端 bbox 增益。

---

## 9. G2：静态 Layer16 residual 的双-dev 分裂

G2 固定 D3_SP000，只训练 Layer16 `project.weight` 和 `fusion_scale`，2048 steps，四 checkpoint；其余 Qwen、LLM、Adapter、共享 heads、Layer8/24 全冻结。

### 9.1 工程异常与修复

v3 在 step 511 保存 checkpoint 时，BF16 tensor 直接转 NumPy 做哈希触发 `TypeError`。v4 仅把哈希改为 `uint8` 原始字节视图；模型、数据、loss、fingerprint 和 checkpoint 口径不变。恢复后四个 checkpoint 均可读。

数值审计发现 `fusion_scale` 以 BF16 保存，在 1536→2048 平台为 `tanh(0.03125)=0.0312398`，但 projector 仍继续更新。该问题保留为后续 FP32 scale 工程修复，不能改变历史 G2 结论。

### 9.2 双 dev

| Split | RGB R@1 | D3_SP000 R@1 | G2 R@1 | G2−RGB | 结论 |
|---|---:|---:|---:|---:|---|
| semantic_dev | 0.11068 | 0.16016 | 0.12760 | +0.01693 | 单 split 通过 |
| multiquery_dev | 0.21074 | 0.19554 | 0.20871 | -0.00203 | 失败 |

multi-query bootstrap CI `[-0.01697,+0.01428]`，且 normal 子群 `-0.01587`。最终 `G2_DEV_NO_GO`，confirmation 保持封存。

解释：Layer16 residual 可以把 D3 特征拉近 RGB Teacher，但在多 Query 干扰下不能稳定超过 RGB。此时问题从“表征是否可读”转为“何时、对谁注入”。

---

## 10. Query gate：会响应 Query，但没有正确因果控制

### 10.1 初始 Query gate

项目先补齐缺失的 `head_train/d3_sp000.pt`：845 records、512 image pairs、4,225 ROI；禁止用 dev/confirmation 代替 train。Query-conditioned Layer16 gate 只训练 FP32 gate/residual-scale，RGB、D3 Adapter、G2 projector 和 heads 均冻结。

初始本地 dev 虽产生正向 R@1，但 correct-vs-wrong Query gap 在 multi-query 为 `-0.00203`，semantic 仅 `+0.00391`，未达到 Query-control 门禁，状态 `QUERY_GATE_LOCAL_DEV_NO_GO`。

### 10.2 QueryControlV2

更直接的 cosine interaction 在一次性双 dev 中：

| Split | RGB R@1 | V2 R@1 | ΔR@1 | bootstrap lower | Correct−wrong R@1 |
|---|---:|---:|---:|---:|---:|
| semantic_dev | 0.11068 | 0.14063 | +0.02995 | +0.00651 | **-0.00781** |
| multiquery_dev | 0.21074 | 0.22391 | +0.01317 | -0.00582 | **-0.00101** |

Gate 表征对 Query 发生变化，query swap rate 约 30%，但错误 Query 的排序略优于正确 Query。说明它学到的是通用分数缩放或数据相关模式，不是 Query-conditioned 红外选择。状态 `QUERY_CONTROL_V2_DUAL_DEV_NO_GO`，路线明确停止继续搜索 Query gate 交互函数。

---

## 11. Quality/Registration gate：静态信号和自适应能力的分离

### 11.1 12 维低容量 gate

该 gate 不接收 Query，只接收透明的 RGB/TIR 质量与粗配准 proxy，在 head_train 内部 pair-disjoint holdout 上比较 RGB、correct quality、shuffled quality、controlled misalignment。

| Arm | R@1 | R@5 | MRR |
|---|---:|---:|---:|
| RGB | 0.38627 | 1.00000 | 0.65127 |
| Correct quality | 0.43725 | 1.00000 | 0.67742 |
| Shuffled quality | 0.43922 | 1.00000 | 0.67840 |
| Misaligned | 0.42941 | 1.00000 | 0.67284 |

- Correct−RGB R@1 `+0.05098`，CI 下界为正；
- Correct−shuffled `-0.00196`，失败；
- Correct−misaligned `+0.00784 < +0.01`，失败；
- 未生成 selection lock，dev/confirmation 保持封存。

这说明低幅度静态 residual 有信号，但 12 维 gate 没有学会利用 quality/registration 变化。

### 11.2 learned gate vs constant gate

- learned R@1 `0.43725`，constant R@1 `0.43725`；
- learned MRR `0.67742`，constant MRR `0.67775`；
- 510 行中只有 5 行排序不同；
- gate 与 full-residual 真实 MRR 收益 Spearman `-0.21727`，CI `[-0.3830,-0.0208]`；
- 对 R@1 收益 Spearman `-0.25288`。

因此诊断为 `SCALE_ONLY_SIGNAL`：收益来自残差被以约 0.089 的总强度打开，而非逐 pair 自适应。

### 11.3 12 维 proxy 上限

在 512 pairs、845 queries、2,535 outcome rows 上，以 `full-gate MRR - RGB MRR` 为标签，严格 pair-disjoint OOF：

| 探针 | Spearman | 95% CI | beneficial-vs-harmful AUC | 95% CI | R² |
|---|---:|---|---:|---|---:|
| ExtraTrees | **0.22666** | [0.13510,0.31343] | **0.59024** | [0.47470,0.70325] | 0.01978 |
| Ridge | 0.12199 | [0.03058,0.21338] | 0.56287 | [0.45621,0.67130] | — |

标签置换 p=`0.004975`，说明弱连续关联不是纯随机；但 AUC 未过 0.60 且 CI 跨 0.5，只解释约 2% 方差，无法可靠区分有益/有害。正式路线为 `STOP_CURRENT_12D_QUALITY_PROXY_ROUTE`。

---

## 12. 所有结果合起来到底说明什么

### 12.1 一个统一解释

1. **TIR representation 不是空的**：D1/D2/D3 的谱改善与 G1R confirmation 都支持这一点；
2. **可用信息是局部且条件性的**：Frozen-8B 有少量 Rescue 也有 Harm，G2 semantic/multi-query 分裂；
3. **当前控制变量粒度不对**：全图 12D quality proxy 无法描述某个 Query 下某个 ROI 的 residual 是帮助还是冲突；
4. **Query gate 也尚未找到正确监督接口**：gate 会变化，但 wrong Query 不更差，说明目标不足以迫使它学习因果 Query 关系；
5. **继续加大训练只会放大旧偏差**：D3A 已见 L8 平台化与 L24 风险，G2 scale 也出现量化平台；
6. **真正缺失的是可靠的局部选择信号，而不是又一个全局 loss 权重。**

### 12.2 当前成熟度

| 维度 | 成熟度 | 说明 |
|---|---|---|
| 数据/processor/坐标/mask | 高 | 已多轮审计并有硬旁路 |
| 表征训练与防坍缩 | 中高 | 有明确正信号和完整资产 |
| 多层谱结构 | 中 | 可改善但层间响应不均衡 |
| 冻结特征任务可读性 | 中 | G1R confirmation GO |
| 端到端 bbox 增益 | 低 | 四臂未过，且旧双图接口有混杂 |
| Query-conditioned 选择 | 低 | correct-wrong Query 失败 |
| quality/registration 自适应 | 低 | learned≈constant，12D 上限不足 |
| AIC 平台搬移 | 未验证 | 没有正式 RGB–TIR 提交 |

---

## 13. 后续优化路线：先做上限，再决定是否继续本分支

### 13.1 下一阶段唯一主问题

> 将控制粒度从“整张 RGB/TIR 图像对”下沉到“Query × ROI／候选”，能否在严格 image-pair/sequence-disjoint 条件下预测 Layer16 residual 对候选排序的 Rescue 或 Harm？

这一步先做本地只读上限诊断，不需要 4090，也不训练 Qwen/Adapter。

### 13.2 候选级可靠性上限的预注册设计

固定已有 RGB/D3_SP000/G2 冻结特征和共享 K=8 head，不再改历史模型。以每个 `Query × candidate ROI` 为基本行，image pair 为分组单位。

#### 标签

- residual 开启相对 RGB 的目标候选 margin 变化；
- best-rank 改善/恶化；
- R@1 Rescue、R@1 Harm、Neutral；
- 对 hardest negative 的 logit gap 变化；
- pair 级最终 MRR/R@1 只作聚合验证。

#### 嵌套特征组

| 组 | 特征 | 回答的问题 |
|---|---|---|
| A | 现有 12D pair-global proxy | 固定旧基线 |
| B | ROI-local 亮度、对比、熵、边缘、有效 FOV、局部配准 | 局部质量是否比全图质量更有用 |
| C | RGB/TIR cosine、residual norm、projected residual 与 RGB 冲突、candidate-local geometry | 表征冲突能否预测 Harm |
| D | Query–RGB、Query–TIR、Query–residual interaction | Query 信息是否在局部候选层才变得可控 |

必须做 A→B→C→D 的增量比较；D 必须同时通过 shuffled/wrong Query 负对照，不能只看总 R@1。

#### 统计合同

- outer folds 按 image pair/sequence 分组；
- 任何 normalization、feature selection 和 classifier fit 只在训练折；
- 固定 seeds；
- 2,000 次 image-pair cluster bootstrap；
- 报告 OOF Spearman、beneficial-vs-harmful AUC、PR-AUC、Brier/校准、harm recall、coverage-risk curve；
- 数据源/low-light/small/occlusion/query count 仅作锁定 subgroup audit；
- 不在同一 dev 上循环改阈值追 GO。

建议的继续门槛：

1. OOF Spearman 至少 `0.25`，95% CI 下界 `>0.10`；
2. beneficial-vs-harmful AUC 至少 `0.65`，95% CI 下界 `>0.55`；
3. 在保留至少 60% 候选/图像对的 coverage 下，Harm 明显低于常数 gate，且 R@1/MRR 不低于常数 gate；
4. B/C 相对 A 有稳定增量；若使用 D，则 correct Query 必须稳定优于 wrong/shuffled Query；
5. 主要子群不能出现材料性反向伤害。

这些门槛应在运行前写入 protocol lock；数值可由团队一次性核定，但运行后不能降低。

### 13.3 三条分支

#### 分支 A：候选级上限明确通过

实现最小 `candidate/ROI-local reliability gate`：

- 仅 Layer16；
- zero-init；
- RGB hard bypass；
- 低容量共享 gate；
- scale 与 optimizer master state 保持 FP32；
- 不加入 Layer8/24、Depth 或第二个 fusion 模块；
- 先 head_train，锁候选后 semantic/multiquery 双 dev 一次；
- 双 dev GO 后才打开新的 image-pair/sequence-disjoint confirmation；
- confirmation GO 后才做 Frozen-8B 端到端接口复核和一次 AIC 8B 平台提交。

#### 分支 B：只有 oracle 上限高，学习器上限不高

说明任务有红外价值，但现有监督不足。此时不继续调 gate 网络，而是重构训练目标：

- 显式 Rescue/Harm ranking loss；
- hardest-negative candidate 级监督；
- 候选局部配准/遮挡标签或可靠伪标签；
- 必要时补充 AIC 同风格训练数据，但继续保持 sequence-disjoint confirmation；
- 先证明标签可靠性，再训练新 gate。

#### 分支 C：候选级上限仍接近随机

停止当前 hidden-residual 选择路线。将 TIR 改为更可审计的辅助候选生成器：

```text
TIR detector/proposal -> candidate boxes
RGB/Qwen -> Query-conditioned candidate ranking / final bbox
```

这种 late rescue 架构能明确统计 TIR 增加了哪些候选、救回了哪些漏检、伤害了哪些 RGB 正确样本。如果连 GT oracle candidate union 的上限也低，则冻结红外主线，把算力转回 RGB baseline、数据清洗或其他模态。

### 13.4 暂时明确不做

- 不继续微调当前 12D quality gate；
- 不继续搜索 QueryControl 的更多相似度函数；
- 不重新降低 D2/D3/G2 旧门禁；
- 不把 SP020 直接作为 selected Adapter；
- 不在同一 dev 反复调阈值；
- 不立即训练 D4 Layer-balanced；
- 不迁移 30B、不加入 Depth、不做多分支 fusion；
- 不把 Track-2 诊断写成 Track-1 泛化。

---

## 14. 下一轮可执行清单

### 14.1 本地阶段（现在做）

1. 冻结并哈希已有 RGB/D3_SP000/G2/head 特征与 protocol；
2. 构建 candidate-level label ledger，逐行记录 Rescue/Harm/Neutral；
3. 审计 image-pair/sequence 泄漏和候选重复；
4. 实现 A/B/C/D 嵌套上限探针与 dummy TDD；
5. 固定 thresholds、seeds、bootstrap 和 subgroup；
6. 先跑 A/B/C；仅在 C 有材料性增量后跑带 wrong-query 控制的 D；
7. 生成 decision/summary/report/manifest，不打开 confirmation。

### 14.2 何时才租 4090

只有出现以下情况之一才需要 GPU：

- 本地缺少必须由固定 8B 重提取的 candidate-local Layer16 特征；
- 候选级上限通过，需要运行最小 Layer16 gate 的端到端 trace/inference；
- 已锁定候选通过双 dev，需要提取全新 sequence-disjoint confirmation 特征。

在此之前，本地 4060/CPU 已足够做统计、上限探针、TDD、报告和协议锁。

### 14.3 最终发布门

```text
candidate-level upper bound GO
    -> minimal Layer16 local gate
    -> semantic + multiquery dual-dev GO
    -> new sequence-disjoint confirmation GO
    -> Frozen-8B causal interface GO
    -> one AIC 8B RGB-TIR platform submission
    -> only then consider 30B migration
```

任一门失败，保留资产并停止该分支，不用后续阶段补救前一阶段失败。

---

## 15. 工程可靠性经验

| 问题 | 根因 | 固化修复 |
|---|---|---|
| 临时实例释放导致成果消失 | 训练与 official-val 长链、无边界归档 | Stage A/B 独立 ZIP、receipt、manifest、本地校验后退实例 |
| `/dev/shm` checkpoint 丢失/悬空链接 | 临时内存盘不持久 | checkpoint 必须真实云盘目录 |
| 正常 NO_GO 被 shell 当故障 | exit code 2 与 ERR trap 混淆 | GO=0、NO_GO=2 但正常归档、其他码才 failure |
| 双图 null 改变 Qwen 输出 | token/位置结构改变 | sidebranch 注入；G0 alpha=0 等价门禁 |
| BF16 checkpoint 哈希失败 | BF16 tensor 直接转 NumPy | 用 uint8 原始字节视图，协议不变并 smoke |
| BF16 fusion scale 平台 | 标量分辨率不足 | 下一实现 scale/optimizer master 保持 FP32 |
| SSH/传输握手不稳 | 长传输与平台连接抖动 | 远端资产不删，续传、receipt SHA、ZIP CRC、manifest 逐项校验 |
| 无效 IR ROI 混入 | mask 后零有效 token 未排除 | 严格无效 ROI 排除并记录覆盖率 |
| dev 资产被当 train 替代风险 | 缺失 head_train feature | fail closed；先云端提取 head_train，再本地训练 |

---

## 16. GitHub 同步审计（本次更新前）

### 16.1 远端状态

- 仓库：`finalymadeout7433-gif/aic-multimodal-visual-grounding`；
- 分支：`agent/rgbtir-module-phase19-sync`；
- 本次审计时本地 HEAD 与远端分支均为 `dd657b848bce4db013499c30e991786c09fd8720`；
- 最近正式记录只到 Phase 1.9-D2 Full 与 D2 Layer Audit；
- `src/aic_rgbtir/README.md` 的 current status 仍为 `PHASE_19_D2_FULL_NO_GO`；
- Living Report v2.0 的最近更新为 2026-08-17。

### 16.2 尚未同步的阶段

以下内容在本地已有代码、测试、报告或输出，但在审计时尚未成为远端提交：

- D3/D3A；
- Frozen-8B 四臂/六臂；
- G0 interface trace；
- G1/G1R；
- G2 Layer16；
- Query gate / QueryControlV2；
- Quality/Registration gate；
- gate adaptivity diagnosis；
- 12D quality proxy upper bound。

### 16.3 本次文档补齐范围

- 新增本完整总报告；
- 更新根 README 的红外路线入口；
- 更新 `src/aic_rgbtir/README.md` 的 current status、模块图和实验序列；
- 更新 Living Report 顶部状态、当前结论与 v3.0 日志；
- 不修改权重、不复制 outputs、不覆盖未提交模型代码；
- 不自动 commit/push，避免把工作区中尚未单独审查的代码和大批未跟踪文件一起推到远端。

---

## 17. 关键本地证据入口

| 内容 | 路径 |
|---|---|
| Living Report | `reports/AIC_RGB_TIR_LIVING_TECHNICAL_REPORT.md` |
| D3A 全路线 | `reports/AIC_RGB_TIR_红外模块全路线截至Phase19_D3A_严格ROI审计与后续路线_2026_08_18.md` |
| Frozen-8B 四臂 | `reports/AIC_RGB_TIR_Phase19_Frozen8B四臂对照_完整结果分析与后续决策_2026_08_20.md` |
| Frozen-8B 六臂 | `reports/AIC_RGB_TIR_Phase19_Frozen8B六臂协议控制_完整结果分析与后续路线_2026_08_21.md` |
| G1/G1R outputs | `outputs/aic_rgbtir_phase19_g1_returned_20260821/` |
| G2 outputs | `outputs/aic_rgbtir_phase19_g2_layer16_returned_20260822/` |
| QueryControlV2 | `reports/AIC_RGB_TIR_Phase19_QueryControlV2_一次性双Dev_NO_GO与路线调整_2026_08_23.md` |
| Quality gate screen | `reports/AIC_RGB_TIR_Phase19_QualityRegistrationGate_head_train筛查结果_2026_08_23.md` |
| Gate adaptivity | `reports/AIC_RGB_TIR_Phase19_QualityGate自适应性归因诊断_2026_08_23.md` |
| 12D upper bound | `reports/AIC_RGB_TIR_Phase19_12维质量Proxy可预测性上限测试_2026_08_23.md` |

`outputs/` 是本地证据资产，不上传 GitHub；网页端主要阅读本报告和上表中的 Markdown 报告即可。

---

## 18. 建议网页端重点复核的问题

1. 是否同意将当前结论写成“局部可读信号存在，但选择器和端到端收益未成立”，而不是简单的“红外有效/无效”？
2. G1R confirmation GO 与 G2/Query/quality NO_GO 是否应解释为信息粒度和控制接口问题？
3. 候选级 A/B/C/D 嵌套上限设计是否足以区分全图质量、局部配准、表征冲突和 Query 交互？
4. 建议的 Spearman/AUC/coverage-risk 门槛是否需要在运行前调整一次？
5. 若候选级上限失败，是否同意停止 hidden residual，转向 TIR proposal + RGB ranking 的可审计 late-rescue 路线？
6. 在没有新 sequence-disjoint confirmation 和 AIC 平台提交前，是否同意继续保持 8B RGB-only 为生产基线？

---

## 19. 最终判断

红外模块已经完成了从“能不能接入”到“表征是否健康”、从“谱结构能不能修”到“任务是否可读”、再到“能否逐样本选择”的完整诊断链。投入并非没有产出：它排除了低秩坍缩、无效 ROI、双图接口混杂、错误 Query 伪控制、静态 scale 冒充自适应 gate 等多个会导致错误结论的陷阱，并保留了 G1R 的真实正证据。

但这些证据尚不足以发布模块。当前最有价值的动作不是继续扩大训练，而是做一次严格的候选级可预测上限测试。它将成为本路线的关键分水岭：

- 通过，则最小化实现 ROI-local Layer16 gate，再走双 dev、独立 confirmation、端到端和平台门；
- 不通过，则停止当前残差选择路线，转向可审计的 TIR 候选生成，或把资源退回 RGB 主线。

这比继续调 loss 更节省算力，也能给后续技术报告一个清晰、诚实、可复现的算法演进故事。
