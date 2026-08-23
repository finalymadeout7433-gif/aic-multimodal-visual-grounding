# AIC RGB–TIR 网页端指导报告审读、论文核验与下一轮决策

> AS_OF: 2026-08-13
> 项目：AIC 赛题一多模态语义目标定位
> 当前基座：Qwen3-VL-8B-Instruct
> 当前阶段：TIR Adapter 表征适配与去坍缩，尚未进入 RGB–TIR grounding 融合
> 本文结论：下一轮应执行 **Phase 1.7A+ 无训练证据审计**，不应立即训练 D1/D2，也不应立即实现 Shared/Complementary 双分支或 CoDAF。

---

## 1. 本轮审读的输入

1. 网页端报告：`C:/Users/12525/Downloads/AIC_RGB_TIR_红外模块问题诊断与下一阶段指导.md`
2. 项目交接报告：`reports/AIC_Qwen3VL_RGBT_module_full_handoff_and_next_plan_2026_08_13.md`
3. Phase 0 / 1 / 1.5 / 1.6 本地代码、manifest 与汇总结果。
4. RGBT-GroundBench / RGBT-VGNet、M2D-LIF、IV-tuning、UniRGB-IR、RSDet、C²Former、CoDAF、M-SpecGene 与 RGBX-R1 的论文或官方项目。

---

## 2. 对网页端报告的总体判断

报告的核心判断正确：

```text
不要将 TIR 单纯训练成“更像 RGB”；
应保留跨模态共享目标语义，同时保留 TIR 独有的热轮廓、低光可见性和热显著性；
最终通过 Query、IR 质量和配准可靠性选择性注入。
```

但报告把三个不同时间尺度的任务混在了一起：

1. **立即可以回答的诊断问题**：对齐负比率、RGB Teacher 可靠性、InfoNCE 假负例。
2. **需要一轮受控训练的问题**：D1 retention、质量加权 paired loss。
3. **只有前两项证明必要后才应实现的架构**：Shared/Complementary 双分支、Query-aware fusion、CoDAF。

因此不能把报告的“最终架构”当作下一轮开发清单。

---

## 3. 五个核心问题的重新评估

### 3.1 问题一：迫使 TIR 模仿 RGB Teacher

#### 报告的判断

报告认为，弱光、模糊和遮挡恰好是 RGB Teacher 可能不可靠、TIR 最有价值的场景。如果仍使用固定权重强制 TIR 向 RGB 对齐，可能将 RGB 的缺陷蒸馏给 TIR。

#### 核验结论

这一担忧有充分依据，但尚未被我们的实验直接证明。

- RGBT-GroundBench 明确报告低照度是 grounding 的主要鲁棒性缺口，热红外在低照度下有帮助，但小目标和恶劣天气仍很难。
- M2D-LIF 证明多模态联合训练可以反过来损害单模态特征学习，并通过单模态蒸馏和局部照度权重来减轻 Fusion Degradation。
- RGBT-VGNet 的融合不是等权蒸馏，而是使用照明、局部语义与全局可靠性决定 RGB/TIR 权重。

但当前不应立即把 `w_RGB-quality` 写进训练损失，因为我们尚未证明：

```text
C2 的对齐漂移是否真正集中在弱光、模糊、遮挡或 RGB target-background margin 低的样本。
```

所以正确的下一步是先做 **Teacher reliability audit**，再决定是否质量加权。

### 3.2 问题二：缺少 Query 下游相关性

这个判断完全正确。当前 Phase 1–1.6 优化的是 RGB ROI 与 TIR ROI 的跨谱表征，不是 Query 条件下的目标选择。

RGBT-VGNet 的 LAVS 先用文本查询两个模态的视觉 token，再进行跨模态交互；这比“先盲目融合、再让 LLM 理解”更贴近 AIC。

还有一篇网页报告未纳入、但与 AIC 高度相关的工作：**RGBX-R1**。它直接研究 MLLM 在 RGB+红外/深度/事件模态上的 grounding，使用 Understand–Associate–Validate 数据构造和两阶段 SFT/RL。它是 2026 年预印本，未检索到可直接使用的官方代码/权重，因此适合用来设计 Phase 1.8 的 Query–TIR probe，不适合作为下一轮主工程基座。

### 3.3 问题三：共享语义与互补信息没有显式解耦

方向正确，但不能仅凭 Phase 1.6 就断定必须改成双分支。

支持证据：

- UniRGB-IR 通过冻结 RGB foundation model，用 Multi-modal Feature Pool 和 Supplementary Feature Injector 注入补充信息。
- RSDet 先用动态频谱滤波去除干扰，再用尺度感知专家选择需要融合的特征。
- 2026 年的 proxy-based RGB-IR 工作显式解耦 modality-shared 和 modality-specific knowledge，再分配给不同专家。
- M-SpecGene 显示专业 RGBT 自监督预训可以获得更强、更通用的跨模态表征。

但当前 C2 已经将有效秩恢复到 Base TIR 的 89.77%，且 R@5 达到 83.56%。这说明“单 Adapter 一定无法兼顾共享与互补”尚未被证明。

应当仅在以下证据出现时升级双分支：

```text
1. 质量加权 D1 仍然在“对齐”与“秩/检索”之间存在稳定的 Pareto 冲突；或
2. Query–TIR probe 显示对齐特征能检索目标，却无法保留低光/热显著性独有证据。
```

### 3.4 问题四：RGB Teacher 可靠性不应固定

这是当前最值得立即验证的新假设。

但质量分数不能只是全图亮度，应同时包括：

```text
全图 RGB brightness / contrast / blur
ROI brightness / contrast / entropy
RGB target-vs-background teacher margin
illumination / occlusion / weather 标签
RGB Teacher paired cosine 与 retrieval 表现
```

需要比较这些质量代理与 C2 的逐层 alignment delta 是否具有稳定关联。如果仅“低照度标签”相关，可以使用简单分组权重；如果局部质量才有预测力，则需要可学习的 patch/ROI quality gate。

### 3.5 问题五：RGBT-GroundBench 与 AIC 的域差异

报告的风险判断正确，但“必须立即加入所有 AIC 风格退化增强”过于激进。

增强会同时改变表征、配准和质量分布，会破坏对 C2 问题的归因。正确顺序是：

```text
先对现有 repair-dev 做弱配准/黑边/小目标分组诊断
→ 再构建小规模 degradation stress set
→ 只在实测证明某类退化会系统性破坏特征时，才将它加入训练。
```

---

## 4. 论文筛选结果

### 4.1 现在必须精读

| 论文 | 完成度 | 对当前问题的直接价值 | 当前用法 |
|---|---|---|---|
| RGBT-GroundBench / RGBT-VGNet, ECCV 2026 | 论文、代码、数据公开 | 与 AIC 任务最接近；AMA、LAVS、TPF 直接对应适配、Query 选择和可靠性融合 | 当前主要方法基准 |
| M2D-LIF, ICCV 2025 | 论文、代码、权重公开 | 定义 Fusion Degradation，用单模态蒸馏保护各自能力，LIF 使用局部照明权重 | 设计 Teacher reliability audit 和 Phase 2 degradation guard |
| IV-tuning, arXiv v5 (2026-02) | 代码公开，尚未确认顶会录用 | 冻结 visible VFM，仅训练约 3% 模态 prompt/adapter，同时学习互补性 | 检查 Adapter 插入层与容量，不直接搬训练脚本 |
| RGBX-R1, arXiv 2026 | 预印本；未找到可直接复用的官方仓库/权重 | 直接研究 RGB+X+Query grounding，弥补当前只做 ROI 对齐的缺口 | 用于设计 Phase 1.8 Query–TIR probe，暂不上 RL |

### 4.2 进入 Phase 2 前阅读

| 论文 | 可借鉴内容 | 不立即实施的原因 |
|---|---|---|
| UniRGB-IR, ACM MM 2025 | 冻结 RGB 基座，通过 MFP + SFI 注入补充特征 | 官方仓库的 TODO 仍显示核心代码/多类任务权重发布状态不完整；且不是 grounding |
| RSDet | 先去除干扰频谱，再动态选择尺度特征 | 会同时改变表征与融合，不适合 Phase 1.7 归因 |
| M-SpecGene, ICCV 2025 | RGBT 大规模自监督预训、CMSS 信息密度与对象中心 mask | 它是长期 TIR 初始化/专业基座方案，替换初始表征会打破当前 C2 控制实验 |

### 4.3 只在弱配准成为主瓶颈时阅读/实施

| 论文 | 方法 | 进入条件 |
|---|---|---|
| C²Former, TGRS 2024 | 跨模态 cross-attention 隐式校准并捕捉互补特征 | weak-alignment 分组显著落后 |
| CAGT, Information Fusion 2024 | ROI 级平移/尺度/旋转级联校准 + 互补 Transformer | 证明几何偏移而不是表征是主因 |
| CoDAF, arXiv 2025 / Applied Soft Computing 2026 | offset-guided deformable alignment + dynamic gated fusion | Phase 2 中 registration-risk 子集存在系统性 Harm |

### 4.4 本轮不采用

- 不使用 M-SpecGene 替换 Qwen 视觉基座。
- 不训练 RGBX-R1 式 RL。
- 不实现 RSDet 频域过滤。
- 不实现 CoDAF/CAGT 几何对齐。
- 不同时加入 Query loss、质量加权、双分支和融合门。

---

## 5. 下一轮唯一推荐任务：Phase 1.7A+ 无训练证据审计

### 5.1 唯一目标

```text
在不训练新 Adapter、不使用 official val 调参的前提下，
确定 C2 的门禁失败究竟来自：
A. 不稳定的相对指标；
B. 低质量 RGB Teacher 的错误约束；
C. InfoNCE 假负例；
D. 真实逐层对齐漂移。
```

### 5.2 执行前的资产门禁

当前本地 `outputs/aic_rgbtir_phase16_v1/` 只有 preflight 和 manifests，**未发现 C1/C2 Adapter、candidate summary 和 Teacher Bank**。

所以首先必须从云端/持久盘拉回并校验：

```text
C1 Adapter + SHA256
C2 Adapter + SHA256
C0/C1/C2 candidate summaries
Teacher Bank 或可重构它的 fingerprint
repair-dev manifest SHA256
Qwen revision + processor fingerprint
Phase 1.6 完整日志
```

如果 C1/C2 不能恢复，不能伪造诊断；必须重现 C1/C2 probe 或回到云端持久资产。

### 5.3 子任务 A：逐层绝对对齐诊断

对 Base TIR / C1 / C2，在固定 repair-dev 上输出 Layer 8/16/24/final：

```text
paired cosine mean / median / P5 / P95
alignment loss
absolute delta
relative delta 和真实分母
R@1 / R@5
effective rank / participation ratio / singular-value entropy
non-paired cosine mean / P95
paired-shuffled margin + bootstrap 95% CI
```

结论分类：

```text
绝对 cosine 下降 <= 0.01，但相对比率巨大为负
→ 指标小分母失真

任一主要层平均 cosine 下降 > 0.02
→ 真实逐层漂移
```

### 5.4 子任务 B：RGB Teacher 可靠性审计

每条 record 添加：

```text
global_rgb_brightness
global_rgb_contrast
global_rgb_blur
roi_rgb_brightness
roi_rgb_contrast
roi_rgb_entropy
rgb_target_background_margin
illumination / weather / occlusion
```

检查它们与下列指标的关联：

```text
C2 - Base paired cosine delta
C2 - Base R@5 delta
C2 worst-layer status
```

使用分组均值、Spearman 相关和固定 bootstrap CI，不训练质量分类器。

### 5.5 子任务 C：InfoNCE 假负例审计

当前每个 TIR ROI 使用 256 个跨图 RGB 负例。需要统计：

```text
同 target superclass 负例比例
高 Query 语义相似负例比例
同条件/同场景/同尺度 hard-negative 比例
最高相似度 top-20 中的潜在假负例比例
```

因 manifest 没有显式 category 字段，需要使用固定、可审计的 Query target-head parser；低置信记录不强制归类。

### 5.6 子任务 D：漂移与风险聚类

将 worst 100 和主要分组按以下字段输出：

```text
source_dataset
illumination
weather
object_size
occlusion
IR valid-FOV ratio
black-border severity
registration-risk proxy
ROI token count
target superclass
RGB quality bin
```

这一步同时判断是否需要在后续启用 CoDAF/C²Former 式配准，但本轮不实现配准模块。

---

## 6. Phase 1.7A+ 结果如何唯一决定下一步

| 诊断结果 | 后续唯一主线 |
|---|---|
| 仅是小分母失真，C2 绝对对齐可接受 | 修正门禁，使用 C2 从全新 Adapter 做 full train；不加 D1 |
| 真实漂移，但与 RGB 质量无明显关系 | 训练 D1：C2 + per-layer base-relative retention |
| 真实漂移明显集中在低质量 RGB | 训练 D1-Q：D1 + 预注册 RGB-quality weighted paired/retention |
| 假负例比例高，且导致同类目标过度分离 | 将 InfoNCE 改为 category-aware multi-positive 或 query-soft-negative，其他保持 C2 |
| 对齐、秩与检索长期存在 Pareto 冲突 | 才启动 Shared/Complementary 双分支小 probe |
| 只有 weak-alignment 簇显著失败 | 在 Phase 2 准备 registration gate；仍不在 Adapter 阶段引入 CoDAF |

---

## 7. 本轮验收边界

Phase 1.7A+ 是一轮可完整结束的诊断任务，交付：

```text
outputs/aic_rgbtir_phase17a_plus_v1/
  asset_preflight.json
  layerwise_metrics.json
  per_record_metrics.jsonl
  rgb_teacher_quality.csv
  negative_sampling_audit.json
  subgroup_metrics.csv
  worst_cases.csv
  decision.json
  sha256_manifest.json

reports/
  aic_rgbtir_phase17a_plus_diagnosis_2026_08_13.md
```

最终 `decision.json` 只能输出一个下一训练分支：

```text
C2_FULL_TRAIN
D1_RETENTION
D1_QUALITY_WEIGHTED
C2_FALSE_NEGATIVE_AWARE
SHARED_COMPLEMENTARY_PROBE
ASSET_BLOCKED
```

本轮不得：

- 更改 Adapter 权重；
- 打开 official val 调参；
- 实现 RGB–TIR fusion；
- 实现 CoDAF/CAGT；
- 使用 AIC 测试集训练或拟合质量阈值；
- 将表征指标称为 AIC ACC 提升。

---

## 8. 资源与时间判断

- 资产拉回和静态审计：本机完成。
- 逐层特征提取：RTX 4060 8GB 可尝试，使用单样本 BF16 + inference mode + 按 image pair 缓存。
- 如本地 worst-case smoke OOM，将同一诊断任务转移 4090，不降低分辨率或修改 processor 口径。
- 本轮无需先租 4090；首先确认 C1/C2 资产能否恢复。

---

## 9. 信息来源表

| 来源 | 年份/状态 | 用途 |
|---|---|---|
| [RGBT-GroundBench / RGBT-VGNet](https://arxiv.org/abs/2512.24561) / [official repo](https://github.com/crazyxiaoxi/RGBT-GroundBench) | ECCV 2026；官方仓库标注 academic research license | 任务最近的主基准；AMA/LAVS/TPF |
| [M2D-LIF](https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_Rethinking_Multi-modal_Object_Detection_from_the_Perspective_of_Mono-Modality_Feature_ICCV_2025_paper.html) / [repo](https://github.com/Zhao-Tian-yi/M2D-LIF) | ICCV 2025；代码与权重可用 | Fusion Degradation、单模态保护、照度融合 |
| [IV-tuning](https://arxiv.org/abs/2412.16654) / [repo](https://github.com/Yummy198913/IV-tuning) | arXiv v5, 2026-02；代码公开 | 参数高效的模态 prompt/adapter |
| [UniRGB-IR](https://arxiv.org/abs/2404.17360) / [repo](https://github.com/PoTsui99/UniRGB-IR) | ACM MM 2025；仓库发布状态不完整 | MFP/SFI 补充特征注入 |
| [RSDet](https://arxiv.org/abs/2401.10731) | 2024 | 去凗余后动态选择 |
| [C²Former](https://arxiv.org/abs/2306.16175) | TGRS 2024 | 隐式跨模态校准与互补交互 |
| [CoDAF](https://arxiv.org/abs/2506.16737) | arXiv 2025 / Applied Soft Computing 2026 | 弱配准 offset-guided alignment + dynamic fusion |
| [M-SpecGene](https://arxiv.org/abs/2507.16318) | ICCV 2025 | RGBT 大规模自监督表征；长期候选 |
| [RGBX-R1](https://arxiv.org/abs/2602.00504) | 2026 预印本；未确认官方可复用代码 | Query–X grounding 与后续 SFT/RL 设计 |

---

## 10. 覆盖缺口、可能过时与冲突信息

### 覆盖缺口

1. 未获得 C1/C2 逐层原始记录，因此不能回答具体哪一层漂移。
2. 本地尚未固化 C1/C2 Adapter 和 Teacher Bank，这是下一轮的首要工程风险。
3. RGBX-R1 暂未检索到官方可直接复用的代码/权重。
4. 目前还没有 Query–TIR grounding probe，因此无法证明高检索指标必然转化为 bbox 收益。

### 可能过时的信息

- GitHub stars、release 和 TODO 状态会变化；实施某论文代码前应再次检查官方仓库。
- IV-tuning 当前确认到 arXiv v5，未在本次检索中确认其正式会议/期刊录用。

### 冲突或需要限定的信息

- UniRGB-IR 仓库 README 同时出现“weights released”与多项核心代码/权重 TODO；应视为部分开源，不视为即插即用。
- RGBT 检测论文的 mAP 增益不能直接等同于 AIC visual grounding ACC@0.5 增益。
- 网页报告提出的 Shared/Complementary 双分支是合理设计假设，不是当前实验已证明的必选架构。

---

## 11. 最终决策

**下一轮不开启云端训练，不开始 Phase 2，不实现双分支。**

下一轮任务固定为：

> **Phase 1.7A+ = C1/C2 资产恢复 + 逐层绝对对齐诊断 + RGB Teacher 可靠性审计 + InfoNCE 假负例审计。**

它的价值不是再生成一组“看起来更好的指标”，而是将下一轮训练从五种可能路线收敛为唯一有证据的分支。
