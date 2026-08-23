# AIC RGB–红外训练数据与融合路线（2026-08-11）

> AS_OF：2026-08-11（Asia/Shanghai）
> 目标：在不污染 AIC 正式测试集、不中断当前 RGB 最优基线的前提下，建立红外优先、深度后置的多模态训练路线。

> **2026-08-11 晚间更新：** RGBT-GroundBench 已完成解压和全量标注/抽样配准审计。最新的数据适配结论、AMA 实施方案、恶劣天气分布和 AIC 风格差异见：`reports/aic_rgbt_vgnet_ama_data_fit_and_training_plan_2026_08_11.md`。该新报告取代本文中“数据尚未审计”的旧状态；按当前项目决策，本阶段不再以 AIC 同源/重叠检查作为训练启动门槛。

## 1. 执行结论

1. **下一步不应直接微调 Qwen3-VL-30B。** 当前 `0.7757` 提交来自 Qwen3-VL-30B-A3B-Instruct-FP8，并有 13 条由 Qwen3-VL-8B 补框；它应作为冻结的 RGB 平台控制结果，而不是多模态训练起点。先用 8B 或专用 grounding 基座验证数据、配准、门控和损失是否真的有效，再移植到 30B；正式新路线应改为同模型重试/修复，不再依赖外部模型 fallback。
2. **本地当前没有可直接、安全训练的高质量 RGB-T grounding 数据。** `D:\AIC赛题一数据集` 现有的是 RGB grounding 与 RGB-D grounding；未发现独立的 RGB-T+Query+bbox 外部数据。
3. **另一个 AIC 赛题的三模态训练包存在已确认泄漏，禁止用于本赛题训练。** 至少图像组 `000002` 的 RGB、红外和 Depth 与本赛题正式测试集逐字节一致。
4. **首选外部红外数据是 RGBT-GroundBench，但必须先做重叠和许可证审计。** 它直接提供 RGB-TIR+Query+bbox，任务最贴近；但数据来自 FLIR、M3FD、MFAD，而 AIC 很可能与这些来源同源。
5. **RGB 仍需要补充训练，但只做“针对性补课”，不做无差别重训。** 小目标用 SOREC/PIZA，结构区域用 PhraseCut，复数/集合用 gRefCOCO，长 Query 保留 RefCOCOg；训练必须保留 RGB replay，防止再次发生 RefCOCO 域负迁移。
6. **融合必须是配准感知、有效区感知和 Query 感知的动态门控。** 黑边、旋转、FOV 差异或属性不可见时，红外权重应自动接近零，而不是强制参与。

## 2. 本地数据资产结论

### 2.1 可安全保留的外部数据

`D:\AIC赛题一数据集` 当前已有：

- RefCOCO / RefCOCO+ / RefCOCOg；
- gRefCOCO；
- SUN-Spot 与 SUN RGB-D；
- COCO2014 共享图像底座。

现有统一清单位于：

`D:\AIC赛题一数据集\manifests\processed`

其中 SUN-Spot 本地统计为 1,948 张图、3,245 个 referent、8,010 条表达。它适合后续验证 Depth 接口和空间关系，但属于室内 RGB-D 域，不能代表 AIC 的户外、无人机、小目标和建筑结构分布。

### 2.2 禁止使用的本地三模态训练包

高风险文件：

`D:\初赛数据集-面向城市场景的多模态目标检测\训练集\AIC2026_Train_2000.zip`

它包含 2,000 组 `visible / infrared / depth / labels`，但已确认以下精确重复：

| 模态 | ZIP 中 `000002.png` SHA-256 | AIC 正式测试同名文件 SHA-256 | 结论 |
|---|---|---|---|
| visible | `77ea606d6f14ee96dc3337435260f269fd53b25043190d6a1591fe6af488ac1e` | 相同 | 精确重复 |
| infrared | `6767f1b6f9c6617d63c44b0c3a982b794f37192b19839cfb92429b9ccc5f1697` | 相同 | 精确重复 |
| depth | `5806cbd9192b3f417dff48be225b6c3d5d2b8271d25f86fbf58e62ee6e14b959` | 相同 | 精确重复 |

因此，该 ZIP 只能用于只读格式核验和重叠审计，不能用于：

- 训练红外/深度分支；
- 生成文本伪标注；
- 训练 bbox head；
- 校准测试图像的专用变换；
- 从相邻帧构造训练数据。

即使删除精确重复帧，也仍可能存在同视频相邻帧泄漏；若来源序列不能隔离，应整体禁用。

## 3. 外部数据优先级

### P0：RGBT-GroundBench（先审计，后决定训练）

- ECCV 2026；
- 21,535 对 RGB-TIR；
- 38,760 条 grounding 实例；
- 训练 26,604、验证 2,032、测试 10,124；
- 每条包含 RGB、TIR、Query、目标 bbox 和光照/天气/尺寸/遮挡标签；
- 总资产约 10.4 GB；代码、训练和评测流程公开。

官方资料：

- 论文：https://arxiv.org/abs/2512.24561
- 代码：https://github.com/crazyxiaoxi/RGBT-GroundBench
- 数据：https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset

为什么第一优先：它是目前公开数据中与 AIC 输入输出最接近的监督源。

为什么不能立刻训练：它由 FLIR、M3FD、MFAD 构建，必须先与 AIC 2,000 个图像组做 SHA-256、pHash、局部特征和同序列近重复审计。若某个子源存在测试重叠，应整段/整源排除，而不是只删同名文件。

### P1：SOREC / PIZA（RGB 小目标专项）

- ICCV 2025；
- 10 万条小目标 Query+bbox；
- 所有目标面积小于图像面积的 1%；
- 表达平均 25.5 个词；
- 公开数据、代码和预训练 adapter。

官方资料：

- 论文：https://openaccess.thecvf.com/content/ICCV2025/html/Goto_Referring_Expression_Comprehension_for_Small_Objects_ICCV_2025_paper.html
- 代码/数据：https://github.com/mmaiLab/sorec

用途：提升 AIC 中摄像头、灯、无人机、标志和远距离目标。它只补 RGB 小目标能力，不解决热红外融合。

### P1：M-SpecGene（红外编码器初始化候选）

- ICCV 2025；
- 从 41 个 RGBT 数据集清洗出 548,238 对样本；
- 公开 ViT-B 预训练权重、检测/分割转换权重和训练代码；
- 代码为 MIT。

官方资料：

- 论文：https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_M-SpecGene_Generalized_Foundation_Model_for_RGBT_Multispectral_Vision_ICCV_2025_paper.html
- 代码/权重：https://github.com/CalayZhou/M-SpecGene

用途：作为独立 IR/RGBT 视觉编码器初始化，比从 RGB ViT 生硬复制更合理。

风险：RGBT550K 聚合了 41 个来源，必须审查底层许可证和 AIC 同源风险。优先考虑公开预训练权重，不自行使用未审计的 RGBT550K 全量重训。

### P2：LLVIP

- 15,488 对严格时空对齐 RGB-IR；
- 主要为低光行人；
- 有行人 bbox，无语言 Query；
- 官方提供原始未配准数据，可用于专门研究配准。

官方资料：https://github.com/bupt-ai-cz/LLVIP

用途：训练 IR 低层特征、有效性评分和低光门控；不能单独训练开放词汇/复杂 Query 模块，且类别过窄。

### P2：FLIR、M3FD、MFAD

- FLIR 官方 starter 数据约 14,000 张、多类别，适合道路热成像检测；
- M3FD 约 4.2K 对，多场景、多类别；
- 三者已被 RGBT-GroundBench转化为语言 grounding 子集。

FLIR 官方入口：https://oem.flir.com/en-au/solutions/automotive/dataset

建议：不要重复下载同一底层源；优先使用 RGBT-GroundBench 的语言标注版本，并完成来源级去重。

### P3：RGBDT500（Depth 阶段再审计）

- NeurIPS 2025 Datasets and Benchmarks；
- 500 个同步 RGB+Depth+Thermal 视频，约 203.7K 帧；
- bbox 为跟踪监督，无语言 Query；
- RDTTrack 使用预训练 RGB 跟踪器、三模态提示和正交投影约束。

官方资料：https://proceedings.neurips.cc/paper_files/paper/2025/hash/b4962fcd5d4410a9f43ef70f528eedd8-Abstract-Datasets_and_Benchmarks_Track.html

它与 AIC 的传感器组合最像，但也最可能同源。完成许可、精确重复、相邻帧和场景近重复审计前，不下载、不训练。

## 4. RGB 模块是否还要训练

答案是：**需要针对性训练，但不能重做一次“RefCOCO 全量微调”。**

当前 RGB-only 主模型对应的平台提交已达 `0.7757`（其中 13 条由 8B 补框），说明通用语义能力很强。下一阶段 RGB 数据只解决明确缺口：

| 缺口 | 建议数据 | 训练作用 |
|---|---|---|
| 极小目标、长 Query | SOREC / PIZA | 渐进缩放、保持高分辨率定位 |
| passage、awning、building part 等结构区域 | PhraseCut / Visual Genome region | 区域、stuff、部件与长尾概念 |
| 复数/群组/无目标 | gRefCOCO | 数量、集合外接框、困难负样本 |
| 长句和主体-参照物 | RefCOCOg | 角色与关系理解 |
| 普通实体与属性 replay | RefCOCO / RefCOCO+ | 防止多模态训练破坏 RGB 基础能力 |

不建议：

- 只用 RefCOCO 继续全模型微调；
- 让 bbox 面积/位置捷径支配训练；
- 用 AIC 测试预测生成伪标签；
- 先把所有 RGB、IR、Depth 无条件拼接后再看平台结果。

## 5. 推荐的 RGB-T 模型结构

第一版应采用“保留 RGB 主干 + 可插拔红外适配器”，而不是双主干全量重训：

```text
RGB -> 当前视觉编码器 ------------------------------┐
                                                     │
IR  -> 有效区 mask -> 配准/对齐 -> IR adapter ------├-> Query-conditioned reliability gate
                                                     │
Query -> target/attribute/relation/visibility tokens -┘
                                                        -> grounding decoder -> RGB bbox
```

关键模块：

1. **IR valid mask**：先屏蔽黑边、坏点和无覆盖区域，黑边不参与归一化、注意力或损失。
2. **弱配准模块**：学习 translation/scale/rotation 或 deformable offsets；不假设同像素对齐。
3. **语言引导融合**：`checkered/white/red/text/logo` 等外观 Query 主要依赖 RGB；`person/animal/low-light/occluded` 可增加 IR 权重。
4. **可靠性门控**：结合 RGB 亮度、IR 信息量、配准置信度、目标是否落在 IR 有效区和 Query 类型，允许 IR 权重退化为 0。
5. **同模型缺失模态鲁棒性**：通过 modality dropout 训练同一个模型在 IR 无效时退回 RGB 路径，不使用另一个模型 fallback。
6. **输出坐标统一**：所有 loss 和最终 bbox 都以 RGB 坐标系为准。

## 6. 训练顺序

### Stage 0：泄漏与许可证门禁

1. RGBT-GroundBench 下载到隔离目录；
2. 与 AIC 测试集比较 SHA-256；
3. 对不同压缩/裁剪版本做 pHash；
4. 对候选近重复做局部特征匹配；
5. 按底层数据源和序列隔离，而非只删重复文件；
6. 记录每个来源的许可证和可否用于竞赛训练。

### Stage 1：轻量原型，不动 30B

- 基座：Qwen3-VL-8B 或 RGBT-VGNet/HiVG 类专用 grounding 基座；
- 冻结 RGB 视觉主干和语言主干；
- 只训练 IR adapter、配准、门控和小型融合层；
- 外部验证同时报告 RGB-only、IR-only、RGB+IR；
- 分层报告低光、小目标、正常光、错位和黑边样本。

### Stage 2：针对性 RGB replay

每个训练批次保留 RGB-only replay。建议初始实验从以下配比起步，最终由外部验证冻结：

- 50%：通过泄漏门禁的 RGB-T grounding；
- 25%：SOREC/PhraseCut/gRefCOCO 困难样本；
- 25%：RefCOCOg/RefCOCO+ RGB replay。

这只是初始实验配比，不是已验证最优值。

### Stage 3：移植到当前最优 30B

只有当 8B/专用基座在外部数据上满足以下条件才进入：

- RGB+IR 相对 RGB-only 有稳定提升；
- 正常光 RGB 性能没有明显下降；
- 错位和黑边分组不发生负增益；
- 外观/颜色 Query 的 IR 权重接近零；
- 无效 IR 时同模型仍能输出合法框。

随后只在 30B 上训练 LoRA、IR projector、门控与配准模块，不全量更新语言模型。

## 7. 最值得阅读的近期论文

### 与 AIC 最直接

1. **RGBT-GroundBench / RGBT-VGNet，ECCV 2026**
   直接研究 RGB+TIR+语言->bbox；提出不对称模态适配、语言感知协同和三先验可靠性融合。第一优先。

2. **Referring Expression Comprehension for Small Objects / PIZA，ICCV 2025**
   解决长 Query+极小目标，和 AIC 的尺度问题高度匹配。

3. **Thermal-Det，2026 预印本**
   研究语言引导的开放词汇热成像检测，包含 RGB teacher 蒸馏、Thermal-Text Alignment 和跨模态注意力。方法值得参考，但尚不是与 RGBT-GroundBench 同等级的成熟竞赛基线。
   https://arxiv.org/abs/2605.10130

### 解决红外编码与模态偏置

4. **M-SpecGene，ICCV 2025**
   大规模 RGBT 自监督基础模型；适合初始化 IR/RGBT encoder。

5. **Causal Mode Multiplexer，CVPR 2024**
   重点解决数据集中“夜间标签总和 thermal 强相关”的模态偏置，防止模型学成只在固定条件信任某一模态。
   https://openaccess.thecvf.com/content/CVPR2024/html/Kim_Causal_Mode_Multiplexer_A_Novel_Framework_for_Unbiased_Multispectral_Pedestrian_CVPR_2024_paper.html

6. **RGB-X Object Detection via Scene-Specific Fusion Modules，WACV 2024**
   用少量配准数据和小型 scene-specific 模块复用单模态预训练模型，适合保留现有 RGB 能力。
   https://openaccess.thecvf.com/content/WACV2024/html/Deevi_RGB-X_Object_Detection_via_Scene-Specific_Fusion_Modules_WACV_2024_paper.html

### 解决错位与跨模态融合

7. **Improving RGB-Infrared Object Detection with Cascade Alignment-Guided Transformer，Information Fusion 2024**
   使用 translation/scale/rotation alignment 与 complementary fusion transformer，直接对应本地审计发现的黑边、旋转和 FOV 差异。
   https://doi.org/10.1016/j.inffus.2024.102246

8. **Pseudo Visible Feature Fine-Grained Fusion，CVPR 2025**
   使用伪可见特征与多层级 Graph-Mamba 融合，公开代码；适合研究 thermal 特征如何保留细节，但任务是 thermal detection，不是语言 grounding。
   https://openaccess.thecvf.com/content/CVPR2025/html/Li_Pseudo_Visible_Feature_Fine-Grained_Fusion_for_Thermal_Object_Detection_CVPR_2025_paper.html

9. **RDTTrack / RGBDT500，NeurIPS 2025**
   后续 Depth 阶段重点参考其三模态 prompt learning 与正交投影约束。

## 8. 下一步三个单变量实验

1. **数据门禁实验**：下载 RGBT-GroundBench，仅做许可证与 AIC 精确/近重复审计，不训练。
2. **红外适配器实验**：在通过门禁的外部 split 上，比较 RGB-only 与 `RGB + valid-mask + registration + gated IR adapter`，基座先用 8B/专用 grounding 模型。
3. **RGB 小目标实验**：只加入 SOREC/PIZA，对 RGB-only 外部验证比较小目标分组，同时检查普通目标是否退化。

三个实验不得合并成一次平台提交。外部结果冻结后，平台第一次只测试“红外门控分支”这一个新增变量；Depth 留到红外路线稳定之后。

## 9. 不确定性边界

- 当前没有 AIC GT，无法直接证明红外能提高多少平台 ACC。
- 外部数据的同源/近重复风险尚未完成全量审计。
- RGBT-GroundBench 的底层数据许可分别继承自 FLIR、M3FD、MFAD，需逐项确认。
- M-SpecGene 的代码和权重公开不等于其聚合数据可无条件用于比赛。
- 本文中的训练配比和门控阈值是实验起点，不是已验证最优参数。
