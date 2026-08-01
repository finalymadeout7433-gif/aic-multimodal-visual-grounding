# AIC 两轮策略、平台结果与 Spatial LTR 负迁移复盘

## 1. 文档目的

本文把两轮实验代码、受控本地证据和真实 AIC 平台结果放在同一条证据链中，回答：

1. 第一轮为什么先诊断 Florence 候选、Tile 和 GroundingDINO；
2. 第二轮为什么训练候选 Ranker，而不是微调 GroundingDINO；
3. 为什么本地 holdout 提升 11.22 个百分点，平台却下降 17.48 个百分点；
4. 哪些方向已经被平台证伪，下一轮应该怎样降低风险。

正式 AIC 9,555 条 Query 没有 bbox。本项目从未使用 AIC 测试标签、人工 bbox 或
伪标签训练。AIC 平台 ACC@0.5 是唯一目标域准确率证据。

## 2. 第一轮：候选上限、Tile 与零样本诊断

### 2.1 边界

- 数据：RefCOCO、RefCOCO+、RefCOCOg validation 固定子集；
- 输入模态：RGB + 英文 Query；
- AIC：仅做无标签 smoke，不计算本地准确率；
- 不训练、不生成正式提交、不读取 holdout；
- 固定 seed、清单和模型哈希。

### 2.2 Florence 结果

`core_eval_1500`：

| 指标 | 结果 |
|---|---:|
| first ACC@0.5 | 0.6893 |
| candidate oracle ACC@0.5 | 0.7533 |
| oracle-first 差值 | +6.40 pp |
| first correct | 1,034 |
| oracle 可救 | 96 |
| 所有候选失败 | 370 |

结论：Florence 候选排序有优化空间，但 24.67% 样本所有候选都失败，单纯重排不是
唯一突破口。

### 2.3 Tile 结果

`tile_probe_600`：

| 组别 | full oracle | full+tile oracle | 增益 | 延迟倍率 |
|---|---:|---:|---:|---:|
| 小目标 | 0.6200 | 0.7933 | +17.33 pp | 4.90× |
| 中/大目标 | 0.8233 | 0.8900 | +6.67 pp | 4.90× |

结论：Tile 能扩展候选召回，但成本高，应作为选择性召回增强，而不是无条件全量推理。

### 2.4 GroundingDINO-Tiny 结果

`core_eval_1500`：

| 指标 | 结果 |
|---|---:|
| Top-1 ACC@0.5 | 0.5627 |
| Top-5 oracle | 0.8820 |
| Top-10 oracle | 0.9073 |
| 无候选率 | 0 |

该结果支持“在 RefCOCO 分布上，GroundingDINO 适合作为高召回候选生成器”。它不
证明 AIC 的 Top-10 oracle 同样高，因为 AIC 没有 GT。

## 3. 第二轮：GroundingDINO Top-10 + Spatial LTR

### 3.1 训练对象

冻结 `IDEA-Research/grounding-dino-tiny`，不更新视觉或文本编码器。GroundingDINO
为每条 Query 生成最多十个候选，实际训练对象是 LightGBM LambdaRank 排序器。

```text
RGB + Query
    -> GroundingDINO-Tiny Top-10
    -> score / bbox geometry / spatial / ordinal / text-role features
    -> LightGBM LambdaRank
    -> selected bbox
```

这不是强化学习，也不是 GroundingDINO 微调。

### 3.2 数据和模态

| Split | Query 数 | 数据来源 |
|---|---:|---|
| train | 50,000 | RefCOCO / RefCOCO+ / RefCOCOg |
| validation | 10,000 | 同系列、image key 隔离 |
| holdout | 10,000 | 同系列、冻结后单次评测 |

训练使用：RGB、英文 Query、外部 GT bbox。未使用 IR、Depth、AIC 标签、AIC
伪标签、Tile、CLIP crop、Florence 候选或人工 Query 改写。

### 3.3 本地结果

| 方法 | Holdout ACC@0.5 | 相对 Top-1 |
|---|---:|---:|
| GroundingDINO Top-1 | 0.5211 | 基准 |
| 保守空间规则 | 0.5369 | +1.58 pp |
| Score + Geometry Ranker | 0.6006 | +7.95 pp |
| 完整 Spatial LTR | 0.6333 | +11.22 pp |
| Top-10 oracle | 0.8936 | +37.25 pp |

在外部同分布 holdout 上，这些结果是真实且可复现的；问题是它们没有迁移到 AIC。

## 4. 真实平台结果

| 提交 | 策略 | AIC ACC@0.5 | 相对 S01 | 相对 S02 |
|---|---|---:|---:|---:|
| S01 | Florence-2 RGB-only first | 0.4980 | 基准 | +0.42 pp |
| S02 | GroundingDINO 原始 Top-1 | 0.4938 | -0.42 pp | 基准 |
| S03 | GroundingDINO + Spatial LTR | 0.3190 | -17.90 pp | -17.48 pp |

S03 修改 4,770/9,555 条预测。根据平台四位小数近似换算，S03 相对 S02 净损失约
1,670 条正确预测；在发生切换的子集中，相当于约 -35.0 个百分点的净变化。

## 5. 已排除的工程问题

对全部 9,555 条候选、选择记录与提交 JSON 重新连接检查：

- 非法 Ranker candidate index：0；
- S02 bbox 与 candidate 0 不一致：0；
- S03 bbox 与 `candidate[ranker_index]` 不一致：0；
- NaN、inf、反向框、越界框：0；
- 零候选 Florence fallback：0；
- 两个 ZIP 都只含一个正确 JSON，哈希与发布记录一致。

因此平台下降不是候选下标、坐标归一化或 ZIP 装包错误，而是 Ranker 真实选择行为。

## 6. 根因一：尺度捷径在 AIC 上反转

Ranker 特征 Gain 前四名：

| 特征 | Gain 占比 |
|---|---:|
| area | 23.58% |
| gdino_score | 8.63% |
| log_area | 6.95% |
| score_gap_to_first | 6.10% |

候选分布对比：

| 统计 | 外部 holdout | AIC |
|---|---:|---:|
| Top-1 候选面积中位数 | 20.57% | 2.57% |
| 全部候选面积中位数 | 14.09% | 1.66% |
| Query 单词数中位数 | 4 | 9 |
| Query 平均单词数 | 5.11 | 10.44 |

这些是候选框统计，不是未知的 AIC GT 面积，但足以证明模型面对的尺度与语言分布发生
明显变化。

Ranker 改选行为：

| 统计 | 外部 holdout | AIC |
|---|---:|---:|
| 切换率 | 35.18% | 49.92% |
| 新框/Top-1 面积比中位数 | 0.73× | 5.67× |
| 新框至少大 2 倍 | 13.79% | 72.81% |
| 新框大于 Top-1 | 33.48% | 88.95% |
| 新旧框相同 label | 64.21% | 42.96% |
| 新旧框 IoU < 0.1 | 57.28% | 70.19% |

在外部数据上，模型常用更小、更精确的候选修正 Top-1；进入 AIC 后却大量选取更大的
建筑、门、路灯和参照区域。绝对面积先验成为错误捷径。

## 7. 根因二：目标被参照物替换

对含参照物且发生切换的 Query，比较候选 label 与目标短语/参照物短语的 token overlap：

| 统计 | 外部 holdout | AIC |
|---|---:|---:|
| 含参照物的切换 | 726 | 1,014 |
| Top-1 更像目标、Ranker 更像参照物 | 7 | 238 |
| 目标到参照物翻转率 | 0.96% | 23.47% |

典型案例：

```text
Query: A man in a checkered shirt beside the white street lamp
Top-1: a man, score=0.7533, area=0.009829
Ranker: the white street lamp, score=0.5713, area=0.047669
```

```text
Query: First white planter ... in front of the utility building
Top-1: first white planter, score=0.3819, area=0.009341
Ranker: the utility building, score=0.2106, area=0.098194
```

Ranker 放弃较高置信度的目标小框，改选更大的参照物，直接违背 referring expression
的目标角色。

## 8. 根因三：序数训练配额未匹配 AIC

| 类别 | train | holdout | AIC |
|---|---:|---:|---:|
| spatial | 31.90% | 32.65% | 30.31% |
| ordinal | 6.96% | 6.10% | 22.09% |
| depth | 18.50% | 18.56% | 17.99% |
| other | 18.24% | 18.98% | 10.48% |
| attribute | 13.30% | 13.17% | 9.75% |

50,000 条训练计划需要 11,046 条 ordinal，RefCOCO 系列实际只能提供 3,480 条，
缺少 7,566 条。AIC 序数比例约为训练集的 3.17 倍，因此“按 AIC 类别匹配”的计划
在源数据供给约束下没有真正实现。

## 9. 根因四：无保护阈值导致过度切换

冻结模型使用 `guard_margin=0.0`，只要 Ranker 分数略高就替换 Top-1。AIC 改选候选
相对 Top-1 的 GroundingDINO score：

- 平均下降 0.2212；
- 中位数下降 0.1958；
- 91.34% 的新旧框 IoU 小于 0.5；
- 57.04% 的切换跨越不同 label。

目标域上缺少置信度、语义角色和面积倍率保护，使错误先验可以影响近一半测试集。

## 10. 对两轮策略的最终判断

### 被证实可保留

- GroundingDINO 可继续作为候选生成器研究；S02 与 Florence 仅差 0.42 pp；
- Tile 对外部小目标候选召回有效，但应选择性启用；
- 外部候选缓存、split 隔离、提交审计和可恢复训练工程有效；
- 规则 Ranker 的低切换策略比无保护全量切换更符合目标域安全要求。

### 被平台证伪

- 不能把 RefCOCO Top-10 oracle 当作 AIC Top-10 oracle；
- 不能依据外部 holdout +11.22 pp 宣称 Ranker 预计提升 AIC；
- 不能在 AIC 上使用当前 `guard_margin=0.0` 的 Spatial LTR；
- 不能让绝对面积成为主导特征；
- 不能只用 token overlap 区分复杂目标、部件和参照物。

## 11. 下一轮最低风险方案

从 S02 候选缓存出发，只在全部保护条件满足时切换：

1. 新候选必须更像目标短语，不能更像参照物；
2. 默认只允许相同或高度兼容 label；
3. GroundingDINO score 降幅不超过 0.05 或 0.10；
4. 面积不得超过 Top-1 的 2 倍，除非 Query 明确要求 largest、group 或 region；
5. front/behind/nearest/farthest 不伪造成二维位置关系；
6. 提高 guard margin，并在外部 AIC-like 验证上单独校准；
7. 将参照物候选加入 hard negative，移除或强正则化 `area/log_area`；
8. 补充长关系句、小目标和序数表达的合规外部训练数据。

下一次平台提交应是单变量的保守 S04，而不是继续扩大当前 LightGBM 训练规模。

## 12. 证据与可复现入口

- 第一轮报告：`reports/diagnostic_oracle_tile_gdino_round.md`；
- 第二轮报告：`reports/gdino_spatial_ltr_v1_training_report.md`；
- 机器摘要：`reports/gdino_spatial_ltr_v1_summary.json`；
- 排行榜记录：`reports/leaderboard_results.md`；
- 公开配置：`configs/gdino_spatial_ltr_v1.example.yaml`；
- 统一训练入口：`tools/run_ranker_training_round.py`；
- 完整候选缓存、权重、数据集、提交 JSON/ZIP 和本机路径均由 `.gitignore` 排除。

平台分数来自用户提供的排行榜截图；平台未提供逐 Query GT，因此不能计算 AIC
按类别 rescue/harm 或真实 Top-10 oracle。本文对根因的判断来自平台总分差异与候选行为
审计，不把未知的逐样本正确性写成已确认事实。
