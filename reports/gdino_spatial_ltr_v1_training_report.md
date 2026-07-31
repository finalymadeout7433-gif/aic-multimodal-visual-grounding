# AIC 赛题一：GroundingDINO Top-10 + 空间关系感知排序器训练报告

> 报告定位：这是给后续 ChatGPT 网页端或其他协作者直接阅读的独立交接文件。文中严格区分平台事实、外部本地验证、工程 sanity check 与尚未完成项。

## 1. 执行摘要

本轮训练的是监督式候选排序器，不是 GroundingDINO 本体，也不是强化学习。输入仍为 RGB 与原始英文 Query；GroundingDINO-Tiny 生成 Top-10，LightGBM LambdaRank 根据模型分数、候选几何、目标/参照物词语匹配、绝对位置、相对关系、面积和序数等特征选择最终 bbox。

已知平台事实只有：

- Florence-2-large-ft RGB-only、first-candidate：AIC 平台 ACC@0.5 = **0.4980**。
- S02 与 S03 的平台成绩仍需用户手动上传后回填，不能由本地 RefCOCO 分数代替。

本轮是否达到预设晋级线：

- Holdout 相对 GroundingDINO Top-1 增益：+11.22 pp。
- 本地晋级结论：通过全部预设本地晋级条件；平台是否提升仍必须由 S02/S03 实测。

- Holdout 整体增益：+11.22 pp；
- spatial + ordinal 合并增益：+16.23 pp；
- 最差单数据集增益：+7.95 pp；
- AIC 合法框率：100.00%；
- 静默 fallback / 数值异常：0 / 0。

## 2. 赛题背景与本轮边界

AIC 任务要求依据 Visible RGB、Infrared、Depth 与英文 Query，在 Visible 图上输出归一化 `[x1,y1,x2,y2]`，指标为 ACC@0.5。正式初赛 9,555 条 Query 没有 bbox，是测试集，只用于推理。

本轮只使用：

- RefCOCO、RefCOCO+、RefCOCOg 的外部 train/validation/holdout；
- Visible RGB 和原始 Query；
- 固定 GroundingDINO-Tiny Top-10；
- LightGBM 4.6.0 LambdaRank。

本轮没有使用：

- AIC 测试集训练、伪标签或人工框；
- IR、Depth、Tile、Query 人工改写；
- Florence 候选融合、CLIP crop 特征；
- GroundingDINO 参数更新。

## 3. 为什么先训练 Ranker，而不是先微调 GroundingDINO

诊断轮的受控外部结果为：

- GroundingDINO Top-1：0.5627；
- GroundingDINO Top-10 oracle：0.9073；
- Florence first：0.6893；
- Florence oracle：0.7533。

这些数字不能直接预测 AIC 平台分数，但它们说明 GroundingDINO 的候选集合明显比当前 Top-1 排序更强。候选 oracle 与最终选择之间的巨大差距，正是 Learning-to-Rank 的直接监督空间。微调检测器会同时改变召回与定位，本轮先用小模块隔离“排序是否有效”这一变量。

## 4. 对 ChatGPT 网页端旧分析的修正

旧分析中有合理方向，也有需要明确纠正的判断：

1. **“60.27% 多候选”不能证明正确框在非首候选。** 多候选可能分别是主体、部件或参照物。只有带 GT 的 oracle 才能证明可救空间。
2. **Florence 没有可用的候选 confidence。** 因此 `0.4 * Florence confidence` 之类公式没有数据基础；本轮没有伪造 confidence。
3. **不能跨语义 label 直接做 CLIP crop 重排。** “man holding a phone” 中 phone crop 可能与整句很相似，却不是应输出的主体框。目标/参照物角色必须先区分。
4. **无校准分数时不应使用 WBF。** Florence Tile 的多个框没有可比较置信度，WBF 会人为制造一个缺乏统计依据的新框。
5. **Tile oracle 提升不等于最终 selected ACC 提升。** Tile 只说明候选召回上限；没有可靠选择器时不能直接声称平台会涨分。
6. **GroundingDINO 外部 Top-1 并不优于 Florence 外部 first。** 它的价值是 Top-K 候选生成，不是已经证明可直接替代 Florence。
7. **规则原型的 +2.6 pp 是受控外部证据，不是 AIC 平台增益。** 本轮两份只差排序器的提交，才用于测量真实目标域收益。

## 5. 固定子集、隔离与哈希

随机种子为 `20260731`。训练、dev、validation、holdout 均按 image key 隔离；holdout 只在模型与保护阈值冻结后生成候选和评测。

| 子集 | Query | 唯一图像 | 单图上限 | 清单 SHA-256 |
|---|---:|---:|---:|---|
| pilot | 20000 | 13137 | 2 | `F63C73FD14C4DF6CC15FAAB6BCFFE928518F2461236EE2F40BBD703C99334410` |
| full | 50000 | 20224 | 3 | `BBC86574DD5FFF71098893269283B02169DC4FF46D1A4B065EFE43C674CCD310` |
| validation | 10000 | 2567 | 5 | `9702ED7D695FF46D177F768C2FA974047E42D48B137B4871A22BA7C265740154` |
| holdout | 10000 | 3346 | 4 | `37C322CA3189EAF4D42A95A3AC8A11AA85810F81B52E8CD6C0E52B7455DC10B6` |

由于 RefCOCO 系列中 ordinal 等类型的自然供给不足，实际类别分布不能完全复制 AIC 比例。报告保留每类请求配额与实际偏差，不能写成“完美匹配”。

## 6. 100 条真实模型冒烟

- 候选记录：100；无候选：0；Top-10 oracle：90.00%；峰值显存：1.98 GiB。

该 100 条只验证缓存、坐标、IoU、可恢复写入和显存，不用于模型结论。

## 7. 空间词强化的实现

每条 Query 是一个排序 group；候选相关性标签由 IoU 转为四级：

```text
0: IoU < 0.20
1: 0.20 <= IoU < 0.50
2: 0.50 <= IoU < 0.70
3: IoU >= 0.70
label_gain = [0, 1, 4, 5]
```

特征覆盖：

- GroundingDINO score、原始 rank、与第一名/下一名分差；
- bbox 中心、宽高、面积、长宽比、四边距离；
- 全候选与目标兼容候选内部的左右/上下/面积排名；
- 候选 label 与完整 Query、目标短语、参照物短语的 token overlap；
- 主体/参照物 overlap 及二者差值（本轮**没有**独立的部件角色特征）；
- `leftmost/rightmost/topmost/bottommost/center`；
- `largest/smallest`；
- `first/second/third... from left/right/top/bottom`；
- `left of/right of/above/below/beside/inside/between` 的候选对关系；
- `nearest/farthest/front/behind` 仅记录文本标志，不伪造二维深度。

例如 `the man left of the bus` 被拆为 target=`man`、relation=`left_of`、reference=`bus`，候选 man 的几何关系相对 bus 候选计算，而不是把 “left” 粗暴作用于全部框。

实现边界需要明确：原计划提到主体、部件、参照物三类角色，但冻结模型只显式建模了主体与参照物，部件候选尚未独立识别。为保持 holdout 单次评测纪律，本轮审查后没有改变特征矩阵或重训；部件角色应作为下一轮独立消融。

## 8. 20k Pilot 与 50k 最终训练

Pilot 选择：grid 3，`num_leaves=63`，`min_child_samples=100`，best iteration=215，guard margin=0.0。

最终完整 Ranker 使用 42260 个有排序信号的 Query group、210446 个候选行；固定 215 轮。

### 主要特征重要性

| 排名 | 特征 | Gain | 占比 |
|---:|---|---:|---:|
| 1 | `area` | 144031.79 | 23.58% |
| 2 | `gdino_score` | 52732.06 | 8.63% |
| 3 | `log_area` | 42462.90 | 6.95% |
| 4 | `score_gap_to_first` | 37273.69 | 6.10% |
| 5 | `max_contains_other` | 24061.53 | 3.94% |
| 6 | `cx` | 21105.68 | 3.46% |
| 7 | `width` | 18677.73 | 3.06% |
| 8 | `edge_left` | 17168.19 | 2.81% |
| 9 | `absolute_relation_satisfaction` | 16167.41 | 2.65% |
| 10 | `edge_right` | 14996.17 | 2.46% |
| 11 | `ordinal_distance` | 14826.74 | 2.43% |
| 12 | `max_iou_other` | 12611.25 | 2.06% |
| 13 | `score_gap_to_next` | 12551.23 | 2.06% |
| 14 | `ordinal_satisfaction` | 10636.02 | 1.74% |
| 15 | `same_label_count` | 9421.41 | 1.54% |
| 16 | `mean_iou_other` | 9057.79 | 1.48% |
| 17 | `height` | 8546.57 | 1.40% |
| 18 | `flag_ordinal_relation` | 8512.07 | 1.39% |
| 19 | `target_compatible` | 8444.29 | 1.38% |
| 20 | `max_containment_in_other` | 8304.60 | 1.36% |

`area` 与 `log_area` 在增益重要性中占比较高，说明模型部分依赖 RefCOCO 的框尺度先验。它在外部 holdout 上有效，但可能遇到 AIC 极小目标域偏移；因此不能只凭本地提升推断平台一定涨分。

## 9. Validation 10k 消融

| 方法 | ACC@0.5 | 相对 Top-1 | Mean IoU | Switch | Rescue | Harm |
|---|---:|---:|---:|---:|---:|---:|
| GroundingDINO Top-1 | 53.65% | +0.00 pp | 0.5357 | 0.00% | 0 | 0 |
| 保守空间规则 | 55.41% | +1.76 pp | 0.5499 | 4.64% | 214 | 38 |
| Score + Geometry Ranker | 61.18% | +7.53 pp | 0.5914 | 32.07% | 1390 | 637 |
| 完整 Ranker（无保护阈值） | 64.78% | +11.13 pp | 0.6219 | 33.92% | 1653 | 540 |
| 完整 Ranker（冻结阈值） | 64.78% | +11.13 pp | 0.6219 | 33.92% | 1653 | 540 |

### 按数据集

| 数据集 | Query | Top-1 ACC | Ranker ACC | 增益 |
|---|---:|---:|---:|---:|
| refcoco | 3862 | 51.06% | 63.13% | +12.07 pp |
| refcoco_plus | 2748 | 49.31% | 58.22% | +8.92 pp |
| refcocog | 3390 | 60.12% | 71.98% | +11.86 pp |

### 按 Query 类型

| Query 类型 | Query | Top-1 ACC | Ranker ACC | 增益 |
|---|---:|---:|---:|---:|
| action | 963 | 63.76% | 71.55% | +7.79 pp |
| attribute | 1564 | 64.90% | 75.26% | +10.36 pp |
| depth | 1751 | 47.74% | 58.99% | +11.25 pp |
| ordinal | 414 | 27.29% | 56.52% | +29.23 pp |
| other | 2185 | 64.71% | 70.39% | +5.68 pp |
| plural_group | 254 | 38.19% | 56.30% | +18.11 pp |
| spatial | 2869 | 44.48% | 58.00% | +13.52 pp |

### 排序边际分桶

| Ranker 分差桶 | Query | Selected ACC@0.5 |
|---|---:|---:|
| `lt_0.05` | 6827 | 68.49% |
| `0.05_to_0.10` | 202 | 50.50% |
| `0.10_to_0.20` | 389 | 48.07% |
| `0.20_to_0.30` | 327 | 52.60% |
| `ge_0.30` | 2249 | 59.63% |

Validation Top-10 不可救样本：1064 条；这些样本无法由任何只做 Top-10 重排的模型修复。

## 10. Holdout 10k 单次评测

| 方法 | ACC@0.5 | 相对 Top-1 | Mean IoU | Switch | Rescue | Harm |
|---|---:|---:|---:|---:|---:|---:|
| GroundingDINO Top-1 | 52.11% | +0.00 pp | 0.5224 | 0.00% | 0 | 0 |
| 保守空间规则 | 53.69% | +1.58 pp | 0.5366 | 5.19% | 205 | 47 |
| Score + Geometry Ranker | 60.06% | +7.95 pp | 0.5817 | 33.06% | 1463 | 668 |
| 完整 Ranker（无保护阈值） | 63.33% | +11.22 pp | 0.6096 | 35.18% | 1740 | 618 |
| 完整 Ranker（冻结阈值） | 63.33% | +11.22 pp | 0.6096 | 35.18% | 1740 | 618 |

### 按数据集

| 数据集 | Query | Top-1 ACC | Ranker ACC | 增益 |
|---|---:|---:|---:|---:|
| refcoco | 4446 | 49.96% | 62.12% | +12.17 pp |
| refcoco_plus | 2704 | 51.81% | 59.76% | +7.95 pp |
| refcocog | 2850 | 55.75% | 68.60% | +12.84 pp |

### 按 Query 类型

| Query 类型 | Query | Top-1 ACC | Ranker ACC | 增益 |
|---|---:|---:|---:|---:|
| action | 828 | 62.80% | 73.19% | +10.39 pp |
| attribute | 1317 | 64.54% | 72.44% | +7.90 pp |
| depth | 1856 | 50.65% | 62.12% | +11.48 pp |
| ordinal | 610 | 27.21% | 55.57% | +28.36 pp |
| other | 1898 | 63.65% | 67.86% | +4.21 pp |
| plural_group | 226 | 46.46% | 50.88% | +4.42 pp |
| spatial | 3265 | 43.55% | 57.52% | +13.97 pp |

### 排序边际分桶

| Ranker 分差桶 | Query | Selected ACC@0.5 |
|---|---:|---:|
| `lt_0.05` | 6710 | 67.15% |
| `0.05_to_0.10` | 233 | 45.49% |
| `0.10_to_0.20` | 390 | 43.85% |
| `0.20_to_0.30` | 317 | 51.10% |
| `ge_0.30` | 2339 | 59.34% |

Holdout Top-10 不可救样本：1064 条。

Holdout 仅允许单次检查；文件已存在时统一入口拒绝再次覆盖评测。本轮代码审查后又增加了模型 SHA、特征 schema、训练/候选配置、validation cache fingerprint 与 holdout evaluation SHA 的绑定门禁；既有 holdout 结果通过原始完成清单回绑，不重新计算指标。

## 11. AIC 9,555 条推理与合法性

- 候选缓存：9555/9555；无候选：0；本轮耗时 4198.0 秒；峰值显存 1.97 GiB。

系统异常、NaN、特征错误不会被中心框掩盖。只有 GroundingDINO 真正返回零候选时，两份提交才共同使用已存在的 Florence 预测，保证 S03−S02 只测排序器选择差异。

## 12. 两份平台提交

- S02 JSON：`outputs/gdino_spatial_ltr_v1/submissions/S02_gdino_top1_control/predictions_submission.json`
- S02 ZIP：`outputs/gdino_spatial_ltr_v1/submissions/S02_gdino_top1_control/predictions_submission.zip`
- S02 ZIP SHA-256：`99F06B050E5478212A26D233F365CCA29B7FF3EA44CF73E171F0EEE4F1B706E5`
- S03 JSON：`outputs/gdino_spatial_ltr_v1/submissions/S03_gdino_spatial_ltr_v1/predictions_submission.json`
- S03 ZIP：`outputs/gdino_spatial_ltr_v1/submissions/S03_gdino_spatial_ltr_v1/predictions_submission.zip`
- S03 ZIP SHA-256：`23819AE2728B7502BE53384C2F5F906EB7F7ED45ACF95EF74E92D10FB4A702CF`
- Ranker 切换率：49.92%
- 共享 Florence 零候选兜底：0 条；两份提交对这些样本使用完全相同的框。

建议用户严格按顺序手动上传：

1. S02 GroundingDINO 原始 Top-1；
2. S03 GroundingDINO + 空间 LTR。

## 13. 平台分数回填

| 提交 | 平台分数 | 提交 ID | 时间 |
|---|---:|---|---|
| Florence S01 | 0.4980 | 待补 | 待补 |
| GroundingDINO S02 | 待用户上传 | 待补 | 待补 |
| GDINO + Ranker S03 | 待用户上传 | 待补 | 待补 |

回填后计算：

```text
Ranker 训练真实收益 = S03 - S02
GroundingDINO 基座变化 = S02 - 0.4980
相对首版系统收益 = S03 - 0.4980
```

## 14. 下一轮决策规则

- 若 S03−S02 明显为正，先做 selective tile 或 crop/global-context embedding 消融；
- 若 spatial/ordinal 本地提升明显、平台提升有限，优先检查 AIC 目标角色和 Query 翻译域偏移；
- 若 S02/S03 均低于 Florence 0.4980，可尝试 Florence/GDINO 候选联合排序，而不是立即微调 GDINO；
- 若 Top-10 recall failure 仍集中在小目标，再做 selective tile；
- Depth late fusion 只服务 nearest/farthest/front/behind，PNG 与 JPG 深度域分开处理；
- 只有候选 oracle 仍不足时，才把 GroundingDINO 本体微调提升为主线。

## 15. 运行时间、异常与最终验证

| 阶段 | 状态 | 最近一次调用耗时（秒） |
|---|---|---:|
| `subsets` | completed | 14.0 |
| `cache-train` | completed | 7219.3 |
| `train-pilot` | completed | 19.2 |
| `cache-train-full` | completed | 11056.5 |
| `train-final` | completed | 46.5 |
| `cache-validation` | completed | 3685.8 |
| `evaluate-validation` | completed | 6.6 |
| `cache-holdout` | completed | 3785.3 |
| `evaluate-holdout` | completed | 0.3 |
| `cache-aic` | completed | 4201.2 |
| `submissions` | completed | 11.4 |
| `summary` | completed | 0.1 |

候选生成的真实 GPU 耗时以各 cache `summary.json` 为准；上表是监督器记录的最近一次阶段调用，其中 validation 重建冻结凭据、holdout provenance 回绑等审查操作不会重跑 GPU 推理或第二次打开 holdout 指标。

- 自动化测试：80 passed，耗时 15.95 秒；compileall：passed；提交独立审计：passed。

本轮最终阶段没有 CUDA OOM、NaN、RuntimeError 或静默 fallback。代码审查发现并修复了缓存依赖闭包、holdout 冻结身份绑定、配置哈希和报告口径问题；这些修复不改变已冻结模型、候选框、holdout 指标或两份提交 bbox。

## 16. 环境与可复现性

- Python：`3.11.15 | packaged by Anaconda, Inc. | (main, Mar 11 2026, 17:12:15) [MSC v.1942 64 bit (AMD64)]`；
- Torch：`2.5.1+cu121`；
- Transformers：`4.57.6`；
- LightGBM：`4.6.0`；
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`；
- 机器可读运行状态：`outputs/gdino_spatial_ltr_v1/run_summary.json`；
- 关键产物哈希：`outputs/gdino_spatial_ltr_v1/sha256_manifest.json`。
- 配置指纹：`outputs/gdino_spatial_ltr_v1/configuration/config_fingerprint.json`。
- Holdout 绑定凭据：`outputs/gdino_spatial_ltr_v1/evaluations/holdout/evaluation_provenance.json`。

完整候选 JSONL、模型权重、外部数据、AIC 数据和本机绝对路径均不进入 Git；Git 只保留代码、公开配置模板、小型摘要和本报告。
