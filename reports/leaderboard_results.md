# 排行榜结果

## Florence-2 RGB-only v0.2.0

| 字段 | 结果 |
|---|---|
| 模型 | `microsoft/Florence-2-large-ft` 的本地开源权重 |
| 输入 | Visible RGB + 原始英文 Query |
| 训练 | 无；零样本推理 |
| 候选选择 | Florence phrase grounding 返回的第一个候选 |
| 正式 Query | 9,555 |
| fallback | 0 |
| 平台 ACC@0.5 | **0.4980** |
| 当时排名 | 36 |
| 平台记录时间 | 2026-07-29 18:52:09 |
| 提交标识 | `AIC-2026-34835309` |

该分数来自用户提供的平台结果截图。它是第一版可复现 RGB-only 工程基线，不代表
后续多尺度、候选重排、Query 规范化或多模态方案的上限。

## GroundingDINO 与 Spatial LTR 平台对照

| 提交 | 输入与选择策略 | 平台 ACC@0.5 | 相对 Florence | 平台记录时间 |
|---|---|---:|---:|---|
| S02 | GroundingDINO-Tiny RGB + Query，原始 Top-1 | **0.4938** | -0.42 pp | 2026-08-01 01:09:49 |
| S03 | 同一 Top-10 候选 + LightGBM Spatial LTR | **0.3190** | -17.90 pp | 2026-07-31 23:52:51 |

两项分数均来自用户提供的平台结果截图。S02 与 Florence 基本持平；S03 相对 S02
下降 17.48 个百分点。S03 在 9,555 条 Query 中改选了 4,770 条，按平台显示精度
近似换算，改选造成约 1,670 条净正确预测损失。

本地 RefCOCO 系列 holdout 上，Spatial LTR 曾将 ACC@0.5 从 0.5211 提高到
0.6333，但该增益没有迁移到 AIC。逐候选审计表明，AIC 改选框的面积中位数是
Top-1 的 5.67 倍，并出现明显的目标到参照物翻转。该结果现被定性为目标域负迁移，
不能再把外部 holdout 增益表述为预期 AIC 增益。完整证据见
[Spatial LTR 平台复盘](gdino_spatial_ltr_v1_platform_postmortem.md)。

## S04 保守 Ranker

| 提交 | 输入与选择策略 | 平台 ACC@0.5 | 相对 Florence | 平台记录时间 |
|---|---|---:|---:|---|
| S04 | S02 Top-1 + 同标签、分数降幅、面积倍率、参照物与 Depth 安全门 | **0.4957** | -0.23 pp | 2026-08-01 13:12:35 |

S04 只在 9,555 条中切换 164 条，避免了 S03 的大规模负迁移，但仍未超过
Florence-2。该结果证明保守门控能控制伤害，不证明现有 LightGBM Ranker 已成为
更优平台基线。

## 三个开源模型零训练全量对照

三份提交均使用 Visible RGB + 原始英文 Query，处理 9,555 条 Query，采用模型
原生最高分 Top-1；没有训练、候选重排、人工规则、IR/Depth 融合或 AIC 伪标签。

| 模型 | 平台 ACC@0.5 | 相对 Florence | 相对 MM-GDINO-T | 平台记录时间 |
|---|---:|---:|---:|---|
| MM-Grounding-DINO-T | **0.5011** | **+0.31 pp** | — | 2026-08-02 16:07:02 |
| LLMDet-Swin-T | **0.4942** | -0.38 pp | -0.69 pp | 2026-08-02 14:44:19 |
| APE-Ti | **0.4424** | -5.56 pp | -5.87 pp | 2026-08-02 13:29:14 |

分数均来自用户提供的平台截图。2026-08-02 最初曾因用户口头表述将 `0.4942`
误归给 MM-Grounding-DINO-T；现已纠正为：

- `0.4942` 属于 **LLMDet-Swin-T**；
- `0.5011` 属于 **MM-Grounding-DINO-T**。

三份推理输出、模型指纹和 ZIP 文件始终按各自模型目录独立保存，错误仅发生在
平台截图对应模型的文字描述，不涉及提交包、预测文件或权重身份混淆。

按平台显示的四位小数近似折算，MM-Grounding-DINO-T 比 Florence-2 多约 30 条
正确预测，比 LLMDet-Swin-T 多约 66 条。由于平台分数存在显示舍入，这些数量只作
量级解释，不作为逐样本真值。

当前平台结论：

1. **MM-Grounding-DINO-T（0.5011）成为当前最高分单模型控制基线。**
2. LLMDet-Swin-T 未超过 Florence，也没有显示出比其 MM-GDINO 基座更好的 AIC
   迁移；在扩大到 Swin-B/L 前应先证明独有收益。
3. APE-Ti 的区域/stuff 能力没有转化为 AIC 总分，不再作为默认主线。
4. 当前三个模型均为 RGB-only，结果不能用于否定 Infrared 或 Depth 的潜在价值。

完整工程与模型资料见
[三模型零训练全量报告](aic_three_model_zero_shot_full_report.md)。
