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
