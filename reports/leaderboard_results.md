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
