# 模型与训练策略

## 已完成并得到平台分数的版本

`v0.2.0` 使用 `Florence-2-large-ft`：

- Visible RGB + 原始 Query；
- 不训练、不使用 AIC 正式数据伪标签；
- FP16、单条推理、确定性生成；
- 输出第一个 phrase-grounding 候选；
- 9,555 条全部生成合法 bbox，fallback 为 0；
- 平台 ACC@0.5：`0.4980`。

它是当前排行榜起点，不是下一阶段训练模型。

## 下一版可训练 RGB baseline

基座固定为 `IDEA-Research/grounding-dino-tiny`（GroundingDINO-T / Swin-T）。
选择理由：

- 原生输出候选 bbox，比生成式坐标更适合 ACC@0.5；
- 接收 image-text 对，任务形式与 AIC Visual Grounding 接近；
- Swin-T 比 Swin-B 更适合 RTX 4060 单卡起步；
- 官方项目采用 Apache-2.0，Transformers 已提供模型接口。

第一轮只使用 `rgb_core_v1_train.jsonl`，不加入 gRefCOCO、SUN-Spot、IR 或 Depth，
确保能与 Florence RGB-only 结果清晰比较。

## 两阶段训练

### Stage 1：稳定检测头

- 冻结视觉和文本 backbone；
- 训练 decoder、跨模态层和 bbox head；
- batch size 1，梯度累积 8；
- FP16、gradient checkpointing；
- 最大边 800；
- 1 epoch，学习率 `1e-4`。

### Stage 2：小学习率适配

- 解冻视觉 backbone 最后若干 stage 和融合层；
- 2 epochs，学习率 `1e-5`；
- 继续保留梯度裁剪与显存监控；
- 按 validation ACC@0.5 选择 checkpoint。

训练损失沿用 Grounding DINO 检测目标，包括分类/对齐、L1 bbox 与 GIoU。比赛比较
主指标统一使用 ACC@0.5，不用训练 loss 代替比赛指标。

## 数据边界

- train：模型参数更新；
- validation：阈值、学习率和 checkpoint 选择；
- holdout：方案冻结后的单次最终检查；
- AIC 9,555 条正式数据：只推理和提交，不训练、不伪标注；
- gRefCOCO：核心 RGB baseline 稳定后做复数/集合消融；
- SUN-Spot：第二阶段用于 Depth-aware reranker，不混入第一轮 RGB 结论。

公开配置模板为
`configs/groundingdino_rgb_core_v1.example.yaml`。本轮只完成训练数据和策略准备，
尚未声称 GroundingDINO 已经训练或取得平台提升。

## 官方实现依据

- Grounding DINO 官方仓库：https://github.com/IDEA-Research/GroundingDINO
- Transformers 模型文档：https://huggingface.co/docs/transformers/model_doc/grounding-dino
- 模型页：https://huggingface.co/IDEA-Research/grounding-dino-tiny
