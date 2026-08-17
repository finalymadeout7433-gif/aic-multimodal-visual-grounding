# AIC RGB–TIR Phase 1.8R-Audit 本地复核计划

## 定位

本轮只使用已返回本地的 Phase 1.8R 资产，不训练、不租 GPU、不打开 AIC 测试集，也不改变 Adapter 权重。目标是区分模型表征问题与验证门禁问题，最终只输出 `GATE_FIX`、`MODEL_FIX` 或 `BOTH_REQUIRED`。

## 固定输入

- Phase 1.8R 最终归档及 receipt；
- official-val `embedding_cache.pt`；
- `per_record_metrics.jsonl`、分组、检索、坍缩与安全结果；
- selected rank-48 TIR Adapter；
- 训练 checkpoint 与 Teacher Bank 继续封存在原始 ZIP 中，不为本轮重复复制。

## 审计边界

1. 安全契约：推理模式、冻结状态、Adapter 张量、有限值、RGB 哈希与无第二模型 fallback；
2. 表征口径：记录级与唯一图像对聚合级的有效秩、participation ratio 和谱能量；
3. 检索口径：严格记录正例、同图像对多正例、唯一图像对聚合三套指标；
4. 分组口径：绝对漂移、全局漂移、ExcessDrift，以及按唯一图像对 bootstrap 的 95% 区间；
5. 决策边界：只有统计/冻结契约问题为 `GATE_FIX`，真实表征问题为 `MODEL_FIX`，两者并存为 `BOTH_REQUIRED`。

## 不在本轮完成

- Query–TIR probe；
- RGB–TIR 残差融合或 UniRGB-IR SFI；
- Adapter 重训、D2 谱保持修复；
- 8B/30B 平台推理；
- 配准与 Depth。

## 退出条件

- 2,032 条记录和 1,115 个图像对完整读取；
- 所有审计产物可重复生成且 JSON/CSV 无非有限值；
- 原始缓存与 Adapter 不被修改；
- 报告明确给出三选一决策及下一轮唯一主分支。
