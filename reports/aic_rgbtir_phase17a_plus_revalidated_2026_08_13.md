# AIC RGB–TIR Phase 1.7A+ 无训练证据审计

> AS_OF: 2026-08-13
> 本轮不训练、不使用 AIC 测试集、不生成提交。

## 1. 执行结论

- 状态：`PHASE_17A_PLUS_COMPLETE`
- 唯一后续分支：`D1_RETENTION`
- 原因：Material drift is present but is not sufficiently associated with RGB quality.
- 已画像 repair-dev：`1024` 条。

## 2. 资产门禁

- 资产状态：`ASSETS_READY`
- 缺失资产：`无`

缺失 C1/C2 与 Teacher Bank 时，不能计算逐层绝对对齐、Teacher margin 或模型空间 top-20 假负例；输入画像不会被冒充为模型诊断。

## 3. 已完成的输入事实与代理

- repair-dev RGB 全图与 ROI 亮度、对比度、模糊度、熵；
- TIR 边界连通黑边有效视场比例；
- source、illumination、weather、size、occlusion 分组；
- 固定 256 跨图负例的 target-head、superclass 与 Query 词集相似代理。

潜在假负例代理率：`0.0850`。它是词法启发式代理，不是模型空间已确认假负例率。

## 4. 已完成的模型证据

- Layer 8 绝对 alignment delta：`0.1289`；
- Layer 16 绝对 alignment delta：`0.1526`；
- Layer 24 绝对 alignment delta：`0.3838`；
- RGB 质量风险与漂移 Spearman 关联：`0.1500`；
- 模型 top-neighbor 潜在假负例代理率：`0.1901`。

绝对逐层指标确认 C2 存在真实漂移；质量关联和假负例指标属于诊断代理，不是因果结论或人工真值。

## 5. 下一操作

只进入 `D1_RETENTION`。保持 C2 训练口径，仅新增预注册的 per-layer Base-TIR retention；先做 repair probe，通过门禁后才允许 full train。

当前仍不能确认 Query grounding、bbox ACC 或 AIC 平台收益；本轮没有训练、没有使用 AIC 测试数据。

## 6. 产物

- `asset_preflight.json`
- `decision.json`
- `layerwise_metrics.json`
- `negative_sampling_audit.json`
- `per_record_metrics.jsonl`
- `rgb_teacher_quality.csv`
- `run_summary.json`
- `sha256_manifest.json`
- `subgroup_metrics.csv`
- `teacher_reliability.json`
- `worst_cases.csv`
