# AIC RGB–TIR Phase 1.7A+ 无训练证据审计

> AS_OF: 2026-08-13
> 本轮不训练、不使用 AIC 测试集、不生成提交。

## 1. 执行结论

- 状态：`PHASE_17A_PLUS_ASSET_BLOCKED`
- 唯一后续分支：`ASSET_BLOCKED`
- 原因：C1/C2 adapters, summaries, or the shared Teacher Bank are unavailable; model-derived drift cannot be recomputed.
- 已画像 repair-dev：`1024` 条。

## 2. 资产门禁

- 资产状态：`ASSETS_MISSING`
- 缺失资产：`c1_adapter, c1_summary, c2_adapter, c2_summary, teacher_bank`

缺失 C1/C2 与 Teacher Bank 时，不能计算逐层绝对对齐、Teacher margin 或模型空间 top-20 假负例；输入画像不会被冒充为模型诊断。

## 3. 已完成的输入事实与代理

- repair-dev RGB 全图与 ROI 亮度、对比度、模糊度、熵；
- TIR 边界连通黑边有效视场比例；
- source、illumination、weather、size、occlusion 分组；
- 固定 256 跨图负例的 target-head、superclass 与 Query 词集相似代理。

潜在假负例代理率：`0.0850`。它是词法启发式代理，不是模型空间已确认假负例率。

## 4. 尚未完成的核心问题

1. C2 的最差层负百分比是否由小 Base loss 分母放大；
2. 真实漂移是否与低质量 RGB 显著相关；
3. Teacher 相似度 top-20 中是否存在模型空间假负例；
4. 哪个风险簇造成 C2 的实际层级漂移。

这些问题必须恢复原始 C1/C2 Adapter 与共同 Teacher Bank 后复算，不能从聚合报告反推。

## 5. 下一操作

从 Phase 1.6 云端持久盘或归档回收 `probe_candidates/C1`、`probe_candidates/C2` 和 `rgb_teacher_bank/teacher_bank.pt`，保持原 fingerprint；不得重训或改变 repair-dev。

预期云端根目录：`/home/featurize/aic_cloud/outputs/aic_rgbtir_phase16_v1/`。建议只打包上述目录以及 `run_summary.json`，回传到本地现有 Phase 1.6 输出根目录。资产到齐后进入 Phase 1.7A+ 的模型特征复算切片。

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
- `worst_cases.csv`
