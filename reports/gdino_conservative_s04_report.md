# AIC S04 GroundingDINO 保守排序恢复实验报告

## 1. 执行结论

本轮已完成一个严格单变量、可手动上传平台的恢复性实验：保持 S02 GroundingDINO Top-1 为默认输出，仅在非 Top-1 候选同时通过全部安全门时允许 LightGBM Ranker 改选。

- 平台上传包：`outputs/gdino_conservative_s04/platform_upload_ready/S04_gdino_conservative_ltr_v1.zip`
- ZIP SHA-256：`002B2399AA14B66695C3DABE162FF63186519CD3E81E99ECCEC59CBC26FF2831`
- AIC 记录数：9,555
- 相对 S02 实际改框：164 条（1.7164%）
- 自动上传：未执行；必须由用户手动上传。

这只能证明策略在外部 RefCOCO 系列隔离数据上更安全，不能证明 AIC 平台一定提升。真实收益必须由 `S04 - 0.4938` 计算。

## 2. 为什么要做恢复性 S04

已知平台结果为：Florence S01 `0.4980`、GroundingDINO Top-1 S02 `0.4938`、无保护 LightGBM Ranker S03 `0.3190`。S03 在 AIC 上大幅负迁移，主要风险是切换过多、训练域面积先验、目标/参照物反转和跨标签改选。

S04 没有重新训练任何模型，也没有修改候选生成。它只缩小 Ranker 的决策权限，因此是对 S02 的保守增量，而不是新的多模态模型。

## 3. 冻结输入与可追溯性

- AIC 候选缓存 SHA-256：`E87C5128A8E3005E501C3626D2366B5F9861F0632AA2B010AB076E21B779B225`
- LightGBM 文本模型 SHA-256：`41F979CC8044B329B9DB98266AF191B28C6DF6F9400AF3BDB8CD281473F336A1`
- AIC Queries SHA-256：`2A08CD3A930D9749EDB86A8B420D8259DCE446116AD98D93338EDF8B3E1EF67C`
- S02 JSON SHA-256：`B813042F3E70E559DF645FEB8839BBCBC09E79A4CDE54A2581C11804433CEB38`
- 本轮代码闭包 SHA-256：`630EAB6F4BF8EE8EC0702FF9B031018D6833A6C81B0DE62D8AA3C1896AB199E6`

候选模型固定为 `IDEA-Research/grounding-dino-tiny`，revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`，float32、box/text threshold 均为 0.15、Top-K=10。没有重跑候选，也没有使用 AIC 图像、伪标签或人工 bbox 训练。

## 4. 保守安全门

候选按 Ranker 分数降序检查，只有首个同时满足下列条件的候选才可替换 Top-1：

1. Ranker 相对 Top-1 margin ≥ `0.5`；
2. GroundingDINO score drop ≤ `0.1`；
3. canonical label 非空且与 Top-1 完全相同；
4. reference overlap 不得高于 target overlap；
5. 面积不得超过 Top-1 的 `1.5×`；
6. parser 或 query category 判为 depth 时禁止切换；
7. bbox 必须有限、合法且位于 `[0,1]`；
8. 任一字段异常时保持 Top-1。

没有启用 largest/group/region 面积例外，也没有加入 IR、Depth、Tile、PIZA、APE 或新候选模型。

## 5. 外部 validation 结果

| 指标 | 数值 |
|---|---:|
| 可评估 Query | 9,994 |
| Top-1 ACC@0.5 | 0.536822 |
| S04 ACC@0.5 | 0.550130 |
| 增益 | +1.3308 pp |
| 切换 | 284 |
| Rescue / Harm | 161 / 28 |
| Rescue/Harm | 5.7500 |

实施计划摘要中的先期只读回放写为 validation `0.55093 / 170/29`、holdout `0.53669 / 186/36`，但这些数字不能由最终书面安全门逐字复现。最终规格额外明确了 `query_category=depth` 也必须禁切换，并要求异常字段回退 Top-1；本次实现严格遵循这些最终门限，得到本报告两张表中的实际数值且通过计划中独立列出的验收下限。该差异作为规格内历史草案与最终实现的证据保留，不能把先期数字冒充为本次实际结果。

## 6. 外部 holdout 单次结果

| 指标 | 数值 |
|---|---:|
| 可评估 Query | 9,989 |
| Top-1 ACC@0.5 | 0.521674 |
| S04 ACC@0.5 | 0.536190 |
| 增益 | +1.4516 pp |
| 切换 | 297 |
| Rescue / Harm | 180 / 35 |
| Rescue/Harm | 5.1429 |

本轮没有使用 holdout 搜索阈值，最终产物只采用预先冻结的唯一策略。代码审查期间为定位“先期回放表与最终书面安全门不一致”额外执行过只读语义对照；这些对照没有改变任何阈值、模型或最终策略。因此 holdout 没有参与模型选择，但“物理上仅执行一次”的理想过程已不能严格宣称，特在此保留审计边界。

## 7. AIC 无标签审计

| 审计项 | 结果 |
|---|---:|
| 总记录 | 9,555 |
| 相对 S02 改框 | 164 |
| 切换率 | 1.7164% |
| Depth 切换 | 0 |
| 跨 canonical label | 0 |
| reference-dominant | 0 |
| 最大面积倍率 | 1.486651× |
| 非法框 | 0 |
| 非 bbox 字段变化 | 0 |
| S02 Top-1 不一致 | 0 |

切换类别分布：`{"action": 5, "attribute": 3, "ordinal": 94, "other": 12, "plural_group": 3, "spatial": 47}`。

## 8. 平台上传与结果解释

只上传 `S04_gdino_conservative_ltr_v1.zip`，不要上传目录、模型文本或调试 JSONL。平台上传后记录提交时间、提交 ID 和 ACC@0.5：

```text
保守 Ranker 真实收益 = S04 - 0.4938
相对 Florence 收益   = S04 - 0.4980
```

- `S04 > 0.4980`：将 S04 作为新稳定基线，下一轮做视觉语义 selector 与 PIZA 小目标分支；
- `0.4938 < S04 <= 0.4980`：安全 Ranker 有迁移价值，但 Florence 仍是最佳单提交；
- `S04 <= 0.4938`：停止当前 LightGBM 主线，下一次使用新单模型。

## 9. 结论边界

S04 是恢复性排序策略，不是 GroundingDINO 微调模型，也没有使用 RGB 以外模态。外部分数来自 RefCOCO 系列，平台结果未知；不得把外部增益写成 AIC 已提升。
