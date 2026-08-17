# AIC RGB–TIR Phase 1.6 资产恢复验收与 Phase 1.8 计划

> 日期：2026-08-13
> 状态：`ASSET_RECOVERY_ACCEPTED_WITH_NOTES`
> 下一条唯一主线：`D1_RETENTION`

## 1. 这次核验解决了什么

之前的 Phase 1.7A+ 计划被 `ASSET_BLOCKED` 卡住，不是算法无法继续，而是本地只剩 manifest 和 preflight，缺少能够复现诊断的模型资产。

旧计划明确需要：

1. C1 Adapter 与 SHA-256；
2. C2 Adapter 与 SHA-256；
3. C0/C1/C2 candidate summaries；
4. 共同 RGB Teacher Bank 或其可验证 fingerprint；
5. 固定 repair-dev manifest 与 SHA-256；
6. Qwen revision 与 processor fingerprint；
7. Phase 1.6 完整训练/复算日志。

本轮重跑的目的不是训练最终融合模型，而是：

```text
重建共同 Teacher Bank
→ 从相同口径重跑 C1/C2 probe
→ 保存 Adapter、summary 和逐记录诊断
→ 执行 Phase 1.7A+ 无训练审计
→ 将五条候选路线收敛为一条有证据的后续路线
```

## 2. 已回传到本地的归档

- 云端：`/home/featurize/aic_cloud/platform_upload_ready/AIC_RGBT_Phase16R_C1_C2_Phase17APlus_20260813.tar.gz`
- 本地：`D:/12525/Documents/pytorch/baseline_v0/outputs/aic_rgbtir_phase16_recovery_returned_20260813/AIC_RGBT_Phase16R_C1_C2_Phase17APlus_20260813.tar.gz`
- 大小：`337,526,255` 字节
- SHA-256：`3AEAF519809D9FE7BE2818F7C36787667A10444422465FE4A98B8E8CBD37DECE`
- 云端与本地哈希：一致
- 解包目录：`D:/12525/Documents/pytorch/baseline_v0/outputs/aic_rgbtir_phase16_recovery_returned_20260813/extracted`

内部 Phase 1.6 的 21 个清单项和 Phase 1.7A+ 的 9 个清单项已逐文件复算，哈希失败数为 `0`。

## 3. 旧需求与恢复结果逐项对照

| 旧计划要求 | 状态 | 本地证据 |
|---|---|---|
| C1 Adapter + SHA-256 | 已恢复 | `extracted/outputs/aic_rgbtir_phase16_recovery_v1/probe_candidates/C1/adapter.pt`；`0D561490...A85EE` |
| C2 Adapter + SHA-256 | 已恢复 | `extracted/outputs/aic_rgbtir_phase16_recovery_v1/probe_candidates/C2/adapter.pt`；`28322253...3E786` |
| C1/C2 summaries | 已恢复 | 对应 `summary.json` 均存在且通过内部哈希 |
| C0 summary | 未重建 | 本轮恢复配置明确只重跑 C1/C2；属于历史完整性缺口 |
| Teacher Bank | 已恢复 | `teacher_bank.pt`，159,587,214 字节，SHA-256 `4A429B24...7DD` |
| Teacher Bank fingerprint | 已恢复 | `79CF814B64EC640AD3D924342FEEED235764F139445F834B2CDB2DD0B580C05C` |
| repair-dev manifest | 已恢复 | 1,024 条，SHA-256 `B9DFB8C4...D1D7` |
| Qwen revision | 配置合同已核实 | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| processor fingerprint | 配置合同已核实 | `preprocessor_config.json` SHA-256 `27225450...E516` |
| Phase 1.6 日志 | 已恢复 | `extracted/logs/aic_rgbtir_phase16_recovery_v1/03_c1_c2_recovery.log` |
| C1/C2 逐记录诊断 | 已恢复 | `recovery_diagnostics/C1.json` 与 `C2.json` |
| Phase 1.7A+ 机器结果 | 已恢复 | `decision.json`、`layerwise_metrics.json`、`per_record_metrics.jsonl` 等 |

说明：C0 未恢复不阻塞 `D1_RETENTION`。下一轮需要的核心对照是 Base TIR、C2、共同 Teacher Bank 与固定 repair-dev，四者均已具备。不得虚构 C0 文件；如将来需要完整复刻历史三候选表，再单独重建 C0。

## 4. 数据隔离与记录数

| 切分 | 记录数 | 唯一图像对 |
|---|---:|---:|
| repair probe train | 4,096 | 4,096 |
| repair dev | 1,024 | 1,024 |
| repair full train | 24,612 | 13,822 |
| official val | 2,032 | 1,115 |

重叠检查：`probe_dev=0`、`full_dev=0`、`train_official_val=0`。

## 5. 本轮重跑得到的算法结论

### 5.1 C1/C2 都不是“完全失败”

| 指标 | C1 | C2 |
|---|---:|---:|
| mean R@1 | 57.36% | **59.80%** |
| mean R@5 | 82.42% | **83.59%** |
| 最低有效秩 / Base TIR | 89.24% | **89.69%** |
| 最低有效秩 / RGB Teacher | 81.41% | **81.82%** |
| 最低 paired-shuffled margin | **0.1800** | 0.1698 |

C2 在检索和有效秩上优于 C1，仍是合理的后续基座。

### 5.2 硬门禁失败来自真实逐层对齐漂移

C2 相对 Base TIR 的平均绝对 alignment delta：

| 层 | delta mean |
|---|---:|
| Layer 8 | 0.1289 |
| Layer 16 | 0.1526 |
| Layer 24 | 0.3838 |

Layer 24 漂移最大。这个结果说明，C2 虽然提升跨图检索并保住有效秩，但中深层表示偏离 Base TIR 太多；不能直接把 C2 扩展为 full train。

### 5.3 为什么选择 D1_RETENTION

机器决策文件给出的证据：

- `real_alignment_drift=true`；
- RGB 质量风险与漂移的 Spearman 关联只有 `0.1500`，未达到质量加权路线阈值；
- 模型近邻中的潜在同类假负例代理率为 `0.1901`，没有越过当前 0.20 分支阈值；
- 因而不选择 `D1_QUALITY_WEIGHTED`、`C2_FALSE_NEGATIVE_AWARE` 或双分支结构；
- 唯一后续分支为 `D1_RETENTION`。

这里的“假负例率”和“RGB 质量关联”仍属于模型/启发式代理，不是人工真值或因果结论。

## 6. 发现的产物一致性问题

回传包中的 `reports/aic_rgbtir_phase17a_plus_recovery_2026_08_13.md` 保留了一段旧模板文字，仍称四个核心问题“尚未完成”。这与同一归档内的以下机器产物冲突：

- `decision.json`：`D1_RETENTION`；
- `run_summary.json`：`PHASE_17A_PLUS_COMPLETE`；
- `layerwise_metrics.json`：逐层漂移已计算；
- `recovery_diagnostics/C2.json`：1,024 条逐记录诊断已完成。

因此后续以 JSON、CSV、日志和 SHA 清单为准，不引用该 Markdown 的过时段落。下一轮报告生成器应增加“报告结论与 run_summary 状态一致”测试。

另外，`subgroup_metrics.csv` 的 `c2_metrics_available` 固定为 `False`；当前 `rgb_quality_association` 实际来自逐记录质量风险与逐层漂移的 Spearman 相关，而不是该分组 CSV。这个字段不能被误读为“没有执行 C2 诊断”。

## 7. 下一轮：Phase 1.8A D1 Retention Probe

### 7.1 唯一目标

在不牺牲 C2 检索和有效秩优势的前提下，显著压低 Layer 8/16/24 的 Base-relative drift，判断 C2 是否可以安全进入 full train。

本轮仍不做 Query、bbox、RGB–TIR fusion、Depth、AIC 测试集训练或平台提交。

### 7.2 单变量改动

控制组保持 C2：

```text
alignment + cross-image InfoNCE + relational distillation
```

D1 只新增逐层 Base TIR retention：

```text
L_D1 = L_C2 + lambda_ret * sum_l w_l * L_ret(l)
```

其中：

- `L_ret(l)` 使用 adapted TIR 与冻结 Base TIR 的方向/关系保持约束；
- 层权重：Layer 8=`1.0`、Layer 16=`1.0`、Layer 24=`1.5`、Final=`0.0`；
- Final 层已经近饱和，不用它驱动训练；
- 不能直接强迫 adapted TIR 完全复制 Base TIR，否则会抵消跨模态学习收益。

### 7.3 两阶段执行

第一阶段只做 repair probe：

1. 固定同一份 fresh rank-48 初始化；
2. C2 使用已恢复结果作为控制；
3. D1 仅比较预注册的低/中两个 retention 强度；
4. 使用 4,096 probe train 训练、1,024 repair-dev 选择；
5. official val 不参与超参数选择。

第二阶段只有 D1 probe 通过才允许：

1. 用选定强度从 fresh Adapter 在 24,612 条 full train 重训；
2. 25%/50%/75%/100% 保存 checkpoint；
3. repair-dev 多目标选择 checkpoint；
4. 最后一次性在 2,032 条 official val 验收。

### 7.4 Probe 硬门禁

D1 相对恢复后的 C2 必须同时满足：

1. Layer 8/16/24 的 absolute alignment delta 均下降；
2. Layer 24 delta 至少相对下降 25%；
3. mean R@5 不低于 C2 超过 0.5 个百分点；
4. mean R@1 不低于 C2 超过 1 个百分点；
5. 最低 effective-rank/Base TIR 不低于 85%；
6. 最低 effective-rank/RGB Teacher 不低于 75%；
7. paired-shuffled margin 保持为正；
8. nonpaired cosine P95 不触发原门禁；
9. RGB base 哈希不变，只有 TIR Adapter 获得梯度；
10. gate=0、`tir=None`、无效 IR 仍严格退化为同模型 RGB-only；
11. 不存在第二模型 fallback。

任一关键项失败就停在 probe，不做 full train。

### 7.5 明确不采用的路线

- 暂不做 RGB-quality weighting：质量关联只有 0.15；
- 暂不改造 false-negative sampling：当前模型代理率 0.1901，证据不足；
- 暂不做 shared/complementary 双分支：尚未证明 C2 + retention 无法解决冲突；
- 暂不进入 Phase 2 融合：当前仍是 TIR 表征安全性修复；
- 不使用 AIC 测试集做训练、阈值选择或伪标签。

## 8. 下一轮开始前的资产门禁

必须使用本次本地回传目录，不重新依赖云端临时盘：

```text
Teacher Bank:
outputs/aic_rgbtir_phase16_recovery_returned_20260813/extracted/
outputs/aic_rgbtir_phase16_recovery_v1/rgb_teacher_bank/teacher_bank.pt

C2 control:
outputs/aic_rgbtir_phase16_recovery_returned_20260813/extracted/
outputs/aic_rgbtir_phase16_recovery_v1/probe_candidates/C2/

repair-dev:
outputs/aic_rgbtir_phase16_recovery_returned_20260813/extracted/
outputs/aic_rgbtir_phase16_recovery_v1/manifests/repair_dev.jsonl
```

运行前必须再次核对 `asset_receipt.json`、归档 SHA-256 和 Teacher Bank fingerprint。`teacher_bank.partial.pt` 只保留作恢复过程证据，禁止作为下一轮输入。

## 9. 当前结论边界

已确认：

- 旧的 C1/C2/Teacher Bank 资产阻塞已经解除；
- C2 存在真实层级漂移，且不是明显由 RGB 质量代理主导；
- D1 retention 是当前唯一有证据支持的下一训练变量。

仍未确认：

- D1 是否能通过 probe；
- TIR Adapter 是否能提升 Query grounding 或 bbox ACC；
- RGB–TIR fusion 是否能提升 AIC 平台成绩；
- 红外收益是否能迁移到 AIC 域。

这些只能由后续 D1 probe、full-val 和 Phase 2 受控平台实验依次回答。
