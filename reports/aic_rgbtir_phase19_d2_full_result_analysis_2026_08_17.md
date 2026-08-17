# AIC RGB–TIR Phase 1.9-D2 Full 训练结果、门禁诊断与后续决策

> 日期：2026-08-17（Asia/Shanghai）
> 候选：`D2_G025`
> 最终状态：`PHASE_19_D2_FULL_NO_GO`
> 性质：双开发集训练选择结论，不是运行故障，不是 AIC 平台结果

## 1. 本轮要回答的问题

Phase 1.8R 的 `D1_L050` 已经通过 retention 明显降低 C2 的中层漂移，但 sealed official-val 的记录级有效秩仍低于门槛。Phase 1.9-D2 在 D1 上增加 Base-TIR 邻域几何保持，先通过 probe 选择 `geometry_weight=0.25`，再回答：

> D2_G025 的局部秩恢复能否从同一 D1 起点扩展到完整训练，并在 semantic 与 multi-query 两个隔离 dev 上同时达到发布门禁？

本轮不包含 Query 训练、RGB–TIR fusion、Depth、AIC 测试集、bbox 平台评测或第二模型 fallback。

## 2. 固定实验合同

### 2.1 模型与训练对象

- 基座：`Qwen/Qwen3-VL-8B-Instruct`；
- revision：`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；
- 初始化：Phase 1.8R `D1_L050` selected Adapter；
- 训练对象：仅 TIR rank-48 LoRA；
- LoRA 位置：vision attention `qkv/proj`；
- RGB LoRA rank：0；
- RGB 基座、LLM 与输出路径冻结；
- 不从 probe 产生的 G025 Adapter 继续训练。

### 2.2 损失

```text
0.25 × paired alignment
+ 1.00 × cross-image InfoNCE
+ 0.50 × relational distillation
+ 0.10 × background margin
+ 0.50 × Base-relative retention
+ 0.25 × Base-TIR neighborhood geometry
```

Layer 权重：

```text
Layer 8  = 1.00
Layer 16 = 1.00
Layer 24 = 0.25
Final    = 0.00
```

负例数固定为 256，其中 128 个同来源、64 个同条件，其余为全局随机；相同 image pair 禁止成为负例。

### 2.3 数据隔离

| 集合 | records | unique pairs | 用途 |
|---|---:|---:|---|
| full train | 23,391 | 13,310 | 唯一训练集 |
| semantic dev | 1,024 | 1,024 | 单 Query 语义与表征选择 |
| multi-query dev | 1,221 | 512 | 同图多 Query/多目标区分选择 |
| official val | 2,032 | 1,115 | 双 dev GO 后才允许打开一次 |

四个集合按 image pair 完全隔离。official val 没有参与 checkpoint 选择。

## 3. 预注册双 dev 门禁

每个 checkpoint 必须同时满足：

1. semantic 与 multi-query 的最低 effective-rank/Base 均 `>=0.85`；
2. 两个 dev 相对 D1 baseline 的最低 effective-rank/Base 均至少提高 `0.02`；
3. semantic 平均 R@5 降幅 `<=0.005`；
4. multi-query 平均 R@5 降幅 `<=0.010`；
5. 两个 dev 的 nonpaired cosine P95 增量 `<=0.02`；
6. 两个 dev 的 paired-shuffled margin 均大于 0。

只有通过全部门禁的 checkpoint 才能进入候选集。选择器优先比较两个 dev 中更差的 effective-rank/Base，然后比较 R@5、margin 和较早 step。

## 4. 执行完整性

- full train：`23,391/23,391` step；
- checkpoint：25%/50%/75%/100% 四个均生成；
- checkpoint step：`5848 / 11696 / 17544 / 23391`；
- 每个 checkpoint 大小：`53,880,066` bytes；
- 四个 checkpoint 均可在 CPU 上 `torch.load`；
- Stage A ZIP、receipt 与内部 manifest 校验通过；
- 没有 Traceback、CUDA OOM、非有限值、磁盘错误或 `runtime_failure_state.json`；
- selected Adapter：未生成，符合 NO-GO 合同；
- official-val：未运行，符合 sealed gate 合同。

## 5. 四个 checkpoint 的双 dev 结果

| 训练进度 | step | MQ rank/Base | Semantic rank/Base | MQ rank 增益 | Semantic rank 增益 | MQ R@5 | Semantic R@5 | MQ margin | Semantic margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | 5,848 | 0.823677 | 0.916189 | +0.016935 | +0.030571 | 0.894076 | 0.898763 | 0.176946 | 0.160095 |
| 50% | 11,696 | **0.840168** | 0.907785 | **+0.033426** | +0.022167 | **0.915097** | 0.904297 | 0.181581 | 0.162684 |
| 75% | 17,544 | 0.834847 | 0.915537 | +0.028105 | +0.029920 | 0.914005 | 0.905273 | **0.184138** | **0.165323** |
| 100% | 23,391 | 0.835484 | **0.916341** | +0.028741 | **+0.030723** | 0.914551 | **0.905599** | 0.183802 | 0.165016 |

### 5.1 门禁结果

| step | 绝对秩 | 双 dev 秩增益 | R@5 保留 | P95 控制 | margin | 最终 |
|---:|---|---|---|---|---|---|
| 5,848 | FAIL | FAIL | PASS | PASS | PASS | NO-GO |
| 11,696 | FAIL | PASS | PASS | PASS | PASS | NO-GO |
| 17,544 | FAIL | PASS | PASS | PASS | PASS | NO-GO |
| 23,391 | FAIL | PASS | PASS | PASS | PASS | NO-GO |

四个 checkpoint 唯一共同失败项是：

```text
both_dev_absolute_rank = false
```

## 6. 未通过的直接原因

`both_dev_absolute_rank` 对 semantic 和 multi-query 使用 `all()`：任一 dev 的最低 Layer 8/16/24 effective-rank/Base 低于 `0.85`，整个 checkpoint 即失败。

Semantic dev 始终在 `0.9078–0.9163`，稳定通过。Multi-query dev 的最好结果出现在 50% checkpoint：

```text
observed = 0.840168
required = 0.850000
gap      = 0.009832
```

即离门槛约 0.98 个百分点。门槛是训练前固定的硬门禁，不能在结果出现后下调到 0.84 追认 GO。

25% checkpoint 还因为 multi-query 秩增益 `0.016935 < 0.02` 未通过相对增益；从 50% 起该项已修复。

## 7. 这不是哪些问题

### 7.1 不是运行失败

训练、评估、归档和回传均正常完成。NO-GO 来自模型指标，而不是云端中断。

### 7.2 不是检索能力不足

50% checkpoint 的平均 R@5：

```text
multi-query: 0.584221 Base → 0.915097 D2
semantic:    0.605794 Base → 0.904297 D2
```

paired-shuffled margin 也稳定为正，说明正确目标和错误目标仍有清晰间隔。

### 7.3 不是 Phase 1 式整体严重坍缩

D1 baseline 的最低 rank/Base：

```text
multi-query = 0.806742
semantic    = 0.885618
```

D2 50% 提高到：

```text
multi-query = 0.840168（+0.033426）
semantic    = 0.907785（+0.022167）
```

所以谱几何保持方向有效；它只是没有把最难分布推过绝对门槛。

## 8. 对训练动力学的解释

Multi-query rank/Base：

```text
25%  0.823677
50%  0.840168  <- best
75%  0.834847
100% 0.835484
```

R@5 在 50% 后也基本饱和。继续使用完全相同配置延长训练，没有证据能跨过 0.85；更可能重复“强检索、轻度压缩”的平衡。

当前最合理的机制解释是：

- InfoNCE/关系蒸馏继续强力优化目标可检索性；
- geometry/retention 确实恢复了 Base-TIR 关系结构；
- 但统一 `geometry_weight=0.25` 无法针对真正的瓶颈层；
- multi-query dev 的同图多目标/多 Query 结构更容易暴露被压缩的细粒度差异；
- 聚合指标只保存三层最小值，尚不能从 Stage A 断言 Layer 8、16 或24 谁是直接瓶颈。

`minimum_alignment_improvement` 仍为较大的负相对数，但它是 `(base-adapted)/base`，Base loss 很小时会放大。本轮 dev 门禁没有使用该指标；下一轮必须同时输出每层绝对 alignment loss。

## 9. 本轮对路线的影响

### 保留

- 冻结 RGB 主路径；
- rank-48 TIR Adapter；
- paired/InfoNCE/relational/background/retention 主体；
- Base-TIR geometry 方向；
- semantic + multi-query 双 dev；
- Stage A 先归档、official-val 后置的云端合同。

### 停止

- 不重复相同 G025 full train；
- 不通过降低门槛追认 GO；
- 不直接把 50% checkpoint 发布为 selected Adapter；
- 不打开 official-val；
- 不提前加入 Query、fusion、Depth 或平台提交。

## 10. 下一轮唯一任务

### Phase 1.9-D2-Layer-Audit

先对四个 checkpoint 在两个 dev 上逐层复核：

1. Layer 8/16/24 effective rank 与 participation ratio；
2. 奇异值谱和累计谱能量；
3. R@1/R@5、paired cosine、margin、nonpaired P95；
4. semantic 与 multi-query 的逐层差异；
5. 同图 Query 数、数据源、目标大小、遮挡、黑边等分组；
6. 绝对 alignment loss；
7. checkpoint/manifest/代码 fingerprint。

审计后只允许选择一个单变量 D3 分支：

- 若单层主导：只提高该层 geometry/retention；
- 若对比损失与秩恢复直接冲突：只降低该层 contrastive 权重；
- 若同图多目标主导：增加同图关系保持，但不得把同 pair 当负例；
- 若所有层同时轻度不足：再评估跨层谱保持，不能直接堆叠多项改动。

representation 双 dev GO 后，才进入 frozen Query–TIR diagnostic；Query 诊断通过后，才进入 Layer-16/少数层 zero-init residual fusion。

## 11. 结论边界

本轮已经验证：D2_G025 在完整训练中保持强检索并改善 D1 的秩，但未达到 multi-query 绝对秩门槛。

本轮没有验证：

- Query grounding；
- bbox ACC；
- RGB–TIR fusion Rescue/Harm；
- AIC 平台增益；
- 30B 迁移；
- Depth 收益。

## 12. 复现指纹与本地证据

- Stage A ZIP SHA-256：`37A1EE5AB2EC52630478269299F32652B9C983750F7D3A5EFD2862A0E133BD26`；
- Stage A ZIP bytes：`165653981`；
- checkpoint steps：`5848 / 11696 / 17544 / 23391`；
- Teacher Bank fingerprint：`D8F01B557D9FAE020767A27DE7DB4602391ACFE5B27141013C3D4D22A2D677E1`；
- 本地交接目录：`outputs/aic_rgbtir_phase19_d2_full_returned_20260817`，该目录受 `.gitignore` 保护，不上传模型资产；
- GitHub 只保存实现代码、配置模板、测试、门禁定义、指标摘要、哈希和本报告。
