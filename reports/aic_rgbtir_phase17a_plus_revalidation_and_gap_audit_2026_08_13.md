# AIC RGB–TIR Phase 1.7A+ 重新验证与缺口审计

> AS_OF: 2026-08-13
> 本轮只恢复并复核既有 Phase 1.6 证据；不训练、不使用 AIC 测试集、不生成比赛提交。

## 1. 最终状态

- 资产门禁：`ASSETS_READY`
- 执行状态：`PHASE_17A_PLUS_COMPLETE`
- repair-dev：`1024/1024` 条完成
- 缺失模型记录：`0`
- 唯一后续分支：`D1_RETENTION`
- 机器产物重复运行：`SHA-256 byte-stable = true`
- 测试：`8 passed`

旧报告中的 `ASSET_BLOCKED` 是当时云端资产尚未回传造成的历史状态，不再代表当前状态。

## 2. 已恢复并核验的关键资产

- RGB Teacher Bank：5,120 个图像对，fingerprint `79CF814B64EC640AD3D924342FEEED235764F139445F834B2CDB2DD0B580C05C`
- C1 Adapter SHA-256：`0D56149083B9EFCFA30DDBB30904A70D631EBA5E080EA2DCD63F123BEFAA85EE`
- C2 Adapter SHA-256：`2832225394CD884D5BA972D9BCD0E19CE8AF8715657DC9675E60AB75B103E786`
- repair-dev manifest：1,024 条，SHA-256 `B9DFB8C4E47CAD542326BD25D5B2054EA7CADD64E6AA12BA72E2E7B84CD5D1D7`
- train/dev/official-val 图像对重叠：`0`

C0 的完整历史总结仍未恢复，但它不是本轮判断 C2 漂移根因和选择 D1 分支的必要输入。

## 3. 本轮修复的工程问题

1. 旧报告生成器无论资产是否齐全，都会输出资产缺失模板；现已改为根据实际门禁状态生成报告。
2. 旧实现先写 subgroup 文件、后连接 C2 诊断，导致分组表中 C2 指标恒为不可用；现已在写文件前完成连接。
3. 新增 RGB Teacher 目标区与背景区可分性指标，并对其与 C2 漂移的关系做固定种子 bootstrap。
4. 新增回归测试，确保恢复资产会进入模型证据路径，完整资产不会再次生成 `ASSET_BLOCKED` 报告。

## 4. 已确认的模型证据

### 4.1 C2 存在真实逐层漂移

`C2 alignment loss - Base TIR alignment loss` 的平均值：

| 层 | 平均绝对漂移 |
|---|---:|
| Layer 8 | 0.1289 |
| Layer 16 | 0.1526 |
| Layer 24 | 0.3838 |

总体 C2 平均漂移为 `0.22177`。Layer 24 最严重，说明冲突不是只发生在最终输出层，而是在较深视觉表征中持续累积。

### 4.2 漂移与普通 RGB 质量只有弱关联

下列值均为 Spearman 相关；负值表示图像质量信号越高，C2 漂移总体越低：

| 信号 | Spearman | bootstrap 95% CI |
|---|---:|---:|
| 全图 RGB 亮度 | -0.1486 | [-0.2077, -0.0870] |
| 全图 RGB 模糊度 | -0.1201 | [-0.1821, -0.0594] |
| ROI RGB 对比度 | -0.1370 | [-0.1981, -0.0780] |
| ROI RGB 熵 | -0.1404 | [-0.2031, -0.0782] |
| IR 有效视场比例 | -0.0716 | [-0.1347, -0.0104] |

这些关联方向稳定，但效应量较弱，不足以说明“只对低质量 RGB 样本降低对齐权重”就能解决问题。

### 4.3 RGB Teacher 可分性是更强但仍非因果的信号

RGB Teacher 的目标区–背景区 margin 与 C2 漂移相关为：

```text
Spearman = -0.2464
bootstrap 95% CI = [-0.3054, -0.1862]
```

该 margin 定义为 Layer 8/16/24 中 RGB Teacher 前景 ROI 与背景表示的平均余弦距离。结果说明：Teacher 越难区分目标与背景，C2 漂移往往越大。但 `|rho| < 0.25`，它仍只能作为风险信号，不能单独证明 RGB Teacher 质量是漂移的根因。

### 4.4 假负例风险存在，但当前不能确认其因果责任

- 256 个固定跨图负例的词法潜在假负例率：`8.50%`
- 高 Query 词集相似率：`0.54%`
- 模型 top-neighbor 潜在假负例代理率：`19.01%`
- same-pair 违规：`0`

这些是 target-head、superclass 和邻近关系代理，不是人工真值。现有资产没有完整的逐负例语义标注，因此不能断言 InfoNCE 假负例就是 C2 漂移的主因。

## 5. 分组观察

漂移较高的分组包括 very-weak-light、rainy、heavy-occlusion、normal-small 等，但不同来源和条件之间的差距远小于所有分组共同存在的 C2 漂移。

这意味着问题更像是训练目标的全局结构冲突，而不是单独由黑边、弱光、遮挡或某一个来源数据造成。

## 6. 原 Phase 1.7A+ 计划的完成边界

### 已完成

- C1/C2、Teacher Bank 和 repair-dev 资产恢复与哈希核验；
- 1,024 条逐记录输入质量画像；
- C2 相对 Base TIR 的 Layer 8/16/24 绝对漂移；
- RGB Teacher 前景–背景可分性及 bootstrap 相关分析；
- 固定 256 负例的词法审计和模型邻居代理；
- source、illumination、weather、size、occlusion、IR FOV 分组；
- 单一后续分支决策。

### 仍不能从现有资产补出的内容

- C1/C2 全量原始 ROI embedding，因此不能重新计算完整 gallery 的逐记录 R@1/R@5 和 paired-shuffled margin bootstrap；
- 每个负例的人工语义真值，因此不能把代理假负例率称为真实假负例率；
- 完整 registration-risk 和 ROI token 级诊断；
- Query grounding、bbox ACC 和 AIC 平台收益。

补齐这些缺口需要重新执行视觉前向或人工标注，不应从当前摘要文件中推断或伪造。

## 7. 为什么选择 D1_RETENTION

当前证据同时满足：

1. C2 对齐与检索方向有效，但存在显著逐层漂移；
2. 漂移不只集中在低质量 RGB 样本；
3. 普通 RGB 质量与漂移的关联不足以支持 `D1_QUALITY_WEIGHTED`；
4. 假负例风险尚未被确认为主要原因，不足以直接进入 `C2_FALSE_NEGATIVE_AWARE`；
5. 暂无必要立即引入 shared/complementary 双分支的大结构变化。

因此下一轮应保持 C2 的 InfoNCE + relational distillation 训练口径，只增加逐层 Base-TIR retention 约束，直接抑制 Layer 8/16/24 的过度漂移。

## 8. 下一轮唯一建议

先执行一个小规模 `D1_RETENTION_PROBE`：

```text
C2 loss
+ per-layer Base-TIR retention (Layer 8/16/24)
+ RGB 主路径冻结
+ 同一 repair-probe / repair-dev 切分
```

只在 probe 同时保留 C2 的检索收益、降低 Layer 24 漂移并通过有效秩门禁后，才允许 full train。此轮不加入质量门控、假负例过滤、Query 融合、bbox head 或 AIC 测试集。

## 9. 关键产物

- 机器结果：`outputs/aic_rgbtir_phase17a_plus_revalidated_v2/`
- 自动报告：`reports/aic_rgbtir_phase17a_plus_revalidated_2026_08_13.md`
- 本审计报告：`reports/aic_rgbtir_phase17a_plus_revalidation_and_gap_audit_2026_08_13.md`
- 恢复资产收据：`reports/aic_rgbtir_phase16_recovery_asset_receipt_and_phase18_plan_2026_08_13.md`
