# AIC RGB–TIR Phase 1.5 全量红外适配验证

**最终状态：`PHASE_15_NO_GO`**

本报告只验证冻结 TIR rank-48 Adapter 在 RGBT-GroundBench official val 上的表征泛化；不验证 Query grounding、bbox ACC 或 AIC 平台收益。

## 执行范围

- 记录：2032/2,032；唯一图像对：1115/1,115；跳过：0。
- 本机最坏样本 smoke：10/10，峰值显存 1.64 GiB。
- 全部权重冻结；无训练、无反向传播、无 Query、无融合门、无 bbox 生成、无 AIC 测试集。

## 核心结果

- Base TIR alignment loss mean：0.160773。
- Adapted TIR alignment loss mean：0.070442。
- 相对改善：56.19%。

### 检索与坍缩门禁

- Layer 8：base/adapted R@1 0.2608/0.5039，R@5 0.4400/0.7003，adapted margin 0.2572，effective rank base/adapted 99.45/30.39。
- Layer 16：base/adapted R@1 0.4222/0.7106，R@5 0.6107/0.8563，adapted margin 0.2932，effective rank base/adapted 77.42/22.72。
- Layer 24：base/adapted R@1 0.3381/0.6501，R@5 0.5379/0.8361，adapted margin 0.0744，effective rank base/adapted 59.87/22.94。
- Layer final：base/adapted R@1 0.2397/0.4690，R@5 0.4050/0.6427，adapted margin 0.0070，effective rank base/adapted 1.02/1.02。

## 硬门禁

- PASS `complete`
- PASS `alignment_improvement`
- PASS `subgroup_degradation`
- PASS `positive_margin`
- PASS `r5_preserved`
- PASS `retrieval_gain`
- FAIL `effective_rank`
- FAIL `nonpaired_cosine_p95`
- PASS `safety_equivalence`

失败原因：
- effective_rank collapse at ['8', '16', '24']
- nonpaired cosine P95 increase exceeded 0.10 at ['16']

## 结论边界

- 已验证事实：full-val RGB/TIR ROI 表征对齐、跨模态检索、非配对相似度、effective rank 与安全等价性。
- 当前不能确认：自然语言 Query 是否受益、最终 bbox 是否改善、AIC ACC 是否提高。
- Phase 2：禁止启动，先修复首要失败项。
