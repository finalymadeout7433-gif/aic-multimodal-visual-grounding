# Phase 1.7A+：无训练证据审计执行计划

> AS_OF: 2026-08-13
> 分支：`exp/aic-rgbtir-phase17a-plus-v1`
> 目标：把 Phase 1.6 的五种可能后续路线压缩为一个有证据支持的训练分支。

## 边界

- 不训练、不反向传播；
- 不使用 AIC 测试集；
- 不改变 repair-dev；
- 不修改 C1/C2 Adapter；
- 不生成平台提交；
- 缺失模型资产时输出 `ASSET_BLOCKED`，不使用聚合报告伪造逐层结论。

## 公共接口与测试 seam

```python
Phase17APlusAuditor.run(...) -> Phase17APlusResult
Phase17DecisionPolicy.decide(metrics) -> Phase17Decision
```

第一个深模块负责资产核验、输入质量画像、负例风险审计、产物和证据边界；第二个模块只根据预注册阈值输出一个后续分支。

## 执行顺序

```text
asset-preflight
→ RGB / ROI / TIR-FOV 输入画像
→ 256 跨图负例词义风险审计
→ C1/C2 逐层绝对对齐（资产存在时）
→ RGB Teacher 可靠性相关性（模型指标存在时）
→ 单一决策与报告
```

## 单一决策空间

```text
C2_FULL_TRAIN
D1_RETENTION
D1_QUALITY_WEIGHTED
C2_FALSE_NEGATIVE_AWARE
SHARED_COMPLEMENTARY_PROBE
ASSET_BLOCKED
```

## 验收

- repair-dev 必须为 1,024 条、1,024 个图像对；
- 固定随机种子 `20260812`；
- 负例不得来自相同 image pair；
- 所有机器产物可重复生成；
- `decision.json` 只能包含一个分支；
- 输入代理与模型事实严格分开。
