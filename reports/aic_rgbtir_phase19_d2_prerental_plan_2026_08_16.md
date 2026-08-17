# AIC RGB–TIR Phase 1.9-D2 租卡前计划

## 目标

修复 Phase 1.8R 在记录级出现的有效秩不足，同时保留 D1 的高检索能力。D2 仍只训练 rank-48 TIR Adapter，不加入 Query、融合、Depth、配准、第二模型 fallback 或 AIC 测试集。

## 唯一因果变量

D1_L050 的全部损失、数据、优化器和初始化保持不变，只增加 Base-TIR 跨记录几何保持项：

```text
L_D2 = L_D1_L050 + lambda_geometry * L_base_tir_geometry
```

当前样本的 Adapted TIR 与一组预计算 Base-TIR anchors 计算关系向量，并匹配当前样本 Base-TIR 与相同 anchors 的关系向量。关系向量先在样本内标准化，避免“所有相似度都接近常数”的低秩捷径。该项不增加第二次 Qwen 前向，也不要求单样本 SVD。

固定层权重：Layer 8=`1.0`、Layer 16=`1.0`、Layer 24=`0.25`、Final=`0`。Layer 24 只保留弱约束，防止其较大漂移主导训练。

候选只改变 `lambda_geometry`：`0.10 / 0.25 / 0.50`。

## 固定验证

- `semantic_dev`：原 1,024 条、1,024 对，用于保留语义检索和对齐；
- `multiquery_dev`：从 full train 中固定抽取 512 个、每对至少两条记录的图像对，并从后续训练池完整排除；
- official val 不参与候选选择，只在未来 D2 全量训练后打开一次。

## Probe 硬门禁

1. 三候选从同一个 Phase 1.8R selected Adapter 初始化；
2. RGB 主干哈希不变，只有 TIR LoRA 获得梯度；
3. semantic dev 的平均 R@5 相对同场重算的 D1 基线不下降超过 0.5 个百分点；
4. multiquery dev 的平均 R@5 相对 D1 不下降超过 1 个百分点；
5. 两个 dev 的最低 Adapted/Base 有效秩均不得下降超过 1 个百分点，并且至少一个 dev 提升 2 个百分点；
6. paired-shuffled margin 保持为正，nonpaired cosine P95 相对 D1 增量不超过 0.02；
7. gate=0、无红外、无效红外、全零 mask 均退化为同模型 RGB-only；
8. 不存在第二模型 fallback。

只有 Probe 通过后才安排 D2 全量训练。Query–TIR 与 Layer-16 融合仍在其后。
