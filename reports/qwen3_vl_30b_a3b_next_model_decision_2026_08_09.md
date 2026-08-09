# Qwen3-VL-30B-A3B-Instruct 下一轮模型决策（2026-08-09）

## 结论

下一轮优先测试 `Qwen/Qwen3-VL-30B-A3B-Instruct`，而不是继续直接全量测试 EGM。

当前最高平台基线是：

| 模型 | 平台 ACC@0.5 |
|---|---:|
| Qwen3-VL-8B-Instruct zero-shot | **0.7582** |
| LocateAnything-3B zero-shot | 0.7210 |
| Qwen3-VL-8B-Thinking cascade | 0.7194 |
| EGM-Qwen3-VL-8B zero-shot | 0.5333 |

因此，下一轮更强模型应优先沿着已经在 AIC 上验证有效的 `Qwen3-VL-Instruct` 路线扩大，而不是继续 EGM。

## 它相对 Qwen3-VL-8B-Instruct 强在哪里

`Qwen3-VL-30B-A3B-Instruct` 是 MoE 架构，模型卡标注为 31B 参数、BF16 权重。官方说明 Qwen3-VL 相比前代具备更强文本理解、视觉感知与推理、空间理解、OCR 和长上下文能力。

对 AIC 来说，最相关的增强点是：

1. **文本理解更强**：AIC Query 平均更长，包含序数、关系、参照物、动作和结构区域描述。
2. **空间感知更强**：官方强调 stronger 2D grounding、物体位置、遮挡和空间关系判断。
3. **细粒度视觉对齐更强**：DeepStack 融合多层 ViT 特征，理论上更利于小目标和细节。
4. **OCR 和结构理解更强**：AIC 中存在 sign、logo、文字区域、建筑结构和道路设施。
5. **仍是 Instruct 路线**：我们已经在平台上验证 Instruct 系列比 Thinking / EGM 更稳。

## 5090 32GB 能不能跑

判断分两种：

### BF16 原始权重

不建议在单张 5090 32GB 上直接跑 BF16 原始权重。

原因：31B BF16 权重本身大约需要 62GB 量级显存，还没有计算 vision encoder、KV cache、激活和 runtime 开销。单卡 32GB 基本放不下。

如果坚持 BF16，建议至少使用 48GB 以上显存，实际更稳的是 80GB 或多卡张量并行。

### FP8 / 量化权重

可以优先尝试 `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` 或其他官方/可信量化版本。

官方 FP8 仓库说明其为 30B-A3B-Instruct 的 fine-grained FP8 量化版本，性能指标接近原 BF16，并提供 vLLM 和 SGLang 部署示例。

在 5090 32GB 上，推荐先测试：

```text
Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
vLLM 或 SGLang
单图输入
max_tokens 128 / 256
temperature 0
concurrency 从 1 / 2 / 4 递增
```

验收顺序：

1. 先启动服务；
2. 跑 10 条 smoke；
3. 跑 100 条测速；
4. 检查 fallback、解析失败、显存峰值；
5. 如果稳定，再跑 9,555 全量。

## 是否需要换 48GB

如果目标是 **FP8/量化全量推理**，先用 5090 32GB 测是合理的。

如果目标是 **BF16 原始权重、较高并发、较大图片 token、较长输出**，建议换 48GB 或更大显存。

当前优先级：

1. 5090 32GB：测试 30B-A3B FP8 smoke；
2. 若 OOM 或吞吐太低：换 48GB；
3. 若 FP8 平台分数明显超过 0.7582，再考虑 BF16 或更高显存复测；
4. 不建议直接上 235B 或 32B dense，成本和不确定性都更高。

## 下一步执行建议

本轮不要训练，只做零样本全量检测。

推荐实验名：

```text
Qwen3_VL_30B_A3B_Instruct_FP8_zero_shot_v1
```

控制变量：

- 只用 Visible RGB + 原始 Query；
- 不加入 IR / Depth；
- 不做 Tile；
- 不做 Query 改写；
- 不做多模型融合；
- 不用 EGM 输出作为 fallback。

只有这样，平台分数才能直接回答一个问题：

> 更强 Qwen3-VL-Instruct 基座是否能超过当前 0.7582？

## GitHub 记录原则

模型权重、全量预测和提交 ZIP 不进 Git。GitHub 只记录：

- 模型名称；
- 平台分数；
- 本地提交包路径；
- SHA-256；
- 审计结果；
- 为什么继续或放弃该路线。
