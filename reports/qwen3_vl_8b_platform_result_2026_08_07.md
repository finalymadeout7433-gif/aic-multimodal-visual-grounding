# Qwen3-VL-8B-Instruct 平台结果与下一轮模型路线

日期：2026-08-07

## 结论

`Qwen3-VL-8B-Instruct` 零训练全量检测刷新了当前项目最高分。

| 字段 | 结果 |
|---|---:|
| 模型 | `Qwen/Qwen3-VL-8B-Instruct` |
| 策略 | Zero-shot，Visible RGB + 原始英文 Query |
| AIC Query | 9,555 |
| 平台 ACC@0.5 | **0.7582** |
| 平台排名截图 | 14 |
| 平台记录时间 | 2026-08-07 19:28:19 |

该成绩来自用户提供的平台截图。它超过 `LocateAnything-3B` 的 `0.7210`，成为当前已确认的最高分单模型基线。

## 本地提交包

本轮提交包保存在 Git 仓库外部，不提交到 GitHub：

```text
D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\platform_upload_ready\AIC_Qwen3_VL_8B_Instruct_zero_shot_20260807.zip
```

提交包 SHA-256：

```text
D3D1757BA6052C117B13DC8175B21475DF22B084D2465570B5E998C1D78E1188
```

## 本地审计

云端输出目录：

```text
/home/featurize/aic_cloud/outputs/aic_qwen3_vl_8b_zero_shot_v1
```

下载回本地的完整证据目录：

```text
D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\platform_upload_ready\qwen3_vl_8b_zero_shot_20260807
```

| 检查项 | 结果 |
|---|---:|
| 完成记录 | 9,555 / 9,555 |
| invalid bbox | 0 |
| no candidate | 0 |
| 非 bbox 字段修改 | 0 |
| Query ID 精确一致 | true |
| ZIP 内容 | 仅 `predictions_submission.json` |
| submission JSON SHA-256 | `39DE2E32B5D47966CEC91E3B9B7478F1A5A59AFAA250C5B543DB27EF5BC3F4CC` |
| submission ZIP SHA-256 | `D3D1757BA6052C117B13DC8175B21475DF22B084D2465570B5E998C1D78E1188` |

运行摘要：

| 项 | 值 |
|---|---|
| GPU | RTX 4090 |
| 模型路径 | `/home/featurize/aic_cloud/models/Qwen3-VL-8B-Instruct` |
| 本轮耗时 | 8,526.92 秒 |
| 候选数量 | 每条 1 个最终框 |

## 与已有模型的关系

| 模型 / 策略 | 平台 ACC@0.5 | 相对 Qwen3-VL-8B |
|---|---:|---:|
| Qwen3-VL-8B-Instruct zero-shot | **0.7582** | — |
| LocateAnything-3B zero-shot | 0.7210 | -3.72 pp |
| MM-Grounding-DINO-T zero-shot | 0.5011 | -25.71 pp |
| Florence-2 RGB-only | 0.4980 | -26.02 pp |
| APE-Ti zero-shot | 0.4424 | -31.58 pp |

平台结论很清楚：AIC 当前更吃长 Query、关系理解、OCR/区域理解和空间语义能力，而不是传统检测器式 Top-1 候选分数。后续主线应从 Qwen3-VL 系列和同类强 VLM grounding 模型继续扩展。

## 下一轮更大或更适合模型

### 1. Qwen3-VL-8B-Thinking

优先级：高。

理由：

- 与当前最高分模型同系列、同量级；
- 改动最小，环境和 runner 最容易复用；
- Thinking 版可能改善 AIC 中的序数、相对位置、主体/参照物和长 Query。

风险：

- 输出更长，bbox 解析失败概率可能上升；
- 推理更慢；
- 如果 prompt 不约束，可能解释过多而不稳定。

建议：先 200 条 smoke，确认 bbox 解析率，再全量平台单变量提交。

### 2. EGM-Qwen3-VL-8B

优先级：高，但先核验权重和许可证。

理由：

- EGM 论文明确针对 visual grounding，不是普通 VQA；
- 其核心判断是：小 VLM 在 grounding 上落后主要来自复杂文本理解，而 AIC 正好大量存在复杂 Query；
- 论文报告 EGM-Qwen3-VL-8B 在 RefCOCO 上超过 Qwen3-VL-235B，同时更快。

风险：

- 需要确认 Hugging Face 权重、推理代码和许可证；
- 如果模型输出格式和 Qwen3-VL 官方不同，需要重新适配解析器。

建议：作为下一轮最有价值的“同等算力增强版 Qwen”候选。

### 3. Qwen3-VL-30B-A3B-Instruct / Thinking

优先级：中高。

理由：

- MoE 总参数约 31B，激活参数约 3B；
- 理论上语言理解更强，可能改善复杂 Query；
- 比 235B 更现实。

硬件判断：

- 单卡 4090 24GB 跑原生 Transformers FP16 不现实；
- 4-bit / FP8 / GGUF 方案可能可测，但要先做 20～100 条 smoke；
- 5090 32GB 或 48GB 级 GPU 更稳。

### 4. Qwen3-VL-32B-Instruct / Thinking

优先级：中高。

理由：

- dense 32B 可能比 8B 有更强文本和空间关系理解；
- Hugging Face 官方集合已提供 Instruct、Thinking、FP8、GGUF 等变体。

硬件判断：

- 24GB 4090 对 32B 原生推理非常紧；
- 建议租 48GB 或更高显存，或先用 4-bit/GGUF 做工程可行性验证；
- 需要严格防止因为量化导致 bbox 数值不稳定。

### 5. Qwen3-VL-235B-A22B

优先级：低，除非有多卡大显存预算。

理由：

- Qwen3-VL 官方有 235B-A22B Instruct/Thinking/FP8/GGUF；
- 理论能力最强。

风险：

- 单 4090/5090 不适合；
- 多卡 80GB 级别更现实；
- 成本高，且 EGM 论文显示在 grounding 上“小模型 + 推理/训练优化”可能比直接上 235B 更划算。

## 模型路线判断

下一轮不建议直接做 ensemble、Depth、IR 或训练。当前最有信息价值的单变量顺序：

1. `Qwen3-VL-8B-Thinking` 全量；
2. `EGM-Qwen3-VL-8B` 全量；
3. `Qwen3-VL-30B-A3B-Instruct` 量化 smoke；
4. `Qwen3-VL-32B-Instruct` 量化 smoke；
5. 只有当 30B/32B 单模型明显超过 8B，才考虑更贵的 235B 或多模型融合。

## 参考来源

- Qwen3-VL 官方 GitHub: https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL Hugging Face Collection: https://huggingface.co/collections/Qwen/qwen3-vl
- Qwen3-VL-8B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- EGM paper: https://arxiv.org/html/2601.13633v3
- LocateAnything model card: https://huggingface.co/nvidia/LocateAnything-3B
- Eagle / LocateAnything repository: https://github.com/NVlabs/Eagle
