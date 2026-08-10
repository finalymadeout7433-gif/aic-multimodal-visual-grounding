# InternVL / Step3-VL 候选模型评估（2026-08-10）

## 当前基线

当前 AIC 最高分：

```text
Qwen3-VL-30B-A3B-Instruct-FP8 + Qwen3-VL-8B fallback
ACC@0.5 = 0.7757
```

因此后续模型不是“能不能超过 Florence 或 MM-GDINO”，而是要判断是否有机会超过 0.7757。

## 候选 1：InternVL2-8B-Instruct / InternVL 系列

### 可取性

有可取性，但不建议只测截图里的旧 `InternVL2-8B-Instruct`。如果要测 InternVL 系列，优先看更新的 `InternVL3.5-8B-Instruct` 或至少 `InternVL3-8B-Instruct`。

依据：

- InternVL2-8B 模型卡明确给出了 grounding prompt，用 `<ref>{}</ref>` 请求区域 bbox，说明它具备 referring grounding 能力。
- InternVL 官方仓库持续更新到 InternVL3 / InternVL3.5；InternVL3.5 模型卡说明其引入 Visual Resolution Router、推理效率提升和更强多模态能力。

参考：

- InternVL2-8B 模型卡：https://huggingface.co/OpenGVLab/InternVL2-8B
- InternVL 官方仓库：https://github.com/OpenGVLab/InternVL
- InternVL3.5-8B-Instruct：https://huggingface.co/OpenGVLab/InternVL3_5-8B-Instruct

### 对 AIC 的潜在优势

1. 原生支持 bbox/grounding 风格输出。
2. 对高分辨率、多图像切片和细粒度视觉可能较友好。
3. 8B 级别成本低，适合做全量单变量测试。

### 主要风险

1. 旧 InternVL2-8B 不一定强于 Qwen3-VL-8B，更不一定接近 Qwen3-VL-30B 的 0.7757。
2. 生成式 bbox 仍有格式和拒答风险，需要沿用 parser + retry + fallback。
3. 仍然只吃 RGB；不会自然利用 AIC 的 IR/Depth。

### 推荐硬件

| 目标 | 推荐卡 |
|---|---|
| InternVL2/3/3.5 8B zero-shot 全量 | 4090 24GB 即可 |
| 更高分辨率 / 更稳吞吐 | 5090 32GB 更好，但不是必须 |
| LoRA 微调 | 4090 24GB 可做 8B LoRA；更大 batch 用 5090/48GB |

### 是否建议全量 AIC

建议，但优先级低于 Step3-VL-10B。原因是当前 Qwen 系已经到 0.7757，InternVL 8B 要直接超过它难度较大。

推荐只测一个更新版本：

```text
InternVL3.5-8B-Instruct zero-shot
```

如果用户坚持截图里的模型，则测：

```text
OpenGVLab/InternVL2-8B
```

但预期应设为探索性，不应假设能超过 0.7757。

## 候选 2：Step3-VL-10B-Instruct

### 可取性

有可取性，而且比 InternVL2-8B 更值得作为下一次全量单变量测试。

依据：

- Step3-VL 官方说明其统一预训练覆盖 reasoning、perception、grounding、counting、OCR、GUI interactions。
- 官方仓库/项目页称其 10B 级别在多模态 benchmark 上表现强，并提供 Hugging Face 与 ModelScope 下载。
- AIC 的高分瓶颈包含长 Query、序数、空间关系、小目标和区域定位，Step3-VL 的定位、计数、OCR/GUI grounding 方向与这些难点有重合。

参考：

- Step3-VL-10B 官方仓库：https://github.com/stepfun-ai/Step3-VL-10B
- Step3-VL-10B 模型卡：https://huggingface.co/stepfun-ai/Step3-VL-10B-Base
- Step3-VL-10B 项目页：https://stepfun-ai.github.io/Step3-VL-10B/

### 对 AIC 的潜在优势

1. 10B 参数量，比 8B 略大，但成本远低于 30B。
2. 官方强调 grounding、counting、OCR 和 GUI interaction，这些能力与 AIC 的小目标/区域/文字/结构目标相关。
3. 如果 bbox 输出稳定，它可能成为“低成本强基座”。

### 主要风险

1. 它未必像 Qwen3-VL 系列一样在我们当前 runner 中直接稳定输出归一化 bbox。
2. 若官方接口或 chat template 不兼容 OpenAI/vLLM，需要单独适配。
3. 它的公开 benchmark 不是 AIC，不能直接推断平台分数。

### 推荐硬件

| 目标 | 推荐卡 |
|---|---|
| Step3-VL-10B zero-shot 全量 | 4090 24GB 应可优先尝试 |
| 更稳 vLLM / 高分辨率 / 更少 OOM 风险 | 5090 32GB |
| 4-bit 量化 LoRA | 4090 24GB 可尝试 |
| BF16 全量高吞吐 | 32GB 或 48GB 更稳 |

### 是否建议全量 AIC

建议。下一次如果只测一个新系列，优先测：

```text
Step3-VL-10B-Instruct zero-shot
```

并沿用这轮成熟策略：

```text
新模型正常 bbox
新模型 fallback 样本用 Qwen3-VL-30B 当前最高分 0.7757 或 Qwen3-VL-8B 0.7582 补位
```

注意：如果目标是判断 Step3 原始能力，应同时保留“纯 Step3 包”和“Step3 + 历史最佳 fallback 包”两个结果，但平台优先上传级联包。

## 下一轮建议顺序

| 优先级 | 模型 | 卡 | 目的 |
|---:|---|---|---|
| 1 | Step3-VL-10B-Instruct | 4090 24GB 起步；5090 32GB 更稳 | 判断是否有低成本模型能接近或超过 0.7757 |
| 2 | InternVL3.5-8B-Instruct | 4090 24GB | 测 InternVL 最新 8B 系列，而不是旧 InternVL2 |
| 3 | InternVL2-8B-Instruct | 4090 24GB | 仅作截图候选复核 |

## 不建议的方向

1. 不建议直接对 InternVL2-8B 做 LoRA 微调再上平台。先零样本全量看平台得分。
2. 不建议在未验证 Step3 输出格式前直接全量跑 9555 条。先做 10/100 条 smoke。
3. 不建议用 IR/Depth 直接拼图喂这些 RGB-VL 模型；这会引入新的域偏移变量。
4. 不建议同时改 prompt、tile、fallback、模型和分辨率；下一轮仍保持单变量。

## 最终判断

如果现在还想再测两个系列：

```text
第一优先级：Step3-VL-10B-Instruct
第二优先级：InternVL3.5-8B-Instruct
第三优先级：InternVL2-8B-Instruct
```

硬件选择：

```text
4090 24GB：足够做 Step3/InternVL 8B-10B zero-shot 全量
5090 32GB：更适合节省时间、减少环境和显存风险
48GB：只有在 BF16、大图 token、高并发或 LoRA 微调时才更有必要
```
