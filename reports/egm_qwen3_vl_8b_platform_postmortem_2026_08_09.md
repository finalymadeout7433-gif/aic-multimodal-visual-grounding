# EGM-Qwen3-VL-8B 平台结果与路线复盘（2026-08-09）

## 结论

本轮 `EGM-Qwen3-VL-8B` 不作为后续主线。它的提交格式和 bbox 合法性没有问题，但平台 `ACC@0.5 = 0.5333`，明显低于当前最高的 `Qwen3-VL-8B-Instruct = 0.7582`，也低于 `LocateAnything-3B = 0.7210` 和 `Qwen3-VL-8B-Thinking cascade = 0.7194`。

因此，后续不继续直接全量押注 EGM。若还要验证 EGM，只做小规模 prompt / token / runtime 诊断，不再直接作为下一轮平台提交主模型。

## 平台与提交包记录

| 字段 | 结果 |
|---|---|
| 模型 | `nvidia/EGM-8B` / EGM-Qwen3-VL-8B |
| 输入 | Visible RGB + 原始英文 Query |
| 训练 | 无；零样本推理 |
| GPU | RTX 5090 32GB 云端实例 |
| 正式 Query | 9,555 |
| fallback | 0 |
| retry | 0 |
| invalid bbox | 0 |
| modified non-bbox fields | 0 |
| 平台 ACC@0.5 | **0.5333** |
| 本地提交 ZIP | `D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\platform_upload_ready\egm_qwen3_vl_8b_zero_shot_5090_20260809\AIC_EGM_Qwen3_VL_8B_zero_shot_5090_v1_20260809.zip` |
| ZIP SHA-256 | `33F3284CA1C38E1B1A364CDBB86B83D467F02C59E76FC1CC82B4346BC8E788CD` |

本地审计通过：ZIP 只包含 `predictions_submission.json`，Query ID 完全一致，bbox 全部合法，非 bbox 字段未修改。

## 与 Qwen3-VL-8B-Instruct 的无标签对比

对比对象：`Qwen3-VL-8B-Instruct zero-shot = 0.7582`。

| 指标 | Qwen3-VL-8B-Instruct | EGM-Qwen3-VL-8B |
|---|---:|---:|
| 预测框面积均值 | 4.16% | 6.54% |
| 预测框面积中位数 | 1.10% | 2.93% |
| 面积 <0.1% | 838 | 3 |
| 面积 <1% | 4,589 | 1,798 |
| 面积 >=10% | 1,094 | 1,727 |
| 面积 >=25% | 318 | 532 |

两模型逐样本预测框关系：

| 指标 | 数值 |
|---|---:|
| IoU(EGM, Qwen) >= 0.5 | 5,561 / 9,555 |
| IoU(EGM, Qwen) >= 0.7 | 4,818 / 9,555 |
| IoU(EGM, Qwen) < 0.1 | 2,607 / 9,555 |
| EGM 面积超过 Qwen 2 倍 | 3,548 / 9,555 |
| EGM 面积小于 Qwen 一半 | 88 / 9,555 |

## 主要失败模式

EGM 的主要问题不是无法输出 bbox，而是系统性偏向更大的上下文框。它在 AIC 的小目标、序数、远距离目标上经常从紧框扩大到背景区域、参照物或目标附近上下文。

按 Query 粗分类：

| Query 类型 | 样本数 | EGM 与 Qwen IoU>=0.5 | EGM 面积超过 Qwen 2 倍 | Qwen 面积中位数 | EGM 面积中位数 |
|---|---:|---:|---:|---:|---:|
| small_object | 1,881 | 51.3% | 848 | 0.74% | 3.11% |
| ordinal | 2,301 | 44.9% | 1,061 | 0.69% | 2.19% |
| depth_relation | 1,725 | 57.3% | 640 | 1.20% | 3.08% |
| region_structure | 1,716 | 58.7% | 646 | 1.24% | 3.67% |
| person | 2,247 | 67.6% | 618 | 1.01% | 2.03% |

典型高风险 Query 包括：

- `The leftmost drone.`
- `The soccer ball in the center of the image`
- `The fourth lantern from right to left`
- `The leftmost ball`
- `A black bird in the distance on the left side of the sky, above the trees.`

这些样本中，Qwen 往往给出极小框，而 EGM 给出大区域框，导致 ACC@0.5 明显受损。

## 原因判断

1. EGM 的公开强项主要体现在 RefCOCO 系列 grounding，但 AIC 的目标域更偏小目标、序数、无人机/鸟/球/灯/结构细节。
2. EGM 基于 `Qwen3-VL-8B-Thinking`，而 Thinking 在 AIC 上已低于 Instruct；EGM 的 grounding 强化没有抵消 AIC 域偏移。
3. 本轮使用短输出配置，未充分发挥 EGM 官方强调的 test-time compute；但即使如此，其主要错误表现是框尺度偏大，而不是解析失败。
4. EGM 更像在框“句子描述区域”，而 AIC 平台按目标 tight bbox 的 IoU>=0.5 计分，大框会严重损失 IoU。

## 后续策略

- 当前主基线仍是 `Qwen3-VL-8B-Instruct = 0.7582`。
- 下一轮若测试更强模型，优先 `Qwen3-VL-30B-A3B-Instruct`，而不是继续 EGM。
- EGM 若复测，只允许小规模消融：prompt、max_tokens、SGLang/vLLM，不再直接全量提交。
- 不把 EGM 作为融合默认候选，避免把大框偏差引入 ensemble。
