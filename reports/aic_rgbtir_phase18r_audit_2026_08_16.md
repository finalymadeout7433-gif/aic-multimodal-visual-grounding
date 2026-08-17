# AIC RGB–TIR Phase 1.8R-Audit 本地复核报告

**最终决策：`BOTH_REQUIRED`**

## 结论

- 历史安全失败仅来自 official-val 前未关闭训练状态的冻结契约错误
- 记录级有效秩未达门槛：layer_8:record_level, layer_16:record_level, layer_24:record_level
- 全局绝对对齐漂移 0.098007 超过 0.020000 门槛

## 完整性与安全契约

- 输入完整：`True`；输入复核前后保持不变：`True`。
- 历史安全失败项：`no_trainable_parameters`。
- 是否属于可修复的推理冻结契约问题：`True`。
- Adapter：`108` 个张量，全部有限：`True`。

## 三套检索口径

| 层 | 严格记录 R@5 | 同图像对多正例 R@5 | 图像对聚合 R@5 |
|---|---:|---:|---:|
| 8 | 0.9134 | 0.9149 | 0.9561 |
| 16 | 0.9070 | 0.9099 | 0.9614 |
| 24 | 0.7539 | 0.7589 | 0.8879 |

## 有效秩复核

| 层 | 记录级 Adapted/Base | 图像对聚合 Adapted/Base |
|---|---:|---:|
| 8 | 0.8038 | 0.8700 |
| 16 | 0.8180 | 0.9090 |
| 24 | 0.8179 | 0.8862 |

## ExcessDrift 最大的分组

全局绝对漂移：`0.098007`。正值表示 Adapted 比 Base 更偏离 RGB Teacher。

| 分组 | 记录/图像对 | ExcessDrift | 95% CI |
|---|---:|---:|---:|
| illumination=VWL | 16/10 | 0.031383 | [0.017082, 0.044622] |
| illumination=SL | 410/213 | 0.019456 | [0.016227, 0.022613] |
| source_dataset=flir | 608/325 | 0.018672 | [0.016197, 0.021160] |
| weather=SY | 481/253 | 0.015556 | [0.012915, 0.018430] |
| black_border_severity=mild | 294/155 | 0.014427 | [0.011008, 0.018187] |
| weather=RY | 127/61 | 0.008023 | [-0.002132, 0.017933] |
| source_dataset=m3fd | 168/93 | 0.006191 | [0.001737, 0.010362] |
| occlusion=PO | 110/101 | 0.004939 | [-0.000474, 0.010253] |

## 下一轮唯一主分支

先固定验证器，再运行单变量 D2 谱保持修复；通过后才进入冻结 Query–TIR probe。

## 结论边界

本报告只复核 RGBT-GroundBench official-val 表征和历史安全门禁；不证明 Query grounding、bbox ACC、AIC 平台分数或 UniRGB-IR 融合收益。

输入指纹：`66EC9A9C87A5D9069C53FFD27C08F2CB0535A3CD0082576BA252453E35A3C413`。
