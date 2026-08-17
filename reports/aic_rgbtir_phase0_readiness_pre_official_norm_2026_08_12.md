# AIC RGB–TIR Phase 0 可训练性验收

**最终状态：`PHASE_0_GO`**

本轮仅完成数据、成对预处理和模型接口验证；未训练、未生成 checkpoint、未生成 AIC 提交。

## Manifest

- 原始实例：38760；唯一图像对：21535；
- train all / clean：26604 / 26477；
- val / test：2032 / 10124；
- 排除训练记录：127；原因：`{'bbox_meta_language': 106, 'cross_split_image_pair': 2, 'negative_or_absent_language': 108}`；
- 跨 split 图像对：`['m3fd/02627.png', 'mfad/cali_l_2023_10_17_17_46_05_549.jpg']`，对应 train 记录已排除。

## RGB/TIR 图像与 Grid

- RGBT：21535/21535 成功，黑边 4056，低信息 IR 0，可用 IR 21534；
- AIC：2000/2000 成功，黑边 1439，低信息 IR 0，可用 IR 1998；
- RGBT/AIC grid 相等：21535/21535、2000/2000。

## Processor 与模型等价性

- tracer processor：500/500；
- gate=0 final 最大绝对误差：0.0；
- gate=0 DeepStack 最大绝对误差：0.0；
- RGB-only 缺失 TIR 最大绝对误差：0.0；
- 包装器不包含跨模型 fallback；IR 不可用时走同一个模型的 RGB-only 路径。

## Tracer

- train / val：400 / 100；
- 合并条件统计：`{'adverse_weather': 139, 'high_occlusion': 187, 'low_light': 220, 'small': 306}`；
- 来源统计：`{'flir': 168, 'm3fd': 166, 'mfad': 166}`。

## 测试

- pytest：PASS；测试文件 3 个；
- paired processor failure：0。

## Phase 1 进入条件

所有硬门槛通过，可以进入 Qwen3-VL-8B BF16 的 TIR adapter warmup。
进入训练前仍需在真实 8B BF16 视觉权重上完成一次 10 条无训练 smoke。

## 结论边界

- RGBT 的天气、尺寸和遮挡字段是外部数据 GT；
- AIC 的黑边、低信息和 grid 结果是输入审计事实；
- 本轮没有 AIC bbox GT，因此没有多模态精度提升结论。
