# 官方示例 bbox 独立核验

## 样例

- Query ID：`000108_001`
- Query：`The silver light bulb inside the bubble house`
- RGB 尺寸：`1920 × 1080`
- 官方归一化 bbox：`[0.7718, 0.9249, 0.8129, 0.9762]`
- 官方像素 bbox：`[1481.856, 998.892, 1560.768, 1054.296]`

## 独立核验结论

将官方坐标按 RGB 原始宽高转换并绘制后，框位于画面右下角透明泡泡屋内部，
紧密覆盖一只银色灯泡。目标类别、属性和关系均与 Query 一致：

- `silver`：框内物体呈银白色；
- `light bulb`：可见灯泡状发光体及下方灯座/连接结构；
- `inside the bubble house`：该物体位于右下角透明穹顶结构内部。

因此，官方标注在这个样例上是**视觉上合理且坐标制式正确**的。该结论属于
对公开样例的可视化核验，不能推广为“全部官方标注均无误”。

## 与 Florence-2 零样本结果对比

- Florence 像素框：`[843.84, 442.26, 928.32, 530.82]`
- Florence 归一化框：约 `[0.4395, 0.4095, 0.4835, 0.4915]`
- IoU：`0`
- ACC@0.5：`False`

Florence 框住的是树上的蓝色球形装饰，而非泡泡屋内的银色灯泡。这说明失败
来自目标识别/关系理解，而不是官方小数 bbox 与模型整数像素 bbox 的转换错误。

官方框面积约占整图 `0.210843%`，属于明显小目标；后续应重点测试高分辨率切片、
多尺度候选生成和关系重排序。

## 证据文件

- `outputs/official_sample_review/gt_vs_florence.png`
- `outputs/official_sample_review/official_gt_zoom.png`
- `outputs/official_sample_review/florence_pred_zoom.png`
- `outputs/official_sample_review/bbox_comparison.json`
