# 下一轮三个单变量实验

> 证据边界：Query 与图像统计是输入事实；模型框面积、一致性和“小目标”分级是无标签代理；模型优劣与 ACC 必须等待平台验证。


## 画像冻结后的具体顺序

1. **APE-Ti 单模型对照**：第一优先。只替换候选/定位模型，其余 Query、预处理和提交格式全部保持不变。
2. **PIZA/SOREC 小目标离线专项**：第二优先但暂不直接上平台。先补齐并核验 SOREC 资产，在公开有标签小目标集验证；AIC 上只做既定 probe，不训练。
3. **Depth late-fusion tracer**：第三优先。仅对显式 nearest/farthest/front/behind Query 使用真实 Depth 统计；PNG uint16 与 JPG uint8 未知域分开，且不得以二维位置代替深度。

LLMDet、RGBT-VGNet 和 MM-Grounding-DINO-T 均保留为后续路线：前两者分别等待长句角色失败证据和低光/IR 触发，MM-GDINO-T 等 APE/PIZA 单变量结果后再决定是否多域微调。

S04 结果回来后只更换控制基线，不反向修改本画像代理规则。下一份平台提交只能引入一个变量。
