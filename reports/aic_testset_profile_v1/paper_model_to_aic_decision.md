# 论文与开源模型到 AIC 的路线决策

> 证据边界：Query 与图像统计是输入事实；模型框面积、一致性和“小目标”分级是无标签代理；模型优劣与 ACC 必须等待平台验证。


## 按 AIC 用途排序（不是按论文榜单排序）

| 路线 | 已核验公开资产 | AIC 主要覆盖 | 本轮决定 |
|---|---|---|---|
| [PIZA + SOREC（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Goto_Referring_Expression_Comprehension_for_Small_Objects_ICCV_2025_paper.html) | [代码、标注与 adapter](https://github.com/mmaiLab/sorec) | 极小目标、渐进式缩放 | `not_triggered`；只有尺度探针也达到冻结阈值才优先 |
| [APE-Ti（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Aligning_and_Prompting_Everything_All_at_Once_for_Universal_Visual_CVPR_2024_paper.html) | [推理、训练与 Ti/L 权重](https://github.com/shenyunhang/APE) | 区域、stuff、建筑结构、复杂句子 | `triggered` |
| [MM-Grounding-DINO-T](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino) | 公开训练配置与 T/B/L checkpoint | 多域训练和可复现微调 | 无单一难点主导时先做验证，不在本轮训练 |
| [RGBT-GroundBench/VGNet](https://arxiv.org/abs/2512.24561) | [代码](https://github.com/crazyxiaoxi/RGBT-GroundBench)、[数据](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset)、[模型资源](https://huggingface.co/JiawenXi/RGBT-Ground-Model) | RGB+TIR+Query、低光 | `not_triggered`；先做许可、文件—配置映射和近重复审计，本轮不下载大模型包 |
| [LLMDet（CVPR 2025 Highlight）](https://openaccess.thecvf.com/content/CVPR2025/html/Fu_LLMDet_Learning_Strong_Open-Vocabulary_Object_Detectors_under_the_Supervision_of_CVPR_2025_paper.html) | [Swin-T/B/L 代码与权重](https://github.com/iSEE-Laboratory/LLMDet) | 长句、属性、角色语义 | deferred_until_long_query_role_cluster_is_validated |
| [RGBDT500/RDTTrack（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b4962fcd5d4410a9f43ef70f528eedd8-Abstract-Datasets_and_Benchmarks_Track.html) | 三模态跟踪数据、代码与权重 | RGB/Depth/TIR 融合结构 | 任务无语言；只借鉴结构，重叠审计前不训练 |

## 纠正两个容易误读的结论

- PIZA、APE-Ti 和 MM-GDINO-T 都是 RGB+文本路线，不能替代 IR/Depth；它们分别解决尺度、区域语义和可训练基座问题。
- RGBDT500 的视觉模态最接近 AIC，但其监督是单目标跟踪，不是自然语言指代定位；“有三模态 bbox”不等于“可直接训练 AIC 模型”。

## 本机外部验证资产状态

```json
{
  "grefcoco": {
    "path": "D:\\AIC赛题一数据集\\manifests\\processed\\grefcoco_validation.jsonl",
    "records": 5324,
    "sha256": "6FFBCECC736E845BFE6127D6B46AD8D0FDAD48AEBBA331B06F682112DC9EA347",
    "status": "available"
  },
  "refcoco": {
    "path": "D:\\AIC赛题一数据集\\manifests\\processed\\refcoco_validation.jsonl",
    "records": 10834,
    "sha256": "8D60DB928D6223C5805874A904C29671B065E3F5A54C25297E4C7871904DDFC0",
    "status": "available"
  },
  "refcoco_plus": {
    "path": "D:\\AIC赛题一数据集\\manifests\\processed\\refcoco_plus_validation.jsonl",
    "records": 10758,
    "sha256": "56B6059E3DBBCA1C565540444838116810DDC61DDFDC844C5B5B63714005C653",
    "status": "available"
  },
  "refcocog": {
    "path": "D:\\AIC赛题一数据集\\manifests\\processed\\refcocog_validation.jsonl",
    "records": 4896,
    "sha256": "A9866D0A51CAE68F938546AD06D236DCFBDE5F9BEB35A4B00764916E67571D2C",
    "status": "available"
  },
  "rgbt_groundbench": {
    "path": null,
    "status": "missing_not_verified"
  },
  "sorec": {
    "path": null,
    "status": "missing_not_verified"
  }
}
```

`missing_not_verified` 只表示本机未发现对应 manifest，本轮没有下载或伪造数据；不表示官方资产不存在。
