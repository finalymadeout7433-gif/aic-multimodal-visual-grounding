# AIC 三模型零训练全量检测与平台提交报告

日期：2026-08-02（Asia/Shanghai）

## 结论

APE-Ti、MM-Grounding-DINO-T 和 LLMDet-Swin-T 均已使用原始 AIC Visible RGB
与原始英文 Query 完成 9,555 条零训练全量检测。三份预测均采用模型原生最高分
Top-1，不使用外部验证集筛选、不做候选重排、不加入人工规则、不使用 AIC
伪标签训练，也未自动上传平台。

三份平台结果已经返回：MM-Grounding-DINO-T 为 **0.5011**，LLMDet-Swin-T 为
**0.4942**，APE-Ti 为 **0.4424**。MM-Grounding-DINO-T 以 0.31 个百分点超过
Florence-2 的 0.4980，成为当前最高分单模型控制基线。

三份平台 ZIP 已通过统一审计：Query ID 精确一致、非 bbox 字段修改为 0、
非法框为 0，且每个 ZIP 只包含一个 `predictions_submission.json`。
最终包进一步统一为 LF JSON 和固定 ZIP 时间戳；相同预测在当前受控打包环境中
重复生成时将得到逐字节一致的 ZIP。

## 模型基础资料

| 模型 | 论文/项目定位 | 当前使用权重 | 主要能力与本轮作用 | 官方来源 |
|---|---|---|---|---|
| APE-Ti | CVPR 2024 通用视觉感知模型 | APE-Ti 官方 checkpoint | 同时定位前景物体、background stuff 和区域描述；用于检验 AIC 区域/结构 Query 假设 | [APE](https://github.com/shenyunhang/APE) |
| MM-Grounding-DINO-T | OpenMMLab 的统一开放词汇检测、Phrase Grounding 与 REC 基座 | `openmmlab-community/mm-grounding-dino-tiny-o365v1-goldg` | 训练配置与 T/B/L 权重完整；用于检验可复现 GDINO 改进是否迁移到 AIC | [MMDetection 配置](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino) |
| LLMDet-Swin-T | CVPR 2025 Highlight 开放词汇检测器 | `fushh7/llmdet-swin-tiny-hf` 完整安全重封装 | 在 MM-GDINO 基础上加入长描述和大语言模型监督；用于检验语义监督是否改善 AIC 长 Query | [LLMDet](https://github.com/iSEE-Laboratory/LLMDet) |

上述三模型在本轮都只接收 Visible RGB 和 Query。它们不是 AIC 完整 RGB、Infrared、
Depth 多模态方案；本轮结果只比较 RGB grounding 基座能力。

## 平台上传包

| 模型 | 文件 | SHA-256 | 大小 |
|---|---|---|---:|
| APE-Ti | `AIC_APE_Ti_zero_shot_v1.zip` | `5618E32ED4A3CE0786C80004E98E8720A871A71C313965CBDBE14BC41B5B1189` | 402,169 bytes |
| MM-Grounding-DINO-T | `AIC_MM_Grounding_DINO_T_zero_shot_v1.zip` | `302572F8E0922ACE40475F1D349BD9E5477022B16432C9E8B4254A1F4C5F599E` | 642,700 bytes |
| LLMDet-Swin-T | `AIC_LLMDet_Swin_T_zero_shot_v1.zip` | `27090EDE56141789415E474951FEE6D991C5D5014179F925F4276BA2958E68D9` | 641,937 bytes |

统一目录：

```text
outputs/aic_zero_shot_full_v1/platform_upload_ready/
```

其中 `release_manifest.json` 记录源包、输出包、大小、Query 数与哈希；
`SHA256SUMS.txt` 可用于上传前复核。

## 平台结果

| 模型 | AIC ACC@0.5 | 相对 Florence-2 | 平台记录时间 |
|---|---:|---:|---|
| MM-Grounding-DINO-T | **0.5011** | **+0.31 pp** | 2026-08-02 16:07:02 |
| LLMDet-Swin-T | **0.4942** | -0.38 pp | 2026-08-02 14:44:19 |
| APE-Ti | **0.4424** | -5.56 pp | 2026-08-02 13:29:14 |

分数归属说明：最初曾因用户口头表述将 `0.4942` 误写为 MM-Grounding-DINO-T；
用户随后提供新截图并纠正。正确映射是 `0.4942 = LLMDet-Swin-T`、
`0.5011 = MM-Grounding-DINO-T`。三份模型目录、预测文件、哈希和提交 ZIP 从未
混用，修正仅涉及平台结果的文字归属。

## 输入与推理边界

- AIC Query：9,555 条；图像组：2,000 组。
- `queries.json` SHA-256：
  `2A08CD3A930D9749EDB86A8B420D8259DCE446116AD98D93338EDF8B3E1EF67C`。
- 输入模态：Visible RGB + 原始 Query。
- 未向这三个 RGB grounding 模型伪造 IR 或 Depth 输入。
- 候选阈值设为 0，仅用于保证保留模型产生的候选；最终仍取原生最高分 Top-1。
- 每条保存 Top-20 调试候选，但提交 JSON 只包含最终 bbox。
- AIC 没有公开 GT，因此本报告不报告本地 ACC、oracle 或“救回数量”。

## 模型、环境与运行结果

### APE-Ti

- 官方 APE commit：`8c4920e2014d4818ec3ecafb7b526d896a659d1c`。
- checkpoint SHA-256：
  `B5D793E960515A6D1AA4B8A55B61DA990B0B4184B510D3D7AD3BB37526FB8007`。
- 权重已在完成后重新读取并核对 SHA-256；实际 APE 补丁目标文件、Git HEAD、
  完整 tracked diff 和反向补丁检查另存于
  `ape_ti/post_run_provenance_audit.json`。该文件属于完成后的只读复核，
  不冒充原始运行时快照。
- checkpoint 装载：0 missing / 0 unexpected / 0 incorrect shape，strict-equivalent。
- 环境：WSL2 Ubuntu 22.04、Python 3.10.20、PyTorch 1.12.1+cu116。
- 推理：FP16 autocast、batch size 1、bbox-only。
- bbox-only 只关闭比赛不需要的 semantic/panoptic/mask 输出；已在同一 AIC
  Query 上核对原路径与 bbox-only 路径的 Top-1 bbox 和 score 完全相同。
- 完成：9,555/9,555；耗时 7,717.37 秒；空候选 0；非法框 0。
- 运行期间在温度接近驱动热控上限时执行了四次 45–60 秒进程暂停冷却；
  未重载模型、未改变预测配置或断点指纹。
- 预测 JSON SHA-256：
  `1F3AB3A935F798E86C28351D3E89051EE2CFAB24D0EE510C6DA83D169A979054`。

### MM-Grounding-DINO-T

- 模型：`openmmlab-community/mm-grounding-dino-tiny-o365v1-goldg`。
- weight SHA-256：
  `2D9BDFD0247F167A00E00FFFA1DEC1EE5D5861FDFB4E27FD8CC2279AF6820688`。
- 环境：Windows、Python 3.11.15、PyTorch 2.5.1+cu121、Transformers 4.57.6。
- 推理：FP32、无 autocast、batch size 1。
- 完成：9,555/9,555；耗时 9,270.12 秒；空候选 0；非法框 0。
- 预测 JSON SHA-256：
  `1DA3F5764A2B3F183BCC1E45A8675CC801E11BF77AEEBFA4D24CDEA6CC3C104F`。

### LLMDet-Swin-T

- 模型：`fushh7/llmdet-swin-tiny-hf` 完整 safetensors 转换包。
- weight SHA-256：
  `1D9E7CB0C53DB1A8EA6D26D23248247F489FBD6F64302821CCA64FB557653A3B`。
- 环境：Windows、Python 3.11.15、PyTorch 2.5.1+cu121、Transformers 4.57.6。
- 推理：FP32、无 autocast、batch size 1。
- 官方自定义模型在本机缺少 Windows deformable-attention CUDA kernel，按官方
  代码自动回退到纯 PyTorch 实现；推理有效但速度较慢。
- 完成：9,555/9,555；耗时 8,853.14 秒；空候选 0；非法框 0。
- 预测 JSON SHA-256：
  `97D2C75408DF951C9126BF3FA8F249B2157F7801687A2AED649E4BDF057346CE`。

## 无标签模型行为

以下都是模型预测代理，不是真实目标尺寸或准确率。

| 模型 | 预测框面积中位数 | 面积 <0.1% | 面积 <1% | 面积 >=25% |
|---|---:|---:|---:|---:|
| APE-Ti | 1.5215% | 430 | 3,892 | 520 |
| MM-Grounding-DINO-T | 2.2794% | 341 | 3,251 | 798 |
| LLMDet-Swin-T | 2.2927% | 408 | 3,276 | 980 |

两两预测 IoU>=0.5：

| 模型对 | Query 数 | 比例 |
|---|---:|---:|
| APE-Ti vs MM-Grounding-DINO-T | 4,133 | 43.25% |
| APE-Ti vs LLMDet-Swin-T | 4,144 | 43.37% |
| MM-Grounding-DINO-T vs LLMDet-Swin-T | 6,672 | 69.83% |

MM-Grounding-DINO-T 与 LLMDet-Swin-T 的输出高度相似，说明二者在 AIC 上
可能共享较强的候选偏好；APE-Ti 的输出更独立。这个结果只说明实验信息互补性，
不能推出 APE 更好或更差。

## 平台结论与下一步

1. **MM-Grounding-DINO-T（0.5011）成为当前最高分单模型控制基线。** 它相对
   Florence-2 的领先只有 0.31 pp，说明当前是小幅但真实的平台改进，不应夸大为
   架构性突破。
2. **LLMDet-Swin-T（0.4942）没有超过其 MM-GDINO 基座。** 本地两模型输出又
   高度相似，因此当前没有证据支持直接扩大到 LLMDet-B/L 或启动全量训练。
3. **APE-Ti（0.4424）明显落后。** AIC 区域/结构 Query 占比较高这一输入画像仍然
   成立，但不能再由此推出 APE-Ti 是更优平台模型；无标签画像只能指出难点，不能
   替代真实平台模型选择。
4. 下一份平台实验继续坚持单变量原则。优先围绕 MM-Grounding-DINO-T 测试受控
   backbone/分辨率变化，或测试真正改变 Grounding 机制的新基座；PIZA、IR、Depth
   和新 selector 不得一次性混入同一提交。

## 机器产物

```text
outputs/aic_zero_shot_full_v1/
  ape_ti/
  mm_grounding_dino_t/
  llmdet_swin_t/
  platform_upload_ready/
  three_model_behavior_summary.json

reports/
  aic_three_model_zero_shot_behavior.md
  aic_three_model_zero_shot_full_report.md
```

每个模型目录均包含 `predictions.jsonl`、`runtime_events.jsonl`、
`run_fingerprint.json`、`run_summary.json`、`environment.json` 和独立提交审计。
