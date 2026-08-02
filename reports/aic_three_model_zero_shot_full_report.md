# AIC 三模型零训练全量检测与平台提交报告

日期：2026-08-02（Asia/Shanghai）

## 结论

APE-Ti、MM-Grounding-DINO-T 和 LLMDet-Swin-T 均已使用原始 AIC Visible RGB
与原始英文 Query 完成 9,555 条零训练全量检测。三份预测均采用模型原生最高分
Top-1，不使用外部验证集筛选、不做候选重排、不加入人工规则、不使用 AIC
伪标签训练，也未自动上传平台。

三份平台 ZIP 已通过统一审计：Query ID 精确一致、非 bbox 字段修改为 0、
非法框为 0，且每个 ZIP 只包含一个 `predictions_submission.json`。
最终包进一步统一为 LF JSON 和固定 ZIP 时间戳；相同预测在当前受控打包环境中
重复生成时将得到逐字节一致的 ZIP。

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

## 平台决策

三份 ZIP 应作为三个独立单模型实验分别手动上传。若只能先上传一个，优先 APE-Ti：
它与现有 GroundingDINO 系输出差异最大，能最快回答“区域/结构型通用 grounding
模型是否更适合 AIC”。随后上传 MM-Grounding-DINO-T 和 LLMDet-Swin-T。

只有平台 ACC 返回后才能确定最优基座：

1. 最高分模型成为下一轮唯一控制基线；
2. 下一轮只增加一个变量，例如 PIZA 小目标分支、提高分辨率或 Depth late fusion；
3. 不在平台比较前训练 selector，也不将三个模型直接混合成一次提交；
4. 若三个新模型均不超过 Florence 0.4980，则继续保留 Florence 为稳定基线，
   并优先测试真正改变小目标机制的 PIZA，而不是继续更换相似的检测器。

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
