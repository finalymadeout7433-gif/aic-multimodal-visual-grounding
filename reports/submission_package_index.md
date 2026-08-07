# AIC 平台提交包索引

目的：集中记录每一轮可上传 ZIP 的本地路径、平台分数和哈希，避免后续找不到提交包或误归属分数。

> 注意：这些 ZIP 是生成结果，通常位于 Git 忽略目录或仓库外部，不提交到 GitHub。GitHub 只保存路径、哈希、审计和分数记录。

## 当前最高分

| 模型 / 策略 | 平台 ACC@0.5 | 本地 ZIP |
|---|---:|---|
| Qwen3-VL-8B-Instruct zero-shot | **0.7582** | `D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\platform_upload_ready\AIC_Qwen3_VL_8B_Instruct_zero_shot_20260807.zip` |

## 全部已知提交包

| 轮次 | 模型 / 策略 | 平台 ACC@0.5 | 平台时间 | ZIP 字节数 | SHA-256 | 本地路径 |
|---|---|---:|---|---:|---|---|
| S01 | Florence-2 RGB-only v0.2.0 | 0.4980 | 2026-07-29 18:52:09 | 385,189 | `52CFF325FBB0841D5E670645965F99BEF46C580BB0000798FC4CE7FE18081393` | `D:\12525\Documents\pytorch\baseline_v0\outputs\florence2_rgb_only_full\predictions_submission.zip` |
| S02 | GroundingDINO-Tiny Top-1 | 0.4938 | 2026-08-01 01:09:49 | 642,640 | `99F06B050E5478212A26D233F365CCA29B7FF3EA44CF73E171F0EEE4F1B706E5` | `D:\12525\Documents\pytorch\baseline_v0\outputs\gdino_spatial_ltr_v1\platform_upload_ready\S02_gdino_top1_control.zip` |
| S03 | GroundingDINO + Spatial LTR | 0.3190 | 2026-07-31 23:52:51 | 642,507 | `23819AE2728B7502BE53384C2F5F906EB7F7ED45ACF95EF74E92D10FB4A702CF` | `D:\12525\Documents\pytorch\baseline_v0\outputs\gdino_spatial_ltr_v1\platform_upload_ready\S03_gdino_spatial_ltr_v1.zip` |
| S04 | GroundingDINO Conservative LTR | 0.4957 | 2026-08-01 13:12:35 | 642,739 | `002B2399AA14B66695C3DABE162FF63186519CD3E81E99ECCEC59CBC26FF2831` | `D:\12525\Documents\pytorch\baseline_v0\outputs\gdino_conservative_s04\platform_upload_ready\S04_gdino_conservative_ltr_v1.zip` |
| M01 | APE-Ti zero-shot | 0.4424 | 2026-08-02 13:29:14 | 402,169 | `5618E32ED4A3CE0786C80004E98E8720A871A71C313965CBDBE14BC41B5B1189` | `D:\12525\Documents\pytorch\baseline_v0\outputs\aic_zero_shot_full_v1\platform_upload_ready\AIC_APE_Ti_zero_shot_v1.zip` |
| M02 | LLMDet-Swin-T zero-shot | 0.4942 | 2026-08-02 14:44:19 | 641,937 | `27090EDE56141789415E474951FEE6D991C5D5014179F925F4276BA2958E68D9` | `D:\12525\Documents\pytorch\baseline_v0\outputs\aic_zero_shot_full_v1\platform_upload_ready\AIC_LLMDet_Swin_T_zero_shot_v1.zip` |
| M03 | MM-Grounding-DINO-T zero-shot | 0.5011 | 2026-08-02 16:07:02 | 642,700 | `302572F8E0922ACE40475F1D349BD9E5477022B16432C9E8B4254A1F4C5F599E` | `D:\12525\Documents\pytorch\baseline_v0\outputs\aic_zero_shot_full_v1\platform_upload_ready\AIC_MM_Grounding_DINO_T_zero_shot_v1.zip` |
| M04 | LocateAnything-3B zero-shot | 0.7210 | 2026-08-05 21:31:36 | 324,710 | `4763A3627FFE9E11898E9205C967F7FAB41482AD7464C6B07B87ABF55859C45C` | `D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\platform_upload_ready\AIC_LocateAnything_3B_zero_shot_v1.zip` |
| M05 | Qwen3-VL-8B-Instruct zero-shot | **0.7582** | 2026-08-07 19:28:19 | 336,716 | `D3D1757BA6052C117B13DC8175B21475DF22B084D2465570B5E998C1D78E1188` | `D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\platform_upload_ready\AIC_Qwen3_VL_8B_Instruct_zero_shot_20260807.zip` |

## 找文件的最短路径

1. 先打开本文件。
2. 找到对应模型行。
3. 复制“本地路径”到 Windows 文件资源管理器地址栏。
4. 上传平台时只选 `.zip` 文件，不要上传同目录下的 `.jsonl`、`run_summary.json` 或其它调试文件。

## 上传前复核

对任意 ZIP，可用 PowerShell 复核：

```powershell
Get-FileHash -Algorithm SHA256 "D:\path\to\submission.zip"
```

结果应与本表 SHA-256 一致。
