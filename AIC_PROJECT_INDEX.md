# AIC 项目导航

这个文件是给 GitHub 浏览用的入口。以后如果找不到某个模型实现、报告或提交包位置，优先从这里进入。

## 先看这三个文件

| 你要找什么 | 打开哪个文件 |
|---|---|
| 当前分数排行榜、哪一轮最高 | [`reports/leaderboard_results.md`](reports/leaderboard_results.md) |
| 本地提交包 ZIP 在哪里 | [`reports/submission_package_index.md`](reports/submission_package_index.md) |
| 当前最高分 Qwen3-VL-8B 的详细记录 | [`reports/qwen3_vl_8b_platform_result_2026_08_07.md`](reports/qwen3_vl_8b_platform_result_2026_08_07.md) |
| 上一轮 0.7210 是哪个模型 | [`reports/locateanything_3b_platform_result_2026_08_05.md`](reports/locateanything_3b_platform_result_2026_08_05.md) |
| EGM 为什么只有 0.5333 | [`reports/egm_qwen3_vl_8b_platform_postmortem_2026_08_09.md`](reports/egm_qwen3_vl_8b_platform_postmortem_2026_08_09.md) |
| 下一轮为什么测 Qwen3-VL-30B-A3B | [`reports/qwen3_vl_30b_a3b_next_model_decision_2026_08_09.md`](reports/qwen3_vl_30b_a3b_next_model_decision_2026_08_09.md) |

## 当前最高分

| 模型 | 平台 ACC@0.5 | 本地提交包 |
|---|---:|---|
| Qwen3-VL-8B-Instruct zero-shot | **0.7582** | `D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\platform_upload_ready\AIC_Qwen3_VL_8B_Instruct_zero_shot_20260807.zip` |
| LocateAnything-3B zero-shot | 0.7210 | `D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\platform_upload_ready\AIC_LocateAnything_3B_zero_shot_v1.zip` |

## 代码目录怎么读

| 目录 | 用途 |
|---|---|
| `tools/cloud/` | 云端模型推理脚本，例如 Qwen3-VL、LocateAnything 的云端入口 |
| `src/aic_baseline/` | 本地 baseline 工程代码：数据读取、bbox、提交包审计等 |
| `configs/` | 配置模板。本地私有路径配置通常是 `.local.yaml`，不会提交 |
| `reports/` | 每一轮实验结论、平台分数、模型路线和问题复盘 |
| `outputs/` | 本地生成结果，Git 忽略。GitHub 上不会看到里面的大量预测文件 |

## 为什么 GitHub 上看不到 ZIP 或输出目录

`outputs/`、模型权重和云端大文件都被 Git 忽略。这是有意设计：

- 避免把比赛提交包、预测明细、模型权重、大数据上传到 GitHub；
- GitHub 只保存可复现代码、报告、哈希和本地路径；
- 真正要上传平台的 ZIP 位置记录在 [`reports/submission_package_index.md`](reports/submission_package_index.md)。

## 按任务找实现

| 任务 | 关键文件 |
|---|---|
| Qwen3-VL-8B 全量 AIC 推理 | `tools/cloud/run_qwen3vl_aic.py` |
| Qwen3-VL 云端环境安装 | `tools/cloud/install_qwen3vl_env.sh` |
| LocateAnything 云端流程 | `reports/locateanything_3b_platform_result_2026_08_05.md` 和 `tools/cloud/` |
| 三模型 APE/MM-GDINO/LLMDet 对照 | `reports/aic_three_model_zero_shot_full_report.md` |
| GroundingDINO Ranker 失败复盘 | `reports/gdino_spatial_ltr_v1_platform_postmortem.md` |
| 数据集审计结论 | `reports/dataset_audit_summary.md` |
| Qwen3-VL-30B-A3B 下一轮决策 | `reports/qwen3_vl_30b_a3b_next_model_decision_2026_08_09.md` |

## 后续路线

当前主线不再是传统检测器或 RefCOCO LightGBM Ranker，而是：

1. 以 Qwen3-VL-8B-Instruct `0.7582` 作为当前主控制基线；
2. 放弃 EGM-Qwen3-VL-8B 作为主线，只保留小规模 prompt/token 诊断；
3. 下一轮优先测试 Qwen3-VL-30B-A3B-Instruct；
4. 确认更强单模型收益后，再考虑多模型融合或多模态 IR/Depth。
