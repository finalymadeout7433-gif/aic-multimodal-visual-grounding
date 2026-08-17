# 2026 AIC 多模态视觉定位 Baseline

## 先从这里看

如果是在 GitHub 页面浏览本仓库，优先打开：

- [RGB–TIR 红外模块持续更新技术报告](reports/AIC_RGB_TIR_LIVING_TECHNICAL_REPORT.md)
- [RGB–TIR 模块代码与复现入口](src/aic_rgbtir/README.md)
- [Phase 1.9-D2 Full 结果与未通过原因](reports/aic_rgbtir_phase19_d2_full_result_analysis_2026_08_17.md)
- [AIC 项目导航](AIC_PROJECT_INDEX.md)
- [排行榜结果](reports/leaderboard_results.md)
- [提交包索引](reports/submission_package_index.md)
- [Qwen3-VL-8B 平台结果与下一轮模型路线](reports/qwen3_vl_8b_platform_result_2026_08_07.md)

当前最高分：`Qwen3-VL-8B-Instruct zero-shot = 0.7582`。

对应本地提交包：

```text
D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\platform_upload_ready\AIC_Qwen3_VL_8B_Instruct_zero_shot_20260807.zip
```

上一轮 `0.7210` 的模型是 `nvidia/LocateAnything-3B`，它是当前第二高分的零训练模型。对应报告与提交包：

- [LocateAnything-3B 平台结果报告](reports/locateanything_3b_platform_result_2026_08_05.md)
- 本地提交包：

```text
D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\platform_upload_ready\AIC_LocateAnything_3B_zero_shot_v1.zip
```

第一版可完整运行的 RGB-only 基线：

```text
Visible + Query → Florence-2-large-ft → 归一化 bbox → 提交 JSON/ZIP
```

当前版本：`v0.2.1`。正式初赛数据没有 bbox，只用于推理，不用于训练。

## 诊断与 Ranker 实验分支

当前实验分支记录了两轮受控工作：

1. Florence 候选上限、Tile 小目标与 GroundingDINO-Tiny 零样本诊断；
2. GroundingDINO Top-10 + LightGBM Spatial LTR 候选排序训练。

对应报告：

- [诊断实验报告](reports/diagnostic_oracle_tile_gdino_round.md)
- [Spatial LTR 训练报告](reports/gdino_spatial_ltr_v1_training_report.md)
- [平台结果与负迁移复盘](reports/gdino_spatial_ltr_v1_platform_postmortem.md)
- [排行榜记录](reports/leaderboard_results.md)

平台结果必须和外部本地验证分开解释：

| 系统 | 外部 holdout ACC@0.5 | AIC 平台 ACC@0.5 |
|---|---:|---:|
| Florence-2 RGB-only first | 未在本轮 holdout 重测 | **0.4980** |
| GroundingDINO-Tiny Top-1 | 0.5211 | **0.4938** |
| GroundingDINO + Spatial LTR | 0.6333 | **0.3190** |

Spatial LTR 的外部提升没有迁移到 AIC，因此该模型被标记为研究失败，不是推荐提交。
代码仍保留，用于复现负迁移、改进目标/参照物建模和设计保守切换消融。

## 三个开源模型零训练全量对照

APE-Ti、MM-Grounding-DINO-T 和 LLMDet-Swin-T 均已使用 Visible RGB 与原始
英文 Query 完成 9,555 条 AIC 测试集零训练推理。三者均使用模型原生 Top-1，
没有使用 AIC 图像、Query、候选框或伪标签训练。

| 模型 | 基础定位 | AIC 平台 ACC@0.5 | 相对 Florence-2 |
|---|---|---:|---:|
| MM-Grounding-DINO-T | OpenMMLab 的可训练统一检测/grounding 基座 | **0.5011** | **+0.31 pp** |
| Florence-2-large-ft | 当前 RGB-only 控制基线 | 0.4980 | — |
| LLMDet-Swin-T | 在 MM-GDINO 上加入长描述/LLM 监督 | 0.4942 | -0.38 pp |
| APE-Ti | 同时覆盖前景物体、background stuff 和区域描述 | 0.4424 | -5.56 pp |

`0.4942` 的平台结果属于 LLMDet-Swin-T；MM-Grounding-DINO-T 的正确平台结果
是 `0.5011`。该归属在 2026-08-02 根据用户提供的平台截图完成纠正。
MM-Grounding-DINO-T 因此成为当前已验证的最高分单模型，但领先 Florence-2
只有 0.31 个百分点，后续仍应采用单变量平台实验确认改动收益。

对应资料：

- [三模型基础资料、环境、权重指纹与全量报告](reports/aic_three_model_zero_shot_full_report.md)
- [无标签预测行为对照](reports/aic_three_model_zero_shot_behavior.md)
- [完整排行榜记录](reports/leaderboard_results.md)

## 安装

```powershell
conda activate aic-baseline
python -m pip install -e .
```

已验证环境：

- Python 3.11
- PyTorch 2.5.1+cu121
- Transformers 4.57.6
- RTX 4060 Laptop GPU

## 配置

复制公开模板，本机路径只写入 `.local.yaml`：

```powershell
Copy-Item configs\florence2_rgb_only.example.yaml configs\florence2_rgb_only.local.yaml
Copy-Item configs\official_sample.example.yaml configs\official_sample.local.yaml
```

编辑两个本地配置中的数据集、模型和输出路径。本地配置已被 Git 忽略。

## 验证与推理

```powershell
# 自动化测试
python -m pytest -q

# 检查 9,555 条 Query 及文件路径
python -m aic_baseline.cli validate --config configs\florence2_rgb_only.local.yaml

# 官方单样例 sanity check
python -m aic_baseline.cli sample --config configs\official_sample.local.yaml

# 10 条 smoke test
python -m aic_baseline.cli infer `
  --config configs\florence2_rgb_only.local.yaml `
  --limit 10 `
  --output-dir outputs\smoke_10

# 全量推理、断点续跑并生成提交文件
python -m aic_baseline.cli infer `
  --config configs\florence2_rgb_only.local.yaml `
  --build-submission `
  --resume
```

只有完整处理全部 Query 后才允许生成提交 ZIP。

`v0.2.0` 已完成正式 9,555 条 Query 的 Florence-2 RGB-only 全量推理；
本地输出位于 Git 忽略的 `outputs/florence2_rgb_only_full`，提交前应查看其中的
`submission_audit.json`。

平台结果：Florence-2 RGB-only v0.2.0 的 ACC@0.5 为 `0.4980`。详见
[排行榜记录](reports/leaderboard_results.md)。

## 外部训练数据

外部数据保存在本机 D 盘，不提交 Git。预处理工具按以下顺序执行完整 ZIP 校验、
安全解压、标注引用核对、图像隔离 split 和可视化：

```powershell
python tools\prepare_external_data.py `
  --data-root "D:\AIC赛题一数据集" all
```

`all` 仅适用于两个原始大 ZIP 仍存在的首次准备过程。2026-07-29 完整核验后，
本机已删除 `train2014.zip` 和 `SUNRGBD.zip`，保留解压数据并释放 18.995 GiB。
当前重新生成清单应使用：

```powershell
python tools\prepare_external_data.py `
  --data-root "D:\AIC赛题一数据集" prepare
```

后续训练代码可读取 `manifests/processed/rgb_core_v1_*.jsonl`，通过
`GroundingManifestDataset` 懒加载图像、Query 和归一化 bbox。数据统计见
[外部数据准备报告](reports/external_data_preparation_report.md)，下一模型与训练边界见
[模型和训练策略](reports/model_and_training_strategy.md)。

当前仓库已完成训练数据清单和加载接口；GroundingDINO 的 processor、collator、
训练循环、验证与 checkpoint 入口尚未实现，因此本版本不称为“一键训练完成”。

## 数据审计

```powershell
python tools\audit_dataset.py `
  --dataset-root "D:\path\to\aic-dataset"
```

核心结论见 [数据集审计摘要](reports/dataset_audit_summary.md)。

## 项目结构

```text
configs/   公开配置模板
src/       数据、模型、推理和提交代码
tests/     自动化测试
tools/     数据审计和外部数据准备工具
reports/   实验、排行榜、数据和训练策略结论
outputs/   本地输出，不上传 Git
```

当前限制：已实验 RGB 候选框重排序，但尚未使用 Infrared、Depth 或 Query 改写；
现有 Spatial LTR 在 AIC 平台发生严重负迁移，不能作为生产提交策略。
