# Phase 1.8A 云端 4090 D1 Retention Probe 运行手册

## 任务边界

本轮只运行三条 D1 retention 候选的固定 probe/dev 对照：

- `D1_L010`：C2 + retention λ=0.10；
- `D1_L025`：C2 + retention λ=0.25；
- `D1_L050`：C2 + retention λ=0.50。

三者使用同一 Qwen3-VL-8B 视觉权重、全新 rank-48 TIR Adapter 初始化、Teacher Bank、4,096/1,024 固定切分、256 个负例、随机种子、优化器和 4,096 steps。唯一变量是 retention λ。

本轮禁止 full train、official val、AIC 测试集推理、Query/bbox、融合、Depth、第二模型 fallback 和平台提交。

## 云端预期路径

```text
/home/featurize/aic_rgbtir_phase18_bundle
/home/featurize/data/RGBT_GroundBench/extracted
/home/featurize/work/model_cache/Qwen3-VL-8B-Instruct
/home/featurize/aic_cloud/outputs/aic_rgbtir_phase18_d1_probe_v1
```

## 上传和解压

将本地 ZIP 上传到 `/home/featurize/`，然后执行：

```bash
cd /home/featurize
unzip -q aic_rgbtir_phase18_d1_cloud_bundle_20260814.zip
test -f /home/featurize/aic_rgbtir_phase18_bundle/BUNDLE_MANIFEST.json
test -d /home/featurize/data/RGBT_GroundBench/extracted/image_data
```

## 安装环境

```bash
bash /home/featurize/aic_rgbtir_phase18_bundle/tools/cloud/install_rgbtir_phase18_env.sh
```

环境目录固定为 `/home/featurize/work/envs/aic-rgbtir-phase18`，模型缓存放在持久盘，不写入临时工程目录。

## 正式执行

建议在 `tmux` 中运行：

```bash
tmux new -s phase18
bash /home/featurize/aic_rgbtir_phase18_bundle/tools/cloud/run_rgbtir_phase18_probe.sh
```

后台恢复：

```bash
tmux attach -t phase18
```

日志：

```text
/home/featurize/aic_cloud/logs/rgbtir_phase18_d1_probe.log
```

断点续跑由 `--resume` 控制；候选 summary 与 adapter 同时存在且 Teacher Bank fingerprint 一致时才允许复用。

## 完成验收

必须存在：

```text
/home/featurize/aic_cloud/outputs/aic_rgbtir_phase18_d1_probe_v1/phase18_probe_summary.json
/home/featurize/aic_cloud/outputs/aic_rgbtir_phase18_d1_probe_v1/safety_equivalence.json
/home/featurize/aic_cloud/outputs/aic_rgbtir_phase18_d1_probe_v1/environment.json
/home/featurize/aic_cloud/outputs/aic_rgbtir_phase18_d1_probe_v1/sha256_manifest.json
/home/featurize/aic_cloud/reports/aic_rgbtir_phase18_d1_probe_2026_08_14.md
```

`PHASE_18_D1_GO` 只表示至少一个 λ 同时降低 L8/L16/L24 drift、使 L24 drift 至少降低 25%，并保留 C2 检索和有效秩。它只授权下一轮 full train，不代表 AIC 平台会提升。

## 必须回传

回传整个输出目录、报告和日志。不要只回传 adapter；否则无法复核公平性、门禁与环境。
