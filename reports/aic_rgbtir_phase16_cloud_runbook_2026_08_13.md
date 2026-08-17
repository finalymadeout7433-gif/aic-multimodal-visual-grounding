# AIC RGB–TIR Phase 1.6 云端运行手册

## 1. 上传与解压

本地云端启动包：

```text
D:\12525\Documents\pytorch\aic_rgbtir_phase16_prerental_bundle_20260813.zip
```

上传后解压到：

```text
/home/featurize/aic_rgbtir_phase16_bundle
```

RGBT-GroundBench 数据挂载或解压路径必须为：

```text
/home/featurize/data/RGBT_GroundBench/extracted
```

该目录下应存在 `image_data/`。

## 2. 安装或复用环境

```bash
cd /home/featurize/aic_rgbtir_phase16_bundle/repo
bash tools/cloud/install_rgbtir_phase16_env.sh
```

脚本优先复用已验证的 Phase 1 环境；不存在时才创建隔离环境并安装依赖。

## 3. 启动可恢复全流程

推荐在 tmux 内启动：

```bash
tmux new -s phase16
cd /home/featurize/aic_rgbtir_phase16_bundle/repo
bash tools/cloud/run_rgbtir_phase16_gated.sh
```

按 `Ctrl+B`，再按 `D` 可退出 tmux 而不中断任务。重新连接：

```bash
tmux attach -t phase16
```

脚本执行顺序：

```text
build-manifests
→ preflight
→ 下载并核验固定 revision 的 Qwen 视觉 shard
→ RGB Teacher Bank / Base-TIR Bank
→ C0/C1/C2 repair probe
→ 胜者 fresh-init full retrain
→ repair-dev checkpoint 选择
→ 2,032 条 official val
→ PHASE_16_GO / PHASE_16_NO_GO
```

网络或 SSH 中断后，重新运行同一命令即可；`--resume` 会复用 fingerprint 一致的 Teacher Bank、候选结果和梯度累积边界 checkpoint。

## 4. 监控

```bash
tail -f /home/featurize/aic_cloud/logs/aic_rgbtir_phase16_v1/03_phase16_all.log
nvidia-smi
du -sh /home/featurize/aic_cloud/outputs/aic_rgbtir_phase16_v1
```

## 5. 结果判定

最终查看：

```text
/home/featurize/aic_cloud/outputs/aic_rgbtir_phase16_v1/official_val/phase16_gate.json
/home/featurize/aic_cloud/reports/aic_rgbtir_phase16_repair_validation_2026_08_13.md
```

只有 `phase16_gate.json` 中 `status=PHASE_16_GO` 才能进入 Phase 2。`PHASE_16_NO_GO` 时不得为追求进度跳过门禁，也不得让另一个模型 fallback 代替红外分支。
