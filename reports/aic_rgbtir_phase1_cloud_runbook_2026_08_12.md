# AIC RGB–TIR Phase 1 云端运行手册

## 租卡前

1. 在 Featurize“我的数据集”确认已经存在 `RGBT_GroundBench_official_archives_20260812.zip`。
2. 记录其挂载/展开路径；必须能找到 6 个 `ann_*.tar` / `data_*.tar`。
3. 准备本地工程包：

```text
D:\12525\Documents\pytorch\aic_rgbtir_phase1_prerental_bundle_20260812.zip
```

Phase 1 不需要 AIC 测试集。

## 开机后的固定步骤

以下命令均在云端 Linux 终端执行。

### 1. 解开工程包

```bash
mkdir -p /home/featurize/aic_rgbtir_phase1_bundle
unzip -q /path/to/aic_rgbtir_phase1_prerental_bundle_20260812.zip \
  -d /home/featurize/aic_rgbtir_phase1_bundle
```

### 2. 展开并校验 RGBT 数据

将 `/path/to/persistent_rgbt_archives` 替换为平台持久化数据集中 6 个 tar 所在目录：

```bash
bash /home/featurize/aic_rgbtir_phase1_bundle/repo/tools/cloud/extract_rgbt_groundbench.sh \
  /path/to/persistent_rgbt_archives \
  /home/featurize/data/RGBT_GroundBench/extracted
```

脚本会先核对 6 个官方 SHA-256，再展开；不匹配会立刻停止。

### 3. 安装隔离环境

```bash
export REPO_ROOT=/home/featurize/aic_rgbtir_phase1_bundle/repo
bash "$REPO_ROOT/tools/cloud/install_rgbtir_phase1_env.sh"
```

不允许为了安装 Transformers 而覆盖平台 CUDA/PyTorch。若镜像 `torch<2.6`，停止并换官方 PyTorch 2 镜像。

### 4. 启动分门控任务

先不允许全量 warmup：

```bash
export REPO_ROOT=/home/featurize/aic_rgbtir_phase1_bundle/repo
export RUN_FULL=0
nohup bash "$REPO_ROOT/tools/cloud/run_rgbtir_phase1_gated.sh" \
  > /home/featurize/aic_cloud/logs/aic_rgbtir_phase1_master.log 2>&1 &
echo $! > /home/featurize/aic_cloud/logs/aic_rgbtir_phase1_master.pid
```

该脚本依次执行 `preflight → 视觉 shard 下载 → 10 条真实权重 smoke → overfit100 → tracer400`。任何门槛失败都会退出，不会自动进入下一阶段。

## 监控

```bash
tail -f /home/featurize/aic_cloud/logs/aic_rgbtir_phase1_master.log
nvidia-smi
```

关键结果：

```text
/home/featurize/aic_cloud/outputs/aic_rgbtir_phase1_v1/smoke_summary.json
/home/featurize/aic_cloud/outputs/aic_rgbtir_phase1_v1/overfit100/run_summary.json
/home/featurize/aic_cloud/outputs/aic_rgbtir_phase1_v1/tracer400/run_summary.json
```

## 断点恢复

重复运行 gated 脚本即可。`warmup` 使用 `--resume`，会读取 `checkpoint_last.pt`，并校验配置 fingerprint。不得跳过 smoke 文件或手工修改 GO 状态。

## 启动 clean-train 全量 warmup 的条件

必须同时满足：

- smoke 为 `PHASE_1_SMOKE_GO`；
- gate=0 BF16 误差 `≤1e-3`；
- 梯度只存在于 TIR LoRA；
- overfit100 loss 至少改善 10%；
- tracer official-val loss 至少改善 2%；
- base 参数 SHA-256 前后不变；
- 无 NaN/OOM/大规模 ROI 被黑边清空。

通过后另行执行：

```bash
source /home/featurize/work/envs/aic_rgbtir_phase1/bin/activate
cd /home/featurize/aic_rgbtir_phase1_bundle/repo
export PYTHONPATH=$PWD/src
python tools/run_rgbtir_phase1.py warmup \
  --config configs/aic_rgbtir_phase1.cloud.local.yaml \
  --mode full --resume
```

## 结果边界

本阶段产物是 TIR Adapter checkpoint，不是 AIC submission。不得把 RGBT alignment 指标解释为 AIC ACC，也不使用其他模型 fallback。
