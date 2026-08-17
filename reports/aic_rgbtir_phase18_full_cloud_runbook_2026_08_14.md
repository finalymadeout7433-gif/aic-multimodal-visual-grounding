# AIC RGB–TIR Phase 1.8A-Q0 + Full 云端运行手册

## 1. 本轮唯一目标

本轮使用 `Qwen/Qwen3-VL-8B-Instruct` 的冻结 RGB/LLM 主路径，从**全新初始化**的 rank-48 TIR Adapter 开始，仅训练 `D1_L050`：

```text
C2 对齐/InfoNCE/关系蒸馏/背景约束
+ Base-TIR retention，lambda=0.50
```

训练集为 24,612 条 `repair_full_train`，只用 1,024 条 `repair_dev` 选择 25%/50%/75%/100% 四个 checkpoint；方案冻结后，2,032 条 official val 只打开一次。

本轮不训练 Query、不做 RGB–TIR 融合、不加 Depth、不使用 AIC 测试集、不生成比赛提交、不使用第二模型 fallback。

## 2. 本地已准备资产

云端包：

```text
D:\12525\Documents\pytorch\aic_rgbtir_phase18_full_cloud_bundle_20260814.zip
SHA256: 106FFE33B094DD6ED768BB75C7AACD725027331E3DBA173FFA4AB59AFCD5047A
```

包内包含：

- 完整源码、测试、配置和启动脚本；
- 固定的 full/dev/official manifests；
- Q0 候选池、Tokenizer 文件和 D1 probe Adapter 只读参考；
- 产物自动归档脚本。

不包含 8B 完整权重。云端必须把固定 revision 的完整四个权重分片放在实例本地盘
`/home/featurize/full_model_cache`；不要把补下载写入容量较小的持久 NFS 云盘。

## 3. 租卡后执行

推荐 RTX 4090 24GB，系统使用已有 PyTorch/CUDA 镜像。将 ZIP 解压为：

```text
/home/featurize/aic_rgbtir_phase18_full_bundle
```

数据集必须可见：

```text
/home/featurize/data/RGBT_GroundBench/extracted/image_data
```

Featurize 的 `dataset extract` 对本轮 28.1GB ZIP64 包可能触发
`UnexpectedHeaderError`。已验证的恢复路径是：

1. 用 `featurize dataset download` 将平台缓存包下载到持久盘；
2. 用系统 `7z t` 验证 ZIP64 完整性；
3. 将外层 ZIP 解压到实例本地盘；
4. 校验六个 TAR 的固定 SHA-256 后再解包到上述 `extracted` 目录。

本轮实测持久 NFS 云盘在保存环境、旧模型缓存和 28GB ZIP 后会达到配额上限；
数据解压、完整模型缓存和训练输出均使用实例本地盘，结束前必须回传归档。

安装环境：

```bash
export BUNDLE_ROOT=/home/featurize/aic_rgbtir_phase18_full_bundle
export ENV_ROOT=/home/featurize/work/envs/aic-rgbtir-phase18
bash "$BUNDLE_ROOT/tools/cloud/install_rgbtir_phase18_env.sh"
```

后台执行：

```bash
nohup bash "$BUNDLE_ROOT/tools/cloud/run_rgbtir_phase18_full_gated.sh" \
  > /home/featurize/aic_cloud/logs/rgbtir_phase18_full_bootstrap.log 2>&1 &
echo $!
```

执行顺序固定为：

1. 全部相关测试；
2. 本地等价的 Q0 候选/接口 smoke；
3. 真实 8B 权重 Query hidden-state smoke（10 条，无训练）；
4. Full preflight；
5. 构建 full/dev/official Teacher Bank，可断点恢复；
6. D1_L050 全量训练并保存四个 checkpoint；
7. repair-dev 密封选点；
8. official val 一次性验收；
9. 自动生成云端归档和 SHA-256 receipt。

## 4. 监控与恢复

日志：

```text
/home/featurize/aic_cloud/logs/rgbtir_phase18_full_bootstrap.log
/home/featurize/aic_cloud/logs/rgbtir_phase18_full.log
```

中断后重复运行同一个启动命令即可；`--resume` 会校验 fingerprint，并只接受相同模型 revision、代码、配置、manifest 和 Teacher Bank。

若 fingerprint 不一致，必须停止，不能静默复用旧 checkpoint。

## 5. 完成产物与退卡边界

主目录：

```text
/home/featurize/aic_cloud/outputs/aic_rgbtir_phase18_full_v1
```

归档：

```text
/home/featurize/aic_cloud/platform_upload_ready/aic_rgbtir_phase18_full_artifacts_20260814.zip
/home/featurize/aic_cloud/platform_upload_ready/aic_rgbtir_phase18_full_artifacts_20260814.receipt.json
```

必须先把归档拉回本地、核验 ZIP SHA-256、解压并检查 `run_summary.json`、选中 Adapter、四个 checkpoint、official-val 指标和 `sha256_manifest.json`，之后才能退还 4090。

该归档是研究训练产物，不是 AIC 平台提交包。

## 6. GO / NO-GO

只有输出 `PHASE_18_FULL_GO` 才允许进入 Phase 1.8B Frozen Query–TIR Grounding Probe。任何门禁失败均停止，不通过增加融合、Depth 或第二模型 fallback 掩盖问题。
