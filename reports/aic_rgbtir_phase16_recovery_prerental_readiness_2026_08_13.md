# Phase 1.6-R 最小恢复重训：租卡前就绪说明

> AS_OF: 2026-08-13
> 状态：本轮只恢复 C1/C2、精简 Teacher Bank 和 Phase 1.7A+ 诊断证据。

## 固定边界

- 使用 `Qwen/Qwen3-VL-8B-Instruct` 固定 revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；
- C1/C2 从相同 fresh rank-48 Adapter 初始化；
- Teacher Bank 仅覆盖 4,096 probe + 1,024 dev 图像对；
- 不运行 C0；
- 不进行 full train；
- 不打开 2,032 条 official val；
- 不使用 AIC 测试集；
- 不生成比赛提交；
- 不使用第二模型 fallback。

## 云端路径

上传并解压到：

```text
/home/featurize/aic_rgbtir_phase16_recovery_bundle
```

RGBT-GroundBench 必须可见于：

```text
/home/featurize/data/RGBT_GroundBench/extracted/image_data
```

## 执行命令

```bash
cd /home/featurize/aic_rgbtir_phase16_recovery_bundle/repo
bash tools/cloud/install_rgbtir_phase16_env.sh
tmux new -s phase16r
bash tools/cloud/run_rgbtir_phase16_recovery.sh
```

退出 tmux：`Ctrl+B`，然后按 `D`。

## 最终必须回传

```text
/home/featurize/aic_cloud/platform_upload_ready/AIC_RGBT_Phase16R_C1_C2_Phase17APlus_20260813.tar.gz
/home/featurize/aic_cloud/platform_upload_ready/AIC_RGBT_Phase16R_C1_C2_Phase17APlus_20260813.tar.gz.sha256
```

在本地完成 SHA256、C1/C2、Teacher Bank、C2 diagnostics 和 `decision.json` 审计前，不得释放实例。

## 中断恢复

重复运行同一脚本即可：

- Teacher Bank 每 25 对保存 partial；
- C1 完成后立即生成独立 ZIP 与 SHA256；
- C2 完成后同样独立持久化；
- fingerprint 不一致时拒绝复用旧缓存。
