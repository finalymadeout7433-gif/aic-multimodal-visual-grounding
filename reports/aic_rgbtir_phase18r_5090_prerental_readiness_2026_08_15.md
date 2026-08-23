# AIC RGB–TIR Phase 1.8R：5090 租卡前就绪报告

## 1. 本轮目标

本轮不是改变算法、不是加入 Query/Fusion/Depth，也不是重新选择候选路线。唯一目标是安全重跑已经预注册的 `D1_L050` 全量训练与 sealed official-val，并避免再次因验证中断而丢失已经完成的训练成果。

固定边界：

- 模型：`Qwen/Qwen3-VL-8B-Instruct`，固定 revision；
- 只训练 rank-48 TIR Adapter，RGB/LLM 主路径冻结；
- 训练集 `24,612` 条，repair-dev `1,024` 条；
- 保存 25%/50%/75%/100% 四个 checkpoint；
- official val 为 `2,032` 条、`1,115` 个唯一图像对；
- 不使用 AIC 测试集，不生成比赛提交，不使用第二模型 fallback。

## 2. 上轮为何停在 official-val 569/1115

已确认事实：Teacher Bank、24,612 个训练 step、四个 checkpoint 和 dev 选点均已完成；official-val 在 `569/1115` 后进程被系统以 `Killed` 终止。日志没有 Python Traceback，也没有代码主动抛出的 OOM。

最合理解释是外部 `SIGKILL`：当时云平台余额耗尽/实例释放，同时旧实现把训练和 official-val 放在同一个长 Python 进程，训练对象、Teacher Bank、优化器和验证缓存共同提高了主机内存压力。不能仅凭 `Killed` 精确区分平台计费终止与 Linux OOM killer，因此本轮同时修复两类风险。

## 3. Phase 1.8R 的工程修复

执行链改为：

```text
preflight + Q0 smoke
→ 独立进程 A：Teacher Bank + 全量训练 + dev 选点
→ 原子归档 Stage A（四个 checkpoint、selected Adapter、summary、SHA256 receipt）
→ 释放进程 A 的训练资源
→ 独立进程 B：只加载选中的 Adapter，断点续跑 official-val
→ GO 或 NO-GO 均生成最终归档
```

关键保障：

1. Stage A 完成后立即产生可下载归档，official-val 失败不会抹掉训练成果；
   如果四个 checkpoint 都未通过 repair-dev 门禁，也会先归档四个 checkpoint 和 NO-GO 决策，再停止而不打开 official-val；
2. official-val 不再构建 Teacher Bank，不保留训练器和优化器；
3. 每个图像对的缓存只保存 compact FP16 ROI pooled features，不保存完整 token 序列；
4. 验证提取阶段定期记录 RSS、CUDA allocated/reserved/peak 和进度；
5. 缓存绑定 fingerprint，可从已完成 pair 继续；
6. 云端启动前强制检查主机内存不少于 48 GiB；
7. official-val 的 `NO-GO` 是有效科研结论，仍然必须归档；只有异常退出才阻止最终归档。

## 4. 5090 实例要求

- GPU：RTX 5090 32GB；
- 主机内存：最低 48 GiB，推荐 64 GiB 或以上；
- 实例本地盘：建议至少 200GB 可用；
- 镜像：现成 PyTorch 2/CUDA 镜像；
- 数据：`/home/featurize/data/RGBT_GroundBench/extracted/image_data` 必须存在；
- 运行期间余额必须覆盖训练、official-val、归档和下载缓冲时间。

如果平台页面显示的 5090 实例主机内存不足 48 GiB，不要启动任务，也不要通过降低图像分辨率或验证样本数绕过门禁。

## 5. 云端运行与产物

云端包解压后的固定目录：

```text
/home/featurize/aic_rgbtir_phase18_full_bundle
```

启动：

```bash
export BUNDLE_ROOT=/home/featurize/aic_rgbtir_phase18_full_bundle
export ENV_ROOT=/home/featurize/work/envs/aic-rgbtir-phase18
bash "$BUNDLE_ROOT/tools/cloud/install_rgbtir_phase18_env.sh"
nohup bash "$BUNDLE_ROOT/tools/cloud/run_rgbtir_phase18_full_gated.sh" \
  > /home/featurize/aic_cloud/logs/rgbtir_phase18_full_bootstrap.log 2>&1 &
```

Stage A 训练产物归档：

```text
/home/featurize/aic_cloud/platform_upload_ready/aic_rgbtir_phase18_train_select_artifacts_20260815.zip
/home/featurize/aic_cloud/platform_upload_ready/aic_rgbtir_phase18_train_select_artifacts_20260815.receipt.json
```

最终归档：

```text
/home/featurize/aic_cloud/platform_upload_ready/aic_rgbtir_phase18_full_artifacts_20260815.zip
/home/featurize/aic_cloud/platform_upload_ready/aic_rgbtir_phase18_full_artifacts_20260815.receipt.json
```

监控程序在 Stage A 归档生成后应立即先下载一份到本地；最终归档生成并通过本地 SHA256/解压/manifest 校验后，才可退还实例。

## 6. 租卡前验收门禁

必须同时满足：

- 新增分阶段、恢复、流式缓存和包契约测试全部通过；
- 全部 `test_rgbtir_*.py` 回归通过；
- `py_compile` 与 `git diff --check` 通过；
- 云端 ZIP 能完整解压；
- 解压包内测试再次通过；
- ZIP SHA256 与文件清单已经记录。

只有完成上述检查，状态才是 `PHASE_18R_PRERENTAL_GO`。

## 7. 本地最终验收结果

- Python 语法编译：通过；
- 本地全部 RGB–TIR 回归：`65/65` 通过；
- 云端包解压后相关测试：`41/41` 通过；
- 包内 `BUNDLE_MANIFEST.json` 文件哈希：全部匹配；
- ZIP 完整解压：通过；
- shell 启动脚本语法：通过；
- 最终状态：`PHASE_18R_PRERENTAL_GO`。

最终云端包的绝对路径和 SHA256 记录在包外的租卡前交付回执中，避免把 ZIP 自身哈希写入 ZIP 导致自引用变化。
