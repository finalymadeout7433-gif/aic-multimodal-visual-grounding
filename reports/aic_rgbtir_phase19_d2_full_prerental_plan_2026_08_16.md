# AIC RGB–TIR Phase 1.9-D2 Full：租卡前执行计划与验收边界

日期：2026-08-16
状态：`LOCAL_PRERENTAL_READY`（待云端 RTX 4090 执行）

## 1. 本轮为什么要做

Phase 1.9-D2 probe 已在两个相互隔离的开发集上比较 `D2_G010 / D2_G025 / D2_G050`。其中 `D2_G025` 在检索保持、有效秩恢复和非配对相似度控制之间取得最好的门禁内折中；`D2_G050` 虽然秩恢复更强，但非配对 P95 漂移超过预设上限。因此本轮只验证一个问题：

> `D2_G025` 的局部 probe 收益，在完整 clean train 上从相同 D1 起点重新训练后，能否在双开发集和一次性 official val 上稳定成立？

这不是 Query、融合或比赛平台测试。本轮没有 AIC 测试集、Depth、第二模型 fallback，也不生成提交文件。

## 2. 唯一训练变量

- 基座：`Qwen/Qwen3-VL-8B-Instruct`，固定 revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；
- 初始化：Phase 1.8 的 `D1_L050` selected TIR Adapter；
- 不从 Phase 1.9 probe 的 G025 Adapter 续训，避免把候选选择信息带入 full train；
- 训练对象：仅 TIR rank-48 Adapter；RGB 主路径继续冻结；
- D1 损失、负样本、优化器、分辨率和数据口径保持不变；
- 唯一新增项：Base-TIR 邻域几何保持损失，`geometry_weight = 0.25`；
- 训练量：`23,391` 条、`13,310` 个图像对，一次完整遍历；
- 持久 checkpoint：25% / 50% / 75% / 100%，全部保留在云盘真实目录。

## 3. 数据切分与禁止泄漏

| 集合 | records | unique pairs | 用途 |
|---|---:|---:|---|
| full train | 23,391 | 13,310 | 唯一训练集 |
| semantic dev | 1,024 | 1,024 | 单 Query 语义保持与秩恢复选优 |
| multi-query dev | 1,221 | 512 | 同图多 Query 区分能力选优 |
| official val | 2,032 | 1,115 | 双 dev 通过后仅打开一次 |

租卡前 preflight 已验证四者图像对隔离、manifest 数量与 SHA-256、Teacher Bank 覆盖和 D1 Adapter 指纹。official val 不参与 checkpoint 选择。

## 4. 执行顺序

1. 云端只做轻量环境安装和版本检查；
2. 在下载/加载模型前运行 tests 与资产 preflight；
3. 加载固定模型 revision、D1 Adapter 和 Teacher Bank；
4. 从 D1 同一起点训练唯一候选 `D2_G025`；
5. 生成并持久保存四个 checkpoint；
6. 每个 checkpoint 同时跑 semantic dev 与 multi-query dev；
7. 只有满足双 dev 硬门禁的 checkpoint 才能被选中；
8. 立即生成 Stage A 归档，先把四个 checkpoint、selected Adapter、Teacher Bank 指纹、双 dev 指标和日志传回本地；
9. 双 dev 为 GO 时，才对 selected Adapter 运行一次 official val；
10. 生成最终归档并传回本地，完成 SHA-256、ZIP 解压和清单校验后才允许退还实例。

## 5. 双开发集硬门禁

一个 checkpoint 必须同时满足：

- semantic 与 multi-query 的最低有效秩/Base 均 `>= 0.85`；
- 两个 dev 相对 D1 baseline 的最低有效秩均至少提升 `0.02`；
- semantic 平均 R@5 相对 D1 降幅不超过 `0.005`；
- multi-query 平均 R@5 相对 D1 降幅不超过 `0.010`；
- 两个 dev 的 non-paired cosine P95 增量均不超过 `0.02`；
- paired-shuffled margin 均为正。

若四个 checkpoint 均不通过，结果为 `PHASE_19_D2_TRAIN_SELECT_NO_GO`，不打开 official val。NO-GO 是有效实验结论，不等于任务程序失败。

## 6. official val 硬门禁

只有 dev-selected checkpoint 接受一次 `2,032/2,032`、`1,115/1,115` official val。要求：

- skipped = 0；
- Layer 8/16/24 最大绝对表征漂移 `<= 0.02`；
- Layer 8/16/24 最低有效秩/Base `>= 0.85`，最低有效秩/RGB Teacher `>= 0.75`；
- 严格 record-level R@5 保持：Layer 8 相对 `0.9134` 降幅 `<=0.005`，Layer 16 相对 `0.9070` 降幅 `<=0.005`，Layer 24 相对 `0.7539` 降幅 `<=0.010`；
- non-paired cosine P95 增量 `<=0.10`；
- paired-shuffled margin 及 bootstrap 95% 下界均大于 0；
- 主要分组退化不超过 5%；
- RGB base 哈希、gate=0、无 IR/无效 IR、全零 mask 和无第二模型 fallback 等安全检查全部通过。

全部通过才输出 `PHASE_19_D2_FULL_GO`。该 GO 只说明 TIR 表征修复可进入下一阶段，仍不代表 AIC 平台分数已经提升。

## 7. 云端文件保全要求

- checkpoint 目录必须是云盘真实目录，禁止 `/dev/shm` 和符号链接；
- 四个 checkpoint、Teacher Bank、selected Adapter、双 dev 摘要不得在 official val 前删除；
- official-val 使用分片缓存和最终 `embedding_cache.pt`，允许断点恢复；
- trap 必须写 `runtime_failure_state.json`；
- Stage A 归档优先于 official val；即使 official 阶段中断，训练产物也已返回本地；
- 只有本地 receipt SHA-256、ZIP 完整性和文件清单全部通过，才建议退还实例。

## 8. 时间与算力预算

RTX 4090 可以执行本轮。准确耗时必须以云端最前面的 200/1,000 step 实测速率重算；租卡前不承诺一个虚假的固定 ETA。建议按 **8–12 小时余额**准备，并在 Stage A 完成后重新评估 official-val 剩余时间。若早期吞吐明显偏离预期，先诊断 I/O、CPU preprocessing 和 GPU 利用率，不改变模型与验证口径来追求表面加速。

## 9. 本地租卡前验收

- 相关测试：`33 passed`；
- Python 编译：通过；
- 两个 cloud shell：`bash -n` 通过；
- `git diff --check`：通过；
- 本地真实资产 preflight：`PHASE_19_D2_FULL_PREFLIGHT_GO`；
- Ruff：本地环境未安装，因此未作为阻塞项；由编译、测试、运行时 preflight 和哈希验收覆盖当前发布检查。

最终云端运行包及其 SHA-256 以同目录 bundle receipt 为准。
