# AIC RGB–TIR Phase 1.9-D2 租卡前就绪报告

## 最终状态

`PHASE_19_D2_PRERENTAL_GO`

本地数据、损失函数、双验证集门禁、云端入口、环境检查和归档脚本已准备完成。本状态只表示可以启动 D2 probe，不表示 D2 已训练成功，更不表示 AIC 平台分数会提升。

## 本轮唯一变量

从 Phase 1.8R 返回的 `D1_L050` selected Adapter 开始，保持模型、rank-48 TIR LoRA、RGB 冻结路径、C2/D1 损失、采样和优化器不变，只增加：

```text
L_D2 = L_D1_L050 + lambda_geometry * L_base_tir_geometry
lambda_geometry ∈ {0.10, 0.25, 0.50}
```

Base-TIR 关系向量按样本标准化后再匹配，避免原始相似度集中时“常数向量”成为低秩捷径。D2 不要求第二次 Qwen 前向；Base-TIR anchors 直接从已返回并校验的 Teacher Bank 读取。

## 固定数据

| 数据 | records | pairs | SHA256 |
|---|---:|---:|---|
| probe train | 4096 | 4096 | `B13FF30E0CD22D1B81CF9939B154BE1B38D28A7C3D0BDF8C45464468965C5D3B` |
| semantic dev | 1024 | 1024 | `B9DFB8C4E47CAD542326BD25D5B2054EA7CADD64E6AA12BA72E2E7B84CD5D1D7` |
| multiquery dev | 1221 | 512 | `A54490B10BF9FC1BFB455C8052F883422D65038603E72CF650749E41D4DF363D` |

三个集合图像对交集均为 0。multiquery dev 的所有图像对也已从未来 full-train 池完整排除。official val 保持封存，不参与本次候选选择。

## 资产复用

- Phase 1.8R D1 Adapter：108 个 TIR 张量，SHA256 `03D2C3D595495162DFCF2F0044CF06CE9B05AE2B52C1186B9A1434F64A5CED06`；
- Teacher Bank：27,668 条，SHA256 `B43CA908381DD42C4421FB3BF1A895DDD1C9396F40ED8B5BE2B51F2D2E966E08`；
- Teacher Bank fingerprint：`D8F01B557D9FAE020767A27DE7DB4602391ACFE5B27141013C3D4D22A2D677E1`；
- 需要的 6,341 条 probe/dev record 在 Teacher Bank 中缺失数为 0。

因此云端不需要重算 Teacher Bank，也不会从随机 Adapter 重新开始。

## 离线敏感性验证

使用 Phase 1.8R official-val 已返回 embedding cache，只验证 D2 损失是否能发现坍缩：

| 输入 | 平均 geometry loss |
|---|---:|
| Base-TIR 恒等关系 | 0.0000 |
| 当前 D1 Adapted TIR | 0.4234 |
| 合成坍缩表示 | 1.4778 |

Layer 8/16/24 均满足“坍缩损失大于当前 D1 损失”。这只是损失敏感性证据，不是训练效果。

## Probe 门禁

每个候选都在同一运行中重新测量 D1 baseline，并同时满足：

1. semantic 与 multiquery 的最低 effective-rank/Base 均不得下降超过 0.01；
2. 至少一个 dev 的最低 effective-rank/Base 提升不低于 0.02；
3. semantic R@5 下降不超过 0.005；
4. multiquery R@5 下降不超过 0.010；
5. nonpaired cosine P95 相对 D1 增量不超过 0.02；
6. 两个 dev 的 paired-shuffled margin 均为正；
7. RGB base 哈希不变、不运行 official val、不存在第二模型 fallback。

候选全部失败会正常输出 `PHASE_19_D2_PROBE_NO_GO` 并归档；不会继续到 full train。

## 验证结果

- 全量 RGB–TIR 回归：`80 passed`；
- D2/Phase16/Phase18 关键回归：`22 passed`；
- Ruff：通过；
- Python compile：通过；
- 两个 Bash 脚本 `bash -n`：通过；
- 本地 preflight：`PHASE_19_D2_PREFLIGHT_GO`；
- ZIP `testzip`：通过；
- ZIP 内 31 个受清单约束文件：缺失 0、哈希不匹配 0。

## 云端包

- 路径：`D:\12525\Documents\pytorch\aic_rgbtir_phase19_d2_cloud_bundle_20260816.zip`
- 字节：`761277588`
- SHA256：`22D9F3CFF9B38E2C548D07D82410B8B46BACC35F5C75A8225452DD268CA23BC9`

4090 可直接使用系统 CUDA PyTorch；若使用 5090，安装脚本会检测 `sm_120`，仅在当前 PyTorch 不支持时安装已验证的 PyTorch `2.7.1+cu128`，并在任何模型下载前执行 CUDA smoke。

## 后续决策

- `D2_PROBE_GO`：只允许进入 D2 full train；
- `D2_PROBE_NO_GO`：保留全部候选，停止扩大训练，回到 shared/complementary 双分支设计；
- Query–TIR、Layer-16 zero-init fusion、Depth 和 AIC 平台测试继续后置，不能用本轮表征 probe 代替最终 bbox/platform 证据。
