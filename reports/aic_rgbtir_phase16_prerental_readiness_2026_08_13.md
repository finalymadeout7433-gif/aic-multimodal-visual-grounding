# AIC RGB–TIR Phase 1.6 租卡前就绪报告

## 结论

本地工程已达到 **PHASE_16_LOCAL_READY**。它只表示代码、数据边界、测试和云端入口已准备好，不代表去坍缩训练已经完成，更不代表已经获得 AIC 平台提升。

下一步可以租用 RTX 4090 24GB，并在云端执行固定流程。只有 official val 的 2,032 条记录全部通过硬门禁后，才允许进入 Phase 2 融合。

## 本轮修复对象

Phase 1 Adapter 显著改善 RGB–TIR 配对对齐和检索，但 Layer 8/16/24 的 effective rank 明显下降，说明特征更像 RGB 的同时，不同目标之间的区分结构被压缩。Phase 1.6 不改 LoRA rank，也不训练 RGB 主路径，而是比较：

- C0：原始配对 alignment + background margin；
- C1：C0 + 256 个跨图 RGB 负例的 InfoNCE；
- C2：C1 + RGB teacher relational distillation。

Layer 8/16 是主要判别层，权重为 1.0；Layer 24 为 0.5；Final 只保留 0.05 的配对约束，不再让已经趋于饱和的 Final 层主导训练。

## 已确认的数据边界

- 原始 clean train：26,477 条记录；
- repair probe：4,096 条、4,096 个图像对；
- repair dev：1,024 条、1,024 个图像对；
- repair full train：24,612 条、13,822 个图像对；
- official val：2,032 条、1,115 个图像对；
- probe/dev、full/dev、train/official-val 图像对交集均为 0。

repair full train 包含 probe 图像对，但不包含 repair-dev 图像对；这是为了让候选选择完成后，胜者从全新相同初始化在全部非 dev clean train 上重训。

## 工程保障

- C0/C1/C2 从逐张量一致的全新 rank-48 TIR Adapter 开始；
- RGB/Qwen 视觉主路径完全冻结并进行参数哈希校验；
- Teacher Bank 按模型、processor、manifest、代码和权重哈希绑定 fingerprint；
- 负样本不来自同一图像对，并固定为 128 个同来源、64 个同条件、64 个全局随机；
- full 训练保存 25%/50%/75%/100% checkpoint，由 repair-dev 多指标选择；
- checkpoint 位于梯度累积边界，支持 `--resume`；
- 旧 fingerprint 缓存会被明确拒绝；
- 无第二模型 fallback；
- Phase 2 前仍不接入 Query、bbox、融合门、Depth 或 AIC 测试集。

## 本地验收结果

- `tests/test_rgbtir_*.py`：32 项全部通过；
- Phase 1.6 专项：9 项全部通过；
- Python 编译、两份 YAML 配置解析、PowerShell AST 和两份 Bash 脚本语法均通过；
- 云端 ZIP 解压后再次执行 32 项测试，全部通过；
- ZIP 内部 SHA-256 manifest 不匹配数为 0；
- 启动包 SHA-256 以同目录 `.zip.sha256` 旁车文件为准。

## 硬门禁

最终状态只有同时满足下列条件才是 `PHASE_16_GO`：

1. official val 完成 2,032/2,032，skipped=0；
2. alignment loss 相对 Base TIR 至少改善 2%；
3. 主要分组不得恶化超过 5%；
4. Layer 8/16/24 的 paired-shuffled margin 置信区间下界大于 0；
5. Layer 8/16 的 R@5 不比 Base TIR 低超过 0.5 pp，且 R@1/R@5 至少一项提升 1 pp；
6. Layer 8/16/24 effective rank 至少达到 Base TIR 的 70% 和 RGB 的 50%；
7. nonpaired cosine P95 增量不超过 0.10；
8. 至少保留 Phase 1 Adapter 70% 的 R@5 增益；
9. Adapter/checkpoint、gate=0、`tir=None`、无效 IR、全零 mask、RGB 参数哈希和有限值检查全部通过。

## 云端固定入口

```bash
cd /home/featurize/aic_rgbtir_phase16_bundle/repo
bash tools/cloud/install_rgbtir_phase16_env.sh
bash tools/cloud/run_rgbtir_phase16_gated.sh
```

运行过程中可通过以下日志观察：

```text
/home/featurize/aic_cloud/logs/aic_rgbtir_phase16_v1/
```

所有训练、checkpoint、Teacher Bank、official val 指标和最终报告写入持久盘：

```text
/home/featurize/aic_cloud/outputs/aic_rgbtir_phase16_v1/
/home/featurize/aic_cloud/reports/aic_rgbtir_phase16_repair_validation_2026_08_13.md
```

## 结论边界

Phase 1.6 即使通过，也只证明红外 Adapter 的表征适配兼顾了对齐与区分能力。它仍不能证明 AIC Query grounding、bbox ACC 或平台分数提升。真正的平台收益需要 Phase 2 将修复后的 TIR 特征以零初始化、可退化的融合模块接回 RGB 主模型，并以 RGB-only 0.7582 为控制基线；首个实质提升门槛为 0.7632。
