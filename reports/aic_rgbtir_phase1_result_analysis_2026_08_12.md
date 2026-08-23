# AIC RGB–TIR Phase 1 结果分析与后续验证决策

日期：2026-08-12
结论：Phase 1 训练成功，但应在进入 query-aware fusion 前补做 Phase 1.5 全量独立验证。

## 1. 当前进度

| 阶段 | 状态 | 已证明 | 尚未证明 |
|---|---|---|---|
| Phase 0 数据/几何/接口 | 完成 | manifest、黑边 mask、grid、坐标和 gate=0 等价链路可信 | 实际融合增益 |
| Phase 1 TIR Adapter warmup | 完成 | TIR rank-48 Adapter 能稳定学习冻结 RGB teacher 表征 | Query 控制、bbox 改善和 AIC 增益 |
| Phase 1.5 完整验证 | 待执行 | — | 2,032 条 full-val 泛化、检索区分度和表征坍缩 |
| Phase 2 query-aware fusion | 未开始 | — | RGB+TIR+Query 是否优于 RGB-only |
| AIC 受控提交 | 未开始 | — | 官方 ACC@0.5 |

## 2. Phase 1 核心结果

- clean train：`26,477 / 26,477`
- skipped / fallback：`0`
- 可训练参数：`8,957,952`，全部为 TIR `qkv/proj` rank-48 Adapter
- 冻结 Qwen RGB 基座哈希前后一致
- official-val-100 loss：`0.152710 → 0.073343`
- 绝对减少：`0.079367`
- 相对减少：`51.97%`
- 设定 GO 门槛：`2%`

分层表征余弦相似度：

| hook 层 | 训练前 | 训练后 | 变化 |
|---|---:|---:|---:|
| layer 8 | 0.71715 | 0.87767 | +0.16051 |
| layer 16 | 0.78759 | 0.90987 | +0.12228 |
| layer 24 | 0.95479 | 0.97669 | +0.02190 |
| final | 0.99539 | 0.99694 | +0.00155 |

改善主要发生在前中层，符合“红外与 RGB 的域差异首先存在于底层/中层视觉表示”的预期。final 层本来已经接近饱和，因此不应只用 final cosine 评估 Adapter。

## 3. 收敛质量

基于 2,648 条定期训练日志：

| 区间 | loss 均值 | 中位数 | 标准差 |
|---|---:|---:|---:|
| step 1–1,000 | 0.10068 | 0.08741 | 0.04951 |
| step 12,000–14,000 | 0.07973 | 0.06556 | 0.05436 |
| step 25,000–26,477 | 0.07377 | 0.05992 | 0.04206 |
| 最后 500 step | 0.07187 | 0.05825 | 0.03768 |

- 全局日志 loss 斜率为负，末期没有系统性反弹。
- loss 和 layer cosine 中均无 NaN/Inf。
- 个别高 loss 点仍存在，更像样本难度差异，而不是优化发散。
- overfit100、tracer400 与 full 三级门槛分别得到 `34.52% / 38.88% / 51.97%` 改善，证据方向一致。

## 4. 可以得出的结论

1. Qwen3-VL-8B 视觉塔可以用非对称 TIR rank-48 Adapter 稳定适配红外输入。
2. 一轮 clean-train 比 400 条 tracer 进一步改善独立 val-100，说明增加数据量在当前指标上有效。
3. RGB 主路没有被更改，后续可用零初始化融合门保留历史 RGB-only 能力。
4. 黑边无效区没有造成训练样本跳过，Phase 0 mask 链路在当前数据上可用。

## 5. 当前不能得出的结论

1. `51.97%` 是表征对齐 loss 改善，不是 ACC@0.5。
2. 当前 loss 使用真实 bbox 构造 ROI mask，未检验模型能否从 Query 自主找到该区域。
3. 高配对 cosine 不必然表示特征具有足够的跨样本区分度；它仍可能受通用场景特征或表征集中影响。
4. 目前没有证明 RGB+TIR 会在正常光样本上不产生负迁移。
5. 当前没有证明 AIC 上的弱配准、倾斜黑边与视差问题已被解决。

## 6. 为什么需要 Phase 1.5

当前验证集是固定分层的 100 条，覆盖 34/33/33 个 FLIR/M3FD/MFAD 样本，并包含 60 个小目标、42 个低光、31 个恶劣天气和 41 个高遮挡样本。它适合作为训练门槛，但不足以描述全部 2,032 条 official val 的尾部分布。

进入 Phase 2 前建议完成：

### A. 全量 official val-2032

报告 mean/median/P90 loss、四层 cosine、ROI 清空数量和失败样本。按 FLIR/M3FD/MFAD、白天/低光、天气、目标尺度、遮挡、黑边比例与配准风险分组。

### B. 跨样本检索验证

对每个 TIR ROI，比较配对 RGB ROI 与其他样本 RGB ROI，计算 R@1/R@5、paired-vs-shuffled margin。如果只有配对 cosine 高，但检索率很低，则表示特征还不具备足够的目标区分度。

### C. 表征坍缩检查

统计各 hook 层的跨样本方差、effective rank 和非配对样本平均 cosine。不能只依赖 final 层近乎饱和的 cosine。

### D. 安全退化验证

在加载最终 Adapter 后重新验证：

- `gate=0` 与原生 RGB-only 结果误差仍在 BF16 `1e-3` 以内；
- `tir=None` 与 IR 无效 mask 都严格退化为同一 RGB 路径；
- Adapter-only release 与原 full checkpoint 的 TIR 特征逐层一致。

## 7. 决策门槛

建议 Phase 1.5 GO 同时满足：

- full-val-2032 无 NaN/Inf，ROI 无效率极低且有明确清单；
- 总体与主要来源子集均优于未训 TIR Adapter，不只是 val-100 提升；
- paired-vs-shuffled 有稳定正 margin，且检索率明显优于随机；
- 无明显表征坍缩；
- 训练后 gate=0 / `tir=None` 等价性仍通过。

如果 full val 通过而某些弱配准子集明显落后，不应否定整个 Adapter；应在 Phase 2 中加入 alignment confidence / IR quality gate，或后续再引入 CoDAF 式弱配准模块。

## 8. 下一步

当前不建议立即训练完整融合模型。先补做 Phase 1.5，再用固定的三组对照进入 Phase 2：

1. RGB-only；
2. RGB+TIR，无 Query gate；
3. RGB+TIR，Query + IR quality + alignment confidence gate。

Phase 2 首轮只训 fusion/gate，继续冻结 RGB 主路和当前 TIR Adapter。这样一旦融合无效，可以明确将失败归因到 fusion/gate，而不是同时改变多个模块。
