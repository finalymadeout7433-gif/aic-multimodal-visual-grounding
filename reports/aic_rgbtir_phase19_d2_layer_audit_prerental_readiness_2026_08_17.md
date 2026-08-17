# AIC RGB–TIR Phase 1.9-D2 Layer Audit 本地准备与执行边界

> 日期：2026-08-17（Asia/Shanghai）
> 状态：`PHASE_19_D2_LAYER_AUDIT_PREFLIGHT_GO`
> 性质：无训练、双 dev、四 checkpoint 的逐层表征审计

## 1. 本轮唯一问题

Phase 1.9-D2 Full 的四个 checkpoint 都没有通过 `both_dev_absolute_rank`。现有 Stage A 摘要只保存 Layer 8/16/24 中的最小值，因此当前不能诚实断言到底是哪一层拖后腿。

本轮只回答：

1. 哪一层在 multi-query dev 上形成最低 effective-rank/Base；
2. 该瓶颈是否贯穿 25%/50%/75%/100% checkpoint；
3. semantic 与 multi-query 的差距来自单层还是多层；
4. 50% checkpoint 是否只适合作为诊断参考，而不能作为正式 selected Adapter；
5. 下一轮 D3 是否有证据只修改一个层和一个损失权重。

## 2. 明确禁止

- 不训练或反向传播；
- 不改变四个 checkpoint；
- 不生成 selected Adapter；
- 不打开 official val；
- 不加入 Query、fusion、Depth、AIC test 或 fallback；
- 不把阈值从 `0.85` 事后降低到 `0.84`；
- 不把 feature audit 写成 bbox 或平台收益。

## 3. 固定资产核验

本地 preflight 已逐项通过：

| 资产 | 记录/图像对或 SHA-256 |
|---|---|
| semantic dev | `1,024 / 1,024`；`B9DFB8C4...D5D1D7` |
| multi-query dev | `1,221 / 512`；`A54490B1...F363D` |
| Teacher Bank | `B43CA908...66E08` |
| Teacher fingerprint | `D8F01B55...677E1` |
| checkpoint 5,848 | `82DDB36A...A4BA` |
| checkpoint 11,696 | `6D30D87D...BFC6` |
| checkpoint 17,544 | `D878E212...4555` |
| checkpoint 23,391 | `07090D4B...610` |

两个 dev 与训练集、official val 的原始 image-pair 隔离合同不变。

## 4. 每个 checkpoint、每个 dev、每一层输出

Layer 8/16/24 分别保存：

- absolute effective rank；
- Base/Teacher effective-rank ratio；
- participation ratio；
- 完整奇异值、能量概率、累计能量；
- 90%/95%/99% 能量所需分量数；
- adapted/Base R@1、R@5；
- paired cosine 与绝对 alignment loss；
- nonpaired cosine mean/P95；
- 严格跨 image-pair cosine mean/P95；
- 历史 `torch.roll` margin；
- 严格跨 image-pair 的 pair-safe margin；
- 记录级与 image-pair 聚合后的 effective rank。

旧评估器用 `torch.roll` 构造 shuffled 对象。在 multi-query 记录相邻且同图多 Query 时，它可能选中同一 image pair。因此本审计不会删除历史指标，而是并列新增 pair-safe margin；这只修正诊断口径，不影响 D2 Full 的 effective-rank NO-GO 事实。

## 5. 执行效率与恢复

同一 checkpoint 内按 image pair 编码一次，再对该图对应的不同 bbox 分别 ROI pooling：

```text
semantic:   1,024 unique pairs
multiquery:   512 unique pairs
合计:       1,536 pairs / checkpoint
四 checkpoint: 6,144 visual forwards
```

每 100 个 image pair 原子写入 partial cache；缓存绑定：

- 模型 revision 与视觉资产哈希；
- checkpoint SHA-256；
- Teacher fingerprint；
- 两个 manifest 的 SHA-256；
- processor 与 ROI expansion；
- 审计代码和配置。

fingerprint 不同拒绝恢复。审计完成后保留 feature cache、逐层 JSON、summary、sha256 manifest 和日志。

## 6. 本机为什么不直接跑

本机 RTX 4060 Laptop 只有 8GB VRAM，审计按原配置要求至少 23GB。执行入口已实测安全停止：

```text
PHASE_16_HARDWARE_NO_GO: GPU has 8.00 GiB, requires 23.00 GiB
```

此外本机 Hugging Face snapshot 只有 config/tokenizer，没有四个模型权重 shard。改变分辨率、量化或降低模型口径都不允许，因此真实逐层结果必须在 4090/5090 上生成。

## 7. 云端包

- 路径：`D:\12525\Documents\pytorch\aic_rgbtir_phase19_d2_layer_audit_cloud_bundle_20260817.zip`
- 字节：`910,643,044`
- SHA-256：`90F1E673CDBB90C9FFFD5A067499E73D244A91782F79615B4B0AE6C5288B3748`
- ZIP 条目：35；`testzip() = None`
- manifest 文件：34；全部字节数与 SHA-256 通过

云端启动前还必须验证固定 revision 的完整 Qwen 模型缓存和 RGBT 数据根目录存在。未验证前不得开始计费较长的 6,144 次视觉前向。

## 8. 审计后的唯一决策规则

- 若同一层在四 checkpoint 的 multi-query 中持续最低：D3 只提高该层 geometry/retention，其他层保持 G025；
- 若单层的 R@5/秩存在冲突：D3 只降低该层 contrastive 权重；
- 若 record-level 失败但 pair-level 全部健康且差异集中在同图多目标：增加同图关系保持，禁止把同 pair 当负例；
- 若三个层都轻度不足：才考虑跨层谱保持；不能一次叠加多项改动；
- 只有新候选在 semantic 与 multi-query 都稳定 `>=0.85`，才允许重新讨论 official val；
- 在 representation GO 前不进行 Query 训练。
