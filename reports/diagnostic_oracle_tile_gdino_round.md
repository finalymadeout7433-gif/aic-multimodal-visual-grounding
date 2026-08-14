# Florence 候选上限、Tile 与 GroundingDINO 零样本诊断报告

## 1. 目的与边界

本轮只做外部 validation 诊断与无标签 AIC smoke；没有训练模型、没有生成正式提交、没有使用 AIC 测试标签，也没有触碰 holdout。

## 2. 固定数据清单

- `core_eval_1500`：1500 条，1500 张互不重复图像；选择哈希 `02B2A9AEB79BEB4788DBD01013A348AC5E6830FA37846386BDA8ACC232BF3464`。
- `tile_probe_600`：600 条；选择哈希 `D14D796D09E55985B930FE2E594DD0A5883F79EADB16647269640D3B6AE037EF`。
- 源 validation 清单哈希：`C141D13839C057AED276BAE9A927C5B77567EDA3362D0D806154808BE394B388`。

### 环境与模型身份

- Python `3.11.15`，PyTorch `2.5.1+cu121`，Transformers `4.57.6`；GPU `NVIDIA GeForce RTX 4060 Laptop GPU`。
- Florence 权重 SHA-256：`8B4E610C952EEF90A836C56CDA0F398A672A3A6CA7B4D96B0E09A86DEE42E2C3`；本地快照没有暴露源 commit，因此哈希是本轮不可变身份。
- GroundingDINO revision：`a2bb814dd30d776dcf7e30523b00659f4f141c71`；权重 SHA-256：`1A2412EF99BD74BCD3C2A246FA1E48581F8889A1300C9051974741314FC042F3`。

## 3. Florence full-image first 与 oracle

- first ACC@0.5：0.6893
- candidate oracle ACC@0.5：0.7533
- oracle 差值：0.0640
- first mean IoU：0.6716
- oracle mean IoU：0.7329
- 无候选率：0.0013
- 平均候选数：1.692
- 平均延迟：309.7 ms
- GPU 峰值分配显存：1.75 GiB
- 完整 1,500 条墙钟时间：7.84 分钟

候选结果计数：

- `first_correct`：1034
- `oracle_rescue`：96
- `all_candidates_fail`：370

## 4. 候选短语角色

- `whole_query`：635
- `query_phrase`：1899
- `partial_query_phrase`：3
- `divergent_or_reference`：1

详细案例见 `reports/florence_candidate_analysis.md` 和本地 `outputs/diagnostics/analysis/visualizations/`。

## 5. Tile pilot

| 组别 | full first | full oracle | full+tile oracle | oracle 增益 | 候选数增长 | 延迟倍率 |
|---|---:|---:|---:|---:|---:|---:|
| 小目标 | 0.4967 | 0.6200 | 0.7933 | +0.1733 | 1.77→7.20 | 4.90× |
| 中/大目标对照 | 0.7700 | 0.8233 | 0.8900 | +0.0667 | 1.70→7.33 | 4.90× |

Tile 600 条墙钟时间：15.23 分钟；无候选 1 条。

官方 `000108_001` 极小灯泡仅作 sanity：full+tile 共得到 3 个候选，oracle IoU 0.0000，仍未定位成功。

## 6. GroundingDINO-Tiny

- fp16 smoke 在首条前向即因文本增强层 Float/Half dtype 不一致而中止；没有写入候选或 fallback。
- 后续结果使用独立的 fp32 配置，不与失败的 fp16 运行混写。

- 样本数：1500（完整 1,500 条，可与 Florence 受控比较）
- top-1 ACC@0.5：0.5627
- top-5 oracle ACC@0.5：0.8820
- top-10 oracle ACC@0.5：0.9073
- 无候选率：0.0000
- 平均候选数：4.671
- 平均延迟：366.4 ms
- 完整评测墙钟时间：9.25 分钟

AIC 100 条无标签 smoke：

- 合法框率：1.0000
- 无候选率：0.0000
- PNG/JPG 数量：{'png': 50, 'jpg': 50}
- 平均延迟：399.6 ms

## 7. 固定阈值决策

- Florence：`target-role-aware reranker`；oracle 差值 +0.0640。
- Tile：`selective tile`；小目标增益 +0.1733，对照组下降 -0.0667。
- GroundingDINO top-1：`not a platform candidate`。
- GroundingDINO 候选：`candidate generator`。

## 8. 下一步训练与提交建议

1. 下一轮优先训练/开发候选重排器，而不是直接微调 GroundingDINO。GroundingDINO top-10 的召回上限很高，但 top-1 明显较弱，说明主要缺口是排序。
2. 第一阶段用外部 train/validation 生成 top-10 候选，以 GT IoU 构造候选级监督；输入至少包含模型 score、bbox 几何、Query 类型、候选 label 与 Query 的角色关系。先做轻量 ranking/MLP，再决定是否引入视觉 crop 编码器。
3. Florence 的同 label 多实例最值得先做保守选择：在明确包含 left/right/top/bottom/largest/smallest/ordinal 的 Query 上按几何关系排序。不同 label 候选不能直接当成等价目标做无约束相似度排序。
4. Tile 只进入选择性策略：优先用于小目标、原模型无候选或低召回类别；不建议对 9,555 条全部运行，官方极小灯泡样例也证明 Tile 不能解决所有极小目标。
5. 推荐的下一次单变量平台提交，是经过外部验证的 Florence 同-label/显式空间词保守重排；不要直接提交纯 GroundingDINO top-1。随后再单独测试 GroundingDINO top-k + 学习式 reranker。

本地外部数据可能与模型预训练集重叠，因此绝对分数只用于回归；AIC 平台分数仍是目标域证据。每次只提交一种变化，并保留 Florence RGB-only 0.4980 作为平台基线。

## 9. 可复现性与产物

- 机器可读汇总：`outputs/diagnostics/round_summary.json`
- 完整预测：本地 `outputs/diagnostics/`（已被 Git 忽略）
- 候选分析：`reports/florence_candidate_analysis.md`
- 模型 revision 与哈希：见机器可读汇总中的 `models`。
