# AIC RGB–TIR 下一轮修订计划：Phase 1.8A-Q0 + Phase 1.8A-Full

> 日期：2026-08-14（Asia/Shanghai）
> 当前任务：赛题一，`RGB + TIR + Depth + English Query -> RGB 坐标系单目标 bbox`
> 当前基座：`Qwen/Qwen3-VL-8B-Instruct`
> 当前已选红外候选：`D1_L050`
> 修订依据：`AIC_RGB_TIR_Query验证与AIC真实场景下一阶段指导报告_2026_08_14.md`
> 本文目的：冻结下一轮的低成本 Query 接口预检、D1 全量训练目标和 sealed official-val 验收边界。

---

## 1. 对 Query 验证指导报告的审阅结论

这份修正后的报告与当前赛题一和 D1_L050 状态高度一致，主体路线可以采用，但必须保持严格的实验边界。

### 1.1 本计划直接采纳的内容

1. **全量训练前增加 Q0 接口 smoke。** 这是低成本工程保险，可以避免在云端完整训练后才发现 Query token、ROI、候选或角色接口错误。
2. **Query 必须成为进入融合前的独立门禁。** ROI 配对检索不能替代自然语言消歧和 bbox grounding。
3. **Query 候选必须包含困难负例。** 除跨图负例外，还要包含同图同类实例、reference、part 和几何 near-miss。
4. **主 Query probe 使用 RGB-Teacher-Anchored Shared Head。** 同一冻结匹配头评估 Base TIR、C2 和 D1，避免每个表征单独训练强头掩盖差异。
5. **静态融合、Query gate、质量 gate、配准 gate 必须逐项增加。** 每一步均保留 RGB-only 控制和反事实安全测试。
6. **AIC 无标签数据只做安全 dry-run。** 不训练、不选 checkpoint、不拟合阈值、不根据预测推断 GT。

### 1.2 本计划修正或推迟的内容

1. 报告前文称 external official val 可用于 checkpoint 和 gate 校准，但后文又要求 sealed official val。为避免验证集泄漏，本计划固定为：repair-dev 选择和冻结，official val 只做一次最终验收。
2. 报告列出的 fusion、gating、AIC dry-run 全套代码不在下一轮一次实现；下一轮只实现 Q0 和 Full 所需接口，其他模块按阶段增量开发。
3. 第一份平台实验不能一次混入 Query gate、IR quality gate 和 alignment gate。它们必须先在外部有标签数据上逐变量验证，再决定唯一提交变量。
4. AIC dry-run 不能反向缩小启用簇或修改阈值；若发现严重安全问题，只能停止提交并回到外部 train/dev 重新提出预注册方案。

---

## 2. 当前证据与下一轮唯一问题

Phase 1.8A D1 retention probe 已从三条候选中选出 `D1_L050`：

| 指标 | D1_L050 |
|---|---:|
| 平均 R@1 | 61.52% |
| 平均 R@5 | 84.38% |
| Layer 8 drift | 0.08302 |
| Layer 16 drift | 0.10177 |
| Layer 24 drift | 0.21814 |
| Layer 24 相对 C2 漂移下降 | 43.16% |
| 最低有效秩 / Base TIR | 88.24% |
| 最低有效秩 / RGB Teacher | 80.50% |
| 最低 paired-shuffled margin | 0.1516 |

这些结果只来自固定 probe train 和 repair-dev。它们证明 `D1_L050` 值得扩大训练，但没有证明它在完整训练后仍能保留上述平衡。

因此下一轮唯一要回答的问题是：

> `D1_L050` 从全新 rank-48 Adapter 初始化，在完整 clean-train 上训练后，能否在全部 2,032 条 official val 上继续同时保持跨模态检索、低漂移、有效秩和 RGB 安全等价性？

在回答这个模型问题前，增加一个不改变实验变量的工程预检：确认现有 D1 特征可以无歧义地接入 Query 编码、ROI 候选和角色负例链路。该预检不评价模型优劣，也不取代后续 Query grounding。

---

## 3. 下一轮名称、目标和边界

### 3.1 阶段名称

```text
Phase 1.8A-Q0：Query 接口工程 Smoke
        ↓
Phase 1.8A-Full：D1_L050 全量训练与 official-val 表征验收
```

### 3.2 唯一目标

Q0 只排除 Query 接口工程风险；Full 阶段产出一个经过完整训练和 official-val 硬门禁的 TIR rank-48 Adapter，作为 Phase 1.8B Query–TIR grounding probe 的唯一冻结红外资产。

### 3.3 本轮不做

- 不使用 Query 训练或选择 D1 Adapter；
- 不生成 bbox；
- 不做 RGB–TIR 融合；
- 不训练融合门；
- 不使用 AIC 测试集训练、选 checkpoint 或拟合阈值；
- 不加入 Depth；
- 不迁移到 30B；
- 不使用第二模型 fallback；
- 不生成 AIC 提交包。

Query 非常重要，但本轮只允许做 Q0 接口 smoke，不允许做 Query 语义训练或把 Query 指标纳入 D1 checkpoint 选择。完整 Query grounding 仍放在 Phase 1.8B。若全量训练时同时加入 Query loss，就无法区分失败来自 Adapter 全量泛化、Query 接口还是融合结构。

### 3.4 Phase 1.8A-Q0 的精确边界

固定从 repair-dev 抽取 64～128 条样本，使用当前 probe `D1_L050`，只验证：

- Query tokenizer、prompt 和 pooled hidden state 可重复；
- Query token 与 Layer 8/16/24 TIR 特征维度匹配；
- ROI pooling、`grid_thw`、valid-FOV mask 和 bbox 几何一致；
- 同图候选、同类实例、target/reference/part 和几何 near-miss 候选可构造；
- 同一图像对不会被错误用作跨图负例；
- 候选顺序变化不改变指标；
- 全部输出有限，不修改 D1 Adapter，不产生梯度。

Q0 不训练任何参数、不输出模型优劣结论、不参与 GO/NO-GO。Q0 失败只表示接口需要修复；通过后才允许租用云卡执行 Full。

---

## 4. 固定资产与数据边界

| 项目 | 固定值 |
|---|---|
| 模型 | `Qwen/Qwen3-VL-8B-Instruct` |
| revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| TIR Adapter | rank 48，视觉注意力 `qkv` 与 `proj` |
| 初始化 | 全新初始化，不从 probe Adapter 继续训练 |
| 目标函数 | C2 + D1 retention |
| retention lambda | `0.50` |
| seed | `20260812` |
| RGB、LLM、基础视觉塔 | 全部冻结 |
| full train | 24,612 条、13,822 个唯一图像对 |
| repair dev | 1,024 条、1,024 个唯一图像对 |
| official val | 2,032 条、1,115 个唯一图像对 |
| Teacher Bank | 必须匹配已冻结 fingerprint 和 SHA-256 |

训练、dev 和 official val 必须按图像对完全隔离。official val 不能参与超参数、checkpoint 或阈值选择。

---

## 5. 训练和 checkpoint 选择

### 5.1 全量训练

保持 probe 中除训练规模外的全部设置：

- BF16；
- AdamW；
- 固定学习率、warmup、cosine decay 和 gradient accumulation；
- 固定 256 个跨图负例采样协议；
- 固定 Layer 8/16/24/final 损失权重；
- 固定 retention 容差；
- 只更新 TIR Adapter。

保存：

```text
25%
50%
75%
100%
```

四个 checkpoint。

### 5.2 repair-dev 选择

checkpoint 只根据 repair-dev 选择，采用“硬门禁 + 词典序”，不得使用未经解释的加权总分。

第一层：候选 checkpoint 必须全部满足：

1. 无 skipped、NaN、Inf 或空 ROI；
2. RGB/base 参数哈希不变；
3. Layer 8/16/24 漂移均低于 C2；
4. Layer 24 漂移相对 C2 至少下降 25%；
5. R@1/R@5 达到预注册保留阈值；
6. effective-rank/Base TIR 不低于 85%；
7. effective-rank/RGB Teacher 不低于 75%；
8. paired-shuffled margin 为正；
9. nonpaired cosine P95 增量不超过 0.10；
10. 主要分组不存在系统性恶化；
11. 安全等价性通过且不存在第二模型 fallback。

第二层：在全部通过的 checkpoint 中按以下词典序选择：

1. 最大化 Layer 8/16/24 中最差的 R@5；
2. 若差异小于 0.5 个百分点，选择 Layer 24 drift 更低者；
3. 再选择最低 effective-rank ratio 更高者；
4. 再选择 paired-shuffled margin 更高者；
5. 最后选择更早 checkpoint，降低过拟合风险。

不得默认选择最后一步，也不得查看 official val 后反向更换 checkpoint。

### 5.3 official-val 一次性验收

方案和 checkpoint 冻结后，official val 只执行一次。其作用是判断是否允许进入 Query probe，不再承担调参功能。

指导报告中“official val 可用于选择 checkpoint 或校准 gate”的表述与其后文的 sealed-val 规则冲突。本计划采用更严格边界：repair-dev 负责选择和阈值冻结，official val 只做一次最终验收。

---

## 6. 硬门禁

只有全部满足才输出 `PHASE_18_FULL_GO`：

1. full train 完成 `24,612/24,612`，无跳样本、NaN、Inf 或空 ROI；
2. official val 完成 `2,032/2,032`、`1,115/1,115` 图像对，skipped 为 0；
3. RGB base、LLM 和冻结视觉参数训练前后 SHA-256 不变；
4. Layer 8/16/24 drift 全部低于 C2 参考值；
5. Layer 24 drift 相对 C2 至少下降 25%；
6. mean R@1 相对 D1 probe 下降不超过 1.0 个百分点；
7. mean R@5 相对 D1 probe 下降不超过 0.5 个百分点；
8. Layer 8/16/24 的 effective-rank/Base TIR 最低值不少于 85%；
9. Layer 8/16/24 的 effective-rank/RGB Teacher 最低值不少于 75%；
10. paired-shuffled margin 为正，2,000 次固定种子 bootstrap 的 95% 下界大于 0；
11. nonpaired cosine P95 相对 Base TIR 增量不超过 0.10；
12. 样本数不少于 100 的来源、光照、天气、尺寸和遮挡分组不得系统性恶化超过 5%；
13. `gate=0`、`tir=None`、无效 IR、全零 mask 均与同模型 RGB-only 等价；
14. 不存在第二模型 fallback。

若任一核心门禁失败，输出 `PHASE_18_FULL_NO_GO`，停止 Query 和融合实验，优先分析 full-train 规模导致的过拟合、checkpoint 时机或 retention 强度，不通过增加更多模块掩盖问题。

---

## 7. 工程实现要求

现有 `run_rgbtir_phase18.py` 被明确保护为 probe-only：

```text
probe_only = true
forbid_full_training = true
forbid_official_val = true
full_steps = 0
```

不能修改或解除这些保护来运行全量。下一轮应新增独立实现：

```text
src/aic_rgbtir/phase18_full.py
src/aic_rgbtir/query_candidates.py
src/aic_rgbtir/query_roles.py
src/aic_rgbtir/query_probe.py
tools/run_rgbtir_phase18_full.py
tools/run_rgbtir_query_interface_smoke.py
configs/aic_rgbtir_phase18_full.example.yaml
configs/aic_rgbtir_phase18_full.cloud.example.yaml
tests/test_rgbtir_phase18_full.py
tests/test_rgbtir_query_interface_smoke.py
tools/prepare_rgbtir_phase18_full_cloud_bundle.ps1
tools/cloud/run_rgbtir_phase18_full_gated.sh
```

核心工程保护：

- fresh-init 必须有单元测试，拒绝把 `D1_L050` probe Adapter 当作续训起点；
- checkpoint selector 只能读取 repair-dev；
- official-val 文件在 checkpoint 冻结前不可访问；
- fingerprint 不一致拒绝 resume；
- 中断后可从最近 checkpoint 恢复；
- 云端产物必须在退还 GPU 前打包并拉回本地；
- probe 配置、代码和产物不得被覆盖。

Q0 的单元测试必须覆盖：

- target/reference/part 角色解析；
- 同图候选与跨图负例隔离；
- hard-negative IoU 分桶；
- 候选顺序无关性；
- Query tokenizer 和 hidden shape 固定；
- smoke 过程中 D1/RGB/LLM 参数均无梯度且哈希不变。

---

## 8. 本轮产物

```text
outputs/aic_rgbtir_phase18_full_v1/
  preflight.json
  run_fingerprint.json
  train_events.jsonl
  checkpoints/
    step_25/
    step_50/
    step_75/
    step_100/
  repair_dev_metrics/
  selected_checkpoint.json
  selected_adapter.pt
  official_val/
    per_record_metrics.jsonl
    alignment_summary.json
    retrieval_metrics.json
    collapse_metrics.json
    subgroup_metrics.csv
    safety_equivalence.json
  run_summary.json
  sha256_manifest.json

outputs/aic_rgbtir_phase18_q0_v1/
  smoke_manifest.jsonl
  candidate_audit.jsonl
  interface_checks.json
  run_summary.json
  sha256_manifest.json

reports/
  aic_rgbtir_phase18_full_validation_2026_08_XX.md
```

云端完整目录必须额外生成压缩归档和 SHA-256，拉回本地并完成解压审计后才能退还实例。

---

## 9. 下一阶段预注册

若 Q0 工程 smoke 通过且 Full 为 `PHASE_18_FULL_GO`，下一阶段固定为：

```text
Phase 1.8B：Frozen Query–TIR Grounding Probe
```

它将使用冻结的完整 D1 Adapter，在 RGBT-GroundBench Query+bbox 上比较：

```text
RGB-only + Query
Base TIR + Query
C2 TIR + Query
D1 Full TIR + Query
```

并报告候选级 Query–ROI 排序、ACC@0.5、mIoU、主体/参照物错误、小目标、低光和遮挡分组。只有 Query probe 证明 D1 特征可被语言定位任务利用，才进入最小静态 RGB–TIR 融合；Query-conditioned gate 再作为单变量增量加入。

Phase 1.8B 的主 Query probe 应采用 RGB-Teacher-Anchored Shared Head：只在 RGB Teacher ROI + Query 上训练一个低容量共享匹配头，冻结后使用同一参数评估 Base TIR、C2 和 D1 Full，避免为每种表征单独训练强头掩盖差异。候选必须包含同图其他目标、同类实例、reference、part 和 IoU 分桶几何 near-miss，不能只使用容易的跨图随机负例。

若本轮为 `NO_GO`，不得进入 Query、融合或 Depth。

---

## 10. 最终决策

下一轮不测试新模型，也不直接实现复杂融合。唯一正确动作是：

> 用固定的 `D1_L050` 配方，从全新 rank-48 Adapter 初始化，在完整 clean-train 上训练，经 repair-dev 选择 checkpoint，再在 2,032 条 official val 上做一次性表征验收。

在租用云卡前，先完成不训练的 Q0 Query 接口 smoke。它只降低工程风险，不改变 Full 阶段的模型目标和证据边界。

这一轮的成功标准不是平台分数，而是得到一份可被 Query 模块安全消费、不会破坏 RGB 主路径、不会再次低秩坍缩的完整红外 Adapter。平台实质增益必须经过 Query probe、最小融合和一次受控 AIC 提交后才能确认。
