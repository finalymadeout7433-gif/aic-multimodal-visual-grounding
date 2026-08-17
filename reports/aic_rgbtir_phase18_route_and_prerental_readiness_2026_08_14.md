# Phase 1.8 路线与租卡前准备报告

## 1. 路线结论

Phase 1.8 拆为两个严格串行阶段：

1. **Phase 1.8A — D1 Retention Probe（本轮）**：修复 C2 检索增强但向 RGB Teacher 绝对漂移的问题；只比较 retention λ。
2. **Phase 1.8B — Query–TIR Grounding Probe（后续）**：仅在 D1 通过 full train 与 official val 后，验证红外表征是否真正服务自然语言定位。

当前不得直接实现 shared/complementary 双分支，也不得进入融合。原因是现有证据只能确认 C2 的结构冲突，尚未证明 D1 能在不牺牲检索与有效秩的前提下减少漂移。

## 2. D1 的可归因定义

基线保持 C2：

```text
alignment + cross-image InfoNCE + relational distillation + background margin
```

只增加：

```text
L_retention = mean_l ReLU(cos(BaseTIR_l, RGBTeacher_l) - epsilon_l
                          - cos(AdaptedTIR_l, RGBTeacher_l))
```

固定：

- ε8=0.02、ε16=0.02、ε24=0.01；Final 不进入 retention；
- λ∈{0.10, 0.25, 0.50}；
- 三条候选从完全相同的新 rank-48 Adapter 初始化；
- 4,096 probe train、1,024 repair dev；
- Teacher Bank、256 个负例、seed=20260812、AdamW、学习率、steps 完全一致。

因此任何差异只能归因于 retention 强度。

## 3. 预注册硬门禁

候选必须同时满足：

1. Layer 8/16/24 drift 均低于恢复后的 C2；
2. Layer 24 drift 至少降低 25%；
3. R@5 相对 C2 下降不超过 0.5 个百分点；
4. R@1 相对 C2 下降不超过 1.0 个百分点；
5. 最低 effective rank/Base ≥85%；
6. 最低 effective rank/RGB Teacher ≥75%；
7. paired-shuffled margin >0；
8. nonpaired cosine P95 增量 ≤0.10；
9. RGB/base hash 不变；
10. 无 NaN/Inf、无空 ROI、无第二模型 fallback。

恢复后的 C2 对照固定为：R@1=0.59798177、R@5=0.8359375；L8/L16/L24 drift 分别为 0.12893361、0.15261441、0.38376218。

## 4. 本地已准备资产

- 深模块：`src/aic_rgbtir/phase18.py`；
- 入口：`tools/run_rgbtir_phase18.py`；
- 单元与合同测试：`tests/test_rgbtir_phase18.py`、`tests/test_rgbtir_phase18_bundle_contract.py`；
- 固定云端配置：`configs/aic_rgbtir_phase18.cloud.example.yaml`；
- 三份候选配置：`configs/phase18_candidates/`；
- 云端安装与执行脚本：`tools/cloud/install_rgbtir_phase18_env.sh`、`run_rgbtir_phase18_probe.sh`；
- 打包器：`tools/prepare_rgbtir_phase18_cloud_bundle.ps1`；
- 固定 Teacher Bank 与 manifests 来自已回传并校验的 Phase 1.6 recovery 资产。

## 5. 本地事实核验

- probe：4,096 条 / 4,096 pairs；
- dev：1,024 条 / 1,024 pairs；
- full：24,612 条 / 13,822 pairs（仅打包，禁止本轮执行）；
- official val：2,032 条 / 1,115 pairs（仅固定边界，禁止本轮执行）；
- split overlap：0；
- Teacher Bank 覆盖 probe+dev 全部 5,120 record IDs；
- Teacher Bank SHA-256：`4A429B24353171CA1A887059A55D5597EF73A6E94E87B1C00D8582515E1567DD`；
- fingerprint：`79CF814B64EC640AD3D924342FEEED235764F139445F834B2CDB2DD0B580C05C`。

## 6. 当前边界

本地准备完成后可以租 4090，但本轮只允许运行 retention probe。即使候选 GO，也必须先回传完整产物并复核，再单独授权 full train。不得在同一次租卡中自动继续训练。

## 7. 云端运行包

- ZIP：`D:\12525\Documents\pytorch\aic_rgbtir_phase18_d1_cloud_bundle_20260814.zip`；
- 大小：141,131,935 bytes；
- SHA-256：`816D2018D6355F75057FCA4F1594EFCD21E0ADA478709A515DF2CB3ECC19F2CA`；
- ZIP entry：34；
- 路径分隔符：全部为 `/`，可在 Linux 正常解压；
- 非法 `__pycache__` / `.pyc`：0；
- 本地 RGB–TIR 回归测试：50 passed；
- Phase 1.8 专项与合同测试：19 passed；
- 本地真实资产 preflight：`PHASE_18_LOCAL_OR_CLOUD_PREFLIGHT_GO`。

结论：`PHASE_18_PRERENTAL_GO`。可以租用 RTX 4090 24GB，执行一次 D1 retention probe。
