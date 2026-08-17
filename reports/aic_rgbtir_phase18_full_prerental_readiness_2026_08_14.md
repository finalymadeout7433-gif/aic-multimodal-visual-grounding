# Phase 1.8A-Q0 + Full 租卡前准备验收

## 结论

状态：`LOCAL_PRERENTAL_GO`

本地能够完成的代码、测试、固定数据、候选配置、云端启动包和归档链路已准备完毕。现在租用 RTX 4090 24GB 是合理的；租卡后不需要临时设计训练逻辑，只需要核验数据挂载、解压包并执行固定命令。

## 已验证事实

### 数据与隔离

| Split | 记录数 | 唯一图像对 |
|---|---:|---:|
| repair probe | 4,096 | 4,096 |
| repair dev | 1,024 | 1,024 |
| repair full train | 24,612 | 13,822 |
| official val | 2,032 | 1,115 |

Full/dev/official 按图像对交集为 0。固定 manifest SHA-256：

- full：`8E2998C9FC3D20BFB06085F9A0439FB9D353E7D19C5AAFB5902127706FB19874`
- dev：`B9DFB8C4E47CAD542326BD25D5B2054EA7CADD64E6AA12BA72E2E7B84CD5D1D7`
- official：`002FFB665CA8D0CBA545F2A7C4C342ED3B5708274922163B82E66955AF8E5E30`

### Q0

本地无训练 Q0 完成 128 条 Query、752 个候选，状态 `PHASE_18_Q0_READY`。重复执行后 `sha256_manifest.json` 逐字节一致：

```text
60B891C8438EED8A1A5521355DDB029D9D6E9ECBBF8FA17EAF94AFF4FE0FD637
```

边界：本地 Q0 使用真实 tokenizer，但用确定性 sketch 验证 Query/候选/维度接口，不是 learned Query 语义结果。云端启动器因此额外安排 10 条真实 8B hidden-state smoke；该 smoke 仍不训练，也不声称 grounding 效果。

### 全量编排

已实现并测试：

- 只允许 `D1_L050 / lambda=0.50`；
- Adapter 必须 fresh-init；
- 四个固定 checkpoint；
- checkpoint 只读 repair-dev；
- official val 仅在 dev 冻结选择后打开一次；
- R@5 差异小于等于 0.5 个百分点时，按 Layer 24 drift、有效秩、margin、较早 checkpoint 依次选择；
- fingerprint 漂移拒绝 resume；
- 无合格 dev checkpoint 时绝不打开 official val；
- 训练完成后自动归档，且不存在比赛 submission 或第二模型 fallback。

### 测试

- RGB–TIR 全量回归：`59 passed`；
- 新增 Phase 1.8 Full/Bundle 专项：`9 passed`；
- Python 入口编译检查：通过；
- 本地 Full preflight：`PHASE_18_FULL_PREFLIGHT_GO`；
- `git diff --check`：通过。

### 云端包

```text
路径：D:\12525\Documents\pytorch\aic_rgbtir_phase18_full_cloud_bundle_20260814.zip
大小：21,758,876 bytes
SHA256：106FFE33B094DD6ED768BB75C7AACD725027331E3DBA173FFA4AB59AFCD5047A
条目：51
结构审计：PASS
```

## 云端仍需验证的事项

以下无法在本机 4060 和缺失完整权重的条件下伪装为已完成：

1. 真实 8B Query hidden-state smoke；
2. 4090 BF16 视觉 shard smoke 和显存峰值；
3. full Teacher Bank 构建；
4. 24,612 步 D1_L050 训练；
5. dev 四 checkpoint 选择；
6. 2,032 条 official val 一次性验收。

这些均已写成自动化固定流程，租卡后只执行，不再现场改变方案。

## 下一动作

可以租用 RTX 4090 24GB。租用后先核验 RGBT 数据集挂载路径，再上传/解压固定云端包并执行运行手册中的命令。
