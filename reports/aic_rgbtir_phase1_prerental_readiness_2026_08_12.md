# AIC RGB–TIR Phase 1 租卡前方案审核

日期：2026-08-12
状态：本地准备中，真实权重 smoke 需 RTX 4090 后执行

## 1. Phase 1 的准确目标

Phase 1 不是训练完整的 `RGB + TIR + Query → bbox` 模型，也不训练融合门。它只做一件事：

> 在冻结的 Qwen3-VL-8B RGB 视觉塔监督下，让 TIR rank-48 LoRA 学会将红外目标区域映射到 Qwen 已熟悉的 RGB 视觉表征空间。

训练监督来自 RGBT-GroundBench 的配对 RGB/TIR、目标 bbox。Query 在本阶段保留于 manifest，但不进入损失，因此不得把 Phase 1 指标称为语言定位准确率。

## 2. 为什么先做真实权重 smoke

Phase 0 用结构等价的轻量测试确认了接口，但只有真实权重才能验证：

- 官方 BF16 权重能否只加载视觉塔；
- 第 8、16、24、final 视觉层 hook 能否捕获真实 pre-merger 特征；
- Qwen 官方图像归一化和真实 token grid 是否一致；
- 4090 的峰值显存；
- gate=0 时包装器与 RGB-only 是否等价；
- 反向传播是否只进入 TIR LoRA。

真实 smoke 固定 10 条、不更新权重。任一条件失败即停止，不进入 warmup。

## 3. 权重与环境方案

- 模型：`Qwen/Qwen3-VL-8B-Instruct`
- revision：`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- Transformers：`4.57.6`
- 精度：BF16
- GPU：RTX 4090 24GB
- 不安装 FlashAttention，不从零重装 CUDA/PyTorch；首轮使用 PyTorch SDPA，减少配置时间和编译风险。
- 不加载 8B 语言模型。根据官方 checkpoint index，仅下载包含全部 `model.visual.*` 张量的一个 shard，约 2.72 GB；完整模型约 17.53 GB。

## 4. 训练边界

可训练：

- 视觉注意力 `qkv` 和 `proj` 的 TIR LoRA；
- rank = 48，约 900 万参数。

冻结：

- Qwen 视觉主干；
- RGB LoRA（rank=0，实际不存在）；
- 语言模型；
- residual fusion、`fusion_scale`；
- bbox head。

因此本阶段不会破坏当前 RGB 主路径；它也不会直接产生 AIC 提交包。

## 5. 损失与黑边处理

每条训练记录使用真实 bbox 生成目标区域 patch mask，并与 Phase 0 的 `ir_valid_mask` 相乘，排除 TIR 外边界黑框。损失包含：

1. 目标区域 RGB teacher 与 TIR student 的多层余弦对齐；
2. 轻量前景/背景 hard-negative margin，降低 TIR 把背景学成目标的风险。

不做逐图 min-max，不自动配准，不用 AIC 测试集训练。

## 6. 云端执行门槛

执行顺序固定：

1. `preflight`：GPU、磁盘、环境、manifest、Phase 0 GO；
2. `download-assets`：只下载视觉 shard 并记录 SHA-256；
3. `smoke`：10 条真实 BF16、无训练；
4. `overfit100`：100 条、200 step，验证 loss 和梯度链路；
5. `tracer400`：400 train / 100 official val；
6. 只有 tracer 达标后，才由人工确认是否启动 26,477 条 clean train 的完整 warmup。

硬停止条件：

- smoke OOM、NaN、hook 层缺失；
- gate=0 最大误差大于 `1e-3`；
- 任一非 TIR-LoRA 参数获得梯度；
- Qwen base hash 改变；
- 100 条过拟合诊断 loss 相对改善不足 10%，或 tracer official-val loss 相对改善不足 2%；
- 红外表征塌缩或大量 ROI 被黑边 mask 清空。

## 7. 数据与成本准备

Phase 1 不需要 AIC 正式测试集，只需要：

- RGBT-GroundBench 官方图像与标注，约 10 GB；
- Phase 0 manifests 和摘要，约 36 MB；
- Qwen 视觉 shard，约 2.72 GB；
- 工程包，不足 1 MB（不含 manifests）。

为避免 GPU 计费浪费，RGBT 数据应先放到平台持久化“我的数据集”，云端只挂载或解压，不应每次从本机临时上传。

本地已生成：

- 云端工程包：`D:\12525\Documents\pytorch\aic_rgbtir_phase1_prerental_bundle_20260812.zip`
- RGBT 官方归档上传包：`D:\AIC赛题一数据集\03_RGBT_GroundBench\RGBT_GroundBench_official_archives_20260812.zip`
  - SHA-256：`16DFDEF217B526AE37187EE1898E7D97833FD587FEF2611CCC5C77C1FD1A2447`

浏览器只读检查时 Featurize 会话已经退出登录，因此当前尚未确认第二个大包是否已存在于“我的数据集”。租卡前必须先登录并确认/上传该数据包；这属于当前唯一的外部状态检查项。

## 8. 本阶段能与不能回答的问题

能回答：

- Qwen 视觉塔是否能稳定接入 TIR LoRA；
- TIR 表征是否向冻结 RGB teacher 收敛；
- 黑边 mask、真实 hook、显存和断点恢复是否可靠。

不能回答：

- AIC 平台分数是否提高；
- Query 是否正确控制 IR 参与；
- 融合后 bbox 是否更准；
- 红外是否对所有 Query 都有益。

这些属于 Phase 2 的 query-aware fusion 与 grounding 验证。

## 9. 参考实现

- Qwen3-VL 官方仓库：https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL 官方微调环境：https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune
- 官方模型配置：https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json
