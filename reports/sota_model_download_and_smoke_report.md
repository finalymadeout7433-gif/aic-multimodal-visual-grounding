# AIC 候选基座模型下载与首轮测试报告

日期：2026-08-01

## 结论

三个模型的官方代码和权重均已下载并完成 SHA-256 校验。

- **MM-Grounding-DINO-T 已完整跑通**，是本轮三个候选中最适合继续扩展验证的可训练基座。
- **LLMDet-Swin-T 已跑通**，但固定 60 条小型探针上没有超过 MM-Grounding-DINO-T，也没有提供候选并集增益，因此暂不作为第一主线。
- **APE-Ti 权重中存在规模可观的语言模型前缀**，但是否与当前配置严格完整匹配，仍需在官方运行栈中进行 strict load 验证。官方实现依赖 Detectron2、Detrex 和 `ape._C` 原生扩展；本机没有已配置的 WSL2/Linux、MSVC 与 CUDA Toolkit 编译链，因此只完成了权重结构测试，不能伪称完成了模型推理。

当前建议不是直接训练或生成平台提交，而是先把 MM-Grounding-DINO-T 扩大到严格的多域验证，并另建 Linux/WSL2 环境完成 APE-Ti 公平对照。

## 资产与可复现身份

| 模型 | 官方代码版本 | 主要权重 | 大小 | SHA-256 | 状态 |
|---|---|---|---:|---|---|
| APE-Ti | `8c4920e2014d4818ec3ecafb7b526d896a659d1c` | `model_final.pth` | 2,316,022,287 B | `B5D793E960515A6D1AA4B8A55B61DA990B0B4184B510D3D7AD3BB37526FB8007` | 权重通过；运行环境阻塞 |
| MM-Grounding-DINO-T | `cfd5d3a985b0249de009b67d04f37263e11cdf3d` | OpenMMLab 原始 `.pth` | 960,771,250 B | `4EA751CECB6741437290A1352A1561AF365084672FFEA1B2115070B07BCCE134` | 通过 |
| MM-Grounding-DINO-T | HF revision `80b3081e...` | Transformers `model.safetensors` | 692,015,412 B | `2D9BDFD0247F167A00E00FFFA1DEC1EE5D5861FDFB4E27FD8CC2279AF6820688` | 通过 |
| LLMDet-Swin-T | `53366243fba758ad3d7a042c0c1457dee91a02b2` | 官方原始 `tiny.pth` | 2,262,911,558 B | `9F0EB4DE246C9C8DF7194A921A335F8C3F24C95D3C0DCB1AFD22CF3DB4CAD042` | 权重通过 |
| LLMDet-Swin-T | HF revision `6719e6ec...` | 官方 `pytorch_model.bin` | 692,228,571 B | `3D7157A358780627EB824BACDBE3B8A13671E4D5FADD860841144EAB77420476` | 通过 |
| LLMDet-Swin-T | 本地安全重封装 | 完整 safetensors | 695,202,284 B | `1D9E7CB0C53DB1A8EA6D26D23248247F489FBD6F64302821CCA64FB557653A3B` | 通过 |

三个原始主权重合计约 5.16 GiB；包含 Transformers 转换和安全重封装副本时，本轮模型文件约 7.74 GiB。

官方来源：

- APE：[代码](https://github.com/shenyunhang/APE)、[权重](https://huggingface.co/shenyunhang/APE)
- MM-Grounding-DINO：[官方配置与权重表](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md)、[Transformers 转换集合](https://huggingface.co/collections/openmmlab-community/mm-grounding-dino)
- LLMDet：[代码](https://github.com/iSEE-Laboratory/LLMDet)、[原始权重](https://huggingface.co/fushh7/LLMDet)、[Transformers 权重](https://huggingface.co/fushh7/llmdet_swin_tiny_hf)

以上仓库均采用 Apache-2.0 许可证。

## 测试一：AIC 官方单样例

样例 `000108_001` 的 GT 来自官方公开样例，只用于链路和候选冒烟。

| 模型 | Top-1 IoU | Top-20 最佳 IoU | 最佳候选排名 | 判断 |
|---|---:|---:|---:|---|
| MM-Grounding-DINO-T | 0.4082 | 0.7241 | 13 | Top-1 未过线，但候选集中存在正确框 |
| LLMDet-Swin-T | 0.0000 | 0.6196 | 3 | Top-1 错误，但第三候选过线 |

单样例不能用于决定平台强弱，只能说明两个模型都存在明显的“候选召回强、Top-1 选择弱”现象。

## 测试二：固定 60 条外部有标签探针

探针来自 RefCOCO、RefCOCO+、RefCOCOg，各 20 条、每条对应不同图像。配置统一为 FP32、阈值 0.05、最多 20 个候选。

| 模型 | Top-1 ACC@0.5 | Top-10 Oracle | 全候选 Oracle | 平均推理延迟 |
|---|---:|---:|---:|---:|
| MM-Grounding-DINO-T | **0.5333** | **0.9500** | 0.9500 | 394 ms |
| LLMDet-Swin-T | 0.4500 | 0.9333 | 0.9500 | 396 ms |

两模型 Top-1 框在 83.33% 的样本上达到 IoU≥0.5 的相互一致；候选并集 Oracle 仍为 0.95，没有新增召回。MM-Grounding-DINO-T 有 7 条独有 Top-1 正确，LLMDet 有 2 条。

这些指标是小型、可能与模型训练域重叠的外部数据结果，**不是 AIC 结果，也不能预测平台分数**。它们只支持当前工程决策：LLMDet 暂未显示足够互补性，MM-Grounding-DINO-T 更适合作为下一轮可运行基座。

## LLMDet 格式兼容问题

官方 `model.safetensors` 对重复注册的解码分类标量进行了去重，当前 Transformers 版本加载时会把 5 个中间层标量随机初始化。官方 `pytorch_model.bin` 保留了完整 1,069 个键，但当前 Transformers 由于 PyTorch 2.5 的安全限制拒绝直接加载 pickle 权重。

为避免升级现有训练环境，本轮对已校验的官方 `.bin` 使用 `weights_only=True` 读取，并将所有张量逐个 clone 后写成新的非 pickle safetensors。这个副本只改变序列化格式，不改变数值，并保留官方来源哈希。完整版本用于最终测试。

LLMDet 当前还使用纯 PyTorch deformable-attention fallback，没有原生 kernel；功能可运行，但速度不是官方最佳状态。

## APE-Ti 权重审计与阻塞

APE-Ti 权重结构测试通过：

- 1,151 个 tensor；
- 776,701,310 个参数；
- 414 个语言相关 tensor；
- `model_vision.model_language.net` 前缀下包含 390 个 tensor，说明语言分支权重确实存在。

但仅靠 key 和 tensor 数量不能证明它与当前配置严格完整匹配。是否可以跳过单独的 EVA-CLIP bootstrap，必须在 Linux/WSL2 官方环境中构建模型并检查 missing/unexpected keys 后再确认。

当前不能运行推理的原因是官方 APE 栈需要：

- Detectron2；
- Detrex；
- 编译后的 `ape._C` C++/CUDA 扩展；
- 与其固定版本兼容的 CUDA 编译环境。

本机当前没有已配置的 WSL 发行版、Docker、MSVC `cl` 或 CUDA Toolkit `nvcc`。继续在现有 Windows 环境硬装非官方 wheel 会降低可复现性，因此本轮没有这么做。

## 当前路线决策

1. **第一可运行基座：MM-Grounding-DINO-T。** 扩大多域验证，重点看 AIC 类似的小目标、区域结构和长 Query，不直接依据 60 条 RefCOCO 探针生成平台提交。
2. **APE-Ti：保持高优先级，但先解决官方 Linux/WSL2 环境。** 它仍是区域/结构目标最值得测试的模型；当前“未跑”是工程阻塞，不是模型失败。
3. **LLMDet-Swin-T：保留语义对照。** 当前没有超过 MM-Grounding-DINO-T，也没有提高候选并集 Oracle；除非长句/角色关系专项显示独有收益，否则不优先全量 AIC 推理。
4. 下一步仍坚持单变量：先完成更可靠的单模型比较，再决定是否训练、Tile、PIZA、IR 或 Depth，不把这些一起混入。

## 产物

- 机器摘要：`reports/sota_model_download_and_smoke_summary.json`
- APE checkpoint 审计：`outputs/sota_model_smoke/ape_ti/checkpoint_audit.json`
- APE 环境审计：`outputs/sota_model_smoke/ape_ti/runtime_audit.json`
- 固定探针：`outputs/sota_model_smoke/model_comparison_probe_60.jsonl`
- MM-Grounding-DINO-T 结果：`outputs/sota_model_smoke/mm_grounding_dino_t/probe_60/`
- LLMDet-Swin-T 结果：`outputs/sota_model_smoke/llmdet_swin_t/probe_60_complete_safetensors/`
- 两模型对照：`outputs/sota_model_smoke/model_probe_comparison.json`
