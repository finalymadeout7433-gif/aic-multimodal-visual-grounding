# AIC Qwen3-VL RGB–TIR Adapter 启动方案

> 日期：2026-08-12（Asia/Shanghai）
> 输入报告：`D:\ZOTERO\reports\RGBT-GroundBench_Qwen3-VL红外融合深度分析.html`
> 目标：在不破坏当前 RGB 能力、不使用跨模型 fallback 的前提下，把 RGBT-VGNet 的 AMA 思想迁移到 Qwen3-VL，并明确 8B 开发、30B 迁移的接口边界。

## 1. 最终决策

**先在 Qwen3-VL-8B-Instruct BF16 上开发与验证；30B 只作为第二阶段迁移目标。**

这不是因为 30B 不强，而是因为：

1. 8B 已有真实 AIC 平台 `0.7582`，与当前 `0.7757` 参考结果只差 1.75 pp，足以代表真实任务能力；
2. 当前 `0.7757` 是 30B-FP8 主模型加 13 条 8B 补框，不是纯 30B 无 fallback 结果；
3. 8B 是 dense 模型，LoRA、梯度检查点和单卡训练链路更成熟；
4. 30B-A3B 虽然每 token 只激活约 3B 参数，但训练显存仍要保存总计约 30B 参数；
5. 当前 30B 权重是 FP8 推理版本，不应作为首个训练底座；
6. Qwen 官方明确说明 Qwen3-VL MoE 当前不支持 DeepSpeed ZeRO-3，结构调试成本明显更高。

开发顺序：

```text
8B BF16：接口/数据/AMA/融合/回归头验证
        ↓
8B：完整 RGBT 外部验证与 RGB replay 回归测试
        ↓
30B BF16：重新初始化或有条件迁移视觉 Adapter
        ↓
30B：单变量 AIC 平台提交
```

## 2. 8B 与 30B 的关键差别

| 项目 | Qwen3-VL-8B-Instruct | Qwen3-VL-30B-A3B-Instruct | 对方案的影响 |
|---|---|---|---|
| 语言模型 | Dense 8B | MoE，总参数约 30B、激活约 3B | 30B 推理计算较省，但训练存储仍按总参数计 |
| 当前 AIC 结果 | 0.7582，纯 8B 提交 | 0.7757，含 13 条 8B 补框 | 8B 是干净开发控制组；0.7757 是参考上限而非纯 30B 对照 |
| Vision depth/hidden | 27 层 / 1152 | 27 层 / 1152 | pre-merger Adapter/Fusion 可以共用结构 |
| DeepStack | 8/16/24 | 8/16/24 | 两者可共用相同融合插入层 |
| Vision merger 输出 | 4096 | 2048 | merger 后模块不能直接复制 |
| 训练权重 | 使用 BF16 原始版 | 应使用 BF16 原始版 | FP8 checkpoint 仅作为推理参考，不作为默认训练底座 |
| 单卡可行性 | 5090 32GB 可做冻结主干、batch=1 的 Adapter 原型 | 5090 32GB 不建议训练 | 30B 建议至少 80GB，稳妥方案是 2×80GB |

### 最重要的架构结论

**把可迁移模块放在 1152 维 pre-merger hidden states 上。**

如果融合发生在最终 merger 后：

```text
8B feature dim = 4096
30B feature dim = 2048
```

模块形状和训练权重都无法直接迁移。若融合发生在视觉层 8/16/24 和 final 的 1152 维 hidden states 上，模块结构可以共用，再交给各自原生的 DeepStack merger / final merger 投影到 4096 或 2048。

在迁移前还需执行一次视觉权重等价性审计：

- 比较 8B BF16 与 30B BF16 的 `patch_embed / pos_embed / blocks.0..26` tensor；
- 若 tensor hash 完全相同，可直接复制 TIR LoRA 与 pre-merger fusion 权重；
- 若仅形状相同但权重不同，只复制模块结构，以 8B 权重 warm start，并在 30B 上重新适配；
- 不对 FP8 tensor 与 BF16 tensor做直接等价判断。

## 3. HTML 报告中正确且应保留的部分

1. 共享 Qwen vision tower，不复制第二套完整 ViT；
2. RGB/TIR 分别使用低/高容量 LoRA；
3. DeepStack 8/16/24 与 final 都必须处理，不能只改最终 projector；
4. RGB/TIR 必须使用完全相同的 resize、crop、pad 和 `grid_thw`；
5. TPF 必须按 `grid_thw` 支持动态长宽比，不能假设方形 token 网格；
6. 新模块先训练，LLM/vision base 后解冻；
7. 保留 RGB-only、TIR-only、RGB+TIR、分辨率与 modality dropout 消融；
8. 论文原则和仓库 `IAFv3` 实现必须分开复现与比较。

## 4. 报告还需要补足的部分

### 4.1 不能把双图 baseline 当成融合模型

默认 Qwen 多图输入把 RGB/TIR 当成两个普通图片序列，不知道它们空间对齐，也没有模态专用参数。它只用于验证：

- TIR 预处理是否可读；
- 原始分辨率是否保留；
- 两图是否增加无效框/拒答；
- prompt 和坐标解析是否正确。

它不能证明模型已经学会红外表示或 patch 对齐。

### 4.2 AIC 必须显式处理黑边和弱对齐

`PairedRGBTProcessor` 必须同时输出：

```python
rgb_pixel_values
tir_pixel_values
grid_thw
ir_valid_mask
shared_geometry_transform
original_rgb_size
normalized_bbox
```

硬约束：

- `rgb_grid_thw == tir_grid_thw`；
- 黑边不能被当作低温背景；
- `ir_valid_mask` 经过相同 resize/pad 后降采样到 patch/merge grid；
- 最终 bbox 永远定义在 RGB 原始坐标系；
- 所有随机几何增强共享参数；
- 第一版加入小幅 rotation/scale/translation 扰动，提高对 AIC 弱对齐的鲁棒性。

### 4.3 最终输出不应依赖生成式坐标

当前 Qwen 零样本方案的 fallback 来自拒答、空数组、退化框或坐标解析失败。训练后的多模态模型可以从结构上避免这一问题。

推荐增加一个专用 `<bbox_query>` token：

```text
RGB/TIR fused visual tokens + Query tokens + <bbox_query>
                              ↓
                    Qwen last hidden state
                              ↓
              MLP → sigmoid([cx, cy, w, h])
                              ↓
                   deterministic valid bbox
```

主输出使用直接回归头，损失为：

```text
L_bbox = λ1 L1 + λ2 GIoU
```

生成式 JSON bbox 可保留为辅助任务/对照，但不再作为唯一提交来源。这样无需另一个模型补框，最终框可通过有界参数化保证合法。

需要做三组对照：

- G1：生成式 JSON；
- G2：直接 bbox head；
- G3：生成 CE + bbox head 多任务。

### 4.4 AMA 在 Qwen 中的 target modules 必须重选

RGBT-VGNet 的 CLIP target modules 不能原样复制。Qwen3-VL vision attention 使用融合的 `qkv` 与 `proj`。

第一轮仅适配：

```text
visual.blocks.*.attn.qkv
visual.blocks.*.attn.proj
```

消融后再考虑：

```text
visual.blocks.*.mlp.linear_fc1
visual.blocks.*.mlp.linear_fc2
```

初始 rank：

```text
RGB rank ∈ {0, 4, 8}
TIR rank ∈ {16, 32, 48}
```

RGB rank 0 是最重要的保护基线。论文的 `16/48` 只能作为 Qwen 消融点，不能当作答案。

### 4.5 训练数据必须防止道路域主导

RGBT-GroundBench 约 95% 是道路参与者，不能单独训练整个 Qwen。训练策略：

- 清洗 RGBT train 中命中否定/目标缺失/bbox 元语言的记录；当前两个高风险规则在 train 中命中 125 条；
- FLIR/M3FD/MFAD source-balanced sampling；
- 天气/光照用于分层，不人为放大雾雨比例；
- 融合阶段加入 RGB grounding replay，重点补序数、建筑结构、区域与开放词汇；
- 不使用 AIC 测试数据、预测框或伪标签训练。

## 5. 推荐的软件结构

不要修改 `transformers/modeling_qwen3_vl.py`。在项目中新增独立包装层：

```text
src/aic_rgbtir/
  config.py
  paired_processor.py
  modality_lora.py
  qwen_vision_encoder.py
  fusion.py
  bbox_head.py
  model.py
  losses.py
  dataset.py
  trainer.py
```

核心接口：

```python
class Qwen3VLRGBTAdapter(nn.Module):
    def forward(
        self,
        input_ids,
        attention_mask,
        rgb_pixel_values,
        tir_pixel_values,
        image_grid_thw,
        ir_valid_mask,
        bbox_labels=None,
    ):
        ...
```

### 5.1 Vision encoder 输出

两路共享原始 ViT base，分别激活 RGB/TIR adapter，返回未经过 merger 的：

```text
rgb_hidden[8],  rgb_hidden[16], rgb_hidden[24], rgb_hidden[final]
tir_hidden[8],  tir_hidden[16], tir_hidden[24], tir_hidden[final]
```

第一版在每个观测层独立执行零初始化残差融合：

```text
F_l = R_l + sigmoid(g_l) * M_l * Project(T_l)
```

然后把 `F_8/F_16/F_24/F_final` 分别送入模型原生 DeepStack merger 与 final merger。该结构同时适配 8B 和 30B。

### 5.2 为后续模块预留接口

虽然本轮不训练 Query-conditioned gate 和 CoDAF，接口应预留：

```python
fusion(rgb, tir, valid_mask, grid_thw, query_state=None, quality_state=None)
```

第一版 `query_state=None`，只验证 AMA + mask + residual。后续可以单变量替换为 LAVS/TPF/CoDAF，而不重写主模型。

## 6. 分阶段训练计划

### Phase 0：数据和等价性测试，不训练

- 生成 clean RGBT manifests；
- 统一 RGB/TIR 几何预处理；
- 验证所有 paired sample 的 `grid_thw` 一致；
- 验证 gate 初始化接近零时，RGBT wrapper 与原 RGB-only 输出等价；
- 固定 500 条 tracer split，覆盖三 source、低光、正常光、小目标、弱对齐代理。

停止条件：任何坐标、mask、grid 或 RGB 等价性测试失败都不得进入训练。

### Phase 1：8B vision-only TIR warmup

加载 8B BF16 的 vision tower，冻结 base 与 RGB 路径，只训练：

- TIR normalization；
- TIR LoRA；
- modality embedding；
- 临时 bbox-aware/ROI alignment head。

目标不是直接生成最终 AIC 框，而是让 TIR 表示进入 Qwen 的视觉语义空间。使用 bbox 前景区域的语义/几何对齐，避免把整张 RGB/TIR 强制 MSE 成完全相同。

### Phase 2：8B AMA + zero-init fusion

加载完整 8B BF16，冻结 LLM 与 vision base，训练：

- TIR LoRA；
- 可选低 rank RGB LoRA；
- 8/16/24/final fusion；
- `<bbox_query>` embedding；
- bbox head。

数据混合：RGBT + RGB replay；加入 modality dropout 与轻量弱错位增强。

### Phase 3：8B grounding multi-task

在 Phase 2 稳定后，对比直接 bbox head、生成式 bbox 和多任务方案。只有在外部验证满足以下条件才进入 30B：

- RGB+TIR 超过同一模型 RGB-only；
- normal-light 不明显下降；
- small/low-light/weak-aligned 分组同向改善；
- RGB replay 不发生明显遗忘；
- invalid bbox = 0；
- 无跨模型 fallback。

### Phase 4：迁移到 30B

使用 30B BF16 原始权重；冻结 MoE decoder，先只加载/重训视觉 Adapter、fusion 和 bbox head。不要以 FP8 inference checkpoint 开始训练。

若 8B/30B vision base hash 完全一致，可直接移植 pre-merger LoRA/fusion；否则仅 warm start 并重新训练。

30B 训练建议：

- 最低可尝试：1×80GB，batch=1、gradient checkpointing、短 tracer；
- 稳妥：2×80GB；
- 5090 32GB：适合 8B Adapter，不建议 30B 训练；
- 48GB：可做受限试验，但不建议作为主训练环境；
- 不使用当前官方不支持的 MoE ZeRO-3 路线。

## 7. 第一轮必须实现的测试

1. `test_rgb_equivalence_gate_zero`：gate=0 时 wrapper 与原模型 RGB features 等价；
2. `test_paired_grid_thw_equal`：每个 RGB/TIR pair 的 grid 完全一致；
3. `test_ir_valid_mask_downsample`：黑边在 patch/merge grid 上仍被屏蔽；
4. `test_shared_geometry_bbox_roundtrip`：resize/pad/逆变换后 bbox 可逆；
5. `test_modality_lora_isolation`：TIR backward 不更新 RGB adapter/base；
6. `test_bbox_head_validity`：任意有限输入都输出合法 `[0,1]` bbox；
7. `test_rgb_only_missing_tir`：TIR 缺失时同一模型仍输出合法框；
8. `test_checkpoint_portable_shapes`：8B/30B pre-merger 模块形状可兼容；
9. `test_no_upstream_patch`：不修改安装环境中的 Transformers 源文件；
10. `test_no_cross_model_fallback`：提交生成链路不读取第二模型预测。

## 8. 现在应该从哪里开始

本轮只启动 Phase 0，不立刻租多卡训练：

1. 固定 Qwen3-VL-8B-Instruct BF16 revision；
2. 建立 `src/aic_rgbtir` 包装器骨架；
3. 生成 clean train/val/test manifests；
4. 实现 paired processor、`ir_valid_mask`、共同 `grid_thw`；
5. 完成 gate=0 的 RGB 等价性单元测试；
6. 再租 5090 32GB 进行 100/1,000 条 tracer 训练；
7. tracer 通过后才跑 RGBT train；
8. 8B 方案冻结后，再为 30B 计算资源和迁移。

## 9. 结论

8B 不是最终上限，而是当前最有价值的实验载体：它已经在 AIC 得到 0.7582，和 30B 参考结果足够接近，同时训练链路简单得多。只要所有可迁移模块放在 1152 维 pre-merger 接口上，8B 的工程工作不会浪费。

与 HTML 报告相比，本方案新增了四个必要约束：

1. pre-merger 统一接口，解决 8B/30B 输出维度不同；
2. 直接 bbox head，结构性消除 fallback；
3. AIC 黑边 mask 与弱错位增强；
4. clean/source-balanced RGBT + RGB replay，防止道路域负迁移。

因此，正确起点是 **8B BF16 + AMA-only + mask + zero-init residual + deterministic bbox head**，而不是 30B-FP8 全量训练，也不是一次性实现 AMA/LAVS/TPF/CoDAF。

## 10. 一手资料

- Qwen3-VL 官方仓库：https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL 官方微调框架：https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune
- Qwen3-VL-8B 配置：https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json
- Qwen3-VL-30B-A3B-FP8 配置：https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8/blob/main/config.json
- RGBT-GroundBench 论文：https://arxiv.org/html/2512.24561
- RGBT-GroundBench 官方代码：https://github.com/crazyxiaoxi/RGBT-GroundBench
