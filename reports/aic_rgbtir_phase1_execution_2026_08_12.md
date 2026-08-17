# AIC RGB–TIR Phase 1 云端执行记录

日期：2026-08-12
分支：`exp/aic-rgbtir-phase1-v1`
状态：`PHASE_1_RELEASE_READY`

## 1. 运行资源

- GPU：NVIDIA GeForce RTX 4090，24,564 MiB
- 隔离环境：`/home/featurize/work/envs/aic_rgbtir_phase1`
- PyTorch：`2.6.0+cu124`
- Transformers：`4.57.6`
- 模型：`Qwen/Qwen3-VL-8B-Instruct`
- revision：`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- 只加载视觉权重 shard，不加载 8B 语言模型。

## 2. 数据状态

- Featurize 持久数据集：`03_RGBT_GroundBench.zip`
- 平台显示体积：约 28.1 GB
- 下载原始 ZIP：`/home/featurize/dataset_downloads/03_RGBT_GroundBench.zip`
- 实际字节数：`28,072,209,745`
- 展开后根目录：`/home/featurize/data/RGBT_GroundBench_unpacked/03_RGBT_GroundBench`
- `extracted/image_data` 文件数：`43,070`，即 `21,535` 对 RGB/TIR。
- clean train：`26,477`
- tracer：`400 train / 100 official val`

平台持久数据集可在后续新建实例时继续使用，不需要每次从本机重新上传。实例本地的解压目录是临时资源，新实例仍需从持久数据集下载或挂载后解压。

## 3. 已通过门槛

### 真实权重 smoke

- 状态：`PHASE_1_SMOKE_GO`
- 样本：10
- gate=0 最大误差：`0.0`
- hook 层：`8 / 16 / 24 / final`
- 每层真实特征形状：`[1280, 1152]`
- 可训练 TIR adapter 参数：`8,957,952`
- 梯度仅出现在 TIR adapter：是
- Qwen base 参数哈希前后一致：是
- 峰值显存：约 `3.35 GiB`

### overfit100

- 状态：`PHASE_1_WARMUP_GO`
- 200 step，跳过 0
- baseline mean loss：`0.1683081590`
- final mean loss：`0.1102080377`
- 相对改善：`34.52%`
- 要求：至少 `10%`

### tracer400

- 状态：`PHASE_1_WARMUP_GO`
- 800 step，跳过 0
- official-val baseline mean loss：`0.1527101608`
- official-val final mean loss：`0.0933405722`
- 相对改善：`38.88%`
- 要求：至少 `2%`
- Qwen base 未变：是

## 4. Full warmup

- 命令：`python -u tools/run_rgbtir_phase1.py warmup --config configs/aic_rgbtir_phase1.cloud.local.yaml --mode full --resume`
- 日志：`/home/featurize/aic_cloud/logs/rgbtir_phase1_full.log`
- 输出：`/home/featurize/aic_cloud/outputs/aic_rgbtir_phase1_v1/full`
- 训练边界：只训练 TIR rank-48 adapter；RGB 视觉主干、语言模型、fusion gate 和 bbox head 均未训练。
- 本阶段不使用第二模型 fallback，不产生 AIC 提交包。

完成结果：

- 训练步数：`26,477 / 26,477`
- 跳过样本：`0`
- official-val baseline mean loss：`0.1527101608`
- official-val final mean loss：`0.0733431448`
- 相对改善：`51.97%`
- 要求：至少 `2%`
- 四层平均余弦相似度：
  - layer 8：`0.71715 → 0.87767`
  - layer 16：`0.78759 → 0.90987`
  - layer 24：`0.95479 → 0.97669`
  - final：`0.99539 → 0.99694`
- Qwen base 参数未变：是
- 结论：`PHASE_1_WARMUP_GO`

## 5. 实际环境问题与固定解法

1. Featurize `dataset extract` 在含嵌套归档的数据集上发生 `UnexpectedHeaderError`。
   固定解法：使用 `featurize dataset download` 保留原始 ZIP，再手动解压并核对文件数。
2. 旧打包方式在 ZIP 条目中保留 Windows 反斜杠，Linux `unzip` 不能按目录还原。
   固定解法：使用 `ZipArchive` 生成 POSIX `/` 条目。
3. 平台系统 PyTorch 为 `2.2.2+cu121`，与当前 Qwen3-VL 链路不合。
   固定解法：使用完全隔离 venv 安装 `torch 2.6.0+cu124`，不覆盖系统环境。
4. 在 meta device 构建 Qwen 视觉塔后直接赋权重，非持久 buffer 仍可留在 meta，后续 `.to()` 失败。
   固定解法：视觉塔先在 CPU 实体化，再加载单个 vision shard，最后转 BF16/GPU。

## 6. 声明边界

当前 GO 仅证明：TIR adapter 能在配对 RGB/TIR 目标区域上学习冻结 Qwen RGB teacher 的视觉表征。它尚不能证明 Query 已控制红外参与，也不能证明 AIC ACC 会提升。这两项需要 Phase 2 的 query-aware fusion 与 grounding 评估。

## 7. 本地复现包

- 路径：`D:\12525\Documents\pytorch\aic_rgbtir_phase1_prerental_bundle_20260812.zip`
- 字节数：`4,351,088`
- SHA-256：`DA8E194593C99793067CDEF5EAB6B786D57C1105EC93245D44A0FB41663AE5E5`
- 已包含本轮 Linux ZIP 路径、隔离环境和 Qwen 视觉塔 meta-buffer 修复。

## 8. Phase 1 发布产物

本地根目录：

`D:\12525\Documents\pytorch\baseline_v0\outputs\aic_rgbtir_phase1_release_20260812`

主 Adapter：

`qwen3vl8b_tir_rank48_adapter_phase1.pt`

- 张量数：`108`
- 参数数：`8,957,952`
- SHA-256：`F8B81349886702497ABC936BCAF1DB4660CF7B3290B2569AE80C303CE362DF7B`
- 本地已校验：manifest 中 11 个文件的字节数与 SHA-256 全部一致；Adapter 中全部张量有限，键均属于 TIR path。

可续训原始 checkpoint 与日志保存在：

`D:\12525\Documents\pytorch\baseline_v0\outputs\aic_rgbtir_phase1_release_20260812\raw_full_checkpoint`

完整本地归档（不是 AIC 提交包）：

- 路径：`D:\12525\Documents\pytorch\aic_rgbtir_phase1_release_20260812_final.zip`
- 字节数：`56,140,300`
- SHA-256：`DC9925B395EE4FA2C31027FA3247E6F750A29D3984118400C58B11E3A4B3CAEF`

本地验证记录：

`D:\12525\Documents\pytorch\baseline_v0\outputs\aic_rgbtir_phase1_release_20260812\local_validation.json`

## 9. Exit Gate

Phase 1 的所有硬门槛已通过，允许进入 Phase 2，但 Phase 2 必须继续保持单变量：

1. 先冻结 RGB 主路和已训 TIR Adapter，只训零初始化的 query-aware fusion/gate；
2. 必须验证 `gate=0` 仍与历史 RGB-only 路径等价；
3. 先在 RGBT-GroundBench 的独立验证集比较 RGB-only 与 RGB+TIR；
4. 只有融合不破坏正常光 RGB 样本，才接入 Query-to-bbox grounding 并做 AIC 受控平台验证；
5. 不在未完成配准或 mask 安全门槛时强制全量 IR 融合。
