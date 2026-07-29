# 2026 AIC 第一版完整 Baseline 报告

## 1. 执行结论

第一版 `Florence-2 RGB-only` baseline 已建立并完成端到端验证：

```text
queries.json
  → 只读加载 Visible 与原始 Query
  → 本地 Florence-2-large-ft GPU 推理
  → 候选像素框解析
  → 归一化与合法性检查
  → predictions_debug.jsonl 断点日志
  → 完整 JSON 保真复制并只写 bbox
  → 只含提交 JSON 的 ZIP
```

工程已完成正式 9,555 条 Query 的全量推理、提交 JSON/ZIP 生成与独立读回审计。
结果是可复现的 RGB-only 排行榜起点，不代表模型精度上限；官方唯一样例定位
失败和全量结果中的大框偏置仍作为已知限制保留。

## 2. 官方示例 bbox 核验

官方 `000108_001`：

- Query：`The silver light bulb inside the bubble house`
- RGB：`1920 × 1080`
- 官方归一化 bbox：`[0.7718, 0.9249, 0.8129, 0.9762]`
- 转换后像素 bbox：`[1481.856, 998.892, 1560.768, 1054.296]`

原图叠加和局部放大均显示，官方框紧密覆盖右下角透明泡泡屋内部的银色灯泡。
因此该样例的标注语义与坐标均合理。它只能证明这一条公开标注合理，不能据此
断言全部测试标注都无误。

Florence 预测：

- 像素 bbox：`[843.84, 442.26, 928.32, 530.82]`
- 归一化 bbox：约 `[0.4395, 0.4095, 0.4835, 0.4915]`
- IoU：`0`
- ACC@0.5：`False`

预测框覆盖树上的蓝色球形装饰，失败原因是模型选错目标，不是“小数 GT 对整数
像素预测”的坐标转换问题。官方框只占整图约 `0.210843%`，属于明显小目标。

## 3. 对网页端分析的审核

### 保留的正确方向

1. 先建立工程与提交闭环，再加入复杂三模态融合。
2. 第一版保持 RGB-only，避免把模型、Depth、IR、Query 清洗问题混在一起。
3. 正式数据无 bbox，只能做无标签 dry run；最终分数需要排行榜反馈。
4. 后续使用 GroundingDINO/Florence 生成候选，再用关系、Depth、IR 重排序。

### 已修正的内容

1. 官方本地示例只有 `000108_001` 一条，不能按“三个官方样例”实现。
2. `9,615` 是删减前旧版本，当前正式基准固定为 `9,555`，不再追查 60 条差异。
3. 单条样例只能做 sanity test，不能用于比较 Florence 与 GroundingDINO 的总体
   准确率。
4. RefCOCO 可用于回归测试，但不能替代 AIC 目标域验证；同时需要核对模型是否
   已在相关数据上微调，避免把训练内分布结果称为独立泛化证据。
5. v0 先使用已经完整部署并验证可运行的 Florence；GroundingDINO 作为下一版
   候选生成器，而不是阻塞第一版闭环。
6. 有限样本运行禁止生成正式 ZIP；只有 9,555 条全部完成才开放提交生成。
7. fallback 不能静默发生。v0 对全量提交采用固定中心框作为最后兜底，并在
   debug JSONL、汇总和 `failure_cases.csv` 中逐条标记。

## 4. 环境与模型

- Python：`3.11.15`
- PyTorch：`2.5.1+cu121`
- Transformers：`4.57.6`
- Accelerate：`1.14.0`
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
- 模型：本地 `Florence-2-large-ft`（路径由私有配置指定）
- 权重大小：`1,540,980,506` bytes
- 权重 SHA-256：
  `8B4E610C952EEF90A836C56CDA0F398A672A3A6CA7B4D96B0E09A86DEE42E2C3`
- 完整模型工件清单 SHA-256：
  `5FD1E3D6A5859942A39A1303AD2BCA0FDF2A2111685B683859F5AC1285319507`
- 兼容配置：`attn_implementation="eager"`、`use_cache=False`

完整工件指纹在全量运行后的代码审查阶段补算。模型目录内所有文件的修改时间均早于
本次全量推理，且权重与 Query 哈希未变化；运行指纹已升级为 schema v2，使当前输出
可以继续接受新版 `--resume` 的一致性校验。

## 5. 测试与验证

### 自动化测试

`23 passed`，覆盖：

- 官方 bbox 已知坐标换算；
- pixel/normalized 双向换算；
- IoU 与 ACC@0.5；
- 非法、越界、NaN 和反向框拒绝；
- 中文 Windows 路径下的官方样例读取；
- Florence processor 结果解析与候选选择；
- 推理日志与显式 fallback；
- 中断后的 checkpoint 复用；
- 原始 JSON 字段保真；
- Query ID 集合一致性；
- ZIP 中只含正确提交 JSON。
- 空输出目录可用 `--resume` 初始化；
- resume 指纹不一致时拒绝复用旧 checkpoint；
- 完整模型工件中非权重文件变化也会改变运行指纹；
- CUDA/OOM 等 RuntimeError 和配置错误不会被静默转换为中心 fallback；
- 非数字候选坐标按非法模型输出进入显式 fallback。

另外：

- Python 编译检查通过；
- `pip check`：`No broken requirements found`；
- 所有包导入成功。

### 正式数据只读核验

- Query：`9,555`
- 唯一 Visible：`2,000`
- 三模态缺失文件：`0`
- 按 Query 路径计数：PNG `9,378`，JPG `177`

### 正式数据 100 条 smoke test

- 处理：`100/100`
- 合法 bbox：`100/100`
- fallback：`0`
- 系统错误：`0`
- 平均延迟：约 `1,019 ms/Query`
- P50：约 `944 ms/Query`
- P95：约 `1,572 ms/Query`
- 多候选输出：`70/100`
- bbox 面积大于整图 50%：`11/100`
- bbox 面积中位数：约 `3.83%`
- 最大 bbox 面积：约 `99.60%`

这些面积统计不是 ACC，但仍说明固定选择第一个 Florence 候选存在大框偏置。

### 正式数据 9,555 条全量推理

- 合法 bbox：`9,555/9,555`（100%）
- fallback：`0`
- 系统错误：`0`
- PNG Query：`9,378`
- JPG Query：`177`
- 候选框数量均值：约 `2.183`
- 多候选比例：约 `60.27%`
- bbox 面积中位数：约 `2.35%`
- bbox 面积 P95：约 `35.33%`
- bbox 面积超过整图 50%：`264`（约 `2.76%`）
- 平均模型延迟：约 `925 ms/Query`
- P50 / P95：约 `867 / 1,337 ms`
- 模型推理时间合计：约 `8,843 秒`
- 本次墙钟耗时：约 `9,187 秒`（2 小时 33 分 7 秒）
- PyTorch 峰值显存分配：`1,876,689,408` bytes

提交 ZIP SHA-256：

```text
52CFF325FBB0841D5E670645965F99BEF46C580BB0000798FC4CE7FE18081393
```

独立审计确认 Query ID 集合完全一致、原字段保真、所有 bbox 为合法有限浮点数、
debug 日志恰好 9,555 条且唯一，ZIP 中只包含与磁盘 JSON 字节一致的
`predictions_submission.json`。

### JPG 域 smoke test

对 `004006_001`（640×360 JPG）完成 GPU 推理：

- Query：`The rear wheel of the van on the far left.`
- 返回候选：2 个
- 选中归一化 bbox：
  `[0.4695, 0.6815, 0.4865, 0.7085]`
- 延迟：约 `1,058 ms`

这证明 v0 的 RGB 读取与模型推理可覆盖 JPG Visible 域。它不验证 JPG Depth 的
物理含义，因为 v0 根本不把 Depth 输入模型。

### 提交链路

使用官方单样例完整执行了：

```text
推理 → predictions_submission.json → predictions_submission.zip
```

ZIP 中只有 `predictions_submission.json`；记录只含原始
`visible/infrared/depth/query` 与算法生成的 `bbox`。

## 6. 代码审查

当前目录是有效 Git 仓库；`v0.2.0` 变更通过自动化测试并以提交前后的 Git diff
进行 Standards/Spec 双轴审查。

### Standards

- 未发现正式源代码中硬编码 `000108_001`、`004006_001` 或测试集 bbox。
- 所有官方数据访问均为只读。
- 写操作只落到用户指定的 baseline 输出目录。
- 未发现宽泛 `except Exception`、动态 `eval/exec` 或静默失败。
- fallback、错误、原始模型文本和耗时均可审计。
- 23 项自动化测试、编译与依赖完整性检查均通过。

### Spec

- 已实现数据→模型→bbox→日志→JSON→ZIP 完整闭环。
- 已阻止有限样本运行生成“正式提交”。
- 已实现并验证全量长任务所需的逐条落盘和安全 `--resume`。
- resume 指纹覆盖 queries、完整模型目录工件及全部关键推理参数。
- 配置错误和系统错误立即中止；只有无候选或非法框可使用显式 fallback。
- 已保留正式数据和官方样例原文件。
- v0 明确只使用 RGB 与原始 Query，符合阶段边界。

双轴审查提出的完整模型指纹、配置错误分类、非数字坐标分类和重复写入问题均已修复，
未留下阻止 baseline v0 运行的关键问题。

## 7. 当前限制与下一步

第一版是“完整、可复现、可测量”的工程基线，但不是高质量方案。最优先的三个
改进是：

1. **小目标多尺度/切片**：官方唯一样例目标仅占 0.21%，需要在全图之外增加
   高分辨率 tile 候选。
2. **候选选择器**：保留 Florence/GroundingDINO 的全部框，利用实体词、面积、
   序数和空间关系进行重排序，不能固定选择第一个候选。
3. **受控加入 IR/Depth**：先做 `RGB`、`RGB+IR`、`RGB+Depth` 消融；JPG Depth
   只能作为未知相对信号，不能解释为毫米。

本次全量实际墙钟耗时约 `2 小时 33 分`。正式启动与断点续跑命令见项目 README。
