# Changelog

## v0.2.1 — 2026-07-29

- 记录 Florence-2 RGB-only v0.2.0 平台 ACC@0.5 `0.4980`；
- 新增外部 grounding 数据的 ZIP 校验、安全解压、统一 JSONL 和可视化工具；
- 完成 RefCOCO、RefCOCO+、RefCOCOg、gRefCOCO 与 SUN-Spot 数据准备；
- 核对全部训练 RGB/Depth 引用，核心 RGB split 保证图像互斥；
- 增加训练清单懒加载接口与 GroundingDINO Swin-T 下一阶段策略。
- 固定外部标注 SHA-256，并用受限反序列化读取 Pickle；
- 预览改为不同图像抽样，并在图上显示 Query ID 与文本。

## v0.2.0 — 2026-07-29

- 为断点续跑增加 queries、完整模型工件清单和推理参数指纹，拒绝不一致的旧 checkpoint；
- 仅对无候选框或非法 bbox 使用显式 fallback，CUDA/OOM 等系统错误立即中止；
- 新增候选数量、bbox 面积、PNG/JPG 分域、耗时和 GPU 峰值审计指标；
- 23 项自动化测试全部通过；
- 完成正式 9,555 条 Query 的 Florence-2 RGB-only 全量推理和提交 ZIP 校验。

## v0.1.2 — 2026-07-29

- 新增公开安全的数据审计脚本和关键结论摘要；
- 将本机绝对路径移出公开配置，本地配置改为 Git 忽略文件；
- 精简 README 和仓库结构；
- 补齐远程版本标签。

## v0.1.1 — 2026-07-29

- 新增正式测试集隐藏标签下的五级验证路线；
- 说明正式 9,555 条 Query 只用于推理，不能用于训练；
- 说明初赛上传预测 JSON/ZIP，而非模型文件；
- 新增 Git 工作区、暂存区、本地仓库和 GitHub 远程仓库教学。

## v0.1.0 — 2026-07-29

第一版可完整运行的 AIC RGB-only baseline：

- 只读加载正式 9,555 条 Query；
- 本地 Florence-2-large-ft GPU 推理；
- bbox 转换、IoU、ACC@0.5 和合法性校验；
- debug JSONL、fallback 日志与断点续跑；
- 完整提交 JSON 和 ZIP 生成；
- 官方单样例、PNG smoke、JPG smoke 和 15 项自动化测试；
- VS Code 解释器、测试任务、调试入口和长任务入口。

已知限制：

- 官方唯一小目标样例定位失败；
- 复杂关系 Query 存在大框偏置；
- v0 尚未使用 IR、Depth、多尺度切片或候选框重排序。
