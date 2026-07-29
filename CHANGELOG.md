# Changelog

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
