# 2026 AIC RGB-only Baseline v0

这是第一版可完整运行的视觉定位基线：

```text
Visible RGB + 原始 Query
        ↓
Florence-2-large-ft
        ↓
像素 bbox → 归一化 bbox
        ↓
逐条调试日志 + 合法性检查
        ↓
完整 submission.json + ZIP
```

v0 **不使用** Infrared、Depth、Query 改写、模型集成或人工修框。官方原始
JSON 和图像只读，所有结果写入本目录的 `outputs`。

## 环境

每次新开终端先激活独立环境：

```powershell
conda activate aic-baseline
```

然后在本目录安装项目本身（首次执行一次即可）：

```powershell
python -m pip install -e .
```

模型已经位于：

```text
D:\AI_Models\modelscope\AI-ModelScope\Florence-2-large-ft
```

已验证的兼容设置是 `attn_implementation="eager"` 与
`generate(use_cache=False)`。

## 运行顺序

1. 只读核验正式数据的 9,555 条 Query 和全部三模态路径：

```powershell
python -m aic_baseline.cli validate --config configs\florence2_rgb_only.yaml
```

2. 跑官方唯一带标签样例：

```powershell
python -m aic_baseline.cli sample --config configs\official_sample.yaml
```

3. 正式数据小规模 smoke test（不会生成提交 ZIP）：

```powershell
python -m aic_baseline.cli infer `
  --config configs\florence2_rgb_only.yaml `
  --limit 10 `
  --output-dir outputs\smoke_10
```

4. 全量 9,555 条推理并生成提交 JSON/ZIP：

```powershell
python -m aic_baseline.cli infer `
  --config configs\florence2_rgb_only.yaml `
  --build-submission `
  --resume
```

只有完整处理全部 Query 时，程序才允许生成正式提交 ZIP。有限样本运行被明确
标记为 smoke run，防止误提交。`--resume` 会从
`predictions_debug.jsonl` 断点继续，避免长时间全量推理因中断而从头开始。

## 输出

- `predictions_debug.jsonl`：原始生成文本、候选框、最终框、耗时和 fallback。
- `predictions.json`：`query_id → bbox` 的内部结果。
- `inference_summary.json`：有效框率、fallback 率和耗时统计。
- `environment.json`：Python、PyTorch、Transformers、GPU 和模型权重哈希。
- `predictions_submission.json`：只在全量运行后生成。
- `predictions_submission.zip`：ZIP 中只含提交 JSON。

## 测试

```powershell
python -m pytest -q
```

官方单样例只能证明数据、bbox、模型和评测链路能够工作，不能代表模型在比赛
中的总体 ACC@0.5。
