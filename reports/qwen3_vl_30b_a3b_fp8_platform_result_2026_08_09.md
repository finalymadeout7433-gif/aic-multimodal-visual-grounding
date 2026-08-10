# Qwen3-VL-30B-A3B-Instruct-FP8 平台结果记录（2026-08-09）

## 结论

这一轮 `Qwen3-VL-30B-A3B-Instruct-FP8 + 8B 历史最佳 fallback` 刷新了当前项目最高分：

| 字段 | 结果 |
|---|---:|
| 平台 ACC@0.5 | **0.7757** |
| 平台记录时间 | 2026-08-09 22:14:57 |
| 平台排名截图 | 用户提供截图，排名 10 |
| 主模型 | `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` |
| fallback 来源 | `Qwen3-VL-8B-Instruct` 平台 0.7582 那轮 |
| 输入模态 | Visible RGB + 原始英文 Query |
| 训练 | 无；零样本全量推理 |
| AIC Query 数 | 9,555 |
| 主模型 fallback | 13 |
| fallback 替换 | 13 条均替换为 8B 历史最佳 bbox |
| invalid bbox | 0 |
| 非 bbox 字段修改 | 0 |

这个结果说明：沿着 Qwen3-VL Instruct 系列扩大模型规模，在 AIC 上确实带来了收益。相比 `Qwen3-VL-8B-Instruct` 的 0.7582，本轮提升：

```text
0.0175 ACC@0.5
约 +1.75 pp
折合约 167 条 Query 的净正确数提升（按 9555 条估算）
```

## 可上传提交包

```text
D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\platform_upload_ready\AIC_Qwen3_VL_30B_A3B_FP8_8BInstructFallback_20260809.zip
```

完整本地归档目录：

```text
D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\platform_upload_ready\qwen3_vl_30b_a3b_fp8_cascade_20260809
```

SHA-256：

```text
f2486b013d740b5c89cc782e95b7eb4c11c8cbd1b8899879e9146b415ea8f9d8
```

ZIP 审计：

```text
zip_entries: ["predictions_submission.json"]
query_count: 9555
invalid_bbox_count: 0
modified_non_bbox_count: 0
```

## 策略说明

运行中发现 30B 仍会在极少数样本上输出无效结果，例如：

- `There is no ...`
- `[]`
- `[0.0, 0.0, 0.0, 0.0]`

因此最终提交没有直接使用 center fallback，而是采用级联：

```text
30B 正常 bbox
  ↓
如果该 query 在 runtime_events.jsonl 中 fallback=true
  ↓
使用 Qwen3-VL-8B-Instruct 0.7582 那轮对应 bbox
```

最终替换数量：

```text
13 / 9555
```

该比例很低，且 fallback 来源是已经通过平台验证的历史最佳模型，所以比中心框更稳。

## 与已有路线对比

| 模型 / 策略 | ACC@0.5 | 结论 |
|---|---:|---|
| Qwen3-VL-30B-A3B-FP8 + 8B fallback | **0.7757** | 当前最高分 |
| Qwen3-VL-8B-Instruct zero-shot | 0.7582 | 旧最高分；仍是 fallback 基线 |
| LocateAnything-3B zero-shot | 0.7210 | 小目标能力有效，但弱于 Qwen3-VL |
| Qwen3-VL-8B-Thinking cascade | 0.7194 | 推理时间长，分数下降 |
| EGM-Qwen3-VL-8B zero-shot | 0.5333 | 放弃主线 |

## 后续含义

1. 当前主线应从 `Qwen3-VL-8B-Instruct` 升级为 `Qwen3-VL-30B-A3B-Instruct-FP8`。
2. Thinking 分支暂不优先，成本更高且已验证不如 Instruct。
3. EGM 分支暂不继续全量投入。
4. 下一轮如果测新模型，仍应保留“历史最佳 fallback”机制，避免少量解析失败拖累提交合法性。
5. 继续测试新模型时，应优先和 0.7757 比，而不是再和 0.7582 比。
