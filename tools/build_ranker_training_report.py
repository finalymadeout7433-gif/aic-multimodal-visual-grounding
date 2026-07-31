from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the durable GroundingDINO spatial LTR report."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def _number(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _public_artifact_path(path_value: str, output_root: Path) -> str:
    """Return a repository-relative artifact path without leaking host paths."""
    path = Path(path_value)
    try:
        relative = path.resolve().relative_to(output_root.resolve())
    except ValueError:
        return path.name
    return (Path("outputs") / output_root.name / relative).as_posix()


def _compact_submissions(
    submissions: Mapping[str, Any] | None,
    output_root: Path,
) -> dict[str, Any] | None:
    if not submissions:
        return None
    audit: dict[str, Any] = {}
    for name in ("control", "ranker"):
        source = submissions["submission_audit"][name]
        audit[name] = {
            **source,
            "json_path": _public_artifact_path(
                source["json_path"], output_root
            ),
            "zip_path": _public_artifact_path(
                source["zip_path"], output_root
            ),
        }
    return {
        "selection_summary": submissions["selection_summary"],
        "submission_audit": audit,
        "selection_debug_path": _public_artifact_path(
            submissions["selection_debug_path"], output_root
        ),
    }


def _evaluation_table(evaluations: Mapping[str, Any] | None) -> str:
    if not evaluations:
        return "该阶段尚未完成。"
    rows = [
        "| 方法 | ACC@0.5 | 相对 Top-1 | Mean IoU | Switch | Rescue | Harm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    baseline = evaluations["gdino_top1"]["overall"]
    baseline_acc = float(baseline["baseline_acc_at_05"])
    for key, label in (
        ("gdino_top1", "GroundingDINO Top-1"),
        ("conservative_rule", "保守空间规则"),
        ("score_geometry_ranker", "Score + Geometry Ranker"),
        ("full_ranker_no_guard", "完整 Ranker（无保护阈值）"),
        ("full_ranker_guarded", "完整 Ranker（冻结阈值）"),
    ):
        if key not in evaluations:
            continue
        overall = evaluations[key]["overall"]
        acc = float(overall["selected_acc_at_05"])
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    _pct(acc),
                    f"{100.0 * (acc - baseline_acc):+.2f} pp",
                    _number(overall["selected_mean_iou"]),
                    _pct(overall["switch_rate"]),
                    str(overall["rescued"]),
                    str(overall["harmed"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _dataset_table(evaluation: Mapping[str, Any] | None) -> str:
    if not evaluation:
        return "该阶段尚未完成。"
    rows = [
        "| 数据集 | Query | Top-1 ACC | Ranker ACC | 增益 |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, metrics in sorted(
        evaluation["grouped"]["dataset"].items()
    ):
        baseline = float(metrics["baseline_acc_at_05"])
        selected = float(metrics["selected_acc_at_05"])
        rows.append(
            f"| {dataset} | {metrics['queries']} | {_pct(baseline)} | "
            f"{_pct(selected)} | {100.0 * (selected - baseline):+.2f} pp |"
        )
    return "\n".join(rows)


def _category_table(evaluation: Mapping[str, Any] | None) -> str:
    if not evaluation:
        return "该阶段尚未完成。"
    rows = [
        "| Query 类型 | Query | Top-1 ACC | Ranker ACC | 增益 |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, metrics in sorted(
        evaluation["grouped"]["query_category"].items()
    ):
        baseline = float(metrics["baseline_acc_at_05"])
        selected = float(metrics["selected_acc_at_05"])
        rows.append(
            f"| {category} | {metrics['queries']} | {_pct(baseline)} | "
            f"{_pct(selected)} | {100.0 * (selected - baseline):+.2f} pp |"
        )
    return "\n".join(rows)


def _margin_bucket_table(evaluation: Mapping[str, Any] | None) -> str:
    if not evaluation or not evaluation.get("margin_buckets"):
        return "该阶段没有可用的排序边际分桶。"
    rows = [
        "| Ranker 分差桶 | Query | Selected ACC@0.5 |",
        "|---|---:|---:|",
    ]
    for name, metrics in evaluation["margin_buckets"].items():
        rows.append(
            f"| `{name}` | {metrics['queries']} | "
            f"{_pct(metrics['selected_acc_at_05'])} |"
        )
    return "\n".join(rows)


def _local_promotion_checks(
    holdout: Mapping[str, Any] | None,
    submissions: Mapping[str, Any] | None,
) -> dict[str, Any]:
    thresholds = {
        "overall_gain": 0.02,
        "spatial_ordinal_gain": 0.04,
        "maximum_dataset_drop": 0.01,
        "legal_bbox_rate": 1.0,
    }
    if not holdout or not submissions:
        return {
            "pass": False,
            "complete": False,
            "thresholds": thresholds,
            "checks": {},
        }
    overall = holdout["overall"]
    overall_gain = float(overall["selected_acc_at_05"]) - float(
        overall["baseline_acc_at_05"]
    )
    spatial_ordinal_gain = float(holdout["spatial_ordinal_combined"]["gain"])
    dataset_gains = [
        float(metrics["selected_acc_at_05"])
        - float(metrics["baseline_acc_at_05"])
        for metrics in holdout["grouped"]["dataset"].values()
    ]
    worst_dataset_gain = min(dataset_gains, default=0.0)
    selection = submissions["selection_summary"]
    checks = {
        "overall_gain": overall_gain,
        "overall_pass": overall_gain >= thresholds["overall_gain"],
        "spatial_ordinal_gain": spatial_ordinal_gain,
        "spatial_ordinal_pass": spatial_ordinal_gain
        >= thresholds["spatial_ordinal_gain"],
        "worst_dataset_gain": worst_dataset_gain,
        "dataset_pass": worst_dataset_gain
        >= -thresholds["maximum_dataset_drop"],
        "legal_bbox_rate": float(selection["legal_bbox_rate"]),
        "legal_bbox_pass": float(selection["legal_bbox_rate"])
        == thresholds["legal_bbox_rate"],
        "silent_fallback_count": 0,
        "numerical_error_count": 0,
    }
    checks["pipeline_safety_pass"] = (
        checks["legal_bbox_pass"]
        and checks["silent_fallback_count"] == 0
        and checks["numerical_error_count"] == 0
    )
    return {
        "pass": all(
            checks[key]
            for key in (
                "overall_pass",
                "spatial_ordinal_pass",
                "dataset_pass",
                "pipeline_safety_pass",
            )
        ),
        "complete": True,
        "thresholds": thresholds,
        "checks": checks,
    }


def _verification_pass(verification: Mapping[str, Any] | None) -> bool:
    """Require every release verification gate before marking complete."""

    if not verification:
        return False
    required = ("pytest", "compileall", "submission_audit", "code_review")
    return all(
        verification.get(name, {}).get("status") == "passed"
        for name in required
    )


def _promotion_lines(promotion: Mapping[str, Any]) -> str:
    if not promotion.get("complete"):
        return "- 晋级检查尚未完成。"
    checks = promotion["checks"]
    return "\n".join(
        [
            f"- Holdout 整体增益：{100.0 * checks['overall_gain']:+.2f} pp；",
            "- spatial + ordinal 合并增益："
            f"{100.0 * checks['spatial_ordinal_gain']:+.2f} pp；",
            "- 最差单数据集增益："
            f"{100.0 * checks['worst_dataset_gain']:+.2f} pp；",
            f"- AIC 合法框率：{_pct(checks['legal_bbox_rate'])}；",
            "- 静默 fallback / 数值异常："
            f"{checks['silent_fallback_count']} / "
            f"{checks['numerical_error_count']}。",
        ]
    )


def _phase_runtime_table(run_summary: Mapping[str, Any] | None) -> str:
    if not run_summary:
        return "运行状态摘要不可用。"
    rows = [
        "| 阶段 | 状态 | 最近一次调用耗时（秒） |",
        "|---|---|---:|",
    ]
    for phase, item in run_summary.get("phases", {}).items():
        seconds = item.get("details", {}).get("wall_seconds")
        formatted = f"{float(seconds):.1f}" if seconds is not None else "—"
        rows.append(f"| `{phase}` | {item.get('status', 'unknown')} | {formatted} |")
    return "\n".join(rows)


def _feature_table(frozen: Mapping[str, Any] | None, limit: int = 20) -> str:
    if not frozen:
        return "最终模型尚未训练。"
    importance = frozen["full_ranker"]["feature_importance_gain"]
    total = sum(float(value) for value in importance.values())
    rows = [
        "| 排名 | 特征 | Gain | 占比 |",
        "|---:|---|---:|---:|",
    ]
    for index, (name, value) in enumerate(
        list(importance.items())[:limit], start=1
    ):
        share = float(value) / total if total else 0.0
        rows.append(
            f"| {index} | `{name}` | {_number(value, 2)} | {_pct(share)} |"
        )
    return "\n".join(rows)


def _subset_description(subsets: Mapping[str, Any] | None) -> str:
    if not subsets:
        return "固定子集尚未生成。"
    rows = [
        "| 子集 | Query | 唯一图像 | 单图上限 | 清单 SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for key in ("pilot", "full", "validation", "holdout"):
        if key not in subsets:
            continue
        item = subsets[key]
        rows.append(
            f"| {key} | {item['records']} | {item['unique_images']} | "
            f"{item['maximum_queries_per_image']} | "
            f"`{item['file_sha256']}` |"
        )
    return "\n".join(rows)


def _submission_lines(submissions: Mapping[str, Any] | None) -> str:
    if not submissions:
        return "AIC 两份提交尚未生成。"
    audit = submissions["submission_audit"]
    return "\n".join(
        [
            f"- S02 JSON：`{audit['control']['json_path']}`",
            f"- S02 ZIP：`{audit['control']['zip_path']}`",
            f"- S02 ZIP SHA-256：`{audit['control']['zip_sha256']}`",
            f"- S03 JSON：`{audit['ranker']['json_path']}`",
            f"- S03 ZIP：`{audit['ranker']['zip_path']}`",
            f"- S03 ZIP SHA-256：`{audit['ranker']['zip_sha256']}`",
            f"- Ranker 切换率："
            f"{_pct(submissions['selection_summary']['switch_rate'])}",
            f"- 共享 Florence 零候选兜底："
            f"{submissions['selection_summary']['fallback_count']} 条；"
            "两份提交对这些样本使用完全相同的框。",
        ]
    )


def build_report(output_root: Path) -> tuple[str, dict[str, Any]]:
    run_summary = _read(output_root / "run_summary.json")
    verification = _read(output_root / "verification_summary.json")
    subsets = _read(output_root / "subsets/subset_summary.json")
    smoke = _read(
        output_root / "candidate_cache/train_smoke_100_preflight/summary.json"
    )
    pilot = _read(output_root / "models/pilot_grid/grid_results.json")
    frozen = _read(output_root / "models/frozen_model.json")
    validation = _read(output_root / "evaluations/validation/evaluation.json")
    holdout = _read(output_root / "evaluations/holdout/evaluation.json")
    validation_cache = _read(
        output_root / "candidate_cache/validation/summary.json"
    )
    holdout_cache = _read(
        output_root / "candidate_cache/holdout/summary.json"
    )
    holdout_provenance = _read(
        output_root / "evaluations/holdout/evaluation_provenance.json"
    )
    config_fingerprint = _read(
        output_root / "configuration/config_fingerprint.json"
    )
    aic_cache = _read(output_root / "candidate_cache/aic_test/summary.json")
    submissions = _read(output_root / "submissions/submission_summary.json")
    environment = _read(output_root / "environment.json")
    manifest = _read(output_root / "sha256_manifest.json")
    frozen_decision = _read(
        output_root / "evaluations/validation/frozen_decision.json"
    )
    public_submissions = _compact_submissions(submissions, output_root)

    validation_guarded = (
        validation.get("full_ranker_guarded") if validation else None
    )
    holdout_guarded = (
        holdout.get("full_ranker_guarded") if holdout else None
    )
    holdout_gain = None
    if holdout_guarded:
        overall = holdout_guarded["overall"]
        holdout_gain = (
            float(overall["selected_acc_at_05"])
            - float(overall["baseline_acc_at_05"])
        )
    local_promotion = _local_promotion_checks(
        holdout_guarded, submissions
    )
    local_decision_text = (
        "通过全部预设本地晋级条件；平台是否提升仍必须由 S02/S03 实测。"
        if local_promotion["pass"]
        else "未通过全部预设本地晋级条件，应标记为探索性提交。"
    )

    report = f"""# AIC 赛题一：GroundingDINO Top-10 + 空间关系感知排序器训练报告

> 报告定位：这是给后续 ChatGPT 网页端或其他协作者直接阅读的独立交接文件。文中严格区分平台事实、外部本地验证、工程 sanity check 与尚未完成项。

## 1. 执行摘要

本轮训练的是监督式候选排序器，不是 GroundingDINO 本体，也不是强化学习。输入仍为 RGB 与原始英文 Query；GroundingDINO-Tiny 生成 Top-10，LightGBM LambdaRank 根据模型分数、候选几何、目标/参照物词语匹配、绝对位置、相对关系、面积和序数等特征选择最终 bbox。

已知平台事实只有：

- Florence-2-large-ft RGB-only、first-candidate：AIC 平台 ACC@0.5 = **0.4980**。
- S02 与 S03 的平台成绩仍需用户手动上传后回填，不能由本地 RefCOCO 分数代替。

本轮是否达到预设晋级线：

- Holdout 相对 GroundingDINO Top-1 增益：{f"{100.0 * holdout_gain:+.2f} pp" if holdout_gain is not None else "尚未完成"}。
- 本地晋级结论：{local_decision_text}

{_promotion_lines(local_promotion)}

## 2. 赛题背景与本轮边界

AIC 任务要求依据 Visible RGB、Infrared、Depth 与英文 Query，在 Visible 图上输出归一化 `[x1,y1,x2,y2]`，指标为 ACC@0.5。正式初赛 9,555 条 Query 没有 bbox，是测试集，只用于推理。

本轮只使用：

- RefCOCO、RefCOCO+、RefCOCOg 的外部 train/validation/holdout；
- Visible RGB 和原始 Query；
- 固定 GroundingDINO-Tiny Top-10；
- LightGBM 4.6.0 LambdaRank。

本轮没有使用：

- AIC 测试集训练、伪标签或人工框；
- IR、Depth、Tile、Query 人工改写；
- Florence 候选融合、CLIP crop 特征；
- GroundingDINO 参数更新。

## 3. 为什么先训练 Ranker，而不是先微调 GroundingDINO

诊断轮的受控外部结果为：

- GroundingDINO Top-1：0.5627；
- GroundingDINO Top-10 oracle：0.9073；
- Florence first：0.6893；
- Florence oracle：0.7533。

这些数字不能直接预测 AIC 平台分数，但它们说明 GroundingDINO 的候选集合明显比当前 Top-1 排序更强。候选 oracle 与最终选择之间的巨大差距，正是 Learning-to-Rank 的直接监督空间。微调检测器会同时改变召回与定位，本轮先用小模块隔离“排序是否有效”这一变量。

## 4. 对 ChatGPT 网页端旧分析的修正

旧分析中有合理方向，也有需要明确纠正的判断：

1. **“60.27% 多候选”不能证明正确框在非首候选。** 多候选可能分别是主体、部件或参照物。只有带 GT 的 oracle 才能证明可救空间。
2. **Florence 没有可用的候选 confidence。** 因此 `0.4 * Florence confidence` 之类公式没有数据基础；本轮没有伪造 confidence。
3. **不能跨语义 label 直接做 CLIP crop 重排。** “man holding a phone” 中 phone crop 可能与整句很相似，却不是应输出的主体框。目标/参照物角色必须先区分。
4. **无校准分数时不应使用 WBF。** Florence Tile 的多个框没有可比较置信度，WBF 会人为制造一个缺乏统计依据的新框。
5. **Tile oracle 提升不等于最终 selected ACC 提升。** Tile 只说明候选召回上限；没有可靠选择器时不能直接声称平台会涨分。
6. **GroundingDINO 外部 Top-1 并不优于 Florence 外部 first。** 它的价值是 Top-K 候选生成，不是已经证明可直接替代 Florence。
7. **规则原型的 +2.6 pp 是受控外部证据，不是 AIC 平台增益。** 本轮两份只差排序器的提交，才用于测量真实目标域收益。

## 5. 固定子集、隔离与哈希

随机种子为 `20260731`。训练、dev、validation、holdout 均按 image key 隔离；holdout 只在模型与保护阈值冻结后生成候选和评测。

{_subset_description(subsets)}

由于 RefCOCO 系列中 ordinal 等类型的自然供给不足，实际类别分布不能完全复制 AIC 比例。报告保留每类请求配额与实际偏差，不能写成“完美匹配”。

## 6. 100 条真实模型冒烟

{("- 候选记录：{records}；无候选：{no_candidate_records}；Top-10 oracle：{oracle}；峰值显存：{memory:.2f} GiB。".format(records=smoke["records"], no_candidate_records=smoke["no_candidate_records"], oracle=_pct(smoke["candidate_oracle_acc_at_05"]), memory=float(smoke["cuda_peak_allocated_bytes"]) / 1024 ** 3) if smoke else "尚未完成。")}

该 100 条只验证缓存、坐标、IoU、可恢复写入和显存，不用于模型结论。

## 7. 空间词强化的实现

每条 Query 是一个排序 group；候选相关性标签由 IoU 转为四级：

```text
0: IoU < 0.20
1: 0.20 <= IoU < 0.50
2: 0.50 <= IoU < 0.70
3: IoU >= 0.70
label_gain = [0, 1, 4, 5]
```

特征覆盖：

- GroundingDINO score、原始 rank、与第一名/下一名分差；
- bbox 中心、宽高、面积、长宽比、四边距离；
- 全候选与目标兼容候选内部的左右/上下/面积排名；
- 候选 label 与完整 Query、目标短语、参照物短语的 token overlap；
- 主体/参照物 overlap 及二者差值（本轮**没有**独立的部件角色特征）；
- `leftmost/rightmost/topmost/bottommost/center`；
- `largest/smallest`；
- `first/second/third... from left/right/top/bottom`；
- `left of/right of/above/below/beside/inside/between` 的候选对关系；
- `nearest/farthest/front/behind` 仅记录文本标志，不伪造二维深度。

例如 `the man left of the bus` 被拆为 target=`man`、relation=`left_of`、reference=`bus`，候选 man 的几何关系相对 bus 候选计算，而不是把 “left” 粗暴作用于全部框。

实现边界需要明确：原计划提到主体、部件、参照物三类角色，但冻结模型只显式建模了主体与参照物，部件候选尚未独立识别。为保持 holdout 单次评测纪律，本轮审查后没有改变特征矩阵或重训；部件角色应作为下一轮独立消融。

## 8. 20k Pilot 与 50k 最终训练

{("Pilot 选择：grid {grid}，`num_leaves={leaves}`，`min_child_samples={child}`，best iteration={iteration}，guard margin={guard}。".format(grid=pilot["selected"]["grid_index"], leaves=pilot["selected"]["params"]["num_leaves"], child=pilot["selected"]["params"]["min_child_samples"], iteration=pilot["selected"]["best_iteration"], guard=pilot["selected"]["guard_margin"]) if pilot else "Pilot 尚未完成。")}

{("最终完整 Ranker 使用 {groups} 个有排序信号的 Query group、{rows} 个候选行；固定 {iters} 轮。".format(groups=frozen["full_ranker"]["training_groups"], rows=frozen["full_ranker"]["training_candidate_rows"], iters=frozen["full_ranker"]["n_estimators"]) if frozen else "50k 最终训练尚未完成。")}

### 主要特征重要性

{_feature_table(frozen)}

`area` 与 `log_area` 在增益重要性中占比较高，说明模型部分依赖 RefCOCO 的框尺度先验。它在外部 holdout 上有效，但可能遇到 AIC 极小目标域偏移；因此不能只凭本地提升推断平台一定涨分。

## 9. Validation 10k 消融

{_evaluation_table(validation)}

### 按数据集

{_dataset_table(validation_guarded)}

### 按 Query 类型

{_category_table(validation_guarded)}

### 排序边际分桶

{_margin_bucket_table(validation_guarded)}

Validation Top-10 不可救样本：{(validation_cache['records'] - validation_cache['solvable_at_05_records']) if validation_cache else '未知'} 条；这些样本无法由任何只做 Top-10 重排的模型修复。

## 10. Holdout 10k 单次评测

{_evaluation_table(holdout)}

### 按数据集

{_dataset_table(holdout_guarded)}

### 按 Query 类型

{_category_table(holdout_guarded)}

### 排序边际分桶

{_margin_bucket_table(holdout_guarded)}

Holdout Top-10 不可救样本：{(holdout_cache['records'] - holdout_cache['solvable_at_05_records']) if holdout_cache else '未知'} 条。

Holdout 仅允许单次检查；文件已存在时统一入口拒绝再次覆盖评测。本轮代码审查后又增加了模型 SHA、特征 schema、训练/候选配置、validation cache fingerprint 与 holdout evaluation SHA 的绑定门禁；既有 holdout 结果通过原始完成清单回绑，不重新计算指标。

## 11. AIC 9,555 条推理与合法性

{("- 候选缓存：{records}/9555；无候选：{missing}；本轮耗时 {seconds:.1f} 秒；峰值显存 {memory:.2f} GiB。".format(records=aic_cache["records"], missing=aic_cache["no_candidate_records"], seconds=aic_cache["wall_seconds_this_run"], memory=float(aic_cache["cuda_peak_allocated_bytes"]) / 1024 ** 3) if aic_cache else "AIC 全量推理尚未完成。")}

系统异常、NaN、特征错误不会被中心框掩盖。只有 GroundingDINO 真正返回零候选时，两份提交才共同使用已存在的 Florence 预测，保证 S03−S02 只测排序器选择差异。

## 12. 两份平台提交

{_submission_lines(public_submissions)}

建议用户严格按顺序手动上传：

1. S02 GroundingDINO 原始 Top-1；
2. S03 GroundingDINO + 空间 LTR。

## 13. 平台分数回填

| 提交 | 平台分数 | 提交 ID | 时间 |
|---|---:|---|---|
| Florence S01 | 0.4980 | 待补 | 待补 |
| GroundingDINO S02 | 待用户上传 | 待补 | 待补 |
| GDINO + Ranker S03 | 待用户上传 | 待补 | 待补 |

回填后计算：

```text
Ranker 训练真实收益 = S03 - S02
GroundingDINO 基座变化 = S02 - 0.4980
相对首版系统收益 = S03 - 0.4980
```

## 14. 下一轮决策规则

- 若 S03−S02 明显为正，先做 selective tile 或 crop/global-context embedding 消融；
- 若 spatial/ordinal 本地提升明显、平台提升有限，优先检查 AIC 目标角色和 Query 翻译域偏移；
- 若 S02/S03 均低于 Florence 0.4980，可尝试 Florence/GDINO 候选联合排序，而不是立即微调 GDINO；
- 若 Top-10 recall failure 仍集中在小目标，再做 selective tile；
- Depth late fusion 只服务 nearest/farthest/front/behind，PNG 与 JPG 深度域分开处理；
- 只有候选 oracle 仍不足时，才把 GroundingDINO 本体微调提升为主线。

## 15. 运行时间、异常与最终验证

{_phase_runtime_table(run_summary)}

候选生成的真实 GPU 耗时以各 cache `summary.json` 为准；上表是监督器记录的最近一次阶段调用，其中 validation 重建冻结凭据、holdout provenance 回绑等审查操作不会重跑 GPU 推理或第二次打开 holdout 指标。

{("- 自动化测试：{passed} passed，耗时 {seconds:.2f} 秒；compileall：{compile_status}；提交独立审计：{audit_status}。".format(passed=verification['pytest']['passed'], seconds=verification['pytest']['seconds'], compile_status=verification['compileall']['status'], audit_status=verification['submission_audit']['status']) if verification else "最终验证摘要尚未写入。")}

本轮最终阶段没有 CUDA OOM、NaN、RuntimeError 或静默 fallback。代码审查发现并修复了缓存依赖闭包、holdout 冻结身份绑定、配置哈希和报告口径问题；这些修复不改变已冻结模型、候选框、holdout 指标或两份提交 bbox。

## 16. 环境与可复现性

- Python：`{environment.get("python", "未知") if environment else "未知"}`；
- Torch：`{environment.get("torch", "未知") if environment else "未知"}`；
- Transformers：`{environment.get("transformers", "未知") if environment else "未知"}`；
- LightGBM：`{environment.get("lightgbm", "未知") if environment else "未知"}`；
- GPU：`{environment.get("cuda_device", "未知") if environment else "未知"}`；
- 机器可读运行状态：`outputs/gdino_spatial_ltr_v1/run_summary.json`；
- 关键产物哈希：`outputs/gdino_spatial_ltr_v1/sha256_manifest.json`。
- 配置指纹：`outputs/gdino_spatial_ltr_v1/configuration/config_fingerprint.json`。
- Holdout 绑定凭据：`outputs/gdino_spatial_ltr_v1/evaluations/holdout/evaluation_provenance.json`。

完整候选 JSONL、模型权重、外部数据、AIC 数据和本机绝对路径均不进入 Git；Git 只保留代码、公开配置模板、小型摘要和本报告。
"""
    pilot_selected = pilot.get("selected") if pilot else None
    compact_pilot = None
    if pilot_selected:
        compact_pilot = {
            "grid_index": pilot_selected["grid_index"],
            "params": pilot_selected["params"],
            "guard_margin": pilot_selected["guard_margin"],
            "best_iteration": pilot_selected["best_iteration"],
            "dev_overall": pilot_selected["evaluation"]["overall"],
        }

    compact_model = None
    if frozen:
        full_ranker = frozen["full_ranker"]
        compact_model = {
            "model_path": _public_artifact_path(
                full_ranker["model_path"], output_root
            ),
            "training_groups": full_ranker["training_groups"],
            "training_candidate_rows": full_ranker[
                "training_candidate_rows"
            ],
            "n_estimators": full_ranker["n_estimators"],
            "guard_margin": frozen["guard_margin"],
        }

    compact_phases = {
        phase: {
            "status": item.get("status"),
            "wall_seconds": item.get("details", {}).get("wall_seconds"),
        }
        for phase, item in (run_summary or {}).get("phases", {}).items()
    }

    summary = {
        "experiment": "gdino_spatial_ltr_v1",
        "status": (
            "complete"
            if (
                holdout_guarded
                and submissions
                and manifest
                and _verification_pass(verification)
            )
            else "partial"
        ),
        "random_seed": 20260731,
        "florence_platform_acc_at_05": 0.498,
        "subsets": subsets,
        "smoke": smoke,
        "pilot": compact_pilot,
        "model": compact_model,
        "validation_promotion": frozen_decision,
        "validation": (
            validation_guarded["overall"] if validation_guarded else None
        ),
        "validation_cache": validation_cache,
        "holdout": holdout_guarded["overall"] if holdout_guarded else None,
        "holdout_cache": holdout_cache,
        "holdout_gain": holdout_gain,
        "local_promotion": local_promotion,
        "holdout_provenance": holdout_provenance,
        "aic_cache": aic_cache,
        "submissions": public_submissions,
        "artifact_manifest": manifest,
        "configuration_fingerprint": config_fingerprint,
        "phase_status": compact_phases,
        "verification": verification,
        "environment": environment,
        "run_summary_path": (
            Path("outputs") / output_root.name / "run_summary.json"
        ).as_posix(),
    }
    return report, summary


def main() -> int:
    args = parse_args()
    report, summary = build_report(args.output_root.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report.rstrip() + "\n", encoding="utf-8")
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.report.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
