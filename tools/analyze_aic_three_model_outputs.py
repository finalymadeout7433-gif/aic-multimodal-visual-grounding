from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from aic_baseline.bbox import intersection_over_union, validate_normalized_bbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize three unlabeled AIC model outputs without accuracy claims."
    )
    parser.add_argument("--ape", type=Path, required=True)
    parser.add_argument("--mm", type=Path, required=True)
    parser.add_argument("--llmdet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id in rows:
                raise ValueError(f"duplicate query_id at {path}:{line_number}")
            validate_normalized_bbox(row["selected_bbox"])
            rows[query_id] = row
    return rows


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        label: float(np.percentile(array, percentile))
        for label, percentile in (
            ("min", 0),
            ("p01", 1),
            ("p05", 5),
            ("p25", 25),
            ("median", 50),
            ("p75", 75),
            ("p95", 95),
            ("p99", 99),
            ("max", 100),
        )
    }


def _model_summary(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    areas: list[float] = []
    scores: list[float] = []
    for row in rows.values():
        x1, y1, x2, y2 = row["selected_bbox"]
        areas.append((x2 - x1) * (y2 - y1))
        if row["selected_score"] is not None:
            scores.append(float(row["selected_score"]))
    return {
        "query_count": len(rows),
        "predicted_bbox_area_quantiles": _quantiles(areas),
        "selected_score_quantiles": _quantiles(scores) if scores else None,
        "predicted_area_lt_0_1_percent": sum(area < 0.001 for area in areas),
        "predicted_area_lt_1_percent": sum(area < 0.01 for area in areas),
        "predicted_area_ge_25_percent": sum(area >= 0.25 for area in areas),
        "empty_selected_label_count": sum(
            not str(row["selected_label"]).strip() for row in rows.values()
        ),
    }


def main() -> int:
    args = parse_args()
    models = {
        "APE-Ti": _load(args.ape),
        "MM-Grounding-DINO-T": _load(args.mm),
        "LLMDet-Swin-T": _load(args.llmdet),
    }
    id_sets = {name: set(rows) for name, rows in models.items()}
    if len({frozenset(ids) for ids in id_sets.values()}) != 1:
        raise ValueError("the three prediction files do not share identical query IDs")

    model_summaries = {
        name: _model_summary(rows) for name, rows in models.items()
    }
    pairwise: dict[str, Any] = {}
    query_ids = list(next(iter(models.values())))
    for (first_name, first), (second_name, second) in combinations(models.items(), 2):
        values = [
            intersection_over_union(
                first[query_id]["selected_bbox"],
                second[query_id]["selected_bbox"],
            )
            for query_id in query_ids
        ]
        pairwise[f"{first_name}__vs__{second_name}"] = {
            "iou_quantiles": _quantiles(values),
            "agree_iou_ge_0_3": sum(value >= 0.3 for value in values),
            "agree_iou_ge_0_5": sum(value >= 0.5 for value in values),
            "agree_iou_ge_0_7": sum(value >= 0.7 for value in values),
        }
    result = {
        "scope": "unlabeled_AIC_model_behavior_only",
        "accuracy_available": False,
        "model_summaries": model_summaries,
        "pairwise_prediction_agreement": pairwise,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# AIC 三模型全量检测行为摘要",
        "",
        "> 本报告没有 AIC 真值，只描述预测分布与模型一致性，不能替代平台 ACC。",
        "",
        "## 单模型预测分布",
        "",
        "| 模型 | Query | 预测面积中位数 | 面积 <0.1% | 面积 <1% | 面积 ≥25% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in model_summaries.items():
        lines.append(
            f"| {name} | {summary['query_count']:,} | "
            f"{summary['predicted_bbox_area_quantiles']['median']:.4%} | "
            f"{summary['predicted_area_lt_0_1_percent']:,} | "
            f"{summary['predicted_area_lt_1_percent']:,} | "
            f"{summary['predicted_area_ge_25_percent']:,} |"
        )
    lines.extend(
        [
            "",
            "## 两两预测一致性",
            "",
            "| 模型对 | IoU 中位数 | IoU≥0.3 | IoU≥0.5 | IoU≥0.7 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, summary in pairwise.items():
        lines.append(
            f"| {name.replace('__vs__', ' vs ')} | "
            f"{summary['iou_quantiles']['median']:.4f} | "
            f"{summary['agree_iou_ge_0_3']:,} | "
            f"{summary['agree_iou_ge_0_5']:,} | "
            f"{summary['agree_iou_ge_0_7']:,} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

