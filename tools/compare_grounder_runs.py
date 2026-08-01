from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from aic_baseline.bbox import intersection_over_union
from aic_baseline.diagnostics import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two labeled grounder runs.")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compare_runs(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left = {str(row["query_id"]): row for row in left_records}
    right = {str(row["query_id"]): row for row in right_records}
    if set(left) != set(right):
        raise ValueError("run Query ID sets differ")
    rows: list[dict[str, Any]] = []
    for query_id in sorted(left):
        left_row = left[query_id]
        right_row = right[query_id]
        left_selected = bool(left_row["selected_acc_at_05"])
        right_selected = bool(right_row["selected_acc_at_05"])
        left_oracle = bool(left_row["oracle_acc_at_05"])
        right_oracle = bool(right_row["oracle_acc_at_05"])
        left_box = left_row.get("selected_bbox")
        right_box = right_row.get("selected_bbox")
        agreement = (
            intersection_over_union(left_box, right_box)
            if left_box is not None and right_box is not None
            else 0.0
        )
        rows.append(
            {
                "query_id": query_id,
                "left_selected": left_selected,
                "right_selected": right_selected,
                "left_oracle": left_oracle,
                "right_oracle": right_oracle,
                "selected_bbox_iou": agreement,
            }
        )
    count = len(rows)
    if not count:
        raise ValueError("runs are empty")
    return {
        "status": "ok",
        "record_count": count,
        "left_name": left_name,
        "right_name": right_name,
        "left_top1_acc_at_05": statistics.fmean(
            float(row["left_selected"]) for row in rows
        ),
        "right_top1_acc_at_05": statistics.fmean(
            float(row["right_selected"]) for row in rows
        ),
        "left_oracle_acc_at_05": statistics.fmean(
            float(row["left_oracle"]) for row in rows
        ),
        "right_oracle_acc_at_05": statistics.fmean(
            float(row["right_oracle"]) for row in rows
        ),
        "candidate_union_oracle_acc_at_05": statistics.fmean(
            float(row["left_oracle"] or row["right_oracle"]) for row in rows
        ),
        "left_only_top1_correct": sum(
            row["left_selected"] and not row["right_selected"] for row in rows
        ),
        "right_only_top1_correct": sum(
            row["right_selected"] and not row["left_selected"] for row in rows
        ),
        "left_only_oracle_recall": sum(
            row["left_oracle"] and not row["right_oracle"] for row in rows
        ),
        "right_only_oracle_recall": sum(
            row["right_oracle"] and not row["left_oracle"] for row in rows
        ),
        "selected_bbox_iou_mean": statistics.fmean(
            float(row["selected_bbox_iou"]) for row in rows
        ),
        "selected_bbox_agreement_at_05": statistics.fmean(
            float(row["selected_bbox_iou"] >= 0.5) for row in rows
        ),
    }


def main() -> int:
    args = parse_args()
    comparison = compare_runs(
        read_jsonl(args.left),
        read_jsonl(args.right),
        left_name=args.left_name,
        right_name=args.right_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
