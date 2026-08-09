from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_baseline.aic_full_detection import (  # noqa: E402
    finalize_aic_full_detection,
    sha256_file,
)
from aic_baseline.bbox import validate_normalized_bbox  # noqa: E402
from aic_baseline.data import AICDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cascade AIC submission by replacing selected fallback rows "
            "from a primary prediction file with rows from a stable fallback model."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--fallback-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fallback-source",
        default="qwen3_vl_8b_instruct_fallback",
        help="Source label written for rows replaced from the fallback model.",
    )
    parser.add_argument(
        "--replace-source",
        action="append",
        default=["qwen3vl_fallback_center"],
        help="Candidate source value in the primary predictions that should be replaced.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id in rows:
                raise ValueError(f"duplicate query_id in {path}:{line_number}: {query_id}")
            validate_normalized_bbox(row["selected_bbox"])
            rows[query_id] = row
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _selected_source(row: dict[str, Any]) -> str:
    selected_index = int(row.get("selected_index", 0))
    candidates = row.get("candidates") or []
    if candidates and 0 <= selected_index < len(candidates):
        return str(candidates[selected_index].get("source", ""))
    return ""


def main() -> int:
    args = parse_args()
    dataset = AICDataset(dataset_root=args.dataset_root, queries_path=args.queries)
    expected_ids = [dataset[index].query_id for index in range(len(dataset))]
    expected_set = set(expected_ids)

    primary = _read_jsonl(args.primary_predictions)
    fallback = _read_jsonl(args.fallback_predictions)
    if set(primary) != expected_set:
        missing = sorted(expected_set - set(primary))
        extra = sorted(set(primary) - expected_set)
        raise ValueError(f"primary IDs mismatch; missing={missing[:5]}, extra={extra[:5]}")
    if set(fallback) != expected_set:
        missing = sorted(expected_set - set(fallback))
        extra = sorted(set(fallback) - expected_set)
        raise ValueError(f"fallback IDs mismatch; missing={missing[:5]}, extra={extra[:5]}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    replace_sources = set(args.replace_source)
    replacement_ids: list[str] = []
    source_counts: dict[str, int] = {}

    with predictions_path.open("w", encoding="utf-8", newline="\n") as sink:
        for query_id in expected_ids:
            row = dict(primary[query_id])
            source = _selected_source(row)
            if source in replace_sources:
                fb = fallback[query_id]
                row["selected_bbox"] = validate_normalized_bbox(fb["selected_bbox"])
                row["selected_score"] = fb.get("selected_score")
                row["selected_label"] = fb.get("selected_label", row["query"])
                row["selected_index"] = 0
                row["selection_policy"] = (
                    "primary_model_unless_parser_fallback_then_stable_model"
                )
                row["candidate_count"] = 1
                row["candidates"] = [
                    {
                        "bbox": row["selected_bbox"],
                        "label": row["selected_label"],
                        "score": row["selected_score"],
                        "source": args.fallback_source,
                    }
                ]
                row["cascade_replacement"] = {
                    "primary_source": source,
                    "fallback_prediction_file": str(args.fallback_predictions),
                    "fallback_source": args.fallback_source,
                }
                replacement_ids.append(query_id)
                source = args.fallback_source
            source_counts[source] = source_counts.get(source, 0) + 1
            sink.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "schema": "aic-cascade-submission-v1",
        "query_count": len(expected_ids),
        "primary_predictions": str(args.primary_predictions),
        "primary_sha256": sha256_file(args.primary_predictions),
        "fallback_predictions": str(args.fallback_predictions),
        "fallback_sha256": sha256_file(args.fallback_predictions),
        "output_predictions": str(predictions_path),
        "output_predictions_sha256": sha256_file(predictions_path),
        "replace_sources": sorted(replace_sources),
        "fallback_source": args.fallback_source,
        "replacement_count": len(replacement_ids),
        "replacement_ids": replacement_ids,
        "source_counts": source_counts,
        "training_on_aic": False,
    }
    _write_json(output / "cascade_summary.json", summary)
    finalized = finalize_aic_full_detection(
        dataset=dataset,
        detection_dir=output,
        submission_dir=output / "submission",
    )
    _write_json(
        output / "cascade_final_audit.json",
        {
            "cascade_summary": summary,
            "submission_zip": str(finalized.zip_path),
            "submission_audit": finalized.audit,
        },
    )
    print(
        json.dumps(
            {
                "replacement_count": len(replacement_ids),
                "source_counts": source_counts,
                "submission_zip": str(finalized.zip_path),
                "submission_audit": finalized.audit,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
