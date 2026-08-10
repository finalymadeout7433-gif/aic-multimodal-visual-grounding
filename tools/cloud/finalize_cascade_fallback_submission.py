#!/usr/bin/env python3
"""Build an AIC submission by replacing primary fallback rows with a baseline.

This is intentionally post-processing only. It does not train on AIC data and
does not infer new boxes. It keeps all successful primary model predictions and
uses an already validated baseline submission only for rows where the primary
runner recorded a fallback event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--primary-output-dir", type=Path, required=True)
    parser.add_argument("--fallback-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform-zip", type=Path, required=True)
    parser.add_argument("--primary-name", default="qwen3_vl_30b_a3b_instruct_fp8")
    parser.add_argument("--fallback-name", default="qwen3_vl_8b_instruct_20260807")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in bbox):
        return False
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def read_primary_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id in rows:
                raise ValueError(f"duplicate primary query_id at line {line_number}: {query_id}")
            if not valid_bbox(row.get("selected_bbox")):
                raise ValueError(f"invalid primary bbox at line {line_number}: {query_id}")
            rows[query_id] = row
    return rows


def read_fallback_ids(path: Path) -> set[str]:
    fallback_ids: set[str] = set()
    if not path.exists():
        return fallback_ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("fallback"):
                fallback_ids.add(str(row["query_id"]))
    return fallback_ids


def main() -> int:
    args = parse_args()
    original = read_json(args.queries)
    fallback_submission = read_json(args.fallback_submission)
    primary_rows = read_primary_rows(args.primary_output_dir / "predictions.jsonl")
    fallback_ids = read_fallback_ids(args.primary_output_dir / "runtime_events.jsonl")

    original_ids = set(original)
    primary_ids = set(primary_rows)
    fallback_source_ids = set(fallback_submission)
    missing_primary = sorted(original_ids - primary_ids)
    missing_fallback_source = sorted(original_ids - fallback_source_ids)
    if missing_primary:
        raise ValueError(f"primary run incomplete: missing {len(missing_primary)} query ids")
    if missing_fallback_source:
        raise ValueError(f"fallback submission incomplete: missing {len(missing_fallback_source)} query ids")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.platform_zip.parent.mkdir(parents=True, exist_ok=True)

    final_submission: dict[str, dict[str, Any]] = {}
    cascade_rows: list[dict[str, Any]] = []
    invalid: list[str] = []
    modified_non_bbox = 0
    replaced = 0

    for query_id in original:
        base_record = dict(original[query_id])
        if query_id in fallback_ids:
            bbox = fallback_submission[query_id].get("bbox")
            source = args.fallback_name
            replaced += 1
        else:
            bbox = primary_rows[query_id]["selected_bbox"]
            source = args.primary_name
        if not valid_bbox(bbox):
            invalid.append(query_id)
        base_record["bbox"] = [float(value) for value in bbox]
        final_submission[query_id] = base_record

        original_without_bbox = dict(original[query_id])
        final_without_bbox = dict(base_record)
        final_without_bbox.pop("bbox", None)
        if original_without_bbox != final_without_bbox:
            modified_non_bbox += 1

        primary_row = dict(primary_rows[query_id])
        primary_row["cascade_source"] = source
        primary_row["cascade_used_fallback"] = query_id in fallback_ids
        primary_row["cascade_bbox"] = base_record["bbox"]
        cascade_rows.append(primary_row)

    if invalid:
        raise ValueError(f"invalid final bbox count={len(invalid)} examples={invalid[:5]}")
    if modified_non_bbox:
        raise ValueError(f"modified non-bbox fields count={modified_non_bbox}")

    submission_json = args.output_dir / "predictions_submission.json"
    cascade_jsonl = args.output_dir / "cascade_predictions.jsonl"
    summary_json = args.output_dir / "cascade_summary.json"
    audit_json = args.output_dir / "submission_audit.json"

    write_json(submission_json, final_submission)
    with cascade_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in cascade_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    with zipfile.ZipFile(args.platform_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_json, arcname="predictions_submission.json")
    local_zip = args.output_dir / args.platform_zip.name
    with zipfile.ZipFile(local_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_json, arcname="predictions_submission.json")

    with zipfile.ZipFile(args.platform_zip, "r") as zf:
        zip_entries = zf.namelist()

    summary = {
        "primary_name": args.primary_name,
        "fallback_name": args.fallback_name,
        "query_count": len(original),
        "primary_rows": len(primary_rows),
        "primary_fallback_count": len(fallback_ids),
        "replaced_with_fallback_count": replaced,
        "invalid_bbox_count": 0,
        "modified_non_bbox_count": 0,
        "source_policy": "use_primary_bbox_unless_primary_runtime_event_fallback_true",
        "primary_predictions_sha256": sha256_file(args.primary_output_dir / "predictions.jsonl"),
        "primary_runtime_events_sha256": sha256_file(args.primary_output_dir / "runtime_events.jsonl"),
        "fallback_submission_sha256": sha256_file(args.fallback_submission),
        "submission_json_sha256": sha256_file(submission_json),
        "platform_zip_sha256": sha256_file(args.platform_zip),
        "local_zip_sha256": sha256_file(local_zip),
        "platform_zip": str(args.platform_zip),
        "local_output_zip": str(local_zip),
    }
    audit = {
        "query_count": len(original),
        "query_ids_exact": set(final_submission) == original_ids,
        "invalid_bbox_count": 0,
        "modified_non_bbox_count": 0,
        "zip_contains_only_predictions_submission_json": zip_entries == ["predictions_submission.json"],
        "zip_entries": zip_entries,
        "predictions_json_sha256": summary["submission_json_sha256"],
        "predictions_zip_sha256": summary["platform_zip_sha256"],
    }
    write_json(summary_json, summary)
    write_json(audit_json, audit)
    print(json.dumps({"summary": summary, "audit": audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
