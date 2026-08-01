from __future__ import annotations

import math
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .conservative_selector import (
    ConservativeSelectionResult,
    ConservativeSwitchPolicy,
    select_conservative_predictions,
)
from .external_data import sha256_file
from .submission import build_submission


def _write_deterministic_zip(json_path: Path, zip_path: Path) -> None:
    payload = json_path.read_bytes()
    entry = zipfile.ZipInfo(
        filename="predictions_submission.json",
        date_time=(1980, 1, 1, 0, 0, 0),
    )
    entry.compress_type = zipfile.ZIP_DEFLATED
    entry.create_system = 3
    entry.external_attr = 0o600 << 16
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(entry, payload)


def write_conservative_submission(
    *,
    original_records: Mapping[str, Mapping[str, Any]],
    selection: ConservativeSelectionResult,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Build a deterministic, one-entry S04 submission archive."""

    destination = Path(output_dir)
    json_path = destination / "predictions_submission.json"
    zip_path = destination / "predictions_submission.zip"
    build_submission(
        original_records=original_records,
        predictions=selection.conservative_predictions,
        output_json=json_path,
        output_zip=zip_path,
    )
    _write_deterministic_zip(json_path, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        bad_member = archive.testzip()
    if names != ["predictions_submission.json"] or bad_member is not None:
        raise ValueError("conservative submission ZIP failed integrity audit")
    return {
        "json_path": str(json_path),
        "json_sha256": sha256_file(json_path),
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "zip_entries": names,
        "zip_bad_member": bad_member,
    }


def _record_ious(record: Mapping[str, Any]) -> list[float]:
    candidates = list(record.get("candidates", []))
    source = record.get("candidate_ious")
    if source is None:
        source = [candidate.get("iou") for candidate in candidates]
    try:
        values = [float(value) for value in source]
    except (TypeError, ValueError) as error:
        raise ValueError("candidate IoUs are missing or invalid") from error
    if len(values) != len(candidates) or not all(
        math.isfinite(value) for value in values
    ):
        raise ValueError("candidate IoUs do not match the candidate list")
    return values


def _evaluation_bucket(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    queries = len(outcomes)
    if not queries:
        return {
            "queries": 0,
            "baseline_acc_at_05": 0.0,
            "selected_acc_at_05": 0.0,
            "gain_pp": 0.0,
            "switch_count": 0,
            "rescues": 0,
            "harms": 0,
        }
    baseline = sum(float(row["baseline_iou"]) >= 0.5 for row in outcomes)
    selected = sum(float(row["selected_iou"]) >= 0.5 for row in outcomes)
    return {
        "queries": queries,
        "baseline_acc_at_05": baseline / queries,
        "selected_acc_at_05": selected / queries,
        "gain_pp": (selected - baseline) * 100.0 / queries,
        "switch_count": sum(bool(row["switched"]) for row in outcomes),
        "rescues": sum(bool(row["rescued"]) for row in outcomes),
        "harms": sum(bool(row["harmed"]) for row in outcomes),
    }


def evaluate_conservative_policy(
    *,
    records: Sequence[Mapping[str, Any]],
    ranker: Any,
    policy: ConservativeSwitchPolicy,
) -> dict[str, Any]:
    """Evaluate S04 on labelled cache records without inventing fallbacks."""

    eligible = [record for record in records if record.get("candidates")]
    selection = select_conservative_predictions(
        records=eligible,
        ranker=ranker,
        policy=policy,
        fallback_predictions={},
    )
    outcomes: list[dict[str, Any]] = []
    for record, debug in zip(eligible, selection.debug_records):
        ious = _record_ious(record)
        selected_index = int(debug["selected_index"])
        baseline_iou = ious[0]
        selected_iou = ious[selected_index]
        outcomes.append(
            {
                "query_id": str(record.get("query_id", "")),
                "dataset": str(record.get("dataset", "unknown")),
                "query_category": str(
                    record.get("query_category", "other")
                ),
                "baseline_iou": baseline_iou,
                "selected_iou": selected_iou,
                "selected_index": selected_index,
                "switched": selected_index != 0,
                "rescued": baseline_iou < 0.5 <= selected_iou,
                "harmed": baseline_iou >= 0.5 > selected_iou,
            }
        )
    overall = _evaluation_bucket(outcomes)
    rescues = int(overall["rescues"])
    harms = int(overall["harms"])
    categories = sorted({str(row["query_category"]) for row in outcomes})
    grouped = {
        category: _evaluation_bucket(
            [row for row in outcomes if row["query_category"] == category]
        )
        for category in categories
    }
    return {
        "cache_records": len(records),
        "evaluated_queries": len(eligible),
        "no_candidate_records": len(records) - len(eligible),
        **overall,
        "baseline_mean_iou": (
            sum(float(row["baseline_iou"]) for row in outcomes)
            / len(outcomes)
            if outcomes
            else 0.0
        ),
        "selected_mean_iou": (
            sum(float(row["selected_iou"]) for row in outcomes)
            / len(outcomes)
            if outcomes
            else 0.0
        ),
        "switch_rate": (
            int(overall["switch_count"]) / len(outcomes)
            if outcomes
            else 0.0
        ),
        "rescue_harm_ratio": rescues / harms if harms else None,
        "depth_switch_count": sum(
            row["switched"] and row["query_category"] == "depth"
            for row in outcomes
        ),
        "selection_safety": {
            key: selection.summary[key]
            for key in (
                "cross_canonical_label_switch_count",
                "reference_dominant_switch_count",
                "max_switched_area_ratio",
            )
        },
        "by_query_category": grouped,
        "policy": asdict(policy),
    }
