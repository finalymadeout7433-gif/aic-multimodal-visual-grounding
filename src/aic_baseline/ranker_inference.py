from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bbox import validate_normalized_bbox
from .external_data import sha256_file, verify_zip_archive
from .ranker_features import FEATURE_NAMES, build_feature_rows
from .submission import build_submission


@dataclass
class RankerSelectionResult:
    top1_predictions: dict[str, list[float]]
    ranker_predictions: dict[str, list[float]]
    debug_records: list[dict[str, Any]]
    summary: dict[str, Any]


def _predict_scores(ranker: Any, features: np.ndarray) -> np.ndarray:
    scores = np.asarray(ranker.predict(features), dtype=np.float64)
    if scores.shape != (features.shape[0],):
        raise ValueError("ranker returned an unexpected score shape")
    if not np.isfinite(scores).all():
        raise FloatingPointError("ranker returned NaN or infinity")
    return scores


def select_aic_predictions(
    *,
    records: Sequence[Mapping[str, Any]],
    ranker: Any,
    guard_margin: float,
    fallback_predictions: Mapping[str, Sequence[float]],
) -> RankerSelectionResult:
    """Select control/ranked boxes from one shared AIC candidate cache."""

    if guard_margin < 0.0:
        raise ValueError("guard_margin must be non-negative")
    top1_predictions: dict[str, list[float]] = {}
    ranker_predictions: dict[str, list[float]] = {}
    debug_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    fallback_count = 0
    switch_count = 0

    for record in records:
        query_id = str(record["query_id"])
        if query_id in seen:
            raise ValueError(f"duplicate AIC candidate record: {query_id}")
        seen.add(query_id)
        candidates = list(record.get("candidates", []))
        if not candidates:
            if query_id not in fallback_predictions:
                raise ValueError(
                    f"zero-candidate query has no shared Florence fallback: "
                    f"{query_id}"
                )
            fallback = validate_normalized_bbox(
                fallback_predictions[query_id]
            )
            top1_predictions[query_id] = fallback
            ranker_predictions[query_id] = fallback
            fallback_count += 1
            debug_records.append(
                {
                    "query_id": query_id,
                    "candidate_count": 0,
                    "control_index": None,
                    "ranker_index": None,
                    "ranker_margin_over_control": None,
                    "switched": False,
                    "fallback_used": True,
                }
            )
            continue

        rows = build_feature_rows(record)
        features = np.asarray(
            [
                [float(row["features"][name]) for name in FEATURE_NAMES]
                for row in rows
            ],
            dtype=np.float32,
        )
        scores = _predict_scores(ranker, features)
        best = int(np.argmax(scores))
        margin = float(scores[best] - scores[0])
        if best != 0 and margin < guard_margin:
            best = 0
        control_bbox = validate_normalized_bbox(candidates[0]["bbox"])
        ranked_bbox = validate_normalized_bbox(candidates[best]["bbox"])
        top1_predictions[query_id] = control_bbox
        ranker_predictions[query_id] = ranked_bbox
        switched = best != 0
        switch_count += int(switched)
        debug_records.append(
            {
                "query_id": query_id,
                "query": str(record.get("query", "")),
                "query_category": str(
                    record.get("query_category", "other")
                ),
                "candidate_count": len(candidates),
                "control_index": 0,
                "ranker_index": best,
                "control_score": float(scores[0]),
                "ranker_score": float(scores[best]),
                "ranker_margin_over_control": margin,
                "switched": switched,
                "fallback_used": False,
            }
        )

    total = len(records)
    return RankerSelectionResult(
        top1_predictions=top1_predictions,
        ranker_predictions=ranker_predictions,
        debug_records=debug_records,
        summary={
            "records": total,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / total if total else 0.0,
            "switch_count": switch_count,
            "switch_rate": switch_count / total if total else 0.0,
            "legal_bbox_rate": (
                len(ranker_predictions) / total if total else 0.0
            ),
            "guard_margin": float(guard_margin),
        },
    )


def load_fallback_predictions(path: Path | str) -> dict[str, list[float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fallback prediction file must contain an object")
    return {
        str(query_id): validate_normalized_bbox(bbox)
        for query_id, bbox in payload.items()
    }


def _submission_audit(json_path: Path, zip_path: Path) -> dict[str, Any]:
    archive = verify_zip_archive(zip_path)
    if archive.bad_member is not None or archive.entry_count != 1:
        raise ValueError("submission ZIP failed integrity audit")
    return {
        "json_path": str(json_path),
        "json_sha256": sha256_file(json_path),
        "zip_path": str(zip_path),
        "zip_sha256": archive.sha256,
        "zip_entry_count": archive.entry_count,
        "zip_bad_member": archive.bad_member,
    }


def write_aic_control_and_ranker_submissions(
    *,
    original_records: Mapping[str, Mapping[str, Any]],
    selection: RankerSelectionResult,
    output_root: Path | str,
) -> dict[str, Any]:
    output = Path(output_root)
    control_dir = output / "S02_gdino_top1_control"
    ranker_dir = output / "S03_gdino_spatial_ltr_v1"
    control_json = control_dir / "predictions_submission.json"
    control_zip = control_dir / "predictions_submission.zip"
    ranker_json = ranker_dir / "predictions_submission.json"
    ranker_zip = ranker_dir / "predictions_submission.zip"
    control = build_submission(
        original_records=original_records,
        predictions=selection.top1_predictions,
        output_json=control_json,
        output_zip=control_zip,
    )
    ranked = build_submission(
        original_records=original_records,
        predictions=selection.ranker_predictions,
        output_json=ranker_json,
        output_zip=ranker_zip,
    )
    if set(control) != set(ranked):
        raise ValueError("control and ranker submissions have different IDs")
    for query_id in control:
        control_non_bbox = {
            key: value
            for key, value in control[query_id].items()
            if key != "bbox"
        }
        ranked_non_bbox = {
            key: value
            for key, value in ranked[query_id].items()
            if key != "bbox"
        }
        if control_non_bbox != ranked_non_bbox:
            raise ValueError(
                f"non-bbox fields differ between submissions: {query_id}"
            )
    return {
        "control": _submission_audit(control_json, control_zip),
        "ranker": _submission_audit(ranker_json, ranker_zip),
    }
