from __future__ import annotations

import math
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bbox import BBoxError, validate_normalized_bbox
from .external_data import sha256_file
from .ranker_features import (
    FEATURE_NAMES,
    build_feature_rows,
    canonical_candidate_label,
    parse_query_semantics,
)
from .submission import build_submission


_FLOAT_TOLERANCE = 1e-12
_DEPTH_QUERY_RE = re.compile(
    r"\b(?:front|frontmost|behind|rearmost|nearest|closest|farthest|furthest)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ConservativeSwitchPolicy:
    ranker_margin_min: float = 0.5
    gdino_score_drop_max: float = 0.10
    max_area_ratio: float = 1.5
    require_same_canonical_label: bool = True
    reject_reference_dominant: bool = True
    reject_depth_queries: bool = True
    search_best_passing_candidate: bool = True

    def __post_init__(self) -> None:
        numeric = {
            "ranker_margin_min": self.ranker_margin_min,
            "gdino_score_drop_max": self.gdino_score_drop_max,
            "max_area_ratio": self.max_area_ratio,
        }
        for name, value in numeric.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_area_ratio <= 0.0:
            raise ValueError("max_area_ratio must be positive")


@dataclass(frozen=True)
class ConservativeSelectionDecision:
    query_id: str
    control_index: int
    selected_index: int
    selected_bbox: list[float]
    switched: bool
    ranker_margin: float
    gdino_score_drop: float
    control_label: str
    selected_label: str
    area_ratio: float
    target_overlap: float
    reference_overlap: float
    query_category: str
    depth_rejected: bool
    rejection_reasons: tuple[str, ...]
    candidate_count: int
    fallback_used: bool = False

    def to_debug_record(self) -> dict[str, Any]:
        result = asdict(self)
        result["rejection_reasons"] = list(self.rejection_reasons)
        return result


@dataclass
class ConservativeSelectionResult:
    top1_predictions: dict[str, list[float]]
    conservative_predictions: dict[str, list[float]]
    debug_records: list[dict[str, Any]]
    summary: dict[str, Any]


def _bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = validate_normalized_bbox(bbox)
    return (x2 - x1) * (y2 - y1)


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _token_overlap(label: str, phrase: str | None) -> float:
    left = set(canonical_candidate_label(label).split())
    right = set(canonical_candidate_label(phrase or "").split())
    if not left or not right:
        return 0.0
    return len(left & right) / len(right)


def is_depth_query(record: Mapping[str, Any]) -> bool:
    if str(record.get("query_category", "other")).lower() == "depth":
        return True
    query = str(record.get("query", ""))
    semantics = parse_query_semantics(query)
    return semantics.depth_relation is not None or bool(
        _DEPTH_QUERY_RE.search(query)
    )


def _control_decision(
    record: Mapping[str, Any],
    *,
    rejection_reasons: Sequence[str],
    depth_rejected: bool,
) -> ConservativeSelectionDecision:
    candidates = list(record.get("candidates", []))
    if not candidates:
        raise ValueError("a control decision requires at least one candidate")
    bbox = validate_normalized_bbox(candidates[0]["bbox"])
    label = canonical_candidate_label(str(candidates[0].get("label", "")))
    semantics = parse_query_semantics(str(record.get("query", "")))
    return ConservativeSelectionDecision(
        query_id=str(record.get("query_id", "")),
        control_index=0,
        selected_index=0,
        selected_bbox=bbox,
        switched=False,
        ranker_margin=0.0,
        gdino_score_drop=0.0,
        control_label=label,
        selected_label=label,
        area_ratio=1.0,
        target_overlap=_token_overlap(label, semantics.target_phrase),
        reference_overlap=_token_overlap(label, semantics.reference_phrase),
        query_category=str(record.get("query_category", "other")),
        depth_rejected=depth_rejected,
        rejection_reasons=tuple(rejection_reasons),
        candidate_count=len(candidates),
    )


def select_conservative_candidate(
    record: Mapping[str, Any],
    ranker_scores: Sequence[float] | np.ndarray,
    policy: ConservativeSwitchPolicy,
) -> ConservativeSelectionDecision:
    """Select the first ranker-ordered candidate passing every safety gate."""

    candidates = list(record.get("candidates", []))
    if not candidates:
        raise ValueError("candidate selection requires at least one candidate")
    validate_normalized_bbox(candidates[0]["bbox"])
    depth_rejected = policy.reject_depth_queries and is_depth_query(record)
    if depth_rejected:
        return _control_decision(
            record,
            rejection_reasons=("depth_query",),
            depth_rejected=True,
        )

    try:
        scores = np.asarray(ranker_scores, dtype=np.float64)
    except (TypeError, ValueError):
        return _control_decision(
            record,
            rejection_reasons=("invalid_ranker_scores",),
            depth_rejected=False,
        )
    if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
        return _control_decision(
            record,
            rejection_reasons=("invalid_ranker_scores",),
            depth_rejected=False,
        )

    control_score = _finite_number(candidates[0].get("score"))
    control_label = canonical_candidate_label(
        str(candidates[0].get("label", ""))
    )
    control_area = _bbox_area(candidates[0]["bbox"])
    semantics = parse_query_semantics(str(record.get("query", "")))
    order = sorted(range(1, len(candidates)), key=lambda i: (-scores[i], i))
    if not policy.search_best_passing_candidate:
        order = order[:1]
    rejection_reasons: list[str] = []

    for index in order:
        candidate = candidates[index]
        reasons: list[str] = []
        margin = float(scores[index] - scores[0])
        if margin + _FLOAT_TOLERANCE < policy.ranker_margin_min:
            reasons.append("ranker_margin")

        candidate_score = _finite_number(candidate.get("score"))
        if control_score is None or candidate_score is None:
            score_drop = math.inf
            reasons.append("invalid_gdino_score")
        else:
            score_drop = control_score - candidate_score
            if (
                score_drop
                > policy.gdino_score_drop_max + _FLOAT_TOLERANCE
            ):
                reasons.append("gdino_score_drop")

        candidate_label = canonical_candidate_label(
            str(candidate.get("label", ""))
        )
        if policy.require_same_canonical_label and (
            not control_label
            or not candidate_label
            or candidate_label != control_label
        ):
            reasons.append("canonical_label")

        target_overlap = _token_overlap(
            str(candidate.get("label", "")), semantics.target_phrase
        )
        reference_overlap = _token_overlap(
            str(candidate.get("label", "")), semantics.reference_phrase
        )
        if (
            policy.reject_reference_dominant
            and reference_overlap > target_overlap + _FLOAT_TOLERANCE
        ):
            reasons.append("reference_dominant")

        try:
            candidate_bbox = validate_normalized_bbox(candidate["bbox"])
            area_ratio = _bbox_area(candidate_bbox) / control_area
        except (BBoxError, KeyError, TypeError, ValueError, ZeroDivisionError):
            candidate_bbox = None
            area_ratio = math.inf
            reasons.append("invalid_bbox")
        if (
            area_ratio > policy.max_area_ratio + _FLOAT_TOLERANCE
            and "invalid_bbox" not in reasons
        ):
            reasons.append("area_ratio")

        if reasons:
            rejection_reasons.extend(
                f"candidate[{index}]:{reason}" for reason in reasons
            )
            continue

        assert candidate_bbox is not None
        return ConservativeSelectionDecision(
            query_id=str(record.get("query_id", "")),
            control_index=0,
            selected_index=index,
            selected_bbox=candidate_bbox,
            switched=True,
            ranker_margin=margin,
            gdino_score_drop=score_drop,
            control_label=control_label,
            selected_label=candidate_label,
            area_ratio=area_ratio,
            target_overlap=target_overlap,
            reference_overlap=reference_overlap,
            query_category=str(record.get("query_category", "other")),
            depth_rejected=False,
            rejection_reasons=tuple(rejection_reasons),
            candidate_count=len(candidates),
        )

    if not order:
        rejection_reasons.append("no_non_control_candidate")
    return _control_decision(
        record,
        rejection_reasons=rejection_reasons,
        depth_rejected=False,
    )


def _ranker_scores(ranker: Any, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    features = np.asarray(
        [
            [float(row["features"][name]) for name in FEATURE_NAMES]
            for row in rows
        ],
        dtype=np.float32,
    )
    raw_scores = ranker.predict(features)
    try:
        return np.asarray(raw_scores, dtype=np.float64)
    except (TypeError, ValueError):
        return np.full((len(rows),), np.nan, dtype=np.float64)


def select_conservative_predictions(
    *,
    records: Sequence[Mapping[str, Any]],
    ranker: Any,
    policy: ConservativeSwitchPolicy,
    fallback_predictions: Mapping[str, Sequence[float]],
) -> ConservativeSelectionResult:
    """Apply the frozen conservative policy to a shared AIC cache."""

    top1_predictions: dict[str, list[float]] = {}
    conservative_predictions: dict[str, list[float]] = {}
    debug_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    fallback_count = 0

    for record in records:
        query_id = str(record.get("query_id", ""))
        if not query_id:
            raise ValueError("candidate record has no query_id")
        if query_id in seen:
            raise ValueError(f"duplicate AIC candidate record: {query_id}")
        seen.add(query_id)
        candidates = list(record.get("candidates", []))
        if not candidates:
            if query_id not in fallback_predictions:
                raise ValueError(
                    "zero-candidate query has no shared Florence fallback: "
                    f"{query_id}"
                )
            fallback = validate_normalized_bbox(
                fallback_predictions[query_id]
            )
            top1_predictions[query_id] = fallback
            conservative_predictions[query_id] = fallback
            fallback_count += 1
            debug_records.append(
                {
                    "query_id": query_id,
                    "control_index": None,
                    "selected_index": None,
                    "switched": False,
                    "ranker_margin": 0.0,
                    "gdino_score_drop": 0.0,
                    "control_label": "",
                    "selected_label": "",
                    "area_ratio": 1.0,
                    "target_overlap": 0.0,
                    "reference_overlap": 0.0,
                    "query_category": str(
                        record.get("query_category", "other")
                    ),
                    "depth_rejected": False,
                    "rejection_reasons": ["zero_candidates_fallback"],
                    "candidate_count": 0,
                    "fallback_used": True,
                }
            )
            continue

        control_bbox = validate_normalized_bbox(candidates[0]["bbox"])
        top1_predictions[query_id] = control_bbox
        try:
            rows = build_feature_rows(record)
        except (BBoxError, KeyError, TypeError, ValueError) as error:
            decision = _control_decision(
                record,
                rejection_reasons=(
                    f"record_feature_error:{type(error).__name__}",
                ),
                depth_rejected=False,
            )
        else:
            scores = _ranker_scores(ranker, rows)
            decision = select_conservative_candidate(record, scores, policy)
        conservative_predictions[query_id] = decision.selected_bbox
        debug_records.append(decision.to_debug_record())

    switches = [row for row in debug_records if row["switched"]]
    reason_counts: Counter[str] = Counter()
    for row in debug_records:
        for reason in row["rejection_reasons"]:
            reason_counts[str(reason).split(":")[-1]] += 1
    total = len(records)
    summary = {
        "records": total,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / total if total else 0.0,
        "switch_count": len(switches),
        "switch_rate": len(switches) / total if total else 0.0,
        "depth_rejected_count": sum(
            bool(row["depth_rejected"]) for row in debug_records
        ),
        "cross_canonical_label_switch_count": sum(
            row["control_label"] != row["selected_label"]
            for row in switches
        ),
        "reference_dominant_switch_count": sum(
            row["reference_overlap"] > row["target_overlap"]
            for row in switches
        ),
        "max_switched_area_ratio": max(
            (float(row["area_ratio"]) for row in switches),
            default=1.0,
        ),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "policy": asdict(policy),
    }
    return ConservativeSelectionResult(
        top1_predictions=top1_predictions,
        conservative_predictions=conservative_predictions,
        debug_records=debug_records,
        summary=summary,
    )


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
    grouped: dict[str, dict[str, Any]] = {}
    categories = sorted({str(row["query_category"]) for row in outcomes})
    for category in categories:
        grouped[category] = _evaluation_bucket(
            [row for row in outcomes if row["query_category"] == category]
        )
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
        "rescue_harm_ratio": (
            rescues / harms if harms else None
        ),
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
