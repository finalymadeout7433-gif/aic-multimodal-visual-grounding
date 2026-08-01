from __future__ import annotations

import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from aic_baseline.conservative_selector import (
    ConservativeSwitchPolicy,
    ConservativeSelectionResult,
    evaluate_conservative_policy,
    select_conservative_candidate,
    select_conservative_predictions,
    write_conservative_submission,
)


def _candidate(
    bbox: list[float],
    score: float,
    *,
    label: str = "person",
) -> dict[str, object]:
    return {
        "bbox": bbox,
        "label": label,
        "score": score,
        "original_rank": 0,
        "source": "groundingdino",
    }


def _record(
    *,
    query: str = "the person",
    query_category: str = "other",
    candidates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "query_id": "q1",
        "query": query,
        "query_category": query_category,
        "candidates": candidates
        if candidates is not None
        else [
            _candidate([0.1, 0.1, 0.3, 0.3], 0.90),
            _candidate([0.5, 0.1, 0.7, 0.3], 0.80),
        ],
    }


def test_ranker_margin_boundary_is_inclusive() -> None:
    accepted = select_conservative_candidate(
        _record(), [0.1, 0.6], ConservativeSwitchPolicy()
    )
    rejected = select_conservative_candidate(
        _record(), [0.1, 0.599999], ConservativeSwitchPolicy()
    )

    assert accepted.selected_index == 1
    assert rejected.selected_index == 0
    assert any("ranker_margin" in reason for reason in rejected.rejection_reasons)


def test_gdino_score_drop_boundary_is_inclusive() -> None:
    accepted = select_conservative_candidate(
        _record(), [0.0, 1.0], ConservativeSwitchPolicy()
    )
    rejected_record = _record(
        candidates=[
            _candidate([0.1, 0.1, 0.3, 0.3], 0.90),
            _candidate([0.5, 0.1, 0.7, 0.3], 0.799999),
        ]
    )
    rejected = select_conservative_candidate(
        rejected_record, [0.0, 1.0], ConservativeSwitchPolicy()
    )

    assert accepted.selected_index == 1
    assert rejected.selected_index == 0
    assert any("gdino_score_drop" in reason for reason in rejected.rejection_reasons)


@pytest.mark.parametrize(
    ("selected_label", "should_switch"),
    [("the person", True), ("car", False), ("", False)],
)
def test_canonical_label_gate(
    selected_label: str, should_switch: bool
) -> None:
    record = _record(
        candidates=[
            _candidate([0.1, 0.1, 0.3, 0.3], 0.90, label="person"),
            _candidate(
                [0.5, 0.1, 0.7, 0.3],
                0.80,
                label=selected_label,
            ),
        ]
    )
    decision = select_conservative_candidate(
        record, [0.0, 1.0], ConservativeSwitchPolicy()
    )

    assert decision.switched is should_switch


def test_reference_dominant_candidate_is_rejected() -> None:
    record = _record(
        query="the red car beside the car",
        candidates=[
            _candidate([0.1, 0.1, 0.3, 0.3], 0.90, label="car"),
            _candidate([0.5, 0.1, 0.7, 0.3], 0.80, label="car"),
        ],
    )
    decision = select_conservative_candidate(
        record, [0.0, 1.0], ConservativeSwitchPolicy()
    )

    assert decision.selected_index == 0
    assert decision.reference_overlap > decision.target_overlap
    assert any(
        "reference_dominant" in reason
        for reason in decision.rejection_reasons
    )


def test_area_ratio_boundary_is_inclusive() -> None:
    accepted = _record(
        candidates=[
            _candidate([0.1, 0.1, 0.3, 0.3], 0.90),
            _candidate([0.5, 0.1, 0.8, 0.3], 0.80),
        ]
    )
    rejected = _record(
        candidates=[
            _candidate([0.1, 0.1, 0.3, 0.3], 0.90),
            _candidate([0.5, 0.1, 0.800001, 0.3], 0.80),
        ]
    )

    accepted_decision = select_conservative_candidate(
        accepted, [0.0, 1.0], ConservativeSwitchPolicy()
    )
    rejected_decision = select_conservative_candidate(
        rejected, [0.0, 1.0], ConservativeSwitchPolicy()
    )

    assert accepted_decision.selected_index == 1
    assert accepted_decision.area_ratio == pytest.approx(1.5)
    assert rejected_decision.selected_index == 0


@pytest.mark.parametrize(
    "query",
    [
        "the person in front of the car",
        "the person behind the car",
        "the nearest person",
        "the farthest person",
    ],
)
def test_depth_phrases_never_switch(query: str) -> None:
    decision = select_conservative_candidate(
        _record(query=query), [0.0, 1.0], ConservativeSwitchPolicy()
    )

    assert decision.selected_index == 0
    assert decision.depth_rejected is True


def test_depth_category_never_switches() -> None:
    decision = select_conservative_candidate(
        _record(query_category="depth"),
        [0.0, 1.0],
        ConservativeSwitchPolicy(),
    )

    assert decision.selected_index == 0
    assert decision.depth_rejected is True


def test_searches_next_candidate_when_best_ranker_candidate_is_unsafe() -> None:
    record = _record(
        candidates=[
            _candidate([0.1, 0.1, 0.3, 0.3], 0.90, label="person"),
            _candidate([0.4, 0.1, 0.6, 0.3], 0.85, label="car"),
            _candidate([0.7, 0.1, 0.9, 0.3], 0.82, label="person"),
        ]
    )
    decision = select_conservative_candidate(
        record, [0.0, 1.0, 0.8], ConservativeSwitchPolicy()
    )

    assert decision.selected_index == 2
    assert decision.switched is True
    assert any("candidate[1]" in reason for reason in decision.rejection_reasons)


def test_no_safe_candidate_falls_back_exactly_to_top1() -> None:
    record = _record(
        candidates=[
            _candidate([0.1, 0.1, 0.3, 0.3], 0.90, label="person"),
            _candidate([0.5, 0.1, 0.7, 0.3], 0.80, label="car"),
        ]
    )
    decision = select_conservative_candidate(
        record, [0.0, 1.0], ConservativeSwitchPolicy()
    )

    assert decision.selected_index == 0
    assert decision.selected_bbox == [0.1, 0.1, 0.3, 0.3]


class _NeverCalledRanker:
    def predict(self, features: np.ndarray) -> np.ndarray:
        raise AssertionError("ranker must not run for a zero-candidate record")


def test_zero_candidates_use_the_same_florence_fallback_as_s02() -> None:
    result = select_conservative_predictions(
        records=[_record(candidates=[])],
        ranker=_NeverCalledRanker(),
        policy=ConservativeSwitchPolicy(),
        fallback_predictions={"q1": [0.2, 0.2, 0.7, 0.7]},
    )

    assert result.top1_predictions["q1"] == [0.2, 0.2, 0.7, 0.7]
    assert result.conservative_predictions["q1"] == [0.2, 0.2, 0.7, 0.7]
    assert result.summary["fallback_count"] == 1


def test_nonfinite_candidate_data_keeps_top1_and_is_audited() -> None:
    record = _record(
        candidates=[
            _candidate([0.1, 0.1, 0.3, 0.3], 0.90),
            _candidate([0.5, 0.1, math.nan, 0.3], 0.80),
        ]
    )
    decision = select_conservative_candidate(
        record, [0.0, 1.0], ConservativeSwitchPolicy()
    )

    assert decision.selected_index == 0
    assert any("invalid_bbox" in reason for reason in decision.rejection_reasons)


def test_conservative_submission_is_byte_reproducible_and_single_entry(
    tmp_path: Path,
) -> None:
    original = {
        "q1": {
            "visible": "Images/visible/1.png",
            "infrared": "Images/infrared/1.png",
            "depth": "Images/depth/1.png",
            "query": "the person",
        }
    }
    selection = ConservativeSelectionResult(
        top1_predictions={"q1": [0.1, 0.1, 0.3, 0.3]},
        conservative_predictions={"q1": [0.5, 0.1, 0.7, 0.3]},
        debug_records=[],
        summary={},
    )

    first = write_conservative_submission(
        original_records=original,
        selection=selection,
        output_dir=tmp_path / "first",
    )
    second = write_conservative_submission(
        original_records=original,
        selection=selection,
        output_dir=tmp_path / "second",
    )

    assert first["json_sha256"] == second["json_sha256"]
    assert first["zip_sha256"] == second["zip_sha256"]
    assert (tmp_path / "first" / "predictions_submission.json").read_bytes() == (
        tmp_path / "second" / "predictions_submission.json"
    ).read_bytes()
    assert (tmp_path / "first" / "predictions_submission.zip").read_bytes() == (
        tmp_path / "second" / "predictions_submission.zip"
    ).read_bytes()
    with zipfile.ZipFile(first["zip_path"]) as archive:
        assert archive.namelist() == ["predictions_submission.json"]


class _AlwaysPreferSecondRanker:
    def predict(self, features: np.ndarray) -> np.ndarray:
        assert features.shape[0] == 2
        return np.asarray([0.0, 1.0], dtype=np.float64)


def test_external_evaluation_reports_rescue_harm_and_empty_records() -> None:
    rescue = _record()
    rescue["query_id"] = "rescue"
    rescue["candidate_ious"] = [0.1, 0.8]
    harm = _record()
    harm["query_id"] = "harm"
    harm["candidate_ious"] = [0.8, 0.1]
    empty = _record(candidates=[])
    empty["query_id"] = "empty"

    evaluation = evaluate_conservative_policy(
        records=[rescue, harm, empty],
        ranker=_AlwaysPreferSecondRanker(),
        policy=ConservativeSwitchPolicy(),
    )

    assert evaluation["cache_records"] == 3
    assert evaluation["evaluated_queries"] == 2
    assert evaluation["no_candidate_records"] == 1
    assert evaluation["baseline_acc_at_05"] == 0.5
    assert evaluation["selected_acc_at_05"] == 0.5
    assert evaluation["rescues"] == 1
    assert evaluation["harms"] == 1
    assert evaluation["rescue_harm_ratio"] == 1.0
