from __future__ import annotations

import pytest

from aic_baseline.ranker_features import (
    FEATURE_NAMES,
    build_feature_rows,
    deduplicate_rank_candidates,
    parse_query_semantics,
    relevance_from_iou,
)


def _candidate(
    bbox: list[float],
    label: str,
    score: float,
    iou: float = 0.0,
) -> dict:
    return {
        "bbox": bbox,
        "label": label,
        "score": score,
        "source": "groundingdino",
        "iou": iou,
    }


def test_query_parser_separates_target_relation_and_reference() -> None:
    parsed = parse_query_semantics("The man left of the blue bus")

    assert parsed.target_phrase == "the man"
    assert parsed.reference_phrase == "the blue bus"
    assert parsed.relative_relation == "left_of"
    assert parsed.depth_relation is None


def test_query_parser_handles_ordinal_and_does_not_fake_depth_geometry() -> None:
    ordinal = parse_query_semantics("the second umbrella from the right")
    depth = parse_query_semantics("the person farthest from the camera")

    assert ordinal.ordinal_index == 1
    assert ordinal.ordinal_axis == "x"
    assert ordinal.ordinal_reverse is True
    assert depth.depth_relation == "farthest"
    assert depth.relative_relation is None


@pytest.mark.parametrize(
    ("iou", "expected"),
    [(0.0, 0), (0.1999, 0), (0.2, 1), (0.4999, 1), (0.5, 2), (0.7, 3)],
)
def test_relevance_boundaries(iou: float, expected: int) -> None:
    assert relevance_from_iou(iou) == expected


def test_candidate_dedup_only_merges_same_canonical_label() -> None:
    candidates = [
        _candidate([0.1, 0.1, 0.4, 0.4], "The Person", 0.9),
        _candidate([0.101, 0.101, 0.399, 0.399], "the person.", 0.8),
        _candidate([0.1, 0.1, 0.4, 0.4], "shirt", 0.7),
    ]

    result = deduplicate_rank_candidates(candidates, iou_threshold=0.95)

    assert [candidate["label"] for candidate in result] == [
        "The Person",
        "shirt",
    ]


def test_leftmost_feature_is_computed_within_target_compatible_candidates() -> None:
    record = {
        "query_id": "q1",
        "dataset": "refcoco",
        "query_category": "ordinal",
        "query": "the leftmost car beside the licenseplate",
        "gt_bbox": [0.05, 0.2, 0.25, 0.7],
        "candidates": [
            _candidate([0.55, 0.2, 0.8, 0.7], "car", 0.9, 0.0),
            _candidate([0.05, 0.2, 0.25, 0.7], "car", 0.5, 1.0),
            _candidate([0.9, 0.4, 0.98, 0.5], "licenseplate", 0.7, 0.0),
        ],
    }

    rows = build_feature_rows(record)
    left_car = rows[1]
    plate = rows[2]

    assert set(left_car["features"]) == set(FEATURE_NAMES)
    assert left_car["features"]["target_compatible"] == 1.0
    assert left_car["features"]["absolute_relation_satisfaction"] == 1.0
    assert plate["features"]["reference_overlap"] > plate["features"]["target_overlap"]


def test_relative_relation_uses_reference_candidate_geometry() -> None:
    record = {
        "query_id": "q2",
        "dataset": "refcoco",
        "query_category": "spatial",
        "query": "the man left of the bus",
        "gt_bbox": [0.1, 0.2, 0.3, 0.8],
        "candidates": [
            _candidate([0.1, 0.2, 0.3, 0.8], "man", 0.5, 1.0),
            _candidate([0.7, 0.2, 0.95, 0.8], "bus", 0.9, 0.0),
        ],
    }

    rows = build_feature_rows(record)

    assert rows[0]["features"]["reference_available"] == 1.0
    assert rows[0]["features"]["relative_relation_satisfaction"] == 1.0
    assert rows[0]["features"]["target_to_reference_dx"] > 0.0
    assert rows[1]["features"]["target_compatible"] == 0.0


def test_depth_relation_has_flag_but_no_2d_relation_satisfaction() -> None:
    record = {
        "query_id": "q3",
        "dataset": "refcoco",
        "query_category": "depth",
        "query": "the farthest person",
        "gt_bbox": [0.1, 0.1, 0.3, 0.8],
        "candidates": [
            _candidate([0.1, 0.1, 0.3, 0.8], "person", 0.4, 1.0),
            _candidate([0.7, 0.1, 0.9, 0.8], "person", 0.5, 0.0),
        ],
    }

    rows = build_feature_rows(record)

    assert rows[0]["features"]["flag_depth_relation"] == 1.0
    assert rows[0]["features"]["relative_relation_satisfaction"] == 0.0
