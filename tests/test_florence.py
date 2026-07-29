from __future__ import annotations

import pytest

from aic_baseline.florence import (
    FlorenceConfigurationError,
    FlorenceOutputError,
    extract_candidates,
    select_candidate,
    validate_selection_strategy,
)


def test_extract_candidates_reads_processor_public_result() -> None:
    parsed = {
        "<CAPTION_TO_PHRASE_GROUNDING>": {
            "bboxes": [[10.0, 20.0, 30.0, 40.0], [50, 60, 70, 80]],
            "labels": ["first object", "second object"],
        }
    }

    candidates = extract_candidates(parsed)

    assert [candidate.pixel_bbox for candidate in candidates] == [
        [10.0, 20.0, 30.0, 40.0],
        [50.0, 60.0, 70.0, 80.0],
    ]
    assert [candidate.label for candidate in candidates] == [
        "first object",
        "second object",
    ]
    assert select_candidate(candidates, strategy="first") == candidates[0]


def test_extract_candidates_rejects_empty_model_output() -> None:
    with pytest.raises(FlorenceOutputError, match="候选框"):
        extract_candidates(
            {"<CAPTION_TO_PHRASE_GROUNDING>": {"bboxes": [], "labels": []}}
        )


def test_extract_candidates_wraps_non_numeric_coordinates() -> None:
    parsed = {
        "<CAPTION_TO_PHRASE_GROUNDING>": {
            "bboxes": [["left", 20.0, 30.0, 40.0]],
            "labels": ["object"],
        }
    }

    with pytest.raises(FlorenceOutputError, match="坐标"):
        extract_candidates(parsed)


def test_invalid_selection_strategy_is_a_configuration_error() -> None:
    with pytest.raises(FlorenceConfigurationError, match="策略"):
        validate_selection_strategy("largest")
