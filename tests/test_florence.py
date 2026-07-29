from __future__ import annotations

import pytest

from aic_baseline.florence import (
    FlorenceOutputError,
    extract_candidates,
    select_candidate,
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
