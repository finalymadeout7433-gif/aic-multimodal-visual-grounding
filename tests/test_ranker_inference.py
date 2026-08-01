from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from aic_baseline.ranker_inference import (
    RankerSelectionResult,
    select_aic_predictions,
    write_aic_control_and_ranker_submissions,
)


class PreferSecondRanker:
    def predict(self, features: np.ndarray) -> np.ndarray:
        assert features.shape[0] == 2
        return np.asarray([0.1, 0.9], dtype=np.float64)


def _candidate(bbox: list[float], score: float) -> dict:
    return {
        "bbox": bbox,
        "label": "person",
        "score": score,
        "original_rank": 0,
        "source": "groundingdino",
    }


def test_aic_selection_uses_same_fallback_for_both_variants() -> None:
    records = [
        {
            "query_id": "q1",
            "query": "the leftmost person",
            "query_category": "ordinal",
            "candidates": [
                _candidate([0.5, 0.1, 0.8, 0.9], 0.9),
                _candidate([0.1, 0.1, 0.3, 0.9], 0.5),
            ],
        },
        {
            "query_id": "q2",
            "query": "the car",
            "query_category": "other",
            "candidates": [],
        },
    ]
    result = select_aic_predictions(
        records=records,
        ranker=PreferSecondRanker(),
        guard_margin=0.0,
        fallback_predictions={"q2": [0.2, 0.2, 0.7, 0.7]},
    )

    assert result.top1_predictions["q1"] == [0.5, 0.1, 0.8, 0.9]
    assert result.ranker_predictions["q1"] == [0.1, 0.1, 0.3, 0.9]
    assert result.top1_predictions["q2"] == [0.2, 0.2, 0.7, 0.7]
    assert result.ranker_predictions["q2"] == [0.2, 0.2, 0.7, 0.7]
    assert result.summary["fallback_count"] == 1


def test_two_submissions_only_differ_in_bbox(tmp_path: Path) -> None:
    original = {
        "q1": {
            "visible": "Images/visible/1.png",
            "infrared": "Images/infrared/1.png",
            "depth": "Images/depth/1.png",
            "query": "person",
        }
    }
    result = RankerSelectionResult(
        top1_predictions={"q1": [0.1, 0.1, 0.4, 0.4]},
        ranker_predictions={"q1": [0.2, 0.2, 0.5, 0.5]},
        debug_records=[],
        summary={"fallback_count": 0},
    )

    audit = write_aic_control_and_ranker_submissions(
        original_records=original,
        selection=result,
        output_root=tmp_path,
    )

    control = json.loads(
        (tmp_path / "S02_gdino_top1_control" / "predictions_submission.json")
        .read_text(encoding="utf-8")
    )
    ranked = json.loads(
        (
            tmp_path
            / "S03_gdino_spatial_ltr_v1"
            / "predictions_submission.json"
        ).read_text(encoding="utf-8")
    )
    assert {**control["q1"], "bbox": None} == {
        **ranked["q1"],
        "bbox": None,
    }
    assert audit["control"]["zip_entry_count"] == 1
    with zipfile.ZipFile(audit["ranker"]["zip_path"]) as archive:
        assert archive.namelist() == ["predictions_submission.json"]
