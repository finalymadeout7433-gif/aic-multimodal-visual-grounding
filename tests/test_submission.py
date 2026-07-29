from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from aic_baseline.submission import SubmissionError, build_submission


def test_submission_preserves_every_original_field_and_adds_only_bbox(
    tmp_path: Path,
) -> None:
    original = {
        "q1": {
            "visible": "Images/visible/a.png",
            "infrared": "Images/infrared/a.png",
            "depth": "Images/depth/a.png",
            "query": "the object",
        }
    }
    output_json = tmp_path / "submission.json"
    output_zip = tmp_path / "submission.zip"

    build_submission(
        original_records=original,
        predictions={"q1": [0.1, 0.2, 0.5, 0.8]},
        output_json=output_json,
        output_zip=output_zip,
    )

    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["q1"] == {
        **original["q1"],
        "bbox": [0.1, 0.2, 0.5, 0.8],
    }
    assert original["q1"].get("bbox") is None
    with zipfile.ZipFile(output_zip) as archive:
        assert archive.namelist() == ["submission.json"]
        zipped = json.loads(archive.read("submission.json").decode("utf-8"))
    assert zipped == written


def test_submission_rejects_missing_or_extra_prediction_ids(tmp_path: Path) -> None:
    original = {"q1": {"query": "x"}}

    with pytest.raises(SubmissionError, match="ID 集合"):
        build_submission(
            original_records=original,
            predictions={"q2": [0.1, 0.2, 0.5, 0.8]},
            output_json=tmp_path / "submission.json",
            output_zip=tmp_path / "submission.zip",
        )
