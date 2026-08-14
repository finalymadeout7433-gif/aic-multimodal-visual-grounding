from __future__ import annotations

import json
import zipfile
from pathlib import Path

from PIL import Image
import pytest

from aic_baseline.aic_full_detection import (
    finalize_aic_full_detection,
    run_aic_full_detection,
)
from aic_baseline.data import AICDataset
from aic_baseline.external_eval import ExternalPrediction, ModelCandidate


FINGERPRINT = {
    "model": "fake-grounder",
    "queries_sha256": "queries-a",
    "weights_sha256": "weights-a",
}


class FakePredictor:
    model_name = "fake-grounder"

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        self.calls += 1
        assert image.size == (100, 50)
        assert query == "the object"
        return ExternalPrediction(
            raw_output={"ok": True},
            candidates=[
                ModelCandidate(
                    pixel_bbox=[20.0, 10.0, 80.0, 40.0],
                    label="lower-score",
                    score=0.25,
                ),
                ModelCandidate(
                    pixel_bbox=[-5.0, 5.0, 50.0, 30.0],
                    label="higher-score",
                    score=0.75,
                ),
            ],
        )


class EmptyPredictor:
    model_name = "empty-grounder"

    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        return ExternalPrediction(raw_output={}, candidates=[])


class MustNotRunPredictor:
    model_name = "fake-grounder"

    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        raise AssertionError("resume must not repeat completed inference")


class FakeBatchPredictor(FakePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0

    def predict_batch(
        self, *, images: list[Image.Image], queries: list[str]
    ) -> list[ExternalPrediction]:
        self.batch_calls += 1
        return [self.predict(image=image, query=query) for image, query in zip(images, queries)]


def make_dataset(tmp_path: Path) -> AICDataset:
    image_path = tmp_path / "Images" / "visible" / "a.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 50)).save(image_path)
    payload = {
        "q1": {
            "visible": "Images/visible/a.png",
            "infrared": "Images/infrared/a.png",
            "depth": "Images/depth/a.png",
            "query": "the object",
        }
    }
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps(payload), encoding="utf-8")
    return AICDataset(dataset_root=tmp_path, queries_path=queries)


def make_same_image_dataset(tmp_path: Path) -> AICDataset:
    image_path = tmp_path / "Images" / "visible" / "a.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 50)).save(image_path)
    payload = {
        f"q{index}": {
            "visible": "Images/visible/a.png",
            "infrared": "Images/infrared/a.png",
            "depth": "Images/depth/a.png",
            "query": "the object",
        }
        for index in range(1, 4)
    }
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps(payload), encoding="utf-8")
    return AICDataset(dataset_root=tmp_path, queries_path=queries)


def test_full_detection_selects_highest_score_and_clips_bbox(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    output = tmp_path / "run"

    result = run_aic_full_detection(
        dataset=dataset,
        predictor=FakePredictor(),
        output_dir=output,
        run_fingerprint=FINGERPRINT,
    )

    assert result.predictions == {"q1": [0.0, 0.1, 0.5, 0.6]}
    record = json.loads((output / "predictions.jsonl").read_text())
    assert record["selected_label"] == "higher-score"
    assert record["selected_score"] == 0.75
    assert record["candidate_count"] == 2
    assert "latency_ms" not in record
    runtime = json.loads((output / "runtime_events.jsonl").read_text())
    assert runtime["query_id"] == "q1"
    assert runtime["latency_ms"] >= 0.0


def test_full_detection_resume_reuses_completed_records(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    output = tmp_path / "run"
    run_aic_full_detection(
        dataset=dataset,
        predictor=FakePredictor(),
        output_dir=output,
        run_fingerprint=FINGERPRINT,
    )

    result = run_aic_full_detection(
        dataset=dataset,
        predictor=MustNotRunPredictor(),
        output_dir=output,
        run_fingerprint=FINGERPRINT,
        resume=True,
    )

    assert result.summary["records_reused"] == 1
    assert result.summary["records_processed"] == 0


def test_full_detection_batches_queries_that_share_an_image(tmp_path: Path) -> None:
    dataset = make_same_image_dataset(tmp_path)
    predictor = FakeBatchPredictor()

    result = run_aic_full_detection(
        dataset=dataset,
        predictor=predictor,
        output_dir=tmp_path / "run",
        run_fingerprint={**FINGERPRINT, "batch_size": 3},
        batch_size=3,
    )

    assert predictor.batch_calls == 1
    assert predictor.calls == 3
    assert result.summary["records_processed"] == 3
    runtime_rows = [
        json.loads(line)
        for line in result.runtime_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["batch_size"] for row in runtime_rows} == {3}


def test_full_detection_resume_rejects_fingerprint_change(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    output = tmp_path / "run"
    run_aic_full_detection(
        dataset=dataset,
        predictor=FakePredictor(),
        output_dir=output,
        run_fingerprint=FINGERPRINT,
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_aic_full_detection(
            dataset=dataset,
            predictor=MustNotRunPredictor(),
            output_dir=output,
            run_fingerprint={**FINGERPRINT, "weights_sha256": "changed"},
            resume=True,
        )


def test_full_detection_aborts_on_no_candidate(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)

    with pytest.raises(RuntimeError, match="returned no candidate"):
        run_aic_full_detection(
            dataset=dataset,
            predictor=EmptyPredictor(),
            output_dir=tmp_path / "run",
            run_fingerprint={**FINGERPRINT, "model": "empty-grounder"},
        )


def test_finalize_builds_platform_zip_and_only_changes_bbox(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    output = tmp_path / "run"
    run_aic_full_detection(
        dataset=dataset,
        predictor=FakePredictor(),
        output_dir=output,
        run_fingerprint=FINGERPRINT,
    )

    result = finalize_aic_full_detection(
        dataset=dataset,
        detection_dir=output,
        submission_dir=tmp_path / "submission",
    )

    submission = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert submission["q1"]["query"] == "the object"
    assert submission["q1"]["bbox"] == [0.0, 0.1, 0.5, 0.6]
    assert set(submission["q1"]) == {
        "visible",
        "infrared",
        "depth",
        "query",
        "bbox",
    }
    with zipfile.ZipFile(result.zip_path) as archive:
        assert archive.namelist() == ["predictions_submission.json"]
    assert result.audit["query_count"] == 1
    assert result.audit["invalid_bbox_count"] == 0
