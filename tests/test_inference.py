from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from aic_baseline.data import AICDataset
from aic_baseline.florence import (
    FlorenceCandidate,
    FlorencePrediction,
    FlorenceOutputError,
)
from aic_baseline.inference import run_inference


class FakeGrounder:
    def predict(self, *, image: Image.Image, query: str) -> FlorencePrediction:
        assert image.size == (100, 50)
        assert query == "the object"
        candidate = FlorenceCandidate([10.0, 5.0, 50.0, 25.0], "object")
        return FlorencePrediction("raw", [candidate], candidate)


class EmptyGrounder:
    def predict(self, *, image: Image.Image, query: str) -> FlorencePrediction:
        raise FlorenceOutputError("none")


class MustNotRunGrounder:
    def predict(self, *, image: Image.Image, query: str) -> FlorencePrediction:
        raise AssertionError("resume 不应重复推理已经完成的 Query")


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


def test_inference_writes_normalized_prediction_and_debug_record(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path)

    result = run_inference(
        dataset=dataset,
        grounder=FakeGrounder(),
        output_dir=tmp_path / "run",
        fallback_mode="error",
    )

    assert result.predictions == {"q1": [0.1, 0.1, 0.5, 0.5]}
    debug = json.loads((tmp_path / "run" / "predictions_debug.jsonl").read_text())
    assert debug["query_id"] == "q1"
    assert debug["fallback_used"] is False
    assert debug["normalized_bbox"] == [0.1, 0.1, 0.5, 0.5]
    assert result.summary["valid_bbox_rate"] == 1.0


def test_submission_fallback_is_explicitly_logged(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)

    result = run_inference(
        dataset=dataset,
        grounder=EmptyGrounder(),
        output_dir=tmp_path / "run",
        fallback_mode="center",
    )

    assert result.predictions == {"q1": [0.25, 0.25, 0.75, 0.75]}
    debug = json.loads((tmp_path / "run" / "predictions_debug.jsonl").read_text())
    assert debug["fallback_used"] is True
    assert debug["error"] == "none"
    assert result.summary["fallback_rate"] == 1.0


def test_resume_reuses_checkpointed_prediction(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    output = tmp_path / "run"
    run_inference(
        dataset=dataset,
        grounder=FakeGrounder(),
        output_dir=output,
        fallback_mode="error",
    )

    resumed = run_inference(
        dataset=dataset,
        grounder=MustNotRunGrounder(),
        output_dir=output,
        fallback_mode="error",
        resume=True,
    )

    assert resumed.predictions == {"q1": [0.1, 0.1, 0.5, 0.5]}
    assert resumed.summary["records_reused_from_checkpoint"] == 1
    assert resumed.summary["records_processed_this_run"] == 0
