from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from aic_baseline.data import AICDataset
from aic_baseline.florence import (
    FlorenceCandidate,
    FlorenceConfigurationError,
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


class RuntimeFailureGrounder:
    def predict(self, *, image: Image.Image, query: str) -> FlorencePrediction:
        raise RuntimeError("CUDA out of memory")


class ConfigurationFailureGrounder:
    def predict(self, *, image: Image.Image, query: str) -> FlorencePrediction:
        raise FlorenceConfigurationError("unsupported selection strategy")


RUN_FINGERPRINT = {
    "queries_sha256": "queries-a",
    "model_weights_sha256": "weights-a",
    "max_new_tokens": 256,
}


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
        run_fingerprint=RUN_FINGERPRINT,
    )

    resumed = run_inference(
        dataset=dataset,
        grounder=MustNotRunGrounder(),
        output_dir=output,
        fallback_mode="error",
        resume=True,
        run_fingerprint=RUN_FINGERPRINT,
    )

    assert resumed.predictions == {"q1": [0.1, 0.1, 0.5, 0.5]}
    assert resumed.summary["records_reused_from_checkpoint"] == 1
    assert resumed.summary["records_processed_this_run"] == 0


def test_resume_can_initialize_an_empty_output_directory(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    output = tmp_path / "run"

    result = run_inference(
        dataset=dataset,
        grounder=FakeGrounder(),
        output_dir=output,
        fallback_mode="error",
        resume=True,
        run_fingerprint=RUN_FINGERPRINT,
    )

    assert result.predictions == {"q1": [0.1, 0.1, 0.5, 0.5]}
    assert json.loads(
        (output / "run_fingerprint.json").read_text(encoding="utf-8")
    ) == RUN_FINGERPRINT


def test_resume_rejects_mismatched_run_fingerprint(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    output = tmp_path / "run"
    run_inference(
        dataset=dataset,
        grounder=FakeGrounder(),
        output_dir=output,
        fallback_mode="error",
        run_fingerprint=RUN_FINGERPRINT,
    )

    with pytest.raises(ValueError, match="指纹"):
        run_inference(
            dataset=dataset,
            grounder=MustNotRunGrounder(),
            output_dir=output,
            fallback_mode="error",
            resume=True,
            run_fingerprint={**RUN_FINGERPRINT, "max_new_tokens": 128},
        )


def test_runtime_error_aborts_instead_of_using_center_fallback(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run_inference(
            dataset=dataset,
            grounder=RuntimeFailureGrounder(),
            output_dir=tmp_path / "run",
            fallback_mode="center",
        )

    debug_path = tmp_path / "run" / "predictions_debug.jsonl"
    assert debug_path.read_text(encoding="utf-8") == ""


def test_configuration_error_aborts_instead_of_using_center_fallback(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path)

    with pytest.raises(FlorenceConfigurationError, match="strategy"):
        run_inference(
            dataset=dataset,
            grounder=ConfigurationFailureGrounder(),
            output_dir=tmp_path / "run",
            fallback_mode="center",
        )
