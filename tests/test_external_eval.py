from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from aic_baseline.external_eval import (
    ExternalPrediction,
    ModelCandidate,
    normalize_grounding_query,
    run_external_evaluation,
)


class FakePredictor:
    model_name = "fake"

    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        assert image.size == (100, 50)
        assert query == "the target"
        return ExternalPrediction(
            raw_output="raw",
            candidates=[
                ModelCandidate([0, 0, 20, 10], "reference", 0.9),
                ModelCandidate([10, 5, 50, 25], "target", 0.8),
            ],
        )


class MustNotRunPredictor:
    model_name = "fake"

    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        raise AssertionError("resume must not repeat completed records")


class RuntimeFailurePredictor:
    model_name = "broken"

    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        raise RuntimeError("CUDA out of memory")


def _record(tmp_path: Path) -> dict[str, object]:
    image = tmp_path / "images" / "sample.jpg"
    image.parent.mkdir(exist_ok=True)
    Image.new("RGB", (100, 50), "white").save(image)
    return {
        "dataset": "refcoco",
        "query_id": "refcoco:1",
        "image_key": "coco:1",
        "query": "the target",
        "image_relpath": "images/sample.jpg",
        "bbox_xyxy_normalized": [0.1, 0.1, 0.5, 0.5],
    }


def test_external_eval_records_first_and_oracle_without_fallback(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    result = run_external_evaluation(
        records=[_record(tmp_path)],
        data_root=tmp_path,
        predictor=FakePredictor(),
        output_dir=output,
        run_fingerprint={"model": "fake", "subset": "one"},
    )

    assert result["overall"]["selected_acc_at_05"] == 0.0
    assert result["overall"]["oracle_acc_at_05"] == 1.0
    record = json.loads(
        (output / "predictions.jsonl").read_text(encoding="utf-8")
    )
    assert record["candidates"][0]["score"] == 0.9
    assert record["selected_bbox"] == [0.0, 0.0, 0.2, 0.2]
    assert record["best_candidate_iou"] == 1.0
    assert record["error"] is None


def test_external_eval_resume_does_not_repeat_completed_record(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    fingerprint = {"model": "fake", "subset": "one"}
    run_external_evaluation(
        records=[_record(tmp_path)],
        data_root=tmp_path,
        predictor=FakePredictor(),
        output_dir=output,
        run_fingerprint=fingerprint,
    )
    summary = run_external_evaluation(
        records=[_record(tmp_path)],
        data_root=tmp_path,
        predictor=MustNotRunPredictor(),
        output_dir=output,
        run_fingerprint=fingerprint,
        resume=True,
    )

    assert summary["execution"]["records_reused"] == 1
    assert summary["execution"]["records_processed"] == 0
    assert len(
        (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ) == 1


def test_external_eval_resume_rejects_changed_fingerprint(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    run_external_evaluation(
        records=[_record(tmp_path)],
        data_root=tmp_path,
        predictor=FakePredictor(),
        output_dir=output,
        run_fingerprint={"model": "fake", "subset": "one"},
    )
    with pytest.raises(ValueError, match="fingerprint"):
        run_external_evaluation(
            records=[_record(tmp_path)],
            data_root=tmp_path,
            predictor=MustNotRunPredictor(),
            output_dir=output,
            run_fingerprint={"model": "fake", "subset": "changed"},
            resume=True,
        )


def test_external_eval_runtime_error_aborts_without_creating_candidate(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run_external_evaluation(
            records=[_record(tmp_path)],
            data_root=tmp_path,
            predictor=RuntimeFailurePredictor(),
            output_dir=output,
            run_fingerprint={"model": "broken"},
        )

    assert (output / "predictions.jsonl").read_text(encoding="utf-8") == ""


def test_normalize_grounding_query_is_lowercase_with_terminal_period() -> None:
    assert normalize_grounding_query("  The RED car!  ") == "the red car."
    assert normalize_grounding_query("A person.") == "a person."
