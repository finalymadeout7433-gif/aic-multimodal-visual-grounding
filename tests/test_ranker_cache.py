from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from aic_baseline.external_eval import ExternalPrediction, ModelCandidate
from aic_baseline.ranker_cache import (
    read_candidate_cache,
    run_external_candidate_cache,
)


class FakePredictor:
    model_name = "fake-gdino"

    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        return ExternalPrediction(
            raw_output={"query": query},
            candidates=[
                ModelCandidate([10, 5, 50, 25], "target", 0.8),
                ModelCandidate([60, 5, 90, 25], "reference", 0.7),
            ],
        )


class MustNotRunPredictor:
    model_name = "fake-gdino"

    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        raise AssertionError("resume repeated a completed record")


def _record(tmp_path: Path) -> dict:
    image_path = tmp_path / "images" / "sample.jpg"
    image_path.parent.mkdir(exist_ok=True)
    Image.new("RGB", (100, 50), "white").save(image_path)
    return {
        "dataset": "refcoco",
        "query_id": "refcoco:1",
        "image_key": "coco:1",
        "query": "the target",
        "query_category": "other",
        "image_relpath": "images/sample.jpg",
        "bbox_xyxy_normalized": [0.1, 0.1, 0.5, 0.5],
    }


def test_external_candidate_cache_records_iou_rank_and_solvability(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cache"
    summary = run_external_candidate_cache(
        records=[_record(tmp_path)],
        data_root=tmp_path,
        predictor=FakePredictor(),
        output_dir=output,
        run_fingerprint={"model": "fake", "subset": "one"},
    )

    record = read_candidate_cache(output / "candidates.jsonl")[0]
    assert record["candidates"][0]["original_rank"] == 0
    assert record["candidates"][0]["iou"] == 1.0
    assert record["candidate_ious"] == [1.0, 0.0]
    assert record["best_candidate_iou"] == 1.0
    assert record["solvable_at_05"] is True
    assert summary["records"] == 1


def test_candidate_cache_resume_ignores_only_partial_final_line(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cache"
    fingerprint = {"model": "fake", "subset": "one"}
    run_external_candidate_cache(
        records=[_record(tmp_path)],
        data_root=tmp_path,
        predictor=FakePredictor(),
        output_dir=output,
        run_fingerprint=fingerprint,
    )
    with (output / "candidates.jsonl").open("ab") as handle:
        handle.write(b'{"query_id":"partial"')

    summary = run_external_candidate_cache(
        records=[_record(tmp_path)],
        data_root=tmp_path,
        predictor=MustNotRunPredictor(),
        output_dir=output,
        run_fingerprint=fingerprint,
        resume=True,
    )

    assert summary["records_reused"] == 1
    assert len(read_candidate_cache(output / "candidates.jsonl")) == 1
    assert json.loads(
        (output / "run_fingerprint.json").read_text(encoding="utf-8")
    ) == fingerprint
