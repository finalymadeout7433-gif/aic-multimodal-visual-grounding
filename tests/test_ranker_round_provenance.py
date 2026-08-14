from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_ranker_training_round import (
    _candidate_fingerprint,
    _candidate_health,
    _ranker_semantics_sha256,
    _validate_frozen_decision,
)


def test_candidate_fingerprint_covers_model_config_processor_and_source(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    source = tmp_path / "subset.jsonl"
    source.write_text('{"query_id":"one"}\n', encoding="utf-8")
    config = {
        "grounding_dino": {
            "revision": "fixed-revision",
            "dtype": "float32",
            "box_threshold": 0.15,
            "text_threshold": 0.15,
            "max_candidates": 10,
            "dedup_iou": 0.95,
        }
    }
    paths = {"grounding_dino_model": model}

    fingerprint = _candidate_fingerprint(
        config,
        paths,
        source_name="subset",
        source_path=source,
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert set(fingerprint["model_artifacts"]) == {
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    }
    assert fingerprint["subset_sha256"]
    assert fingerprint["model_artifacts_sha256"]
    assert fingerprint["code_sha256"]
    assert set(fingerprint["runtime_versions"]) == {
        "python",
        "torch",
        "transformers",
        "pillow",
        "cuda",
    }


def test_ranker_semantics_digest_changes_with_feature_code(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src/aic_baseline"
    package.mkdir(parents=True)
    for name in (
        "bbox.py",
        "ranker_features.py",
        "ranker_training.py",
        "ranker_inference.py",
    ):
        (package / name).write_text(f"# {name}\n", encoding="utf-8")

    before = _ranker_semantics_sha256(tmp_path)
    (package / "ranker_features.py").write_text(
        "# changed feature semantics\n",
        encoding="utf-8",
    )
    after = _ranker_semantics_sha256(tmp_path)

    assert before != after


def test_candidate_health_rejects_non_finite_or_reversed_boxes() -> None:
    health = _candidate_health(
        [
            {
                "error": None,
                "candidates": [
                    {"bbox": [0.1, 0.1, 0.4, 0.4]},
                    {"bbox": [0.5, 0.1, 0.4, 0.4]},
                    {"bbox": [0.1, 0.1, float("nan"), 0.4]},
                ],
            },
            {"error": "runtime_error", "candidates": []},
        ]
    )

    assert health["candidate_count"] == 3
    assert health["legal_candidate_count"] == 1
    assert health["candidate_bbox_legal_rate"] == pytest.approx(1 / 3)
    assert health["unexpected_error_count"] == 1


def test_holdout_gate_rejects_failed_validation_without_opening_artifacts(
    tmp_path: Path,
) -> None:
    decision = tmp_path / "evaluations/validation/frozen_decision.json"
    decision.parent.mkdir(parents=True)
    decision.write_text(
        json.dumps({"validation_promotion_pass": False}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="holdout remains sealed"):
        _validate_frozen_decision(
            {
                "ranker": {},
                "grounding_dino": {},
                "subsets": {},
                "promotion": {},
            },
            {"output_root": tmp_path},
        )
