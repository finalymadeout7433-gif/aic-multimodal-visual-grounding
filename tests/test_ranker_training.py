from __future__ import annotations

from pathlib import Path

import numpy as np

from aic_baseline.ranker_training import (
    RankerMatrix,
    evaluate_ranker_scores,
    load_ranker_model,
    save_ranker_model,
    select_indices_from_scores,
)


def _matrix() -> RankerMatrix:
    return RankerMatrix(
        features=np.asarray(
            [
                [0.9, 0.9],
                [0.4, 0.1],
                [0.8, 0.8],
                [0.3, 0.2],
            ],
            dtype=np.float32,
        ),
        relevance=np.asarray([0, 3, 3, 0], dtype=np.int32),
        groups=np.asarray([2, 2], dtype=np.int32),
        candidate_ious=np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        feature_names=["score", "leftmost"],
        query_metadata=[
            {
                "query_id": "q1",
                "dataset": "refcoco",
                "query_category": "spatial",
                "area_bin": "large_ge_10pct",
            },
            {
                "query_id": "q2",
                "dataset": "refcoco_plus",
                "query_category": "other",
                "area_bin": "medium_1_to_10pct",
            },
        ],
    )


def test_guard_margin_preserves_top1_when_ranker_is_uncertain() -> None:
    matrix = _matrix()
    scores = np.asarray([0.4, 0.45, 0.7, 0.1], dtype=np.float64)

    guarded = select_indices_from_scores(matrix, scores, guard_margin=0.1)
    unguarded = select_indices_from_scores(matrix, scores, guard_margin=0.0)

    assert guarded == [0, 0]
    assert unguarded == [1, 0]


def test_ranker_evaluation_reports_rescue_and_harm() -> None:
    matrix = _matrix()
    scores = np.asarray([0.1, 0.9, 0.8, 0.2], dtype=np.float64)

    result = evaluate_ranker_scores(matrix, scores, guard_margin=0.0)

    assert result["overall"]["baseline_acc_at_05"] == 0.5
    assert result["overall"]["selected_acc_at_05"] == 1.0
    assert result["overall"]["rescued"] == 1
    assert result["overall"]["harmed"] == 0


def test_ranker_model_text_roundtrip(tmp_path: Path) -> None:
    lightgbm = __import__("lightgbm")
    model = lightgbm.LGBMRanker(
        objective="lambdarank",
        n_estimators=5,
        learning_rate=0.1,
        num_leaves=3,
        min_child_samples=1,
        verbosity=-1,
    )
    matrix = _matrix()
    model.fit(matrix.features, matrix.relevance, group=matrix.groups)
    path = tmp_path / "ranker.txt"

    save_ranker_model(model, path)
    loaded = load_ranker_model(path)

    np.testing.assert_allclose(
        model.predict(matrix.features),
        loaded.predict(matrix.features),
    )
