from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from .ranker_features import FEATURE_NAMES, build_feature_rows


SCORE_GEOMETRY_FEATURE_NAMES: tuple[str, ...] = (
    "gdino_score",
    "original_rank",
    "score_gap_to_first",
    "score_gap_to_next",
    "cx",
    "cy",
    "width",
    "height",
    "area",
    "log_area",
    "aspect_ratio",
    "edge_left",
    "edge_right",
    "edge_top",
    "edge_bottom",
    "rank_left_all",
    "rank_right_all",
    "rank_top_all",
    "rank_bottom_all",
    "rank_area_large_all",
    "rank_area_small_all",
    "candidate_count",
    "same_label_count",
    "mean_iou_other",
    "max_iou_other",
    "max_containment_in_other",
    "max_contains_other",
)


@dataclass
class RankerMatrix:
    features: np.ndarray
    relevance: np.ndarray
    groups: np.ndarray
    candidate_ious: np.ndarray
    feature_names: list[str]
    query_metadata: list[dict[str, Any]]
    empty_query_metadata: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("ranker feature matrix must be two-dimensional")
        rows = self.features.shape[0]
        if rows != len(self.relevance) or rows != len(self.candidate_ious):
            raise ValueError("ranker arrays have inconsistent row counts")
        if int(self.groups.sum()) != rows:
            raise ValueError("ranker group sum does not equal candidate rows")
        if len(self.groups) != len(self.query_metadata):
            raise ValueError("ranker metadata does not match query groups")
        if self.features.shape[1] != len(self.feature_names):
            raise ValueError("ranker feature names do not match matrix width")
        if not np.isfinite(self.features).all():
            raise FloatingPointError("ranker features contain NaN or infinity")
        if not np.isfinite(self.candidate_ious).all():
            raise FloatingPointError("candidate IoUs contain NaN or infinity")


def _metadata_for_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_id": str(record.get("query_id", "")),
        "dataset": str(record.get("dataset", "unknown")),
        "image_key": str(record.get("image_key", "")),
        "query": str(record.get("query", "")),
        "query_category": str(record.get("query_category", "other")),
        "area_bin": str(record.get("area_bin", "unknown")),
        "solvable_at_05": bool(record.get("solvable_at_05", False)),
    }


def build_ranker_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    drop_no_signal: bool = False,
) -> RankerMatrix:
    feature_rows: list[list[float]] = []
    relevance: list[int] = []
    candidate_ious: list[float] = []
    groups: list[int] = []
    metadata: list[dict[str, Any]] = []
    empty_metadata: list[dict[str, Any]] = []

    for record in records:
        rows = build_feature_rows(record)
        record_metadata = _metadata_for_record(record)
        if not rows:
            empty_metadata.append(record_metadata)
            continue
        labels = [int(row["relevance"]) for row in rows]
        if drop_no_signal and len(set(labels)) <= 1:
            continue
        for row in rows:
            feature_rows.append(
                [float(row["features"][name]) for name in FEATURE_NAMES]
            )
            relevance.append(int(row["relevance"]))
            candidate_ious.append(float(row["iou"]))
        groups.append(len(rows))
        metadata.append(record_metadata)

    matrix = RankerMatrix(
        features=np.asarray(feature_rows, dtype=np.float32).reshape(
            (-1, len(FEATURE_NAMES))
        ),
        relevance=np.asarray(relevance, dtype=np.int32),
        groups=np.asarray(groups, dtype=np.int32),
        candidate_ious=np.asarray(candidate_ious, dtype=np.float32),
        feature_names=list(FEATURE_NAMES),
        query_metadata=metadata,
        empty_query_metadata=empty_metadata,
    )
    matrix.validate()
    return matrix


def save_ranker_matrix(matrix: RankerMatrix, output_prefix: Path | str) -> None:
    matrix.validate()
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        features=matrix.features,
        relevance=matrix.relevance,
        groups=matrix.groups,
        candidate_ious=matrix.candidate_ious,
    )
    prefix.with_suffix(".schema.json").write_text(
        json.dumps(
            {
                "feature_names": matrix.feature_names,
                "query_metadata": matrix.query_metadata,
                "empty_query_metadata": matrix.empty_query_metadata,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def load_ranker_matrix(input_prefix: Path | str) -> RankerMatrix:
    prefix = Path(input_prefix)
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as arrays:
        features = arrays["features"]
        relevance = arrays["relevance"]
        groups = arrays["groups"]
        candidate_ious = arrays["candidate_ious"]
    schema = json.loads(
        prefix.with_suffix(".schema.json").read_text(encoding="utf-8")
    )
    matrix = RankerMatrix(
        features=features,
        relevance=relevance,
        groups=groups,
        candidate_ious=candidate_ious,
        feature_names=list(schema["feature_names"]),
        query_metadata=list(schema["query_metadata"]),
        empty_query_metadata=list(schema.get("empty_query_metadata", [])),
    )
    matrix.validate()
    return matrix


def select_matrix_features(
    matrix: RankerMatrix,
    feature_names: Sequence[str],
) -> RankerMatrix:
    indices = []
    for name in feature_names:
        if name not in matrix.feature_names:
            raise ValueError(f"ranker matrix has no feature named {name}")
        indices.append(matrix.feature_names.index(name))
    selected = RankerMatrix(
        features=matrix.features[:, indices],
        relevance=matrix.relevance,
        groups=matrix.groups,
        candidate_ious=matrix.candidate_ious,
        feature_names=list(feature_names),
        query_metadata=matrix.query_metadata,
        empty_query_metadata=matrix.empty_query_metadata,
    )
    selected.validate()
    return selected


def _group_offsets(groups: np.ndarray) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for group in groups:
        next_cursor = cursor + int(group)
        offsets.append((cursor, next_cursor))
        cursor = next_cursor
    return offsets


def select_indices_from_scores(
    matrix: RankerMatrix,
    scores: Sequence[float] | np.ndarray,
    *,
    guard_margin: float,
) -> list[int]:
    matrix.validate()
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (matrix.features.shape[0],):
        raise ValueError("ranker score count does not match candidate rows")
    if not np.isfinite(values).all():
        raise FloatingPointError("ranker scores contain NaN or infinity")
    if guard_margin < 0.0:
        raise ValueError("guard_margin must be non-negative")
    selected: list[int] = []
    for start, end in _group_offsets(matrix.groups):
        group_scores = values[start:end]
        best = int(np.argmax(group_scores))
        if best != 0 and (
            float(group_scores[best]) - float(group_scores[0])
            < float(guard_margin)
        ):
            best = 0
        selected.append(best)
    return selected


def _summarize_outcomes(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(outcomes)
    if not count:
        return {
            "queries": 0,
            "baseline_acc_at_05": 0.0,
            "selected_acc_at_05": 0.0,
            "oracle_acc_at_05": 0.0,
            "baseline_mean_iou": 0.0,
            "selected_mean_iou": 0.0,
            "oracle_mean_iou": 0.0,
            "switch_rate": 0.0,
            "rescued": 0,
            "harmed": 0,
        }
    return {
        "queries": count,
        "baseline_acc_at_05": sum(
            outcome["baseline_iou"] >= 0.5 for outcome in outcomes
        )
        / count,
        "selected_acc_at_05": sum(
            outcome["selected_iou"] >= 0.5 for outcome in outcomes
        )
        / count,
        "oracle_acc_at_05": sum(
            outcome["oracle_iou"] >= 0.5 for outcome in outcomes
        )
        / count,
        "baseline_mean_iou": sum(
            float(outcome["baseline_iou"]) for outcome in outcomes
        )
        / count,
        "selected_mean_iou": sum(
            float(outcome["selected_iou"]) for outcome in outcomes
        )
        / count,
        "oracle_mean_iou": sum(
            float(outcome["oracle_iou"]) for outcome in outcomes
        )
        / count,
        "switch_rate": sum(bool(outcome["switched"]) for outcome in outcomes)
        / count,
        "rescued": sum(bool(outcome["rescued"]) for outcome in outcomes),
        "harmed": sum(bool(outcome["harmed"]) for outcome in outcomes),
    }


def evaluate_ranker_scores(
    matrix: RankerMatrix,
    scores: Sequence[float] | np.ndarray,
    *,
    guard_margin: float,
) -> dict[str, Any]:
    selected_indices = select_indices_from_scores(
        matrix, scores, guard_margin=guard_margin
    )
    outcomes: list[dict[str, Any]] = []
    for group_index, ((start, end), selected) in enumerate(
        zip(_group_offsets(matrix.groups), selected_indices)
    ):
        ious = matrix.candidate_ious[start:end]
        baseline_iou = float(ious[0])
        selected_iou = float(ious[selected])
        oracle_iou = float(np.max(ious))
        metadata = matrix.query_metadata[group_index]
        outcomes.append(
            {
                **metadata,
                "baseline_iou": baseline_iou,
                "selected_iou": selected_iou,
                "oracle_iou": oracle_iou,
                "selected_index": selected,
                "switched": selected != 0,
                "rescued": baseline_iou < 0.5 <= selected_iou,
                "harmed": baseline_iou >= 0.5 > selected_iou,
            }
        )
    for metadata in matrix.empty_query_metadata:
        outcomes.append(
            {
                **metadata,
                "baseline_iou": 0.0,
                "selected_iou": 0.0,
                "oracle_iou": 0.0,
                "selected_index": None,
                "switched": False,
                "rescued": False,
                "harmed": False,
            }
        )

    grouped: dict[str, dict[str, Any]] = {}
    for field_name in ("dataset", "query_category", "area_bin"):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for outcome in outcomes:
            buckets[str(outcome.get(field_name, "unknown"))].append(outcome)
        grouped[field_name] = {
            key: _summarize_outcomes(values)
            for key, values in sorted(buckets.items())
        }
    margins: list[dict[str, float]] = []
    values = np.asarray(scores, dtype=np.float64)
    for start, end in _group_offsets(matrix.groups):
        group_scores = values[start:end]
        order = np.argsort(-group_scores)
        best = float(group_scores[order[0]])
        second = (
            float(group_scores[order[1]]) if len(order) > 1 else best
        )
        margins.append(
            {
                "best_minus_second": best - second,
                "best_minus_baseline": best - float(group_scores[0]),
            }
        )
    return {
        "overall": _summarize_outcomes(outcomes),
        "grouped": grouped,
        "outcomes": outcomes,
        "score_margins": margins,
        "guard_margin": float(guard_margin),
    }


def conservative_rule_scores(matrix: RankerMatrix) -> np.ndarray:
    """A fixed, label-aware geometry rule used only as an ablation."""

    index = {name: matrix.feature_names.index(name) for name in matrix.feature_names}
    required = {
        "gdino_score",
        "target_compatible",
        "absolute_relation_satisfaction",
        "size_relation_satisfaction",
        "ordinal_satisfaction",
        "relative_relation_satisfaction",
        "flag_absolute_relation",
        "flag_size_relation",
        "flag_ordinal_relation",
        "flag_relative_relation",
    }
    missing = required - set(index)
    if missing:
        raise ValueError(f"rule ablation is missing features: {sorted(missing)}")
    scores = matrix.features[:, index["gdino_score"]].astype(np.float64).copy()
    for start, end in _group_offsets(matrix.groups):
        group = matrix.features[start:end]
        flags = {
            name: float(group[0, index[name]])
            for name in (
                "flag_absolute_relation",
                "flag_size_relation",
                "flag_ordinal_relation",
                "flag_relative_relation",
            )
        }
        if not any(flags.values()):
            continue
        target = group[:, index["target_compatible"]]
        relation_bonus = (
            flags["flag_absolute_relation"]
            * group[:, index["absolute_relation_satisfaction"]]
            + flags["flag_size_relation"]
            * group[:, index["size_relation_satisfaction"]]
            + flags["flag_ordinal_relation"]
            * group[:, index["ordinal_satisfaction"]]
            + flags["flag_relative_relation"]
            * group[:, index["relative_relation_satisfaction"]]
        )
        scores[start:end] += 0.20 * target + 0.35 * relation_bonus
    return scores


def save_ranker_model(model: Any, path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    booster = model.booster_ if hasattr(model, "booster_") else model
    booster.save_model(str(destination))


def load_ranker_model(path: Path | str) -> lgb.Booster:
    return lgb.Booster(model_file=str(Path(path)))


def _fit_one_ranker(
    train: RankerMatrix,
    dev: RankerMatrix,
    *,
    params: Mapping[str, Any],
) -> lgb.LGBMRanker:
    train.validate()
    dev.validate()
    if not len(train.groups) or not len(dev.groups):
        raise ValueError("ranker train/dev matrices must contain query groups")
    parameters = dict(params)
    early_stopping_rounds = int(parameters.pop("early_stopping_rounds", 100))
    eval_at = list(parameters.pop("eval_at", [1, 3, 5]))
    model = lgb.LGBMRanker(**parameters)
    model.fit(
        train.features,
        train.relevance,
        group=train.groups,
        feature_name=train.feature_names,
        eval_set=[(dev.features, dev.relevance)],
        eval_group=[dev.groups],
        eval_at=eval_at,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def train_ranker_grid(
    *,
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    base_params: Mapping[str, Any],
    parameter_grid: Sequence[Mapping[str, Any]],
    guard_margins: Sequence[float],
    output_dir: Path | str,
) -> dict[str, Any]:
    """Train the fixed small grid and select by ACC, IoU, then simplicity."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = build_ranker_matrix(train_records, drop_no_signal=True)
    dev = build_ranker_matrix(dev_records, drop_no_signal=False)
    save_ranker_matrix(train, output / "rank_train")
    save_ranker_matrix(dev, output / "rank_dev")
    results: list[dict[str, Any]] = []
    models: list[lgb.LGBMRanker] = []
    for grid_index, override in enumerate(parameter_grid):
        params = {**dict(base_params), **dict(override)}
        model = _fit_one_ranker(train, dev, params=params)
        scores = np.asarray(model.predict(dev.features), dtype=np.float64)
        best_for_model: dict[str, Any] | None = None
        for guard_margin in guard_margins:
            evaluation = evaluate_ranker_scores(
                dev, scores, guard_margin=float(guard_margin)
            )
            candidate = {
                "grid_index": grid_index,
                "params": params,
                "guard_margin": float(guard_margin),
                "best_iteration": int(
                    model.best_iteration_ or params.get("n_estimators", 0)
                ),
                "evaluation": evaluation,
            }
            if best_for_model is None or _selection_key(candidate) > _selection_key(
                best_for_model
            ):
                best_for_model = candidate
        assert best_for_model is not None
        model_path = output / f"grid_{grid_index}.txt"
        save_ranker_model(model, model_path)
        best_for_model["model_path"] = str(model_path)
        results.append(best_for_model)
        models.append(model)

    selected = max(results, key=_selection_key)
    selected_index = int(selected["grid_index"])
    selected_path = output / "selected_pilot_ranker.txt"
    save_ranker_model(models[selected_index], selected_path)
    payload = {
        "selected": {
            **selected,
            "model_path": str(selected_path),
        },
        "grid_results": results,
        "train_groups": int(len(train.groups)),
        "train_candidate_rows": int(train.features.shape[0]),
        "dev_groups": int(len(dev.groups)),
        "dev_candidate_rows": int(dev.features.shape[0]),
    }
    (output / "grid_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _selection_key(result: Mapping[str, Any]) -> tuple[float, float, int, int]:
    overall = result["evaluation"]["overall"]
    params = result["params"]
    return (
        float(overall["selected_acc_at_05"]),
        float(overall["selected_mean_iou"]),
        -int(params.get("num_leaves", 31)),
        -int(result.get("best_iteration", params.get("n_estimators", 0))),
    )


def train_final_ranker(
    *,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    best_iteration: int,
    output_model: Path | str,
) -> dict[str, Any]:
    matrix = build_ranker_matrix(records, drop_no_signal=True)
    parameters = dict(params)
    parameters.pop("early_stopping_rounds", None)
    parameters.pop("eval_at", None)
    parameters["n_estimators"] = max(1, int(best_iteration))
    model = lgb.LGBMRanker(**parameters)
    model.fit(
        matrix.features,
        matrix.relevance,
        group=matrix.groups,
        feature_name=matrix.feature_names,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    save_ranker_model(model, output_model)
    importance = {
        name: float(value)
        for name, value in sorted(
            zip(
                matrix.feature_names,
                model.booster_.feature_importance(importance_type="gain"),
            ),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    return {
        "model_path": str(Path(output_model)),
        "training_groups": int(len(matrix.groups)),
        "training_candidate_rows": int(matrix.features.shape[0]),
        "dropped_empty_queries": len(matrix.empty_query_metadata),
        "n_estimators": int(parameters["n_estimators"]),
        "feature_importance_gain": importance,
    }


def train_fixed_ranker(
    *,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    n_estimators: int,
    output_model: Path | str,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> dict[str, Any]:
    """Train a deterministic fixed-iteration ablation/final ranker."""

    full = build_ranker_matrix(records, drop_no_signal=True)
    matrix = select_matrix_features(full, feature_names)
    parameters = dict(params)
    parameters.pop("early_stopping_rounds", None)
    parameters.pop("eval_at", None)
    parameters["n_estimators"] = max(1, int(n_estimators))
    model = lgb.LGBMRanker(**parameters)
    model.fit(
        matrix.features,
        matrix.relevance,
        group=matrix.groups,
        feature_name=matrix.feature_names,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    save_ranker_model(model, output_model)
    importance = {
        name: float(value)
        for name, value in sorted(
            zip(
                matrix.feature_names,
                model.booster_.feature_importance(importance_type="gain"),
            ),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    return {
        "model_path": str(Path(output_model)),
        "training_groups": int(len(matrix.groups)),
        "training_candidate_rows": int(matrix.features.shape[0]),
        "n_estimators": int(parameters["n_estimators"]),
        "feature_names": matrix.feature_names,
        "feature_importance_gain": importance,
    }
