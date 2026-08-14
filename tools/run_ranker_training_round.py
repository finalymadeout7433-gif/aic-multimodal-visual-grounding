from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightgbm
import numpy as np
import torch
import transformers
import yaml
from PIL import __version__ as PILLOW_VERSION

from aic_baseline.diagnostics import query_category, read_jsonl, write_jsonl
from aic_baseline.external_data import sha256_file
from aic_baseline.model_adapters import GroundingDinoExternalPredictor
from aic_baseline.ranker_cache import (
    read_candidate_cache,
    run_aic_candidate_cache,
    run_external_candidate_cache,
)
from aic_baseline.ranker_features import FEATURE_NAMES
from aic_baseline.ranker_inference import (
    load_fallback_predictions,
    select_aic_predictions,
    write_aic_control_and_ranker_submissions,
)
from aic_baseline.ranker_subsets import (
    build_evaluation_subset,
    build_nested_ranker_subsets,
    split_by_image,
)
from aic_baseline.ranker_training import (
    SCORE_GEOMETRY_FEATURE_NAMES,
    build_ranker_matrix,
    conservative_rule_scores,
    evaluate_ranker_scores,
    load_ranker_model,
    select_matrix_features,
    train_fixed_ranker,
    train_ranker_grid,
)


PHASES = (
    "subsets",
    "cache-train",
    "train-pilot",
    "cache-train-full",
    "train-final",
    "cache-validation",
    "evaluate-validation",
    "cache-holdout",
    "evaluate-holdout",
    "cache-aic",
    "submissions",
    "summary",
    "all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the staged GroundingDINO spatial LTR training round."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--allow-holdout", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ranker config must be a YAML object")
    return payload


def _required_path(value: Any, *, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"config path is required: {name}")
    return Path(str(value)).resolve()


def _manifest_path(data_root: Path, value: Any, *, name: str) -> Path:
    path = _required_path(value, name=name)
    if path.is_file():
        return path
    candidate = data_root / str(value)
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"manifest does not exist: {name}={value}")


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    values = config["paths"]
    data_root = _required_path(values["data_root"], name="data_root")
    return {
        "data_root": data_root,
        "train_manifest": _manifest_path(
            data_root, values["train_manifest"], name="train_manifest"
        ),
        "validation_manifest": _manifest_path(
            data_root,
            values["validation_manifest"],
            name="validation_manifest",
        ),
        "holdout_manifest": _manifest_path(
            data_root, values["holdout_manifest"], name="holdout_manifest"
        ),
        "aic_dataset_root": _required_path(
            values["aic_dataset_root"], name="aic_dataset_root"
        ),
        "aic_queries": _required_path(
            values["aic_queries"], name="aic_queries"
        ),
        "grounding_dino_model": _required_path(
            values["grounding_dino_model"], name="grounding_dino_model"
        ),
        "florence_fallback_predictions": _required_path(
            values["florence_fallback_predictions"],
            name="florence_fallback_predictions",
        ),
        "output_root": _required_path(
            values["output_root"], name="output_root"
        ),
    }


def _code_sha256(repo_root: Path) -> str:
    """Hash the complete Python dependency closure used by this round.

    A candidate cache is only reusable when both the model snapshot and every
    local module that can alter preprocessing, inference, normalization, or
    deduplication are unchanged.  Hashing the package is deliberately broader
    than trying to maintain a fragile hand-written import list.
    """

    digest = hashlib.sha256()
    package_root = repo_root / "src/aic_baseline"
    paths = sorted(package_root.glob("*.py")) + [Path(__file__).resolve()]
    for path in paths:
        relative = (
            path.relative_to(repo_root).as_posix()
            if path.is_relative_to(repo_root)
            else path.name
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest().upper()


def _ranker_semantics_sha256(repo_root: Path) -> str:
    """Hash feature extraction, ranking, evaluation, and selection semantics."""

    digest = hashlib.sha256()
    paths = [
        repo_root / "src/aic_baseline/bbox.py",
        repo_root / "src/aic_baseline/ranker_features.py",
        repo_root / "src/aic_baseline/ranker_training.py",
        repo_root / "src/aic_baseline/ranker_inference.py",
        Path(__file__).resolve(),
    ]
    for path in paths:
        relative = (
            path.relative_to(repo_root).as_posix()
            if path.is_relative_to(repo_root)
            else path.name
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest().upper()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _model_artifact_manifest(model_dir: Path) -> dict[str, str]:
    """Hash every local model artifact that can affect Transformers loading."""

    artifacts: dict[str, str] = {}
    for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
        if path.suffix.lower() in {".lock", ".tmp"}:
            continue
        relative = path.relative_to(model_dir).as_posix()
        artifacts[relative] = sha256_file(path)
    if not artifacts:
        raise FileNotFoundError(
            f"GroundingDINO model directory is empty: {model_dir}"
        )
    return artifacts


def _write_environment(output_root: Path) -> dict[str, Any]:
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "lightgbm": lightgbm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "cuda_total_memory_bytes": (
            int(torch.cuda.get_device_properties(0).total_memory)
            if torch.cuda.is_available()
            else 0
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return environment


def _update_run_summary(
    output_root: Path,
    *,
    phase: str,
    status: str,
    details: Mapping[str, Any],
) -> None:
    path = output_root / "run_summary.json"
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"schema_version": 1, "phases": {}}
    )
    payload["phases"][phase] = {
        "status": status,
        "timestamp_unix": time.time(),
        "details": dict(details),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _subset_paths(output_root: Path) -> dict[str, Path]:
    root = output_root / "subsets"
    return {
        "pilot": root / "rank_train_pilot_20000.jsonl",
        "full": root / "rank_train_full_50000.jsonl",
        "validation": root / "rank_validation_10000.jsonl",
        "holdout": root / "rank_holdout_10000.jsonl",
        "summary": root / "subset_summary.json",
    }


def _assert_image_disjoint(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    names: str,
) -> None:
    left_images = {str(record["image_key"]) for record in left}
    right_images = {str(record["image_key"]) for record in right}
    overlap = left_images & right_images
    if overlap:
        raise ValueError(
            f"image-key leakage between {names}: {len(overlap)} images"
        )


def phase_subsets(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    include_holdout: bool,
) -> dict[str, Any]:
    subset_config = config["subsets"]
    destination = _subset_paths(paths["output_root"])
    destination["pilot"].parent.mkdir(parents=True, exist_ok=True)
    train = read_jsonl(paths["train_manifest"])
    validation_source = read_jsonl(paths["validation_manifest"])
    nested = build_nested_ranker_subsets(
        train,
        pilot_size=int(subset_config["pilot_size"]),
        full_size=int(subset_config["full_size"]),
        seed=int(config["seed"]),
        pilot_image_cap=int(subset_config["pilot_image_cap"]),
        full_image_cap=int(subset_config["full_image_cap"]),
    )
    validation, validation_summary = build_evaluation_subset(
        validation_source,
        size=int(subset_config["validation_size"]),
        seed=int(config["seed"]),
        name="rank_validation",
        image_cap=int(subset_config["evaluation_image_cap"]),
    )
    _assert_image_disjoint(nested["full"], validation, names="train/validation")
    write_jsonl(destination["pilot"], nested["pilot"])
    write_jsonl(destination["full"], nested["full"])
    write_jsonl(destination["validation"], validation)
    summary: dict[str, Any] = {
        **nested["summary"],
        "validation": validation_summary,
        "source_manifests": {
            "train_sha256": sha256_file(paths["train_manifest"]),
            "validation_sha256": sha256_file(paths["validation_manifest"]),
        },
    }
    if include_holdout:
        holdout_source = read_jsonl(paths["holdout_manifest"])
        holdout, holdout_summary = build_evaluation_subset(
            holdout_source,
            size=int(subset_config["holdout_size"]),
            seed=int(config["seed"]),
            name="rank_holdout",
            image_cap=int(subset_config["evaluation_image_cap"]),
        )
        _assert_image_disjoint(nested["full"], holdout, names="train/holdout")
        _assert_image_disjoint(validation, holdout, names="validation/holdout")
        write_jsonl(destination["holdout"], holdout)
        summary["holdout"] = holdout_summary
        summary["source_manifests"]["holdout_sha256"] = sha256_file(
            paths["holdout_manifest"]
        )
    for name in ("pilot", "full", "validation", "holdout"):
        if destination[name].exists():
            summary.setdefault(name, {})["file_sha256"] = sha256_file(
                destination[name]
            )
    destination["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_predictor(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> GroundingDinoExternalPredictor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GroundingDINO caching")
    settings = config["grounding_dino"]
    dtype_name = str(settings["dtype"])
    if dtype_name != "float32":
        raise ValueError("this validated GroundingDINO snapshot requires float32")
    return GroundingDinoExternalPredictor(
        model_path=paths["grounding_dino_model"],
        device="cuda",
        dtype=torch.float32,
        box_threshold=float(settings["box_threshold"]),
        text_threshold=float(settings["text_threshold"]),
        max_candidates=int(settings["max_candidates"]),
    )


def _candidate_fingerprint(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    source_name: str,
    source_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    settings = config["grounding_dino"]
    model_artifacts = _model_artifact_manifest(paths["grounding_dino_model"])
    return {
        "model": "IDEA-Research/grounding-dino-tiny",
        "revision": str(settings["revision"]),
        "model_artifacts": model_artifacts,
        "model_artifacts_sha256": _canonical_sha256(model_artifacts),
        f"{source_name}_sha256": sha256_file(source_path),
        "dtype": str(settings["dtype"]),
        "box_threshold": float(settings["box_threshold"]),
        "text_threshold": float(settings["text_threshold"]),
        "max_candidates": int(settings["max_candidates"]),
        "dedup_iou": float(settings["dedup_iou"]),
        "query_normalization": "lowercase_with_terminal_period",
        "runtime_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pillow": PILLOW_VERSION,
            "cuda": torch.version.cuda,
        },
        "code_sha256": _code_sha256(repo_root),
    }


def _progress(completed: int, total: int, record: dict[str, Any]) -> None:
    if completed <= 10 or completed % 100 == 0 or completed == total:
        print(
            f"[ranker-cache] {completed}/{total} "
            f"last={record['query_id']}",
            flush=True,
        )


def phase_cache_external(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    subset_name: str,
    cache_name: str,
    resume: bool,
    stop_after: int | None,
    repo_root: Path,
) -> dict[str, Any]:
    subset_path = _subset_paths(paths["output_root"])[subset_name]
    records = read_jsonl(subset_path)
    predictor = _build_predictor(config, paths)
    return run_external_candidate_cache(
        records=records,
        data_root=paths["data_root"],
        predictor=predictor,
        output_dir=paths["output_root"] / "candidate_cache" / cache_name,
        run_fingerprint=_candidate_fingerprint(
            config,
            paths,
            source_name="subset",
            source_path=subset_path,
            repo_root=repo_root,
        ),
        resume=resume,
        dedup_iou=float(config["grounding_dino"]["dedup_iou"]),
        stop_after=stop_after,
        progress_callback=_progress,
    )


def _ranker_base_params(config: Mapping[str, Any]) -> dict[str, Any]:
    ranker = config["ranker"]
    excluded = {"grid", "guard_margins"}
    return {key: value for key, value in ranker.items() if key not in excluded}


def phase_train_pilot(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    cache_path = (
        paths["output_root"]
        / "candidate_cache"
        / "train_full"
        / "candidates.jsonl"
    )
    records = read_candidate_cache(cache_path)
    pilot_size = int(config["subsets"]["pilot_size"])
    if len(records) < pilot_size:
        raise ValueError(
            f"pilot needs {pilot_size} cached records; found {len(records)}"
        )
    train, dev = split_by_image(
        records[:pilot_size],
        dev_fraction=float(config["subsets"]["dev_fraction"]),
        seed=int(config["seed"]),
    )
    grid = []
    for item in config["ranker"]["grid"]:
        copied = dict(item)
        copied.pop("name", None)
        grid.append(copied)
    result = train_ranker_grid(
        train_records=train,
        dev_records=dev,
        base_params=_ranker_base_params(config),
        parameter_grid=grid,
        guard_margins=[float(v) for v in config["ranker"]["guard_margins"]],
        output_dir=paths["output_root"] / "models" / "pilot_grid",
    )
    result["train_query_ids_sha256"] = hashlib.sha256(
        "\n".join(str(record["query_id"]) for record in train).encode("utf-8")
    ).hexdigest().upper()
    result["dev_query_ids_sha256"] = hashlib.sha256(
        "\n".join(str(record["query_id"]) for record in dev).encode("utf-8")
    ).hexdigest().upper()
    return result


def phase_train_final(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    pilot_result_path = (
        paths["output_root"] / "models/pilot_grid/grid_results.json"
    )
    pilot = json.loads(pilot_result_path.read_text(encoding="utf-8"))
    selected = pilot["selected"]
    records = read_candidate_cache(
        paths["output_root"] / "candidate_cache/train_full/candidates.jsonl"
    )
    full_size = int(config["subsets"]["full_size"])
    if len(records) != full_size:
        raise ValueError(
            f"final training requires {full_size} cached records; "
            f"found {len(records)}"
        )
    model_dir = paths["output_root"] / "models"
    full_result = train_fixed_ranker(
        records=records,
        params=selected["params"],
        n_estimators=int(selected["best_iteration"]),
        output_model=model_dir / "gdino_spatial_ltr_v1.txt",
        feature_names=FEATURE_NAMES,
    )
    geometry_result = train_fixed_ranker(
        records=records,
        params=selected["params"],
        n_estimators=int(selected["best_iteration"]),
        output_model=model_dir / "gdino_score_geometry_ablation.txt",
        feature_names=SCORE_GEOMETRY_FEATURE_NAMES,
    )
    result = {
        "full_ranker": full_result,
        "score_geometry_ranker": geometry_result,
        "guard_margin": float(selected["guard_margin"]),
        "pilot_selection": selected,
    }
    (model_dir / "frozen_model.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def _freeze_identity(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    """Build the immutable identity required before holdout can be opened."""

    output = paths["output_root"]
    frozen_model_path = output / "models/frozen_model.json"
    frozen_model = json.loads(frozen_model_path.read_text(encoding="utf-8"))
    feature_names = frozen_model["full_ranker"]["feature_names"]
    validation_fingerprint = (
        output / "candidate_cache/validation/run_fingerprint.json"
    )
    identity = {
        "ranker_model_sha256": sha256_file(
            output / "models/gdino_spatial_ltr_v1.txt"
        ),
        "geometry_model_sha256": sha256_file(
            output / "models/gdino_score_geometry_ablation.txt"
        ),
        "frozen_model_sha256": sha256_file(frozen_model_path),
        "feature_schema_sha256": _canonical_sha256(feature_names),
        "ranker_config_sha256": _canonical_sha256(config["ranker"]),
        "grounding_dino_config_sha256": _canonical_sha256(
            config["grounding_dino"]
        ),
        "subset_config_sha256": _canonical_sha256(config["subsets"]),
        "promotion_config_sha256": _canonical_sha256(config["promotion"]),
        "ranker_semantics_code_sha256": _ranker_semantics_sha256(
            Path(__file__).resolve().parents[1]
        ),
        "validation_cache_fingerprint_sha256": sha256_file(
            validation_fingerprint
        ),
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _candidate_health(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    total = 0
    legal = 0
    unexpected_errors = 0
    for record in records:
        if record.get("error") not in {None, "no_candidate"}:
            unexpected_errors += 1
        for candidate in record.get("candidates", []):
            total += 1
            try:
                x1, y1, x2, y2 = [
                    float(value) for value in candidate["bbox"]
                ]
            except (KeyError, TypeError, ValueError):
                continue
            if (
                all(math.isfinite(value) for value in (x1, y1, x2, y2))
                and 0.0 <= x1 < x2 <= 1.0
                and 0.0 <= y1 < y2 <= 1.0
            ):
                legal += 1
    return {
        "candidate_count": total,
        "legal_candidate_count": legal,
        "candidate_bbox_legal_rate": legal / total if total else 1.0,
        "unexpected_error_count": unexpected_errors,
        "silent_fallback_count": 0,
    }


def _validate_frozen_decision(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    decision_path = (
        paths["output_root"]
        / "evaluations/validation/frozen_decision.json"
    )
    if not decision_path.exists():
        raise RuntimeError("validation model has not been frozen")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not bool(decision.get("validation_promotion_pass")):
        raise RuntimeError("validation promotion failed; holdout remains sealed")
    expected = _freeze_identity(config, paths)
    recorded = decision.get("freeze_identity")
    if recorded != expected:
        raise RuntimeError(
            "frozen model/config/feature identity changed after validation"
        )
    if decision.get("freeze_identity_sha256") != expected["identity_sha256"]:
        raise RuntimeError("frozen decision identity digest is inconsistent")
    return decision


def _holdout_ledger_payload(
    paths: Mapping[str, Path], decision: Mapping[str, Any]
) -> dict[str, Any]:
    output = paths["output_root"]
    return {
        "schema_version": 1,
        "freeze_identity_sha256": decision["freeze_identity_sha256"],
        "ranker_model_sha256": decision["model_sha256"],
        "holdout_cache_fingerprint_sha256": sha256_file(
            output / "candidate_cache/holdout/run_fingerprint.json"
        ),
        "holdout_evaluation_sha256": sha256_file(
            output / "evaluations/holdout/evaluation.json"
        ),
    }


def _verify_or_create_holdout_ledger(
    paths: Mapping[str, Path],
    decision: Mapping[str, Any],
    *,
    allow_legacy_backfill: bool,
) -> dict[str, Any]:
    output = paths["output_root"]
    ledger_path = output / "evaluations/holdout/evaluation_provenance.json"
    expected = _holdout_ledger_payload(paths, decision)
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if ledger.get(key) != value:
                raise RuntimeError(f"holdout provenance mismatch: {key}")
        return ledger
    if allow_legacy_backfill:
        manifest_path = output / "sha256_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("cannot bind legacy holdout without artifact manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "models/gdino_spatial_ltr_v1.txt": expected[
                "ranker_model_sha256"
            ],
            "evaluations/holdout/evaluation.json": expected[
                "holdout_evaluation_sha256"
            ],
        }
        for relative, digest in required.items():
            if manifest.get(relative, {}).get("sha256") != digest:
                raise RuntimeError(
                    f"legacy holdout is not bound by the completed manifest: {relative}"
                )
    ledger = {
        **expected,
        "legacy_manifest_backfill": bool(allow_legacy_backfill),
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return ledger


def _combined_category_metrics(
    outcomes: Sequence[Mapping[str, Any]], categories: set[str]
) -> dict[str, Any]:
    selected = [
        outcome
        for outcome in outcomes
        if str(outcome.get("query_category")) in categories
    ]
    count = len(selected)
    baseline = (
        sum(float(outcome["baseline_iou"]) >= 0.5 for outcome in selected)
        / count
        if count
        else 0.0
    )
    ranked = (
        sum(float(outcome["selected_iou"]) >= 0.5 for outcome in selected)
        / count
        if count
        else 0.0
    )
    return {
        "queries": count,
        "baseline_acc_at_05": baseline,
        "selected_acc_at_05": ranked,
        "gain": ranked - baseline,
    }


def _margin_buckets(
    evaluation: Mapping[str, Any],
) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[bool]] = {
        "lt_0.05": [],
        "0.05_to_0.10": [],
        "0.10_to_0.20": [],
        "0.20_to_0.30": [],
        "ge_0.30": [],
    }
    for outcome, margin in zip(
        evaluation["outcomes"], evaluation["score_margins"]
    ):
        value = float(margin["best_minus_baseline"])
        if value < 0.05:
            key = "lt_0.05"
        elif value < 0.10:
            key = "0.05_to_0.10"
        elif value < 0.20:
            key = "0.10_to_0.20"
        elif value < 0.30:
            key = "0.20_to_0.30"
        else:
            key = "ge_0.30"
        buckets[key].append(float(outcome["selected_iou"]) >= 0.5)
    return {
        key: {
            "queries": len(values),
            "selected_acc_at_05": sum(values) / len(values) if values else 0.0,
        }
        for key, values in buckets.items()
    }


def phase_evaluate(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    split: str,
) -> dict[str, Any]:
    cache_path = (
        paths["output_root"]
        / "candidate_cache"
        / split
        / "candidates.jsonl"
    )
    records = read_candidate_cache(cache_path)
    expected = int(config["subsets"][f"{split}_size"])
    if len(records) != expected:
        raise ValueError(
            f"{split} evaluation needs {expected} records; found {len(records)}"
        )
    matrix = build_ranker_matrix(records, drop_no_signal=False)
    full_model = load_ranker_model(
        paths["output_root"] / "models/gdino_spatial_ltr_v1.txt"
    )
    geometry_model = load_ranker_model(
        paths["output_root"] / "models/gdino_score_geometry_ablation.txt"
    )
    frozen = json.loads(
        (paths["output_root"] / "models/frozen_model.json").read_text(
            encoding="utf-8"
        )
    )
    guard_margin = float(frozen["guard_margin"])
    full_scores = np.asarray(full_model.predict(matrix.features), dtype=np.float64)
    geometry_matrix = select_matrix_features(
        matrix, SCORE_GEOMETRY_FEATURE_NAMES
    )
    geometry_scores = np.asarray(
        geometry_model.predict(geometry_matrix.features), dtype=np.float64
    )
    top1_scores = np.zeros(matrix.features.shape[0], dtype=np.float64)
    evaluations = {
        "gdino_top1": evaluate_ranker_scores(
            matrix, top1_scores, guard_margin=0.0
        ),
        "conservative_rule": evaluate_ranker_scores(
            matrix, conservative_rule_scores(matrix), guard_margin=0.0
        ),
        "score_geometry_ranker": evaluate_ranker_scores(
            matrix, geometry_scores, guard_margin=guard_margin
        ),
        "full_ranker_no_guard": evaluate_ranker_scores(
            matrix, full_scores, guard_margin=0.0
        ),
        "full_ranker_guarded": evaluate_ranker_scores(
            matrix, full_scores, guard_margin=guard_margin
        ),
    }
    guarded = evaluations["full_ranker_guarded"]
    evaluations["full_ranker_guarded"]["spatial_ordinal_combined"] = (
        _combined_category_metrics(
            guarded["outcomes"], {"spatial", "ordinal"}
        )
    )
    evaluations["full_ranker_guarded"]["margin_buckets"] = _margin_buckets(
        guarded
    )
    output_dir = paths["output_root"] / "evaluations" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation.json"
    output_path.write_text(
        json.dumps(evaluations, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if split == "validation":
        overall = guarded["overall"]
        dataset_drops = []
        for metrics in guarded["grouped"]["dataset"].values():
            dataset_drops.append(
                float(metrics["selected_acc_at_05"])
                - float(metrics["baseline_acc_at_05"])
            )
        promotion = config["promotion"]
        health = _candidate_health(records)
        freeze_identity = _freeze_identity(config, paths)
        decision = {
            "model_sha256": freeze_identity["ranker_model_sha256"],
            "guard_margin": guard_margin,
            "validation_overall_gain": (
                float(overall["selected_acc_at_05"])
                - float(overall["baseline_acc_at_05"])
            ),
            "spatial_ordinal_gain": guarded[
                "spatial_ordinal_combined"
            ]["gain"],
            "worst_dataset_gain": min(dataset_drops, default=0.0),
            "candidate_health": health,
            "freeze_identity": freeze_identity,
            "freeze_identity_sha256": freeze_identity["identity_sha256"],
        }
        decision["validation_promotion_pass"] = bool(
            decision["validation_overall_gain"]
            >= float(promotion["holdout_overall_gain"])
            and decision["spatial_ordinal_gain"]
            >= float(promotion["spatial_ordinal_gain"])
            and decision["worst_dataset_gain"]
            >= -float(promotion["maximum_dataset_drop"])
            and float(health["candidate_bbox_legal_rate"]) == 1.0
            and int(health["unexpected_error_count"]) == 0
            and int(health["silent_fallback_count"]) == 0
        )
        (output_dir / "frozen_decision.json").write_text(
            json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        evaluations["frozen_decision"] = decision
        output_path.write_text(
            json.dumps(evaluations, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    elif split == "holdout":
        decision = _validate_frozen_decision(config, paths)
        _verify_or_create_holdout_ledger(
            paths,
            decision,
            allow_legacy_backfill=False,
        )
    return evaluations


def _aic_records(queries_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    original = json.loads(queries_path.read_text(encoding="utf-8-sig"))
    if not isinstance(original, dict):
        raise ValueError("AIC queries JSON must contain an object")
    records = [
        {
            "query_id": query_id,
            "query": record["query"],
            "visible": record["visible"],
            "image_key": record["visible"],
            "query_category": query_category(str(record["query"])),
        }
        for query_id, record in original.items()
    ]
    return original, records


def phase_cache_aic(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    resume: bool,
    repo_root: Path,
) -> dict[str, Any]:
    _, records = _aic_records(paths["aic_queries"])
    predictor = _build_predictor(config, paths)
    settings = config["grounding_dino"]
    fingerprint = _candidate_fingerprint(
        config,
        paths,
        source_name="queries",
        source_path=paths["aic_queries"],
        repo_root=repo_root,
    )
    return run_aic_candidate_cache(
        records=records,
        dataset_root=paths["aic_dataset_root"],
        predictor=predictor,
        output_dir=paths["output_root"] / "candidate_cache/aic_test",
        run_fingerprint=fingerprint,
        resume=resume,
        dedup_iou=float(settings["dedup_iou"]),
        progress_callback=_progress,
    )


def phase_submissions(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    original, _ = _aic_records(paths["aic_queries"])
    cache_records = read_candidate_cache(
        paths["output_root"] / "candidate_cache/aic_test/candidates.jsonl"
    )
    if len(cache_records) != len(original):
        raise ValueError(
            f"AIC cache incomplete: {len(cache_records)}/{len(original)}"
        )
    frozen = json.loads(
        (paths["output_root"] / "models/frozen_model.json").read_text(
            encoding="utf-8"
        )
    )
    ranker = load_ranker_model(
        paths["output_root"] / "models/gdino_spatial_ltr_v1.txt"
    )
    fallback = load_fallback_predictions(
        paths["florence_fallback_predictions"]
    )
    selection = select_aic_predictions(
        records=cache_records,
        ranker=ranker,
        guard_margin=float(frozen["guard_margin"]),
        fallback_predictions=fallback,
    )
    submission_root = paths["output_root"] / "submissions"
    audit = write_aic_control_and_ranker_submissions(
        original_records=original,
        selection=selection,
        output_root=submission_root,
    )
    debug_path = submission_root / "selection_debug.jsonl"
    write_jsonl(debug_path, selection.debug_records)
    result = {
        "selection_summary": selection.summary,
        "submission_audit": audit,
        "selection_debug_path": str(debug_path),
    }
    (submission_root / "submission_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def phase_summary(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    output = paths["output_root"]
    reproducible_config = {
        key: config[key]
        for key in (
            "seed",
            "grounding_dino",
            "subsets",
            "ranker",
            "promotion",
        )
    }
    config_fingerprint = {
        "schema_version": 1,
        "full_config_sha256": _canonical_sha256(config),
        "reproducible_config_sha256": _canonical_sha256(
            reproducible_config
        ),
        "included_public_sections": list(reproducible_config),
    }
    config_path = output / "configuration/config_fingerprint.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config_fingerprint, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest: dict[str, dict[str, Any]] = {}
    tracked = [
        config_path,
        output / "environment.json",
        output / "verification_summary.json",
        output / "models/gdino_spatial_ltr_v1.txt",
        output / "models/gdino_score_geometry_ablation.txt",
        output / "models/frozen_model.json",
        output / "evaluations/validation/evaluation.json",
        output / "evaluations/validation/frozen_decision.json",
        output / "evaluations/holdout/evaluation.json",
        output / "evaluations/holdout/evaluation_provenance.json",
        output
        / "submissions/S02_gdino_top1_control/predictions_submission.json",
        output
        / "submissions/S02_gdino_top1_control/predictions_submission.zip",
        output
        / "submissions/S03_gdino_spatial_ltr_v1/predictions_submission.json",
        output
        / "submissions/S03_gdino_spatial_ltr_v1/predictions_submission.zip",
    ]
    for path in tracked:
        if path.exists():
            manifest[str(path.relative_to(output)).replace("\\", "/")] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    (output / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "artifact_count": len(manifest),
        "artifacts": manifest,
        "configuration": config_fingerprint,
    }


def _run_phase(
    phase: str,
    *,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    args: argparse.Namespace,
    repo_root: Path,
) -> dict[str, Any]:
    if phase == "subsets":
        return phase_subsets(
            config, paths, include_holdout=args.allow_holdout
        )
    if phase in {"cache-train", "cache-train-full"}:
        stop_after = (
            int(config["subsets"]["pilot_size"])
            if phase == "cache-train" and args.stop_after is None
            else args.stop_after
        )
        return phase_cache_external(
            config,
            paths,
            subset_name="full",
            cache_name="train_full",
            resume=args.resume or phase == "cache-train-full",
            stop_after=stop_after,
            repo_root=repo_root,
        )
    if phase == "train-pilot":
        return phase_train_pilot(config, paths)
    if phase == "train-final":
        return phase_train_final(config, paths)
    if phase == "cache-validation":
        return phase_cache_external(
            config,
            paths,
            subset_name="validation",
            cache_name="validation",
            resume=args.resume,
            stop_after=args.stop_after,
            repo_root=repo_root,
        )
    if phase == "evaluate-validation":
        return phase_evaluate(config, paths, split="validation")
    if phase in {"cache-holdout", "evaluate-holdout"}:
        if not args.allow_holdout:
            raise PermissionError(
                "holdout phase requires --allow-holdout after validation freeze"
            )
        frozen_decision = _validate_frozen_decision(config, paths)
        if phase == "cache-holdout":
            if not _subset_paths(paths["output_root"])["holdout"].exists():
                phase_subsets(config, paths, include_holdout=True)
            return phase_cache_external(
                config,
                paths,
                subset_name="holdout",
                cache_name="holdout",
                resume=args.resume,
                stop_after=args.stop_after,
                repo_root=repo_root,
            )
        holdout_output = (
            paths["output_root"] / "evaluations/holdout/evaluation.json"
        )
        if holdout_output.exists():
            if args.resume:
                _verify_or_create_holdout_ledger(
                    paths,
                    frozen_decision,
                    allow_legacy_backfill=True,
                )
                return json.loads(holdout_output.read_text(encoding="utf-8"))
            raise RuntimeError(
                "holdout was already evaluated; refusing a second evaluation"
            )
        return phase_evaluate(config, paths, split="holdout")
    if phase == "cache-aic":
        return phase_cache_aic(
            config, paths, resume=args.resume, repo_root=repo_root
        )
    if phase == "submissions":
        return phase_submissions(config, paths)
    if phase == "summary":
        return phase_summary(config, paths)
    raise ValueError(f"unknown phase: {phase}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = _paths(config)
    output_root = paths["output_root"]
    repo_root = Path(__file__).resolve().parents[1]
    _write_environment(output_root)
    phases = (
        [
            "subsets",
            "cache-train",
            "train-pilot",
            "cache-train-full",
            "train-final",
            "cache-validation",
            "evaluate-validation",
            "cache-holdout",
            "evaluate-holdout",
            "cache-aic",
            "submissions",
            "summary",
        ]
        if args.phase == "all"
        else [args.phase]
    )
    for phase in phases:
        print(f"[ranker-round] phase={phase} starting", flush=True)
        started = time.perf_counter()
        try:
            details = _run_phase(
                phase,
                config=config,
                paths=paths,
                args=args,
                repo_root=repo_root,
            )
        except Exception as exc:
            _update_run_summary(
                output_root,
                phase=phase,
                status="failed",
                details={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "wall_seconds": time.perf_counter() - started,
                },
            )
            raise
        _update_run_summary(
            output_root,
            phase=phase,
            status="completed",
            details={
                "wall_seconds": time.perf_counter() - started,
                "result": details,
            },
        )
        print(
            f"[ranker-round] phase={phase} completed in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
