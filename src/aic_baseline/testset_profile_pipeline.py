from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from .external_data import sha256_file
from .testset_profile import (
    build_model_behavior_record,
    classify_small_target_proxy,
    parse_query_profile,
    profile_image_group,
    select_scale_probe,
    summarize_external_domains,
)


PROFILE_STAGES = (
    "verify-inputs",
    "profile-queries",
    "profile-modalities",
    "profile-existing-models",
    "build-scale-proxies",
    "run-scale-probe",
    "compare-external-domains",
    "build-trigger-matrix",
    "build-report",
)

REPORT_NAMES = (
    "aic_testset_full_profile.md",
    "paper_model_to_aic_decision.md",
    "model_disagreement_and_scale.md",
    "domain_shift_analysis.md",
    "optimization_trigger_matrix.md",
    "next_three_experiments.md",
)

FROZEN_TRIGGER_THRESHOLDS = {
    "piza_high_small_ratio": 0.15,
    "piza_high_plus_medium_ratio": 0.30,
    "piza_scale_gain_pp": 10.0,
    "ape_region_ratio": 0.10,
    "ape_region_all_disagree_ratio": 0.40,
    "rgbt_low_light_ratio": 0.15,
    "rgbt_usable_ir_ratio": 0.80,
    "depth_relation_ratio": 0.15,
    "depth_rgb_stability_ratio": 0.50,
}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def resolve_frozen_trigger_thresholds(config: Mapping[str, Any]) -> dict[str, float]:
    """Reject post-hoc threshold drift while allowing omitted default values."""

    configured = config.get("frozen_trigger_thresholds")
    if configured is None:
        return dict(FROZEN_TRIGGER_THRESHOLDS)
    if not isinstance(configured, Mapping):
        raise TypeError("frozen_trigger_thresholds must be a mapping")
    normalized = {str(key): float(value) for key, value in configured.items()}
    if normalized != FROZEN_TRIGGER_THRESHOLDS:
        raise ValueError(
            "frozen_trigger_thresholds differ from the pre-S04 policy; "
            f"expected {FROZEN_TRIGGER_THRESHOLDS}, got {normalized}"
        )
    return normalized


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _resolve_path(value: Any, repo_root: Path, *, required: bool = True) -> Path | None:
    if value is None or not str(value).strip():
        if required:
            raise ValueError("required config path is empty")
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def _prediction_boxes(path: Path) -> dict[str, list[float]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"prediction file must be a JSON object: {path}")
    result: dict[str, list[float]] = {}
    for query_id, record in payload.items():
        box = record.get("bbox") if isinstance(record, dict) else record
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"invalid bbox for {query_id} in {path}")
        result[str(query_id)] = [float(value) for value in box]
    return result


def _records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in _load_jsonl(path):
        query_id = str(record.get("query_id", ""))
        if not query_id or query_id in result:
            raise ValueError(f"missing or duplicate query_id in {path}: {query_id!r}")
        result[query_id] = record
    return result


def _assert_exact_ids(name: str, expected: set[str], actual: set[str]) -> None:
    if actual == expected:
        return
    missing = sorted(expected - actual)[:5]
    extra = sorted(actual - expected)[:5]
    raise ValueError(
        f"{name} Query ID set mismatch: missing={missing}, extra={extra}, "
        f"expected_count={len(expected)}, actual_count={len(actual)}"
    )


def _hash_file_tree(files: Sequence[Path]) -> tuple[str, dict[str, str]]:
    hashes = {str(path): sha256_file(path).upper() for path in sorted(files)}
    digest = hashlib.sha256()
    for path, value in hashes.items():
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper(), hashes


def _query_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("queries.json must contain an object keyed by Query ID")
    return {str(key): dict(value) for key, value in payload.items()}


def _image_paths(
    queries: Mapping[str, Mapping[str, Any]], dataset_root: Path
) -> tuple[dict[str, tuple[Path, Path, Path]], list[Path]]:
    groups: dict[str, tuple[Path, Path, Path]] = {}
    files: set[Path] = set()
    for query_id, record in queries.items():
        paths = tuple(
            (dataset_root / str(record[field])).resolve()
            for field in ("visible", "infrared", "depth")
        )
        if not all(path.is_file() for path in paths):
            missing = [str(path) for path in paths if not path.is_file()]
            raise FileNotFoundError(f"missing modality files for {query_id}: {missing}")
        group_id = Path(str(record["visible"])).stem
        previous = groups.get(group_id)
        if previous is not None and previous != paths:
            raise ValueError(
                f"image group {group_id} maps to inconsistent modality paths"
            )
        groups[group_id] = paths
        files.update(paths)
    return groups, sorted(files)


def _verify_and_load(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    raw_paths = dict(config.get("paths", {}))
    required_names = (
        "dataset_root",
        "queries",
        "s01_predictions",
        "s02_predictions",
        "s03_predictions",
        "s04_predictions",
        "candidate_cache",
        "s03_debug",
        "s04_debug",
        "output_root",
        "report_root",
    )
    paths = {
        name: _resolve_path(raw_paths.get(name), repo_root) for name in required_names
    }
    research_word_report = _resolve_path(
        raw_paths.get("research_word_report"), repo_root, required=False
    )
    if research_word_report is not None:
        paths["research_word_report"] = research_word_report
    scale_settings = dict(config.get("scale_probe", {}))
    scale_model_files: list[Path] = []
    scale_model_fingerprint: dict[str, Any] = {"status": "not_enabled"}
    if scale_settings.get("enabled"):
        scale_model_path = _resolve_path(scale_settings.get("model_path"), repo_root)
        if not scale_model_path.is_dir():
            raise FileNotFoundError(
                f"scale-probe model directory is missing: {scale_model_path}"
            )
        paths["scale_model_path"] = scale_model_path
        required_model_files = (
            "model.safetensors",
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        )
        scale_model_files = [scale_model_path / name for name in required_model_files]
        missing_model_files = [
            str(path) for path in scale_model_files if not path.is_file()
        ]
        if missing_model_files:
            raise FileNotFoundError(
                f"scale-probe model files are missing: {missing_model_files}"
            )
        actual_weights_hash = sha256_file(
            scale_model_path / "model.safetensors"
        ).upper()
        expected_weights_hash = str(
            scale_settings.get("expected_model_weights_sha256", "")
        ).upper()
        if expected_weights_hash and actual_weights_hash != expected_weights_hash:
            raise ValueError(
                "scale-probe model weights SHA-256 mismatch: "
                f"expected={expected_weights_hash}, actual={actual_weights_hash}"
            )
        scale_model_fingerprint = {
            "status": "verified",
            "path": str(scale_model_path),
            "revision": scale_settings.get("model_revision"),
            "weights_file": "model.safetensors",
            "weights_sha256": actual_weights_hash,
            "files": {
                path.name: sha256_file(path).upper() for path in scale_model_files
            },
        }
    for name in required_names:
        if name in {"dataset_root", "output_root", "report_root"}:
            continue
        if not paths[name].is_file():
            raise FileNotFoundError(f"required input is missing: {name}={paths[name]}")
    if not paths["dataset_root"].is_dir():
        raise FileNotFoundError(f"dataset root is missing: {paths['dataset_root']}")

    configured_hashes = dict(config.get("expected_sha256", {}))
    verified_expected_hashes: dict[str, str] = {}
    for name, expected_hash in configured_hashes.items():
        if expected_hash is None or not str(expected_hash).strip():
            continue
        if name not in paths or not paths[name].is_file():
            raise FileNotFoundError(f"fingerprinted input is missing: {name}")
        actual_hash = sha256_file(paths[name]).upper()
        expected_value = str(expected_hash).upper()
        if actual_hash != expected_value:
            raise ValueError(
                f"input SHA-256 mismatch for {name}: "
                f"expected={expected_value}, actual={actual_hash}"
            )
        verified_expected_hashes[name] = actual_hash

    queries = _query_records(paths["queries"])
    query_ids = set(queries)
    expected = dict(config.get("expected_counts", {}))
    expected_queries = int(expected.get("queries", len(queries)))
    if len(queries) != expected_queries:
        raise ValueError(
            f"query count mismatch: expected={expected_queries}, actual={len(queries)}"
        )

    predictions = {
        name: _prediction_boxes(paths[f"{name}_predictions"])
        for name in ("s01", "s02", "s03", "s04")
    }
    for name, boxes in predictions.items():
        _assert_exact_ids(f"{name}_predictions", query_ids, set(boxes))
    candidates = _records_by_id(paths["candidate_cache"])
    s03_debug = _records_by_id(paths["s03_debug"])
    s04_debug = _records_by_id(paths["s04_debug"])
    _assert_exact_ids("candidate_cache", query_ids, set(candidates))
    _assert_exact_ids("s03_debug", query_ids, set(s03_debug))
    _assert_exact_ids("s04_debug", query_ids, set(s04_debug))

    groups, image_files = _image_paths(queries, paths["dataset_root"])
    expected_groups = int(expected.get("image_groups", len(groups)))
    if len(groups) != expected_groups:
        raise ValueError(
            f"image-group count mismatch: expected={expected_groups}, actual={len(groups)}"
        )

    protected_files = [
        paths["queries"],
        paths["s01_predictions"],
        paths["s02_predictions"],
        paths["s03_predictions"],
        paths["s04_predictions"],
        paths["candidate_cache"],
        paths["s03_debug"],
        paths["s04_debug"],
        *image_files,
    ]
    if research_word_report is not None:
        if not research_word_report.is_file():
            raise FileNotFoundError(
                f"research Word report is missing: {research_word_report}"
            )
        protected_files.append(research_word_report)
    protected_files.extend(scale_model_files)
    tree_hash, file_hashes = _hash_file_tree(protected_files)
    return {
        "paths": paths,
        "queries": queries,
        "predictions": predictions,
        "candidates": candidates,
        "s03_debug": s03_debug,
        "s04_debug": s04_debug,
        "groups": groups,
        "protected_files": protected_files,
        "input_tree_sha256_before": tree_hash,
        "input_file_sha256": file_hashes,
        "verified_expected_sha256": verified_expected_hashes,
        "scale_model_fingerprint": scale_model_fingerprint,
    }


def _build_query_profiles(
    queries: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        parse_query_profile(query_id, queries[query_id]) for query_id in sorted(queries)
    ]


def _build_image_profiles(
    groups: Mapping[str, tuple[Path, Path, Path]],
) -> list[dict[str, Any]]:
    return [
        profile_image_group(
            image_group_id=group_id,
            visible_path=groups[group_id][0],
            infrared_path=groups[group_id][1],
            depth_path=groups[group_id][2],
        )
        for group_id in sorted(groups)
    ]


def _selected_index(record: Mapping[str, Any], *names: str) -> int:
    for name in names:
        if record.get(name) is not None:
            return int(record[name])
    return 0


def _build_model_profiles(
    *,
    query_profiles: Sequence[Mapping[str, Any]],
    loaded: Mapping[str, Any],
) -> list[dict[str, Any]]:
    profile_by_id = {str(row["query_id"]): row for row in query_profiles}
    result: list[dict[str, Any]] = []
    for query_id in sorted(profile_by_id):
        candidate_record = loaded["candidates"][query_id]
        candidates = list(candidate_record.get("candidates", []))
        result.append(
            build_model_behavior_record(
                query_id=query_id,
                query_profile=profile_by_id[query_id],
                boxes={
                    name: loaded["predictions"][name][query_id]
                    for name in ("s01", "s02", "s03", "s04")
                },
                candidates=candidates,
                s03_selected_index=_selected_index(
                    loaded["s03_debug"][query_id], "ranker_index", "selected_index"
                ),
                s04_selected_index=_selected_index(
                    loaded["s04_debug"][query_id], "selected_index", "ranker_index"
                ),
            )
        )
    return result


def _build_small_proxies(
    query_profiles: Sequence[Mapping[str, Any]],
    model_profiles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    queries = {str(row["query_id"]): row for row in query_profiles}
    result: list[dict[str, Any]] = []
    for model in model_profiles:
        query_id = str(model["query_id"])
        all_areas = dict(model.get("areas", {}))
        target_compatible_areas = [
            float(value)
            for value in model.get("topk_target_compatible_areas", [])
            if value is not None and math.isfinite(float(value))
        ]
        # S03 and S04 are selectors over the same GroundingDINO candidates, not
        # independent visual models. Counting them as separate scale evidence
        # would let one GDINO box satisfy a two-model rule three times.
        independent_scale_evidence: dict[str, float | None] = {
            "florence_s01": all_areas.get("s01"),
            "gdino_s02_top1": all_areas.get("s02"),
        }
        if target_compatible_areas:
            independent_scale_evidence["gdino_topk_target_median"] = statistics.median(
                target_compatible_areas
            )
        pairwise = dict(model.get("pairwise_iou", {}))
        proxy = classify_small_target_proxy(
            query_profile=queries[query_id],
            model_areas=independent_scale_evidence,
            model_ious={"florence_s01_gdino_s02": pairwise.get("s01_s02")},
        )
        result.append(
            {
                "query_id": query_id,
                "image_group_id": queries[query_id]["image_group_id"],
                "dependent_selector_areas_excluded": ["s03", "s04"],
                **proxy,
            }
        )
    return result


def _probe_records(
    query_profiles: Sequence[Mapping[str, Any]],
    small_proxies: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    proxies = {str(row["query_id"]): row for row in small_proxies}
    result: list[dict[str, Any]] = []
    for profile in query_profiles:
        query_id = str(profile["query_id"])
        proxy_label = proxies[query_id]["small_target_proxy"]
        if proxy_label in {"high_confidence_small", "medium_confidence_small"}:
            group = proxy_label
        elif profile.get("region_or_structure") or profile.get("plural_flag"):
            group = "region_group"
        else:
            group = "ordinary_control"
        result.append(
            {
                "query_id": query_id,
                "image_group_id": profile["image_group_id"],
                "probe_group": group,
                "small_target_proxy": proxy_label,
            }
        )
    return result


def _external_records(
    config: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, list[dict[str, Any]] | None], dict[str, Any]]:
    records: dict[str, list[dict[str, Any]] | None] = {}
    assets: dict[str, Any] = {}
    for name, raw_path in sorted(dict(config.get("external_manifests", {})).items()):
        path = _resolve_path(raw_path, repo_root, required=False)
        if path is None or not path.is_file():
            records[name] = None
            assets[name] = {
                "status": "missing_not_verified",
                "path": None if path is None else str(path),
            }
        else:
            rows = _load_jsonl(path)
            records[name] = rows
            assets[name] = {
                "status": "available",
                "path": str(path),
                "records": len(rows),
                "sha256": sha256_file(path).upper(),
            }
    return records, assets


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _summary_metrics(
    query_profiles: Sequence[Mapping[str, Any]],
    image_profiles: Sequence[Mapping[str, Any]],
    model_profiles: Sequence[Mapping[str, Any]],
    small_proxies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    query_count = len(query_profiles)
    small_counts = Counter(str(row["small_target_proxy"]) for row in small_proxies)
    region_ids = {
        str(row["query_id"]) for row in query_profiles if row.get("region_or_structure")
    }
    model_by_id = {str(row["query_id"]): row for row in model_profiles}
    region_disagree = sum(
        model_by_id[query_id].get("agreement_at_05") == "all_disagree"
        for query_id in region_ids
    )
    depth_ids = {
        str(row["query_id"])
        for row in query_profiles
        if row.get("depth_relation") is not None
    }
    depth_rgb_stable = sum(
        float(model_by_id[query_id].get("pairwise_iou", {}).get("s01_s02", 0.0)) >= 0.5
        for query_id in depth_ids
    )
    low_light = [
        row
        for row in image_profiles
        if float(row.get("visible_brightness", 1.0)) < 0.18
        or float(row.get("visible_low_light_score", 0.0)) >= 0.5
    ]
    usable_ir = sum(not bool(row.get("infrared_low_information")) for row in low_light)
    token_counts = [int(row["token_count"]) for row in query_profiles]
    profile_by_id = {str(row["query_id"]): row for row in query_profiles}
    s03_switched = [
        row for row in model_profiles if int(row.get("s03_selected_index", 0)) != 0
    ]
    s04_switched = [
        row for row in model_profiles if int(row.get("s04_selected_index", 0)) != 0
    ]

    def switch_breakdown(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for row in rows:
            profile = profile_by_id[str(row["query_id"])]
            flags = {
                "small_object_lexical_prior": bool(profile.get("instance_small_prior")),
                "region_structure": bool(profile.get("region_or_structure")),
                "plural_group": bool(profile.get("plural_flag")),
                "ordinal": profile.get("ordinal") is not None,
                "depth_relation": profile.get("depth_relation") is not None,
                "spatial_relation": bool(profile.get("relations")),
                "action": bool(profile.get("actions")),
                "attribute": bool(profile.get("attributes")),
            }
            for name, enabled in flags.items():
                if enabled:
                    counts[name] += 1
        return dict(sorted(counts.items()))

    agreement_counts = Counter(
        str(row.get("agreement_at_05", "unknown")) for row in model_profiles
    )
    s03_area_ratios = [
        float(row["s03_area_ratio_vs_s02"])
        for row in s03_switched
        if row.get("s03_area_ratio_vs_s02") is not None
    ]
    s04_area_ratios = [
        float(row["s04_area_ratio_vs_s02"])
        for row in s04_switched
        if row.get("s04_area_ratio_vs_s02") is not None
    ]
    return {
        "query_count": query_count,
        "image_group_count": len(image_profiles),
        "query_token_mean": statistics.fmean(token_counts) if token_counts else 0.0,
        "query_token_median": statistics.median(token_counts) if token_counts else 0.0,
        "high_confidence_small_ratio": _ratio(
            small_counts["high_confidence_small"], query_count
        ),
        "medium_confidence_small_ratio": _ratio(
            small_counts["medium_confidence_small"], query_count
        ),
        "high_medium_small_ratio": _ratio(
            small_counts["high_confidence_small"]
            + small_counts["medium_confidence_small"],
            query_count,
        ),
        "small_target_proxy_counts": dict(sorted(small_counts.items())),
        "region_structure_ratio": _ratio(len(region_ids), query_count),
        "region_all_disagree_ratio": _ratio(region_disagree, len(region_ids)),
        "ordinal_ratio": _ratio(
            sum(row.get("ordinal") is not None for row in query_profiles), query_count
        ),
        "spatial_relation_ratio": _ratio(
            sum(bool(row.get("relations")) for row in query_profiles), query_count
        ),
        "depth_relation_ratio": _ratio(len(depth_ids), query_count),
        "depth_rgb_baseline_agreement_proxy_ratio": _ratio(
            depth_rgb_stable, len(depth_ids)
        ),
        "plural_group_ratio": _ratio(
            sum(bool(row.get("plural_flag")) for row in query_profiles), query_count
        ),
        "attribute_ratio": _ratio(
            sum(bool(row.get("attributes")) for row in query_profiles), query_count
        ),
        "action_ratio": _ratio(
            sum(bool(row.get("actions")) for row in query_profiles), query_count
        ),
        "ocr_text_ratio": _ratio(
            sum(bool(row.get("ocr_or_text")) for row in query_profiles), query_count
        ),
        "low_light_group_ratio": _ratio(len(low_light), len(image_profiles)),
        "usable_ir_within_low_light_ratio": _ratio(usable_ir, len(low_light)),
        "all_disagree_ratio": _ratio(
            sum(row.get("agreement_at_05") == "all_disagree" for row in model_profiles),
            query_count,
        ),
        "agreement_at_05_counts": dict(sorted(agreement_counts.items())),
        "s03_switch_count": len(s03_switched),
        "s03_switch_ratio": _ratio(len(s03_switched), query_count),
        "s03_switch_multilabel_breakdown": switch_breakdown(s03_switched),
        "s03_switched_area_ratio_median": (
            statistics.median(s03_area_ratios) if s03_area_ratios else None
        ),
        "s03_switched_area_ratio_gt_2_count": sum(
            value > 2.0 for value in s03_area_ratios
        ),
        "s03_small_to_large_proxy_count": sum(
            bool(row.get("s03_small_to_large_proxy")) for row in s03_switched
        ),
        "s03_cross_label_switch_count": sum(
            bool(row.get("s03_label_changed_from_top1")) for row in s03_switched
        ),
        "s04_switch_count": len(s04_switched),
        "s04_switch_ratio": _ratio(len(s04_switched), query_count),
        "s04_switch_multilabel_breakdown": switch_breakdown(s04_switched),
        "s04_switched_area_ratio_median": (
            statistics.median(s04_area_ratios) if s04_area_ratios else None
        ),
        "s04_switched_area_ratio_max": (
            max(s04_area_ratios) if s04_area_ratios else None
        ),
        "s04_cross_label_switch_count": sum(
            bool(row.get("s04_label_changed_from_top1")) for row in s04_switched
        ),
        "topk_candidate_count_mean": statistics.fmean(
            float(row.get("topk_candidate_count", 0)) for row in model_profiles
        ),
    }


def build_optimization_triggers(
    metrics: Mapping[str, Any],
    scale_summary: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = dict(thresholds or FROZEN_TRIGGER_THRESHOLDS)
    high = float(metrics.get("high_confidence_small_ratio", 0.0))
    high_medium = high + float(metrics.get("medium_confidence_small_ratio", 0.0))
    scale_gain = scale_summary.get("small_stable_candidate_gain_pp")
    small_population = (
        high >= thresholds["piza_high_small_ratio"]
        or high_medium >= thresholds["piza_high_plus_medium_ratio"]
    )
    if scale_gain is None or scale_summary.get("status") != "complete":
        piza_triggered = False
        piza_status = "pending_scale_probe" if small_population else "not_triggered"
    else:
        piza_triggered = (
            small_population and float(scale_gain) >= thresholds["piza_scale_gain_pp"]
        )
        piza_status = "triggered" if piza_triggered else "not_triggered"

    ape_triggered = (
        float(metrics.get("region_structure_ratio", 0.0))
        >= thresholds["ape_region_ratio"]
        or float(metrics.get("region_all_disagree_ratio", 0.0))
        >= thresholds["ape_region_all_disagree_ratio"]
    )
    rgbt_triggered = (
        float(metrics.get("low_light_group_ratio", 0.0))
        >= thresholds["rgbt_low_light_ratio"]
        and float(metrics.get("usable_ir_within_low_light_ratio", 0.0))
        >= thresholds["rgbt_usable_ir_ratio"]
    )
    depth_triggered = (
        float(metrics.get("depth_relation_ratio", 0.0))
        >= thresholds["depth_relation_ratio"]
        and float(metrics.get("depth_rgb_baseline_agreement_proxy_ratio", 0.0))
        >= thresholds["depth_rgb_stability_ratio"]
    )

    if piza_triggered:
        recommendation = "PIZA"
        rationale = "small-target proxy prevalence and scale-probe gain crossed both frozen thresholds"
    elif ape_triggered:
        recommendation = "APE-Ti"
        rationale = (
            "region/structure prevalence or disagreement crossed the frozen threshold"
        )
    elif rgbt_triggered:
        recommendation = "RGBT asset audit"
        rationale = (
            "low-light prevalence and usable infrared coverage crossed both thresholds"
        )
    else:
        recommendation = "MM-Grounding-DINO-T multi-domain validation"
        rationale = "no single verified difficulty branch dominates"
    return {
        "thresholds_frozen_before_s04": True,
        "piza": {
            "triggered": piza_triggered,
            "status": piza_status,
            "high_small_threshold": thresholds["piza_high_small_ratio"],
            "high_plus_medium_threshold": thresholds["piza_high_plus_medium_ratio"],
            "scale_gain_threshold_pp": thresholds["piza_scale_gain_pp"],
        },
        "ape_ti": {
            "triggered": ape_triggered,
            "status": "triggered" if ape_triggered else "not_triggered",
        },
        "rgbt_audit": {
            "triggered": rgbt_triggered,
            "status": "triggered" if rgbt_triggered else "not_triggered",
        },
        "depth_late_fusion_preparation": {
            "triggered": depth_triggered,
            "status": (
                "prepare_only_no_2d_substitution"
                if depth_triggered
                else "not_triggered"
            ),
        },
        "frozen_thresholds": dict(sorted(thresholds.items())),
        "llmdet": {"status": "deferred_until_long_query_role_cluster_is_validated"},
        "recommended_next_model": recommendation,
        "recommendation_rationale": rationale,
    }


def _fmt_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _write_reports(
    *,
    report_root: Path,
    metrics: Mapping[str, Any],
    domain_metrics: Mapping[str, Any],
    triggers: Mapping[str, Any],
    scale_summary: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    evidence_note = (
        "> 证据边界：Query 与图像统计是输入事实；模型框面积、一致性和“小目标”分级是无标签代理；"
        "模型优劣与 ACC 必须等待平台验证。\n"
    )
    main = f"""# AIC 测试域全量画像 v1

{evidence_note}

## 执行摘要

- Query：{metrics['query_count']:,} 条；图像组：{metrics['image_group_count']:,} 组。
- Query 平均长度：{metrics['query_token_mean']:.2f} 词，中位数：{metrics['query_token_median']:.1f} 词。
- 高置信小目标代理：{_fmt_percent(metrics['high_confidence_small_ratio'])}；高+中置信：{_fmt_percent(metrics['high_medium_small_ratio'])}。
- 区域/结构输入占比：{_fmt_percent(metrics['region_structure_ratio'])}。
- 空间关系：{_fmt_percent(metrics['spatial_relation_ratio'])}；序数：{_fmt_percent(metrics['ordinal_ratio'])}；深度关系：{_fmt_percent(metrics['depth_relation_ratio'])}；属性：{_fmt_percent(metrics['attribute_ratio'])}；动作：{_fmt_percent(metrics['action_ratio'])}。
- S04 在冻结候选缓存上切换 {metrics['s04_switch_count']:,} 条（{_fmt_percent(metrics['s04_switch_ratio'])}）。

## 冻结路线结论

下一受控模型候选：**{triggers['recommended_next_model']}**。

依据：{triggers['recommendation_rationale']}。

尺度探针状态：`{scale_summary.get('status', 'unknown')}`。若探针未运行，PIZA 的最终触发结论保持 pending，不能把词义或单模型小框直接当成真实小目标。

## 结论分层

- 输入事实：Query 原文、词长、显式关系词、图像格式、亮度/IR/Depth 质量。
- 模型代理：S01-S04 框面积、候选标签、模型间 IoU、跨尺度稳定性。
- 启发式代理：small-target risk、结构目标分类、模态效用风险。
- 待平台验证：任何新模型是否提高 ACC@0.5，以及 S04 是否优于 0.4980。
"""
    main += f"""
## 关键代理与决策解释

| 项目 | 结果 | 证据类型 | 允许的结论 |
|---|---:|---|---|
| 高置信小目标 | {_fmt_percent(metrics['high_confidence_small_ratio'])} | heuristic proxy | 只能说明多证据支持小目标风险，不是真实 GT 面积 |
| 高+中置信小目标 | {_fmt_percent(metrics['high_medium_small_ratio'])} | heuristic proxy | 可用于构建分层 probe，不能直接估计小目标 ACC |
| 区域/结构目标 | {_fmt_percent(metrics['region_structure_ratio'])} | input-derived heuristic taxonomy | 已按目标主体分类，参照物中的 building/path 不会污染该字段 |
| 小目标跨尺度稳定候选增益 | {scale_summary.get('small_stable_candidate_gain_pp')} pp | model-derived proxy | 衡量候选稳定性，不等于救回正确样本 |
| 普通对照候选膨胀 | {float(scale_summary.get('ordinary_control_candidate_inflation') or 0.0):.2f}x | model-derived proxy | 反对全量 Tile，支持选择性使用 |
| Depth Query | {_fmt_percent(metrics['depth_relation_ratio'])} | input-derived fact | 仅触发 late-fusion 准备，禁止用二维位置替代物理深度 |

## 等待 S04 后的单变量原则

- `S04 > 0.4980`：S04 成为控制基线；下一次只引入本画像触发的 APE-Ti。
- `0.4938 < S04 <= 0.4980`：Florence 仍为稳定基线，S04 仅作诊断。
- `S04 <= 0.4938`：停止 LightGBM 提交路线，回到 Florence，并测试 APE-Ti 单模型。
- 无论平台结果如何，都不能反向修改本轮代理阈值，也不能把 AIC 预测用于训练。
"""
    (report_root / "aic_testset_full_profile.md").write_text(main, encoding="utf-8")

    paper = f"""# 论文与开源模型到 AIC 的路线决策

{evidence_note}

## 按 AIC 用途排序（不是按论文榜单排序）

| 路线 | 已核验公开资产 | AIC 主要覆盖 | 本轮决定 |
|---|---|---|---|
| [PIZA + SOREC（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Goto_Referring_Expression_Comprehension_for_Small_Objects_ICCV_2025_paper.html) | [代码、标注与 adapter](https://github.com/mmaiLab/sorec) | 极小目标、渐进式缩放 | `{triggers['piza']['status']}`；只有尺度探针也达到冻结阈值才优先 |
| [APE-Ti（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Aligning_and_Prompting_Everything_All_at_Once_for_Universal_Visual_CVPR_2024_paper.html) | [推理、训练与 Ti/L 权重](https://github.com/shenyunhang/APE) | 区域、stuff、建筑结构、复杂句子 | `{triggers['ape_ti']['status']}` |
| [MM-Grounding-DINO-T](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino) | 公开训练配置与 T/B/L checkpoint | 多域训练和可复现微调 | 无单一难点主导时先做验证，不在本轮训练 |
| [RGBT-GroundBench/VGNet](https://arxiv.org/abs/2512.24561) | [代码](https://github.com/crazyxiaoxi/RGBT-GroundBench)、[数据](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset)、[模型资源](https://huggingface.co/JiawenXi/RGBT-Ground-Model) | RGB+TIR+Query、低光 | `{triggers['rgbt_audit']['status']}`；先做许可、文件—配置映射和近重复审计，本轮不下载大模型包 |
| [LLMDet（CVPR 2025 Highlight）](https://openaccess.thecvf.com/content/CVPR2025/html/Fu_LLMDet_Learning_Strong_Open-Vocabulary_Object_Detectors_under_the_Supervision_of_CVPR_2025_paper.html) | [Swin-T/B/L 代码与权重](https://github.com/iSEE-Laboratory/LLMDet) | 长句、属性、角色语义 | {triggers['llmdet']['status']} |
| [RGBDT500/RDTTrack（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b4962fcd5d4410a9f43ef70f528eedd8-Abstract-Datasets_and_Benchmarks_Track.html) | 三模态跟踪数据、代码与权重 | RGB/Depth/TIR 融合结构 | 任务无语言；只借鉴结构，重叠审计前不训练 |

## 纠正两个容易误读的结论

- PIZA、APE-Ti 和 MM-GDINO-T 都是 RGB+文本路线，不能替代 IR/Depth；它们分别解决尺度、区域语义和可训练基座问题。
- RGBDT500 的视觉模态最接近 AIC，但其监督是单目标跟踪，不是自然语言指代定位；“有三模态 bbox”不等于“可直接训练 AIC 模型”。

## 本机外部验证资产状态

```json
{json.dumps(assets, ensure_ascii=False, indent=2, sort_keys=True)}
```

`missing_not_verified` 只表示本机未发现对应 manifest，本轮没有下载或伪造数据；不表示官方资产不存在。
"""
    (report_root / "paper_model_to_aic_decision.md").write_text(paper, encoding="utf-8")

    disagreement = f"""# 模型分歧与尺度代理

{evidence_note}

- 三基线在 IoU@0.5 下全部分歧代理比例：{_fmt_percent(metrics['all_disagree_ratio'])}。
- 区域/结构簇全部分歧比例：{_fmt_percent(metrics['region_all_disagree_ratio'])}。
- S03 切换 {metrics['s03_switch_count']:,} 条，其中小框到大框代理 {metrics['s03_small_to_large_proxy_count']:,} 条、跨标签 {metrics['s03_cross_label_switch_count']:,} 条；这些是负迁移风险代理，不是逐条错误真值。
- S04 仅有 {metrics['s04_switch_count']:,} 条切换，最大面积倍率 {metrics['s04_switched_area_ratio_max']}; 任务多标签分布为 `{json.dumps(metrics['s04_switch_multilabel_breakdown'], ensure_ascii=False, sort_keys=True)}`。
- IoU@0.5 一致性分组：`{json.dumps(metrics['agreement_at_05_counts'], ensure_ascii=False, sort_keys=True)}`。
- 尺度探针：`{json.dumps(scale_summary, ensure_ascii=False, sort_keys=True)}`。

机器明细位于 `outputs/aic_testset_profile_v1/model_behavior_profile.jsonl` 与 `small_target_proxy.csv`。
"""
    (report_root / "model_disagreement_and_scale.md").write_text(
        disagreement, encoding="utf-8"
    )

    domain = f"""# AIC 与外部训练域差异

{evidence_note}

外部数据有 bbox 时使用 `ground_truth` 面积；AIC 只保留 `heuristic_proxy`，二者未合并为真实尺度曲线。

```json
{json.dumps(domain_metrics, ensure_ascii=False, indent=2, sort_keys=True)}
```
"""
    (report_root / "domain_shift_analysis.md").write_text(domain, encoding="utf-8")

    trigger_report = f"""# 优化触发矩阵

{evidence_note}

```json
{json.dumps(triggers, ensure_ascii=False, indent=2, sort_keys=True)}
```

冻结规则优先级：PIZA（须同时满足规模与尺度收益）→ APE-Ti → RGBT 资产审计 → MM-Grounding-DINO-T 多域验证。Depth 只准备 late fusion，禁止用二维位置代替物理深度。
"""
    trigger_report += f"""

## 本轮冻结判定

- APE-Ti：`{triggers['ape_ti']['status']}`，区域/结构比例为 {_fmt_percent(metrics['region_structure_ratio'])}。
- PIZA：`{triggers['piza']['status']}`；虽然高+中置信小目标代理达到 {_fmt_percent(metrics['high_medium_small_ratio'])}，尺度稳定候选增益仅 {scale_summary.get('small_stable_candidate_gain_pp')} pp，未越过 10 pp 门槛。
- RGBT 专项：`{triggers['rgbt_audit']['status']}`；低光图像组比例仅 {_fmt_percent(metrics['low_light_group_ratio'])}。
- Depth：`{triggers['depth_late_fusion_preparation']['status']}`；只进入数据与后融合准备，不生成本轮提交。
"""
    (report_root / "optimization_trigger_matrix.md").write_text(
        trigger_report, encoding="utf-8"
    )

    next_experiments = f"""# 下一轮三个单变量实验

{evidence_note}

## 画像冻结后的具体顺序

1. **APE-Ti 单模型对照**：第一优先。只替换候选/定位模型，其余 Query、预处理和提交格式全部保持不变。
2. **PIZA/SOREC 小目标离线专项**：第二优先但暂不直接上平台。先补齐并核验 SOREC 资产，在公开有标签小目标集验证；AIC 上只做既定 probe，不训练。
3. **Depth late-fusion tracer**：第三优先。仅对显式 nearest/farthest/front/behind Query 使用真实 Depth 统计；PNG uint16 与 JPG uint8 未知域分开，且不得以二维位置代替深度。

LLMDet、RGBT-VGNet 和 MM-Grounding-DINO-T 均保留为后续路线：前两者分别等待长句角色失败证据和低光/IR 触发，MM-GDINO-T 等 APE/PIZA 单变量结果后再决定是否多域微调。

S04 结果回来后只更换控制基线，不反向修改本画像代理规则。下一份平台提交只能引入一个变量。
"""
    (report_root / "next_three_experiments.md").write_text(
        next_experiments, encoding="utf-8"
    )


def _contact_selection_key(seed: int, cluster: str, query_id: str) -> str:
    return hashlib.sha256(f"{seed}:{cluster}:{query_id}".encode("utf-8")).hexdigest()


def _safe_display_text(value: str, limit: int = 72) -> str:
    return value[:limit].encode("ascii", errors="replace").decode("ascii")


def _build_contact_sheets(
    *,
    output_root: Path,
    dataset_root: Path,
    queries: Mapping[str, Mapping[str, Any]],
    query_profiles: Sequence[Mapping[str, Any]],
    image_profiles: Sequence[Mapping[str, Any]],
    model_profiles: Sequence[Mapping[str, Any]],
    small_proxies: Sequence[Mapping[str, Any]],
    seed: int,
    max_per_cluster: int = 25,
) -> dict[str, Any]:
    query_by_id = {str(row["query_id"]): row for row in query_profiles}
    model_by_id = {str(row["query_id"]): row for row in model_profiles}
    proxy_by_id = {str(row["query_id"]): row for row in small_proxies}
    image_by_group = {str(row["image_group_id"]): row for row in image_profiles}
    clusters: dict[str, list[str]] = {
        "small_target_high": [],
        "baseline_agree_ranker_differs": [],
        "all_models_disagree": [],
        "plural_group": [],
        "structural_region": [],
        "ordinal": [],
        "depth_relation": [],
        "low_light": [],
        "query_noise": [],
        "s04_switches": [],
    }
    for query_id, profile in query_by_id.items():
        model = model_by_id[query_id]
        proxy = proxy_by_id[query_id]
        image = image_by_group[str(profile["image_group_id"])]
        if proxy.get("small_target_proxy") == "high_confidence_small":
            clusters["small_target_high"].append(query_id)
        if model.get("agreement_at_05") == "florence_gdino_agree_s03_differs":
            clusters["baseline_agree_ranker_differs"].append(query_id)
        if model.get("agreement_at_05") == "all_disagree":
            clusters["all_models_disagree"].append(query_id)
        if profile.get("plural_flag"):
            clusters["plural_group"].append(query_id)
        if profile.get("region_or_structure"):
            clusters["structural_region"].append(query_id)
        if profile.get("ordinal") is not None:
            clusters["ordinal"].append(query_id)
        if profile.get("depth_relation") is not None:
            clusters["depth_relation"].append(query_id)
        if (
            float(image.get("visible_brightness", 1.0)) < 0.18
            or float(image.get("visible_low_light_score", 0.0)) >= 0.5
        ):
            clusters["low_light"].append(query_id)
        if profile.get("noise_flags"):
            clusters["query_noise"].append(query_id)
        if int(model.get("s04_selected_index", 0)) != 0:
            clusters["s04_switches"].append(query_id)

    contact_root = output_root / "contact_sheets"
    contact_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "warning": "MODEL PREDICTIONS — NOT GROUND TRUTH",
        "seed": seed,
        "max_per_cluster": max_per_cluster,
        "clusters": {},
    }
    colors = {
        "s01": (0, 220, 255),
        "s02": (0, 255, 80),
        "s03": (255, 60, 60),
        "s04": (255, 210, 0),
    }
    cell_width, cell_height = 320, 240
    columns = 5
    for cluster, query_ids in clusters.items():
        selected = sorted(
            query_ids,
            key=lambda query_id: _contact_selection_key(seed, cluster, query_id),
        )[:max_per_cluster]
        manifest["clusters"][cluster] = {
            "available": len(query_ids),
            "selected": selected,
        }
        if not selected:
            continue
        rows = (len(selected) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        for index, query_id in enumerate(selected):
            query_record = queries[query_id]
            with Image.open(dataset_root / str(query_record["visible"])) as source:
                visible = ImageOps.exif_transpose(source).convert("RGB")
            rendered = ImageOps.contain(visible, (cell_width, cell_height - 42))
            canvas = Image.new("RGB", (cell_width, cell_height), (20, 20, 20))
            offset_x = (cell_width - rendered.width) // 2
            offset_y = 24 + (cell_height - 42 - rendered.height) // 2
            canvas.paste(rendered, (offset_x, offset_y))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (4, 3), "MODEL PREDICTIONS — NOT GROUND TRUTH", fill=(255, 255, 255)
            )
            scale_x = rendered.width
            scale_y = rendered.height
            for name, box in model_by_id[query_id]["boxes"].items():
                x1, y1, x2, y2 = (float(value) for value in box)
                draw.rectangle(
                    (
                        offset_x + x1 * scale_x,
                        offset_y + y1 * scale_y,
                        offset_x + x2 * scale_x,
                        offset_y + y2 * scale_y,
                    ),
                    outline=colors.get(name, (255, 255, 255)),
                    width=2,
                )
            label = f"{query_id}: {_safe_display_text(str(query_record['query']))}"
            draw.text((4, cell_height - 16), label, fill=(255, 255, 255))
            sheet.paste(
                canvas,
                ((index % columns) * cell_width, (index // columns) * cell_height),
            )
        sheet.save(contact_root / f"{cluster}.png", optimize=False)
    _write_json(output_root / "contact_sheet_manifest.json", manifest)
    return manifest


def _manifest(
    *,
    output_root: Path,
    report_root: Path,
    input_fingerprints: Mapping[str, str],
    configuration_sha256: str,
    code_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    output_files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and path.name != "sha256_manifest.json"
        and "telemetry" not in path.relative_to(output_root).parts
    )
    report_files = sorted(path for path in report_root.rglob("*") if path.is_file())
    return {
        "inputs": dict(sorted(input_fingerprints.items())),
        "configuration_sha256": configuration_sha256,
        "code": dict(sorted(code_fingerprints.items())),
        "outputs": {str(path): sha256_file(path).upper() for path in output_files},
        "reports": {str(path): sha256_file(path).upper() for path in report_files},
    }


def run_profile_pipeline(
    config: Mapping[str, Any],
    *,
    repo_root: Path | str,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the deterministic, read-only profile and optional GPU scale probe."""

    root = Path(repo_root).resolve()
    trigger_thresholds = resolve_frozen_trigger_thresholds(config)
    configuration_sha256 = _canonical_sha256(config)
    loaded = _verify_and_load(config, root)
    paths = loaded["paths"]
    output_root: Path = paths["output_root"]
    report_root: Path = paths["report_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    query_profiles = _build_query_profiles(loaded["queries"])
    image_profiles = _build_image_profiles(loaded["groups"])
    model_profiles = _build_model_profiles(query_profiles=query_profiles, loaded=loaded)
    small_proxies = _build_small_proxies(query_profiles, model_profiles)

    probe = dict(config.get("probe", {}))
    probe_manifest = select_scale_probe(
        _probe_records(query_profiles, small_proxies),
        per_group=int(probe.get("per_group", 100)),
        seed=int(probe.get("seed", 20260801)),
    )
    scale_config = dict(config.get("scale_probe", {}))
    scale_rows: list[dict[str, Any]] = []
    if scale_config.get("enabled"):
        from .testset_scale_probe import run_scale_probe

        scale_config["model_path"] = str(paths["scale_model_path"])
        scale_config["stage2_max_queries"] = int(probe.get("stage2_max_queries", 200))
        scale_rows, scale_summary = run_scale_probe(
            settings=scale_config,
            dataset_root=paths["dataset_root"],
            queries=loaded["queries"],
            query_profiles={str(row["query_id"]): row for row in query_profiles},
            florence_boxes=loaded["predictions"]["s01"],
            candidate_records=loaded["candidates"],
            probe_manifest=probe_manifest,
            output_path=output_root / "scale_probe_results.jsonl",
            telemetry_path=output_root / "telemetry" / "scale_probe_telemetry.jsonl",
            resume=resume,
        )
        scale_summary["enabled"] = True
        scale_summary["selected_queries"] = len(probe_manifest)
        scale_by_id = {str(row["query_id"]): row for row in scale_rows}
        for proxy in small_proxies:
            scale_row = scale_by_id.get(str(proxy["query_id"]))
            if scale_row is None or not scale_row.get("stage1_complete"):
                proxy["scale_probe_support"] = None
                continue
            support = bool(scale_row.get("enhanced_small_support"))
            proxy["scale_probe_support"] = support
            proxy["scale_probe_evidence_kind"] = "model_derived_proxy"
            if support and proxy["small_target_proxy"] == "medium_confidence_small":
                proxy["small_target_proxy"] = "high_confidence_small"
            elif support and proxy["small_target_proxy"] == "uncertain":
                proxy["small_target_proxy"] = "medium_confidence_small"
    else:
        scale_summary = {
            "status": "not_run",
            "enabled": False,
            "selected_queries": len(probe_manifest),
            "small_stable_candidate_gain_pp": None,
            "note": "no AIC ACC or oracle is computed without ground truth",
        }

    external_records, external_assets = _external_records(config, root)
    domain_metrics = summarize_external_domains(
        aic_query_profiles=query_profiles,
        aic_small_proxies=small_proxies,
        external_records=external_records,
    )
    metrics = _summary_metrics(
        query_profiles, image_profiles, model_profiles, small_proxies
    )
    triggers = build_optimization_triggers(
        metrics, scale_summary, thresholds=trigger_thresholds
    )

    _write_jsonl(output_root / "query_profile.jsonl", query_profiles)
    _write_csv(output_root / "image_group_profile.csv", image_profiles)
    _write_jsonl(output_root / "model_behavior_profile.jsonl", model_profiles)
    _write_csv(output_root / "small_target_proxy.csv", small_proxies)
    if not scale_config.get("enabled"):
        _write_jsonl(output_root / "scale_probe_results.jsonl", [])
    _write_json(output_root / "scale_probe_manifest.json", probe_manifest)
    _write_json(output_root / "domain_shift_metrics.json", domain_metrics)

    _build_contact_sheets(
        output_root=output_root,
        dataset_root=paths["dataset_root"],
        queries=loaded["queries"],
        query_profiles=query_profiles,
        image_profiles=image_profiles,
        model_profiles=model_profiles,
        small_proxies=small_proxies,
        seed=int(probe.get("seed", 20260801)),
        max_per_cluster=25,
    )

    tree_hash_after, _ = _hash_file_tree(loaded["protected_files"])
    if tree_hash_after != loaded["input_tree_sha256_before"]:
        raise RuntimeError("protected AIC inputs changed during profiling")

    run_summary = {
        "pipeline": "aic_testset_profile_v1",
        "configuration_sha256": configuration_sha256,
        "stages": list(PROFILE_STAGES),
        "resume_requested": bool(resume),
        "query_count": metrics["query_count"],
        "image_group_count": metrics["image_group_count"],
        "input_tree_sha256_before": loaded["input_tree_sha256_before"],
        "input_tree_sha256_after": tree_hash_after,
        "inputs_unchanged": True,
        "verified_expected_sha256": loaded["verified_expected_sha256"],
        "scale_model_fingerprint": loaded["scale_model_fingerprint"],
        "metrics": metrics,
        "scale_probe": scale_summary,
        "external_assets": external_assets,
        "research_word_report": (
            {
                "status": "available",
                "path": str(paths["research_word_report"]),
                "sha256": sha256_file(paths["research_word_report"]).upper(),
            }
            if paths.get("research_word_report") is not None
            else {"status": "not_configured"}
        ),
        "optimization_triggers": triggers,
        "frozen_trigger_thresholds": trigger_thresholds,
        "forbidden_artifacts_created": False,
    }
    # Resume is an execution hint, not a scientific result; exclude it from
    # deterministic machine output so identical inputs stay byte-identical.
    run_summary["resume_requested"] = False
    _write_json(output_root / "run_summary.json", run_summary)
    _write_reports(
        report_root=report_root,
        metrics=metrics,
        domain_metrics=domain_metrics,
        triggers=triggers,
        scale_summary=scale_summary,
        assets=external_assets,
    )
    module_root = Path(__file__).resolve().parent
    labelled_code_paths = {
        "src/aic_baseline/testset_profile.py": module_root / "testset_profile.py",
        "src/aic_baseline/testset_profile_pipeline.py": Path(__file__).resolve(),
        "src/aic_baseline/testset_scale_probe.py": module_root
        / "testset_scale_probe.py",
        "tools/run_aic_testset_profile.py": root / "tools/run_aic_testset_profile.py",
        "configs/aic_testset_profile.example.yaml": root
        / "configs/aic_testset_profile.example.yaml",
    }
    code_fingerprints = {
        label: sha256_file(path).upper()
        for label, path in labelled_code_paths.items()
        if path.is_file()
    }
    manifest = _manifest(
        output_root=output_root,
        report_root=report_root,
        input_fingerprints=loaded["input_file_sha256"],
        configuration_sha256=configuration_sha256,
        code_fingerprints=code_fingerprints,
    )
    _write_json(output_root / "sha256_manifest.json", manifest)
    return run_summary
