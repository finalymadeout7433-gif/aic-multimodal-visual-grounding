from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .bbox import intersection_over_union, pixel_to_normalized, validate_normalized_bbox
from .diagnostics import TileWindow


_CANDIDATE_SETTING_KEYS = (
    "model_path",
    "model_revision",
    "dtype",
    "box_threshold",
    "text_threshold",
    "max_candidates",
    "high_resolution_shortest_edge",
    "high_resolution_longest_edge",
    "tile_overlap",
)


def candidate_generation_fingerprint(settings: Mapping[str, Any]) -> str:
    payload = {
        key: settings.get(key) for key in _CANDIDATE_SETTING_KEYS if key in settings
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def grid_windows(
    *,
    width: int,
    height: int,
    rows: int,
    columns: int,
    overlap: float,
) -> list[TileWindow]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must lie in [0, 1)")

    tile_width = min(
        width,
        round(width / (columns - (columns - 1) * overlap)),
    )
    tile_height = min(
        height,
        round(height / (rows - (rows - 1) * overlap)),
    )
    x_positions = (
        [0]
        if columns == 1
        else [
            round(index * (width - tile_width) / (columns - 1))
            for index in range(columns)
        ]
    )
    y_positions = (
        [0]
        if rows == 1
        else [
            round(index * (height - tile_height) / (rows - 1)) for index in range(rows)
        ]
    )
    return [
        TileWindow(x1=x, y1=y, x2=x + tile_width, y2=y + tile_height)
        for y in y_positions
        for x in x_positions
    ]


def _canonical_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in {"the", "a", "an", "of", "with"}
    }


def _target_compatible(candidate: Mapping[str, Any], target_phrase: str) -> bool:
    target = _canonical_tokens(target_phrase)
    label = _canonical_tokens(str(candidate.get("label", "")))
    return bool(target and label and target & label)


def _area(candidate: Mapping[str, Any]) -> float:
    x1, y1, x2, y2 = validate_normalized_bbox(candidate["bbox"])
    return (x2 - x1) * (y2 - y1)


def _agrees(
    box: Sequence[float], other: Sequence[float], threshold: float = 0.3
) -> bool:
    return (
        intersection_over_union(
            validate_normalized_bbox(box), validate_normalized_bbox(other)
        )
        >= threshold
    )


def scale_record_metrics(
    *,
    target_phrase: str,
    florence_bbox: Sequence[float],
    full_candidates: Sequence[Mapping[str, Any]],
    high_resolution_candidates: Sequence[Mapping[str, Any]],
    tile_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    full_target = [
        candidate
        for candidate in full_candidates
        if _target_compatible(candidate, target_phrase)
    ]
    high_target = [
        candidate
        for candidate in high_resolution_candidates
        if _target_compatible(candidate, target_phrase)
    ]
    tile_target = [
        candidate
        for candidate in tile_candidates
        if _target_compatible(candidate, target_phrase)
    ]
    full_stable = any(
        _agrees(candidate["bbox"], florence_bbox) for candidate in full_target
    )
    enhanced_to_florence = any(
        _agrees(candidate["bbox"], florence_bbox)
        for candidate in [*high_target, *tile_target]
    )
    cross_scale = any(
        _agrees(high["bbox"], tile["bbox"])
        for high in high_target
        for tile in tile_target
    )
    enhanced_stable = enhanced_to_florence or cross_scale
    enhanced_small = enhanced_stable and any(
        _area(candidate) < 0.01 for candidate in [*high_target, *tile_target]
    )
    return {
        "evidence_kind": "model_derived_proxy",
        "full_candidate_count": len(full_candidates),
        "high_resolution_candidate_count": len(high_resolution_candidates),
        "tile_candidate_count": len(tile_candidates),
        "full_target_compatible_count": len(full_target),
        "high_resolution_target_compatible_count": len(high_target),
        "tile_target_compatible_count": len(tile_target),
        "full_stable_target_candidate": full_stable,
        "enhanced_stable_target_candidate": enhanced_stable,
        "enhanced_agrees_with_florence": enhanced_to_florence,
        "enhanced_cross_scale_agreement": cross_scale,
        "enhanced_small_support": enhanced_small,
        "full_miss_tile_only_target_candidate": not full_target and bool(tile_target),
    }


def summarize_scale_probe(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [record for record in records if record.get("stage1_complete")]
    small = [
        record
        for record in complete
        if record.get("probe_group")
        in {"high_confidence_small", "medium_confidence_small"}
    ]
    control = [
        record for record in complete if record.get("probe_group") == "ordinary_control"
    ]

    def rate(key: str, rows: Sequence[Mapping[str, Any]]) -> float | None:
        if not rows:
            return None
        return sum(bool(row.get(key)) for row in rows) / len(rows)

    full_rate = rate("full_stable_target_candidate", small)
    enhanced_rate = rate("enhanced_stable_target_candidate", small)
    gain = (
        (enhanced_rate - full_rate) * 100.0
        if full_rate is not None and enhanced_rate is not None
        else None
    )
    control_inflation_values = [
        float(row.get("tile_candidate_count", 0))
        / max(float(row.get("full_candidate_count", 0)), 1.0)
        for row in control
    ]
    return {
        "evidence_kind": "model_derived_proxy",
        "status": (
            "complete" if complete and len(complete) == len(records) else "partial"
        ),
        "records": len(records),
        "stage1_complete_records": len(complete),
        "small_records": len(small),
        "small_full_stable_candidate_rate": full_rate,
        "small_enhanced_stable_candidate_rate": enhanced_rate,
        "small_stable_candidate_gain_pp": None if gain is None else round(gain, 6),
        "ordinary_control_candidate_inflation": (
            sum(control_inflation_values) / len(control_inflation_values)
            if control_inflation_values
            else None
        ),
        "full_miss_tile_only_target_candidate_rate": rate(
            "full_miss_tile_only_target_candidate", complete
        ),
        "aic_accuracy_computed": False,
        "aic_oracle_computed": False,
    }


def _write_records(path: Path, records: Mapping[str, Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(
            records[key], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
        for key in sorted(records)
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["query_id"])] = row
    return result


def _normalize_prediction(
    prediction: Any,
    *,
    image_width: int,
    image_height: int,
    source: str,
    window: TileWindow | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in prediction.candidates:
        values = [float(value) for value in candidate.pixel_bbox]
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise ValueError("scale-probe candidate contains invalid coordinates")
        if window is None:
            global_box = values
        else:
            global_box = [
                values[0] + window.x1,
                values[1] + window.y1,
                values[2] + window.x1,
                values[3] + window.y1,
            ]
        clipped = [
            max(0.0, min(float(image_width), global_box[0])),
            max(0.0, min(float(image_height), global_box[1])),
            max(0.0, min(float(image_width), global_box[2])),
            max(0.0, min(float(image_height), global_box[3])),
        ]
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            continue
        result.append(
            {
                "bbox": pixel_to_normalized(
                    clipped, width=image_width, height=image_height
                ),
                "label": candidate.label,
                "score": candidate.score,
                "source": source,
            }
        )
    return result


def _predict_tiles(
    predictor: Any,
    *,
    image: Image.Image,
    query: str,
    rows: int,
    columns: int,
    overlap: float,
    source_prefix: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, window in enumerate(
        grid_windows(
            width=image.width,
            height=image.height,
            rows=rows,
            columns=columns,
            overlap=overlap,
        )
    ):
        tile = image.crop((window.x1, window.y1, window.x2, window.y2))
        prediction = predictor.predict(image=tile, query=query)
        result.extend(
            _normalize_prediction(
                prediction,
                image_width=image.width,
                image_height=image.height,
                source=f"{source_prefix}_{index}",
                window=window,
            )
        )
    return result


def run_scale_probe(
    *,
    settings: Mapping[str, Any],
    dataset_root: Path,
    queries: Mapping[str, Mapping[str, Any]],
    query_profiles: Mapping[str, Mapping[str, Any]],
    florence_boxes: Mapping[str, Sequence[float]],
    candidate_records: Mapping[str, Mapping[str, Any]],
    probe_manifest: Sequence[Mapping[str, Any]],
    output_path: Path,
    telemetry_path: Path,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the frozen no-GT scale probe with per-query crash-safe resume."""

    import torch

    from .model_adapters import GroundingDinoExternalPredictor

    dtype_name = str(settings.get("dtype", "float32"))
    dtype = {"float32": torch.float32, "float16": torch.float16}.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported scale-probe dtype: {dtype_name}")
    predictor = GroundingDinoExternalPredictor(
        model_path=Path(str(settings["model_path"])),
        device=str(settings.get("device", "cuda")),
        dtype=dtype,
        box_threshold=float(settings.get("box_threshold", 0.15)),
        text_threshold=float(settings.get("text_threshold", 0.15)),
        max_candidates=int(settings.get("max_candidates", 10)),
    )
    shortest = int(settings.get("high_resolution_shortest_edge", 1024))
    longest = int(settings.get("high_resolution_longest_edge", 1706))
    default_processor_size = dict(predictor.processor.image_processor.size)
    high_resolution_size = {
        "shortest_edge": shortest,
        "longest_edge": longest,
    }
    overlap = float(settings.get("tile_overlap", 0.2))
    inter_query_cooldown_seconds = float(
        settings.get("inter_query_cooldown_seconds", 0.0)
    )
    stage2_cooldown_seconds = float(
        settings.get(
            "stage2_inter_query_cooldown_seconds", inter_query_cooldown_seconds
        )
    )
    if inter_query_cooldown_seconds < 0 or stage2_cooldown_seconds < 0:
        raise ValueError("scale-probe cooldown values must be non-negative")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {str(row["query_id"]): dict(row) for row in probe_manifest}
    generation_fingerprint = candidate_generation_fingerprint(settings)
    loaded_completed = _load_records(output_path) if resume else {}
    loaded_telemetry = _load_records(telemetry_path) if resume else {}
    completed: dict[str, dict[str, Any]] = {}
    legacy_resume_records_migrated = 0
    telemetry = {
        query_id: loaded_telemetry[query_id]
        for query_id in manifest
        if query_id in loaded_telemetry
    }
    for query_id, manifest_row in manifest.items():
        cached = loaded_completed.get(query_id)
        if cached is None:
            continue
        cached_fingerprint = cached.get("candidate_generation_fingerprint")
        if cached_fingerprint not in {None, generation_fingerprint}:
            continue
        if cached_fingerprint is None:
            legacy_resume_records_migrated += 1
        cached = {**cached, **manifest_row}
        cached["candidate_generation_fingerprint"] = generation_fingerprint
        if (
            cached.get("stage1_complete")
            and isinstance(cached.get("high_resolution_candidates"), list)
            and isinstance(cached.get("tile_2x2_candidates"), list)
        ):
            target_head = str(query_profiles[query_id].get("target_head", ""))
            cached.update(
                scale_record_metrics(
                    target_phrase="" if target_head == "unknown" else target_head,
                    florence_bbox=florence_boxes[query_id],
                    full_candidates=list(
                        candidate_records[query_id].get("candidates", [])
                    ),
                    high_resolution_candidates=cached["high_resolution_candidates"],
                    tile_candidates=cached["tile_2x2_candidates"],
                )
            )
        completed[query_id] = cached

    if torch.cuda.is_available() and str(settings.get("device", "cuda")).startswith(
        "cuda"
    ):
        torch.cuda.reset_peak_memory_stats()
    stage1_ids = sorted(manifest)
    for stage1_position, query_id in enumerate(stage1_ids, 1):
        if completed.get(query_id, {}).get("stage1_complete"):
            continue
        start = time.perf_counter()
        query_record = queries[query_id]
        visible_path = dataset_root / str(query_record["visible"])
        try:
            with Image.open(visible_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            predictor.processor.image_processor.size = high_resolution_size
            high_prediction = predictor.predict(
                image=image, query=str(query_record["query"])
            )
            high_candidates = _normalize_prediction(
                high_prediction,
                image_width=image.width,
                image_height=image.height,
                source="high_resolution_full",
            )
            predictor.processor.image_processor.size = default_processor_size
            tile_candidates = _predict_tiles(
                predictor,
                image=image,
                query=str(query_record["query"]),
                rows=2,
                columns=2,
                overlap=overlap,
                source_prefix="tile_2x2",
            )
            full_candidates = list(candidate_records[query_id].get("candidates", []))
            metrics = scale_record_metrics(
                target_phrase=(
                    ""
                    if query_profiles[query_id].get("target_head") == "unknown"
                    else str(query_profiles[query_id].get("target_head", ""))
                ),
                florence_bbox=florence_boxes[query_id],
                full_candidates=full_candidates,
                high_resolution_candidates=high_candidates,
                tile_candidates=tile_candidates,
            )
            completed[query_id] = {
                **manifest[query_id],
                **metrics,
                "high_resolution_candidates": high_candidates,
                "tile_2x2_candidates": tile_candidates,
                "candidate_generation_fingerprint": generation_fingerprint,
                "stage1_complete": True,
                "stage1_error": None,
            }
            telemetry[query_id] = {
                "query_id": query_id,
                "stage1_latency_ms": (time.perf_counter() - start) * 1000.0,
            }
        except (
            Exception
        ) as exc:  # Recorded explicitly; never converted to an empty success.
            completed[query_id] = {
                **manifest[query_id],
                "evidence_kind": "model_derived_proxy",
                "candidate_generation_fingerprint": generation_fingerprint,
                "stage1_complete": False,
                "stage1_error": f"{type(exc).__name__}: {exc}",
            }
            telemetry[query_id] = {
                "query_id": query_id,
                "stage1_latency_ms": (time.perf_counter() - start) * 1000.0,
            }
        _write_records(output_path, completed)
        _write_records(telemetry_path, telemetry)
        if (
            stage1_position == 1
            or stage1_position % 10 == 0
            or stage1_position == len(stage1_ids)
        ):
            print(
                f"[scale-probe] stage1 {stage1_position}/{len(stage1_ids)}",
                flush=True,
            )
        if inter_query_cooldown_seconds:
            time.sleep(inter_query_cooldown_seconds)

    stage2_limit = int(settings.get("stage2_max_queries", 200))
    unstable = sorted(
        (row for row in completed.values() if row.get("stage1_complete")),
        key=lambda row: (
            bool(row.get("enhanced_stable_target_candidate")),
            -float(row.get("tile_candidate_count", 0)),
            str(row["query_id"]),
        ),
    )[:stage2_limit]
    selected_ranks = {
        str(row["query_id"]): selection_rank
        for selection_rank, row in enumerate(unstable)
    }
    stage2_fields = {
        "stage2_selected",
        "stage2_selection_rank",
        "tile_3x3_candidates",
        "stage2_complete",
        "stage2_error",
    }
    for query_id, row in completed.items():
        if query_id in selected_ranks:
            row["stage2_selected"] = True
            row["stage2_selection_rank"] = selected_ranks[query_id]
        else:
            for field in stage2_fields:
                row.pop(field, None)
            row["stage2_selected"] = False
    for selection_rank, row in enumerate(unstable):
        query_id = str(row["query_id"])
        if completed[query_id].get("stage2_complete"):
            continue
        start = time.perf_counter()
        query_record = queries[query_id]
        visible_path = dataset_root / str(query_record["visible"])
        try:
            with Image.open(visible_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            predictor.processor.image_processor.size = default_processor_size
            tile_3x3 = _predict_tiles(
                predictor,
                image=image,
                query=str(query_record["query"]),
                rows=3,
                columns=3,
                overlap=overlap,
                source_prefix="tile_3x3",
            )
            completed[query_id]["tile_3x3_candidates"] = tile_3x3
            completed[query_id]["stage2_complete"] = True
            completed[query_id]["stage2_error"] = None
            telemetry.setdefault(query_id, {"query_id": query_id})[
                "stage2_latency_ms"
            ] = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            completed[query_id]["stage2_complete"] = False
            completed[query_id]["stage2_error"] = f"{type(exc).__name__}: {exc}"
            telemetry.setdefault(query_id, {"query_id": query_id})[
                "stage2_latency_ms"
            ] = (time.perf_counter() - start) * 1000.0
        _write_records(output_path, completed)
        _write_records(telemetry_path, telemetry)
        stage2_position = selection_rank + 1
        if (
            stage2_position == 1
            or stage2_position % 10 == 0
            or stage2_position == len(unstable)
        ):
            print(
                f"[scale-probe] stage2 {stage2_position}/{len(unstable)}",
                flush=True,
            )
        if stage2_cooldown_seconds:
            time.sleep(stage2_cooldown_seconds)

    for query_id in completed:
        completed[query_id].setdefault("stage2_selected", False)
    _write_records(output_path, completed)
    rows = [completed[key] for key in sorted(completed)]
    summary = summarize_scale_probe(rows)
    summary["stage2_selected_records"] = sum(
        bool(row.get("stage2_selected")) for row in rows
    )
    summary["stage2_complete_records"] = sum(
        bool(row.get("stage2_complete")) for row in rows
    )
    stage1_latencies = [
        float(row["stage1_latency_ms"])
        for row in telemetry.values()
        if row.get("stage1_latency_ms") is not None
    ]
    stage2_latencies = [
        float(row["stage2_latency_ms"])
        for row in telemetry.values()
        if row.get("stage2_latency_ms") is not None
    ]
    telemetry_summary = {
        "evidence_kind": "operational_telemetry_non_deterministic",
        "records": len(telemetry),
        "stage1_latency_ms_mean": (
            sum(stage1_latencies) / len(stage1_latencies) if stage1_latencies else None
        ),
        "stage2_latency_ms_mean": (
            sum(stage2_latencies) / len(stage2_latencies) if stage2_latencies else None
        ),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            and str(settings.get("device", "cuda")).startswith("cuda")
            else None
        ),
        "inter_query_cooldown_seconds": inter_query_cooldown_seconds,
        "stage2_inter_query_cooldown_seconds": stage2_cooldown_seconds,
        "legacy_resume_records_migrated": legacy_resume_records_migrated,
        "candidate_generation_fingerprint": generation_fingerprint,
    }
    telemetry_path.with_suffix(".summary.json").write_text(
        json.dumps(telemetry_summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    summary["high_resolution_shortest_edge"] = shortest
    summary["high_resolution_longest_edge"] = longest
    summary["tile_processor_size"] = default_processor_size
    summary["tile_overlap"] = overlap
    summary["telemetry_path"] = "telemetry/scale_probe_telemetry.jsonl"
    summary["telemetry_is_excluded_from_deterministic_manifest"] = True
    return rows, summary
