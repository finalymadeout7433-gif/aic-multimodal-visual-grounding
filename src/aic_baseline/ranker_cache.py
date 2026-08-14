from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image, ImageOps
import torch

from .bbox import intersection_over_union, pixel_to_normalized
from .external_eval import ExternalPredictor
from .ranker_features import deduplicate_rank_candidates


ProgressCallback = Callable[[int, int, dict[str, Any]], None]


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _read_jsonl_checkpoint(
    path: Path,
    *,
    tolerate_partial_final_line: bool,
) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    raw = path.read_bytes()
    if not raw:
        return [], 0
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    completed: set[str] = set()
    valid_bytes = 0
    for index, raw_line in enumerate(lines):
        is_last = index == len(lines) - 1
        has_newline = raw_line.endswith((b"\n", b"\r"))
        if not raw_line.strip():
            valid_bytes += len(raw_line)
            continue
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if tolerate_partial_final_line and is_last and not has_newline:
                break
            raise ValueError(
                f"damaged JSONL checkpoint at line {index + 1}: {path}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"checkpoint line {index + 1} is not an object: {path}"
            )
        query_id = str(record.get("query_id", ""))
        if not query_id:
            raise ValueError(
                f"checkpoint line {index + 1} has no query_id: {path}"
            )
        if query_id in completed:
            raise ValueError(
                f"duplicate query_id in checkpoint: {query_id}"
            )
        completed.add(query_id)
        records.append(record)
        valid_bytes += len(raw_line)
    return records, valid_bytes


def read_candidate_cache(path: Path | str) -> list[dict[str, Any]]:
    records, _ = _read_jsonl_checkpoint(
        Path(path),
        tolerate_partial_final_line=True,
    )
    return records


def _prepare_checkpoint(
    output_dir: Path,
    run_fingerprint: Mapping[str, Any],
    *,
    resume: bool,
) -> tuple[Path, list[dict[str, Any]], set[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_path = output_dir / "run_fingerprint.json"
    fingerprint = dict(run_fingerprint)
    if resume and fingerprint_path.exists():
        stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if stored != fingerprint:
            raise ValueError(
                "resume fingerprint mismatch; refusing to mix candidate caches"
            )
    elif resume and not fingerprint_path.exists():
        raise ValueError("resume requested but run_fingerprint.json is missing")
    else:
        fingerprint_path.write_text(
            json.dumps(fingerprint, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    candidates_path = output_dir / "candidates.jsonl"
    if resume:
        records, valid_bytes = _read_jsonl_checkpoint(
            candidates_path,
            tolerate_partial_final_line=True,
        )
        if candidates_path.exists() and candidates_path.stat().st_size != valid_bytes:
            with candidates_path.open("r+b") as handle:
                handle.truncate(valid_bytes)
    else:
        records = []
        candidates_path.write_text("", encoding="utf-8")
    completed = {str(record["query_id"]) for record in records}
    return candidates_path, records, completed


def _normalized_candidate(
    candidate: Any,
    *,
    width: int,
    height: int,
    original_rank: int,
) -> dict[str, Any]:
    if len(candidate.pixel_bbox) != 4:
        raise ValueError("candidate bbox must contain four coordinates")
    values = [float(value) for value in candidate.pixel_bbox]
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("candidate bbox contains NaN or infinity")
    score = candidate.score
    if score is not None and not math.isfinite(float(score)):
        raise FloatingPointError("candidate score contains NaN or infinity")
    x1, y1, x2, y2 = values
    clipped = [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError("candidate bbox is empty after clipping")
    return {
        "bbox": pixel_to_normalized(clipped, width=width, height=height),
        "label": str(candidate.label),
        "score": None if score is None else float(score),
        "original_rank": int(original_rank),
        "source": str(candidate.source),
    }


def _predict_candidates(
    *,
    image: Image.Image,
    query: str,
    predictor: ExternalPredictor,
    dedup_iou: float,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    prediction = predictor.predict(image=image, query=query)
    latency_ms = (time.perf_counter() - started) * 1000.0
    normalized = [
        _normalized_candidate(
            candidate,
            width=image.width,
            height=image.height,
            original_rank=index,
        )
        for index, candidate in enumerate(prediction.candidates)
    ]
    normalized = deduplicate_rank_candidates(
        normalized,
        iou_threshold=dedup_iou,
    )
    return normalized, latency_ms


def _area_bin(bbox: Sequence[float]) -> str:
    area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )
    if area < 0.01:
        return "small_lt_1pct"
    if area < 0.1:
        return "medium_1_to_10pct"
    return "large_ge_10pct"


def _write_summary(
    output_dir: Path,
    *,
    requested: int,
    records: Sequence[Mapping[str, Any]],
    reused: int,
    processed: int,
    wall_seconds: float,
) -> dict[str, Any]:
    no_candidates = sum(not record.get("candidates") for record in records)
    solvable = sum(bool(record.get("solvable_at_05")) for record in records)
    summary = {
        "requested_records": int(requested),
        "records": len(records),
        "records_reused": int(reused),
        "records_processed": int(processed),
        "no_candidate_records": int(no_candidates),
        "solvable_at_05_records": int(solvable),
        "candidate_oracle_acc_at_05": (
            solvable / len(records) if records else 0.0
        ),
        "wall_seconds_this_run": float(wall_seconds),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def run_external_candidate_cache(
    *,
    records: Sequence[Mapping[str, Any]],
    data_root: Path | str,
    predictor: ExternalPredictor,
    output_dir: Path | str,
    run_fingerprint: Mapping[str, Any],
    resume: bool = False,
    dedup_iou: float = 0.95,
    stop_after: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Cache GroundingDINO candidates for labelled external records."""

    root = Path(data_root).resolve()
    output = Path(output_dir)
    candidates_path, cached, completed = _prepare_checkpoint(
        output,
        run_fingerprint,
        resume=resume,
    )
    reused = len(cached)
    target_total = len(records) if stop_after is None else min(
        len(records), int(stop_after)
    )
    if target_total < 0:
        raise ValueError("stop_after must be non-negative")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processed = 0
    with candidates_path.open("a", encoding="utf-8", newline="\n") as sink:
        for source in records[:target_total]:
            query_id = str(source["query_id"])
            if query_id in completed:
                continue
            image_path = root / str(source["image_relpath"])
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            candidates, latency_ms = _predict_candidates(
                image=image,
                query=str(source["query"]),
                predictor=predictor,
                dedup_iou=dedup_iou,
            )
            gt_bbox = [float(value) for value in source["bbox_xyxy_normalized"]]
            candidate_ious = [
                intersection_over_union(candidate["bbox"], gt_bbox)
                for candidate in candidates
            ]
            for candidate, iou in zip(candidates, candidate_ious):
                candidate["iou"] = float(iou)
            record = {
                "query_id": query_id,
                "dataset": str(source["dataset"]),
                "image_key": str(source["image_key"]),
                "image_relpath": str(source["image_relpath"]),
                "query": str(source["query"]),
                "gt_bbox": gt_bbox,
                "query_category": str(
                    source.get("query_category", "other")
                ),
                "area_bin": _area_bin(gt_bbox),
                "candidates": candidates,
                "candidate_ious": candidate_ious,
                "best_candidate_iou": max(candidate_ious, default=0.0),
                "solvable_at_05": max(candidate_ious, default=0.0) >= 0.5,
                "latency_ms": float(latency_ms),
                "error": None if candidates else "no_candidate",
            }
            sink.write(_json_dump(record) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            cached.append(record)
            completed.add(query_id)
            processed += 1
            if progress_callback is not None:
                progress_callback(len(cached), target_total, record)
    return _write_summary(
        output,
        requested=target_total,
        records=cached,
        reused=reused,
        processed=processed,
        wall_seconds=time.perf_counter() - started,
    )


def run_aic_candidate_cache(
    *,
    records: Sequence[Mapping[str, Any]],
    dataset_root: Path | str,
    predictor: ExternalPredictor,
    output_dir: Path | str,
    run_fingerprint: Mapping[str, Any],
    resume: bool = False,
    dedup_iou: float = 0.95,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Cache candidates for the label-free AIC test set."""

    root = Path(dataset_root).resolve()
    output = Path(output_dir)
    candidates_path, cached, completed = _prepare_checkpoint(
        output,
        run_fingerprint,
        resume=resume,
    )
    reused = len(cached)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processed = 0
    with candidates_path.open("a", encoding="utf-8", newline="\n") as sink:
        for source in records:
            query_id = str(source["query_id"])
            if query_id in completed:
                continue
            visible_path = str(source.get("visible", source.get("image_relpath")))
            image_path = root / visible_path
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            candidates, latency_ms = _predict_candidates(
                image=image,
                query=str(source["query"]),
                predictor=predictor,
                dedup_iou=dedup_iou,
            )
            record = {
                "query_id": query_id,
                "dataset": "aic_test",
                "image_key": str(source.get("image_key", visible_path)),
                "image_relpath": visible_path,
                "query": str(source["query"]),
                "query_category": str(
                    source.get("query_category", "other")
                ),
                "candidates": candidates,
                "latency_ms": float(latency_ms),
                "error": None if candidates else "no_candidate",
            }
            sink.write(_json_dump(record) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            cached.append(record)
            completed.add(query_id)
            processed += 1
            if progress_callback is not None:
                progress_callback(len(cached), len(records), record)
    return _write_summary(
        output,
        requested=len(records),
        records=cached,
        reused=reused,
        processed=processed,
        wall_seconds=time.perf_counter() - started,
    )
