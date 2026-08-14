from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .bbox import BBoxError, pixel_to_normalized, validate_normalized_bbox
from .data import AICDataset, AICRecord
from .external_eval import ExternalPrediction, ExternalPredictor, ModelCandidate
from .submission import build_submission


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class FullDetectionResult:
    predictions: dict[str, list[float]]
    summary: dict[str, Any]
    predictions_path: Path
    runtime_path: Path


@dataclass(frozen=True)
class FinalizedSubmission:
    json_path: Path
    zip_path: Path
    audit_path: Path
    audit: dict[str, Any]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare_fingerprint(
    output_dir: Path,
    fingerprint: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    path = output_dir / "run_fingerprint.json"
    normalized = dict(fingerprint)
    if resume and path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != normalized:
            raise ValueError(
                "resume fingerprint mismatch; refusing to mix model runs"
            )
        return
    _write_json(path, normalized)


def _read_prediction_checkpoint(
    path: Path,
    *,
    expected_records: Mapping[str, AICRecord],
) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    if not path.exists():
        return rows, completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id in completed:
                raise ValueError(
                    f"duplicate query_id in checkpoint line {line_number}: {query_id}"
                )
            if query_id not in expected_records:
                raise ValueError(f"unknown query_id in checkpoint: {query_id}")
            if row.get("query") != expected_records[query_id].query:
                raise ValueError(f"query changed for checkpointed record: {query_id}")
            validate_normalized_bbox(row["selected_bbox"])
            rows.append(row)
            completed.add(query_id)
    return rows, completed


def _normalized_candidate(
    candidate: ModelCandidate,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    if len(candidate.pixel_bbox) != 4:
        raise BBoxError("candidate bbox must contain four coordinates")
    values = [float(value) for value in candidate.pixel_bbox]
    if not all(math.isfinite(value) for value in values):
        raise BBoxError("candidate bbox contains NaN or infinity")
    x1, y1, x2, y2 = values
    clipped = [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]
    bbox = pixel_to_normalized(clipped, width=width, height=height)
    score = None if candidate.score is None else float(candidate.score)
    if score is not None and not math.isfinite(score):
        raise BBoxError("candidate score contains NaN or infinity")
    return {
        "bbox": bbox,
        "label": str(candidate.label),
        "score": score,
        "source": str(candidate.source),
    }


def _select_native_top1(candidates: Sequence[dict[str, Any]]) -> int:
    if not candidates:
        raise RuntimeError("model returned no candidate")
    if not any(candidate["score"] is not None for candidate in candidates):
        return 0
    return max(
        range(len(candidates)),
        key=lambda index: (
            float("-inf")
            if candidates[index]["score"] is None
            else float(candidates[index]["score"])
        ),
    )


def _record_prediction(
    *,
    record: AICRecord,
    prediction: ExternalPrediction,
    width: int,
    height: int,
) -> dict[str, Any]:
    candidates = [
        _normalized_candidate(candidate, width=width, height=height)
        for candidate in prediction.candidates
    ]
    try:
        selected_index = _select_native_top1(candidates)
    except RuntimeError as error:
        raise RuntimeError(
            f"{record.query_id}: model returned no candidate"
        ) from error
    selected = candidates[selected_index]
    return {
        "query_id": record.query_id,
        "query": record.query,
        "visible": record.source["visible"],
        "image_size": [width, height],
        "selection_policy": "native_highest_score_top1",
        "selected_index": selected_index,
        "selected_bbox": selected["bbox"],
        "selected_label": selected["label"],
        "selected_score": selected["score"],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def run_aic_full_detection(
    *,
    dataset: AICDataset,
    predictor: ExternalPredictor,
    output_dir: Path | str,
    run_fingerprint: Mapping[str, Any],
    resume: bool = False,
    limit: int | None = None,
    batch_size: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> FullDetectionResult:
    """Run a zero-training native Top-1 model over AIC with safe resume.

    Model outputs are checkpointed without latency fields so the machine result
    remains deterministic. Runtime measurements are written separately.
    System/model errors propagate; the function never fabricates a bbox.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    records = [dataset[index] for index in range(len(dataset))]
    if limit is not None:
        records = records[:limit]
    expected = {record.query_id: record for record in records}

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    runtime_path = output / "runtime_events.jsonl"
    _prepare_fingerprint(output, run_fingerprint, resume=resume)

    if resume:
        evaluated, completed = _read_prediction_checkpoint(
            predictions_path,
            expected_records=expected,
        )
    else:
        predictions_path.write_text("", encoding="utf-8")
        runtime_path.write_text("", encoding="utf-8")
        evaluated, completed = [], set()

    started = time.perf_counter()
    processed = 0
    cached_visible_path: Path | None = None
    cached_image: Any = None
    with (
        predictions_path.open("a", encoding="utf-8", newline="\n") as sink,
        runtime_path.open("a", encoding="utf-8", newline="\n") as runtime_sink,
    ):
        pending = [record for record in records if record.query_id not in completed]
        cursor = 0
        while cursor < len(pending):
            first = pending[cursor]
            batch = [first]
            while (
                len(batch) < batch_size
                and cursor + len(batch) < len(pending)
                and pending[cursor + len(batch)].visible_path == first.visible_path
            ):
                batch.append(pending[cursor + len(batch)])
            cursor += len(batch)

            if first.visible_path != cached_visible_path:
                cached_image = dataset.load_visible(first)
                cached_visible_path = first.visible_path
            image = cached_image
            inference_started = time.perf_counter()
            predict_batch = getattr(predictor, "predict_batch", None)
            if len(batch) > 1 and callable(predict_batch):
                batch_predictions = predict_batch(
                    images=[image] * len(batch),
                    queries=[record.query for record in batch],
                )
            else:
                batch_predictions = [
                    predictor.predict(image=image, query=record.query)
                    for record in batch
                ]
            batch_latency_ms = (time.perf_counter() - inference_started) * 1000.0
            if len(batch_predictions) != len(batch):
                raise RuntimeError(
                    "batch predictor returned a different number of predictions"
                )

            for record, prediction in zip(batch, batch_predictions):
                row = _record_prediction(
                    record=record,
                    prediction=prediction,
                    width=image.width,
                    height=image.height,
                )
                sink.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                sink.flush()
                runtime_sink.write(
                    json.dumps(
                        {
                            "query_id": record.query_id,
                            "batch_size": len(batch),
                            "batch_latency_ms": batch_latency_ms,
                            "latency_ms": batch_latency_ms / len(batch),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                runtime_sink.flush()
                evaluated.append(row)
                completed.add(record.query_id)
                processed += 1
                if progress_callback is not None:
                    progress_callback(len(completed), len(records), record.query_id)

    wall_seconds = time.perf_counter() - started
    predictions = {
        str(row["query_id"]): validate_normalized_bbox(row["selected_bbox"])
        for row in evaluated
    }
    candidate_counts = [int(row["candidate_count"]) for row in evaluated]
    summary = {
        "model": predictor.model_name,
        "requested_records": len(records),
        "completed_records": len(evaluated),
        "records_reused": len(evaluated) - processed,
        "records_processed": processed,
        "wall_seconds_this_run": wall_seconds,
        "no_candidate_count": 0,
        "invalid_bbox_count": 0,
        "min_candidate_count": min(candidate_counts) if candidate_counts else 0,
        "max_candidate_count": max(candidate_counts) if candidate_counts else 0,
        "mean_candidate_count": (
            sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
        ),
        "complete": len(evaluated) == len(records),
    }
    _write_json(output / "run_summary.json", summary)
    return FullDetectionResult(
        predictions=predictions,
        summary=summary,
        predictions_path=predictions_path,
        runtime_path=runtime_path,
    )


def finalize_aic_full_detection(
    *,
    dataset: AICDataset,
    detection_dir: Path | str,
    submission_dir: Path | str,
) -> FinalizedSubmission:
    detection = Path(detection_dir)
    expected = {
        dataset[index].query_id: dataset[index] for index in range(len(dataset))
    }
    rows, completed = _read_prediction_checkpoint(
        detection / "predictions.jsonl",
        expected_records=expected,
    )
    expected_ids = set(expected)
    if completed != expected_ids:
        missing = sorted(expected_ids - completed)
        extra = sorted(completed - expected_ids)
        raise ValueError(
            "full detection is incomplete; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    predictions = {
        str(row["query_id"]): validate_normalized_bbox(row["selected_bbox"])
        for row in rows
    }

    output = Path(submission_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "predictions_submission.json"
    zip_path = output / "predictions_submission.zip"
    submission = build_submission(
        original_records=dataset.raw_records,
        predictions=predictions,
        output_json=json_path,
        output_zip=zip_path,
    )

    modified_non_bbox = 0
    invalid_bbox = 0
    for query_id, source in dataset.raw_records.items():
        submitted = dict(submission[query_id])
        bbox = submitted.pop("bbox", None)
        if submitted != dict(source):
            modified_non_bbox += 1
        try:
            validate_normalized_bbox(bbox)
        except (BBoxError, TypeError):
            invalid_bbox += 1
    with zipfile.ZipFile(zip_path) as archive:
        zip_entries = archive.namelist()
    audit = {
        "query_count": len(submission),
        "query_ids_exact": set(submission) == set(dataset.raw_records),
        "modified_non_bbox_count": modified_non_bbox,
        "invalid_bbox_count": invalid_bbox,
        "zip_entries": zip_entries,
        "zip_contains_only_predictions_submission_json": zip_entries
        == ["predictions_submission.json"],
        "predictions_json_sha256": sha256_file(json_path),
        "predictions_zip_sha256": sha256_file(zip_path),
    }
    if not (
        audit["query_ids_exact"]
        and modified_non_bbox == 0
        and invalid_bbox == 0
        and audit["zip_contains_only_predictions_submission_json"]
    ):
        raise ValueError(f"submission audit failed: {audit}")
    audit_path = output / "submission_audit.json"
    _write_json(audit_path, audit)
    return FinalizedSubmission(
        json_path=json_path,
        zip_path=zip_path,
        audit_path=audit_path,
        audit=audit,
    )
