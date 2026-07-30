from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from PIL import Image
import torch

from .bbox import pixel_to_normalized
from .diagnostics import (
    DiagnosticCandidate,
    bbox_area_bin,
    candidate_semantic_type,
    evaluate_candidates,
    query_category,
    summarize_diagnostic_records,
)


@dataclass(frozen=True)
class ModelCandidate:
    """A model candidate in pixel ``xyxy`` coordinates."""

    pixel_bbox: list[float]
    label: str
    score: float | None
    source: str = "full"


@dataclass(frozen=True)
class ExternalPrediction:
    raw_output: Any
    candidates: list[ModelCandidate]


class ExternalPredictor(Protocol):
    model_name: str

    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        ...


class CandidateNumericalError(RuntimeError):
    """A non-finite model output that must abort the experiment."""


def normalize_grounding_query(query: str) -> str:
    """Apply the fixed GroundingDINO text normalization policy."""

    normalized = re.sub(r"\s+", " ", query.strip().lower())
    normalized = normalized.rstrip(" .,:;!?")
    return f"{normalized}."


def _jsonable_raw_output(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _candidate_to_normalized(
    candidate: ModelCandidate,
    *,
    width: int,
    height: int,
) -> DiagnosticCandidate:
    if len(candidate.pixel_bbox) != 4:
        raise ValueError("candidate bbox must have four coordinates")
    values = [float(value) for value in candidate.pixel_bbox]
    if not all(math.isfinite(value) for value in values):
        raise CandidateNumericalError(
            "candidate bbox contains NaN or infinity"
        )
    x1, y1, x2, y2 = values
    clipped = [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]
    return DiagnosticCandidate(
        bbox=pixel_to_normalized(clipped, width=width, height=height),
        label=candidate.label,
        score=None if candidate.score is None else float(candidate.score),
        source=candidate.source,
    )


def _read_completed(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    records: list[dict[str, Any]] = []
    completed: set[str] = set()
    if not path.exists():
        return records, completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            query_id = str(record["query_id"])
            if query_id in completed:
                raise ValueError(
                    f"duplicate query_id in checkpoint line {line_number}: "
                    f"{query_id}"
                )
            completed.add(query_id)
            records.append(record)
    return records, completed


def _prepare_fingerprint(
    output_dir: Path,
    run_fingerprint: dict[str, Any],
    *,
    resume: bool,
) -> None:
    path = output_dir / "run_fingerprint.json"
    if resume and path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != run_fingerprint:
            raise ValueError(
                "resume fingerprint mismatch; refusing to reuse old results"
            )
        return
    if resume and not path.exists():
        path.write_text(
            json.dumps(run_fingerprint, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return
    path.write_text(
        json.dumps(run_fingerprint, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_external_evaluation(
    *,
    records: Sequence[dict[str, Any]],
    data_root: Path | str,
    predictor: ExternalPredictor,
    output_dir: Path | str,
    run_fingerprint: dict[str, Any],
    resume: bool = False,
) -> dict[str, Any]:
    """Run deterministic external grounding evaluation with no fallback.

    Runtime/system exceptions deliberately propagate and abort the experiment.
    Completed records are appended one at a time so an interrupted run can
    resume only when its fingerprint is identical.
    """

    root = Path(data_root).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    _prepare_fingerprint(output, run_fingerprint, resume=resume)

    if resume:
        evaluated, completed = _read_completed(predictions_path)
        mode = "a"
    else:
        evaluated, completed = [], set()
        predictions_path.write_text("", encoding="utf-8")
        mode = "a"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    processed = 0
    started = time.perf_counter()
    with predictions_path.open(mode, encoding="utf-8", newline="\n") as sink:
        for source_record in records:
            query_id = str(source_record["query_id"])
            if query_id in completed:
                continue
            image_path = root / str(source_record["image_relpath"])
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            width, height = image.size

            inference_started = time.perf_counter()
            prediction = predictor.predict(
                image=image,
                query=str(source_record["query"]),
            )
            latency_ms = (time.perf_counter() - inference_started) * 1000.0

            candidates: list[DiagnosticCandidate] = []
            candidate_errors: list[str] = []
            for index, candidate in enumerate(prediction.candidates):
                try:
                    candidates.append(
                        _candidate_to_normalized(
                            candidate,
                            width=width,
                            height=height,
                        )
                    )
                except (TypeError, ValueError) as exc:
                    candidate_errors.append(f"candidate[{index}]: {exc}")

            metrics = evaluate_candidates(
                source_record["bbox_xyxy_normalized"],
                candidates,
            )
            record = {
                "query_id": query_id,
                "dataset": source_record["dataset"],
                "image_key": source_record["image_key"],
                "image_relpath": source_record["image_relpath"],
                "query": source_record["query"],
                "gt_bbox": source_record["bbox_xyxy_normalized"],
                "probe_group": source_record.get("probe_group"),
                "area_bin": bbox_area_bin(
                    source_record["bbox_xyxy_normalized"]
                ),
                "query_category": query_category(
                    str(source_record["query"])
                ),
                "candidates": [
                    candidate.as_dict() for candidate in candidates
                ],
                "candidate_semantic_type": candidate_semantic_type(candidates),
                **metrics,
                "latency_ms": latency_ms,
                "raw_output": _jsonable_raw_output(prediction.raw_output),
                "error": (
                    "; ".join(candidate_errors)
                    if candidate_errors
                    else ("no_candidate" if not candidates else None)
                ),
            }
            sink.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            sink.flush()
            evaluated.append(record)
            completed.add(query_id)
            processed += 1

    wall_seconds = time.perf_counter() - started
    summary = summarize_diagnostic_records(evaluated)
    summary["execution"] = {
        "model": predictor.model_name,
        "requested_records": len(records),
        "completed_records": len(evaluated),
        "records_reused": len(evaluated) - processed,
        "records_processed": processed,
        "wall_seconds_this_run": wall_seconds,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
