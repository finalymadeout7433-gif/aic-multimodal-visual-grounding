from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from .bbox import BBoxError, pixel_to_normalized, validate_normalized_bbox
from .data import AICDataset
from .florence import FlorencePrediction


class Grounder(Protocol):
    def predict(self, *, image: Image.Image, query: str) -> FlorencePrediction: ...


@dataclass(frozen=True)
class InferenceRunResult:
    predictions: dict[str, list[float]]
    summary: dict[str, Any]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _normalize_candidate(
    pixel_bbox: list[float], *, width: int, height: int
) -> list[float]:
    x1, y1, x2, y2 = pixel_bbox
    clamped = [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]
    return pixel_to_normalized(clamped, width=width, height=height)


def run_inference(
    *,
    dataset: AICDataset,
    grounder: Grounder,
    output_dir: Path | str,
    limit: int | None = None,
    fallback_mode: str = "error",
    resume: bool = False,
    progress_interval: int = 100,
) -> InferenceRunResult:
    """逐 Query 推理并写入可审计日志；有限样本运行不会冒充正式提交。"""

    if fallback_mode not in {"error", "center"}:
        raise ValueError("fallback_mode 只能是 error 或 center")
    selected_records = [
        dataset[index]
        for index in range(min(len(dataset), limit or len(dataset)))
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    debug_path = output / "predictions_debug.jsonl"
    predictions: dict[str, list[float]] = {}
    latencies_ms: list[float] = []
    fallback_count = 0
    invalid_checkpoint_lines = 0
    failure_records: dict[str, dict[str, str]] = {}
    selected_ids = {record.query_id for record in selected_records}

    if resume and debug_path.exists():
        with debug_path.open("r", encoding="utf-8") as checkpoint:
            for line in checkpoint:
                if not line.strip():
                    continue
                try:
                    saved = json.loads(line)
                    query_id = saved["query_id"]
                    if query_id not in selected_ids:
                        continue
                    predictions[query_id] = validate_normalized_bbox(
                        saved["normalized_bbox"]
                    )
                    latencies_ms.append(float(saved.get("latency_ms", 0.0)))
                    fallback_count += int(bool(saved.get("fallback_used", False)))
                    if saved.get("fallback_used", False):
                        failure_records[query_id] = {
                            "query_id": query_id,
                            "error": str(saved.get("error", "")),
                            "fallback_bbox": json.dumps(
                                saved["normalized_bbox"],
                                ensure_ascii=False,
                            ),
                        }
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    invalid_checkpoint_lines += 1

    reused_count = len(predictions)
    processed_this_run = 0

    mode = "a" if resume else "w"
    with debug_path.open(mode, encoding="utf-8", newline="\n") as debug_file:
        for record in selected_records:
            if record.query_id in predictions:
                continue
            image = dataset.load_visible(record)
            started = time.perf_counter()
            fallback_used = False
            error_message: str | None = None
            raw_output: str | None = None
            candidates: list[dict[str, Any]] = []
            try:
                model_prediction = grounder.predict(image=image, query=record.query)
                raw_output = model_prediction.raw_output
                candidates = [
                    {
                        "pixel_bbox": candidate.pixel_bbox,
                        "label": candidate.label,
                    }
                    for candidate in model_prediction.candidates
                ]
                normalized = _normalize_candidate(
                    model_prediction.selected.pixel_bbox,
                    width=image.width,
                    height=image.height,
                )
            except (BBoxError, ValueError, RuntimeError) as error:
                if fallback_mode == "error":
                    raise
                normalized = [0.25, 0.25, 0.75, 0.75]
                fallback_used = True
                fallback_count += 1
                error_message = str(error)
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(latency_ms)
            processed_this_run += 1
            predictions[record.query_id] = validate_normalized_bbox(normalized)
            if fallback_used:
                failure_records[record.query_id] = {
                    "query_id": record.query_id,
                    "error": error_message or "",
                    "fallback_bbox": json.dumps(
                        predictions[record.query_id],
                        ensure_ascii=False,
                    ),
                }
            debug_record = {
                "query_id": record.query_id,
                "query_original": record.query,
                "visible": str(record.visible_path),
                "source_suffix": record.visible_path.suffix.lower(),
                "image_size": [image.width, image.height],
                "raw_output": raw_output,
                "candidates": candidates,
                "normalized_bbox": predictions[record.query_id],
                "fallback_used": fallback_used,
                "error": error_message,
                "latency_ms": latency_ms,
            }
            debug_file.write(
                json.dumps(debug_record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            debug_file.flush()
            if (
                progress_interval > 0
                and processed_this_run % progress_interval == 0
            ):
                completed = reused_count + processed_this_run
                print(
                    f"[AIC] 已完成 {completed}/{len(selected_records)} 条 Query",
                    flush=True,
                )

    count = len(selected_records)
    summary = {
        "records_in_dataset": len(dataset),
        "records_processed": count,
        "records_processed_this_run": processed_this_run,
        "records_reused_from_checkpoint": reused_count,
        "invalid_checkpoint_lines": invalid_checkpoint_lines,
        "is_full_dataset_run": count == len(dataset),
        "valid_bbox_count": len(predictions),
        "valid_bbox_rate": len(predictions) / count if count else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / count if count else 0.0,
        "latency_ms_mean": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
        "latency_ms_p50": _percentile(latencies_ms, 0.50),
        "latency_ms_p95": _percentile(latencies_ms, 0.95),
    }
    (output / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "failure_cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as failure_file:
        writer = csv.DictWriter(
            failure_file,
            fieldnames=["query_id", "error", "fallback_bbox"],
        )
        writer.writeheader()
        writer.writerows(failure_records.values())
    return InferenceRunResult(predictions=predictions, summary=summary)
