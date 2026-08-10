from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_baseline.aic_full_detection import (  # noqa: E402
    finalize_aic_full_detection,
    sha256_file,
)
from aic_baseline.bbox import validate_normalized_bbox  # noqa: E402
from aic_baseline.data import AICDataset, AICRecord  # noqa: E402


JSON_ARRAY_PATTERN = re.compile(r"\[[^\[\]]+\]")
FLOAT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
CANONICAL_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run EGM-Qwen3-VL-8B over AIC via a local OpenAI-compatible "
            "vLLM/SGLang server."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="EGM-8B")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--fallback",
        choices=("center", "error"),
        default="center",
        help="center keeps the submission legal; error aborts on parser/model failures.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _encode_image_data_url(path: Path, *, jpeg_quality: int) -> tuple[str, int, int]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        sink = io.BytesIO()
        image.save(sink, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(sink.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", width, height


def _prompt(query: str) -> str:
    # This follows the official EGM inference prompt style.
    return f"Locate {query}, output its bbox coordinates using JSON format"


def _numbers_from_jsonish(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        for key in ("bbox_2d", "bbox", "box", "boxes", "answer"):
            found = _numbers_from_jsonish(value.get(key))
            if found is not None:
                return found
        return None
    if isinstance(value, list):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            return [float(item) for item in value]
        for item in value:
            found = _numbers_from_jsonish(item)
            if found is not None:
                return found
    return None


def _extract_box(text: str) -> list[float]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        found = _numbers_from_jsonish(parsed)
        if found is not None:
            return found
    except json.JSONDecodeError:
        pass

    for match in JSON_ARRAY_PATTERN.finditer(stripped):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        found = _numbers_from_jsonish(parsed)
        if found is not None:
            return found

    numbers = [float(item.group(0)) for item in FLOAT_PATTERN.finditer(stripped)]
    if len(numbers) >= 4:
        return numbers[:4]
    raise ValueError(f"no bbox found in model output: {text[:200]}")


def _box_to_normalized(box: list[float], *, width: int, height: int) -> list[float] | None:
    if len(box) != 4:
        return None
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values):
        return None
    scale = max(abs(value) for value in values)
    if scale <= 1.5:
        x1, y1, x2, y2 = values
    elif scale <= 1000.0:
        x1, y1, x2, y2 = [value / 1000.0 for value in values]
    else:
        x1, y1, x2, y2 = [
            values[0] / float(width),
            values[1] / float(height),
            values[2] / float(width),
            values[3] / float(height),
        ]
    left, right = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    top, bottom = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if right - left <= 1e-6 or bottom - top <= 1e-6:
        return None
    return validate_normalized_bbox([left, top, right, bottom])


def _fallback_bbox() -> list[float]:
    return [0.25, 0.25, 0.75, 0.75]


def _read_checkpoint(path: Path, expected: dict[str, AICRecord]) -> tuple[list[dict[str, Any]], set[str]]:
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
                raise ValueError(f"duplicate query_id in checkpoint line {line_number}: {query_id}")
            if query_id not in expected:
                raise ValueError(f"unknown query_id in checkpoint: {query_id}")
            if row.get("query") != expected[query_id].query:
                raise ValueError(f"query changed for checkpointed record: {query_id}")
            validate_normalized_bbox(row["selected_bbox"])
            completed.add(query_id)
            rows.append(row)
    return rows, completed


def _prepare_fingerprint(path: Path, payload: dict[str, Any], *, resume: bool) -> None:
    if resume and path.exists():
        stored = _read_json(path)
        if stored != payload:
            raise ValueError("resume fingerprint mismatch; refusing to mix model runs")
        return
    _write_json(path, payload)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


@dataclass(frozen=True)
class RequestResult:
    row: dict[str, Any]
    runtime: dict[str, Any]


async def _predict_one(
    *,
    client: Any,
    record: AICRecord,
    data_url: str,
    width: int,
    height: int,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
    max_retries: int,
    fallback: str,
) -> RequestResult:
    started = time.perf_counter()
    raw_text = ""
    source = "egm_openai"
    score = 1.0
    error_text: str | None = None
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": _prompt(record.query)},
            ],
        }
    ]
    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=timeout_seconds,
            )
            raw_text = response.choices[0].message.content or ""
            bbox = _box_to_normalized(_extract_box(raw_text), width=width, height=height)
            if bbox is None:
                raise ValueError(f"invalid parsed bbox from output: {raw_text[:200]}")
            latency_ms = (time.perf_counter() - started) * 1000.0
            row = {
                "query_id": record.query_id,
                "query": record.query,
                "visible": record.source["visible"],
                "image_size": [width, height],
                "selection_policy": "egm_openai_single_bbox",
                "selected_index": 0,
                "selected_bbox": bbox,
                "selected_label": record.query,
                "selected_score": score,
                "candidate_count": 1,
                "candidates": [
                    {
                        "bbox": bbox,
                        "label": record.query,
                        "score": score,
                        "source": source,
                    }
                ],
                "raw_output": raw_text,
                "attempts": attempt + 1,
                "parser_fallback": False,
            }
            runtime = {
                "query_id": record.query_id,
                "latency_ms": latency_ms,
                "attempts": attempt + 1,
                "fallback": False,
            }
            return RequestResult(row=row, runtime=runtime)
        except Exception as exc:  # noqa: BLE001 - keep full AIC run resumable.
            error_text = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                await asyncio.sleep(min(2.0 * (attempt + 1), 5.0))
                continue

    if fallback == "error":
        raise RuntimeError(f"{record.query_id}: EGM prediction failed: {error_text}")
    bbox = _fallback_bbox()
    latency_ms = (time.perf_counter() - started) * 1000.0
    row = {
        "query_id": record.query_id,
        "query": record.query,
        "visible": record.source["visible"],
        "image_size": [width, height],
        "selection_policy": "egm_openai_single_bbox",
        "selected_index": 0,
        "selected_bbox": bbox,
        "selected_label": record.query,
        "selected_score": 0.0,
        "candidate_count": 1,
        "candidates": [
            {
                "bbox": bbox,
                "label": record.query,
                "score": 0.0,
                "source": "egm_fallback_center",
            }
        ],
        "raw_output": raw_text,
        "error": error_text,
        "attempts": max_retries + 1,
        "parser_fallback": True,
    }
    runtime = {
        "query_id": record.query_id,
        "latency_ms": latency_ms,
        "attempts": max_retries + 1,
        "fallback": True,
        "error": error_text,
    }
    return RequestResult(row=row, runtime=runtime)


async def _run(args: argparse.Namespace) -> int:
    from openai import AsyncOpenAI

    dataset = AICDataset(dataset_root=args.dataset_root, queries_path=args.queries)
    records = [dataset[index] for index in range(len(dataset))]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        records = records[: args.limit]
    expected = {record.query_id: record for record in records}

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    runtime_path = output / "runtime_events.jsonl"
    fingerprint = {
        "schema": "aic-egm-qwen3-vl-8b-openai-zero-shot-v1",
        "model_name": args.model,
        "base_url": args.base_url,
        "queries_sha256": sha256_file(args.queries),
        "query_count": len(dataset),
        "runner_sha256": sha256_file(Path(__file__)),
        "selection_policy": "egm_openai_single_bbox",
        "training_on_aic": False,
        "modalities": ["visible_rgb", "query_original"],
        "record_limit": args.limit,
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout_seconds,
        "max_retries": args.max_retries,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "jpeg_quality": args.jpeg_quality,
        "fallback": args.fallback,
    }
    _prepare_fingerprint(output / "run_fingerprint.json", fingerprint, resume=args.resume)

    if args.resume:
        evaluated, completed = _read_checkpoint(predictions_path, expected)
    else:
        predictions_path.write_text("", encoding="utf-8")
        runtime_path.write_text("", encoding="utf-8")
        evaluated, completed = [], set()

    pending = [record for record in records if record.query_id not in completed]
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    image_cache: dict[Path, tuple[str, int, int]] = {}
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    processed = 0
    fallback_count = sum(1 for row in evaluated if row.get("parser_fallback"))
    retry_count = sum(max(0, int(row.get("attempts", 1)) - 1) for row in evaluated)

    async def worker(record: AICRecord) -> RequestResult:
        async with semaphore:
            cached = image_cache.get(record.visible_path)
            if cached is None:
                cached = _encode_image_data_url(
                    record.visible_path,
                    jpeg_quality=args.jpeg_quality,
                )
                image_cache[record.visible_path] = cached
            data_url, width, height = cached
            return await _predict_one(
                client=client,
                record=record,
                data_url=data_url,
                width=width,
                height=height,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                fallback=args.fallback,
            )

    cursor = 0
    # Keep bounded batches so completed rows are checkpointed frequently.
    batch_size = max(args.concurrency * 4, 1)
    while cursor < len(pending):
        chunk = pending[cursor : cursor + batch_size]
        cursor += len(chunk)
        results = await asyncio.gather(*(worker(record) for record in chunk))
        rows = [result.row for result in results]
        runtimes = [result.runtime for result in results]
        _append_jsonl(predictions_path, rows)
        _append_jsonl(runtime_path, runtimes)
        evaluated.extend(rows)
        processed += len(rows)
        fallback_count += sum(1 for row in rows if row.get("parser_fallback"))
        retry_count += sum(max(0, int(row.get("attempts", 1)) - 1) for row in rows)
        completed_now = len(evaluated)
        if (
            completed_now == len(records)
            or completed_now % args.progress_every == 0
            or processed == len(rows)
        ):
            elapsed = max(time.perf_counter() - started, 1e-9)
            rate = processed / elapsed if processed else 0.0
            eta = (len(records) - completed_now) / rate if rate > 0 else None
            print(
                json.dumps(
                    {
                        "completed": completed_now,
                        "total": len(records),
                        "processed_this_run": processed,
                        "queries_per_second": round(rate, 4),
                        "eta_seconds": None if eta is None else round(eta, 1),
                        "fallback_count": fallback_count,
                        "retry_count": retry_count,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    wall_seconds = time.perf_counter() - started
    summary = {
        "model": args.model,
        "requested_records": len(records),
        "completed_records": len(evaluated),
        "records_reused": len(evaluated) - processed,
        "records_processed": processed,
        "wall_seconds_this_run": wall_seconds,
        "fallback_count": fallback_count,
        "retry_count": retry_count,
        "no_candidate_count": 0,
        "invalid_bbox_count": 0,
        "min_candidate_count": 1 if evaluated else 0,
        "max_candidate_count": 1 if evaluated else 0,
        "mean_candidate_count": 1.0 if evaluated else 0.0,
        "complete": len(evaluated) == len(records),
    }
    _write_json(output / "run_summary.json", summary)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "runner": str(Path(__file__).resolve()),
        "model": args.model,
        "base_url": args.base_url,
    }
    _write_json(output / "environment.json", environment)

    finalized = None
    if args.finalize:
        if args.limit is not None:
            raise ValueError("cannot finalize a limited probe run")
        finalized = finalize_aic_full_detection(
            dataset=dataset,
            detection_dir=output,
            submission_dir=output / "submission",
        )

    print(
        json.dumps(
            {
                "summary": summary,
                "submission_zip": None if finalized is None else str(finalized.zip_path),
                "submission_audit": None if finalized is None else finalized.audit,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
