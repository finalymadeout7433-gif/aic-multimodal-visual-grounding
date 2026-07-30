from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from aic_baseline.bbox import pixel_to_normalized
from aic_baseline.model_adapters import GroundingDinoExternalPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed, unlabeled AIC candidate smoke test."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def _stable_key(seed: int, *parts: object) -> str:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _domain(record: dict[str, Any]) -> str:
    suffix = Path(str(record["visible"])).suffix.lower()
    return "jpg" if suffix in {".jpg", ".jpeg"} else "png"


def _fixed_sample(
    records: dict[str, dict[str, Any]],
    *,
    per_domain: int,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[tuple[str, dict[str, Any]]]]] = {
        "png": defaultdict(list),
        "jpg": defaultdict(list),
    }
    for query_id, record in records.items():
        grouped[_domain(record)][str(record["visible"])].append(
            (query_id, record)
        )
    selected: list[tuple[str, dict[str, Any]]] = []
    for domain in ("png", "jpg"):
        image_keys = sorted(
            grouped[domain],
            key=lambda key: _stable_key(seed, domain, key),
        )
        if len(image_keys) < per_domain:
            raise ValueError(
                f"{domain} has only {len(image_keys)} unique images"
            )
        for image_key in image_keys[:per_domain]:
            expressions = sorted(
                grouped[domain][image_key],
                key=lambda item: _stable_key(seed, item[0]),
            )
            selected.append(expressions[0])
    return selected


def main() -> int:
    args = parse_args()
    records = json.loads(args.queries.read_text(encoding="utf-8"))
    sample = _fixed_sample(
        records,
        per_domain=args.per_domain,
        seed=args.seed,
    )
    predictor = GroundingDinoExternalPredictor(
        model_path=args.model_path,
        device="cuda",
        dtype=torch.float32,
        box_threshold=0.15,
        text_threshold=0.15,
        max_candidates=20,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    output_path.write_text("", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    candidate_counts: list[int] = []
    valid_count = 0
    domain_counts: Counter[str] = Counter()
    with output_path.open("a", encoding="utf-8", newline="\n") as sink:
        for query_id, record in sample:
            image_path = args.dataset_root / str(record["visible"])
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            started = time.perf_counter()
            prediction = predictor.predict(
                image=image,
                query=str(record["query"]),
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            candidate_counts.append(len(prediction.candidates))
            latencies.append(latency_ms)
            domain = _domain(record)
            domain_counts[domain] += 1
            selected_bbox = None
            error = None
            if prediction.candidates:
                try:
                    selected_bbox = pixel_to_normalized(
                        prediction.candidates[0].pixel_bbox,
                        width=image.width,
                        height=image.height,
                    )
                    if all(math.isfinite(value) for value in selected_bbox):
                        valid_count += 1
                except (TypeError, ValueError) as exc:
                    error = str(exc)
            debug = {
                "query_id": query_id,
                "visible": record["visible"],
                "query": record["query"],
                "domain": domain,
                "candidate_count": len(prediction.candidates),
                "selected_bbox": selected_bbox,
                "latency_ms": latency_ms,
                "error": error,
            }
            sink.write(json.dumps(debug, ensure_ascii=False) + "\n")
            sink.flush()
    total = len(sample)
    summary = {
        "records": total,
        "seed": args.seed,
        "domain_counts": dict(domain_counts),
        "valid_bbox_rate": valid_count / total,
        "no_candidate_rate": sum(
            count == 0 for count in candidate_counts
        )
        / total,
        "candidate_count_mean": statistics.fmean(candidate_counts),
        "latency_ms_mean": statistics.fmean(latencies),
        "latency_ms_p95": sorted(latencies)[
            min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)
        ],
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
