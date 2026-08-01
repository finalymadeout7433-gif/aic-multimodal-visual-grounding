from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from aic_baseline.diagnostics import read_jsonl, write_jsonl
from aic_baseline.external_data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, image-isolated model comparison probe."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def _stable_key(seed: int, record: dict[str, Any]) -> str:
    payload = "\t".join(
        (
            str(seed),
            str(record.get("dataset", "")),
            str(record.get("image_key", "")),
            str(record.get("query_id", "")),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_probe(
    records: Sequence[dict[str, Any]],
    *,
    per_dataset: int,
    seed: int,
) -> list[dict[str, Any]]:
    if per_dataset <= 0:
        raise ValueError("per_dataset must be positive")
    datasets = sorted({str(record["dataset"]) for record in records})
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    for dataset in datasets:
        candidates = sorted(
            (record for record in records if str(record["dataset"]) == dataset),
            key=lambda record: _stable_key(seed, record),
        )
        chosen: list[dict[str, Any]] = []
        for record in candidates:
            image_key = str(record["image_key"])
            if image_key in used_images:
                continue
            copied = dict(record)
            copied["model_comparison_probe"] = True
            chosen.append(copied)
            used_images.add(image_key)
            if len(chosen) == per_dataset:
                break
        if len(chosen) != per_dataset:
            raise ValueError(
                f"dataset {dataset} provided {len(chosen)} unique images, "
                f"expected {per_dataset}"
            )
        selected.extend(chosen)
    selected.sort(key=lambda row: (str(row["dataset"]), str(row["query_id"])))
    return selected


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.source)
    selected = select_probe(
        records,
        per_dataset=args.per_dataset,
        seed=args.seed,
    )
    write_jsonl(args.output, selected)
    summary = {
        "status": "ok",
        "seed": args.seed,
        "per_dataset": args.per_dataset,
        "record_count": len(selected),
        "unique_image_count": len({row["image_key"] for row in selected}),
        "dataset_counts": dict(
            sorted(Counter(str(row["dataset"]) for row in selected).items())
        ),
        "source_path": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
