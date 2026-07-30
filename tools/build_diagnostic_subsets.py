from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from aic_baseline.diagnostics import (
    build_core_subset,
    build_tile_probe,
    read_jsonl,
    write_jsonl,
)
from aic_baseline.external_data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed external diagnostic subsets."
    )
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--per-dataset", type=int, default=500)
    return parser.parse_args()


def _subset_fingerprint(records: list[dict[str, object]]) -> str:
    payload = "\n".join(
        f"{record['dataset']}\t{record['query_id']}\t{record['image_key']}"
        for record in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.validation_manifest)
    core = build_core_subset(
        records,
        per_dataset=args.per_dataset,
        seed=args.seed,
    )
    tile = build_tile_probe(core, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    core_path = args.output_dir / "core_eval_1500.jsonl"
    tile_path = args.output_dir / "tile_probe_600.jsonl"
    write_jsonl(core_path, core)
    write_jsonl(tile_path, tile)
    summary = {
        "seed": args.seed,
        "source_manifest_sha256": sha256_file(args.validation_manifest),
        "core": {
            "records": len(core),
            "unique_image_keys": len(
                {str(record["image_key"]) for record in core}
            ),
            "dataset_counts": dict(
                Counter(str(record["dataset"]) for record in core)
            ),
            "selection_sha256": _subset_fingerprint(core),
            "file_sha256": sha256_file(core_path),
        },
        "tile_probe": {
            "records": len(tile),
            "dataset_counts": dict(
                Counter(str(record["dataset"]) for record in tile)
            ),
            "group_counts": dict(
                Counter(str(record["probe_group"]) for record in tile)
            ),
            "selection_sha256": _subset_fingerprint(tile),
            "file_sha256": sha256_file(tile_path),
        },
    }
    (args.output_dir / "subset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
