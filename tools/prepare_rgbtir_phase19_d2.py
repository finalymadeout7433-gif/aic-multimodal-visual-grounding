from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_rgbtir.data import read_jsonl_records, write_jsonl  # noqa: E402
from aic_rgbtir.phase19_d2 import (  # noqa: E402
    D2CandidateSpec,
    D2GeometryRetentionObjective,
    build_d2_multiquery_split,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _offline_sensitivity(cache_path: Path) -> dict[str, Any]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    rows = payload["records"]
    layers = ("8", "16", "24")
    base = {
        layer: torch.stack([row["pooled"]["base"][layer].float() for row in rows])
        for layer in layers
    }
    adapted = {
        layer: torch.stack([row["pooled"]["adapted"][layer].float() for row in rows])
        for layer in layers
    }
    objective = D2GeometryRetentionObjective(D2CandidateSpec.d2(0.25))
    generator = torch.Generator().manual_seed(20260816)
    sample_count = min(256, len(rows))
    query_ids = torch.randperm(len(rows), generator=generator)[:sample_count]
    anchor_ids = torch.randperm(len(rows), generator=generator)[:256]
    values: dict[str, list[float]] = {"base_identity": [], "adapted": [], "collapsed": []}
    per_layer: dict[str, dict[str, float]] = {}
    for layer in layers:
        anchors = {name: base[name][anchor_ids] for name in layers}
        layer_weights = {name: (1.0 if name == layer else 1e-8) for name in layers}
        layer_objective = D2GeometryRetentionObjective(
            D2CandidateSpec.d2(0.25), layer_weights=layer_weights
        )
        layer_values = {name: [] for name in values}
        collapsed_vector = adapted[layer].mean(dim=0)
        for index in query_ids.tolist():
            base_row = {name: base[name][index] for name in layers}
            adapted_row = {name: adapted[name][index] for name in layers}
            collapsed_row = dict(adapted_row)
            collapsed_row[layer] = collapsed_vector
            layer_values["base_identity"].append(
                float(layer_objective.geometry_loss(base_row, base_row, anchors))
            )
            layer_values["adapted"].append(
                float(layer_objective.geometry_loss(adapted_row, base_row, anchors))
            )
            layer_values["collapsed"].append(
                float(layer_objective.geometry_loss(collapsed_row, base_row, anchors))
            )
        per_layer[layer] = {
            name: sum(items) / len(items) for name, items in layer_values.items()
        }
    for index in query_ids.tolist():
        anchors = {name: base[name][anchor_ids] for name in layers}
        base_row = {name: base[name][index] for name in layers}
        adapted_row = {name: adapted[name][index] for name in layers}
        collapsed_row = {
            name: adapted[name].mean(dim=0) for name in layers
        }
        values["base_identity"].append(
            float(objective.geometry_loss(base_row, base_row, anchors))
        )
        values["adapted"].append(
            float(objective.geometry_loss(adapted_row, base_row, anchors))
        )
        values["collapsed"].append(
            float(objective.geometry_loss(collapsed_row, base_row, anchors))
        )
    means = {name: sum(items) / len(items) for name, items in values.items()}
    return {
        "status": (
            "D2_OFFLINE_SENSITIVITY_PASS"
            if means["base_identity"] < 1e-8
            and means["adapted"] > means["base_identity"]
            and means["collapsed"] > means["adapted"]
            else "D2_OFFLINE_SENSITIVITY_FAIL"
        ),
        "claim_boundary": "Diagnostic sensitivity only; this is not a training or platform result.",
        "cache_fingerprint": payload.get("fingerprint"),
        "record_count": len(rows),
        "sample_count": sample_count,
        "anchor_count": len(anchor_ids),
        "mean_geometry_loss": means,
        "per_layer_mean_geometry_loss": per_layer,
    }


def run(args: argparse.Namespace) -> int:
    manifest_root = args.manifest_root.resolve()
    output_root = args.output_root.resolve()
    manifests = output_root / "manifests"
    full = read_jsonl_records(manifest_root / "repair_full_train.jsonl")
    probe = read_jsonl_records(manifest_root / "repair_probe_train.jsonl")
    semantic_dev = read_jsonl_records(manifest_root / "repair_dev.jsonl")
    official = read_jsonl_records(manifest_root / "official_val.jsonl")
    split = build_d2_multiquery_split(
        full,
        probe,
        pair_count=args.multiquery_pairs,
        seed=args.seed,
    )
    outputs = {
        "d2_probe_train.jsonl": probe,
        "d2_semantic_dev.jsonl": semantic_dev,
        "d2_multiquery_dev.jsonl": split.multiquery_dev,
        "d2_remaining_full_train.jsonl": split.remaining_full_train,
        "official_val.jsonl": official,
    }
    for name, records in outputs.items():
        write_jsonl(manifests / name, records)

    pair_sets = {
        name: {record.image_pair_key for record in records}
        for name, records in outputs.items()
    }
    overlaps = {
        f"{left}__{right}": len(pair_sets[left] & pair_sets[right])
        for index, left in enumerate(outputs)
        for right in list(outputs)[index + 1 :]
    }
    expected_allowed = {"d2_probe_train.jsonl__d2_remaining_full_train.jsonl"}
    forbidden = {key: value for key, value in overlaps.items() if value and key not in expected_allowed}
    if forbidden:
        raise RuntimeError(f"D2 split leakage: {forbidden}")

    summary = {
        "status": "PHASE_19_D2_LOCAL_DATA_READY",
        "seed": args.seed,
        "multiquery_pair_target": args.multiquery_pairs,
        "manifests": {
            name: {
                "records": len(records),
                "pairs": len(pair_sets[name]),
                "source_records": dict(Counter(record.source_dataset for record in records)),
                "sha256": _sha256(manifests / name),
            }
            for name, records in outputs.items()
        },
        "pair_overlaps": overlaps,
        "candidate_specs": [candidate.__dict__ for candidate in D2CandidateSpec.defaults()],
        "selection_policy": (
            "Candidates are selected on semantic-dev plus multiquery-dev; official val remains sealed."
        ),
    }
    _write_json(output_root / "split_summary.json", summary)
    sensitivity = _offline_sensitivity(args.embedding_cache.resolve())
    _write_json(output_root / "offline_sensitivity.json", sensitivity)
    if sensitivity["status"] != "D2_OFFLINE_SENSITIVITY_PASS":
        raise RuntimeError(sensitivity["status"])
    hashes = {
        path.relative_to(output_root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    _write_json(output_root / "sha256_manifest.json", {"schema_version": 1, "files": hashes})
    print(json.dumps({"summary": summary, "sensitivity": sensitivity}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare fixed Phase 1.9-D2 assets")
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=REPO_ROOT / "outputs/aic_rgbtir_phase16_v1/manifests",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=REPO_ROOT
        / "outputs/aic_rgbtir_phase18r_audit_inputs_20260816/outputs/aic_rgbtir_phase18_full_v1/official_val/embedding_cache.pt",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs/aic_rgbtir_phase19_d2_v1",
    )
    parser.add_argument("--multiquery-pairs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260816)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
