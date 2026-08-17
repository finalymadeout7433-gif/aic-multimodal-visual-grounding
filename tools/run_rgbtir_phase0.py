from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_rgbtir.data import (  # noqa: E402
    ManifestBundle,
    RGBTRecord,
    build_manifest_bundle,
    build_tracer_records,
    read_jsonl_records,
    write_jsonl,
)
from aic_rgbtir.processing import PairedRGBTProcessor  # noqa: E402
from aic_rgbtir.validation import (  # noqa: E402
    audit_aic_pairs,
    audit_rgbt_pairs,
    build_readiness_report,
    build_sha256_manifest,
    exclusion_reason_counts,
    run_phase0_pytest,
    sha256_file,
    summarize_pair_audit,
    validate_zero_gate_equivalence,
    write_csv,
    write_json,
)


STAGES = (
    "verify-inputs",
    "build-manifests",
    "audit-pairs",
    "build-tracer",
    "validate-processor",
    "validate-model-bridge",
    "build-report",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 0 config must be a YAML mapping")
    return payload


def _annotation_paths(rgbt_root: Path) -> list[Path]:
    return [
        rgbt_root / f"rgbtvg_{dataset}" / f"rgbtvg_{dataset}_{split}.pth"
        for dataset in ("flir", "m3fd", "mfad")
        for split in ("train", "val", "test")
    ]


def _transformers_source() -> Path:
    import inspect

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    return Path(inspect.getfile(Qwen3VLVisionModel)).resolve()


def _verify_inputs(
    *, config: dict[str, Any], config_path: Path, output_root: Path
) -> dict[str, Any]:
    paths = config["paths"]
    rgbt_root = _resolve_path(paths["rgbt_root"], base=config_path.parent)
    aic_root = _resolve_path(paths["aic_root"], base=config_path.parent)
    aic_queries = _resolve_path(paths["aic_queries"], base=config_path.parent)
    required = [rgbt_root / "image_data", aic_root / "Images", aic_queries]
    required.extend(_annotation_paths(rgbt_root))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 0 inputs: {missing}")

    query_payload = _read_json(aic_queries)
    if not isinstance(query_payload, dict) or len(query_payload) != 9555:
        raise ValueError(
            f"AIC queries must contain 9,555 records, got {len(query_payload)}"
        )
    upstream = _transformers_source()
    annotation_hashes = {
        path.relative_to(rgbt_root).as_posix(): sha256_file(path)
        for path in _annotation_paths(rgbt_root)
    }
    fingerprint = {
        "schema_version": 1,
        "config_sha256": _canonical_hash(config),
        "rgbt_root": str(rgbt_root),
        "aic_root": str(aic_root),
        "aic_queries": str(aic_queries),
        "aic_query_count": len(query_payload),
        "aic_queries_sha256": sha256_file(aic_queries),
        "rgbt_annotation_sha256": annotation_hashes,
        "transformers_source": str(upstream),
        "transformers_source_sha256_before": sha256_file(upstream),
    }
    target = output_root / "input_fingerprint.json"
    if target.is_file():
        previous = _read_json(target)
        stable_fields = {
            key: value
            for key, value in fingerprint.items()
            if key != "transformers_source_sha256_before"
        }
        previous_stable = {
            key: value
            for key, value in previous.items()
            if key != "transformers_source_sha256_before"
        }
        if previous_stable != stable_fields:
            raise RuntimeError(
                "--resume input fingerprint differs from the existing Phase 0 run"
            )
    write_json(target, fingerprint)
    return fingerprint


def _bundle_from_outputs(manifest_root: Path) -> ManifestBundle:
    train_all = tuple(read_jsonl_records(manifest_root / "train_all.jsonl"))
    train_clean = tuple(read_jsonl_records(manifest_root / "train_clean.jsonl"))
    val = tuple(read_jsonl_records(manifest_root / "val_official.jsonl"))
    test = tuple(read_jsonl_records(manifest_root / "test_official.jsonl"))
    excluded = tuple(read_jsonl_records(manifest_root / "excluded_train.jsonl"))
    all_records = train_all + val + test
    split_membership: dict[str, set[str]] = {}
    for record in all_records:
        split_membership.setdefault(record.image_pair_key, set()).add(record.split)
    cross = tuple(
        sorted(key for key, splits in split_membership.items() if len(splits) > 1)
    )
    return ManifestBundle(
        train_all=train_all,
        train_clean=train_clean,
        val_official=val,
        test_official=test,
        excluded_train=excluded,
        raw_count=len(all_records),
        unique_pair_count=len({record.image_pair_key for record in all_records}),
        cross_split_pairs=cross,
    )


def _build_manifests(rgbt_root: Path, manifest_root: Path) -> tuple[ManifestBundle, dict[str, Any]]:
    bundle = build_manifest_bundle(rgbt_root)
    manifest_root.mkdir(parents=True, exist_ok=True)
    paths_and_records: list[tuple[str, Iterable[RGBTRecord]]] = [
        ("train_all.jsonl", bundle.train_all),
        ("train_clean.jsonl", bundle.train_clean),
        ("val_official.jsonl", bundle.val_official),
        ("test_official.jsonl", bundle.test_official),
        ("excluded_train.jsonl", bundle.excluded_train),
    ]
    for name, records in paths_and_records:
        write_jsonl(manifest_root / name, records)
    summary = {
        "raw_count": bundle.raw_count,
        "train_all_count": len(bundle.train_all),
        "train_clean_count": len(bundle.train_clean),
        "val_count": len(bundle.val_official),
        "test_count": len(bundle.test_official),
        "excluded_train_count": len(bundle.excluded_train),
        "unique_pair_count": bundle.unique_pair_count,
        "cross_split_pairs": list(bundle.cross_split_pairs),
        "exclusion_reason_counts": exclusion_reason_counts(bundle),
    }
    write_json(manifest_root / "manifest_summary.json", summary)
    return bundle, summary


def _combined_tracer_summary(
    train: list[RGBTRecord], val: list[RGBTRecord], train_summary: dict[str, Any], val_summary: dict[str, Any]
) -> dict[str, Any]:
    records = train + val
    condition_counts = Counter()
    source_counts = Counter()
    for record in records:
        source_counts[record.source_dataset] += 1
        condition_counts.update(
            {
                "small": record.object_size == "SS",
                "low_light": record.illumination in {"WL", "VWL"},
                "adverse_weather": record.weather in {"FY", "RY"},
                "high_occlusion": record.occlusion == "HO",
            }
        )
    minima = {
        "small": 200,
        "low_light": 150,
        "adverse_weather": 50,
        "high_occlusion": 50,
    }
    actual = {key: int(condition_counts[key]) for key in minima}
    unmet = {
        key: {"required": target, "actual": actual[key]}
        for key, target in minima.items()
        if actual[key] < target
    }
    return {
        "train": train_summary,
        "val": val_summary,
        "combined": {
            "total": len(records),
            "unique_pair_count": len({record.image_pair_key for record in records}),
            "source_counts": dict(sorted(source_counts.items())),
            "condition_counts": actual,
            "condition_minima": minima,
            "unmet_minima": unmet,
            "train_val_pair_overlap": sorted(
                {record.image_pair_key for record in train}
                & {record.image_pair_key for record in val}
            ),
        },
    }


def _build_tracer(bundle: ManifestBundle, manifest_root: Path) -> tuple[list[RGBTRecord], list[RGBTRecord], dict[str, Any]]:
    train, train_summary = build_tracer_records(
        bundle.train_clean,
        total=400,
        source_quotas={"flir": 134, "m3fd": 133, "mfad": 133},
        condition_minima={
            "small": 160,
            "low_light": 120,
            "adverse_weather": 40,
            "high_occlusion": 40,
        },
        seed=20260812,
    )
    val, val_summary = build_tracer_records(
        bundle.val_official,
        total=100,
        source_quotas={"flir": 34, "m3fd": 33, "mfad": 33},
        condition_minima={
            "small": 40,
            "low_light": 30,
            "adverse_weather": 10,
            "high_occlusion": 10,
        },
        seed=20260812,
    )
    summary = _combined_tracer_summary(train, val, train_summary, val_summary)
    write_jsonl(manifest_root / "tracer_train_400.jsonl", train)
    write_jsonl(manifest_root / "tracer_val_100.jsonl", val)
    write_json(manifest_root / "tracer_summary.json", summary)
    return train, val, summary


def _validate_processor(
    *, records: list[RGBTRecord], rgbt_root: Path, processor_config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    processor = PairedRGBTProcessor(**processor_config)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    max_roundtrip_error = 0.0
    for record in records:
        try:
            batch = processor.process(record, root=rgbt_root)
            original_pixel = (
                record.bbox_xyxy_normalized[0] * record.width,
                record.bbox_xyxy_normalized[1] * record.height,
                record.bbox_xyxy_normalized[2] * record.width,
                record.bbox_xyxy_normalized[3] * record.height,
            )
            resized = batch.shared_geometry_transform.original_to_resized(original_pixel)
            restored = batch.shared_geometry_transform.resized_to_original(resized)
            error = max(abs(a - b) for a, b in zip(original_pixel, restored))
            max_roundtrip_error = max(max_roundtrip_error, error)
            expected_patch_count = int(batch.image_grid_thw.prod().item())
            merge = processor.spatial_merge_size
            expected_merged_count = expected_patch_count // (merge * merge)
            if batch.ir_patch_valid_mask.numel() != expected_patch_count:
                raise ValueError("IR patch mask count does not match image_grid_thw")
            if batch.ir_merged_valid_mask.numel() != expected_merged_count:
                raise ValueError("IR merged-token mask count is invalid")
            rows.append(
                {
                    "record_id": record.record_id,
                    "image_pair_key": record.image_pair_key,
                    "grid_thw": batch.image_grid_thw.squeeze(0).tolist(),
                    "patch_count": expected_patch_count,
                    "merged_token_count": expected_merged_count,
                    "ir_valid_ratio": batch.ir_valid_ratio,
                    "ir_usable": batch.ir_usable,
                    "bbox_roundtrip_max_error_pixel": error,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            failure = {"record_id": record.record_id, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            rows.append(
                {
                    "record_id": record.record_id,
                    "image_pair_key": record.image_pair_key,
                    "grid_thw": [],
                    "patch_count": 0,
                    "merged_token_count": 0,
                    "ir_valid_ratio": None,
                    "ir_usable": False,
                    "bbox_roundtrip_max_error_pixel": None,
                    "status": "error",
                    "error": failure["error"],
                }
            )
    summary = {
        "record_count": len(records),
        "success_count": len(records) - len(failures),
        "failure_count": len(failures),
        "bbox_roundtrip_max_error_pixel": max_roundtrip_error,
        "all_grid_masks_valid": not failures,
        "failures": failures,
    }
    return summary, rows


def _write_grid_mask_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def run(config_path: Path, *, resume: bool) -> str:
    config = _load_config(config_path)
    paths = config["paths"]
    output_root = _resolve_path(paths["output_root"], base=config_path.parent)
    report_path = _resolve_path(paths["report_path"], base=config_path.parent)
    rgbt_root = _resolve_path(paths["rgbt_root"], base=config_path.parent)
    aic_root = _resolve_path(paths["aic_root"], base=config_path.parent)
    aic_queries = _resolve_path(paths["aic_queries"], base=config_path.parent)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_root = output_root / "manifests"
    audit_root = output_root / "audits"
    processor_config = dict(config.get("processor", {}))
    workers = int(config.get("audit", {}).get("workers", 4))

    print("[1/7] verify-inputs", flush=True)
    fingerprint = _verify_inputs(
        config=config, config_path=config_path, output_root=output_root
    )

    print("[2/7] build-manifests", flush=True)
    summary_path = manifest_root / "manifest_summary.json"
    if resume and summary_path.is_file():
        bundle = _bundle_from_outputs(manifest_root)
        manifest_summary = _read_json(summary_path)
    else:
        bundle, manifest_summary = _build_manifests(rgbt_root, manifest_root)

    print("[3/7] audit-pairs", flush=True)
    rgbt_csv = audit_root / "rgbt_pair_audit.csv"
    aic_csv = audit_root / "aic_pair_audit.csv"
    audit_summary_path = audit_root / "pair_audit_summary.json"
    if resume and rgbt_csv.is_file() and aic_csv.is_file() and audit_summary_path.is_file():
        pair_summary = _read_json(audit_summary_path)
        rgbt_summary = pair_summary["rgbt"]
        aic_summary = pair_summary["aic"]
    else:
        rgbt_rows = audit_rgbt_pairs(
            bundle.train_all + bundle.val_official + bundle.test_official,
            root=rgbt_root,
            min_pixels=int(processor_config["min_pixels"]),
            max_pixels=int(processor_config["max_pixels"]),
            black_threshold=int(processor_config.get("black_threshold", 3)),
            workers=workers,
        )
        aic_rows = audit_aic_pairs(
            dataset_root=aic_root,
            queries_path=aic_queries,
            min_pixels=int(processor_config["min_pixels"]),
            max_pixels=int(processor_config["max_pixels"]),
            black_threshold=int(processor_config.get("black_threshold", 3)),
            workers=workers,
        )
        write_csv(rgbt_csv, rgbt_rows)
        write_csv(aic_csv, aic_rows)
        rgbt_summary = summarize_pair_audit(rgbt_rows)
        aic_summary = summarize_pair_audit(aic_rows)
        write_json(audit_summary_path, {"rgbt": rgbt_summary, "aic": aic_summary})

    print("[4/7] build-tracer", flush=True)
    tracer_summary_path = manifest_root / "tracer_summary.json"
    if resume and tracer_summary_path.is_file():
        train_tracer = read_jsonl_records(manifest_root / "tracer_train_400.jsonl")
        val_tracer = read_jsonl_records(manifest_root / "tracer_val_100.jsonl")
        tracer_summary = _read_json(tracer_summary_path)
    else:
        train_tracer, val_tracer, tracer_summary = _build_tracer(bundle, manifest_root)

    print("[5/7] validate-processor", flush=True)
    processor_summary_path = output_root / "processor_summary.json"
    grid_mask_path = audit_root / "grid_mask_audit.jsonl"
    if resume and processor_summary_path.is_file() and grid_mask_path.is_file():
        processor_summary = _read_json(processor_summary_path)
    else:
        processor_summary, grid_rows = _validate_processor(
            records=train_tracer + val_tracer,
            rgbt_root=rgbt_root,
            processor_config=processor_config,
        )
        write_json(processor_summary_path, processor_summary)
        _write_grid_mask_jsonl(grid_mask_path, grid_rows)

    print("[6/7] validate-model-bridge", flush=True)
    fp32_equivalence = validate_zero_gate_equivalence(
        device="cpu", dtype=torch.float32
    )
    bf16_equivalence = validate_zero_gate_equivalence(
        device="cpu", dtype=torch.bfloat16
    )
    equivalence_summary = {
        "passed": fp32_equivalence["passed"] and bf16_equivalence["passed"],
        "final_max_abs_diff": max(
            fp32_equivalence["final_max_abs_diff"],
            bf16_equivalence["final_max_abs_diff"],
        ),
        "deepstack_max_abs_diff": max(
            fp32_equivalence["deepstack_max_abs_diff"],
            bf16_equivalence["deepstack_max_abs_diff"],
        ),
        "rgb_only_max_abs_diff": max(
            fp32_equivalence["rgb_only_max_abs_diff"],
            bf16_equivalence["rgb_only_max_abs_diff"],
        ),
        "fp32": fp32_equivalence,
        "bf16": bf16_equivalence,
    }
    write_json(output_root / "equivalence_summary.json", equivalence_summary)

    print("[7/7] build-report", flush=True)
    pytest_summary = run_phase0_pytest(REPO_ROOT)
    write_json(output_root / "pytest_summary.json", pytest_summary)
    upstream_after = sha256_file(Path(fingerprint["transformers_source"]))
    upstream_unchanged = upstream_after == fingerprint["transformers_source_sha256_before"]
    if not upstream_unchanged:
        pytest_summary["passed"] = False
        pytest_summary["stderr"] += "\nTransformers Qwen source changed during Phase 0."
    report, status, blockers = build_readiness_report(
        manifest_summary=manifest_summary,
        rgbt_summary=rgbt_summary,
        aic_summary=aic_summary,
        processor_summary=processor_summary,
        equivalence_summary=equivalence_summary,
        tracer_summary=tracer_summary,
        pytest_summary=pytest_summary,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    run_summary = {
        "schema_version": 1,
        "status": status,
        "blockers": blockers,
        "stages": list(STAGES),
        "manifest_summary": manifest_summary,
        "rgbt_pair_summary": rgbt_summary,
        "aic_pair_summary": aic_summary,
        "processor_summary": processor_summary,
        "equivalence_summary": equivalence_summary,
        "tracer_summary": tracer_summary,
        "pytest_passed": pytest_summary["passed"],
        "transformers_source_unchanged": upstream_unchanged,
        "report_path": str(report_path),
    }
    write_json(output_root / "run_summary.json", run_summary)
    sha_manifest = build_sha256_manifest(output_root)
    write_json(output_root / "sha256_manifest.json", sha_manifest)
    print(f"Phase 0 result: {status}", flush=True)
    print(f"Report: {report_path}", flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate the AIC RGB-TIR Phase 0 engineering seam."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    status = run(args.config.resolve(), resume=args.resume)
    return 0 if status == "PHASE_0_GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
