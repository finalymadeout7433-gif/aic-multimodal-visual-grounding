from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aic_baseline.external_data import (
    DatasetSpec,
    audit_extracted_zip,
    build_dataset_records,
    build_image_isolated_splits,
    extract_verified_zip,
    sha256_file,
    verify_zip_archive,
    write_jsonl,
    write_preview,
)


ARCHIVES = {
    "coco_train2014": (
        "shared/COCO2014/archives/train2014.zip",
        "shared/COCO2014/images",
    ),
    "sunrgbd": (
        "02_RGBD_Grounding/SUN-Spot/SUNRGBD_base/archives/SUNRGBD.zip",
        "02_RGBD_Grounding/SUN-Spot/SUNRGBD_base/raw",
    ),
}


def _dataset_specs(root: Path) -> list[DatasetSpec]:
    coco_images = root / "shared/COCO2014/images/train2014"
    family = root / "01_RGB_Grounding/RefCOCO_family/raw"
    specs = [
        DatasetSpec(
            name="refcoco",
            refs_path=family / "refcoco/refs(unc).p",
            refs_sha256="D4A8DD3152F130127924F9D0F0EA30F4DC43EC19E1B47BC55AF972F90614F8BA",
            instances_path=family / "refcoco/instances.json",
            image_root=coco_images,
            image_prefix="shared/COCO2014/images/train2014",
            image_namespace="coco2014",
        ),
        DatasetSpec(
            name="refcoco_plus",
            refs_path=family / "refcoco+/refs(unc).p",
            refs_sha256="7DE24C182449758B9D9AE87D968B1F25831236A54D38D45408E2E0FDD318E802",
            instances_path=family / "refcoco+/instances.json",
            image_root=coco_images,
            image_prefix="shared/COCO2014/images/train2014",
            image_namespace="coco2014",
        ),
        DatasetSpec(
            name="refcocog",
            refs_path=family / "refcocog/refs(umd).p",
            refs_sha256="0331C7533537B67C2F7AC8C8BAB0DA2D379D1754C6F5C110FD79F70E17E7BDDB",
            instances_path=family / "refcocog/instances.json",
            image_root=coco_images,
            image_prefix="shared/COCO2014/images/train2014",
            image_namespace="coco2014",
        ),
        DatasetSpec(
            name="grefcoco",
            refs_path=root / "01_RGB_Grounding/gRefCOCO/archives/grefs_unc.json",
            refs_sha256="CC37C5FF95373C78A6A3F98B4C7BC67FDE387EA8514752A1392DB64223EB3366",
            instances_path=root / "01_RGB_Grounding/gRefCOCO/archives/instances.json",
            image_root=coco_images,
            image_prefix="shared/COCO2014/images/train2014",
            image_namespace="coco2014",
        ),
    ]
    sun_root = root / "02_RGBD_Grounding/SUN-Spot/SUNRGBD_base/raw"
    specs.append(
        DatasetSpec(
            name="sunspot",
            refs_path=root / "02_RGBD_Grounding/SUN-Spot/archives/refs(boulder).p",
            refs_sha256="6EE465478306DFC8776CACF86B4AE570A079991212E269C22D7829AFA2A3C18B",
            instances_path=root / "02_RGBD_Grounding/SUN-Spot/archives/instances.json",
            image_root=sun_root,
            image_prefix="02_RGBD_Grounding/SUN-Spot/SUNRGBD_base/raw",
            image_namespace="sunrgbd",
            depth_root=sun_root,
            depth_prefix="02_RGBD_Grounding/SUN-Spot/SUNRGBD_base/raw",
        )
    )
    return specs


def command_verify(root: Path, *, check_crc: bool) -> None:
    results: dict[str, Any] = {}
    for name, (archive_relpath, _) in ARCHIVES.items():
        result = verify_zip_archive(root / archive_relpath, check_crc=check_crc)
        results[name] = result.__dict__
        print(
            f"[verified] {name}: {result.entry_count} entries, "
            f"sha256={result.sha256}",
            flush=True,
        )
        if result.bad_member is not None:
            raise ValueError(f"{name} CRC 失败: {result.bad_member}")
    output = root / "manifests/archive_verification.json"
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def command_extract(root: Path) -> None:
    audits: dict[str, Any] = {}
    for name, (archive_relpath, destination_relpath) in ARCHIVES.items():
        archive_path = root / archive_relpath
        destination = root / destination_relpath
        count = extract_verified_zip(
            archive_path,
            destination,
        )
        print(f"[extracted] {name}: {count} new files", flush=True)
        audit = audit_extracted_zip(archive_path, destination)
        audits[name] = audit.__dict__
        print(
            f"[audited] {name}: missing={audit.missing_count}, "
            f"size_mismatch={audit.size_mismatch_count}, "
            f"sanitized={audit.sanitized_member_count}, "
            f"collisions={audit.sanitized_collision_count}",
            flush=True,
        )
        if (
            audit.missing_count
            or audit.size_mismatch_count
            or audit.sanitized_collision_count
        ):
            raise ValueError(f"{name} 解压审计失败")
    (root / "manifests/extraction_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_split_manifests(
    output_root: Path,
    dataset_name: str,
    records: list[dict[str, Any]],
) -> None:
    for split in ("train", "validation", "holdout"):
        write_jsonl(
            output_root / f"{dataset_name}_{split}.jsonl",
            (record for record in records if record["split"] == split),
        )


def command_prepare(root: Path, *, preview_count: int) -> None:
    manifest_root = root / "manifests/processed"
    previews_root = root / "previews"
    all_records: dict[str, list[dict[str, Any]]] = {}
    dataset_summaries: dict[str, Any] = {}
    preview_counts: dict[str, int] = {}
    for spec in _dataset_specs(root):
        records, summary = build_dataset_records(spec)
        assigned, split_summary = build_image_isolated_splits(records)
        summary["split_summary"] = split_summary
        all_records[spec.name] = assigned
        dataset_summaries[spec.name] = summary
        _write_split_manifests(manifest_root, spec.name, assigned)
        for stale_preview in previews_root.glob(f"{spec.name}_*.jpg"):
            stale_preview.unlink()
        preview_candidates: list[dict[str, Any]] = []
        preview_image_keys: set[str] = set()
        for record in assigned:
            image_key = str(record["image_key"])
            if not record["image_exists"] or image_key in preview_image_keys:
                continue
            preview_candidates.append(record)
            preview_image_keys.add(image_key)
            if len(preview_candidates) >= preview_count:
                break
        preview_counts[spec.name] = len(preview_candidates)
        for index, record in enumerate(preview_candidates, start=1):
            write_preview(
                data_root=root,
                record=record,
                output_path=previews_root
                / f"{spec.name}_{index:02d}_{record['split']}.jpg",
            )
        print(
            f"[prepared] {spec.name}: {len(assigned)} expressions",
            flush=True,
        )

    core_names = ("refcoco", "refcoco_plus", "refcocog")
    core_records = [
        record for name in core_names for record in all_records[name]
    ]
    isolated_core, core_split_summary = build_image_isolated_splits(core_records)
    _write_split_manifests(manifest_root, "rgb_core_v1", isolated_core)

    image_missing = sum(
        1 for records in all_records.values() for record in records
        if not record["image_exists"]
    )
    depth_missing = sum(
        1 for record in all_records["sunspot"] if record["depth_exists"] is False
    )
    summary = {
        "schema_version": 1,
        "datasets": dataset_summaries,
        "rgb_core_v1": {
            "datasets": list(core_names),
            "split_summary": core_split_summary,
        },
        "total_expression_count": sum(
            len(records) for records in all_records.values()
        ),
        "missing_image_record_count": image_missing,
        "missing_depth_record_count": depth_missing,
        "preview_counts": preview_counts,
    }
    (manifest_root / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_hashes = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(manifest_root.glob("*.jsonl"))
    }
    (manifest_root / "manifest_sha256.json").write_text(
        json.dumps(manifest_hashes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if image_missing or depth_missing:
        raise ValueError(
            f"引用核验失败: missing images={image_missing}, "
            f"missing depth={depth_missing}"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="准备 AIC 外部 grounding 数据")
    parser.add_argument("--data-root", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="校验两个大 ZIP")
    verify.add_argument("--skip-crc", action="store_true")
    subparsers.add_parser("extract", help="安全解压两个大 ZIP")
    prepare = subparsers.add_parser("prepare", help="生成统一清单与预览")
    prepare.add_argument("--preview-count", type=int, default=3)
    subparsers.add_parser("all", help="依次执行校验、解压、清单和预览")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.data_root.resolve()
    if args.command == "verify":
        command_verify(root, check_crc=not args.skip_crc)
    elif args.command == "extract":
        command_extract(root)
    elif args.command == "prepare":
        command_prepare(root, preview_count=args.preview_count)
    elif args.command == "all":
        command_verify(root, check_crc=True)
        command_extract(root)
        command_prepare(root, preview_count=3)


if __name__ == "__main__":
    main()
