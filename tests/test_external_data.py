from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

import pytest

from aic_baseline.external_data import (
    DatasetSpec,
    GroundingManifestDataset,
    audit_extracted_zip,
    build_dataset_records,
    build_image_isolated_splits,
    extract_verified_zip,
    verify_zip_archive,
)


def test_verify_and_extract_zip_rejects_unsafe_members(tmp_path: Path) -> None:
    safe_zip = tmp_path / "safe.zip"
    with zipfile.ZipFile(safe_zip, "w") as archive:
        archive.writestr("images/example.jpg", b"image")

    verification = verify_zip_archive(safe_zip, check_crc=True)
    destination = tmp_path / "safe"
    extracted = extract_verified_zip(safe_zip, destination)

    assert verification.entry_count == 1
    assert verification.bad_member is None
    assert extracted == 1
    assert (destination / "images" / "example.jpg").read_bytes() == b"image"

    unsafe_zip = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe_zip, "w") as archive:
        archive.writestr("../outside.txt", b"escape")

    with pytest.raises(ValueError, match="不安全"):
        extract_verified_zip(unsafe_zip, tmp_path / "unsafe")


def test_extract_zip_sanitizes_windows_invalid_member_names(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "windows-name.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("status/file_2015-01-05 12:56:14.json", b"metadata")

    destination = tmp_path / "output"
    extract_verified_zip(archive_path, destination)

    assert (
        destination / "status" / "file_2015-01-05 12_56_14.json"
    ).read_bytes() == b"metadata"
    audit = audit_extracted_zip(archive_path, destination)
    assert audit.missing_count == 0
    assert audit.size_mismatch_count == 0
    assert audit.sanitized_member_count == 1
    assert audit.sanitized_collision_count == 0


def test_build_refcoco_records_normalizes_bbox_and_preserves_split(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "COCO_train2014_000000000001.jpg").write_bytes(b"image")
    instances = {
        "images": [
            {
                "id": 1,
                "file_name": "COCO_train2014_000000000001.jpg",
                "width": 100,
                "height": 50,
            }
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "bbox": [10, 5, 40, 20]}
        ],
    }
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(json.dumps(instances), encoding="utf-8")
    refs = [
        {
            "ref_id": 7,
            "image_id": 1,
            "ann_id": 10,
            "split": "train",
            "sentences": [{"sent_id": 9, "sent": "the object"}],
        }
    ]
    refs_path = tmp_path / "refs.p"
    with refs_path.open("wb") as handle:
        pickle.dump(refs, handle)

    records, summary = build_dataset_records(
        DatasetSpec(
            name="refcoco",
            refs_path=refs_path,
            instances_path=instances_path,
            image_root=image_root,
            image_prefix="images",
        )
    )

    assert summary["expression_count"] == 1
    assert summary["missing_image_count"] == 0
    assert records[0]["query_id"] == "refcoco:9"
    assert records[0]["bbox_xyxy_normalized"] == [0.1, 0.1, 0.5, 0.5]
    assert records[0]["source_split"] == "train"


def test_global_split_assignment_keeps_each_image_in_one_split() -> None:
    records = [
        {
            "dataset": "refcoco",
            "image_key": "coco:1",
            "source_split": "train",
        },
        {
            "dataset": "refcocog",
            "image_key": "coco:1",
            "source_split": "val",
        },
        {
            "dataset": "refcoco",
            "image_key": "coco:2",
            "source_split": "testA",
        },
    ]

    assigned, summary = build_image_isolated_splits(records)

    assert [record["split"] for record in assigned] == [
        "validation",
        "validation",
        "holdout",
    ]
    assert summary["image_overlap_count"] == 0


def test_manifest_dataset_returns_training_ready_image_query_and_bbox(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "images" / "sample.jpg"
    image_path.parent.mkdir()
    from PIL import Image

    Image.new("RGB", (20, 10), color="white").save(image_path)
    record = {
        "query_id": "sample:1",
        "query": "the white image",
        "image_relpath": "images/sample.jpg",
        "bbox_xyxy_normalized": [0.1, 0.2, 0.8, 0.9],
    }
    manifest_path = tmp_path / "train.jsonl"
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = GroundingManifestDataset(
        data_root=tmp_path,
        manifest_path=manifest_path,
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["image"].size == (20, 10)
    assert sample["query"] == "the white image"
    assert sample["bbox_xyxy_normalized"] == [0.1, 0.2, 0.8, 0.9]
