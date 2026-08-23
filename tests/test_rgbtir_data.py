from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from PIL import Image

from aic_rgbtir.data import RGBTRecord, build_manifest_bundle, build_tracer_records, write_jsonl


def _annotation_item(file_name: str, query: str, *, small: bool = False) -> list[object]:
    return [
        file_name,
        {"width": 64, "height": 48},
        [8, 6, 20, 16],
        query,
        2,
        "small" if small else "normal",
        0,
        0,
        0,
        2,
    ]


def _write_pair(root: Path, dataset: str, name: str) -> None:
    for modality in ("rgb", "ir"):
        path = root / "image_data" / dataset / modality / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), color=(80, 90, 100)).save(path)


def _write_annotation(root: Path, dataset: str, split: str, rows: list[list[object]]) -> None:
    path = root / f"rgbtvg_{dataset}" / f"rgbtvg_{dataset}_{split}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, path)


def test_manifest_cleaning_and_cross_split_isolation(tmp_path: Path) -> None:
    for dataset in ("flir", "m3fd", "mfad"):
        common = f"{dataset}_shared.png"
        negative = f"{dataset}_negative.png"
        meta = f"{dataset}_meta.png"
        test_name = f"{dataset}_test.png"
        for name in (common, negative, meta, test_name):
            _write_pair(tmp_path, dataset, name)
        _write_annotation(
            tmp_path,
            dataset,
            "train",
            [
                _annotation_item(common, "the small pedestrian", small=True),
                _annotation_item(negative, "no pedestrian is visible"),
                _annotation_item(meta, "the bbox around the vehicle"),
            ],
        )
        _write_annotation(
            tmp_path,
            dataset,
            "val",
            [_annotation_item(common, "the pedestrian in the scene")],
        )
        _write_annotation(
            tmp_path,
            dataset,
            "test",
            [_annotation_item(test_name, "the visible vehicle")],
        )

    bundle = build_manifest_bundle(tmp_path)
    assert bundle.raw_count == 15
    assert len(bundle.train_all) == 9
    assert len(bundle.train_clean) == 0
    assert len(bundle.excluded_train) == 9
    assert len(bundle.val_official) == 3
    assert len(bundle.test_official) == 3
    assert len(bundle.cross_split_pairs) == 3
    reasons = [reason for row in bundle.excluded_train for reason in row.exclusion_reasons]
    assert reasons.count("cross_split_image_pair") == 3
    assert reasons.count("negative_or_absent_language") == 3
    assert reasons.count("bbox_meta_language") == 3
    assert all(not row.exclusion_reasons for row in bundle.val_official)


def _record(index: int, source: str) -> RGBTRecord:
    flags = index % 4
    return RGBTRecord(
        record_id=f"{source}:train:{index}",
        source_dataset=source,
        split="train",
        rgb_relpath=f"image_data/{source}/rgb/{index}.png",
        tir_relpath=f"image_data/{source}/ir/{index}.png",
        query_original=f"target {index}",
        bbox_xywh_pixel=(1.0, 1.0, 5.0, 5.0),
        bbox_xyxy_normalized=(0.01, 0.01, 0.06, 0.06),
        width=100,
        height=100,
        illumination="VWL" if flags in {0, 1} else "NL",
        weather="RY" if flags in {0, 2} else "CY",
        object_size="SS" if flags in {0, 3} else "NS",
        occlusion="HO" if flags in {0, 1} else "NO",
        crowded="NC",
        scene="BG",
    )


def test_tracer_is_deterministic_and_one_query_per_pair(tmp_path: Path) -> None:
    records = [
        _record(index, source)
        for source in ("flir", "m3fd", "mfad")
        for index in range(12)
    ]
    first, first_summary = build_tracer_records(
        records,
        total=12,
        source_quotas={"flir": 4, "m3fd": 4, "mfad": 4},
        condition_minima={
            "small": 3,
            "low_light": 3,
            "adverse_weather": 3,
            "high_occlusion": 3,
        },
        seed=20260812,
    )
    second, second_summary = build_tracer_records(
        records,
        total=12,
        source_quotas={"flir": 4, "m3fd": 4, "mfad": 4},
        condition_minima={
            "small": 3,
            "low_light": 3,
            "adverse_weather": 3,
            "high_occlusion": 3,
        },
        seed=20260812,
    )
    assert [row.record_id for row in first] == [row.record_id for row in second]
    assert first_summary == second_summary
    assert first_summary["unmet_minima"] == {}
    assert len({row.image_pair_key for row in first}) == 12

    path1 = tmp_path / "first.jsonl"
    path2 = tmp_path / "second.jsonl"
    write_jsonl(path1, first)
    write_jsonl(path2, second)
    assert hashlib.sha256(path1.read_bytes()).digest() == hashlib.sha256(path2.read_bytes()).digest()
