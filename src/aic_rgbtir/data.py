from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


DATASETS = ("flir", "m3fd", "mfad")
SPLITS = ("train", "val", "test")
ILLUMINATION = {1: "VWL", 2: "WL", 3: "NL", 4: "SL"}
CROWDED = {0: "NC", 1: "C", 2: "HC"}
WEATHER = {0: "FY", 1: "RY", 2: "SY", 3: "CY"}
OCCLUSION = {0: "NO", 1: "PO", 2: "HO"}
SCENE = {
    0: "UB",
    1: "SU",
    2: "RR",
    3: "HW",
    4: "RS",
    5: "ID",
    6: "PL",
    7: "IT",
    8: "TN",
    9: "BG",
    10: "CP",
    11: "MK",
    12: "WF",
}

NEGATIVE_OR_ABSENT_RE = re.compile(
    r"\b(no|not)\s+(?:\w+\s+){0,5}(present|visible|found|correspond)",
    re.IGNORECASE,
)
BBOX_META_RE = re.compile(
    r"\b(bounding box|bbox)\b|coordinates\s*\[",
    re.IGNORECASE,
)
UNCERTAIN_RE = re.compile(
    r"\b(possibly|appears to be|indistinct|faint)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RGBTRecord:
    record_id: str
    source_dataset: str
    split: str
    rgb_relpath: str
    tir_relpath: str
    query_original: str
    bbox_xywh_pixel: tuple[float, float, float, float]
    bbox_xyxy_normalized: tuple[float, float, float, float]
    width: int
    height: int
    illumination: str
    weather: str
    object_size: str
    occlusion: str
    crowded: str
    scene: str
    quality_flags: tuple[str, ...] = ()
    exclusion_reasons: tuple[str, ...] = ()

    @property
    def image_pair_key(self) -> str:
        return f"{self.source_dataset}/{Path(self.rgb_relpath).name}"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox_xywh_pixel"] = list(self.bbox_xywh_pixel)
        payload["bbox_xyxy_normalized"] = list(self.bbox_xyxy_normalized)
        payload["quality_flags"] = list(self.quality_flags)
        payload["exclusion_reasons"] = list(self.exclusion_reasons)
        payload["image_pair_key"] = self.image_pair_key
        return payload


@dataclass(frozen=True)
class ManifestBundle:
    train_all: tuple[RGBTRecord, ...]
    train_clean: tuple[RGBTRecord, ...]
    val_official: tuple[RGBTRecord, ...]
    test_official: tuple[RGBTRecord, ...]
    excluded_train: tuple[RGBTRecord, ...]
    raw_count: int
    unique_pair_count: int
    cross_split_pairs: tuple[str, ...]


def _load_pth_records(path: Path) -> list[list[Any]]:
    records = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(records, list):
        raise TypeError(f"Expected list in {path}, got {type(records)!r}")
    return records


def _resolve_pair(root: Path, dataset: str, file_name: str) -> tuple[Path, Path]:
    rgb = root / "image_data" / dataset / "rgb" / file_name
    tir = root / "image_data" / dataset / "ir" / file_name
    if rgb.is_file() and tir.is_file():
        return rgb, tir
    stem = Path(file_name).stem
    rgb_matches = sorted((root / "image_data" / dataset / "rgb").glob(f"{stem}.*"))
    tir_matches = sorted((root / "image_data" / dataset / "ir").glob(f"{stem}.*"))
    if len(rgb_matches) == 1 and len(tir_matches) == 1:
        return rgb_matches[0], tir_matches[0]
    raise FileNotFoundError(f"Cannot resolve RGB/TIR pair for {dataset}/{file_name}")


def _validate_bbox(
    bbox: Sequence[Any], *, width: int, height: int, record_id: str
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    if len(bbox) != 4:
        raise ValueError(f"{record_id}: bbox must contain four values")
    x, y, box_width, box_height = (float(value) for value in bbox)
    values = (x, y, box_width, box_height)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{record_id}: bbox contains non-finite values")
    tolerance = 1e-4
    if (
        x < -tolerance
        or y < -tolerance
        or box_width <= 0
        or box_height <= 0
        or x + box_width > width + tolerance
        or y + box_height > height + tolerance
    ):
        raise ValueError(f"{record_id}: bbox is outside {width}x{height}: {values}")
    xywh = (x, y, box_width, box_height)
    normalized = (
        x / width,
        y / height,
        (x + box_width) / width,
        (y + box_height) / height,
    )
    return xywh, normalized


def _query_flags(query: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    quality: list[str] = []
    exclusions: list[str] = []
    if NEGATIVE_OR_ABSENT_RE.search(query):
        quality.append("negative_or_absent_language")
        exclusions.append("negative_or_absent_language")
    if BBOX_META_RE.search(query):
        quality.append("bbox_meta_language")
        exclusions.append("bbox_meta_language")
    if UNCERTAIN_RE.search(query):
        quality.append("uncertain_language")
    return tuple(sorted(quality)), tuple(sorted(exclusions))


def _parse_source_record(
    *, root: Path, dataset: str, split: str, index: int, item: Sequence[Any]
) -> RGBTRecord:
    if len(item) != 10:
        raise ValueError(
            f"{dataset}:{split}:{index}: expected 10 fields, got {len(item)}"
        )
    file_name, size, bbox, query = item[0], item[1], item[2], str(item[3])
    record_id = f"{dataset}:{split}:{index}"
    if not query.strip():
        raise ValueError(f"{record_id}: empty query")
    width, height = int(size["width"]), int(size["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"{record_id}: invalid image size {width}x{height}")
    rgb, tir = _resolve_pair(root, dataset, str(file_name))
    xywh, normalized = _validate_bbox(
        bbox, width=width, height=height, record_id=record_id
    )
    quality_flags, exclusion_reasons = _query_flags(query)
    return RGBTRecord(
        record_id=record_id,
        source_dataset=dataset,
        split=split,
        rgb_relpath=rgb.relative_to(root).as_posix(),
        tir_relpath=tir.relative_to(root).as_posix(),
        query_original=query,
        bbox_xywh_pixel=xywh,
        bbox_xyxy_normalized=normalized,
        width=width,
        height=height,
        illumination=ILLUMINATION.get(int(item[4]), f"UNKNOWN_{item[4]}"),
        object_size="SS" if str(item[5]).lower() == "small" else "NS",
        crowded=CROWDED.get(int(item[6]), f"UNKNOWN_{item[6]}"),
        scene=SCENE.get(int(item[7]), f"UNKNOWN_{item[7]}"),
        weather=WEATHER.get(int(item[8]), f"UNKNOWN_{item[8]}"),
        occlusion=OCCLUSION.get(int(item[9]), f"UNKNOWN_{item[9]}"),
        quality_flags=quality_flags,
        exclusion_reasons=exclusion_reasons if split == "train" else (),
    )


def build_manifest_bundle(rgbt_root: Path | str) -> ManifestBundle:
    root = Path(rgbt_root).resolve()
    if not (root / "image_data").is_dir():
        raise FileNotFoundError(root / "image_data")

    by_split: dict[str, list[RGBTRecord]] = {split: [] for split in SPLITS}
    for dataset in DATASETS:
        annotation_dir = root / f"rgbtvg_{dataset}"
        for split in SPLITS:
            annotation = annotation_dir / f"rgbtvg_{dataset}_{split}.pth"
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
            for index, item in enumerate(_load_pth_records(annotation)):
                by_split[split].append(
                    _parse_source_record(
                        root=root,
                        dataset=dataset,
                        split=split,
                        index=index,
                        item=item,
                    )
                )

    evaluation_pairs = {
        record.image_pair_key
        for split in ("val", "test")
        for record in by_split[split]
    }
    cross_split_pairs = tuple(
        sorted(
            {
                record.image_pair_key
                for record in by_split["train"]
                if record.image_pair_key in evaluation_pairs
            }
        )
    )
    train_all: list[RGBTRecord] = []
    for record in by_split["train"]:
        reasons = list(record.exclusion_reasons)
        if record.image_pair_key in evaluation_pairs:
            reasons.append("cross_split_image_pair")
        train_all.append(replace(record, exclusion_reasons=tuple(sorted(set(reasons)))))

    train_clean = [record for record in train_all if not record.exclusion_reasons]
    excluded_train = [record for record in train_all if record.exclusion_reasons]
    all_records = train_all + by_split["val"] + by_split["test"]
    unique_pairs = {record.image_pair_key for record in all_records}
    return ManifestBundle(
        train_all=tuple(train_all),
        train_clean=tuple(train_clean),
        val_official=tuple(by_split["val"]),
        test_official=tuple(by_split["test"]),
        excluded_train=tuple(excluded_train),
        raw_count=len(all_records),
        unique_pair_count=len(unique_pairs),
        cross_split_pairs=cross_split_pairs,
    )


def _stable_random(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _condition_flags(record: RGBTRecord) -> dict[str, bool]:
    return {
        "small": record.object_size == "SS",
        "low_light": record.illumination in {"WL", "VWL"},
        "adverse_weather": record.weather in {"FY", "RY"},
        "high_occlusion": record.occlusion == "HO",
    }


def build_tracer_records(
    records: Sequence[RGBTRecord],
    *,
    total: int,
    source_quotas: dict[str, int],
    condition_minima: dict[str, int],
    seed: int = 20260812,
) -> tuple[list[RGBTRecord], dict[str, Any]]:
    if sum(source_quotas.values()) != total:
        raise ValueError("source quotas must sum to tracer total")
    one_per_pair: dict[str, RGBTRecord] = {}
    for record in sorted(records, key=lambda item: item.record_id):
        current = one_per_pair.get(record.image_pair_key)
        if current is None or _stable_random(seed, record.record_id) < _stable_random(
            seed, current.record_id
        ):
            one_per_pair[record.image_pair_key] = record
    candidates = list(one_per_pair.values())
    available = {source: 0 for source in source_quotas}
    for record in candidates:
        if record.source_dataset in available:
            available[record.source_dataset] += 1
    for source, quota in source_quotas.items():
        if available[source] < quota:
            raise ValueError(
                f"not enough unique {source} pairs: {available[source]} < {quota}"
            )

    selected: list[RGBTRecord] = []
    selected_pairs: set[str] = set()
    source_counts = {source: 0 for source in source_quotas}
    condition_counts = {condition: 0 for condition in condition_minima}
    while len(selected) < total:
        best: tuple[float, float, RGBTRecord] | None = None
        for record in candidates:
            if record.image_pair_key in selected_pairs:
                continue
            source = record.source_dataset
            if source not in source_quotas or source_counts[source] >= source_quotas[source]:
                continue
            flags = _condition_flags(record)
            condition_score = 0.0
            for condition, target in condition_minima.items():
                if flags.get(condition, False) and condition_counts[condition] < target:
                    condition_score += (target - condition_counts[condition]) / max(1, target)
            source_score = (
                source_quotas[source] - source_counts[source]
            ) / source_quotas[source]
            score = 10.0 * condition_score + source_score
            tie = -_stable_random(seed, record.record_id)
            candidate = (score, tie, record)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            raise RuntimeError("unable to fill tracer quotas")
        record = best[2]
        selected.append(record)
        selected_pairs.add(record.image_pair_key)
        source_counts[record.source_dataset] += 1
        for condition, enabled in _condition_flags(record).items():
            if condition in condition_counts and enabled:
                condition_counts[condition] += 1

    selected.sort(key=lambda record: record.record_id)
    unmet = {
        condition: {"required": target, "actual": condition_counts[condition]}
        for condition, target in condition_minima.items()
        if condition_counts[condition] < target
    }
    summary = {
        "seed": seed,
        "total": len(selected),
        "unique_pair_count": len(selected_pairs),
        "source_counts": source_counts,
        "condition_counts": condition_counts,
        "condition_minima": condition_minima,
        "unmet_minima": unmet,
    }
    return selected, summary


def write_jsonl(path: Path | str, records: Iterable[RGBTRecord | dict[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = record.as_dict() if isinstance(record, RGBTRecord) else record
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            count += 1
    return count


def read_jsonl_records(path: Path | str) -> list[RGBTRecord]:
    rows: list[RGBTRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            payload.pop("image_pair_key", None)
            for key in (
                "bbox_xywh_pixel",
                "bbox_xyxy_normalized",
                "quality_flags",
                "exclusion_reasons",
            ):
                payload[key] = tuple(payload[key])
            rows.append(RGBTRecord(**payload))
    return rows
