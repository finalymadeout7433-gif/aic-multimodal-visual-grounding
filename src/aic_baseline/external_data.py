from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class ArchiveVerification:
    path: str
    bytes: int
    sha256: str
    entry_count: int
    bad_member: str | None


@dataclass(frozen=True)
class ExtractionAudit:
    archive_file_count: int
    missing_count: int
    size_mismatch_count: int
    sanitized_member_count: int
    sanitized_collision_count: int


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    refs_path: Path
    instances_path: Path
    image_root: Path
    image_prefix: str
    refs_sha256: str | None = None
    image_namespace: str | None = None
    depth_root: Path | None = None
    depth_prefix: str | None = None


class GroundingManifestDataset:
    """按 JSONL 字节偏移懒加载图像、Query 和归一化 bbox。"""

    def __init__(
        self,
        *,
        data_root: Path | str,
        manifest_path: Path | str,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self._offsets: list[int] = []
        with self.manifest_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.manifest_path.open("rb") as handle:
            handle.seek(self._offsets[index])
            record = json.loads(handle.readline().decode("utf-8"))
        image_path = self.data_root / record["image_relpath"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        return {
            **record,
            "image": image,
            "query": record["query"],
            "bbox_xyxy_normalized": record["bbox_xyxy_normalized"],
        }


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_zip_archive(
    path: Path | str, *, check_crc: bool = True
) -> ArchiveVerification:
    archive_path = Path(path)
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip() if check_crc else None
        entry_count = len(archive.infolist())
    return ArchiveVerification(
        path=str(archive_path.resolve()),
        bytes=archive_path.stat().st_size,
        sha256=sha256_file(archive_path),
        entry_count=entry_count,
        bad_member=bad_member,
    )


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _windows_safe_part(part: str) -> str:
    sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', "_", part).rstrip(" .")
    if not sanitized:
        sanitized = "_"
    if sanitized.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        sanitized = f"_{sanitized}"
    return sanitized


def _safe_destination(root: Path, member_name: str) -> Path:
    normalized_name = member_name.replace("\\", "/")
    member_path = PurePosixPath(normalized_name)
    if (
        member_path.is_absolute()
        or ".." in member_path.parts
        or (
            member_path.parts
            and re.fullmatch(r"[A-Za-z]:", member_path.parts[0])
        )
    ):
        raise ValueError(f"ZIP 包含不安全路径: {member_name}")
    safe_parts = [_windows_safe_part(part) for part in member_path.parts if part]
    candidate = root.joinpath(*safe_parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"ZIP 包含不安全路径: {member_name}") from exc
    return candidate


def extract_verified_zip(
    path: Path | str,
    destination: Path | str,
    *,
    overwrite: bool = False,
) -> int:
    archive_path = Path(path)
    output_root = Path(destination)
    output_root.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = _safe_destination(output_root, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not overwrite:
                if target.stat().st_size == member.file_size:
                    continue
                raise FileExistsError(f"已存在但大小不一致: {target}")
            with archive.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
            extracted += 1
    return extracted


def audit_extracted_zip(
    path: Path | str,
    destination: Path | str,
) -> ExtractionAudit:
    archive_path = Path(path)
    output_root = Path(destination)
    missing_count = 0
    size_mismatch_count = 0
    sanitized_member_count = 0
    target_sources: dict[Path, list[str]] = {}
    archive_file_count = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            archive_file_count += 1
            target = _safe_destination(output_root, member.filename)
            target_sources.setdefault(target, []).append(member.filename)
            normalized_source = member.filename.replace("\\", "/").strip("/")
            normalized_target = target.relative_to(output_root.resolve()).as_posix()
            if normalized_source != normalized_target:
                sanitized_member_count += 1
            if not target.is_file():
                missing_count += 1
            elif target.stat().st_size != member.file_size:
                size_mismatch_count += 1
    collision_count = sum(
        1 for sources in target_sources.values() if len(sources) > 1
    )
    return ExtractionAudit(
        archive_file_count=archive_file_count,
        missing_count=missing_count,
        size_mismatch_count=size_mismatch_count,
        sanitized_member_count=sanitized_member_count,
        sanitized_collision_count=collision_count,
    )


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            f"forbidden pickle global: {module}.{name}"
        )


def load_annotation_refs(
    path: Path,
    expected_sha256: str | None = None,
) -> list[dict[str, Any]]:
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(path)
        if actual_sha256.upper() != expected_sha256.upper():
            raise ValueError(
                f"annotation SHA-256 mismatch: {path}; "
                f"expected={expected_sha256.upper()}, actual={actual_sha256}"
            )
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    else:
        if expected_sha256 is None:
            raise ValueError(f"pickle annotation requires SHA-256: {path}")
        with path.open("rb") as handle:
            value = _RestrictedUnpickler(handle, encoding="latin1").load()
    if not isinstance(value, list):
        raise ValueError(f"refs 顶层不是列表: {path}")
    return value


def _normalize_bbox(
    bbox_xywh: Iterable[float], *, width: int, height: int
) -> tuple[list[float], list[float]]:
    values = [float(value) for value in bbox_xywh]
    if len(values) != 4:
        raise ValueError("bbox 必须包含四个数字")
    x, y, box_width, box_height = values
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox 含 NaN 或 inf")
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        raise ValueError("bbox 非法")
    x2 = x + box_width
    y2 = y + box_height
    if x2 > width + 1e-6 or y2 > height + 1e-6:
        raise ValueError("bbox 超出图像范围")
    normalized = [x / width, y / height, x2 / width, y2 / height]
    return values, normalized


def _union_bboxes(
    annotations: list[dict[str, Any]],
) -> list[float]:
    boxes = [annotation["bbox"] for annotation in annotations]
    x1 = min(float(box[0]) for box in boxes)
    y1 = min(float(box[1]) for box in boxes)
    x2 = max(float(box[0]) + float(box[2]) for box in boxes)
    y2 = max(float(box[1]) + float(box[3]) for box in boxes)
    return [x1, y1, x2 - x1, y2 - y1]


def build_dataset_records(
    spec: DatasetSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs = load_annotation_refs(spec.refs_path, spec.refs_sha256)
    with spec.instances_path.open("r", encoding="utf-8") as handle:
        instances = json.load(handle)
    images = {image["id"]: image for image in instances["images"]}
    annotations = {
        annotation["id"]: annotation for annotation in instances["annotations"]
    }
    records: list[dict[str, Any]] = []
    source_splits: Counter[str] = Counter()
    missing_images = 0
    missing_annotations = 0
    invalid_bbox_count = 0
    excluded_no_target = 0

    for ref in refs:
        if ref.get("no_target", False):
            excluded_no_target += len(ref.get("sentences", []))
            continue
        image = images.get(ref["image_id"])
        if image is None:
            missing_images += len(ref.get("sentences", []))
            continue
        ann_ids = ref["ann_id"]
        ann_ids = ann_ids if isinstance(ann_ids, list) else [ann_ids]
        target_annotations = [
            annotations[ann_id] for ann_id in ann_ids if ann_id in annotations
        ]
        if len(target_annotations) != len(ann_ids) or not target_annotations:
            missing_annotations += len(ref.get("sentences", []))
            continue
        bbox_xywh = _union_bboxes(target_annotations)
        try:
            pixel_bbox, normalized_bbox = _normalize_bbox(
                bbox_xywh,
                width=int(image["width"]),
                height=int(image["height"]),
            )
        except ValueError:
            invalid_bbox_count += len(ref.get("sentences", []))
            continue

        file_name = str(image["file_name"]).replace("\\", "/")
        image_path = spec.image_root / Path(file_name)
        image_exists = image_path.is_file()
        if not image_exists:
            missing_images += len(ref.get("sentences", []))
        depth_file_name = image.get("depth_file_name")
        depth_relpath = None
        depth_exists = None
        if depth_file_name and spec.depth_root is not None:
            normalized_depth_name = str(depth_file_name).replace("\\", "/")
            depth_path = spec.depth_root / Path(normalized_depth_name)
            depth_relpath = (
                f"{spec.depth_prefix.rstrip('/')}/{normalized_depth_name}"
                if spec.depth_prefix
                else normalized_depth_name
            )
            depth_exists = depth_path.is_file()

        source_split = str(ref.get("split", "unknown"))
        source_splits[source_split] += len(ref.get("sentences", []))
        namespace = spec.image_namespace or spec.name
        for sentence in ref.get("sentences", []):
            query = str(sentence.get("sent") or sentence.get("raw") or "").strip()
            if not query:
                continue
            sent_id = sentence.get("sent_id")
            records.append(
                {
                    "schema_version": 1,
                    "dataset": spec.name,
                    "query_id": f"{spec.name}:{sent_id}",
                    "image_key": f"{namespace}:{ref['image_id']}",
                    "image_id": ref["image_id"],
                    "ref_id": ref.get("ref_id"),
                    "ann_ids": ann_ids,
                    "target_count": len(ann_ids),
                    "query": query,
                    "source_split": source_split,
                    "image_relpath": (
                        f"{spec.image_prefix.rstrip('/')}/{file_name}"
                    ),
                    "image_exists": image_exists,
                    "depth_relpath": depth_relpath,
                    "depth_exists": depth_exists,
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "bbox_xywh_pixels": pixel_bbox,
                    "bbox_xyxy_normalized": normalized_bbox,
                }
            )

    summary = {
        "dataset": spec.name,
        "reference_count": len(refs),
        "expression_count": len(records),
        "source_split_expression_counts": dict(source_splits),
        "missing_image_count": missing_images,
        "missing_annotation_count": missing_annotations,
        "invalid_bbox_count": invalid_bbox_count,
        "excluded_no_target_expression_count": excluded_no_target,
    }
    return records, summary


def _canonical_split(source_split: str) -> str:
    normalized = source_split.lower()
    if normalized.startswith("test") or normalized in {"holdout"}:
        return "holdout"
    if normalized in {"val", "validation"}:
        return "validation"
    if normalized == "train":
        return "train"
    return "holdout"


def build_image_isolated_splits(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    priority = {"train": 0, "validation": 1, "holdout": 2}
    image_splits: dict[str, str] = {}
    for record in records:
        image_key = str(record["image_key"])
        candidate = _canonical_split(str(record["source_split"]))
        current = image_splits.get(image_key)
        if current is None or priority[candidate] > priority[current]:
            image_splits[image_key] = candidate
    assigned = [
        {**record, "split": image_splits[str(record["image_key"])]}
        for record in records
    ]
    split_images: dict[str, set[str]] = {
        split: {
            str(record["image_key"])
            for record in assigned
            if record["split"] == split
        }
        for split in priority
    }
    overlap = (
        (split_images["train"] & split_images["validation"])
        | (split_images["train"] & split_images["holdout"])
        | (split_images["validation"] & split_images["holdout"])
    )
    summary = {
        "record_counts": dict(Counter(record["split"] for record in assigned)),
        "image_counts": {
            split: len(images) for split, images in split_images.items()
        },
        "image_overlap_count": len(overlap),
    }
    return assigned, summary


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_preview(
    *,
    data_root: Path,
    record: dict[str, Any],
    output_path: Path,
) -> None:
    image_path = data_root / record["image_relpath"]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = record["bbox_xyxy_normalized"]
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(width, height) / 250))
    draw.rectangle(
        (x1 * width, y1 * height, x2 * width, y2 * height),
        outline=(255, 0, 0),
        width=line_width,
    )
    label = f"{record['query_id']} | {record['query']}"
    label = label.encode("ascii", errors="replace").decode("ascii")[:120]
    text_bbox = draw.textbbox((0, 0), label)
    text_height = text_bbox[3] - text_bbox[1] + 8
    draw.rectangle((0, 0, width, text_height), fill=(0, 0, 0))
    draw.text((4, 4), label, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)
