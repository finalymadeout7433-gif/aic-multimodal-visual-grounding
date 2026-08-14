from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .bbox import (
    intersection_over_union,
    pixel_to_normalized,
    validate_normalized_bbox,
)


DIAGNOSTIC_DATASETS = ("refcoco", "refcoco_plus", "refcocog")


@dataclass(frozen=True)
class DiagnosticCandidate:
    bbox: list[float]
    label: str
    score: float | None
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": validate_normalized_bbox(self.bbox),
            "label": self.label,
            "score": None if self.score is None else float(self.score),
            "source": self.source,
        }


@dataclass(frozen=True)
class TileWindow:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_box(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL 第 {line_number} 行无法解析: {path}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL 第 {line_number} 行不是对象: {path}")
            records.append(value)
    return records


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def _stable_key(seed: int, *parts: object) -> str:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bbox_area(record: dict[str, Any]) -> float:
    x1, y1, x2, y2 = validate_normalized_bbox(
        record["bbox_xyxy_normalized"]
    )
    return (x2 - x1) * (y2 - y1)


def build_core_subset(
    records: Sequence[dict[str, Any]],
    *,
    per_dataset: int = 500,
    seed: int = 20260730,
    datasets: Sequence[str] = DIAGNOSTIC_DATASETS,
) -> list[dict[str, Any]]:
    """构建跨数据集 image_key 唯一且可复现的核心评测子集。"""

    by_dataset_image: dict[str, dict[str, list[dict[str, Any]]]] = {
        dataset: defaultdict(list) for dataset in datasets
    }
    for record in records:
        dataset = str(record.get("dataset", ""))
        if dataset not in by_dataset_image:
            continue
        image_key = str(record["image_key"])
        by_dataset_image[dataset][image_key].append(record)

    missing = [
        dataset
        for dataset, images in by_dataset_image.items()
        if len(images) < per_dataset
    ]
    if missing:
        raise ValueError(
            "以下数据集没有足够的唯一图像: " + ", ".join(sorted(missing))
        )

    used_images: set[str] = set()
    chosen: list[dict[str, Any]] = []
    selection_order = sorted(
        datasets,
        key=lambda dataset: (len(by_dataset_image[dataset]), dataset),
    )
    for dataset in selection_order:
        available = [
            image_key
            for image_key in by_dataset_image[dataset]
            if image_key not in used_images
        ]
        available.sort(key=lambda key: _stable_key(seed, dataset, key))
        if len(available) < per_dataset:
            raise ValueError(
                f"无法为 {dataset} 选择 {per_dataset} 个跨数据集唯一图像；"
                f"仅剩 {len(available)} 个"
            )
        for image_key in available[:per_dataset]:
            expressions = sorted(
                by_dataset_image[dataset][image_key],
                key=lambda record: _stable_key(
                    seed,
                    dataset,
                    image_key,
                    record["query_id"],
                ),
            )
            selected = dict(expressions[0])
            selected["diagnostic_subset"] = "core_eval"
            chosen.append(selected)
            used_images.add(image_key)

    dataset_rank = {dataset: index for index, dataset in enumerate(datasets)}
    chosen.sort(
        key=lambda record: (
            dataset_rank[str(record["dataset"])],
            str(record["query_id"]),
        )
    )
    return chosen


def build_tile_probe(
    core_records: Sequence[dict[str, Any]],
    *,
    small_per_dataset: int = 100,
    control_per_dataset: int = 100,
    seed: int = 20260730,
    datasets: Sequence[str] = DIAGNOSTIC_DATASETS,
) -> list[dict[str, Any]]:
    """每个数据集选择最小目标和中/大目标对照。"""

    probe: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_records = [
            record
            for record in core_records
            if str(record.get("dataset")) == dataset
        ]
        required = small_per_dataset + control_per_dataset
        if len(dataset_records) < required:
            raise ValueError(
                f"{dataset} 只有 {len(dataset_records)} 条，无法构建 "
                f"{required} 条 tile probe"
            )
        ordered = sorted(
            dataset_records,
            key=lambda record: (_bbox_area(record), str(record["query_id"])),
        )
        small = ordered[:small_per_dataset]
        remaining = ordered[small_per_dataset:]
        upper_half = remaining[len(remaining) // 2 :]
        control_pool = (
            upper_half
            if len(upper_half) >= control_per_dataset
            else remaining
        )
        control = sorted(
            control_pool,
            key=lambda record: _stable_key(
                seed, dataset, "control", record["query_id"]
            ),
        )[:control_per_dataset]
        for group, selected_records in (("small", small), ("control", control)):
            for record in selected_records:
                copied = dict(record)
                copied["diagnostic_subset"] = "tile_probe"
                copied["probe_group"] = group
                probe.append(copied)
    return probe


def normalize_candidate_label(label: str) -> str:
    lowered = re.sub(r"\s+", " ", label.strip().lower())
    return lowered.rstrip(" .,:;!?")


def candidate_semantic_type(
    candidates: Sequence[DiagnosticCandidate],
) -> str:
    if not candidates:
        return "no_candidate"
    if len(candidates) == 1:
        return "single"
    labels = {normalize_candidate_label(candidate.label) for candidate in candidates}
    return "multi_same_label" if len(labels) == 1 else "multi_distinct_label"


def query_category(query: str) -> str:
    text = f" {query.lower()} "
    patterns = (
        (
            "depth",
            r"\b(nearest|closest|farthest|furthest|front|behind|distance)\b",
        ),
        (
            "ordinal",
            r"\b(first|second|third|fourth|leftmost|rightmost|topmost|"
            r"bottommost)\b",
        ),
        (
            "plural_group",
            r"\b(two|three|four|five|both|group|pair|several|multiple)\b",
        ),
        (
            "spatial",
            r"\b(left|right|above|below|under|over|beside|between|next to|"
            r"near|inside|outside)\b",
        ),
        (
            "action",
            r"\b(standing|sitting|walking|running|riding|holding|flying|"
            r"looking|wearing)\b",
        ),
        (
            "attribute",
            r"\b(red|green|blue|yellow|black|white|silver|golden|wooden|"
            r"metal|striped|checkered|large|small)\b",
        ),
    )
    for category, pattern in patterns:
        if re.search(pattern, text):
            return category
    return "other"


def bbox_area_bin(box: Sequence[float]) -> str:
    x1, y1, x2, y2 = validate_normalized_bbox(box)
    area = (x2 - x1) * (y2 - y1)
    if area < 0.01:
        return "small_lt_1pct"
    if area < 0.10:
        return "medium_1_to_10pct"
    return "large_ge_10pct"


def evaluate_candidates(
    ground_truth: Sequence[float],
    candidates: Sequence[DiagnosticCandidate],
    *,
    selected_index: int | None = None,
) -> dict[str, Any]:
    gt = validate_normalized_bbox(ground_truth)
    if not candidates:
        return {
            "selected_bbox": None,
            "selected_index": None,
            "selected_iou": 0.0,
            "best_candidate_index": None,
            "best_candidate_iou": 0.0,
            "top5_best_iou": 0.0,
            "top10_best_iou": 0.0,
            "selected_acc_at_05": False,
            "oracle_acc_at_05": False,
            "top5_oracle_acc_at_05": False,
            "top10_oracle_acc_at_05": False,
        }
    index = 0 if selected_index is None else selected_index
    if index < 0 or index >= len(candidates):
        raise IndexError("selected_index 超出候选范围")
    ious = [
        intersection_over_union(gt, validate_normalized_bbox(candidate.bbox))
        for candidate in candidates
    ]
    best_index = max(range(len(ious)), key=ious.__getitem__)
    top5_best = max(ious[:5], default=0.0)
    top10_best = max(ious[:10], default=0.0)
    return {
        "selected_bbox": validate_normalized_bbox(candidates[index].bbox),
        "selected_index": index,
        "selected_iou": ious[index],
        "best_candidate_index": best_index,
        "best_candidate_iou": ious[best_index],
        "top5_best_iou": top5_best,
        "top10_best_iou": top10_best,
        "selected_acc_at_05": ious[index] >= 0.5,
        "oracle_acc_at_05": ious[best_index] >= 0.5,
        "top5_oracle_acc_at_05": top5_best >= 0.5,
        "top10_oracle_acc_at_05": top10_best >= 0.5,
    }


def tile_windows(
    *,
    width: int,
    height: int,
    overlap_ratio: float = 0.2,
) -> list[TileWindow]:
    if width <= 0 or height <= 0:
        raise ValueError("图像宽高必须为正数")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio 必须位于 [0, 1)")
    tile_width = min(width, round(width / (2.0 - overlap_ratio)))
    tile_height = min(height, round(height / (2.0 - overlap_ratio)))
    x_positions = (0, width - tile_width)
    y_positions = (0, height - tile_height)
    return [
        TileWindow(x, y, x + tile_width, y + tile_height)
        for y in y_positions
        for x in x_positions
    ]


def map_tile_bbox_to_normalized(
    pixel_bbox: Sequence[float],
    *,
    tile: TileWindow,
    image_width: int,
    image_height: int,
) -> list[float]:
    if len(pixel_bbox) != 4:
        raise ValueError("tile bbox 必须包含 4 个坐标")
    x1, y1, x2, y2 = (float(value) for value in pixel_bbox)
    global_box = [
        max(0.0, min(float(image_width), x1 + tile.x1)),
        max(0.0, min(float(image_height), y1 + tile.y1)),
        max(0.0, min(float(image_width), x2 + tile.x1)),
        max(0.0, min(float(image_height), y2 + tile.y1)),
    ]
    return pixel_to_normalized(
        global_box,
        width=image_width,
        height=image_height,
    )


def deduplicate_candidates(
    candidates: Sequence[DiagnosticCandidate],
    *,
    iou_threshold: float = 0.85,
) -> list[DiagnosticCandidate]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold 必须位于 [0, 1]")
    retained: list[DiagnosticCandidate] = []
    for candidate in candidates:
        normalized_label = normalize_candidate_label(candidate.label)
        duplicate = any(
            normalize_candidate_label(existing.label) == normalized_label
            and intersection_over_union(existing.bbox, candidate.bbox)
            >= iou_threshold
            for existing in retained
        )
        if not duplicate:
            retained.append(candidate)
    return retained


def _metric_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if not count:
        return {
            "count": 0,
            "selected_acc_at_05": 0.0,
            "oracle_acc_at_05": 0.0,
            "top5_oracle_acc_at_05": 0.0,
            "top10_oracle_acc_at_05": 0.0,
            "selected_iou_mean": 0.0,
            "best_candidate_iou_mean": 0.0,
            "oracle_gap": 0.0,
            "no_candidate_rate": 0.0,
            "candidate_count_mean": 0.0,
            "latency_ms_mean": 0.0,
        }
    selected_acc = statistics.fmean(
        float(record["selected_acc_at_05"]) for record in records
    )
    oracle_acc = statistics.fmean(
        float(record["oracle_acc_at_05"]) for record in records
    )
    return {
        "count": count,
        "selected_acc_at_05": selected_acc,
        "oracle_acc_at_05": oracle_acc,
        "top5_oracle_acc_at_05": statistics.fmean(
            float(record["top5_oracle_acc_at_05"]) for record in records
        ),
        "top10_oracle_acc_at_05": statistics.fmean(
            float(record["top10_oracle_acc_at_05"]) for record in records
        ),
        "selected_iou_mean": statistics.fmean(
            float(record["selected_iou"]) for record in records
        ),
        "best_candidate_iou_mean": statistics.fmean(
            float(record["best_candidate_iou"]) for record in records
        ),
        "oracle_gap": oracle_acc - selected_acc,
        "no_candidate_rate": statistics.fmean(
            float(not record.get("candidates")) for record in records
        ),
        "candidate_count_mean": statistics.fmean(
            len(record.get("candidates", [])) for record in records
        ),
        "latency_ms_mean": statistics.fmean(
            float(record.get("latency_ms", 0.0)) for record in records
        ),
    }


def summarize_diagnostic_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    group_fields = (
        "dataset",
        "area_bin",
        "query_category",
        "candidate_semantic_type",
        "probe_group",
    )
    summary: dict[str, Any] = {"overall": _metric_summary(records)}
    for field in group_fields:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            value = record.get(field)
            if value is not None:
                groups[str(value)].append(record)
        if groups:
            summary[f"by_{field}"] = {
                key: _metric_summary(group)
                for key, group in sorted(groups.items())
            }
    return summary


def ensure_finite_score(score: float | None) -> float | None:
    if score is None:
        return None
    value = float(score)
    if not math.isfinite(value):
        raise ValueError("候选 score 必须是有限数值")
    return value
