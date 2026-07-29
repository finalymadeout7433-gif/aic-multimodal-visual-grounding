from __future__ import annotations

import math
from collections.abc import Sequence


class BBoxError(ValueError):
    """边界框格式或取值不合法。"""


def _as_four_floats(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise BBoxError(f"bbox 必须包含 4 个数值，实际为 {len(box)} 个")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise BBoxError("bbox 包含 NaN 或无穷值")
    return values  # type: ignore[return-value]


def validate_normalized_bbox(box: Sequence[float]) -> list[float]:
    """校验 [x1, y1, x2, y2] 归一化框并返回 float 列表。"""

    x1, y1, x2, y2 = _as_four_floats(box)
    if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        raise BBoxError("归一化 bbox 坐标必须位于 [0, 1]")
    if x1 >= x2 or y1 >= y2:
        raise BBoxError("bbox 必须满足 x1 < x2 且 y1 < y2")
    return [x1, y1, x2, y2]


def normalized_to_pixel(
    box: Sequence[float], *, width: int, height: int
) -> list[float]:
    if width <= 0 or height <= 0:
        raise BBoxError("图像宽高必须为正数")
    x1, y1, x2, y2 = validate_normalized_bbox(box)
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def pixel_to_normalized(
    box: Sequence[float], *, width: int, height: int
) -> list[float]:
    if width <= 0 or height <= 0:
        raise BBoxError("图像宽高必须为正数")
    x1, y1, x2, y2 = _as_four_floats(box)
    normalized = [x1 / width, y1 / height, x2 / width, y2 / height]
    return validate_normalized_bbox(normalized)


def intersection_over_union(
    first: Sequence[float], second: Sequence[float]
) -> float:
    ax1, ay1, ax2, ay2 = _as_four_floats(first)
    bx1, by1, bx2, by2 = _as_four_floats(second)
    if ax1 >= ax2 or ay1 >= ay2 or bx1 >= bx2 or by1 >= by2:
        raise BBoxError("计算 IoU 的两个 bbox 都必须具有正面积")

    inter_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_width * inter_height
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union


def acc_at_05(prediction: Sequence[float], ground_truth: Sequence[float]) -> bool:
    return intersection_over_union(prediction, ground_truth) >= 0.5
