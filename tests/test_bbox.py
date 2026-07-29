from __future__ import annotations

import pytest

from aic_baseline.bbox import (
    BBoxError,
    acc_at_05,
    intersection_over_union,
    normalized_to_pixel,
    pixel_to_normalized,
    validate_normalized_bbox,
)


def test_official_bbox_converts_to_documented_pixel_coordinates() -> None:
    normalized = [0.7718, 0.9249, 0.8129, 0.9762]

    pixels = normalized_to_pixel(normalized, width=1920, height=1080)

    assert pixels == pytest.approx([1481.856, 998.892, 1560.768, 1054.296])
    assert pixel_to_normalized(pixels, width=1920, height=1080) == pytest.approx(
        normalized
    )


def test_iou_and_acc_use_independent_known_examples() -> None:
    gt = [0.0, 0.0, 1.0, 1.0]
    half_overlap = [0.0, 0.0, 0.5, 1.0]
    disjoint = [2.0, 2.0, 3.0, 3.0]

    assert intersection_over_union(gt, half_overlap) == pytest.approx(0.5)
    assert acc_at_05(half_overlap, gt) is True
    assert intersection_over_union(gt, disjoint) == 0.0
    assert acc_at_05(disjoint, gt) is False


@pytest.mark.parametrize(
    "box",
    [
        [0.5, 0.2, 0.5, 0.8],
        [0.8, 0.2, 0.5, 0.8],
        [-0.1, 0.2, 0.5, 0.8],
        [0.1, 0.2, float("nan"), 0.8],
    ],
)
def test_invalid_normalized_boxes_are_rejected(box: list[float]) -> None:
    with pytest.raises(BBoxError):
        validate_normalized_bbox(box)
