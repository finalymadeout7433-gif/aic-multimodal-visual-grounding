from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw

from .bbox import normalized_to_pixel


def draw_comparison(
    *,
    image: Image.Image,
    ground_truth: Sequence[float],
    prediction: Sequence[float],
    output_path: Path | str,
) -> None:
    """在 RGB 原图上绘制绿色 GT 与红色预测框。"""

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    gt_pixel = normalized_to_pixel(
        ground_truth, width=image.width, height=image.height
    )
    pred_pixel = normalized_to_pixel(
        prediction, width=image.width, height=image.height
    )
    draw.rectangle(gt_pixel, outline=(0, 255, 80), width=7)
    draw.text(
        (gt_pixel[0], max(0, gt_pixel[1] - 28)),
        "OFFICIAL GT",
        fill=(0, 255, 80),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    draw.rectangle(pred_pixel, outline=(255, 55, 55), width=7)
    draw.text(
        (pred_pixel[0], max(0, pred_pixel[1] - 28)),
        "FLORENCE PRED",
        fill=(255, 55, 55),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)

