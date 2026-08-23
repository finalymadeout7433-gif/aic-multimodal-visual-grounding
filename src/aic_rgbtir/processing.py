from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

from .data import RGBTRecord


@dataclass(frozen=True)
class SharedGeometryTransform:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int

    def original_to_resized(
        self, bbox_xyxy: Sequence[float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
        sx = self.resized_width / self.original_width
        sy = self.resized_height / self.original_height
        return x1 * sx, y1 * sy, x2 * sx, y2 * sy

    def resized_to_original(
        self, bbox_xyxy: Sequence[float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
        sx = self.original_width / self.resized_width
        sy = self.original_height / self.resized_height
        return x1 * sx, y1 * sy, x2 * sx, y2 * sy

    def normalized_to_resized(
        self, bbox_xyxy: Sequence[float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
        return (
            x1 * self.resized_width,
            y1 * self.resized_height,
            x2 * self.resized_width,
            y2 * self.resized_height,
        )

    def resized_to_normalized(
        self, bbox_xyxy: Sequence[float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
        return (
            x1 / self.resized_width,
            y1 / self.resized_height,
            x2 / self.resized_width,
            y2 / self.resized_height,
        )


@dataclass(frozen=True)
class PairInspection:
    rgb_width: int
    rgb_height: int
    tir_width: int
    tir_height: int
    same_spatial_dimensions: bool
    ir_valid_ratio: float
    ir_intensity_p1: float
    ir_intensity_p99: float
    ir_intensity_std: float
    ir_low_information: bool
    ir_usable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "rgb_width": self.rgb_width,
            "rgb_height": self.rgb_height,
            "tir_width": self.tir_width,
            "tir_height": self.tir_height,
            "same_spatial_dimensions": self.same_spatial_dimensions,
            "ir_valid_ratio": self.ir_valid_ratio,
            "ir_intensity_p1": self.ir_intensity_p1,
            "ir_intensity_p99": self.ir_intensity_p99,
            "ir_intensity_std": self.ir_intensity_std,
            "ir_low_information": self.ir_low_information,
            "ir_usable": self.ir_usable,
        }


@dataclass(frozen=True)
class PairedRGBTBatch:
    rgb_pixel_values: torch.Tensor
    tir_pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    ir_pixel_valid_mask: torch.Tensor
    ir_patch_valid_mask: torch.Tensor
    ir_merged_valid_mask: torch.Tensor
    shared_geometry_transform: SharedGeometryTransform
    original_rgb_size: tuple[int, int]
    normalized_bbox: tuple[float, float, float, float] | None
    resized_bbox: tuple[float, float, float, float] | None
    ir_usable: bool
    ir_valid_ratio: float


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _border_connected_mask_fallback(candidate: np.ndarray) -> np.ndarray:
    height, width = candidate.shape
    connected = np.zeros_like(candidate, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        if candidate[0, x]:
            queue.append((0, x))
        if candidate[height - 1, x]:
            queue.append((height - 1, x))
    for y in range(height):
        if candidate[y, 0]:
            queue.append((y, 0))
        if candidate[y, width - 1]:
            queue.append((y, width - 1))
    while queue:
        y, x = queue.popleft()
        if connected[y, x] or not candidate[y, x]:
            continue
        connected[y, x] = True
        if y:
            queue.append((y - 1, x))
        if y + 1 < height:
            queue.append((y + 1, x))
        if x:
            queue.append((y, x - 1))
        if x + 1 < width:
            queue.append((y, x + 1))
    return connected


def border_valid_mask(image: Image.Image, *, black_threshold: int = 3) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    candidate = gray <= int(black_threshold)
    if not candidate.any():
        return np.ones_like(candidate, dtype=bool)
    try:
        import cv2

        _, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=4)
        border_labels = np.unique(
            np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
        )
        border_labels = border_labels[border_labels != 0]
        invalid = np.isin(labels, border_labels) if border_labels.size else np.zeros_like(candidate)
    except Exception:
        invalid = _border_connected_mask_fallback(candidate)
    return ~invalid


def inspect_pair_paths(
    rgb_path: Path | str,
    tir_path: Path | str,
    *,
    black_threshold: int = 3,
) -> PairInspection:
    rgb = _load_rgb(Path(rgb_path))
    tir = _load_rgb(Path(tir_path))
    valid = border_valid_mask(tir, black_threshold=black_threshold)
    gray = np.asarray(tir.convert("L"), dtype=np.float32)
    valid_values = gray[valid]
    if valid_values.size:
        p1, p99 = np.percentile(valid_values, [1, 99]).tolist()
        std = float(valid_values.std())
    else:
        p1 = p99 = std = 0.0
    valid_ratio = float(valid.mean())
    low_information = (p99 - p1) < 8.0 or std < 3.0
    same_size = rgb.size == tir.size
    return PairInspection(
        rgb_width=rgb.width,
        rgb_height=rgb.height,
        tir_width=tir.width,
        tir_height=tir.height,
        same_spatial_dimensions=same_size,
        ir_valid_ratio=valid_ratio,
        ir_intensity_p1=float(p1),
        ir_intensity_p99=float(p99),
        ir_intensity_std=std,
        ir_low_information=low_information,
        ir_usable=same_size and valid_ratio >= 0.5 and not low_information,
    )


class PairedRGBTProcessor:
    """Turn a registered RGB/TIR pair into Qwen-ready patches through one seam."""

    def __init__(
        self,
        *,
        min_pixels: int = 262144,
        max_pixels: int = 1310720,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        black_threshold: int = 3,
        image_mean: Sequence[float] = (0.5, 0.5, 0.5),
        image_std: Sequence[float] = (0.5, 0.5, 0.5),
        image_processor: Any | None = None,
    ) -> None:
        if patch_size <= 0 or spatial_merge_size <= 0:
            raise ValueError("patch and merge sizes must be positive")
        self.patch_size = int(patch_size)
        self.spatial_merge_size = int(spatial_merge_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.black_threshold = int(black_threshold)
        if image_processor is None:
            from transformers import Qwen2VLImageProcessor

            image_processor = Qwen2VLImageProcessor(
                min_pixels=int(min_pixels),
                max_pixels=int(max_pixels),
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                merge_size=self.spatial_merge_size,
                image_mean=list(image_mean),
                image_std=list(image_std),
            )
        self.image_processor = image_processor

    def process(
        self, record: RGBTRecord, *, root: Path | str
    ) -> PairedRGBTBatch:
        base = Path(root).resolve()
        return self.process_paths(
            rgb_path=base / record.rgb_relpath,
            tir_path=base / record.tir_relpath,
            normalized_bbox=record.bbox_xyxy_normalized,
        )

    def process_paths(
        self,
        *,
        rgb_path: Path | str,
        tir_path: Path | str,
        normalized_bbox: Sequence[float] | None = None,
    ) -> PairedRGBTBatch:
        rgb = _load_rgb(Path(rgb_path))
        tir = _load_rgb(Path(tir_path))
        if rgb.size != tir.size:
            raise ValueError(
                f"RGB/TIR source dimensions differ: rgb={rgb.size}, tir={tir.size}"
            )
        valid_mask = border_valid_mask(tir, black_threshold=self.black_threshold)
        inspection = inspect_pair_paths(
            rgb_path, tir_path, black_threshold=self.black_threshold
        )
        processed = self.image_processor(images=[rgb, tir], return_tensors="pt")
        grids = processed["image_grid_thw"]
        if tuple(grids.shape) != (2, 3):
            raise ValueError(f"unexpected paired grid shape: {tuple(grids.shape)}")
        if not torch.equal(grids[0], grids[1]):
            raise ValueError(f"RGB/TIR grid_thw mismatch: {grids.tolist()}")
        grid = grids[0].to(dtype=torch.int64)
        grid_t, grid_h, grid_w = (int(value) for value in grid.tolist())
        if grid_t != 1:
            raise ValueError(f"image grid temporal dimension must be one, got {grid_t}")
        per_image_tokens = grid_t * grid_h * grid_w
        pixels = processed["pixel_values"]
        if pixels.shape[0] != 2 * per_image_tokens:
            raise ValueError(
                f"unexpected flattened patch count {pixels.shape[0]} for grid {grid.tolist()}"
            )
        rgb_pixels, tir_pixels = pixels.split(per_image_tokens, dim=0)

        target_width = grid_w * self.patch_size
        target_height = grid_h * self.patch_size
        resized_mask = Image.fromarray((valid_mask * 255).astype(np.uint8)).resize(
            (target_width, target_height), resample=Image.Resampling.NEAREST
        )
        mask_tensor = torch.from_numpy(
            np.asarray(resized_mask, dtype=np.uint8).copy()
        ).to(dtype=torch.float32) / 255.0
        patches = mask_tensor.reshape(
            grid_h, self.patch_size, grid_w, self.patch_size
        ).mean(dim=(1, 3))
        merge = self.spatial_merge_size
        if grid_h % merge or grid_w % merge:
            raise ValueError(f"grid {grid_h}x{grid_w} is not divisible by merge={merge}")
        ordered_patch_mask = (
            patches.reshape(grid_h // merge, merge, grid_w // merge, merge)
            .permute(0, 2, 1, 3)
            .reshape(-1)
        )
        merged_mask = (
            patches.reshape(grid_h // merge, merge, grid_w // merge, merge)
            .mean(dim=(1, 3))
            .reshape(-1)
        )
        geometry = SharedGeometryTransform(
            original_width=rgb.width,
            original_height=rgb.height,
            resized_width=target_width,
            resized_height=target_height,
        )
        normalized = (
            None
            if normalized_bbox is None
            else tuple(float(value) for value in normalized_bbox)
        )
        resized_bbox = (
            None if normalized is None else geometry.normalized_to_resized(normalized)
        )
        return PairedRGBTBatch(
            rgb_pixel_values=rgb_pixels,
            tir_pixel_values=tir_pixels,
            image_grid_thw=grid.unsqueeze(0),
            ir_pixel_valid_mask=mask_tensor,
            ir_patch_valid_mask=ordered_patch_mask,
            ir_merged_valid_mask=merged_mask,
            shared_geometry_transform=geometry,
            original_rgb_size=rgb.size,
            normalized_bbox=normalized,
            resized_bbox=resized_bbox,
            ir_usable=inspection.ir_usable,
            ir_valid_ratio=inspection.ir_valid_ratio,
        )


def qwen_grid_for_size(
    width: int,
    height: int,
    *,
    min_pixels: int,
    max_pixels: int,
    patch_size: int = 16,
    merge_size: int = 2,
) -> tuple[int, int, int]:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if not all(math.isfinite(value) for value in (resized_height, resized_width)):
        raise ValueError("non-finite smart-resize output")
    return 1, resized_height // patch_size, resized_width // patch_size
