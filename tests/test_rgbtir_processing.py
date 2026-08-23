from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from aic_rgbtir.processing import PairedRGBTProcessor, border_valid_mask, inspect_pair_paths


def test_border_mask_handles_tilted_padding_without_masking_internal_dark_target() -> None:
    array = np.full((64, 96, 3), 120, dtype=np.uint8)
    for y in range(64):
        array[y, : 4 + y // 8] = 0
    array[25:35, 45:55] = 0
    mask = border_valid_mask(Image.fromarray(array), black_threshold=3)
    assert not mask[50, 3]
    assert mask[30, 50]
    assert mask[10, 70]


def test_processor_uses_shared_grid_masks_and_reversible_bbox(tmp_path: Path) -> None:
    rgb = np.full((64, 96, 3), 150, dtype=np.uint8)
    tir = np.full((64, 96, 3), 100, dtype=np.uint8)
    tir[:, :8] = 0
    tir[20:30, 50:60] = 0
    rgb_path = tmp_path / "rgb.png"
    tir_path = tmp_path / "tir.png"
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(tir).save(tir_path)

    processor = PairedRGBTProcessor(
        min_pixels=64 * 96,
        max_pixels=64 * 96,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        black_threshold=3,
    )
    assert processor.image_processor.image_mean == [0.5, 0.5, 0.5]
    assert processor.image_processor.image_std == [0.5, 0.5, 0.5]
    normalized = (0.1, 0.2, 0.6, 0.8)
    batch = processor.process_paths(
        rgb_path=rgb_path,
        tir_path=tir_path,
        normalized_bbox=normalized,
    )
    grid_t, grid_h, grid_w = batch.image_grid_thw.squeeze(0).tolist()
    assert grid_t == 1
    assert batch.rgb_pixel_values.shape == batch.tir_pixel_values.shape
    assert batch.ir_patch_valid_mask.numel() == grid_t * grid_h * grid_w
    assert batch.ir_merged_valid_mask.numel() == grid_t * grid_h * grid_w // 4
    restored_normalized = batch.shared_geometry_transform.resized_to_normalized(
        batch.resized_bbox
    )
    assert max(abs(a - b) for a, b in zip(normalized, restored_normalized)) < 1e-9
    inspection = inspect_pair_paths(rgb_path, tir_path)
    assert inspection.same_spatial_dimensions
    assert inspection.ir_valid_ratio < 1.0
    assert inspection.ir_usable


def test_processor_rejects_unregistered_source_dimensions(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    tir_path = tmp_path / "tir.png"
    Image.new("RGB", (96, 64), color=(100, 100, 100)).save(rgb_path)
    Image.new("RGB", (80, 64), color=(100, 100, 100)).save(tir_path)
    processor = PairedRGBTProcessor(min_pixels=4096, max_pixels=8192)
    try:
        processor.process_paths(rgb_path=rgb_path, tir_path=tir_path)
    except ValueError as exc:
        assert "source dimensions differ" in str(exc)
    else:
        raise AssertionError("mismatched source dimensions must be rejected")
