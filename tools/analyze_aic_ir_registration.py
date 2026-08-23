"""Read-only AIC RGB/infrared registration and black-border audit.

The script never edits the official dataset. It writes derived CSV/JSON/Markdown
artifacts below the repository output/report directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def gray8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return image
    valid = image[np.isfinite(image)]
    if valid.size == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    return np.clip((image.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def border_connected_dark_mask(image: np.ndarray, threshold: int = 3) -> np.ndarray:
    """Return dark components connected to an image edge.

    This isolates padding/wrap borders without labeling isolated dark objects in
    the scene as padding.
    """
    gray = gray8(image)
    dark = (gray <= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    invalid = np.zeros(gray.shape, dtype=bool)
    height, width = gray.shape
    for label in range(1, count):
        x, y, w, h, _area = stats[label]
        if x == 0 or y == 0 or x + w == width or y + h == height:
            invalid |= labels == label
    return invalid


def boundary_metrics(image: np.ndarray) -> dict[str, float | int]:
    invalid = border_connected_dark_mask(image)
    valid = ~invalid
    height, width = valid.shape
    ys: list[int] = []
    lefts: list[int] = []
    rights: list[int] = []
    for y in range(height):
        xs = np.flatnonzero(valid[y])
        if xs.size:
            ys.append(y)
            lefts.append(int(xs[0]))
            rights.append(int(width - 1 - xs[-1]))

    result: dict[str, float | int] = {
        "width": width,
        "height": height,
        "border_invalid_fraction": float(invalid.mean()),
    }
    if len(ys) < max(10, height // 10):
        result.update(
            {
                "left_top_px": width,
                "left_mid_px": width,
                "left_bottom_px": width,
                "right_top_px": width,
                "right_mid_px": width,
                "right_bottom_px": width,
                "left_boundary_angle_deg": float("nan"),
                "right_boundary_angle_deg": float("nan"),
            }
        )
        return result

    y_arr = np.asarray(ys, dtype=np.float64)
    left_arr = np.asarray(lefts, dtype=np.float64)
    right_arr = np.asarray(rights, dtype=np.float64)
    keep = (y_arr >= height * 0.02) & (y_arr <= height * 0.98)
    left_slope = float(np.polyfit(y_arr[keep], left_arr[keep], 1)[0])
    right_slope = float(np.polyfit(y_arr[keep], right_arr[keep], 1)[0])

    def at(values: np.ndarray, fraction: float) -> int:
        index = int(round((len(values) - 1) * fraction))
        return int(values[index])

    result.update(
        {
            "left_top_px": at(left_arr, 0.0),
            "left_mid_px": at(left_arr, 0.5),
            "left_bottom_px": at(left_arr, 1.0),
            "right_top_px": at(right_arr, 0.0),
            "right_mid_px": at(right_arr, 0.5),
            "right_bottom_px": at(right_arr, 1.0),
            "left_boundary_angle_deg": math.degrees(math.atan(left_slope)),
            # right_arr stores padding width, so negate it to recover x-boundary slope.
            "right_boundary_angle_deg": math.degrees(math.atan(-right_slope)),
        }
    )
    return result


def structure_map(image: np.ndarray, size: tuple[int, int] = (320, 180)) -> np.ndarray:
    gray = cv2.resize(gray8(image), size, interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray_f = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    scale = float(np.percentile(magnitude, 98))
    if scale > 0:
        magnitude = np.clip(magnitude / scale, 0, 1)
    return cv2.GaussianBlur(magnitude, (7, 7), 0)


def correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    x = first[mask].astype(np.float64)
    y = second[mask].astype(np.float64)
    if x.size < 100 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def registration_proxy(visible: np.ndarray, infrared: np.ndarray) -> dict[str, float | str]:
    """Estimate a conservative structure-only affine proxy.

    Cross-spectral ECC is not a calibration ground truth. The result is only
    accepted when it is plausible and improves structure correlation.
    """
    width, height = 320, 180
    visible_structure = structure_map(visible, (width, height))
    infrared_structure = structure_map(infrared, (width, height))
    invalid = border_connected_dark_mask(infrared)
    valid_mask = cv2.resize((~invalid).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    valid_mask = cv2.erode(valid_mask, np.ones((5, 5), np.uint8))
    before = correlation(visible_structure, infrared_structure, valid_mask > 0)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _ecc, warp = cv2.findTransformECC(
            visible_structure,
            infrared_structure,
            warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 180, 1e-6),
            valid_mask,
            5,
        )
    except cv2.error:
        return {
            "registration_status": "ecc_failed",
            "structure_corr_before": before,
            "structure_corr_after": float("nan"),
            "estimated_rotation_deg": float("nan"),
            "estimated_scale": float("nan"),
            "estimated_translation_x_fraction": float("nan"),
            "estimated_translation_y_fraction": float("nan"),
        }

    warped = cv2.warpAffine(
        infrared_structure,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )
    warped_mask = cv2.warpAffine(
        valid_mask,
        warp,
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
    )
    after = correlation(visible_structure, warped, warped_mask > 0)
    ir_to_rgb = cv2.invertAffineTransform(warp)
    a, b = float(ir_to_rgb[0, 0]), float(ir_to_rgb[0, 1])
    c, d = float(ir_to_rgb[1, 0]), float(ir_to_rgb[1, 1])
    rotation = math.degrees(math.atan2(c, a))
    scale = math.sqrt(abs(a * d - b * c))
    tx_fraction = float(ir_to_rgb[0, 2] / width)
    ty_fraction = float(ir_to_rgb[1, 2] / height)
    plausible = (
        np.isfinite(after)
        and np.isfinite(before)
        and after - before >= 0.025
        and abs(rotation) <= 10
        and 0.8 <= scale <= 1.2
        and abs(tx_fraction) <= 0.2
        and abs(ty_fraction) <= 0.2
    )
    return {
        "registration_status": "accepted_proxy" if plausible else "low_confidence_proxy",
        "structure_corr_before": before,
        "structure_corr_after": after,
        "estimated_rotation_deg": rotation,
        "estimated_scale": scale,
        "estimated_translation_x_fraction": tx_fraction,
        "estimated_translation_y_fraction": ty_fraction,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scan-count", type=int, default=400)
    parser.add_argument("--probe-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    queries_path = args.dataset_root / "queries" / "queries.json"
    records = json.loads(queries_path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    groups: OrderedDict[tuple[str, str, str], None] = OrderedDict()
    for record in records.values():
        key = (record["visible"], record["infrared"], record["depth"])
        groups.setdefault(key, None)

    group_items = list(groups)
    rng = np.random.default_rng(args.seed)
    required_index = next(
        (idx for idx, group in enumerate(group_items) if Path(group[0]).stem == "000002"),
        None,
    )
    all_indices = np.arange(len(group_items))
    scan_indices = list(
        rng.choice(all_indices, size=min(args.scan_count, len(all_indices)), replace=False)
    )
    if required_index is not None and required_index not in scan_indices:
        scan_indices[-1] = required_index
    scan_indices = sorted(set(scan_indices))

    border_rows: list[dict[str, object]] = []
    for index in scan_indices:
        visible_rel, infrared_rel, depth_rel = group_items[index]
        infrared = read_image(args.dataset_root / infrared_rel)
        row: dict[str, object] = {
            "group_index": index,
            "group_id": Path(visible_rel).stem,
            "visible": visible_rel,
            "infrared": infrared_rel,
            "depth": depth_rel,
        }
        row.update(boundary_metrics(infrared))
        border_rows.append(row)

    chosen = list(
        rng.choice(all_indices, size=min(args.probe_count, len(all_indices)), replace=False)
    )
    if required_index is not None and required_index not in chosen:
        chosen[-1] = required_index
    chosen = sorted(set(chosen))

    probe_rows: list[dict[str, object]] = []
    for index in chosen:
        visible_rel, infrared_rel, _depth_rel = group_items[index]
        visible = read_image(args.dataset_root / visible_rel)
        infrared = read_image(args.dataset_root / infrared_rel)
        probe_rows.append(
            {
                "group_index": index,
                "group_id": Path(visible_rel).stem,
                **registration_proxy(visible, infrared),
            }
        )

    output_dir = args.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "infrared_border_geometry.csv", border_rows)
    write_csv(output_dir / "rgb_ir_registration_proxy.csv", probe_rows)

    invalid = np.asarray([float(row["border_invalid_fraction"]) for row in border_rows])
    angles = np.asarray(
        [
            max(abs(float(row["left_boundary_angle_deg"])), abs(float(row["right_boundary_angle_deg"])))
            for row in border_rows
        ]
    )
    accepted = [row for row in probe_rows if row["registration_status"] == "accepted_proxy"]
    group_000002 = next(row for row in border_rows if row["group_id"] == "000002")
    probe_000002 = next(row for row in probe_rows if row["group_id"] == "000002")
    summary = {
        "query_count": len(records),
        "image_group_count": len(group_items),
        "border_scan_sample_count": len(border_rows),
        "sampling_seed": args.seed,
        "border_threshold": 3,
        "ir_border_invalid_fraction": {
            "median": float(np.median(invalid)),
            "p90": float(np.percentile(invalid, 90)),
            "max": float(np.max(invalid)),
            "count_ge_1pct": int(np.sum(invalid >= 0.01)),
            "count_ge_10pct": int(np.sum(invalid >= 0.10)),
        },
        "ir_boundary_angle_abs_deg": {
            "median": float(np.nanmedian(angles)),
            "p90": float(np.nanpercentile(angles, 90)),
            "count_ge_1deg": int(np.sum(angles >= 1.0)),
        },
        "registration_probe": {
            "sample_count": len(probe_rows),
            "accepted_proxy_count": len(accepted),
            "interpretation": "Cross-spectral affine proxy only; not calibration ground truth.",
        },
        "group_000002": {**group_000002, **probe_000002},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
