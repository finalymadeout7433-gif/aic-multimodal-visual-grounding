"""Read-only RGBT-GroundBench audit and AIC domain comparison.

The source archives, extracted images, and annotations are never modified.  All
derived files are written below ``--output-root``.  RGBT condition statistics
come from released labels; AIC weather/scale statements remain input-derived or
heuristic proxies because the competition test set has no public ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import cv2
import numpy as np
import torch

from analyze_aic_ir_registration import boundary_metrics, read_image, registration_proxy


DATASETS = ("flir", "m3fd", "mfad")
SPLITS = ("train", "val", "test")

ILLUMINATION = {1: "VWL", 2: "WL", 3: "NL", 4: "SL"}
WEATHER = {0: "FY", 1: "RY", 2: "SY", 3: "CY"}
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
OCCLUSION = {0: "NO", 1: "PO", 2: "HO"}
CROWDED = {0: "NC", 1: "C", 2: "HC"}

RELATION_RE = re.compile(
    r"\b(left|right|above|below|under|over|beside|near|next to|between|inside|outside|"
    r"behind|in front of|center|middle|corner|edge)\b",
    re.IGNORECASE,
)
ORDINAL_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|leftmost|rightmost|topmost|bottommost|last)\b",
    re.IGNORECASE,
)
DEPTH_RE = re.compile(
    r"\b(nearest|farthest|closest|furthest|distant|foreground|background|behind|in front of)\b",
    re.IGNORECASE,
)
ACTION_RE = re.compile(
    r"\b(standing|walking|sitting|riding|holding|carrying|wearing|running|driving|parked)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(red|green|blue|yellow|white|black|gray|grey|silver|brown|orange|pink|purple)\b",
    re.IGNORECASE,
)
STRICT_WEATHER_RE = re.compile(
    r"\b(foggy|rainy|raining|snowy|snowing|misty|hazy|stormy|overcast)\b|"
    r"\b(in|during|through) (the )?(rain|fog|snow|mist|storm)\b",
    re.IGNORECASE,
)
INDIRECT_WET_RE = re.compile(r"\b(wet|puddle|waterlogged|rain-soaked)\b", re.IGNORECASE)
NEGATIVE_OR_ABSENT_RE = re.compile(
    r"\b(no|not)\s+(?:\w+\s+){0,5}(present|visible|found|correspond)", re.IGNORECASE
)
BBOX_META_RE = re.compile(r"\b(bounding box|bbox)\b|coordinates\s*\[", re.IGNORECASE)
UNCERTAIN_RE = re.compile(r"\b(possibly|appears to be|indistinct|faint)\b", re.IGNORECASE)

TARGET_PATTERNS = (
    (
        "person",
        re.compile(
            r"\b(person|pedestrian|man|woman|boy|girl|figure|worker|cyclist|motorcyclist|rider)\b",
            re.IGNORECASE,
        ),
    ),
    ("car", re.compile(r"\b(car|sedan|suv|automobile|taxi)\b", re.IGNORECASE)),
    ("truck", re.compile(r"\b(truck|pickup|lorry|semi-truck)\b", re.IGNORECASE)),
    ("bus", re.compile(r"\b(bus|coach)\b", re.IGNORECASE)),
    (
        "two_wheeler",
        re.compile(r"\b(motorcycle|motorbike|bicycle|bike|scooter|electric bicycle)\b", re.IGNORECASE),
    ),
    ("van", re.compile(r"\bvan\b", re.IGNORECASE)),
    ("other_object", re.compile(r"\b(dog|lamp|traffic light|cone|sign|object)\b", re.IGNORECASE)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Iterable[float], q: float) -> float | None:
    seq = list(values)
    if not seq:
        return None
    return float(np.percentile(np.asarray(seq, dtype=np.float64), q))


def load_records(path: Path) -> list[list[Any]]:
    records = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(records, list):
        raise TypeError(f"Expected list in {path}, got {type(records)!r}")
    return records


def query_flags(query: str) -> dict[str, bool]:
    return {
        "has_spatial_relation": bool(RELATION_RE.search(query)),
        "has_ordinal": bool(ORDINAL_RE.search(query)),
        "has_depth_relation": bool(DEPTH_RE.search(query)),
        "has_action": bool(ACTION_RE.search(query)),
        "has_color_attribute": bool(COLOR_RE.search(query)),
        "has_strict_weather_text": bool(STRICT_WEATHER_RE.search(query)),
        "has_indirect_wet_cue": bool(INDIRECT_WET_RE.search(query)),
        "negative_or_absent_language": bool(NEGATIVE_OR_ABSENT_RE.search(query)),
        "bbox_meta_language": bool(BBOX_META_RE.search(query)),
        "uncertain_language": bool(UNCERTAIN_RE.search(query)),
    }


def target_category_proxy(query: str) -> str:
    """Lexical target proxy from the first 12 words, not a released class label."""
    prefix = " ".join(re.findall(r"[A-Za-z-]+", query)[:12])
    for label, pattern in TARGET_PATTERNS:
        if pattern.search(prefix):
            return label
    return "unknown_other"


def summarize_counter(counter: Counter[str], total: int, scope: str) -> list[dict[str, Any]]:
    return [
        {
            "scope": scope,
            "label": label,
            "count": count,
            "ratio": count / total if total else 0.0,
        }
        for label, count in sorted(counter.items())
    ]


def collect_rgbt(annotation_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for dataset in DATASETS:
        base = annotation_root / f"rgbtvg_{dataset}"
        for split in SPLITS:
            path = base / f"rgbtvg_{dataset}_{split}.pth"
            records = load_records(path)
            split_counts[f"{dataset}_{split}"] = len(records)
            for index, item in enumerate(records):
                if len(item) != 10:
                    raise ValueError(f"Unexpected record length {len(item)} in {path}")
                file_name, size, bbox, query = item[0], item[1], item[2], item[3]
                width = int(size["width"])
                height = int(size["height"])
                x, y, box_w, box_h = map(float, bbox)
                area = max(0.0, box_w) * max(0.0, box_h) / max(1.0, width * height)
                flags = query_flags(str(query))
                rows.append(
                    {
                        "record_id": f"{dataset}:{split}:{index}",
                        "dataset": dataset,
                        "split": split,
                        "image_file": str(file_name),
                        "width": width,
                        "height": height,
                        "bbox_x": x,
                        "bbox_y": y,
                        "bbox_width": box_w,
                        "bbox_height": box_h,
                        "bbox_area_ratio": area,
                        "query": str(query),
                        "query_token_count": len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(query))),
                        "target_category_lexical_proxy": target_category_proxy(str(query)),
                        "illumination": ILLUMINATION.get(int(item[4]), f"UNKNOWN_{item[4]}"),
                        "object_size": "SS" if str(item[5]).lower() == "small" else "NS",
                        "crowded": CROWDED.get(int(item[6]), f"UNKNOWN_{item[6]}"),
                        "scene": SCENE.get(int(item[7]), f"UNKNOWN_{item[7]}"),
                        "weather": WEATHER.get(int(item[8]), f"UNKNOWN_{item[8]}"),
                        "occlusion": OCCLUSION.get(int(item[9]), f"UNKNOWN_{item[9]}"),
                        **flags,
                    }
                )
    return rows, {"split_counts": split_counts}


def aggregate_rgbt(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition_rows: list[dict[str, Any]] = []
    for scope, selected in [("all_instances", rows)] + [
        (f"{dataset}_instances", [row for row in rows if row["dataset"] == dataset])
        for dataset in DATASETS
    ]:
        total = len(selected)
        for field in ("illumination", "weather", "object_size", "occlusion", "scene", "crowded"):
            counter = Counter(str(row[field]) for row in selected)
            for entry in summarize_counter(counter, total, scope):
                entry["field"] = field
                condition_rows.append(entry)

    # One released image pair may have multiple grounding instances.  Pair-level
    # weather/illumination statistics are therefore separately de-duplicated.
    pair_rows: dict[tuple[str, str], dict[str, Any]] = {}
    pair_conflicts: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["dataset"]), str(row["image_file"]))
        if key in pair_rows:
            for field in ("width", "height", "illumination", "weather", "scene"):
                if pair_rows[key][field] != row[field]:
                    pair_conflicts.append(
                        {"dataset": key[0], "image_file": key[1], "field": field}
                    )
        else:
            pair_rows[key] = row

    pairs = list(pair_rows.values())
    for field in ("illumination", "weather", "scene"):
        counter = Counter(str(row[field]) for row in pairs)
        for entry in summarize_counter(counter, len(pairs), "unique_image_pairs"):
            entry["field"] = field
            condition_rows.append(entry)

    lengths = [int(row["query_token_count"]) for row in rows]
    areas = [float(row["bbox_area_ratio"]) for row in rows]
    query_summary: dict[str, Any] = {
        "instance_count": len(rows),
        "unique_image_pair_count": len(pairs),
        "query_token_mean": mean(lengths),
        "query_token_median": median(lengths),
        "query_token_p90": percentile(lengths, 90),
        "bbox_area_ratio_median": median(areas),
        "bbox_area_ratio_p10": percentile(areas, 10),
        "bbox_area_ratio_p90": percentile(areas, 90),
        "small_label_ratio": sum(row["object_size"] == "SS" for row in rows) / len(rows),
        "low_light_label_ratio": sum(
            row["illumination"] in {"WL", "VWL"} for row in rows
        )
        / len(rows),
        "adverse_weather_label_ratio": sum(row["weather"] in {"FY", "RY"} for row in rows)
        / len(rows),
        "query_flag_ratios": {
            key: sum(bool(row[key]) for row in rows) / len(rows)
            for key in (
                "has_spatial_relation",
                "has_ordinal",
                "has_depth_relation",
                "has_action",
                "has_color_attribute",
                "has_strict_weather_text",
                "has_indirect_wet_cue",
                "negative_or_absent_language",
                "bbox_meta_language",
                "uncertain_language",
            )
        },
        "target_category_lexical_proxy_counts": dict(
            sorted(Counter(str(row["target_category_lexical_proxy"]) for row in rows).items())
        ),
        "split_statistics": {},
        "pair_label_conflict_count": len(pair_conflicts),
        "pair_label_conflicts": pair_conflicts[:50],
    }
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        split_lengths = [int(row["query_token_count"]) for row in subset]
        split_areas = [float(row["bbox_area_ratio"]) for row in subset]
        query_summary["split_statistics"][split] = {
            "instance_count": len(subset),
            "low_light_label_ratio": sum(
                row["illumination"] in {"WL", "VWL"} for row in subset
            )
            / len(subset),
            "adverse_weather_label_ratio": sum(
                row["weather"] in {"FY", "RY"} for row in subset
            )
            / len(subset),
            "small_label_ratio": sum(row["object_size"] == "SS" for row in subset) / len(subset),
            "query_token_mean": mean(split_lengths),
            "bbox_area_ratio_median": median(split_areas),
            "weather_counts": dict(sorted(Counter(row["weather"] for row in subset).items())),
        }
    return condition_rows, query_summary


def resolve_pair(image_root: Path, dataset: str, file_name: str) -> tuple[Path, Path]:
    rgb = image_root / dataset / "rgb" / file_name
    ir = image_root / dataset / "ir" / file_name
    if rgb.exists() and ir.exists():
        return rgb, ir
    # Some releases preserve the stem but use modality-specific extensions.
    rgb_matches = list((image_root / dataset / "rgb").glob(f"{Path(file_name).stem}.*"))
    ir_matches = list((image_root / dataset / "ir").glob(f"{Path(file_name).stem}.*"))
    if len(rgb_matches) == 1 and len(ir_matches) == 1:
        return rgb_matches[0], ir_matches[0]
    raise FileNotFoundError(f"Cannot resolve pair for {dataset}/{file_name}")


def audit_image_pairs(
    rows: list[dict[str, Any]], image_root: Path, sample_per_dataset: int, probe_per_dataset: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    unique: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["dataset"]), str(row["image_file"]))
        if key not in seen:
            seen.add(key)
            unique[key[0]].append(key[1])

    sampled: list[tuple[str, str, bool]] = []
    for dataset in DATASETS:
        names = sorted(unique[dataset])
        count = min(sample_per_dataset, len(names))
        indices = sorted(rng.choice(len(names), size=count, replace=False).tolist())
        probe_set = set(indices[: min(probe_per_dataset, len(indices))])
        sampled.extend((dataset, names[index], index in probe_set) for index in indices)

    result_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for dataset, file_name, do_probe in sampled:
        try:
            rgb_path, ir_path = resolve_pair(image_root, dataset, file_name)
            rgb = read_image(rgb_path)
            ir = read_image(ir_path)
            border = boundary_metrics(ir)
            row: dict[str, Any] = {
                "dataset": dataset,
                "image_file": file_name,
                "rgb_extension": rgb_path.suffix.lower(),
                "ir_extension": ir_path.suffix.lower(),
                "rgb_dtype": str(rgb.dtype),
                "ir_dtype": str(ir.dtype),
                "rgb_shape": "x".join(map(str, rgb.shape)),
                "ir_shape": "x".join(map(str, ir.shape)),
                "same_spatial_dimensions": rgb.shape[:2] == ir.shape[:2],
                "ir_border_invalid_fraction": border["border_invalid_fraction"],
                "ir_left_boundary_angle_deg": border["left_boundary_angle_deg"],
                "ir_right_boundary_angle_deg": border["right_boundary_angle_deg"],
                "registration_probed": do_probe,
            }
            if do_probe:
                row.update(registration_proxy(rgb, ir))
            result_rows.append(row)
        except Exception as exc:  # keep audit streaming and report exact failures
            failures.append({"dataset": dataset, "image_file": file_name, "error": repr(exc)})

    border_values = [float(row["ir_border_invalid_fraction"]) for row in result_rows]
    probe_rows = [row for row in result_rows if row["registration_probed"]]
    def probe_quantiles(field: str, absolute: bool = False) -> dict[str, Any]:
        values = []
        for row in probe_rows:
            value = row.get(field)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            values.append(abs(value) if absolute else value)
        return {
            "count": len(values),
            "median": median(values) if values else None,
            "p90": percentile(values, 90),
        }
    image_summary = {
        "sample_count": len(result_rows),
        "failure_count": len(failures),
        "failures": failures,
        "same_spatial_dimensions_ratio": sum(
            bool(row["same_spatial_dimensions"]) for row in result_rows
        )
        / max(1, len(result_rows)),
        "ir_border_invalid_fraction_median": median(border_values) if border_values else None,
        "ir_border_invalid_fraction_p90": percentile(border_values, 90),
        "registration_probe_count": len(probe_rows),
        "registration_accepted_proxy_count": sum(
            row.get("registration_status") == "accepted_proxy" for row in probe_rows
        ),
        "registration_proxy_quantiles": {
            "structure_corr_before": probe_quantiles("structure_corr_before"),
            "structure_corr_after": probe_quantiles("structure_corr_after"),
            "abs_rotation_deg": probe_quantiles("estimated_rotation_deg", absolute=True),
            "scale": probe_quantiles("estimated_scale"),
            "abs_translation_x_fraction": probe_quantiles(
                "estimated_translation_x_fraction", absolute=True
            ),
            "abs_translation_y_fraction": probe_quantiles(
                "estimated_translation_y_fraction", absolute=True
            ),
        },
        "registration_interpretation": (
            "Cross-spectral structure-only affine proxy; not calibration ground truth."
        ),
    }
    return result_rows, image_summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_aic(profile_root: Path) -> dict[str, Any]:
    query_path = profile_root / "query_profile.jsonl"
    image_path = profile_root / "image_group_profile.csv"
    run_summary_path = profile_root / "run_summary.json"
    queries = read_jsonl(query_path)
    with image_path.open("r", encoding="utf-8-sig", newline="") as handle:
        images = list(csv.DictReader(handle))
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))

    strict_weather = [row for row in queries if STRICT_WEATHER_RE.search(row["query_original"])]
    indirect_wet = [row for row in queries if INDIRECT_WET_RE.search(row["query_original"])]
    strict_groups = sorted({row["image_group_id"] for row in strict_weather})
    indirect_groups = sorted({row["image_group_id"] for row in indirect_wet})
    metrics = run_summary.get("metrics", {})
    return {
        "query_count": len(queries),
        "image_group_count": len(images),
        "query_token_mean": metrics.get("query_token_mean"),
        "query_token_median": metrics.get("query_token_median"),
        "low_light_group_ratio_proxy": metrics.get("low_light_group_ratio"),
        "region_structure_ratio": metrics.get("region_structure_ratio"),
        "spatial_relation_ratio": metrics.get("spatial_relation_ratio"),
        "ordinal_ratio": metrics.get("ordinal_ratio"),
        "depth_relation_ratio": metrics.get("depth_relation_ratio"),
        "action_ratio": metrics.get("action_ratio"),
        "strict_weather_query_count": len(strict_weather),
        "strict_weather_image_group_count": len(strict_groups),
        "strict_weather_examples": [
            {"query_id": row["query_id"], "query": row["query_original"]}
            for row in strict_weather[:50]
        ],
        "indirect_wet_query_count": len(indirect_wet),
        "indirect_wet_image_group_count": len(indirect_groups),
        "indirect_wet_examples": [
            {"query_id": row["query_id"], "query": row["query_original"]}
            for row in indirect_wet[:50]
        ],
        "weather_evidence_boundary": (
            "AIC has no public weather labels. Query words and low-light image statistics are proxies "
            "and cannot prove that an image group is or is not adverse-weather data."
        ),
    }


def build_comparison(rgbt: dict[str, Any], aic: dict[str, Any], image_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "rgbt_groundbench": {
            **rgbt,
            "weather_and_bbox_evidence_kind": "released_ground_truth_label",
            "image_alignment_evidence_kind": "sampled_structure_proxy",
            "image_audit": image_summary,
        },
        "aic": {
            **aic,
            "weather_evidence_kind": "query_and_image_heuristic_proxy",
            "bbox_scale_evidence_kind": "model_prediction_proxy_not_ground_truth",
        },
        "claim_boundary": {
            "can_confirm": [
                "RGBT-GroundBench contains labeled foggy and rainy samples.",
                "RGBT-GroundBench has released ground-truth boxes and condition labels.",
                "AIC query and image profiles differ substantially from RGBT-GroundBench.",
            ],
            "cannot_confirm_without_aic_ground_truth": [
                "The true adverse-weather share of AIC.",
                "The true target-size distribution of AIC.",
                "Whether a particular AIC prediction is improved by TIR.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgbt-root", type=Path, required=True)
    parser.add_argument("--aic-profile-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-per-dataset", type=int, default=200)
    parser.add_argument("--probe-per-dataset", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    annotation_root = args.rgbt_root / "extracted"
    image_root = annotation_root / "image_data"
    rows, loading = collect_rgbt(annotation_root)
    conditions, rgbt_summary = aggregate_rgbt(rows)
    image_rows, image_summary = audit_image_pairs(
        rows,
        image_root,
        sample_per_dataset=args.sample_per_dataset,
        probe_per_dataset=args.probe_per_dataset,
        seed=args.seed,
    )
    aic_summary = collect_aic(args.aic_profile_root)
    comparison = build_comparison(rgbt_summary, aic_summary, image_summary)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "rgbt_instance_profile.csv", rows)
    write_csv(args.output_root / "rgbt_condition_counts.csv", conditions)
    write_csv(args.output_root / "rgbt_alignment_sample.csv", image_rows)
    json_dump(args.output_root / "rgbt_summary.json", {**loading, **rgbt_summary, **image_summary})
    json_dump(args.output_root / "aic_weather_proxy.json", aic_summary)
    json_dump(args.output_root / "rgbt_aic_comparison.json", comparison)

    manifest: dict[str, Any] = {
        "pipeline": "rgbt_groundbench_aic_fit_audit_v1",
        "seed": args.seed,
        "source_root": str(args.rgbt_root),
        "source_is_read_only": True,
        "outputs": {},
    }
    for path in sorted(args.output_root.glob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            manifest["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    json_dump(args.output_root / "sha256_manifest.json", manifest)

    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
