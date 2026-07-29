"""2026 AIC 数据集只读审计。

只统计 JSON、文件、真实图像格式、dtype、通道和三模态尺寸。
不修改原始数据，不生成 bbox，也不读取模型。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REQUIRED_FIELDS = ("visible", "infrared", "depth", "query")
QUERY_ID_PATTERN = re.compile(r"^\d{6}_\d{3}$")
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, dict):
        raise ValueError("queries JSON 顶层必须是对象")
    return records


def resolve_path(dataset_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def inspect_image(path: Path, modality: str) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image)
        result: dict[str, Any] = {
            "format": image.format or "UNKNOWN",
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "dtype": str(array.dtype),
            "channels": 1 if array.ndim == 2 else int(array.shape[2]),
            "min": int(array.min()),
            "max": int(array.max()),
            "zero_ratio": float(np.mean(array == 0)),
        }
        if modality == "infrared" and array.ndim == 3 and array.shape[2] >= 3:
            signed = array[..., :3].astype(np.int32)
            result["max_channel_difference"] = int(
                max(
                    np.abs(signed[..., 0] - signed[..., 1]).max(),
                    np.abs(signed[..., 0] - signed[..., 2]).max(),
                )
            )
        return result


def audit(dataset_root: Path, queries_path: Path) -> dict[str, Any]:
    records = load_records(queries_path)
    missing_fields: list[dict[str, Any]] = []
    invalid_query_ids: list[str] = []
    empty_queries: list[str] = []
    cjk_queries: list[str] = []
    typhlosolis_queries: list[str] = []
    missing_files: list[str] = []
    referenced: dict[str, set[Path]] = {
        "visible": set(),
        "infrared": set(),
        "depth": set(),
    }
    groups: set[tuple[Path, Path, Path]] = set()

    for query_id, record in records.items():
        if not QUERY_ID_PATTERN.fullmatch(query_id):
            invalid_query_ids.append(query_id)
        absent = [field for field in REQUIRED_FIELDS if field not in record]
        if absent:
            missing_fields.append({"query_id": query_id, "fields": absent})
            continue

        query = str(record["query"])
        if not query.strip():
            empty_queries.append(query_id)
        if CJK_PATTERN.search(query):
            cjk_queries.append(query_id)
        if "typhlosolis" in query.casefold():
            typhlosolis_queries.append(query_id)

        paths = {
            modality: resolve_path(dataset_root, str(record[modality]))
            for modality in referenced
        }
        for modality, path in paths.items():
            referenced[modality].add(path)
            if not path.is_file():
                missing_files.append(str(path))
        groups.add((paths["visible"], paths["infrared"], paths["depth"]))

    format_counts: dict[str, Counter[str]] = {
        modality: Counter() for modality in referenced
    }
    decode_failures: list[dict[str, str]] = []
    image_anomalies: list[dict[str, Any]] = []
    metadata: dict[Path, dict[str, Any]] = {}

    total_images = sum(len(paths) for paths in referenced.values())
    completed = 0
    for modality, paths in referenced.items():
        for path in sorted(paths):
            completed += 1
            if completed % 500 == 0 or completed == total_images:
                print(f"图像审计进度：{completed}/{total_images}")
            if not path.is_file():
                continue
            try:
                info = inspect_image(path, modality)
                metadata[path] = info
                key = (
                    f"{path.suffix.lower()}|{info['format']}|{info['dtype']}|"
                    f"{info['channels']}ch|{info['width']}x{info['height']}"
                )
                format_counts[modality][key] += 1

                unexpected = False
                if modality == "visible":
                    unexpected = info["dtype"] != "uint8" or info["channels"] != 3
                elif modality == "infrared":
                    unexpected = info["dtype"] != "uint8" or info["channels"] != 3
                elif modality == "depth":
                    expected_png = (
                        info["format"] == "PNG"
                        and info["dtype"] == "uint16"
                        and info["channels"] == 1
                    )
                    expected_jpeg_domain = (
                        info["format"] == "JPEG"
                        and info["dtype"] == "uint8"
                        and info["channels"] == 3
                    )
                    unexpected = not (expected_png or expected_jpeg_domain)
                if unexpected or info["zero_ratio"] == 1.0:
                    image_anomalies.append(
                        {
                            "modality": modality,
                            "path": str(path.relative_to(dataset_root)),
                            **info,
                        }
                    )
            except Exception as error:  # 记录坏文件并继续扫描
                decode_failures.append(
                    {
                        "modality": modality,
                        "path": str(path),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    size_mismatches: list[dict[str, Any]] = []
    for visible, infrared, depth in sorted(groups):
        if not all(path in metadata for path in (visible, infrared, depth)):
            continue
        sizes = {
            "visible": [
                metadata[visible]["width"],
                metadata[visible]["height"],
            ],
            "infrared": [
                metadata[infrared]["width"],
                metadata[infrared]["height"],
            ],
            "depth": [metadata[depth]["width"], metadata[depth]["height"]],
        }
        if len({tuple(size) for size in sizes.values()}) != 1:
            size_mismatches.append(
                {"visible": str(visible.relative_to(dataset_root)), "sizes": sizes}
            )

    return {
        "dataset_root": str(dataset_root.resolve()),
        "queries_path": str(queries_path.resolve()),
        "query_count": len(records),
        "image_group_count": len(groups),
        "unique_image_counts": {
            modality: len(paths) for modality, paths in referenced.items()
        },
        "json": {
            "missing_fields": missing_fields,
            "invalid_query_ids": invalid_query_ids,
            "empty_queries": empty_queries,
            "cjk_query_ids": cjk_queries,
            "typhlosolis_query_ids": typhlosolis_queries,
        },
        "missing_files": sorted(set(missing_files)),
        "image_format_counts": {
            modality: dict(counts) for modality, counts in format_counts.items()
        },
        "decode_failures": decode_failures,
        "image_anomalies": image_anomalies,
        "modality_size_mismatches": size_mismatches,
        "interpretation": {
            "jpeg_depth": "三通道 uint8 JPEG 不能直接解释为毫米深度。",
            "alignment": "尺寸一致不等于像素级对齐。",
            "query_flags": "文本异常列表是自动筛查结果，不是人工标注结论。",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只读审计 2026 AIC 三模态数据集")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit_outputs/dataset_audit.json"),
    )
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    queries_path = (
        args.queries.resolve()
        if args.queries
        else dataset_root / "queries" / "queries.json"
    )
    result = audit(dataset_root, queries_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"审计完成：{args.output.resolve()}")


if __name__ == "__main__":
    main()
