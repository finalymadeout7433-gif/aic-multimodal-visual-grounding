from __future__ import annotations

import copy
import json
import shutil
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .aic_full_detection import sha256_file
from .bbox import validate_normalized_bbox


def _write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _verify_package(
    *,
    path: Path,
    original_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        entries = archive.namelist()
        if entries != ["predictions_submission.json"]:
            raise ValueError(f"invalid ZIP entries for {path}: {entries}")
        payload = json.loads(archive.read(entries[0]).decode("utf-8"))
    if set(payload) != set(original_records):
        raise ValueError(f"query ID mismatch in {path}")
    for query_id, source in original_records.items():
        submitted = copy.deepcopy(dict(payload[query_id]))
        bbox = submitted.pop("bbox", None)
        if submitted != dict(source):
            raise ValueError(f"non-bbox field changed for {query_id} in {path}")
        validate_normalized_bbox(bbox)
    return {
        "query_count": len(payload),
        "zip_entries": entries,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_platform_release(
    *,
    original_records: Mapping[str, Mapping[str, Any]],
    packages: Mapping[str, Path | str],
    output_dir: Path | str,
) -> dict[str, Any]:
    if len(packages) != 3:
        raise ValueError("exactly three model packages are required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_packages: dict[str, Any] = {}
    checksum_lines: list[str] = []
    for output_name, source_value in packages.items():
        if Path(output_name).name != output_name or not output_name.endswith(".zip"):
            raise ValueError(f"unsafe output package name: {output_name}")
        source = Path(source_value)
        audit = _verify_package(path=source, original_records=original_records)
        target = output / output_name
        shutil.copyfile(source, target)
        if sha256_file(target) != audit["sha256"]:
            raise RuntimeError(f"copied package hash mismatch: {output_name}")
        manifest_packages[output_name] = {
            **audit,
            "source": str(source.resolve()),
            "output": str(target.resolve()),
        }
        checksum_lines.append(f"{audit['sha256']}  {output_name}")
    manifest = {
        "package_count": len(manifest_packages),
        "query_count_per_package": len(original_records),
        "packages": manifest_packages,
    }
    _write_text_lf(
        output / "release_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text_lf(
        output / "SHA256SUMS.txt",
        "\n".join(checksum_lines) + "\n",
    )
    guide_lines = [
        "# AIC 三模型零训练提交上传说明",
        "",
        (
            f"这三份 ZIP 均已完成 {len(original_records):,} 条 Query 的结构和 "
            "bbox 合法性审计，可直接手动上传。"
        ),
        "",
        "| 建议顺序 | 提交文件 | SHA-256 |",
        "|---:|---|---|",
    ]
    for index, (name, audit) in enumerate(manifest_packages.items(), start=1):
        guide_lines.append(f"| {index} | `{name}` | `{audit['sha256']}` |")
    guide_lines.extend(
        [
            "",
            "不要解压后重新压缩；直接上传本目录中的 ZIP。",
            "上传后记录平台提交 ID、提交时间和 ACC@0.5。",
            "平台分数返回前，不能依据预测框外观判断哪个模型更准确。",
            "",
        ]
    )
    _write_text_lf(output / "UPLOAD_GUIDE.md", "\n".join(guide_lines))
    return manifest
