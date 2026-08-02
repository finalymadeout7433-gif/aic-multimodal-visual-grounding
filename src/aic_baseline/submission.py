from __future__ import annotations

import copy
import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bbox import BBoxError, validate_normalized_bbox


_CANONICAL_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _write_deterministic_zip_member(
    archive: zipfile.ZipFile,
    *,
    name: str,
    payload: bytes,
) -> None:
    info = zipfile.ZipInfo(filename=name, date_time=_CANONICAL_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


class SubmissionError(ValueError):
    """提交记录不完整或 bbox 不合法。"""


def build_submission(
    *,
    original_records: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Sequence[float]],
    output_json: Path | str,
    output_zip: Path | str,
) -> dict[str, dict[str, Any]]:
    """复制原记录并只写入 bbox，然后生成只含该 JSON 的 ZIP。"""

    original_ids = set(original_records)
    prediction_ids = set(predictions)
    if original_ids != prediction_ids:
        missing = sorted(original_ids - prediction_ids)
        extra = sorted(prediction_ids - original_ids)
        raise SubmissionError(
            f"原始记录与预测 ID 集合不一致；缺失={missing[:5]}，多余={extra[:5]}"
        )

    result: dict[str, dict[str, Any]] = {}
    try:
        for query_id, source in original_records.items():
            record = copy.deepcopy(dict(source))
            record["bbox"] = validate_normalized_bbox(predictions[query_id])
            result[query_id] = record
    except BBoxError as error:
        raise SubmissionError(f"{query_id} 的预测框无效: {error}") from error

    json_path = Path(output_json)
    zip_path = Path(output_zip)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with zipfile.ZipFile(zip_path, "w") as archive:
        _write_deterministic_zip_member(
            archive,
            name=json_path.name,
            payload=json_path.read_bytes(),
        )
    return result
