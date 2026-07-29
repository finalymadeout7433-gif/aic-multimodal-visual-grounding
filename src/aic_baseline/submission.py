from __future__ import annotations

import copy
import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bbox import BBoxError, validate_normalized_bbox


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
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname=json_path.name)
    return result
