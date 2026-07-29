from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .bbox import validate_normalized_bbox


class DatasetError(ValueError):
    """数据集结构、字段或路径不合法。"""


@dataclass(frozen=True)
class AICRecord:
    query_id: str
    query: str
    visible_path: Path
    infrared_path: Path
    depth_path: Path
    bbox: list[float] | None
    source: dict[str, Any]


class AICDataset:
    """只读访问 AIC queries JSON 及其三模态文件。"""

    REQUIRED_FIELDS = ("visible", "infrared", "depth", "query")

    def __init__(self, *, dataset_root: Path | str, queries_path: Path | str):
        self.dataset_root = Path(dataset_root).resolve()
        self.queries_path = Path(queries_path).resolve()
        with self.queries_path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise DatasetError("queries JSON 顶层必须是对象")

        self.raw_records: dict[str, dict[str, Any]] = payload
        self._records = [
            self._parse_record(query_id, source)
            for query_id, source in self.raw_records.items()
        ]

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> AICRecord:
        return self._records[index]

    def _resolve_relative_path(self, value: Any, *, field: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise DatasetError(f"{field} 必须是非空字符串")
        relative = Path(value)
        if relative.is_absolute():
            raise DatasetError(f"{field} 必须是数据集根目录下的相对路径")
        resolved = (self.dataset_root / relative).resolve()
        if not resolved.is_relative_to(self.dataset_root):
            raise DatasetError(f"{field} 路径越出了数据集根目录")
        return resolved

    def _parse_record(self, query_id: Any, source: Any) -> AICRecord:
        if not isinstance(query_id, str) or not query_id:
            raise DatasetError("Query ID 必须是非空字符串")
        if not isinstance(source, dict):
            raise DatasetError(f"{query_id} 的记录必须是对象")
        missing = [field for field in self.REQUIRED_FIELDS if field not in source]
        if missing:
            raise DatasetError(f"{query_id} 缺少字段: {', '.join(missing)}")
        query = source["query"]
        if not isinstance(query, str) or not query.strip():
            raise DatasetError(f"{query_id} 的 query 必须是非空字符串")
        bbox = source.get("bbox")
        if bbox is not None:
            bbox = validate_normalized_bbox(bbox)
        return AICRecord(
            query_id=query_id,
            query=query,
            visible_path=self._resolve_relative_path(
                source["visible"], field="visible"
            ),
            infrared_path=self._resolve_relative_path(
                source["infrared"], field="infrared"
            ),
            depth_path=self._resolve_relative_path(source["depth"], field="depth"),
            bbox=bbox,
            source=source,
        )

    def load_visible(self, record: AICRecord) -> Image.Image:
        """读取 Visible，应用 EXIF 方向并统一为 RGB，不修改源文件。"""

        with Image.open(record.visible_path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
