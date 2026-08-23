from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import torch

from .data import RGBTRecord


_ENTITIES = (
    "person",
    "man",
    "woman",
    "child",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "animal",
    "dog",
    "cat",
    "phone",
    "camera",
    "sign",
    "light",
    "lamp",
    "umbrella",
    "bag",
    "ball",
    "boat",
)
_REFERENCE_RE = re.compile(
    r"\b(?:beside|next to|near|left of|right of|above|below|behind|in front of|"
    r"inside|outside|mounted on|attached to|between)\s+(?:the |a |an )?([a-z][a-z -]{1,40})",
    re.IGNORECASE,
)
_PART_RE = re.compile(
    r"\b(?:holding|carrying|wearing|with)\s+(?:the |a |an )?([a-z][a-z -]{1,30})",
    re.IGNORECASE,
)


def _entity_head(text: str) -> str:
    lowered = text.lower()
    matches = [
        (lowered.find(entity), entity)
        for entity in _ENTITIES
        if re.search(rf"\b{re.escape(entity)}\b", lowered)
    ]
    return min(matches)[1] if matches else "unknown"


def _captured_entity(match: re.Match[str] | None) -> str | None:
    if match is None:
        return None
    return _entity_head(match.group(1))


@dataclass(frozen=True)
class QueryRoles:
    target: str
    reference: str | None
    part: str | None


@dataclass(frozen=True)
class QueryCandidate:
    candidate_id: str
    anchor_record_id: str
    source_record_id: str
    kind: str
    bbox_xyxy_normalized: tuple[float, float, float, float]
    target_head: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox_xyxy_normalized"] = list(self.bbox_xyxy_normalized)
        return payload


@dataclass(frozen=True)
class QuerySmokeResult:
    status: str
    record_count: int
    candidate_count: int
    query_dimension: int
    roi_dimension: int
    role_audit: tuple[dict[str, Any], ...]
    candidate_audit: tuple[dict[str, Any], ...]
    checks: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = min(max(x1, 0.0), 1.0 - 1e-4)
    y1 = min(max(y1, 0.0), 1.0 - 1e-4)
    x2 = min(max(x2, x1 + 1e-4), 1.0)
    y2 = min(max(y2, y1 + 1e-4), 1.0)
    return x1, y1, x2, y2


def _geometry_boxes(box: Sequence[float]) -> tuple[tuple[float, float, float, float], ...]:
    x1, y1, x2, y2 = (float(value) for value in box)
    width = x2 - x1
    height = y2 - y1
    variants = (
        (x1 + 0.35 * width, y1, x2 + 0.35 * width, y2),
        (x1, y1 + 0.35 * height, x2, y2 + 0.35 * height),
        (x1 + 0.15 * width, y1 + 0.15 * height, x2 - 0.15 * width, y2 - 0.15 * height),
        (x1 - 0.20 * width, y1 - 0.20 * height, x2 + 0.20 * width, y2 + 0.20 * height),
    )
    unique = []
    for candidate in variants:
        normalized = _clamp_box(candidate)
        if normalized not in unique and normalized != _clamp_box(box):
            unique.append(normalized)
    return tuple(unique)


def _stable_order(records: Sequence[RGBTRecord], *, seed: int) -> list[RGBTRecord]:
    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{seed}:{record.image_pair_key}:{record.record_id}".encode("utf-8")
        ).hexdigest(),
    )


class QueryInterfaceSmoke:
    """Validate the Query/ROI seam without training or changing model state."""

    def __init__(
        self,
        *,
        query_encoder: Callable[[str], torch.Tensor],
        roi_encoder: Callable[[RGBTRecord, QueryCandidate], torch.Tensor],
        maximum_records: int = 128,
        seed: int = 20260812,
    ) -> None:
        if maximum_records < 1:
            raise ValueError("maximum_records must be positive")
        self.query_encoder = query_encoder
        self.roi_encoder = roi_encoder
        self.maximum_records = int(maximum_records)
        self.seed = int(seed)

    @staticmethod
    def parse_roles(query: str) -> QueryRoles:
        return QueryRoles(
            target=_entity_head(query),
            reference=_captured_entity(_REFERENCE_RE.search(query)),
            part=_captured_entity(_PART_RE.search(query)),
        )

    def _candidates(
        self,
        anchor: RGBTRecord,
        roles: QueryRoles,
        pool: Sequence[RGBTRecord],
    ) -> tuple[QueryCandidate, ...]:
        rows: list[QueryCandidate] = []

        def add(kind: str, source: RGBTRecord, box: Sequence[float], target: str) -> None:
            rows.append(
                QueryCandidate(
                    candidate_id=f"{anchor.record_id}:{kind}:{len(rows):03d}",
                    anchor_record_id=anchor.record_id,
                    source_record_id=source.record_id,
                    kind=kind,
                    bbox_xyxy_normalized=_clamp_box(box),
                    target_head=target,
                )
            )

        add("positive", anchor, anchor.bbox_xyxy_normalized, roles.target)
        same_pair = sorted(
            (
                record
                for record in pool
                if record.image_pair_key == anchor.image_pair_key
                and record.record_id != anchor.record_id
            ),
            key=lambda record: record.record_id,
        )
        for record in same_pair:
            target = _entity_head(record.query_original)
            add("same_image_object", record, record.bbox_xyxy_normalized, target)
            if roles.reference not in {None, "unknown"} and target == roles.reference:
                add("reference", record, record.bbox_xyxy_normalized, target)
            if roles.part not in {None, "unknown"} and target == roles.part:
                add("part", record, record.bbox_xyxy_normalized, target)
        for box in _geometry_boxes(anchor.bbox_xyxy_normalized):
            add("geometry_near_miss", anchor, box, roles.target)
        return tuple(rows)

    def run(
        self,
        records: Sequence[RGBTRecord],
        *,
        candidate_pool: Sequence[RGBTRecord],
    ) -> QuerySmokeResult:
        if not records:
            raise ValueError("Q0 records are empty")
        anchors = _stable_order(records, seed=self.seed)[: self.maximum_records]
        role_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        query_dimension = 0
        roi_dimension = 0
        with torch.inference_mode():
            for anchor in anchors:
                roles = self.parse_roles(anchor.query_original)
                first = self.query_encoder(anchor.query_original).detach().cpu().float().flatten()
                second = self.query_encoder(anchor.query_original).detach().cpu().float().flatten()
                if first.numel() == 0:
                    raise ValueError(f"empty query embedding: {anchor.record_id}")
                if not torch.isfinite(first).all():
                    raise ValueError(f"non-finite query embedding: {anchor.record_id}")
                if not torch.equal(first, second):
                    raise ValueError(f"non-deterministic query embedding: {anchor.record_id}")
                query_dimension = query_dimension or int(first.numel())
                if int(first.numel()) != query_dimension:
                    raise ValueError("query embedding dimension changed within Q0")
                candidates = self._candidates(anchor, roles, candidate_pool)
                for candidate in candidates:
                    roi = self.roi_encoder(anchor, candidate).detach().cpu().float().flatten()
                    if roi.numel() == 0 or not torch.isfinite(roi).all():
                        raise ValueError(
                            f"invalid ROI embedding: {anchor.record_id}/{candidate.candidate_id}"
                        )
                    roi_dimension = roi_dimension or int(roi.numel())
                    if int(roi.numel()) != roi_dimension:
                        raise ValueError("ROI embedding dimension changed within Q0")
                    candidate_rows.append(candidate.as_dict())
                role_rows.append(
                    {
                        "record_id": anchor.record_id,
                        "target": roles.target,
                        "reference": roles.reference,
                        "part": roles.part,
                    }
                )
        if query_dimension != roi_dimension:
            raise ValueError(
                f"Query/ROI embedding dimension mismatch: {query_dimension} != {roi_dimension}"
            )
        checks = {
            "record_count_positive": len(anchors) > 0,
            "all_queries_deterministic": True,
            "all_embeddings_finite": True,
            "embedding_dimensions_match": query_dimension == roi_dimension,
            "positive_candidate_per_record": sum(
                row["kind"] == "positive" for row in candidate_rows
            )
            == len(anchors),
            "no_cross_pair_same_image_candidate": True,
        }
        return QuerySmokeResult(
            status="PHASE_18_Q0_READY" if all(checks.values()) else "PHASE_18_Q0_BLOCKED",
            record_count=len(anchors),
            candidate_count=len(candidate_rows),
            query_dimension=query_dimension,
            roi_dimension=roi_dimension,
            role_audit=tuple(role_rows),
            candidate_audit=tuple(candidate_rows),
            checks=checks,
        )
