from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .bbox import intersection_over_union


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "with",
}
_RELATION_WORDS = {
    "above",
    "below",
    "beside",
    "between",
    "bottom",
    "bottommost",
    "center",
    "centre",
    "down",
    "inside",
    "left",
    "leftmost",
    "middle",
    "next",
    "outside",
    "over",
    "right",
    "rightmost",
    "top",
    "topmost",
    "under",
    "underneath",
    "up",
    "within",
}
_ORDINAL_WORDS = {
    "first": 0,
    "1st": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "fourth": 3,
    "4th": 3,
    "fifth": 4,
    "5th": 4,
}
_DEPTH_WORDS = {
    "nearest": "nearest",
    "closest": "nearest",
    "farthest": "farthest",
    "furthest": "farthest",
    "frontmost": "front",
    "foremost": "front",
    "rearmost": "behind",
}


@dataclass(frozen=True)
class QuerySemantics:
    target_phrase: str
    reference_phrase: str | None = None
    relative_relation: str | None = None
    absolute_relation: str | None = None
    size_relation: str | None = None
    depth_relation: str | None = None
    ordinal_index: int | None = None
    ordinal_axis: str | None = None
    ordinal_reverse: bool = False


_RELATIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:to the )?left of\b", "left_of"),
    (r"\b(?:to the )?right of\b", "right_of"),
    (r"\bin front of\b", "front"),
    (r"\bbehind\b", "behind"),
    (r"\b(?:above|over)\b", "above"),
    (r"\b(?:below|under|underneath)\b", "below"),
    (r"\b(?:beside|next to)\b", "beside"),
    (r"\binside\b|\bwithin\b", "inside"),
    (r"\boutside\b", "outside"),
    (r"\bbetween\b", "between"),
)


def _tokens(text: str, *, semantic: bool = False) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if semantic:
        tokens = [
            token
            for token in tokens
            if token not in _STOP_WORDS
            and token not in _RELATION_WORDS
            and token not in _ORDINAL_WORDS
            and token not in _DEPTH_WORDS
            and token not in {"largest", "biggest", "smallest", "tiny"}
        ]
    return tokens


def _clean_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" \t\r\n.,;:!?-"))


def parse_query_semantics(query: str) -> QuerySemantics:
    """Parse conservative geometric semantics from an English query.

    The parser intentionally leaves ``front``/``behind`` as depth flags.  It
    never maps those words to image-plane y coordinates.
    """

    normalized = _clean_phrase(query.lower())
    target_phrase = normalized
    reference_phrase: str | None = None
    relative_relation: str | None = None
    depth_relation: str | None = None

    for pattern, relation in _RELATIVE_PATTERNS:
        match = re.search(pattern, normalized)
        if not match:
            continue
        before = _clean_phrase(normalized[: match.start()])
        after = _clean_phrase(normalized[match.end() :])
        if relation in {"front", "behind"}:
            depth_relation = relation
        else:
            relative_relation = relation
        if before:
            target_phrase = before
        if after:
            reference_phrase = after
        break

    for token, relation in _DEPTH_WORDS.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized):
            depth_relation = relation
            break
    if depth_relation is None:
        if re.search(r"\b(?:closest|nearest) to (?:the )?camera\b", normalized):
            depth_relation = "nearest"
        elif re.search(
            r"\b(?:farthest|furthest) from (?:the )?camera\b", normalized
        ):
            depth_relation = "farthest"

    absolute_relation: str | None = None
    absolute_aliases = (
        ("leftmost", "leftmost"),
        ("rightmost", "rightmost"),
        ("topmost", "topmost"),
        ("uppermost", "topmost"),
        ("bottommost", "bottommost"),
        ("lowermost", "bottommost"),
        ("center", "center"),
        ("centre", "center"),
        ("middle", "center"),
    )
    for token, relation in absolute_aliases:
        if re.search(rf"\b{token}\b", normalized):
            absolute_relation = relation
            break

    size_relation: str | None = None
    if re.search(r"\b(?:largest|biggest)\b", normalized):
        size_relation = "largest"
    elif re.search(r"\b(?:smallest|tiniest)\b", normalized):
        size_relation = "smallest"

    ordinal_index: int | None = None
    for token, index in _ORDINAL_WORDS.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized):
            ordinal_index = index
            break
    ordinal_axis: str | None = None
    ordinal_reverse = False
    if ordinal_index is not None:
        if re.search(r"\b(?:right|rightmost)\b", normalized):
            ordinal_axis = "x"
            ordinal_reverse = True
        elif re.search(r"\b(?:left|leftmost)\b", normalized):
            ordinal_axis = "x"
        elif re.search(r"\b(?:bottom|bottommost|below)\b", normalized):
            ordinal_axis = "y"
            ordinal_reverse = True
        elif re.search(r"\b(?:top|topmost|above)\b", normalized):
            ordinal_axis = "y"

    return QuerySemantics(
        target_phrase=target_phrase,
        reference_phrase=reference_phrase,
        relative_relation=relative_relation,
        absolute_relation=absolute_relation,
        size_relation=size_relation,
        depth_relation=depth_relation,
        ordinal_index=ordinal_index,
        ordinal_axis=ordinal_axis,
        ordinal_reverse=ordinal_reverse,
    )


def relevance_from_iou(iou: float) -> int:
    value = float(iou)
    if not math.isfinite(value):
        raise ValueError("IoU must be finite")
    if value < 0.2:
        return 0
    if value < 0.5:
        return 1
    if value < 0.7:
        return 2
    return 3


def canonical_candidate_label(label: str) -> str:
    return " ".join(_tokens(label, semantic=True))


def deduplicate_rank_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.95,
) -> list[dict[str, Any]]:
    """Deduplicate only boxes with the same semantic label."""

    retained: list[dict[str, Any]] = []
    retained_original_indices: list[int] = []
    for original_index, source in enumerate(candidates):
        candidate = dict(source)
        label = canonical_candidate_label(str(candidate.get("label", "")))
        duplicate_index: int | None = None
        for index, existing in enumerate(retained):
            if label != canonical_candidate_label(
                str(existing.get("label", ""))
            ):
                continue
            if intersection_over_union(
                candidate["bbox"], existing["bbox"]
            ) >= float(iou_threshold):
                duplicate_index = index
                break
        if duplicate_index is None:
            retained.append(candidate)
            retained_original_indices.append(original_index)
            continue
        old_score = retained[duplicate_index].get("score")
        new_score = candidate.get("score")
        old_value = float("-inf") if old_score is None else float(old_score)
        new_value = float("-inf") if new_score is None else float(new_score)
        if new_value > old_value:
            retained[duplicate_index] = candidate
            retained_original_indices[duplicate_index] = original_index
    return [
        candidate
        for _, candidate in sorted(
            zip(retained_original_indices, retained),
            key=lambda pair: pair[0],
        )
    ]


def _token_overlap(label: str, phrase: str | None) -> float:
    if not phrase:
        return 0.0
    left = set(_tokens(label, semantic=True))
    right = set(_tokens(phrase, semantic=True))
    if not left or not right:
        return 0.0
    return len(left & right) / len(right)


def _box_geometry(box: Sequence[float]) -> dict[str, float]:
    if len(box) != 4:
        raise ValueError("bbox must have four values")
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("bbox contains a non-finite value")
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox has non-positive width or height")
    width = x2 - x1
    height = y2 - y1
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": (x1 + x2) / 2.0,
        "cy": (y1 + y2) / 2.0,
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_ratio": width / max(height, 1e-9),
    }


def _rank(values: Sequence[float], index: int, *, reverse: bool) -> float:
    order = sorted(
        range(len(values)),
        key=lambda item: values[item],
        reverse=reverse,
    )
    if len(order) <= 1:
        return 0.0
    return order.index(index) / (len(order) - 1)


def _containment(inner: Mapping[str, float], outer: Mapping[str, float]) -> float:
    intersection_width = max(
        0.0, min(inner["x2"], outer["x2"]) - max(inner["x1"], outer["x1"])
    )
    intersection_height = max(
        0.0, min(inner["y2"], outer["y2"]) - max(inner["y1"], outer["y1"])
    )
    return (intersection_width * intersection_height) / max(
        inner["area"], 1e-9
    )


def _relative_score(
    relation: str,
    target: Mapping[str, float],
    references: Sequence[Mapping[str, float]],
) -> tuple[float, float, float, float]:
    if not references:
        return 0.0, 0.0, 0.0, 0.0
    if relation == "between" and len(references) >= 2:
        ordered = sorted(references, key=lambda item: item["cx"])
        left, right = ordered[0], ordered[-1]
        satisfied = float(left["cx"] <= target["cx"] <= right["cx"])
        return (
            satisfied,
            right["cx"] - target["cx"],
            right["cy"] - target["cy"],
            min(
                math.hypot(target["cx"] - left["cx"], target["cy"] - left["cy"]),
                math.hypot(
                    target["cx"] - right["cx"], target["cy"] - right["cy"]
                ),
            ),
        )
    reference = min(
        references,
        key=lambda item: math.hypot(
            target["cx"] - item["cx"], target["cy"] - item["cy"]
        ),
    )
    dx = reference["cx"] - target["cx"]
    dy = reference["cy"] - target["cy"]
    distance = math.hypot(dx, dy)
    if relation == "left_of":
        score = float(dx > 0.0)
    elif relation == "right_of":
        score = float(dx < 0.0)
    elif relation == "above":
        score = float(dy > 0.0)
    elif relation == "below":
        score = float(dy < 0.0)
    elif relation == "beside":
        score = max(0.0, 1.0 - distance / math.sqrt(2.0))
    elif relation == "inside":
        score = _containment(target, reference)
    elif relation == "outside":
        score = 1.0 - _containment(target, reference)
    else:
        score = 0.0
    return score, dx, dy, distance


FEATURE_NAMES: tuple[str, ...] = (
    "gdino_score",
    "original_rank",
    "score_gap_to_first",
    "score_gap_to_next",
    "cx",
    "cy",
    "width",
    "height",
    "area",
    "log_area",
    "aspect_ratio",
    "edge_left",
    "edge_right",
    "edge_top",
    "edge_bottom",
    "rank_left_all",
    "rank_right_all",
    "rank_top_all",
    "rank_bottom_all",
    "rank_area_large_all",
    "rank_area_small_all",
    "rank_left_target",
    "rank_right_target",
    "rank_top_target",
    "rank_bottom_target",
    "rank_area_large_target",
    "rank_area_small_target",
    "candidate_count",
    "target_candidate_count",
    "same_label_count",
    "mean_iou_other",
    "max_iou_other",
    "max_containment_in_other",
    "max_contains_other",
    "query_overlap",
    "target_overlap",
    "reference_overlap",
    "target_compatible",
    "reference_compatible",
    "role_subject_advantage",
    "role_reference_advantage",
    "absolute_relation_satisfaction",
    "size_relation_satisfaction",
    "ordinal_satisfaction",
    "ordinal_distance",
    "reference_available",
    "relative_relation_satisfaction",
    "target_to_reference_dx",
    "target_to_reference_dy",
    "target_to_reference_distance",
    "flag_absolute_relation",
    "flag_relative_relation",
    "flag_size_relation",
    "flag_ordinal_relation",
    "flag_depth_relation",
    "flag_plural_group",
)


def build_feature_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build one finite, fixed-width feature row per candidate."""

    candidates = [dict(candidate) for candidate in record.get("candidates", [])]
    if not candidates:
        return []
    semantics = parse_query_semantics(str(record.get("query", "")))
    geometries = [_box_geometry(candidate["bbox"]) for candidate in candidates]
    scores = [
        0.0 if candidate.get("score") is None else float(candidate["score"])
        for candidate in candidates
    ]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("candidate score contains a non-finite value")
    labels = [str(candidate.get("label", "")) for candidate in candidates]
    query_overlap = [
        _token_overlap(label, str(record.get("query", ""))) for label in labels
    ]
    target_overlap = [
        _token_overlap(label, semantics.target_phrase) for label in labels
    ]
    reference_overlap = [
        _token_overlap(label, semantics.reference_phrase) for label in labels
    ]
    target_indices = [
        index for index, value in enumerate(target_overlap) if value > 0.0
    ]
    if not target_indices:
        target_indices = list(range(len(candidates)))
    reference_indices = [
        index for index, value in enumerate(reference_overlap) if value > 0.0
    ]
    canonical_counts = Counter(canonical_candidate_label(label) for label in labels)
    all_values = {
        "x": [geometry["cx"] for geometry in geometries],
        "y": [geometry["cy"] for geometry in geometries],
        "area": [geometry["area"] for geometry in geometries],
    }
    target_values = {
        key: [values[index] for index in target_indices]
        for key, values in all_values.items()
    }
    is_plural = bool(
        re.search(
            r"\b(?:two|three|four|five|both|pair|group|several|multiple)\b",
            str(record.get("query", "")).lower(),
        )
    )

    rows: list[dict[str, Any]] = []
    for index, (candidate, geometry) in enumerate(
        zip(candidates, geometries)
    ):
        target_local_index = (
            target_indices.index(index) if index in target_indices else None
        )
        pair_ious = [
            intersection_over_union(candidate["bbox"], other["bbox"])
            for other_index, other in enumerate(candidates)
            if other_index != index
        ]
        contains = [
            _containment(other_geometry, geometry)
            for other_index, other_geometry in enumerate(geometries)
            if other_index != index
        ]
        contained_by = [
            _containment(geometry, other_geometry)
            for other_index, other_geometry in enumerate(geometries)
            if other_index != index
        ]

        absolute_score = 0.0
        if target_local_index is not None:
            if semantics.absolute_relation == "leftmost":
                absolute_score = float(
                    target_local_index
                    == min(
                        range(len(target_values["x"])),
                        key=target_values["x"].__getitem__,
                    )
                )
            elif semantics.absolute_relation == "rightmost":
                absolute_score = float(
                    target_local_index
                    == max(
                        range(len(target_values["x"])),
                        key=target_values["x"].__getitem__,
                    )
                )
            elif semantics.absolute_relation == "topmost":
                absolute_score = float(
                    target_local_index
                    == min(
                        range(len(target_values["y"])),
                        key=target_values["y"].__getitem__,
                    )
                )
            elif semantics.absolute_relation == "bottommost":
                absolute_score = float(
                    target_local_index
                    == max(
                        range(len(target_values["y"])),
                        key=target_values["y"].__getitem__,
                    )
                )
            elif semantics.absolute_relation == "center":
                absolute_score = max(
                    0.0,
                    1.0
                    - math.hypot(geometry["cx"] - 0.5, geometry["cy"] - 0.5)
                    / math.sqrt(0.5),
                )

        size_score = 0.0
        if target_local_index is not None:
            if semantics.size_relation == "largest":
                size_score = float(
                    target_local_index
                    == max(
                        range(len(target_values["area"])),
                        key=target_values["area"].__getitem__,
                    )
                )
            elif semantics.size_relation == "smallest":
                size_score = float(
                    target_local_index
                    == min(
                        range(len(target_values["area"])),
                        key=target_values["area"].__getitem__,
                    )
                )

        ordinal_satisfaction = 0.0
        ordinal_distance = 1.0
        if (
            target_local_index is not None
            and semantics.ordinal_index is not None
            and semantics.ordinal_axis is not None
        ):
            axis_values = target_values[
                "x" if semantics.ordinal_axis == "x" else "y"
            ]
            ordered = sorted(
                range(len(axis_values)),
                key=axis_values.__getitem__,
                reverse=semantics.ordinal_reverse,
            )
            ordinal_rank = ordered.index(target_local_index)
            ordinal_distance = abs(ordinal_rank - semantics.ordinal_index) / max(
                len(ordered) - 1, 1
            )
            ordinal_satisfaction = float(
                ordinal_rank == semantics.ordinal_index
            )

        relative_score = dx = dy = reference_distance = 0.0
        if semantics.relative_relation and reference_indices:
            relative_score, dx, dy, reference_distance = _relative_score(
                semantics.relative_relation,
                geometry,
                [geometries[item] for item in reference_indices],
            )

        feature_values: dict[str, float] = {
            "gdino_score": scores[index],
            "original_rank": float(
                candidate.get("original_rank", index)
            )
            / 9.0,
            "score_gap_to_first": scores[0] - scores[index],
            "score_gap_to_next": (
                scores[index] - scores[index + 1]
                if index + 1 < len(scores)
                else 0.0
            ),
            "cx": geometry["cx"],
            "cy": geometry["cy"],
            "width": geometry["width"],
            "height": geometry["height"],
            "area": geometry["area"],
            "log_area": math.log(max(geometry["area"], 1e-9)),
            "aspect_ratio": geometry["aspect_ratio"],
            "edge_left": geometry["x1"],
            "edge_right": 1.0 - geometry["x2"],
            "edge_top": geometry["y1"],
            "edge_bottom": 1.0 - geometry["y2"],
            "rank_left_all": _rank(all_values["x"], index, reverse=False),
            "rank_right_all": _rank(all_values["x"], index, reverse=True),
            "rank_top_all": _rank(all_values["y"], index, reverse=False),
            "rank_bottom_all": _rank(all_values["y"], index, reverse=True),
            "rank_area_large_all": _rank(
                all_values["area"], index, reverse=True
            ),
            "rank_area_small_all": _rank(
                all_values["area"], index, reverse=False
            ),
            "rank_left_target": (
                _rank(target_values["x"], target_local_index, reverse=False)
                if target_local_index is not None
                else 1.0
            ),
            "rank_right_target": (
                _rank(target_values["x"], target_local_index, reverse=True)
                if target_local_index is not None
                else 1.0
            ),
            "rank_top_target": (
                _rank(target_values["y"], target_local_index, reverse=False)
                if target_local_index is not None
                else 1.0
            ),
            "rank_bottom_target": (
                _rank(target_values["y"], target_local_index, reverse=True)
                if target_local_index is not None
                else 1.0
            ),
            "rank_area_large_target": (
                _rank(
                    target_values["area"], target_local_index, reverse=True
                )
                if target_local_index is not None
                else 1.0
            ),
            "rank_area_small_target": (
                _rank(
                    target_values["area"], target_local_index, reverse=False
                )
                if target_local_index is not None
                else 1.0
            ),
            "candidate_count": float(len(candidates)),
            "target_candidate_count": float(len(target_indices)),
            "same_label_count": float(
                canonical_counts[canonical_candidate_label(labels[index])]
            ),
            "mean_iou_other": (
                sum(pair_ious) / len(pair_ious) if pair_ious else 0.0
            ),
            "max_iou_other": max(pair_ious, default=0.0),
            "max_containment_in_other": max(contained_by, default=0.0),
            "max_contains_other": max(contains, default=0.0),
            "query_overlap": query_overlap[index],
            "target_overlap": target_overlap[index],
            "reference_overlap": reference_overlap[index],
            "target_compatible": float(index in target_indices),
            "reference_compatible": float(index in reference_indices),
            "role_subject_advantage": (
                target_overlap[index] - reference_overlap[index]
            ),
            "role_reference_advantage": (
                reference_overlap[index] - target_overlap[index]
            ),
            "absolute_relation_satisfaction": absolute_score,
            "size_relation_satisfaction": size_score,
            "ordinal_satisfaction": ordinal_satisfaction,
            "ordinal_distance": ordinal_distance,
            "reference_available": float(bool(reference_indices)),
            "relative_relation_satisfaction": relative_score,
            "target_to_reference_dx": dx,
            "target_to_reference_dy": dy,
            "target_to_reference_distance": reference_distance,
            "flag_absolute_relation": float(
                semantics.absolute_relation is not None
            ),
            "flag_relative_relation": float(
                semantics.relative_relation is not None
            ),
            "flag_size_relation": float(semantics.size_relation is not None),
            "flag_ordinal_relation": float(
                semantics.ordinal_index is not None
            ),
            "flag_depth_relation": float(
                semantics.depth_relation is not None
            ),
            "flag_plural_group": float(is_plural),
        }
        if set(feature_values) != set(FEATURE_NAMES):
            raise AssertionError("ranker feature schema is incomplete")
        if not all(math.isfinite(value) for value in feature_values.values()):
            raise ValueError("ranker features contain a non-finite value")
        rows.append(
            {
                "query_id": str(record.get("query_id", "")),
                "candidate_index": index,
                "features": feature_values,
                "iou": float(candidate.get("iou", 0.0)),
                "relevance": relevance_from_iou(
                    float(candidate.get("iou", 0.0))
                ),
            }
        )
    return rows
