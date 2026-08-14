from __future__ import annotations

import hashlib
import math
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .bbox import intersection_over_union, validate_normalized_bbox
from .ranker_features import canonical_candidate_label


_TARGET_LEXICON: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            ("security camera", "camera"),
            ("surveillance camera", "camera"),
            ("traffic light", "traffic_object"),
            ("traffic sign", "sign"),
            ("road sign", "sign"),
            ("street lamp", "light"),
            ("road lamp", "light"),
            ("lamp post", "light"),
            ("lamppost", "light"),
            ("light bulb", "light"),
            ("license plate", "small_daily_object"),
            ("fire hydrant", "traffic_object"),
            ("tactile paving", "road_or_path"),
            ("tactile path", "road_or_path"),
            ("guidance path", "road_or_path"),
            ("promotional sign", "sign"),
            ("mountain range", "natural_object"),
            ("drainage hole", "building_part"),
            ("smoke detector", "small_daily_object"),
            ("power strip", "small_daily_object"),
            ("side mirror", "object_part"),
            ("basketball goal", "building_part"),
            ("flower bed", "region"),
            ("glass window", "building_part"),
            ("entrance mat", "small_daily_object"),
            ("fence pole", "building_part"),
            ("water pipe", "building_part"),
            ("dining area", "region"),
            ("parking area", "region"),
            ("outdoor area", "region"),
            ("bubble house", "building_part"),
            ("air quality monitoring box", "small_daily_object"),
            ("computer monitor", "small_daily_object"),
            ("manhole cover", "road_or_path"),
            ("trash can", "small_daily_object"),
            ("feeding trough", "small_daily_object"),
            ("barrier arm", "building_part"),
            ("metal ladder", "building_part"),
            ("air conditioning outdoor unit", "building_part"),
            ("air conditioner outdoor unit", "building_part"),
            ("outdoor air-conditioning unit", "building_part"),
            ("communication tower", "building_part"),
            ("turn signal", "traffic_object"),
            ("wheelchair", "vehicle"),
            ("passage", "region"),
            ("entrance", "building_part"),
            ("awning", "building_part"),
            ("doorway", "building_part"),
            ("facade", "building_part"),
            ("pillar", "building_part"),
            ("building", "building_part"),
            ("bridge", "building_part"),
            ("sidewalk", "road_or_path"),
            ("pavement", "road_or_path"),
            ("walkway", "road_or_path"),
            ("path", "road_or_path"),
            ("road", "road_or_path"),
            ("ground", "region"),
            ("foreground", "region"),
            ("background", "region"),
            ("floor", "region"),
            ("ceiling", "building_part"),
            ("room", "region"),
            ("scene", "region"),
            ("corner", "region"),
            ("area", "region"),
            ("water", "region"),
            ("pond", "region"),
            ("lake", "region"),
            ("river", "region"),
            ("lawn", "region"),
            ("grass", "region"),
            ("sky", "region"),
            ("bank", "region"),
            ("fence", "building_part"),
            ("railing", "building_part"),
            ("pole", "building_part"),
            ("post", "building_part"),
            ("panel", "building_part"),
            ("platform", "building_part"),
            ("door", "building_part"),
            ("window", "building_part"),
            ("roof", "building_part"),
            ("shelter", "building_part"),
            ("gate", "building_part"),
            ("barrier", "building_part"),
            ("column", "building_part"),
            ("ladder", "building_part"),
            ("staircase", "building_part"),
            ("stairs", "building_part"),
            ("enclosure", "building_part"),
            ("screen", "building_part"),
            ("house", "building_part"),
            ("shed", "building_part"),
            ("structure", "building_part"),
            ("wall", "building_part"),
            ("frame", "building_part"),
            ("pavilion", "building_part"),
            ("canopy", "building_part"),
            ("tent", "building_part"),
            ("glass", "building_part"),
            ("camera", "camera"),
            ("drone", "drone"),
            ("aircraft", "drone"),
            ("flying object", "drone"),
            ("lamp", "light"),
            ("light", "light"),
            ("sign", "sign"),
            ("logo", "text_or_logo"),
            ("text", "text_or_logo"),
            ("person", "person"),
            ("people", "person"),
            ("man", "person"),
            ("woman", "person"),
            ("boy", "person"),
            ("girl", "person"),
            ("pedestrian", "person"),
            ("child", "person"),
            ("runner", "person"),
            ("cyclist", "person"),
            ("car", "vehicle"),
            ("bus", "vehicle"),
            ("truck", "vehicle"),
            ("vehicle", "vehicle"),
            ("bicycle", "vehicle"),
            ("bike", "vehicle"),
            ("motorcycle", "vehicle"),
            ("cart", "vehicle"),
            ("trolley", "vehicle"),
            ("skateboard", "vehicle"),
            ("sedan", "vehicle"),
            ("minivan", "vehicle"),
            ("van", "vehicle"),
            ("scooter", "vehicle"),
            ("boat", "vehicle"),
            ("stroller", "vehicle"),
            ("dog", "animal"),
            ("cat", "animal"),
            ("bird", "animal"),
            ("horse", "animal"),
            ("deer", "animal"),
            ("monkey", "animal"),
            ("flamingo", "animal"),
            ("tiger", "animal"),
            ("bear", "animal"),
            ("swan", "animal"),
            ("elephant", "animal"),
            ("lemur", "animal"),
            ("yak", "animal"),
            ("zebra", "animal"),
            ("peacock", "animal"),
            ("pelican", "animal"),
            ("crane", "animal"),
            ("giraffe", "animal"),
            ("llama", "animal"),
            ("animal", "animal"),
            ("pigeon", "animal"),
            ("parrot", "animal"),
            ("macaw", "animal"),
            ("lion", "animal"),
            ("rhinoceros", "animal"),
            ("alpaca", "animal"),
            ("fox", "animal"),
            ("wolf", "animal"),
            ("camel", "animal"),
            ("umbrella", "small_daily_object"),
            ("bottle", "small_daily_object"),
            ("phone", "small_daily_object"),
            ("ball", "small_daily_object"),
            ("sensor", "small_daily_object"),
            ("button", "small_daily_object"),
            ("handle", "small_daily_object"),
            ("bag", "small_daily_object"),
            ("box", "small_daily_object"),
            ("robot", "small_daily_object"),
            ("stool", "furniture"),
            ("bench", "furniture"),
            ("buoy", "small_daily_object"),
            ("cone", "traffic_object"),
            ("statue", "small_daily_object"),
            ("plaque", "sign"),
            ("badge", "small_daily_object"),
            ("vent", "building_part"),
            ("feeder", "small_daily_object"),
            ("trash", "small_daily_object"),
            ("container", "small_daily_object"),
            ("backpack", "small_daily_object"),
            ("toy", "small_daily_object"),
            ("doll", "small_daily_object"),
            ("book", "small_daily_object"),
            ("equipment", "small_daily_object"),
            ("device", "small_daily_object"),
            ("chassis", "small_daily_object"),
            ("trough", "small_daily_object"),
            ("rod", "small_daily_object"),
            ("object", "small_daily_object"),
            ("flag", "small_daily_object"),
            ("hat", "small_daily_object"),
            ("bucket", "small_daily_object"),
            ("wheel", "object_part"),
            ("horn", "object_part"),
            ("head", "object_part"),
            ("tail", "object_part"),
            ("wing", "object_part"),
            ("leg", "object_part"),
            ("hand", "object_part"),
            ("beak", "object_part"),
            ("lens", "object_part"),
            ("headlight", "object_part"),
            ("pants", "object_part"),
            ("shadow", "region"),
            ("puddle", "region"),
            ("reflection", "region"),
            ("rock", "natural_object"),
            ("stone", "natural_object"),
            ("boulder", "natural_object"),
            ("chair", "furniture"),
            ("table", "furniture"),
            ("sofa", "furniture"),
            ("plant", "plant"),
            ("tree", "plant"),
            ("bush", "plant"),
            ("shrub", "plant"),
            ("flower", "plant"),
            ("leaves", "plant"),
            ("leaf", "plant"),
            ("bud", "plant"),
            ("branch", "plant"),
            ("cabinet", "furniture"),
            ("signboard", "sign"),
            ("net", "small_daily_object"),
            ("streetlight", "light"),
            ("curtain", "building_part"),
            ("menu", "sign"),
            ("signpost", "sign"),
            ("letter", "text_or_logo"),
            ("character", "text_or_logo"),
            ("symbol", "text_or_logo"),
            ("pattern", "text_or_logo"),
            ("label", "sign"),
            ("banner", "sign"),
            ("billboard", "sign"),
            ("poster", "sign"),
            ("sticker", "sign"),
            ("heart", "small_daily_object"),
            ("chain", "small_daily_object"),
            ("cable", "small_daily_object"),
            ("wire", "small_daily_object"),
            ("pipe", "building_part"),
            ("fitting", "building_part"),
            ("insulator", "small_daily_object"),
            ("tire", "object_part"),
            ("slide", "building_part"),
            ("stake", "small_daily_object"),
            ("cap", "object_part"),
            ("helmet", "object_part"),
            ("clothes", "object_part"),
            ("jeans", "object_part"),
            ("shoe", "object_part"),
            ("mask", "object_part"),
            ("tricycle", "vehicle"),
            ("excavator", "vehicle"),
            ("kayak", "vehicle"),
            ("roadblock", "traffic_object"),
            ("gantry", "building_part"),
            ("guardrail", "building_part"),
            ("goalpost", "building_part"),
            ("curbstone", "road_or_path"),
            ("brick", "building_part"),
            ("desk", "furniture"),
            ("bed", "furniture"),
            ("pot", "small_daily_object"),
            ("router", "small_daily_object"),
            ("detector", "small_daily_object"),
            ("thermostat", "small_daily_object"),
            ("outlet", "small_daily_object"),
            ("speaker", "small_daily_object"),
            ("lantern", "light"),
            ("cup", "small_daily_object"),
            ("dustpan", "small_daily_object"),
            ("broom", "small_daily_object"),
            ("handbag", "small_daily_object"),
            ("card", "small_daily_object"),
            ("strip", "small_daily_object"),
            ("mat", "small_daily_object"),
            ("line", "region"),
            ("football", "small_daily_object"),
            ("balloon", "small_daily_object"),
            ("basin", "small_daily_object"),
            ("block", "small_daily_object"),
            ("tube", "small_daily_object"),
            ("cuboid", "small_daily_object"),
            ("sculpture", "small_daily_object"),
            ("hedge", "plant"),
            ("topiary", "plant"),
            ("planter", "plant"),
            ("sapling", "plant"),
            ("tulip", "plant"),
            ("bushes", "plant"),
            ("log", "natural_object"),
            ("stump", "natural_object"),
            ("driftwood", "natural_object"),
            ("rockery", "natural_object"),
            ("mountain", "natural_object"),
            ("slope", "region"),
            ("field", "region"),
            ("pool", "region"),
            ("patio", "region"),
            ("plaza", "region"),
            ("court", "region"),
            ("roadside", "road_or_path"),
            ("waterfall", "region"),
            ("burrow", "region"),
            ("adult", "person"),
            ("human", "person"),
            ("tourist", "person"),
            ("passerby", "person"),
            ("volunteer", "person"),
            ("player", "person"),
            ("figure", "person"),
            ("emu", "animal"),
            ("duck", "animal"),
            ("cow", "animal"),
            ("cockatoo", "animal"),
            ("meerkat", "animal"),
            ("donkey", "animal"),
        ),
        key=lambda item: (-len(item[0].split()), -len(item[0]), item[0]),
    )
)

_SMALL_HEADS = {
    "security camera",
    "surveillance camera",
    "camera",
    "traffic light",
    "traffic sign",
    "street lamp",
    "light bulb",
    "lamp",
    "light",
    "drone",
    "aircraft",
    "flying object",
    "sign",
    "promotional sign",
    "license plate",
    "phone",
    "bottle",
    "ball",
    "sensor",
    "button",
    "handle",
    "bird",
    "logo",
}

_RELATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bin front of\b", "front"),
    (r"\bmounted on\b", "mounted_on"),
    (r"\battached to\b", "attached_to"),
    (r"\bnext to\b", "next_to"),
    (r"\bto the left of\b", "left_of"),
    (r"\bto the right of\b", "right_of"),
    (r"\bon the (?:far )?left(?: side)?(?: of)?\b", "absolute_left"),
    (r"\bon the (?:far )?right(?: side)?(?: of)?\b", "absolute_right"),
    (r"\bfrom left to right\b", "ordinal_axis"),
    (r"\bfrom right to left\b", "ordinal_axis"),
    (r"\bfrom top to bottom\b", "ordinal_axis"),
    (r"\bfrom bottom to top\b", "ordinal_axis"),
    (r"\bbehind\b", "behind"),
    (r"\bbetween\b", "between"),
    (r"\bbeside\b", "beside"),
    (r"\babove\b", "above"),
    (r"\bbelow\b", "below"),
    (r"\bunder\b", "under"),
    (r"\bover\b", "over"),
    (r"\binside\b", "inside"),
    (r"\boutside\b", "outside"),
    (r"\bnear\b", "near"),
)

_ACTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bstanding\b", "standing"),
    (r"\bsitting\b", "sitting"),
    (r"\bwalking\b", "walking"),
    (r"\brunning\b", "running"),
    (r"\briding\b", "riding"),
    (r"\bholding\b", "holding"),
    (r"\bwearing\b", "wearing"),
    (r"\bcarrying\b", "carrying"),
    (r"\bflying\b", "flying"),
    (r"\blooking\b", "looking"),
)

_ATTRIBUTES = (
    "red",
    "green",
    "blue",
    "yellow",
    "black",
    "white",
    "silver",
    "golden",
    "gray",
    "grey",
    "wooden",
    "metal",
    "striped",
    "checkered",
    "largest",
    "biggest",
    "smallest",
    "tiniest",
    "large",
    "small",
)

_ORDINALS = (
    "leftmost",
    "rightmost",
    "topmost",
    "bottommost",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
)

_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "both": 2,
    "pair": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_REGION_WORDS = {
    "area",
    "passage",
    "entrance",
    "awning",
    "building",
    "bridge",
    "road",
    "path",
    "sidewalk",
    "pavement",
    "walkway",
    "roof",
    "wall",
    "floor",
    "doorway",
    "facade",
    "pillar",
}

_OCR_WORDS = {
    "text",
    "logo",
    "word",
    "words",
    "letter",
    "letters",
    "number",
    "characters",
    "writing",
    "printed",
}


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", value.lower())


def _strip_leading_determiner(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = re.sub(r"^(?:the|a|an)\s+", "", value, count=1).strip()
    return stripped or None


def _find_target(target_phrase: str) -> tuple[str, str]:
    normalized = _normalized_text(target_phrase)
    connector = re.search(
        r"\b(?:of|with|showing|depicted|covered|containing)\b", normalized
    )
    matches: list[tuple[int, int, str, str]] = []
    for head, superclass in _TARGET_LEXICON:
        match = re.search(rf"(?<!-)\b{re.escape(head)}s?\b", normalized)
        if match:
            matches.append((match.start(), -len(head), head, superclass))
    if matches:
        start, _, head, superclass = min(matches)
        if connector is not None and start > connector.start():
            prefix_tokens = {
                token
                for token in _tokens(normalized[: connector.start()])
                if token
                not in {
                    "the",
                    "a",
                    "an",
                    "one",
                    "two",
                    "three",
                    "four",
                    "group",
                    "row",
                    "pair",
                    "set",
                    "cluster",
                }
                and token not in _ATTRIBUTES
                and token not in _ORDINALS
            }
            if prefix_tokens:
                return "unknown", "unknown"
        return head, superclass
    # Unknown is preferable to inventing a noun from a trailing action or
    # location word. Low-confidence records remain available for review.
    return "unknown", "unknown"


def _split_target_reference(
    normalized: str,
) -> tuple[str, str | None, str | None]:
    matches: list[tuple[int, int, str]] = []
    for pattern, relation in _RELATION_PATTERNS:
        match = re.search(pattern, normalized)
        if match:
            matches.append((match.start(), match.end(), relation))
    for pattern, relation in _ACTION_PATTERNS:
        if relation not in {"holding", "wearing", "carrying"}:
            continue
        match = re.search(pattern, normalized)
        if not match:
            continue
        prefix_tokens = [
            token
            for token in _tokens(normalized[: match.start()])
            if token not in {"a", "an", "the"}
        ]
        if prefix_tokens:
            matches.append((match.start(), match.end(), relation))
    if not matches:
        return normalized, None, None
    start, end, relation = min(matches, key=lambda item: item[0])
    before = normalized[:start].strip(" ,.;:-")
    after = normalized[end:].strip(" ,.;:-")
    return before or normalized, after or None, relation


def parse_query_profile(
    query_id: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a deterministic, multi-label, input-derived Query profile."""

    query_original = str(record.get("query", ""))
    normalized = _normalized_text(query_original)
    target_phrase, trailing_phrase, split_relation = _split_target_reference(normalized)
    target_head, target_superclass = _find_target(target_phrase)

    relations = [
        relation
        for pattern, relation in _RELATION_PATTERNS
        if re.search(pattern, normalized)
    ]
    actions = [
        action for pattern, action in _ACTION_PATTERNS if re.search(pattern, normalized)
    ]
    attributes = [
        attribute
        for attribute in _ATTRIBUTES
        if re.search(rf"\b{re.escape(attribute)}\b", normalized)
    ]
    ordinal = next(
        (
            value
            for value in _ORDINALS
            if re.search(rf"\b{re.escape(value)}\b", normalized)
        ),
        None,
    )
    count = next(
        (
            value
            for token, value in _COUNT_WORDS.items()
            if re.search(rf"\b{re.escape(token)}\b", normalized)
        ),
        None,
    )
    plural_flag = bool(
        (count is not None and count > 1)
        or re.search(r"\b(group|several|multiple)\b", normalized)
    )

    depth_relation: str | None = None
    for pattern, value in (
        (r"\b(farthest|furthest)\b", "farthest"),
        (r"\b(nearest|closest)\b", "nearest"),
        (r"\bin front of\b", "front"),
        (r"\bbehind\b", "behind"),
    ):
        if re.search(pattern, normalized):
            depth_relation = value
            break

    region_or_structure = target_superclass in {
        "region",
        "building_part",
        "road_or_path",
    }
    ocr_or_text = target_superclass == "text_or_logo" or bool(
        set(_tokens(target_phrase)) & _OCR_WORDS
    )

    part_phrase: str | None = None
    reference_phrase: str | None = None
    if split_relation in {"holding", "wearing", "carrying"}:
        part_phrase = _strip_leading_determiner(trailing_phrase)
    elif trailing_phrase:
        reference_phrase = _strip_leading_determiner(trailing_phrase)

    noise_flags: list[str] = []
    if "typhlosolis" in normalized:
        noise_flags.append("suspected_annotation_term")
    if any("CJK" in unicodedata.name(char, "") for char in query_original):
        noise_flags.append("contains_cjk")
    if any(ord(char) < 32 and char not in "\t\r\n" for char in query_original):
        noise_flags.append("control_character")
    if (
        any(ord(char) > 127 for char in query_original)
        and "contains_cjk" not in noise_flags
    ):
        noise_flags.append("non_ascii")

    instance_small_prior = target_head in _SMALL_HEADS or target_superclass in {
        "camera",
        "drone",
        "light",
        "sign",
        "traffic_object",
        "small_daily_object",
        "text_or_logo",
        "object_part",
    }
    if plural_flag:
        expected_extent = "group"
    elif region_or_structure:
        expected_extent = "region"
    elif instance_small_prior:
        expected_extent = "small_instance"
    else:
        expected_extent = "ordinary_or_unknown"

    if noise_flags or target_head == "unknown" or target_superclass == "unknown":
        confidence = "low"
    else:
        confidence = "high"

    visible = str(record.get("visible", ""))
    image_group_id = Path(visible).stem if visible else query_id.split("_")[0]
    return {
        "evidence_kind": "input_derived_fact",
        "query_id": str(query_id),
        "image_group_id": image_group_id,
        "query_original": query_original,
        "query_normalized_for_analysis": normalized,
        "token_count": len(_tokens(query_original)),
        "target_head": target_head,
        "target_superclass": target_superclass,
        "target_phrase": target_phrase,
        "part_phrase": part_phrase,
        "reference_phrase": reference_phrase,
        "attributes": attributes,
        "actions": actions,
        "relations": relations,
        "ordinal": ordinal,
        "count": count,
        "plural_flag": plural_flag,
        "depth_relation": depth_relation,
        "region_or_structure": region_or_structure,
        "ocr_or_text": ocr_or_text,
        "instance_small_prior": instance_small_prior,
        "expected_bbox_extent_proxy": expected_extent,
        "noise_flags": noise_flags,
        "parser_confidence": confidence,
    }


def _bbox_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = validate_normalized_bbox(box)
    return (x2 - x1) * (y2 - y1)


def _safe_iou(first: Sequence[float], second: Sequence[float]) -> float:
    try:
        return float(intersection_over_union(first, second))
    except (TypeError, ValueError):
        return 0.0


def _token_overlap(label: str, phrase: str) -> float:
    left = set(canonical_candidate_label(label).split())
    right = set(canonical_candidate_label(phrase).split())
    return len(left & right) / len(left) if left else 0.0


def _region_candidate_label(label: str) -> bool:
    tokens = set(canonical_candidate_label(label).split())
    return bool(tokens & _REGION_WORDS) or bool(
        tokens
        & {
            "passage",
            "entrance",
            "awning",
            "building",
            "platform",
            "structure",
            "enclosure",
            "foreground",
            "background",
        }
    )


def _agreement_group(boxes: Mapping[str, Sequence[float]], threshold: float) -> str:
    available = {key: value for key, value in boxes.items() if value is not None}
    if not {"s01", "s02", "s03"}.issubset(available):
        return "incomplete"
    s01_s02 = _safe_iou(available["s01"], available["s02"])
    s01_s03 = _safe_iou(available["s01"], available["s03"])
    s02_s03 = _safe_iou(available["s02"], available["s03"])
    if min(s01_s02, s01_s03, s02_s03) >= threshold:
        return "all_agree"
    if s01_s02 >= threshold and max(s01_s03, s02_s03) < threshold:
        return "florence_gdino_agree_s03_differs"
    if s01_s03 >= threshold and s02_s03 < threshold:
        return "florence_s03_agree_gdino_differs"
    if s02_s03 >= threshold and s01_s02 < threshold:
        return "gdino_s03_agree_florence_differs"
    return "all_disagree"


def build_model_behavior_record(
    *,
    query_id: str,
    query_profile: Mapping[str, Any],
    boxes: Mapping[str, Sequence[float]],
    candidates: Sequence[Mapping[str, Any]],
    s03_selected_index: int,
    s04_selected_index: int,
) -> dict[str, Any]:
    """Profile existing model decisions without asserting correctness."""

    normalized_boxes = {
        key: validate_normalized_bbox(value) for key, value in boxes.items()
    }
    pairwise: dict[str, float] = {}
    names = sorted(normalized_boxes)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            pairwise[f"{left}_{right}"] = _safe_iou(
                normalized_boxes[left], normalized_boxes[right]
            )
    areas = {key: _bbox_area(value) for key, value in normalized_boxes.items()}
    s02_area = areas.get("s02", 0.0)

    def selected_label(index: int) -> str:
        if 0 <= index < len(candidates):
            return str(candidates[index].get("label", ""))
        return ""

    labels = [str(candidate.get("label", "")) for candidate in candidates]
    candidate_count = len(candidates)
    top1_label = canonical_candidate_label(labels[0]) if labels else ""
    target_head = str(query_profile.get("target_head", ""))
    target_term = "" if target_head == "unknown" else target_head
    reference_phrase = str(query_profile.get("reference_phrase") or "")
    part_phrase = str(query_profile.get("part_phrase") or "")
    candidate_areas = [
        _bbox_area(candidate["bbox"])
        for candidate in candidates
        if isinstance(candidate.get("bbox"), Sequence)
    ]
    result = {
        "evidence_kind": "model_derived_proxy",
        "query_id": str(query_id),
        "boxes": normalized_boxes,
        "areas": areas,
        "pairwise_iou": pairwise,
        "agreement_at_03": _agreement_group(normalized_boxes, 0.3),
        "agreement_at_05": _agreement_group(normalized_boxes, 0.5),
        "agreement_at_07": _agreement_group(normalized_boxes, 0.7),
        "s03_selected_index": int(s03_selected_index),
        "s04_selected_index": int(s04_selected_index),
        "s03_selected_label": selected_label(s03_selected_index),
        "s04_selected_label": selected_label(s04_selected_index),
        "s03_area_ratio_vs_s02": (
            areas.get("s03", 0.0) / s02_area if s02_area > 0 else None
        ),
        "s04_area_ratio_vs_s02": (
            areas.get("s04", 0.0) / s02_area if s02_area > 0 else None
        ),
        "topk_candidate_count": candidate_count,
        "topk_same_label_count": sum(
            canonical_candidate_label(label) == top1_label for label in labels
        ),
        "topk_target_compatible_count": sum(
            bool(target_term) and _token_overlap(label, target_term) > 0
            for label in labels
        ),
        "topk_reference_compatible_count": sum(
            bool(reference_phrase) and _token_overlap(label, reference_phrase) > 0
            for label in labels
        ),
        "topk_part_compatible_count": sum(
            bool(part_phrase) and _token_overlap(label, part_phrase) > 0
            for label in labels
        ),
        "topk_region_candidate_count": sum(
            _region_candidate_label(label) for label in labels
        ),
        "topk_target_compatible_areas": [
            _bbox_area(candidate["bbox"])
            for candidate, label in zip(candidates, labels)
            if target_term and _token_overlap(label, target_term) > 0
        ],
        "candidate_area_median": (
            statistics.median(candidate_areas) if candidate_areas else None
        ),
        "s03_label_changed_from_top1": (
            canonical_candidate_label(selected_label(s03_selected_index)) != top1_label
        ),
        "s04_label_changed_from_top1": (
            canonical_candidate_label(selected_label(s04_selected_index)) != top1_label
        ),
        "s03_small_to_large_proxy": bool(
            s02_area < 0.01 and areas.get("s03", 0.0) >= 0.05
        ),
        "s04_small_to_large_proxy": bool(
            s02_area < 0.01 and areas.get("s04", 0.0) >= 0.05
        ),
    }
    for field in (
        "topk_same_label_count",
        "topk_target_compatible_count",
        "topk_reference_compatible_count",
        "topk_part_compatible_count",
        "topk_region_candidate_count",
    ):
        record_key = field.replace("_count", "_ratio")
        result[record_key] = result[field] / candidate_count if candidate_count else 0.0
    return result


def classify_small_target_proxy(
    *,
    query_profile: Mapping[str, Any],
    model_areas: Mapping[str, float | None],
    model_ious: Mapping[str, float | None],
) -> dict[str, Any]:
    valid_areas = {
        key: float(value)
        for key, value in model_areas.items()
        if value is not None and math.isfinite(float(value))
    }
    small_signals = sorted(key for key, value in valid_areas.items() if value < 0.01)
    large_signals = sorted(key for key, value in valid_areas.items() if value >= 0.05)
    position_consistency = any(
        value is not None and math.isfinite(float(value)) and float(value) >= 0.3
        for value in model_ious.values()
    )
    lexical = bool(query_profile.get("instance_small_prior"))
    if lexical and len(small_signals) >= 2 and position_consistency:
        label = "high_confidence_small"
    elif (lexical and small_signals) or (
        len(small_signals) >= 2 and position_consistency
    ):
        label = "medium_confidence_small"
    elif not lexical and len(large_signals) >= 2 and position_consistency:
        label = "not_small_proxy"
    else:
        label = "uncertain"
    return {
        "evidence_kind": "heuristic_proxy",
        "small_target_proxy": label,
        "instance_small_prior": lexical,
        "small_model_signals": small_signals,
        "small_model_signal_count": len(small_signals),
        "large_model_signals": large_signals,
        "position_consistency": position_consistency,
        "model_areas": valid_areas,
        "model_ious": {
            key: value for key, value in model_ious.items() if value is not None
        },
    }


def _stable_key(seed: int, *parts: object) -> str:
    raw = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def select_scale_probe(
    records: Sequence[Mapping[str, Any]],
    *,
    per_group: int,
    seed: int,
) -> list[dict[str, Any]]:
    if per_group <= 0:
        raise ValueError("per_group must be positive")
    group_order = (
        "high_confidence_small",
        "medium_confidence_small",
        "region_group",
        "ordinary_control",
    )
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    for group in group_order:
        pool = sorted(
            (dict(record) for record in records if record.get("probe_group") == group),
            key=lambda row: _stable_key(seed, group, row.get("query_id", "")),
        )
        taken = 0
        for row in pool:
            image_group_id = str(row["image_group_id"])
            if image_group_id in used_images:
                continue
            selected.append(row)
            used_images.add(image_group_id)
            taken += 1
            if taken == per_group:
                break
    return selected


def _entropy(gray: np.ndarray) -> float:
    values = np.clip(gray, 0, 255).astype(np.uint8)
    counts = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _gray(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array.astype(np.float64)
    return (
        0.299 * array[..., 0].astype(np.float64)
        + 0.587 * array[..., 1].astype(np.float64)
        + 0.114 * array[..., 2].astype(np.float64)
    )


def _edge_density(gray: np.ndarray) -> float:
    if min(gray.shape) < 2:
        return 0.0
    normalized = gray / max(float(np.max(gray)), 1.0)
    dx = np.abs(np.diff(normalized, axis=1))[:, :-1]
    dy = np.abs(np.diff(normalized, axis=0))[:-1, :]
    rows = min(dx.shape[0], dy.shape[0])
    cols = min(dx.shape[1], dy.shape[1])
    magnitude = np.hypot(dx[:rows, :cols], dy[:rows, :cols])
    return float(np.mean(magnitude > 0.1)) if magnitude.size else 0.0


def _blur_score(gray: np.ndarray) -> float:
    if min(gray.shape) < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    laplacian = (
        -4 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(laplacian))


def _load_image(path: Path) -> tuple[np.ndarray, str, str]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        image.load()
        return np.asarray(image), str(source.format or "UNKNOWN"), str(image.mode)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if not values.size:
        return {}
    p01, p05, p50, p95, p99 = np.percentile(values, [1, 5, 50, 95, 99])
    return {
        "p01": float(p01),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
    }


def _sample_for_metrics(
    array: np.ndarray, *, max_side: int = 512
) -> tuple[np.ndarray, int]:
    height, width = array.shape[:2]
    stride = max(1, math.ceil(max(height, width) / max_side))
    return array[::stride, ::stride], stride


def profile_image_group(
    *,
    image_group_id: str,
    visible_path: Path | str,
    infrared_path: Path | str,
    depth_path: Path | str,
) -> dict[str, Any]:
    visible, visible_format, visible_mode = _load_image(Path(visible_path))
    infrared, infrared_format, infrared_mode = _load_image(Path(infrared_path))
    depth, depth_format, depth_mode = _load_image(Path(depth_path))

    visible_sample, visible_stride = _sample_for_metrics(visible)
    infrared_sample, infrared_stride = _sample_for_metrics(infrared)
    depth_sample, depth_stride = _sample_for_metrics(depth)

    visible_gray = _gray(visible_sample)
    visible_scale = (
        255.0 if visible.dtype == np.uint8 else max(float(visible_sample.max()), 1.0)
    )
    visible_unit = visible_gray / visible_scale
    visible_entropy = _entropy(visible_gray)
    visible_edge_density = _edge_density(visible_gray)
    visible_contrast = float(np.std(visible_unit))
    if visible_sample.ndim == 3 and visible_sample.shape[2] >= 3:
        channels = visible_sample[..., :3].astype(np.float64)
        saturation = np.mean(
            (channels.max(axis=2) - channels.min(axis=2))
            / np.maximum(channels.max(axis=2), 1.0)
        )
    else:
        saturation = 0.0

    infrared_gray = _gray(infrared_sample)
    infrared_percentiles = _percentiles(infrared_gray)
    if infrared_sample.ndim == 3 and infrared_sample.shape[2] >= 3:
        signed = infrared_sample[..., :3].astype(np.int32)
        max_channel_difference = int(
            max(
                np.abs(signed[..., 0] - signed[..., 1]).max(),
                np.abs(signed[..., 0] - signed[..., 2]).max(),
            )
        )
    else:
        max_channel_difference = 0

    result: dict[str, Any] = {
        "evidence_kind": "input_derived_sampled_statistic",
        "statistics_scope": "deterministic_grid_sample_max_side_512",
        "visible_metric_stride": visible_stride,
        "infrared_metric_stride": infrared_stride,
        "depth_metric_stride": depth_stride,
        "image_group_id": str(image_group_id),
        "visible_format": visible_format,
        "visible_mode": visible_mode,
        "visible_dtype": str(visible.dtype),
        "width": int(visible.shape[1]),
        "height": int(visible.shape[0]),
        "visible_brightness": float(np.mean(visible_unit)),
        "visible_contrast": visible_contrast,
        "visible_saturation": float(saturation),
        "visible_entropy": visible_entropy,
        "visible_blur_score": _blur_score(visible_gray),
        "visible_edge_density": visible_edge_density,
        "visible_scene_complexity_proxy": float(
            min(
                1.0,
                (
                    visible_entropy / 8.0
                    + min(visible_edge_density * 4.0, 1.0)
                    + min(visible_contrast * 4.0, 1.0)
                )
                / 3.0,
            )
        ),
        "visible_low_light_score": float(np.mean(visible_unit < 0.15)),
        "visible_overexposure_ratio": float(np.mean(visible_unit > 0.98)),
        "infrared_format": infrared_format,
        "infrared_mode": infrared_mode,
        "infrared_dtype": str(infrared.dtype),
        "infrared_width": int(infrared.shape[1]),
        "infrared_height": int(infrared.shape[0]),
        "infrared_mean": float(np.mean(infrared_gray)),
        "infrared_median": float(np.median(infrared_gray)),
        "infrared_std": float(np.std(infrared_gray)),
        "infrared_dynamic_range": float(np.ptp(infrared_gray)),
        "infrared_p01": infrared_percentiles.get("p01"),
        "infrared_p05": infrared_percentiles.get("p05"),
        "infrared_p50": infrared_percentiles.get("p50"),
        "infrared_p95": infrared_percentiles.get("p95"),
        "infrared_p99": infrared_percentiles.get("p99"),
        "infrared_entropy": _entropy(infrared_gray),
        "infrared_edge_density": _edge_density(infrared_gray),
        "infrared_thermal_saliency_proxy": float(
            (
                infrared_percentiles.get("p99", 0.0)
                - infrared_percentiles.get("p50", 0.0)
            )
            / max(float(np.ptp(infrared_gray)), 1.0)
        ),
        "infrared_max_channel_difference": max_channel_difference,
        "infrared_low_information": bool(
            np.ptp(infrared_gray) < 10 or np.std(infrared_gray) < 3
        ),
        "depth_format": depth_format,
        "depth_mode": depth_mode,
        "depth_dtype": str(depth.dtype),
        "depth_width": int(depth.shape[1]),
        "depth_height": int(depth.shape[0]),
        "modalities_same_dimensions": bool(
            visible.shape[:2] == infrared.shape[:2] == depth.shape[:2]
        ),
        "depth_valid_ratio": None,
        "depth_zero_ratio": float(np.mean(depth_sample == 0)),
        "depth_lt_300_ratio": None,
        "depth_median_mm": None,
        "depth_p05_mm": None,
        "depth_p01_mm": None,
        "depth_p95_mm": None,
        "depth_p99_mm": None,
        "depth_edge_density": None,
        "depth_relative_variation": None,
        "depth_relative_gray_median": None,
        "depth_relative_gray_p05": None,
        "depth_relative_gray_p95": None,
        "depth_low_information": None,
    }
    if depth_format == "PNG" and depth.dtype == np.uint16 and depth.ndim == 2:
        valid = depth_sample > 0
        valid_values = depth_sample[valid]
        result["depth_encoding"] = "png_uint16_mm"
        result["depth_valid_ratio"] = float(np.mean(valid))
        result["depth_lt_300_ratio"] = float(
            np.mean((depth_sample > 0) & (depth_sample < 300))
        )
        if valid_values.size:
            depth_percentiles = _percentiles(valid_values)
            result["depth_median_mm"] = depth_percentiles["p50"]
            result["depth_p01_mm"] = depth_percentiles["p01"]
            result["depth_p05_mm"] = depth_percentiles["p05"]
            result["depth_p95_mm"] = depth_percentiles["p95"]
            result["depth_p99_mm"] = depth_percentiles["p99"]
            result["depth_relative_variation"] = float(
                np.std(valid_values) / max(float(np.median(valid_values)), 1.0)
            )
        result["depth_edge_density"] = _edge_density(depth_sample.astype(np.float64))
        result["depth_low_information"] = bool(
            valid_values.size == 0 or np.std(valid_values) < 1
        )
    elif depth_format == "JPEG" and depth.dtype == np.uint8:
        gray = _gray(depth_sample)
        gray_percentiles = _percentiles(gray)
        result["depth_encoding"] = "jpeg_uint8_unknown"
        result["depth_relative_gray_median"] = gray_percentiles.get("p50")
        result["depth_relative_gray_p05"] = gray_percentiles.get("p05")
        result["depth_relative_gray_p95"] = gray_percentiles.get("p95")
        result["depth_edge_density"] = _edge_density(gray)
        result["depth_low_information"] = bool(np.ptp(gray) < 5 or np.std(gray) < 2)
    else:
        result["depth_encoding"] = "unknown_unexpected"
    return result


def _summarize_query_profiles(profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(profiles)
    token_counts = [int(profile.get("token_count", 0)) for profile in profiles]
    targets = Counter(
        str(profile.get("target_superclass", "unknown")) for profile in profiles
    )
    vocabulary = {
        token
        for profile in profiles
        for token in _tokens(str(profile.get("query_original", "")))
    }
    return {
        "records": total,
        "query_token_mean": statistics.fmean(token_counts) if token_counts else 0.0,
        "query_token_median": statistics.median(token_counts) if token_counts else 0.0,
        "target_superclass_counts": dict(sorted(targets.items())),
        "vocabulary_size": len(vocabulary),
        "spatial_ratio": (
            sum(bool(profile.get("relations")) for profile in profiles) / total
            if total
            else 0.0
        ),
        "ordinal_ratio": (
            sum(profile.get("ordinal") is not None for profile in profiles) / total
            if total
            else 0.0
        ),
        "plural_ratio": (
            sum(bool(profile.get("plural_flag")) for profile in profiles) / total
            if total
            else 0.0
        ),
        "region_ratio": (
            sum(bool(profile.get("region_or_structure")) for profile in profiles)
            / total
            if total
            else 0.0
        ),
        "small_lexical_prior_ratio": (
            sum(bool(profile.get("instance_small_prior")) for profile in profiles)
            / total
            if total
            else 0.0
        ),
    }


def summarize_external_domains(
    *,
    aic_query_profiles: Sequence[Mapping[str, Any]],
    aic_small_proxies: Sequence[Mapping[str, Any]],
    external_records: Mapping[str, Sequence[Mapping[str, Any]] | None],
) -> dict[str, Any]:
    aic_summary = _summarize_query_profiles(aic_query_profiles)
    aic_summary.update(
        {
            "status": "available",
            "area_evidence": "heuristic_proxy",
            "small_target_proxy_counts": dict(
                sorted(
                    Counter(
                        str(record.get("small_target_proxy", "uncertain"))
                        for record in aic_small_proxies
                    ).items()
                )
            ),
        }
    )
    result: dict[str, Any] = {"aic": aic_summary}
    external_vocabulary: set[str] = set()
    external_target_heads: set[str] = set()
    for dataset, records in external_records.items():
        if records is None:
            result[dataset] = {
                "status": "missing_not_verified",
                "area_evidence": None,
            }
            continue
        profiles = [
            parse_query_profile(
                str(record.get("query_id", f"{dataset}:{index}")),
                {
                    "query": str(record.get("query", "")),
                    "visible": str(record.get("image_relpath", index)),
                },
            )
            for index, record in enumerate(records)
        ]
        external_vocabulary.update(
            token
            for profile in profiles
            for token in _tokens(str(profile.get("query_original", "")))
        )
        external_target_heads.update(
            str(profile.get("target_head", "unknown")) for profile in profiles
        )
        areas = [
            _bbox_area(record["bbox_xyxy_normalized"])
            for record in records
            if record.get("bbox_xyxy_normalized") is not None
        ]
        summary = _summarize_query_profiles(profiles)
        summary.update(
            {
                "status": "available",
                "area_evidence": "ground_truth",
                "gt_area_median": statistics.median(areas) if areas else None,
                "gt_area_lt_001_ratio": (
                    sum(area < 0.001 for area in areas) / len(areas) if areas else None
                ),
                "gt_area_lt_01_ratio": (
                    sum(area < 0.01 for area in areas) / len(areas) if areas else None
                ),
                "image_width_median": (
                    statistics.median(
                        int(record["width"])
                        for record in records
                        if record.get("width") is not None
                    )
                    if any(record.get("width") is not None for record in records)
                    else None
                ),
                "image_height_median": (
                    statistics.median(
                        int(record["height"])
                        for record in records
                        if record.get("height") is not None
                    )
                    if any(record.get("height") is not None for record in records)
                    else None
                ),
                "plural_ground_truth_ratio": (
                    sum(int(record.get("target_count", 1)) > 1 for record in records)
                    / len(records)
                    if records
                    else 0.0
                ),
            }
        )
        result[dataset] = summary
    aic_tokens = [
        token
        for profile in aic_query_profiles
        for token in _tokens(str(profile.get("query_original", "")))
    ]
    aic_heads = [
        str(profile.get("target_head", "unknown")) for profile in aic_query_profiles
    ]
    result["domain_shift_summary"] = {
        "evidence_kind": "input_derived_fact",
        "external_available_domains": sorted(
            name for name, records in external_records.items() if records is not None
        ),
        "aic_token_oov_vs_available_external_ratio": (
            sum(token not in external_vocabulary for token in aic_tokens)
            / len(aic_tokens)
            if aic_tokens and external_vocabulary
            else None
        ),
        "aic_unique_target_head_oov_vs_available_external_ratio": (
            sum(head not in external_target_heads for head in set(aic_heads))
            / len(set(aic_heads))
            if aic_heads and external_target_heads
            else None
        ),
        "aic_area_semantics": "heuristic_proxy_only",
        "external_area_semantics": "ground_truth_when_available",
    }
    return result
