from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from aic_baseline.testset_profile import (
    build_model_behavior_record,
    classify_small_target_proxy,
    parse_query_profile,
    profile_image_group,
    select_scale_probe,
    summarize_external_domains,
)


@pytest.mark.parametrize(
    ("query", "field", "expected"),
    [
        (
            "The leftmost security camera mounted on the stone pillar",
            "target_head",
            "security camera",
        ),
        (
            "The leftmost security camera mounted on the stone pillar",
            "target_superclass",
            "camera",
        ),
        (
            "The leftmost security camera mounted on the stone pillar",
            "ordinal",
            "leftmost",
        ),
        (
            "The leftmost security camera mounted on the stone pillar",
            "relations",
            ["mounted_on"],
        ),
        ("Two white umbrellas above the outdoor dining area", "count", 2),
        ("Two white umbrellas above the outdoor dining area", "plural_flag", True),
        (
            "Two white umbrellas above the outdoor dining area",
            "expected_bbox_extent_proxy",
            "group",
        ),
        ("The farthest four-wing drone from the camera", "depth_relation", "farthest"),
        ("The farthest four-wing drone from the camera", "instance_small_prior", True),
        ("A red promotional sign with food imagery", "target_superclass", "sign"),
        ("The passage to the upper level of the building", "region_or_structure", True),
        ("The person wearing a blue shirt", "actions", ["wearing"]),
        ("The person holding a phone", "part_phrase", "phone"),
        ("The bicycle beside the red car", "reference_phrase", "red car"),
        ("The third person from the right", "ordinal", "third"),
        ("The rightmost street lamp", "ordinal", "rightmost"),
        ("The object behind the bus", "depth_relation", "behind"),
        ("The object in front of the bus", "depth_relation", "front"),
        (
            "The yellow tactile path on the sidewalk",
            "target_superclass",
            "road_or_path",
        ),
        ("The black metal awning over the doors", "target_superclass", "building_part"),
        ("The man walking near the wall", "actions", ["walking"]),
        ("The woman carrying a bag", "actions", ["carrying"]),
        ("The bird flying above the trees", "actions", ["flying"]),
        ("The white car below the bridge", "relations", ["below"]),
        ("The chair between the two tables", "relations", ["between"]),
        ("A pair of bicycles", "count", 2),
        ("The group of people", "plural_flag", True),
        ("The logo on the red sign", "ocr_or_text", True),
        ("The smallest bottle", "attributes", ["smallest"]),
        ("The silver light bulb inside the bubble house", "relations", ["inside"]),
        ("A golden menu standing in front of the barrier", "target_head", "menu"),
        ("The nearest lamppost", "target_head", "lamppost"),
        ("The red letter L closest to the camera", "target_head", "letter"),
        ("The dense hedge on the left of the island", "target_head", "hedge"),
        ("A person walking on the upper plaza path", "region_or_structure", False),
        (
            "The security camera directly below the black awning",
            "region_or_structure",
            False,
        ),
        ("A completely unfamiliar gizmotron", "target_head", "unknown"),
        ("A mysterious gizmo with a red sign", "target_head", "unknown"),
        ("The road lamp on the right", "target_head", "road lamp"),
        ("The reflective glass window", "target_head", "glass window"),
        ("The connecting fitting of the blue water pipe", "target_head", "fitting"),
    ],
)
def test_query_profile_is_multilabel(query: str, field: str, expected: object) -> None:
    profile = parse_query_profile(
        "q1", {"query": query, "visible": "Images/visible/1.png"}
    )

    assert profile["query_original"] == query
    actual = profile[field]
    if isinstance(expected, list):
        assert all(value in actual for value in expected)
    else:
        assert actual == expected


def test_query_profile_keeps_noise_as_a_flag_without_rewriting() -> None:
    profile = parse_query_profile(
        "000681_002",
        {"query": "The typhlosolis", "visible": "Images/visible/681.png"},
    )

    assert profile["query_original"] == "The typhlosolis"
    assert "suspected_annotation_term" in profile["noise_flags"]
    assert profile["parser_confidence"] == "low"


def test_small_target_proxy_requires_independent_evidence_for_high() -> None:
    query_profile = parse_query_profile(
        "q1",
        {"query": "the security camera", "visible": "Images/visible/1.png"},
    )
    result = classify_small_target_proxy(
        query_profile=query_profile,
        model_areas={"s01": 0.004, "s02": 0.006, "topk_target_median": 0.008},
        model_ious={"s01_s02": 0.55},
    )

    assert result["small_target_proxy"] == "high_confidence_small"
    assert result["evidence_kind"] == "heuristic_proxy"
    assert result["small_model_signal_count"] == 3


def test_small_target_proxy_does_not_treat_one_small_prediction_as_truth() -> None:
    query_profile = parse_query_profile(
        "q1",
        {"query": "the camera", "visible": "Images/visible/1.png"},
    )
    result = classify_small_target_proxy(
        query_profile=query_profile,
        model_areas={"s01": 0.004, "s02": 0.20},
        model_ious={"s01_s02": 0.0},
    )

    assert result["small_target_proxy"] == "medium_confidence_small"
    assert result["position_consistency"] is False


def test_model_behavior_distinguishes_baseline_agreement_from_ranker_change() -> None:
    record = build_model_behavior_record(
        query_id="q1",
        query_profile={"target_phrase": "person", "target_head": "person"},
        boxes={
            "s01": [0.1, 0.1, 0.3, 0.3],
            "s02": [0.11, 0.1, 0.31, 0.3],
            "s03": [0.6, 0.6, 0.9, 0.9],
            "s04": [0.11, 0.1, 0.31, 0.3],
        },
        candidates=[
            {"bbox": [0.11, 0.1, 0.31, 0.3], "label": "person", "score": 0.9},
            {"bbox": [0.6, 0.6, 0.9, 0.9], "label": "building", "score": 0.2},
        ],
        s03_selected_index=1,
        s04_selected_index=0,
    )

    assert record["agreement_at_05"] == "florence_gdino_agree_s03_differs"
    assert record["s03_area_ratio_vs_s02"] == pytest.approx(2.25)
    assert record["s03_selected_label"] == "building"
    assert record["topk_same_label_count"] == 1
    assert record["topk_same_label_ratio"] == pytest.approx(0.5)
    assert record["topk_target_compatible_ratio"] == pytest.approx(0.5)
    assert record["topk_region_candidate_ratio"] == pytest.approx(0.5)


def test_target_compatibility_does_not_match_reference_attributes() -> None:
    record = build_model_behavior_record(
        query_id="q1",
        query_profile={
            "target_head": "security camera",
            "target_phrase": "the security camera directly below the black awning",
            "reference_phrase": "black awning",
        },
        boxes={
            "s01": [0.1, 0.1, 0.2, 0.2],
            "s02": [0.1, 0.1, 0.2, 0.2],
            "s03": [0.5, 0.5, 0.9, 0.9],
            "s04": [0.1, 0.1, 0.2, 0.2],
        },
        candidates=[
            {"bbox": [0.1, 0.1, 0.2, 0.2], "label": "security camera"},
            {"bbox": [0.5, 0.5, 0.9, 0.9], "label": "black awning"},
        ],
        s03_selected_index=1,
        s04_selected_index=0,
    )

    assert record["topk_target_compatible_count"] == 1
    assert record["topk_reference_compatible_count"] == 1


def test_scale_probe_is_deterministic_image_isolated_and_does_not_backfill() -> None:
    records: list[dict[str, object]] = []
    for group, count in (
        ("high_confidence_small", 3),
        ("medium_confidence_small", 1),
        ("region_group", 3),
        ("ordinary_control", 3),
    ):
        for index in range(count):
            records.append(
                {
                    "query_id": f"{group}:{index}",
                    "image_group_id": f"image:{group}:{index}",
                    "probe_group": group,
                }
            )
    first = select_scale_probe(records, per_group=2, seed=20260801)
    second = select_scale_probe(records, per_group=2, seed=20260801)

    assert first == second
    assert len(first) == 7
    assert len({row["image_group_id"] for row in first}) == 7
    assert sum(row["probe_group"] == "medium_confidence_small" for row in first) == 1


def test_image_group_profile_keeps_png_depth_in_mm(tmp_path: Path) -> None:
    visible = np.full((8, 10, 3), 120, dtype=np.uint8)
    infrared = np.stack(
        [
            np.full((8, 10), 40, dtype=np.uint8),
            np.full((8, 10), 42, dtype=np.uint8),
            np.full((8, 10), 40, dtype=np.uint8),
        ],
        axis=-1,
    )
    depth = np.full((8, 10), 1500, dtype=np.uint16)
    depth[0, 0] = 0
    visible_path = tmp_path / "visible.png"
    infrared_path = tmp_path / "infrared.png"
    depth_path = tmp_path / "depth.png"
    Image.fromarray(visible).save(visible_path)
    Image.fromarray(infrared).save(infrared_path)
    Image.fromarray(depth).save(depth_path)

    result = profile_image_group(
        image_group_id="g1",
        visible_path=visible_path,
        infrared_path=infrared_path,
        depth_path=depth_path,
    )

    assert result["depth_encoding"] == "png_uint16_mm"
    assert result["depth_valid_ratio"] == pytest.approx(0.9875)
    assert result["depth_median_mm"] == 1500.0
    assert result["infrared_max_channel_difference"] == 2


def test_image_group_profile_never_interprets_jpeg_depth_as_mm(tmp_path: Path) -> None:
    rgb = np.full((8, 10, 3), 100, dtype=np.uint8)
    visible_path = tmp_path / "visible.jpg"
    infrared_path = tmp_path / "infrared.jpg"
    depth_path = tmp_path / "depth.jpg"
    Image.fromarray(rgb).save(visible_path)
    Image.fromarray(rgb).save(infrared_path)
    Image.fromarray(rgb).save(depth_path)

    result = profile_image_group(
        image_group_id="g1",
        visible_path=visible_path,
        infrared_path=infrared_path,
        depth_path=depth_path,
    )

    assert result["depth_encoding"] == "jpeg_uint8_unknown"
    assert result["depth_median_mm"] is None
    assert result["depth_relative_gray_median"] == pytest.approx(100.0)


def test_external_domain_summary_separates_gt_from_aic_proxy() -> None:
    result = summarize_external_domains(
        aic_query_profiles=[
            {
                "query_original": "the camera",
                "target_head": "camera",
                "relations": [],
                "ordinal": None,
                "plural_flag": False,
                "region_or_structure": False,
                "instance_small_prior": True,
            }
        ],
        aic_small_proxies=[{"small_target_proxy": "high_confidence_small"}],
        external_records={
            "refcoco": [
                {
                    "query": "the person",
                    "bbox_xyxy_normalized": [0.0, 0.0, 0.5, 0.5],
                }
            ],
            "sorec": None,
        },
    )

    assert result["aic"]["area_evidence"] == "heuristic_proxy"
    assert result["refcoco"]["area_evidence"] == "ground_truth"
    assert result["refcoco"]["gt_area_median"] == 0.25
    assert result["sorec"]["status"] == "missing_not_verified"
