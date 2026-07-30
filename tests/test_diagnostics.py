from __future__ import annotations

from aic_baseline.diagnostics import (
    DiagnosticCandidate,
    build_core_subset,
    build_tile_probe,
    deduplicate_candidates,
    evaluate_candidates,
    map_tile_bbox_to_normalized,
    tile_windows,
)


def _record(
    dataset: str,
    image_index: int,
    query_index: int,
    *,
    area: float = 0.04,
) -> dict:
    side = area**0.5
    return {
        "dataset": dataset,
        "query_id": f"{dataset}:{image_index}:{query_index}",
        "image_key": f"coco:{image_index}",
        "query": "the object",
        "image_relpath": f"images/{image_index}.jpg",
        "bbox_xyxy_normalized": [0.0, 0.0, side, side],
    }


def test_core_subset_is_deterministic_and_image_unique() -> None:
    records = []
    for dataset_offset, dataset in enumerate(
        ("refcoco", "refcoco_plus", "refcocog")
    ):
        for image_index in range(dataset_offset * 20, dataset_offset * 20 + 12):
            records.append(_record(dataset, image_index, 1))
            records.append(_record(dataset, image_index, 2))

    first = build_core_subset(records, per_dataset=5, seed=20260730)
    second = build_core_subset(records, per_dataset=5, seed=20260730)

    assert [record["query_id"] for record in first] == [
        record["query_id"] for record in second
    ]
    assert len(first) == 15
    assert len({record["image_key"] for record in first}) == 15
    assert {
        dataset: sum(record["dataset"] == dataset for record in first)
        for dataset in ("refcoco", "refcoco_plus", "refcocog")
    } == {"refcoco": 5, "refcoco_plus": 5, "refcocog": 5}


def test_tile_probe_balances_small_and_control_per_dataset() -> None:
    core = []
    for dataset_offset, dataset in enumerate(
        ("refcoco", "refcoco_plus", "refcocog")
    ):
        for index in range(8):
            core.append(
                _record(
                    dataset,
                    dataset_offset * 100 + index,
                    1,
                    area=0.001 * (index + 1),
                )
            )

    probe = build_tile_probe(
        core,
        small_per_dataset=2,
        control_per_dataset=2,
        seed=20260730,
    )

    assert len(probe) == 12
    for dataset in ("refcoco", "refcoco_plus", "refcocog"):
        dataset_records = [r for r in probe if r["dataset"] == dataset]
        assert sum(r["probe_group"] == "small" for r in dataset_records) == 2
        assert sum(r["probe_group"] == "control" for r in dataset_records) == 2


def test_candidate_evaluation_reports_first_and_oracle_headroom() -> None:
    ground_truth = [0.1, 0.1, 0.4, 0.4]
    candidates = [
        DiagnosticCandidate(
            bbox=[0.6, 0.6, 0.9, 0.9],
            label="wrong",
            score=None,
            source="full",
        ),
        DiagnosticCandidate(
            bbox=[0.1, 0.1, 0.4, 0.4],
            label="target",
            score=None,
            source="full",
        ),
    ]

    result = evaluate_candidates(ground_truth, candidates, selected_index=0)

    assert result["selected_iou"] == 0.0
    assert result["best_candidate_iou"] == 1.0
    assert result["selected_acc_at_05"] is False
    assert result["oracle_acc_at_05"] is True
    assert result["top5_oracle_acc_at_05"] is True
    assert result["top10_oracle_acc_at_05"] is True


def test_candidate_evaluation_treats_no_candidates_as_incorrect() -> None:
    result = evaluate_candidates([0.1, 0.1, 0.4, 0.4], [])

    assert result["selected_bbox"] is None
    assert result["selected_iou"] == 0.0
    assert result["best_candidate_iou"] == 0.0
    assert result["selected_acc_at_05"] is False
    assert result["oracle_acc_at_05"] is False


def test_tile_windows_have_four_overlapping_corner_tiles() -> None:
    windows = tile_windows(width=1000, height=500, overlap_ratio=0.2)

    assert len(windows) == 4
    assert windows[0].as_box() == [0, 0, 556, 278]
    assert windows[-1].as_box() == [444, 222, 1000, 500]
    horizontal_overlap = windows[0].x2 - windows[1].x1
    assert horizontal_overlap / windows[0].width == 0.2014388489208633


def test_tile_bbox_maps_back_to_original_and_clamps_edges() -> None:
    windows = tile_windows(width=1000, height=500, overlap_ratio=0.2)
    bottom_right = windows[-1]

    mapped = map_tile_bbox_to_normalized(
        [-10.0, 0.0, 600.0, 300.0],
        tile=bottom_right,
        image_width=1000,
        image_height=500,
    )

    assert mapped == [0.434, 0.444, 1.0, 1.0]


def test_tile_deduplication_only_merges_same_normalized_label() -> None:
    candidates = [
        DiagnosticCandidate([0.1, 0.1, 0.4, 0.4], "The Person", None, "full"),
        DiagnosticCandidate([0.1, 0.1, 0.4, 0.4], "the person.", None, "tile_0"),
        DiagnosticCandidate([0.1, 0.1, 0.4, 0.4], "shirt", None, "tile_0"),
    ]

    deduplicated = deduplicate_candidates(candidates, iou_threshold=0.85)

    assert [(candidate.label, candidate.source) for candidate in deduplicated] == [
        ("The Person", "full"),
        ("shirt", "tile_0"),
    ]
