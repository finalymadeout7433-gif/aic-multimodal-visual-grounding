from __future__ import annotations

from collections import Counter

from aic_baseline.ranker_subsets import (
    build_nested_ranker_subsets,
    split_by_image,
)


def _record(index: int, *, dataset: str, query: str) -> dict:
    return {
        "dataset": dataset,
        "query_id": f"{dataset}:{index}",
        "image_key": f"coco:{index // 2}",
        "query": query,
        "image_relpath": f"images/{index // 2}.jpg",
        "bbox_xyxy_normalized": [0.1, 0.1, 0.5, 0.6],
    }


def test_nested_ranker_subsets_are_deterministic_and_nested() -> None:
    queries = (
        "object on the left",
        "the second person",
        "the nearest car",
        "red umbrella",
        "person walking",
        "two dogs",
        "the object",
    )
    datasets = ("refcoco", "refcoco_plus", "refcocog")
    records = [
        _record(
            index,
            dataset=datasets[index % len(datasets)],
            query=queries[index % len(queries)],
        )
        for index in range(240)
    ]
    proportions = {
        "spatial": 1,
        "ordinal": 1,
        "depth": 1,
        "attribute": 1,
        "action": 1,
        "plural_group": 1,
        "other": 1,
    }

    first = build_nested_ranker_subsets(
        records,
        pilot_size=42,
        full_size=84,
        seed=20260731,
        category_proportions=proportions,
        pilot_image_cap=2,
        full_image_cap=3,
    )
    second = build_nested_ranker_subsets(
        records,
        pilot_size=42,
        full_size=84,
        seed=20260731,
        category_proportions=proportions,
        pilot_image_cap=2,
        full_image_cap=3,
    )

    pilot_ids = [record["query_id"] for record in first["pilot"]]
    full_ids = [record["query_id"] for record in first["full"]]
    assert pilot_ids == [
        record["query_id"] for record in second["pilot"]
    ]
    assert full_ids == [record["query_id"] for record in second["full"]]
    assert len(pilot_ids) == 42
    assert len(full_ids) == 84
    assert set(pilot_ids).issubset(full_ids)
    assert max(
        Counter(record["image_key"] for record in first["pilot"]).values()
    ) <= 2
    assert max(
        Counter(record["image_key"] for record in first["full"]).values()
    ) <= 3


def test_split_by_image_never_leaks_an_image_between_train_and_dev() -> None:
    records = [
        _record(
            index,
            dataset="refcoco",
            query="object on the left",
        )
        for index in range(80)
    ]

    train, dev = split_by_image(
        records,
        dev_fraction=0.2,
        seed=20260731,
    )

    assert train
    assert dev
    assert {record["image_key"] for record in train}.isdisjoint(
        {record["image_key"] for record in dev}
    )
    assert {
        record["query_id"] for record in train + dev
    } == {record["query_id"] for record in records}
