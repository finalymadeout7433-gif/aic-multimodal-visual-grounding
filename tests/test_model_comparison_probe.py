from __future__ import annotations

import pytest

from tools.build_model_comparison_probe import select_probe


def _records() -> list[dict[str, str]]:
    return [
        {
            "dataset": dataset,
            "query_id": f"{dataset}:{index}",
            "image_key": f"{dataset}:image:{index}",
        }
        for dataset in ("a", "b")
        for index in range(5)
    ]


def test_model_comparison_probe_is_balanced_unique_and_deterministic() -> None:
    first = select_probe(_records(), per_dataset=3, seed=7)
    second = select_probe(_records(), per_dataset=3, seed=7)

    assert first == second
    assert len(first) == 6
    assert len({row["image_key"] for row in first}) == 6
    assert sum(row["dataset"] == "a" for row in first) == 3
    assert sum(row["dataset"] == "b" for row in first) == 3


def test_model_comparison_probe_rejects_insufficient_unique_images() -> None:
    with pytest.raises(ValueError, match="expected 6"):
        select_probe(_records(), per_dataset=6, seed=7)
