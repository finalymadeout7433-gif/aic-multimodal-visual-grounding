from __future__ import annotations

from tools.compare_grounder_runs import compare_runs


def _row(query_id: str, *, selected: bool, oracle: bool, box) -> dict:
    return {
        "query_id": query_id,
        "selected_acc_at_05": selected,
        "oracle_acc_at_05": oracle,
        "selected_bbox": box,
    }


def test_compare_grounder_runs_reports_union_and_unique_correctness() -> None:
    left = [
        _row("a", selected=True, oracle=True, box=[0.0, 0.0, 0.5, 0.5]),
        _row("b", selected=False, oracle=False, box=[0.0, 0.0, 0.2, 0.2]),
    ]
    right = [
        _row("a", selected=False, oracle=True, box=[0.0, 0.0, 0.5, 0.5]),
        _row("b", selected=True, oracle=True, box=[0.5, 0.5, 1.0, 1.0]),
    ]

    result = compare_runs(left, right, left_name="left", right_name="right")

    assert result["left_top1_acc_at_05"] == 0.5
    assert result["right_top1_acc_at_05"] == 0.5
    assert result["candidate_union_oracle_acc_at_05"] == 1.0
    assert result["left_only_top1_correct"] == 1
    assert result["right_only_top1_correct"] == 1
