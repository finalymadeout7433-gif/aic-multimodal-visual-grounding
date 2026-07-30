from __future__ import annotations

from aic_baseline.diagnostic_report import decide_next_actions


def test_decision_rules_follow_fixed_thresholds() -> None:
    decisions = decide_next_actions(
        florence_first=0.50,
        florence_oracle=0.56,
        tile_small_full_oracle=0.30,
        tile_small_combined_oracle=0.36,
        tile_control_full_oracle=0.70,
        tile_control_combined_oracle=0.695,
        gdino_top1=0.53,
        gdino_top10_oracle=0.62,
    )

    assert decisions["florence"] == "target-role-aware reranker"
    assert decisions["tile"] == "selective tile"
    assert decisions["grounding_dino_top1"] == "platform candidate"
    assert decisions["grounding_dino_candidates"] == "candidate generator"


def test_decision_rules_reject_weak_changes() -> None:
    decisions = decide_next_actions(
        florence_first=0.50,
        florence_oracle=0.515,
        tile_small_full_oracle=0.30,
        tile_small_combined_oracle=0.34,
        tile_control_full_oracle=0.70,
        tile_control_combined_oracle=0.72,
        gdino_top1=0.49,
        gdino_top10_oracle=0.54,
    )

    assert decisions["florence"] == "defer reranker"
    assert decisions["tile"] == "do not scale tile"
    assert decisions["grounding_dino_top1"] == "not a platform candidate"
    assert decisions["grounding_dino_candidates"] == "no candidate advantage"
