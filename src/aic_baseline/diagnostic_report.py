from __future__ import annotations

from typing import Any


def decide_next_actions(
    *,
    florence_first: float,
    florence_oracle: float,
    tile_small_full_oracle: float,
    tile_small_combined_oracle: float,
    tile_control_full_oracle: float,
    tile_control_combined_oracle: float,
    gdino_top1: float | None,
    gdino_top10_oracle: float | None,
) -> dict[str, Any]:
    """Apply the experiment plan's fixed decision thresholds."""

    florence_gap = florence_oracle - florence_first
    if florence_gap >= 0.05:
        florence = "target-role-aware reranker"
    elif florence_gap >= 0.02:
        florence = "conservative same-label reranker only"
    else:
        florence = "defer reranker"

    tile_small_gain = (
        tile_small_combined_oracle - tile_small_full_oracle
    )
    tile_control_drop = (
        tile_control_full_oracle - tile_control_combined_oracle
    )
    tile = (
        "selective tile"
        if tile_small_gain >= 0.05 and tile_control_drop < 0.01
        else "do not scale tile"
    )

    if gdino_top1 is None:
        gdino_top1_decision = "not evaluated"
    elif gdino_top1 - florence_first >= 0.02:
        gdino_top1_decision = "platform candidate"
    else:
        gdino_top1_decision = "not a platform candidate"

    if gdino_top10_oracle is None:
        gdino_candidate_decision = "not evaluated"
    elif gdino_top10_oracle - florence_oracle >= 0.05:
        gdino_candidate_decision = "candidate generator"
    else:
        gdino_candidate_decision = "no candidate advantage"

    return {
        "florence": florence,
        "florence_oracle_gap": florence_gap,
        "tile": tile,
        "tile_small_oracle_gain": tile_small_gain,
        "tile_control_oracle_drop": tile_control_drop,
        "grounding_dino_top1": gdino_top1_decision,
        "grounding_dino_top1_delta": (
            None if gdino_top1 is None else gdino_top1 - florence_first
        ),
        "grounding_dino_candidates": gdino_candidate_decision,
        "grounding_dino_top10_oracle_delta": (
            None
            if gdino_top10_oracle is None
            else gdino_top10_oracle - florence_oracle
        ),
    }
