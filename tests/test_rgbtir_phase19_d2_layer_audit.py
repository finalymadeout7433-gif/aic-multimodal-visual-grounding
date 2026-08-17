from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from aic_rgbtir.data import RGBTRecord
from aic_rgbtir.phase19_d2_layer_audit import (
    D2LayerAudit,
    LayerAuditFeatures,
    pair_safe_shuffle_indices,
    summarize_checkpoint_audits,
)


def _record(index: int, *, pair: int | None = None) -> RGBTRecord:
    pair_index = index if pair is None else pair
    return RGBTRecord(
        record_id=f"r{index}",
        source_dataset="flir",
        split="train",
        rgb_relpath=f"rgb/pair_{pair_index}.jpg",
        tir_relpath=f"tir/pair_{pair_index}.jpg",
        query_original=f"target {index}",
        bbox_xywh_pixel=(1.0, 1.0, 8.0, 8.0),
        bbox_xyxy_normalized=(0.1, 0.1, 0.9, 0.9),
        width=10,
        height=10,
        illumination="NL" if index % 2 else "VWL",
        weather="FY",
        object_size="small" if index % 3 == 0 else "normal",
        occlusion="NO",
        crowded="NC",
        scene="UB",
    )


def _features(count: int, collapsed_layer: str | None = None) -> LayerAuditFeatures:
    identity = torch.eye(count)
    adapted = {layer: identity.clone() for layer in ("8", "16", "24")}
    if collapsed_layer is not None:
        adapted[collapsed_layer] = torch.ones(count, count)
    return LayerAuditFeatures(
        adapted=adapted,
        base={layer: identity.clone() for layer in ("8", "16", "24")},
        teacher={layer: identity.clone() for layer in ("8", "16", "24")},
    )


def test_layer_audit_identifies_the_single_collapsed_layer() -> None:
    records = [_record(index) for index in range(12)]
    result = D2LayerAudit(minimum_rank_vs_base=0.85).analyze(
        split="multiquery",
        checkpoint_step=11696,
        records=records,
        features=_features(12, collapsed_layer="16"),
        fingerprint="ABC",
    )

    assert result["bottleneck_layer"] == "16"
    assert result["layers"]["16"]["effective_rank_ratio_vs_base"] == 0.0
    assert result["layers"]["8"]["effective_rank_ratio_vs_base"] == pytest.approx(1.0)
    assert result["layers"]["24"]["effective_rank_ratio_vs_teacher"] == pytest.approx(1.0)
    assert result["passed_absolute_rank"] is False
    assert result["layers"]["16"]["singular_spectrum"]["effective_rank"] == 0.0


def test_pair_safe_shuffle_never_uses_the_same_image_pair_and_is_repeatable() -> None:
    records = [_record(index, pair=index // 2) for index in range(12)]
    first = pair_safe_shuffle_indices(records, seed=20260817)
    second = pair_safe_shuffle_indices(records, seed=20260817)

    assert torch.equal(first, second)
    assert all(
        records[index].image_pair_key != records[int(shuffled)].image_pair_key
        for index, shuffled in enumerate(first)
    )


def test_layer_audit_reports_record_and_pair_aggregated_rank() -> None:
    records = [_record(index, pair=index // 2) for index in range(12)]
    result = D2LayerAudit().analyze(
        split="multiquery",
        checkpoint_step=11696,
        records=records,
        features=_features(12),
        fingerprint="ABC",
    )

    assert result["record_count"] == 12
    assert result["pair_count"] == 6
    assert result["layers"]["8"]["pair_aggregated"]["count"] == 6
    assert result["layers"]["8"]["pair_safe_margin_mean"] > 0


def test_layer_audit_rejects_nonfinite_features() -> None:
    records = [_record(index) for index in range(12)]
    features = _features(12)
    broken = features.adapted["8"].clone()
    broken[0, 0] = torch.nan
    features = replace(features, adapted={**features.adapted, "8": broken})

    with pytest.raises(ValueError, match="non-finite"):
        D2LayerAudit().analyze(
            split="semantic",
            checkpoint_step=5848,
            records=records,
            features=features,
            fingerprint="ABC",
        )


def test_checkpoint_summary_uses_best_multiquery_step_as_diagnostic_only() -> None:
    def audit(split: str, step: int, ratio: float, layer: str) -> dict[str, object]:
        return {
            "split": split,
            "checkpoint_step": step,
            "minimum_effective_rank_ratio_vs_base": ratio,
            "bottleneck_layer": layer,
        }

    summary = summarize_checkpoint_audits(
        {
            5848: {
                "semantic": audit("semantic", 5848, 0.91, "24"),
                "multiquery": audit("multiquery", 5848, 0.82, "16"),
            },
            11696: {
                "semantic": audit("semantic", 11696, 0.90, "24"),
                "multiquery": audit("multiquery", 11696, 0.84, "16"),
            },
        }
    )

    assert summary["diagnostic_reference_step"] == 11696
    assert summary["diagnostic_reference_is_release"] is False
    assert summary["single_layer_bottleneck_across_checkpoints"] is True
