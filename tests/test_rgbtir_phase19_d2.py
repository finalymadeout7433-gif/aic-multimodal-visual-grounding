from __future__ import annotations

import pytest
import torch

from aic_rgbtir.data import RGBTRecord
from aic_rgbtir.phase19_d2 import (
    D2CandidateSpec,
    D2GeometryRetentionObjective,
    build_d2_multiquery_split,
    evaluate_d2_gate,
)


LAYERS = ("8", "16", "24", "final")


def _layers(value: torch.Tensor) -> dict[str, torch.Tensor]:
    return {layer: value.clone() for layer in LAYERS}


def _record(record_id: str, pair: str, source: str = "fixture") -> RGBTRecord:
    return RGBTRecord(
        record_id=record_id,
        source_dataset=source,
        split="train",
        rgb_relpath=f"rgb/{pair}.png",
        tir_relpath=f"tir/{pair}.png",
        query_original=record_id,
        bbox_xywh_pixel=(0.0, 0.0, 10.0, 10.0),
        bbox_xyxy_normalized=(0.0, 0.0, 1.0, 1.0),
        width=10,
        height=10,
        illumination="NL",
        weather="FY",
        object_size="NS",
        occlusion="NO",
        crowded="NC",
        scene="UB",
    )


def test_d2_geometry_is_zero_on_base_tir_relations_and_penalises_collapse() -> None:
    base = torch.tensor([1.0, 0.0, 0.0])
    anchors = torch.eye(3)
    objective = D2GeometryRetentionObjective(D2CandidateSpec.d2(0.25))

    preserved = objective.geometry_loss(
        _layers(base), _layers(base), _layers(anchors)
    )
    collapsed_student = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)
    collapsed = objective.geometry_loss(
        _layers(collapsed_student), _layers(base), _layers(anchors)
    )

    assert preserved == pytest.approx(0.0, abs=1e-7)
    assert collapsed > 0
    collapsed.backward()
    assert collapsed_student.grad is not None


def test_d2_excludes_final_and_downweights_layer24() -> None:
    base = _layers(torch.tensor([1.0, 0.0]))
    anchors = _layers(torch.eye(2))
    student = _layers(torch.tensor([1.0, 0.0]))
    student["final"] = torch.tensor([0.0, 1.0])
    objective = D2GeometryRetentionObjective(
        D2CandidateSpec.d2(0.25), layer_weights={"8": 1.0, "16": 1.0, "24": 0.25}
    )

    assert objective.geometry_loss(student, base, anchors) == pytest.approx(0.0)

    student["24"] = torch.tensor([0.0, 1.0])
    layer24_only = objective.geometry_loss(student, base, anchors)
    student["24"] = torch.tensor([1.0, 0.0])
    student["8"] = torch.tensor([0.0, 1.0])
    layer8_only = objective.geometry_loss(student, base, anchors)
    assert layer24_only < layer8_only


def test_d2_candidates_change_only_geometry_weight() -> None:
    candidates = D2CandidateSpec.defaults()
    assert [item.name for item in candidates] == ["D2_G010", "D2_G025", "D2_G050"]
    assert [item.geometry_weight for item in candidates] == [0.10, 0.25, 0.50]
    fixed = {
        (
            item.pair_weight,
            item.contrastive_weight,
            item.relational_weight,
            item.background_weight,
            item.retention_weight,
        )
        for item in candidates
    }
    assert fixed == {(0.25, 1.0, 0.50, 0.10, 0.50)}


def test_d2_multiquery_split_is_deterministic_and_pair_isolated() -> None:
    full = []
    for pair_index in range(8):
        for query_index in range(2 + pair_index % 2):
            full.append(_record(f"r{pair_index}-{query_index}", f"pair-{pair_index}"))
    probe = [_record("probe", "pair-0")]

    first = build_d2_multiquery_split(full, probe, pair_count=3, seed=20260812)
    second = build_d2_multiquery_split(full, probe, pair_count=3, seed=20260812)

    assert [row.record_id for row in first.multiquery_dev] == [
        row.record_id for row in second.multiquery_dev
    ]
    dev_pairs = {row.image_pair_key for row in first.multiquery_dev}
    assert len(dev_pairs) == 3
    assert "fixture/pair-0.png" not in dev_pairs
    assert not dev_pairs & {row.image_pair_key for row in first.remaining_full_train}
    assert all(
        sum(row.image_pair_key == pair for row in first.multiquery_dev) >= 2
        for pair in dev_pairs
    )


def test_d2_split_refuses_insufficient_multiquery_pairs() -> None:
    full = [_record("a", "pa"), _record("b", "pb")]
    with pytest.raises(ValueError, match="multi-query"):
        build_d2_multiquery_split(full, (), pair_count=1, seed=20260812)


def test_d2_gate_requires_rank_gain_without_retrieval_regression() -> None:
    def metrics(rank: float, r5: float, p95: float = 0.02) -> dict[str, float]:
        return {
            "minimum_effective_rank_ratio_vs_base": rank,
            "mean_r_at_5": r5,
            "maximum_nonpaired_cosine_p95_increase": p95,
            "minimum_paired_shuffled_margin": 0.1,
        }

    baseline = {
        "semantic": metrics(0.80, 0.90),
        "multiquery": metrics(0.78, 0.88),
    }
    good = {
        "semantic": metrics(0.81, 0.899),
        "multiquery": metrics(0.81, 0.875),
    }
    no_rank_gain = {
        "semantic": metrics(0.805, 0.90),
        "multiquery": metrics(0.785, 0.88),
    }

    assert evaluate_d2_gate(good, baseline)["status"] == "PHASE_19_D2_CANDIDATE_GO"
    assert (
        evaluate_d2_gate(no_rank_gain, baseline)["status"]
        == "PHASE_19_D2_CANDIDATE_NO_GO"
    )
