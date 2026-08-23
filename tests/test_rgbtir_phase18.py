from __future__ import annotations

from pathlib import Path

import torch

from aic_rgbtir.phase18 import (
    D1CandidateSpec,
    D1RetentionObjective,
    Phase18Config,
    Phase18RetentionProbe,
    evaluate_retention_metrics,
)


LAYERS = ("8", "16", "24", "final")


def _layers(vector: torch.Tensor) -> dict[str, torch.Tensor]:
    return {layer: vector.clone() for layer in LAYERS}


def test_d1_retention_is_hinge_bounded_by_base_tir_and_ignores_final() -> None:
    teacher = torch.tensor([1.0, 0.0])
    base = torch.tensor([0.8, 0.6])  # cosine 0.8
    inside_tolerance = torch.tensor([0.80, 0.60])
    outside_tolerance = torch.tensor([0.50, (1.0 - 0.50**2) ** 0.5])
    objective = D1RetentionObjective(D1CandidateSpec.d1(0.25))

    safe = objective.retention_loss(
        _layers(inside_tolerance), _layers(teacher), _layers(base)
    )
    drifted = objective.retention_loss(
        _layers(outside_tolerance), _layers(teacher), _layers(base)
    )

    assert safe == 0
    assert drifted > 0

    changed_final = _layers(inside_tolerance)
    changed_final["final"] = -teacher
    assert objective.retention_loss(changed_final, _layers(teacher), _layers(base)) == 0


def test_d1_total_preserves_c2_components_and_adds_weighted_retention() -> None:
    teacher = _layers(torch.tensor([1.0, 0.0, 0.0]))
    student = {
        layer: torch.tensor([0.0, 1.0, 0.0], requires_grad=True) for layer in LAYERS
    }
    base = _layers(torch.tensor([0.8, 0.6, 0.0]))
    background = _layers(torch.tensor([-1.0, 0.0, 0.0]))
    negatives = {layer: torch.eye(3)[1:] for layer in LAYERS}
    anchors = {layer: torch.eye(3) for layer in LAYERS}
    objective = D1RetentionObjective(D1CandidateSpec.d1(0.50))

    output = objective(
        student,
        teacher,
        background,
        negatives,
        anchors,
        base_tir=base,
    )

    assert output.retention > 0
    assert torch.allclose(output.total, output.c2_total + 0.50 * output.retention)
    output.total.backward()
    assert all(value.grad is not None for value in student.values())


def test_retention_metrics_report_layer_drift_and_preregistered_gates() -> None:
    teacher = {layer: torch.eye(8) for layer in LAYERS}
    base = {layer: torch.eye(8) * 0.90 + torch.roll(torch.eye(8), 1, 1) * 0.10 for layer in LAYERS}
    adapted = {layer: torch.eye(8) * 0.88 + torch.roll(torch.eye(8), 1, 1) * 0.12 for layer in LAYERS}
    metrics = evaluate_retention_metrics(adapted, teacher, base)

    assert set(metrics["layer_drift"]) == {"8", "16", "24"}
    assert metrics["layer_r_at_1"] == {"8": 1.0, "16": 1.0, "24": 1.0}
    assert metrics["layer_r_at_5"] == {"8": 1.0, "16": 1.0, "24": 1.0}
    assert metrics["mean_r_at_5"] == 1.0
    assert metrics["minimum_paired_shuffled_margin"] > 0

    cfg = Phase18Config(
        c2_layer_drift={"8": 0.20, "16": 0.20, "24": 0.20},
        c2_mean_r_at_1=1.0,
        c2_mean_r_at_5=1.0,
        minimum_layer24_drift_reduction=0.25,
    )
    probe = Phase18RetentionProbe(output_root=Path("unused"), config=cfg)
    decision = probe.evaluate_gate(metrics)
    assert decision["checks"]["all_layer_drift_reduced"] is True
    assert decision["checks"]["layer24_drift_reduction"] is True
    assert decision["status"] == "PHASE_18_D1_GO"


def test_gate_rejects_retrieval_regression_even_when_drift_improves() -> None:
    cfg = Phase18Config(
        c2_layer_drift={"8": 0.20, "16": 0.20, "24": 0.40},
        c2_mean_r_at_1=0.60,
        c2_mean_r_at_5=0.84,
    )
    metrics = {
        "layer_drift": {"8": 0.10, "16": 0.10, "24": 0.20},
        "mean_r_at_1": 0.58,
        "mean_r_at_5": 0.80,
        "minimum_effective_rank_ratio_vs_base": 0.90,
        "minimum_effective_rank_ratio_vs_teacher": 0.82,
        "maximum_nonpaired_cosine_p95_increase": 0.0,
        "minimum_paired_shuffled_margin": 0.10,
    }
    probe = Phase18RetentionProbe(output_root=Path("unused"), config=cfg)

    decision = probe.evaluate_gate(metrics)

    assert decision["checks"]["r5_retained"] is False
    assert decision["status"] == "PHASE_18_D1_NO_GO"


def test_candidates_vary_only_retention_lambda() -> None:
    candidates = D1CandidateSpec.defaults()
    assert [candidate.name for candidate in candidates] == ["D1_L010", "D1_L025", "D1_L050"]
    assert [candidate.retention_weight for candidate in candidates] == [0.10, 0.25, 0.50]
    common = {
        (
            candidate.pair_weight,
            candidate.contrastive_weight,
            candidate.relational_weight,
            candidate.background_weight,
        )
        for candidate in candidates
    }
    assert common == {(0.25, 1.0, 0.50, 0.10)}
