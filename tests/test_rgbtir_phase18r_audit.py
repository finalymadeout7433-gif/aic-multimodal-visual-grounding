from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from aic_rgbtir.phase1 import freeze_for_rgbtir_inference
from aic_rgbtir.phase18r_audit import Phase18RAuditConfig, Phase18RAuditor


LAYERS = ("8", "16", "24")


def _record(record_id: str, pair: str, rgb: list[float], tir: list[float]) -> dict:
    pooled = {
        "rgb": {layer: torch.tensor(rgb) for layer in LAYERS},
        "base": {layer: torch.tensor(rgb) for layer in LAYERS},
        "adapted": {layer: torch.tensor(tir) for layer in LAYERS},
    }
    return {"record_id": record_id, "image_pair_key": pair, "pooled": pooled}


def test_freeze_for_rgbtir_inference_closes_the_validation_contract() -> None:
    module = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
    module.train()
    assert any(parameter.requires_grad for parameter in module.parameters())

    returned = freeze_for_rgbtir_inference(module)

    assert returned is module
    assert module.training is False
    assert not any(parameter.requires_grad for parameter in module.parameters())


def test_retrieval_separates_exact_record_from_same_pair_multi_positive() -> None:
    records = [
        _record("a1", "pair-a", [1.0, 0.0], [0.0, 1.0]),
        _record("a2", "pair-a", [0.0, 1.0], [1.0, 0.0]),
        _record("b1", "pair-b", [-1.0, 0.0], [-1.0, 0.0]),
    ]

    metrics = Phase18RAuditor.retrieval_audit(records, layers=("8",))

    assert metrics["8"]["adapted"]["record_exact"]["r_at_1"] == pytest.approx(1 / 3)
    assert metrics["8"]["adapted"]["pair_multi_positive"]["r_at_1"] == 1.0


def test_spectrum_reports_record_and_unique_pair_levels() -> None:
    records = [
        _record("a1", "pair-a", [1.0, 0.0], [1.0, 0.0]),
        _record("a2", "pair-a", [1.0, 0.0], [1.0, 0.0]),
        _record("b1", "pair-b", [0.0, 1.0], [0.0, 1.0]),
        _record("c1", "pair-c", [-1.0, 0.0], [-1.0, 0.0]),
    ]

    metrics = Phase18RAuditor.spectrum_audit(records, layers=("8",))

    adapted = metrics["8"]["adapted"]
    assert adapted["record_level"]["sample_count"] == 4
    assert adapted["pair_mean_level"]["sample_count"] == 3
    assert adapted["record_level"]["effective_rank"] > 1.0
    assert adapted["record_level"]["participation_ratio"] > 1.0
    assert 0.0 <= adapted["record_level"]["top_energy_fraction"]["2"] <= 1.0


def test_subgroup_excess_drift_removes_the_global_shift() -> None:
    rows = [
        {"image_pair_key": "p1", "group": "stable", "base_alignment_loss": 0.1, "adapted_alignment_loss": 0.2},
        {"image_pair_key": "p2", "group": "stable", "base_alignment_loss": 0.2, "adapted_alignment_loss": 0.3},
        {"image_pair_key": "p3", "group": "harm", "base_alignment_loss": 0.1, "adapted_alignment_loss": 0.4},
        {"image_pair_key": "p4", "group": "harm", "base_alignment_loss": 0.2, "adapted_alignment_loss": 0.5},
    ]

    audit = Phase18RAuditor.subgroup_excess_drift(
        rows,
        group_fields=("group",),
        bootstrap_iterations=100,
        seed=20260812,
    )
    by_group = {row["group_value"]: row for row in audit["groups"]}

    assert abs(audit["global_absolute_drift"] - 0.2) < 1e-7
    assert abs(by_group["stable"]["excess_drift"] + 0.1) < 1e-7
    assert abs(by_group["harm"]["excess_drift"] - 0.1) < 1e-7
    assert by_group["harm"]["unique_pair_count"] == 2
    assert len(by_group["harm"]["excess_drift_ci95"]) == 2


def test_full_audit_returns_both_required_for_phase18r_pattern(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    official = input_root / "official_val"
    selected = input_root / "selected_adapter"
    official.mkdir(parents=True)
    selected.mkdir(parents=True)
    records = [
        _record("a", "pa", [1.0, 0.0], [1.0, 0.0]),
        _record("b", "pb", [0.0, 1.0], [1.0, 0.0]),
        _record("c", "pc", [-1.0, 0.0], [1.0, 0.0]),
    ]
    torch.save({"fingerprint": "fixture", "records": records}, official / "embedding_cache.pt")
    with (official / "per_record_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(json.dumps({
                "record_id": record["record_id"],
                "image_pair_key": record["image_pair_key"],
                "source_dataset": "fixture",
                "base_alignment_loss": 0.10,
                "adapted_alignment_loss": 0.25 + index * 0.01,
            }) + "\n")
    (official / "safety_equivalence.json").write_text(json.dumps({
        "passed": False,
        "checks": {
            "no_trainable_parameters": False,
            "rgb_base_hash_unchanged": True,
            "no_second_model_fallback": True,
        },
    }), encoding="utf-8")
    adapter = {f"tensor_{index}": torch.ones(1) for index in range(108)}
    torch.save({"tir_adapter": adapter}, selected / "adapter.pt")

    result = Phase18RAuditor(
        input_root=input_root,
        output_root=tmp_path / "output",
        config=Phase18RAuditConfig(
            expected_records=3,
            expected_pairs=3,
            bootstrap_iterations=25,
            minimum_effective_rank_vs_base=0.85,
        ),
    ).run()

    assert result.decision == "BOTH_REQUIRED"
    assert result.gate_issue is True
    assert result.model_issue is True
    assert (tmp_path / "output" / "run_summary.json").is_file()
    assert (tmp_path / "output" / "decision.json").is_file()
    assert (tmp_path / "output" / "sha256_manifest.json").is_file()
