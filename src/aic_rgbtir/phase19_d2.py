from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .data import RGBTRecord
from .phase16 import (
    DISCRIMINATIVE_LAYERS,
    LAYER_NAMES,
    Phase16Config,
    Phase16RepairTrainer,
    RGBTeacherBank,
    TeacherEntry,
)
from .phase18 import D1CandidateSpec, D1LossOutput, D1RetentionObjective


@dataclass(frozen=True)
class D2CandidateSpec(D1CandidateSpec):
    """D1_L050 with one controlled Base-TIR geometry weight."""

    geometry_weight: float = 0.0

    @classmethod
    def d2(cls, geometry_weight: float) -> "D2CandidateSpec":
        label = f"D2_G{round(geometry_weight * 100):03d}"
        return cls(label, 0.25, 1.0, 0.50, 0.10, 0.50, float(geometry_weight))

    @classmethod
    def defaults(cls) -> tuple["D2CandidateSpec", ...]:
        return tuple(cls.d2(value) for value in (0.10, 0.25, 0.50))


@dataclass(frozen=True)
class D2LossOutput:
    total: torch.Tensor
    d1_total: torch.Tensor
    pair: torch.Tensor
    contrastive: torch.Tensor
    relational: torch.Tensor
    background: torch.Tensor
    retention: torch.Tensor
    geometry: torch.Tensor


class D2GeometryRetentionObjective(nn.Module):
    """Preserve Base-TIR cross-record geometry while keeping D1 unchanged."""

    requires_base_tir_anchors = True

    def __init__(
        self,
        candidate: D2CandidateSpec,
        *,
        temperature: float = 0.07,
        margin: float = 0.15,
        retention_tolerance: Mapping[str, float] | None = None,
        layer_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.candidate = candidate
        self.d1 = D1RetentionObjective(
            candidate,
            temperature=temperature,
            margin=margin,
            retention_tolerance=retention_tolerance,
        )
        self.layer_weights = dict(
            layer_weights or {"8": 1.0, "16": 1.0, "24": 0.25}
        )
        if set(self.layer_weights) != set(DISCRIMINATIVE_LAYERS):
            raise ValueError("D2 layer weights must cover exactly Layer 8/16/24")
        if any(float(value) <= 0 for value in self.layer_weights.values()):
            raise ValueError("D2 layer weights must be positive")

    def geometry_loss(
        self,
        tir_foreground: Mapping[str, torch.Tensor],
        base_tir: Mapping[str, torch.Tensor],
        base_tir_anchors: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        weighted: list[torch.Tensor] = []
        weights: list[float] = []
        for layer in DISCRIMINATIVE_LAYERS:
            student = F.normalize(tir_foreground[layer].float(), dim=0)
            base = F.normalize(base_tir[layer].float().detach().to(student.device), dim=0)
            raw_anchors = base_tir_anchors[layer]
            if raw_anchors.ndim != 2 or raw_anchors.shape[0] < 2:
                raise ValueError("D2 geometry requires at least two Base-TIR anchors")
            anchors = F.normalize(
                raw_anchors.float().detach().to(student.device), dim=1
            )
            predicted = anchors @ student
            target = anchors @ base
            # The absolute cosine values of Base-TIR are highly concentrated.
            # Matching raw values admits a nearly constant collapsed shortcut.
            # Standardising each relation vector preserves neighbour ordering
            # and relative geometry while removing this dataset-level offset.
            predicted = (predicted - predicted.mean()) / predicted.std().clamp_min(1e-6)
            target = (target - target.mean()) / target.std().clamp_min(1e-6)
            weight = float(self.layer_weights[layer])
            weighted.append(weight * F.mse_loss(predicted, target))
            weights.append(weight)
        return torch.stack(weighted).sum() / sum(weights)

    def forward(
        self,
        tir_foreground: Mapping[str, torch.Tensor],
        rgb_positive: Mapping[str, torch.Tensor],
        rgb_background: Mapping[str, torch.Tensor],
        rgb_negatives: Mapping[str, torch.Tensor],
        rgb_anchors: Mapping[str, torch.Tensor],
        *,
        base_tir: Mapping[str, torch.Tensor] | None = None,
        base_tir_anchors: Mapping[str, torch.Tensor] | None = None,
    ) -> D2LossOutput:
        if base_tir is None or base_tir_anchors is None:
            raise ValueError("D2 requires Base-TIR positive and anchor features")
        d1: D1LossOutput = self.d1(
            tir_foreground,
            rgb_positive,
            rgb_background,
            rgb_negatives,
            rgb_anchors,
            base_tir=base_tir,
        )
        geometry = self.geometry_loss(tir_foreground, base_tir, base_tir_anchors)
        return D2LossOutput(
            total=d1.total + self.candidate.geometry_weight * geometry,
            d1_total=d1.total,
            pair=d1.pair,
            contrastive=d1.contrastive,
            relational=d1.relational,
            background=d1.background,
            retention=d1.retention,
            geometry=geometry,
        )


class D2RepairTrainer(Phase16RepairTrainer):
    """Phase 1.8 trainer with Base-TIR anchor geometry supplied to D2 only."""

    def __init__(
        self,
        *,
        adapter_factory: Callable[[], nn.Module],
        encode_tir: Callable[[nn.Module, RGBTRecord], Mapping[str, torch.Tensor]],
        output_root: Path | str,
        config: Phase16Config,
        candidate_specs: Sequence[D2CandidateSpec] | None = None,
        adapter_state: Callable[[nn.Module], Mapping[str, torch.Tensor]] | None = None,
        load_adapter_state: Callable[[nn.Module, Mapping[str, torch.Tensor]], Any]
        | None = None,
        resume: bool = False,
        stop_after_probe: bool = True,
    ) -> None:
        candidates = tuple(candidate_specs or D2CandidateSpec.defaults())
        super().__init__(
            adapter_factory=adapter_factory,
            encode_tir=encode_tir,
            output_root=output_root,
            config=config,
            adapter_state=adapter_state,
            load_adapter_state=load_adapter_state,
            resume=resume,
            candidate_specs=candidates,
            objective_factory=lambda candidate: D2GeometryRetentionObjective(
                candidate,  # type: ignore[arg-type]
                temperature=config.temperature,
                margin=config.margin,
            ),
            stop_after_probe=stop_after_probe,
        )

    @staticmethod
    def _base_tir_anchor_layers(
        bank: RGBTeacherBank,
        record_ids: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for layer in LAYER_NAMES:
            values = []
            for record_id in record_ids:
                entry = bank[record_id]
                base = entry.base_tir or entry.foreground
                values.append(base[layer])
            result[layer] = torch.stack(values)
        return result

    def _objective_kwargs(
        self,
        *,
        positive: TeacherEntry,
        bank: RGBTeacherBank,
        anchor_ids: Sequence[str],
    ) -> dict[str, Any]:
        if len(anchor_ids) < 2:
            raise ValueError("D2 requires at least two relational anchors")
        return {
            "base_tir": positive.base_tir or positive.foreground,
            "base_tir_anchors": self._base_tir_anchor_layers(bank, anchor_ids),
        }

    def run_dual_dev(
        self,
        train_records: Sequence[RGBTRecord],
        semantic_dev: Sequence[RGBTRecord],
        multiquery_dev: Sequence[RGBTRecord],
        bank: RGBTeacherBank,
    ) -> "D2ProbeResult":
        self.output_root.mkdir(parents=True, exist_ok=True)
        baseline_module = self.adapter_factory()
        baseline = {
            "semantic": self._evaluate(baseline_module, semantic_dev, bank),
            "multiquery": self._evaluate(baseline_module, multiquery_dev, bank),
        }
        candidate_metrics: dict[str, dict[str, Mapping[str, float]]] = {}
        decisions: dict[str, Mapping[str, Any]] = {}
        for candidate in self.candidate_specs:
            module = self.adapter_factory()
            self._train(
                module,
                candidate,
                train_records,
                bank,
                steps=self.config.probe_steps,
            )
            metrics = {
                "semantic": self._evaluate(module, semantic_dev, bank),
                "multiquery": self._evaluate(module, multiquery_dev, bank),
            }
            candidate_metrics[candidate.name] = metrics
            decisions[candidate.name] = evaluate_d2_gate(metrics, baseline)
            candidate_dir = self.output_root / "probe_candidates" / candidate.name
            candidate_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.adapter_state(module), candidate_dir / "adapter.pt")
            (candidate_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "candidate": asdict(candidate),
                        "teacher_bank_fingerprint": bank.fingerprint,
                        "baseline": baseline,
                        "metrics": metrics,
                        "decision": decisions[candidate.name],
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        eligible = [
            candidate
            for candidate in self.candidate_specs
            if decisions[candidate.name]["status"] == "PHASE_19_D2_CANDIDATE_GO"
        ]
        selected = ""
        if eligible:
            selected = max(
                eligible,
                key=lambda candidate: (
                    min(
                        float(
                            candidate_metrics[candidate.name][split][
                                "minimum_effective_rank_ratio_vs_base"
                            ]
                        )
                        for split in ("semantic", "multiquery")
                    ),
                    sum(
                        float(candidate_metrics[candidate.name][split]["mean_r_at_5"])
                        for split in ("semantic", "multiquery")
                    ),
                    -float(candidate.geometry_weight),  # type: ignore[attr-defined]
                ),
            ).name
        result = D2ProbeResult(
            status="PHASE_19_D2_PROBE_GO" if selected else "PHASE_19_D2_PROBE_NO_GO",
            selected_candidate=selected,
            baseline_metrics=baseline,
            candidate_metrics=candidate_metrics,
            candidate_decisions=decisions,
        )
        (self.output_root / "run_summary.json").write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result


@dataclass(frozen=True)
class D2ProbeResult:
    status: str
    selected_candidate: str
    baseline_metrics: Mapping[str, Mapping[str, float]]
    candidate_metrics: Mapping[str, Mapping[str, Mapping[str, float]]]
    candidate_decisions: Mapping[str, Mapping[str, Any]]


def evaluate_d2_gate(
    candidate: Mapping[str, Mapping[str, float]],
    baseline: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    splits = ("semantic", "multiquery")
    rank_delta = {
        split: float(candidate[split]["minimum_effective_rank_ratio_vs_base"])
        - float(baseline[split]["minimum_effective_rank_ratio_vs_base"])
        for split in splits
    }
    r5_drop = {
        split: float(baseline[split]["mean_r_at_5"])
        - float(candidate[split]["mean_r_at_5"])
        for split in splits
    }
    p95_delta = {
        split: float(candidate[split]["maximum_nonpaired_cosine_p95_increase"])
        - float(baseline[split]["maximum_nonpaired_cosine_p95_increase"])
        for split in splits
    }
    checks = {
        "rank_no_regression": all(value >= -0.01 for value in rank_delta.values()),
        "rank_material_improvement": max(rank_delta.values()) >= 0.02,
        "semantic_r5_retained": r5_drop["semantic"] <= 0.005,
        "multiquery_r5_retained": r5_drop["multiquery"] <= 0.010,
        "nonpaired_p95_controlled": all(value <= 0.02 for value in p95_delta.values()),
        "paired_margin_positive": all(
            float(candidate[split]["minimum_paired_shuffled_margin"]) > 0.0
            for split in splits
        ),
    }
    return {
        "status": (
            "PHASE_19_D2_CANDIDATE_GO"
            if all(checks.values())
            else "PHASE_19_D2_CANDIDATE_NO_GO"
        ),
        "checks": checks,
        "rank_delta": rank_delta,
        "r5_drop": r5_drop,
        "nonpaired_p95_delta": p95_delta,
    }


@dataclass(frozen=True)
class D2DataSplit:
    multiquery_dev: tuple[RGBTRecord, ...]
    remaining_full_train: tuple[RGBTRecord, ...]


def build_d2_multiquery_split(
    full_records: Sequence[RGBTRecord],
    probe_records: Sequence[RGBTRecord],
    *,
    pair_count: int,
    seed: int,
) -> D2DataSplit:
    if pair_count <= 0:
        raise ValueError("D2 multi-query pair_count must be positive")
    probe_pairs = {record.image_pair_key for record in probe_records}
    grouped: dict[str, list[RGBTRecord]] = {}
    for record in full_records:
        grouped.setdefault(record.image_pair_key, []).append(record)
    candidates = [
        pair
        for pair, records in grouped.items()
        if pair not in probe_pairs and len(records) >= 2
    ]
    if len(candidates) < pair_count:
        raise ValueError(
            f"insufficient multi-query pairs: {len(candidates)} < {pair_count}"
        )

    def order_key(pair: str) -> str:
        return hashlib.sha256(f"{seed}:{pair}".encode("utf-8")).hexdigest()

    selected_pairs = set(sorted(candidates, key=order_key)[:pair_count])
    multiquery = tuple(
        sorted(
            (
                record
                for pair in selected_pairs
                for record in grouped[pair]
            ),
            key=lambda record: record.record_id,
        )
    )
    remaining = tuple(
        sorted(
            (
                record
                for record in full_records
                if record.image_pair_key not in selected_pairs
            ),
            key=lambda record: record.record_id,
        )
    )
    return D2DataSplit(multiquery_dev=multiquery, remaining_full_train=remaining)
