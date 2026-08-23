from __future__ import annotations

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
    CandidateSpec,
    Phase16Config,
    Phase16RepairObjective,
    Phase16RepairTrainer,
    RGBTeacherBank,
    _candidate_metrics,
)


@dataclass(frozen=True)
class D1CandidateSpec(CandidateSpec):
    """C2 with one controlled variable: the Base-TIR retention weight."""

    retention_weight: float = 0.0

    @classmethod
    def d1(cls, retention_weight: float) -> "D1CandidateSpec":
        label = f"D1_L{round(retention_weight * 100):03d}"
        return cls(label, 0.25, 1.0, 0.50, 0.10, float(retention_weight))

    @classmethod
    def defaults(cls) -> tuple["D1CandidateSpec", ...]:
        return tuple(cls.d1(value) for value in (0.10, 0.25, 0.50))


@dataclass(frozen=True)
class D1LossOutput:
    total: torch.Tensor
    c2_total: torch.Tensor
    pair: torch.Tensor
    contrastive: torch.Tensor
    relational: torch.Tensor
    background: torch.Tensor
    retention: torch.Tensor


@dataclass(frozen=True)
class Phase18Config:
    seed: int = 20260812
    c2_layer_drift: Mapping[str, float] | None = None
    c2_mean_r_at_1: float = 0.5979817708333334
    c2_mean_r_at_5: float = 0.8359375
    maximum_r1_drop: float = 0.010
    maximum_r5_drop: float = 0.005
    minimum_layer24_drift_reduction: float = 0.25
    minimum_effective_rank_vs_base: float = 0.85
    minimum_effective_rank_vs_teacher: float = 0.75
    maximum_nonpaired_cosine_p95_increase: float = 0.10
    retention_tolerance: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.c2_layer_drift is None:
            object.__setattr__(
                self,
                "c2_layer_drift",
                {
                    "8": 0.1289336085319519,
                    "16": 0.15261441469192505,
                    "24": 0.3837621808052063,
                },
            )
        if self.retention_tolerance is None:
            object.__setattr__(
                self,
                "retention_tolerance",
                {"8": 0.02, "16": 0.02, "24": 0.01, "final": 0.0},
            )


@dataclass(frozen=True)
class Phase18Result:
    status: str
    selected_candidate: str
    completed_candidates: tuple[str, ...]
    candidate_decisions: Mapping[str, Mapping[str, Any]]
    probe_metrics: Mapping[str, Mapping[str, Any]]


class D1RetentionObjective(nn.Module):
    """Keep C2 discrimination while preventing teacher drift below Base TIR."""

    def __init__(
        self,
        candidate: D1CandidateSpec,
        *,
        temperature: float = 0.07,
        margin: float = 0.15,
        retention_tolerance: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.candidate = candidate
        self.c2 = Phase16RepairObjective(
            candidate,
            temperature=temperature,
            margin=margin,
        )
        self.retention_tolerance = dict(
            retention_tolerance
            or {"8": 0.02, "16": 0.02, "24": 0.01, "final": 0.0}
        )

    def retention_loss(
        self,
        tir_foreground: Mapping[str, torch.Tensor],
        rgb_positive: Mapping[str, torch.Tensor],
        base_tir: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for layer in DISCRIMINATIVE_LAYERS:
            student = tir_foreground[layer].float()
            device = student.device
            teacher = rgb_positive[layer].float().detach().to(device)
            base = base_tir[layer].float().detach().to(device)
            adapted_cosine = F.cosine_similarity(student[None], teacher[None]).squeeze(0)
            base_cosine = F.cosine_similarity(base[None], teacher[None]).squeeze(0)
            tolerance = float(self.retention_tolerance[layer])
            losses.append(F.relu(base_cosine - tolerance - adapted_cosine))
        return torch.stack(losses).mean()

    def forward(
        self,
        tir_foreground: Mapping[str, torch.Tensor],
        rgb_positive: Mapping[str, torch.Tensor],
        rgb_background: Mapping[str, torch.Tensor],
        rgb_negatives: Mapping[str, torch.Tensor],
        rgb_anchors: Mapping[str, torch.Tensor],
        *,
        base_tir: Mapping[str, torch.Tensor] | None = None,
    ) -> D1LossOutput:
        if base_tir is None:
            raise ValueError("D1 retention requires Base TIR features")
        c2 = self.c2(
            tir_foreground,
            rgb_positive,
            rgb_background,
            rgb_negatives,
            rgb_anchors,
        )
        retention = self.retention_loss(tir_foreground, rgb_positive, base_tir)
        total = c2.total + self.candidate.retention_weight * retention
        return D1LossOutput(
            total=total,
            c2_total=c2.total,
            pair=c2.pair,
            contrastive=c2.contrastive,
            relational=c2.relational,
            background=c2.background,
            retention=retention,
        )


def evaluate_retention_metrics(
    students: Mapping[str, torch.Tensor],
    teachers: Mapping[str, torch.Tensor],
    base_tir: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    metrics: dict[str, Any] = dict(_candidate_metrics(students, teachers, base_tir))
    layer_drift: dict[str, float] = {}
    layer_paired_cosine: dict[str, float] = {}
    layer_base_cosine: dict[str, float] = {}
    layer_r_at_1: dict[str, float] = {}
    layer_r_at_5: dict[str, float] = {}
    for layer in DISCRIMINATIVE_LAYERS:
        student = F.normalize(students[layer].float(), dim=1)
        teacher = F.normalize(teachers[layer].float(), dim=1)
        base = F.normalize(base_tir[layer].float(), dim=1)
        similarity = student @ teacher.T
        order = similarity.argsort(dim=1, descending=True)
        target = torch.arange(similarity.shape[0], device=similarity.device)[:, None]
        layer_r_at_1[layer] = float(
            (order[:, :1] == target).any(dim=1).float().mean()
        )
        layer_r_at_5[layer] = float(
            (order[:, : min(5, order.shape[1])] == target)
            .any(dim=1)
            .float()
            .mean()
        )
        paired = float((student * teacher).sum(dim=1).mean())
        base_paired = float((base * teacher).sum(dim=1).mean())
        layer_drift[layer] = base_paired - paired
        layer_paired_cosine[layer] = paired
        layer_base_cosine[layer] = base_paired
    metrics["layer_drift"] = layer_drift
    metrics["layer_paired_cosine"] = layer_paired_cosine
    metrics["layer_base_cosine"] = layer_base_cosine
    metrics["layer_r_at_1"] = layer_r_at_1
    metrics["layer_r_at_5"] = layer_r_at_5
    return metrics


class _D1Trainer(Phase16RepairTrainer):
    def __init__(self, *, phase18_config: Phase18Config, **kwargs: Any) -> None:
        self.phase18_config = phase18_config
        super().__init__(**kwargs)

    @torch.inference_mode()
    def _evaluate(
        self,
        module: nn.Module,
        records: Sequence[RGBTRecord],
        bank: RGBTeacherBank,
    ) -> dict[str, Any]:
        students = {layer: [] for layer in LAYER_NAMES}
        teachers = {layer: [] for layer in LAYER_NAMES}
        base_tir = {layer: [] for layer in LAYER_NAMES}
        for record in records:
            encoded = self.encode_tir(module, record)
            positive = bank[record.record_id]
            base = positive.base_tir or positive.foreground
            for layer in LAYER_NAMES:
                students[layer].append(encoded[layer].detach().cpu())
                teachers[layer].append(positive.foreground[layer].detach().cpu())
                base_tir[layer].append(base[layer].detach().cpu())
        return evaluate_retention_metrics(
            {layer: torch.stack(values) for layer, values in students.items()},
            {layer: torch.stack(values) for layer, values in teachers.items()},
            {layer: torch.stack(values) for layer, values in base_tir.items()},
        )

    def _passes_probe_gate(self, metrics: Mapping[str, Any]) -> bool:
        return Phase18RetentionProbe(
            output_root=self.output_root,
            config=self.phase18_config,
        ).evaluate_gate(metrics)["status"] == "PHASE_18_D1_GO"

    def _selection_key(self, metrics: Mapping[str, Any]) -> tuple[float, float, float]:
        baseline = self.phase18_config.c2_layer_drift or {}
        reductions = [
            (float(baseline[layer]) - float(metrics["layer_drift"][layer]))
            / max(abs(float(baseline[layer])), 1e-8)
            for layer in DISCRIMINATIVE_LAYERS
        ]
        return (
            sum(reductions) / len(reductions),
            float(metrics["mean_r_at_5"]),
            float(metrics["minimum_effective_rank_ratio_vs_base"]),
        )


class Phase18RetentionProbe:
    """Own D1 candidate fairness, gates, resume behavior, and selection."""

    def __init__(
        self,
        *,
        output_root: Path | str,
        config: Phase18Config | None = None,
        adapter_factory: Callable[[], nn.Module] | None = None,
        encode_tir: Callable[[nn.Module, RGBTRecord], Mapping[str, torch.Tensor]] | None = None,
        phase16_config: Phase16Config | None = None,
        adapter_state: Callable[[nn.Module], Mapping[str, torch.Tensor]] | None = None,
        load_adapter_state: Callable[[nn.Module, Mapping[str, torch.Tensor]], Any] | None = None,
        resume: bool = False,
    ) -> None:
        self.output_root = Path(output_root)
        self.config = config or Phase18Config()
        self.adapter_factory = adapter_factory
        self.encode_tir = encode_tir
        self.phase16_config = phase16_config or Phase16Config(full_steps=0)
        self.adapter_state = adapter_state
        self.load_adapter_state = load_adapter_state
        self.resume = bool(resume)

    def evaluate_gate(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        baseline = self.config.c2_layer_drift or {}
        drift = metrics["layer_drift"]
        reductions = {
            layer: (float(baseline[layer]) - float(drift[layer]))
            / max(abs(float(baseline[layer])), 1e-8)
            for layer in DISCRIMINATIVE_LAYERS
        }
        checks = {
            "all_layer_drift_reduced": all(
                float(drift[layer]) < float(baseline[layer])
                for layer in DISCRIMINATIVE_LAYERS
            ),
            "layer24_drift_reduction": reductions["24"]
            >= self.config.minimum_layer24_drift_reduction,
            "r1_retained": float(metrics["mean_r_at_1"])
            >= self.config.c2_mean_r_at_1 - self.config.maximum_r1_drop,
            "r5_retained": float(metrics["mean_r_at_5"])
            >= self.config.c2_mean_r_at_5 - self.config.maximum_r5_drop,
            "effective_rank_vs_base": float(
                metrics["minimum_effective_rank_ratio_vs_base"]
            )
            >= self.config.minimum_effective_rank_vs_base,
            "effective_rank_vs_teacher": float(
                metrics["minimum_effective_rank_ratio_vs_teacher"]
            )
            >= self.config.minimum_effective_rank_vs_teacher,
            "nonpaired_p95": float(metrics["maximum_nonpaired_cosine_p95_increase"])
            <= self.config.maximum_nonpaired_cosine_p95_increase,
            "paired_margin": float(metrics["minimum_paired_shuffled_margin"]) > 0.0,
        }
        return {
            "status": "PHASE_18_D1_GO" if all(checks.values()) else "PHASE_18_D1_NO_GO",
            "checks": checks,
            "drift_reduction": reductions,
        }

    def run(
        self,
        train_records: Sequence[RGBTRecord],
        dev_records: Sequence[RGBTRecord],
        rgb_teacher_bank: RGBTeacherBank,
    ) -> Phase18Result:
        if self.adapter_factory is None or self.encode_tir is None:
            raise RuntimeError("Phase18RetentionProbe.run requires adapter_factory and encode_tir")
        candidates = D1CandidateSpec.defaults()
        trainer = _D1Trainer(
            phase18_config=self.config,
            adapter_factory=self.adapter_factory,
            encode_tir=self.encode_tir,
            output_root=self.output_root,
            config=self.phase16_config,
            adapter_state=self.adapter_state,
            load_adapter_state=self.load_adapter_state,
            resume=self.resume,
            candidate_specs=candidates,
            objective_factory=lambda candidate: D1RetentionObjective(
                candidate,  # type: ignore[arg-type]
                temperature=self.phase16_config.temperature,
                margin=self.phase16_config.margin,
                retention_tolerance=self.config.retention_tolerance,
            ),
            stop_after_probe=True,
        )
        raw = trainer.run(train_records, dev_records, rgb_teacher_bank)
        decisions = {
            name: self.evaluate_gate(metrics) for name, metrics in raw.probe_metrics.items()
        }
        eligible = [name for name, decision in decisions.items() if decision["status"] == "PHASE_18_D1_GO"]
        selected = ""
        if eligible:
            selected = max(
                eligible,
                key=lambda name: trainer._selection_key(raw.probe_metrics[name]),
            )
        result = Phase18Result(
            status="PHASE_18_D1_GO" if selected else "PHASE_18_D1_NO_GO",
            selected_candidate=selected,
            completed_candidates=tuple(raw.completed_candidates),
            candidate_decisions=decisions,
            probe_metrics=raw.probe_metrics,
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "phase18_probe_summary.json").write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result
