from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from torch import nn

from .data import RGBTRecord


@dataclass(frozen=True)
class CheckpointArtifact:
    step: int
    fraction: float
    path: Path


@dataclass(frozen=True)
class Phase18FullConfig:
    checkpoint_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    expected_full_records: int = 24612
    expected_dev_records: int = 1024
    expected_official_records: int = 2032
    expected_official_pairs: int = 1115
    c2_layer_drift: Mapping[str, float] | None = None
    d1_probe_r1: float = 0.615234375
    d1_probe_r5: float = 0.84375
    maximum_r1_drop: float = 0.010
    maximum_r5_drop: float = 0.005
    minimum_layer24_drift_reduction: float = 0.25
    minimum_effective_rank_vs_base: float = 0.85
    minimum_effective_rank_vs_teacher: float = 0.75
    maximum_nonpaired_p95_increase: float = 0.10
    maximum_subgroup_degradation: float = 0.05

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
        if tuple(self.checkpoint_fractions) != (0.25, 0.50, 0.75, 1.0):
            raise ValueError("Phase 1.8 Full requires 25/50/75/100% checkpoints")


@dataclass(frozen=True)
class Phase18FullResult:
    status: str
    selected_checkpoint_step: int
    selected_checkpoint: str
    checkpoint_decisions: Mapping[str, Mapping[str, Any]]
    official_decision: Mapping[str, Any] | None
    fingerprint: str


@dataclass(frozen=True)
class Phase18TrainSelectionResult:
    status: str
    selected_checkpoint_step: int
    selected_checkpoint: str
    selected_adapter: str
    checkpoint_decisions: Mapping[str, Mapping[str, Any]]
    fingerprint: str


class Phase18FullBackend(Protocol):
    def fresh_adapter(self) -> nn.Module: ...

    def train_full(
        self,
        module: nn.Module,
        records: Sequence[RGBTRecord],
        checkpoint_dir: Path,
        fractions: Sequence[float],
    ) -> Sequence[CheckpointArtifact]: ...

    def load_checkpoint(
        self, module: nn.Module, checkpoint: CheckpointArtifact
    ) -> None: ...

    def evaluate(
        self, module: nn.Module, records: Sequence[RGBTRecord]
    ) -> Mapping[str, Any]: ...

    def release(
        self,
        module: nn.Module,
        checkpoint: CheckpointArtifact,
        output_path: Path,
        metadata: Mapping[str, Any],
    ) -> None: ...

    def validate_official(
        self, module: nn.Module, records: Sequence[RGBTRecord]
    ) -> Mapping[str, Any]: ...


class Phase18FullRunner:
    """Own full D1 training, sealed dev selection and one-shot official validation."""

    def __init__(
        self,
        *,
        backend: Phase18FullBackend,
        output_root: Path | str,
        config: Phase18FullConfig | None = None,
    ) -> None:
        self.backend = backend
        self.output_root = Path(output_root)
        self.config = config or Phase18FullConfig()

    @property
    def fingerprint_path(self) -> Path:
        return self.output_root / "run_fingerprint.json"

    def write_fingerprint(self, fingerprint: str) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        payload = {"fingerprint": str(fingerprint)}
        if self.fingerprint_path.is_file():
            current = json.loads(self.fingerprint_path.read_text(encoding="utf-8"))
            if current != payload:
                raise RuntimeError(
                    f"Phase 1.8 Full fingerprint mismatch: {current.get('fingerprint')} != {fingerprint}"
                )
            return
        self.fingerprint_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _pair_set(records: Sequence[RGBTRecord]) -> set[str]:
        return {record.image_pair_key for record in records}

    def _validate_records(
        self,
        full: Sequence[RGBTRecord],
        dev: Sequence[RGBTRecord],
        official: Sequence[RGBTRecord],
    ) -> None:
        expected = (
            ("full", len(full), self.config.expected_full_records),
            ("dev", len(dev), self.config.expected_dev_records),
            ("official", len(official), self.config.expected_official_records),
        )
        for label, actual, wanted in expected:
            if actual != wanted:
                raise ValueError(f"{label} record count mismatch: {actual} != {wanted}")
        official_pairs = len(self._pair_set(official))
        if official_pairs != self.config.expected_official_pairs:
            raise ValueError(
                f"official pair count mismatch: {official_pairs} != {self.config.expected_official_pairs}"
            )
        pair_sets = {
            "full": self._pair_set(full),
            "dev": self._pair_set(dev),
            "official": self._pair_set(official),
        }
        overlaps = {
            "full_dev": pair_sets["full"] & pair_sets["dev"],
            "full_official": pair_sets["full"] & pair_sets["official"],
            "dev_official": pair_sets["dev"] & pair_sets["official"],
        }
        if any(overlaps.values()):
            raise ValueError(
                "Phase 1.8 Full split leakage: "
                + ", ".join(f"{key}={len(value)}" for key, value in overlaps.items())
            )

    def _gate(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        baseline = self.config.c2_layer_drift or {}
        drift = metrics.get("layer_drift", {})
        reductions = {
            layer: (float(baseline[layer]) - float(drift.get(layer, float("inf"))))
            / max(abs(float(baseline[layer])), 1e-8)
            for layer in ("8", "16", "24")
        }
        checks = {
            "all_layer_drift_reduced": all(
                float(drift.get(layer, float("inf"))) < float(baseline[layer])
                for layer in ("8", "16", "24")
            ),
            "layer24_drift_reduction": reductions["24"]
            >= self.config.minimum_layer24_drift_reduction,
            "r1_retained": float(metrics.get("mean_r_at_1", -1.0))
            >= self.config.d1_probe_r1 - self.config.maximum_r1_drop,
            "r5_retained": float(metrics.get("mean_r_at_5", -1.0))
            >= self.config.d1_probe_r5 - self.config.maximum_r5_drop,
            "effective_rank_vs_base": float(
                metrics.get("minimum_effective_rank_ratio_vs_base", -1.0)
            )
            >= self.config.minimum_effective_rank_vs_base,
            "effective_rank_vs_teacher": float(
                metrics.get("minimum_effective_rank_ratio_vs_teacher", -1.0)
            )
            >= self.config.minimum_effective_rank_vs_teacher,
            "nonpaired_p95": float(
                metrics.get("maximum_nonpaired_cosine_p95_increase", float("inf"))
            )
            <= self.config.maximum_nonpaired_p95_increase,
            "paired_margin": float(
                metrics.get("minimum_paired_shuffled_margin", -1.0)
            )
            > 0.0,
        }
        return {"passed": all(checks.values()), "checks": checks, "reductions": reductions}

    @staticmethod
    def _selection_key(
        checkpoint: CheckpointArtifact, metrics: Mapping[str, Any]
    ) -> tuple[float, float, float, float, int]:
        layer_r5 = metrics.get("layer_r_at_5", {})
        worst_r5 = min(float(layer_r5[layer]) for layer in ("8", "16", "24"))
        return (
            worst_r5,
            -float(metrics["layer_drift"]["24"]),
            min(
                float(metrics["minimum_effective_rank_ratio_vs_base"]),
                float(metrics["minimum_effective_rank_ratio_vs_teacher"]),
            ),
            float(metrics["minimum_paired_shuffled_margin"]),
            -int(checkpoint.step),
        )

    def _select_checkpoint(
        self,
        eligible: Sequence[tuple[CheckpointArtifact, Mapping[str, Any]]],
    ) -> tuple[CheckpointArtifact, Mapping[str, Any]]:
        if not eligible:
            raise ValueError("cannot select from an empty checkpoint set")

        def worst_r5(metrics: Mapping[str, Any]) -> float:
            values = metrics["layer_r_at_5"]
            return min(float(values[layer]) for layer in ("8", "16", "24"))

        def secondary(
            checkpoint: CheckpointArtifact, metrics: Mapping[str, Any]
        ) -> tuple[float, float, float, int]:
            return (
                -float(metrics["layer_drift"]["24"]),
                min(
                    float(metrics["minimum_effective_rank_ratio_vs_base"]),
                    float(metrics["minimum_effective_rank_ratio_vs_teacher"]),
                ),
                float(metrics["minimum_paired_shuffled_margin"]),
                -int(checkpoint.step),
            )

        best_checkpoint, best_metrics = eligible[0]
        tolerance = 0.005
        for checkpoint, metrics in eligible[1:]:
            candidate_r5 = worst_r5(metrics)
            current_r5 = worst_r5(best_metrics)
            if candidate_r5 > current_r5 + tolerance:
                best_checkpoint, best_metrics = checkpoint, metrics
                continue
            if current_r5 > candidate_r5 + tolerance:
                continue
            if secondary(checkpoint, metrics) > secondary(best_checkpoint, best_metrics):
                best_checkpoint, best_metrics = checkpoint, metrics
        return best_checkpoint, best_metrics

    def _official_gate(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        representation = self._gate(metrics)
        checks = dict(representation["checks"])
        checks.update(
            {
                "official_records_complete": int(metrics.get("completed_records", -1))
                == self.config.expected_official_records,
                "official_pairs_complete": int(metrics.get("completed_pairs", -1))
                == self.config.expected_official_pairs,
                "official_skipped_zero": int(metrics.get("skipped_records", -1)) == 0,
                "paired_margin_bootstrap_lower_positive": float(
                    metrics.get("paired_margin_bootstrap_lower", -1.0)
                )
                > 0.0,
                "subgroup_degradation_bounded": float(
                    metrics.get("maximum_subgroup_degradation", float("inf"))
                )
                <= self.config.maximum_subgroup_degradation,
                "safety_passed": bool(metrics.get("safety_passed", False)),
            }
        )
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "reductions": representation["reductions"],
        }

    def _write_result(self, result: Phase18FullResult) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.output_root.joinpath("run_summary.json").write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    def _write_selection(self, result: Phase18TrainSelectionResult) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        target = self.output_root / "train_selection_summary.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    def _train_select_impl(
        self,
        full_records: Sequence[RGBTRecord],
        dev_records: Sequence[RGBTRecord],
        official_records: Sequence[RGBTRecord],
        *,
        fingerprint: str,
    ) -> tuple[Phase18TrainSelectionResult, nn.Module]:
        self._validate_records(full_records, dev_records, official_records)
        self.write_fingerprint(fingerprint)
        module = self.backend.fresh_adapter()
        checkpoints = tuple(
            self.backend.train_full(
                module,
                full_records,
                self.output_root / "checkpoints",
                self.config.checkpoint_fractions,
            )
        )
        if tuple(round(item.fraction, 2) for item in checkpoints) != self.config.checkpoint_fractions:
            raise RuntimeError("Phase 1.8 Full backend did not emit the four frozen checkpoints")
        decisions: dict[str, dict[str, Any]] = {}
        eligible: list[tuple[CheckpointArtifact, Mapping[str, Any]]] = []
        for checkpoint in checkpoints:
            self.backend.load_checkpoint(module, checkpoint)
            metrics = dict(self.backend.evaluate(module, dev_records))
            gate = self._gate(metrics)
            decisions[str(checkpoint.path)] = {
                "step": checkpoint.step,
                "fraction": checkpoint.fraction,
                "metrics": metrics,
                "gate": gate,
            }
            if gate["passed"]:
                eligible.append((checkpoint, metrics))
        if not eligible:
            result = Phase18TrainSelectionResult(
                status="PHASE_18_FULL_NO_GO",
                selected_checkpoint_step=0,
                selected_checkpoint="",
                selected_adapter="",
                checkpoint_decisions=decisions,
                fingerprint=fingerprint,
            )
            self._write_selection(result)
            return result, module
        selected, selected_metrics = self._select_checkpoint(eligible)
        self.backend.load_checkpoint(module, selected)
        release_path = self.output_root / "selected_adapter" / "adapter.pt"
        self.backend.release(
            module,
            selected,
            release_path,
            {
                "fingerprint": fingerprint,
                "selection_source": "repair_dev_only",
                "dev_metrics": selected_metrics,
            },
        )
        result = Phase18TrainSelectionResult(
            status="PHASE_18_TRAIN_SELECT_READY",
            selected_checkpoint_step=selected.step,
            selected_checkpoint=str(selected.path),
            selected_adapter=str(release_path),
            checkpoint_decisions=decisions,
            fingerprint=fingerprint,
        )
        self._write_selection(result)
        return result, module

    def train_select(
        self,
        full_records: Sequence[RGBTRecord],
        dev_records: Sequence[RGBTRecord],
        official_records: Sequence[RGBTRecord],
        *,
        fingerprint: str,
    ) -> Phase18TrainSelectionResult:
        """Train and select an adapter without opening official validation."""
        result, _ = self._train_select_impl(
            full_records, dev_records, official_records, fingerprint=fingerprint
        )
        return result

    def complete_official(
        self,
        selection: Phase18TrainSelectionResult,
        official_metrics: Mapping[str, Any],
    ) -> Phase18FullResult:
        """Finalize the sealed official decision from a persisted selection."""
        if selection.status != "PHASE_18_TRAIN_SELECT_READY":
            raise RuntimeError("official validation requires a ready train selection")
        current = json.loads(self.fingerprint_path.read_text(encoding="utf-8"))
        if current.get("fingerprint") != selection.fingerprint:
            raise RuntimeError("official validation fingerprint mismatch")
        official_gate = self._official_gate(official_metrics)
        result = Phase18FullResult(
            status="PHASE_18_FULL_GO" if official_gate["passed"] else "PHASE_18_FULL_NO_GO",
            selected_checkpoint_step=selection.selected_checkpoint_step,
            selected_checkpoint=selection.selected_checkpoint,
            checkpoint_decisions=selection.checkpoint_decisions,
            official_decision={"metrics": dict(official_metrics), "gate": official_gate},
            fingerprint=selection.fingerprint,
        )
        self._write_result(result)
        return result

    def run(
        self,
        full_records: Sequence[RGBTRecord],
        dev_records: Sequence[RGBTRecord],
        official_records: Sequence[RGBTRecord],
        *,
        fingerprint: str,
    ) -> Phase18FullResult:
        selection, module = self._train_select_impl(
            full_records, dev_records, official_records, fingerprint=fingerprint
        )
        if selection.status != "PHASE_18_TRAIN_SELECT_READY":
            result = Phase18FullResult(
                status="PHASE_18_FULL_NO_GO",
                selected_checkpoint_step=0,
                selected_checkpoint="",
                checkpoint_decisions=selection.checkpoint_decisions,
                official_decision=None,
                fingerprint=fingerprint,
            )
            self._write_result(result)
            return result
        official_metrics = dict(self.backend.validate_official(module, official_records))
        return self.complete_official(selection, official_metrics)
