from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from torch import nn

from .data import RGBTRecord
from .phase18_full import CheckpointArtifact


@dataclass(frozen=True)
class D2FullConfig:
    """Hard boundaries for one D2_G025 full-train and sealed validation run."""

    checkpoint_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    expected_full_records: int = 23391
    expected_semantic_records: int = 1024
    expected_semantic_pairs: int = 1024
    expected_multiquery_records: int = 1221
    expected_multiquery_pairs: int = 512
    expected_official_records: int = 2032
    expected_official_pairs: int = 1115
    minimum_dev_rank_vs_base: float = 0.85
    minimum_rank_gain_each_dev: float = 0.02
    maximum_semantic_r5_drop: float = 0.005
    maximum_multiquery_r5_drop: float = 0.010
    maximum_nonpaired_p95_delta: float = 0.02
    maximum_official_absolute_drift: float = 0.02
    minimum_official_rank_vs_base: float = 0.85
    minimum_official_rank_vs_teacher: float = 0.75
    maximum_official_nonpaired_p95_increase: float = 0.10
    maximum_subgroup_degradation: float = 0.05
    official_layer_r5_baseline: Mapping[str, float] | None = None
    official_layer_r5_maximum_drop: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if tuple(self.checkpoint_fractions) != (0.25, 0.50, 0.75, 1.0):
            raise ValueError("D2 Full requires 25/50/75/100% checkpoints")
        if self.official_layer_r5_baseline is None:
            object.__setattr__(
                self,
                "official_layer_r5_baseline",
                {"8": 0.9134, "16": 0.9070, "24": 0.7539},
            )
        if self.official_layer_r5_maximum_drop is None:
            object.__setattr__(
                self,
                "official_layer_r5_maximum_drop",
                {"8": 0.005, "16": 0.005, "24": 0.010},
            )


@dataclass(frozen=True)
class D2FullSelectionResult:
    status: str
    selected_checkpoint_step: int
    selected_checkpoint: str
    selected_adapter: str
    checkpoint_decisions: Mapping[str, Mapping[str, Any]]
    fingerprint: str


@dataclass(frozen=True)
class D2FullResult:
    status: str
    selected_checkpoint_step: int
    selected_checkpoint: str
    checkpoint_decisions: Mapping[str, Mapping[str, Any]]
    official_decision: Mapping[str, Any] | None
    fingerprint: str


class D2FullBackend(Protocol):
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


class Phase19D2FullRunner:
    """Own D2_G025 full training, dual-dev selection, and sealed official val."""

    def __init__(
        self,
        *,
        backend: D2FullBackend,
        output_root: Path | str,
        baseline_metrics: Mapping[str, Mapping[str, float]],
        config: D2FullConfig | None = None,
    ) -> None:
        self.backend = backend
        self.output_root = Path(output_root)
        self.baseline_metrics = {
            split: dict(values) for split, values in baseline_metrics.items()
        }
        if set(self.baseline_metrics) != {"semantic", "multiquery"}:
            raise ValueError("D2 Full baseline must contain semantic and multiquery")
        self.config = config or D2FullConfig()

    @property
    def fingerprint_path(self) -> Path:
        return self.output_root / "run_fingerprint.json"

    def write_fingerprint(self, fingerprint: str) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        payload = {"fingerprint": str(fingerprint)}
        if self.fingerprint_path.is_file():
            current = json.loads(self.fingerprint_path.read_text(encoding="utf-8"))
            if current != payload:
                raise RuntimeError("Phase 1.9-D2 Full fingerprint mismatch")
            return
        self.fingerprint_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _pair_set(records: Sequence[RGBTRecord]) -> set[str]:
        return {record.image_pair_key for record in records}

    def validate_records(
        self,
        full: Sequence[RGBTRecord],
        semantic: Sequence[RGBTRecord],
        multiquery: Sequence[RGBTRecord],
        official: Sequence[RGBTRecord],
    ) -> None:
        expected = (
            ("full", len(full), self.config.expected_full_records, len(self._pair_set(full)), None),
            (
                "semantic",
                len(semantic),
                self.config.expected_semantic_records,
                len(self._pair_set(semantic)),
                self.config.expected_semantic_pairs,
            ),
            (
                "multiquery",
                len(multiquery),
                self.config.expected_multiquery_records,
                len(self._pair_set(multiquery)),
                self.config.expected_multiquery_pairs,
            ),
            (
                "official",
                len(official),
                self.config.expected_official_records,
                len(self._pair_set(official)),
                self.config.expected_official_pairs,
            ),
        )
        for label, record_count, expected_records, pair_count, expected_pairs in expected:
            if record_count != expected_records:
                raise ValueError(
                    f"{label} record count mismatch: {record_count} != {expected_records}"
                )
            if expected_pairs is not None and pair_count != expected_pairs:
                raise ValueError(
                    f"{label} pair count mismatch: {pair_count} != {expected_pairs}"
                )
        sets = {
            "full": self._pair_set(full),
            "semantic": self._pair_set(semantic),
            "multiquery": self._pair_set(multiquery),
            "official": self._pair_set(official),
        }
        overlaps = {
            f"{left}_{right}": sets[left] & sets[right]
            for index, left in enumerate(sets)
            for right in list(sets)[index + 1 :]
        }
        if any(overlaps.values()):
            raise ValueError(
                "Phase 1.9-D2 Full split leakage: "
                + ", ".join(f"{key}={len(value)}" for key, value in overlaps.items())
            )

    def dev_gate(
        self,
        semantic: Mapping[str, Any],
        multiquery: Mapping[str, Any],
    ) -> dict[str, Any]:
        metrics = {"semantic": semantic, "multiquery": multiquery}
        rank_delta = {
            split: float(values["minimum_effective_rank_ratio_vs_base"])
            - float(self.baseline_metrics[split]["minimum_effective_rank_ratio_vs_base"])
            for split, values in metrics.items()
        }
        r5_drop = {
            split: float(self.baseline_metrics[split]["mean_r_at_5"])
            - float(values["mean_r_at_5"])
            for split, values in metrics.items()
        }
        p95_delta = {
            split: float(values["maximum_nonpaired_cosine_p95_increase"])
            - float(
                self.baseline_metrics[split][
                    "maximum_nonpaired_cosine_p95_increase"
                ]
            )
            for split, values in metrics.items()
        }
        checks = {
            "both_dev_absolute_rank": all(
                float(values["minimum_effective_rank_ratio_vs_base"])
                >= self.config.minimum_dev_rank_vs_base
                for values in metrics.values()
            ),
            "both_dev_rank_gain": all(
                value >= self.config.minimum_rank_gain_each_dev
                for value in rank_delta.values()
            ),
            "semantic_r5_retained": r5_drop["semantic"]
            <= self.config.maximum_semantic_r5_drop,
            "multiquery_r5_retained": r5_drop["multiquery"]
            <= self.config.maximum_multiquery_r5_drop,
            "nonpaired_p95_controlled": all(
                value <= self.config.maximum_nonpaired_p95_delta
                for value in p95_delta.values()
            ),
            "paired_margin_positive": all(
                float(values["minimum_paired_shuffled_margin"]) > 0.0
                for values in metrics.values()
            ),
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "rank_delta": rank_delta,
            "r5_drop": r5_drop,
            "nonpaired_p95_delta": p95_delta,
        }

    def official_gate(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        layer_r5 = metrics.get("layer_r_at_5", {})
        baseline = self.config.official_layer_r5_baseline or {}
        maximum_drop = self.config.official_layer_r5_maximum_drop or {}
        checks = {
            "official_records_complete": int(metrics.get("completed_records", -1))
            == self.config.expected_official_records,
            "official_pairs_complete": int(metrics.get("completed_pairs", -1))
            == self.config.expected_official_pairs,
            "official_skipped_zero": int(metrics.get("skipped_records", -1)) == 0,
            "absolute_drift_bounded": float(
                metrics.get("maximum_absolute_layer_drift", float("inf"))
            )
            <= self.config.maximum_official_absolute_drift,
            "effective_rank_vs_base": float(
                metrics.get("minimum_effective_rank_ratio_vs_base", -1.0)
            )
            >= self.config.minimum_official_rank_vs_base,
            "effective_rank_vs_teacher": float(
                metrics.get("minimum_effective_rank_ratio_vs_teacher", -1.0)
            )
            >= self.config.minimum_official_rank_vs_teacher,
            "strict_r5_retained": all(
                float(layer_r5.get(layer, -1.0))
                >= float(baseline[layer]) - float(maximum_drop[layer])
                for layer in ("8", "16", "24")
            ),
            "nonpaired_p95": float(
                metrics.get("maximum_nonpaired_cosine_p95_increase", float("inf"))
            )
            <= self.config.maximum_official_nonpaired_p95_increase,
            "paired_margin_positive": float(
                metrics.get("minimum_paired_shuffled_margin", -1.0)
            )
            > 0.0,
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
        return {"passed": all(checks.values()), "checks": checks}

    @staticmethod
    def _selection_key(
        checkpoint: CheckpointArtifact,
        semantic: Mapping[str, Any],
        multiquery: Mapping[str, Any],
    ) -> tuple[float, float, float, int]:
        return (
            min(
                float(semantic["minimum_effective_rank_ratio_vs_base"]),
                float(multiquery["minimum_effective_rank_ratio_vs_base"]),
            ),
            float(semantic["mean_r_at_5"]) + float(multiquery["mean_r_at_5"]),
            min(
                float(semantic["minimum_paired_shuffled_margin"]),
                float(multiquery["minimum_paired_shuffled_margin"]),
            ),
            -int(checkpoint.step),
        )

    def _write_json(self, name: str, payload: Mapping[str, Any]) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        target = self.output_root / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    def _train_select_impl(
        self,
        full: Sequence[RGBTRecord],
        semantic: Sequence[RGBTRecord],
        multiquery: Sequence[RGBTRecord],
        official: Sequence[RGBTRecord],
        *,
        fingerprint: str,
    ) -> tuple[D2FullSelectionResult, nn.Module]:
        self.validate_records(full, semantic, multiquery, official)
        self.write_fingerprint(fingerprint)
        module = self.backend.fresh_adapter()
        checkpoints = tuple(
            self.backend.train_full(
                module,
                full,
                self.output_root / "checkpoints",
                self.config.checkpoint_fractions,
            )
        )
        if tuple(round(item.fraction, 2) for item in checkpoints) != self.config.checkpoint_fractions:
            raise RuntimeError("D2 Full backend did not emit 25/50/75/100% checkpoints")
        decisions: dict[str, dict[str, Any]] = {}
        eligible: list[tuple[CheckpointArtifact, Mapping[str, Any], Mapping[str, Any]]] = []
        for checkpoint in checkpoints:
            self.backend.load_checkpoint(module, checkpoint)
            semantic_metrics = dict(self.backend.evaluate(module, semantic))
            multiquery_metrics = dict(self.backend.evaluate(module, multiquery))
            gate = self.dev_gate(semantic_metrics, multiquery_metrics)
            decisions[str(checkpoint.path)] = {
                "step": checkpoint.step,
                "fraction": checkpoint.fraction,
                "semantic": semantic_metrics,
                "multiquery": multiquery_metrics,
                "gate": gate,
            }
            if gate["passed"]:
                eligible.append((checkpoint, semantic_metrics, multiquery_metrics))
        if not eligible:
            result = D2FullSelectionResult(
                status="PHASE_19_D2_FULL_NO_GO",
                selected_checkpoint_step=0,
                selected_checkpoint="",
                selected_adapter="",
                checkpoint_decisions=decisions,
                fingerprint=fingerprint,
            )
            self._write_json("train_selection_summary.json", asdict(result))
            return result, module
        selected, semantic_metrics, multiquery_metrics = max(
            eligible,
            key=lambda row: self._selection_key(row[0], row[1], row[2]),
        )
        self.backend.load_checkpoint(module, selected)
        release_path = self.output_root / "selected_adapter" / "adapter.pt"
        self.backend.release(
            module,
            selected,
            release_path,
            {
                "fingerprint": fingerprint,
                "selection_source": "sealed_semantic_plus_multiquery_dev",
                "semantic_metrics": semantic_metrics,
                "multiquery_metrics": multiquery_metrics,
                "geometry_lambda": 0.25,
            },
        )
        result = D2FullSelectionResult(
            status="PHASE_19_D2_TRAIN_SELECT_READY",
            selected_checkpoint_step=selected.step,
            selected_checkpoint=str(selected.path),
            selected_adapter=str(release_path),
            checkpoint_decisions=decisions,
            fingerprint=fingerprint,
        )
        self._write_json("train_selection_summary.json", asdict(result))
        return result, module

    def train_select(
        self,
        full: Sequence[RGBTRecord],
        semantic: Sequence[RGBTRecord],
        multiquery: Sequence[RGBTRecord],
        official: Sequence[RGBTRecord],
        *,
        fingerprint: str,
    ) -> D2FullSelectionResult:
        result, _ = self._train_select_impl(
            full, semantic, multiquery, official, fingerprint=fingerprint
        )
        return result

    def complete_official(
        self,
        selection: D2FullSelectionResult,
        official_metrics: Mapping[str, Any],
    ) -> D2FullResult:
        if selection.status != "PHASE_19_D2_TRAIN_SELECT_READY":
            raise RuntimeError("D2 official val requires a ready train selection")
        current = json.loads(self.fingerprint_path.read_text(encoding="utf-8"))
        if current.get("fingerprint") != selection.fingerprint:
            raise RuntimeError("D2 official val fingerprint mismatch")
        gate = self.official_gate(official_metrics)
        result = D2FullResult(
            status=("PHASE_19_D2_FULL_GO" if gate["passed"] else "PHASE_19_D2_FULL_NO_GO"),
            selected_checkpoint_step=selection.selected_checkpoint_step,
            selected_checkpoint=selection.selected_checkpoint,
            checkpoint_decisions=selection.checkpoint_decisions,
            official_decision={"metrics": dict(official_metrics), "gate": gate},
            fingerprint=selection.fingerprint,
        )
        self._write_json("run_summary.json", asdict(result))
        return result

    def run(
        self,
        full: Sequence[RGBTRecord],
        semantic: Sequence[RGBTRecord],
        multiquery: Sequence[RGBTRecord],
        official: Sequence[RGBTRecord],
        *,
        fingerprint: str,
    ) -> D2FullResult:
        selection, module = self._train_select_impl(
            full, semantic, multiquery, official, fingerprint=fingerprint
        )
        if selection.status != "PHASE_19_D2_TRAIN_SELECT_READY":
            result = D2FullResult(
                status="PHASE_19_D2_FULL_NO_GO",
                selected_checkpoint_step=0,
                selected_checkpoint="",
                checkpoint_decisions=selection.checkpoint_decisions,
                official_decision=None,
                fingerprint=fingerprint,
            )
            self._write_json("run_summary.json", asdict(result))
            return result
        official_metrics = dict(self.backend.validate_official(module, official))
        return self.complete_official(selection, official_metrics)
