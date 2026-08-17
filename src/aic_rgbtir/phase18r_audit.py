from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Phase18RAuditConfig:
    expected_records: int = 2032
    expected_pairs: int = 1115
    seed: int = 20260812
    bootstrap_iterations: int = 500
    minimum_effective_rank_vs_base: float = 0.85
    maximum_global_absolute_drift: float = 0.02
    expected_adapter_tensors: int = 108
    layers: tuple[str, ...] = ("8", "16", "24")
    group_fields: tuple[str, ...] = (
        "source_dataset",
        "illumination",
        "weather",
        "object_size",
        "occlusion",
        "black_border_severity",
        "weak_alignment_risk",
    )


@dataclass(frozen=True)
class Phase18RAuditResult:
    decision: str
    gate_issue: bool
    model_issue: bool
    complete: bool
    findings: tuple[str, ...]
    input_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["findings"] = list(self.findings)
        return payload


class Phase18RAuditor:
    def __init__(
        self,
        *,
        input_root: Path | str,
        output_root: Path | str,
        config: Phase18RAuditConfig | None = None,
    ) -> None:
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.config = config or Phase18RAuditConfig()

    @staticmethod
    def retrieval_audit(
        records: Sequence[Mapping[str, Any]], *, layers: Sequence[str]
    ) -> dict[str, Any]:
        if not records:
            raise ValueError("retrieval audit requires records")
        pair_ids = [str(record["image_pair_key"]) for record in records]
        result: dict[str, Any] = {}
        for layer in layers:
            rgb = F.normalize(
                torch.stack([record["pooled"]["rgb"][layer] for record in records]).float(),
                dim=1,
            )
            layer_result: dict[str, Any] = {}
            for variant in ("base", "adapted"):
                tir = F.normalize(
                    torch.stack(
                        [record["pooled"][variant][layer] for record in records]
                    ).float(),
                    dim=1,
                )
                similarity = tir @ rgb.T
                order = similarity.argsort(dim=1, descending=True)
                exact = torch.arange(len(records))
                same_pair = torch.tensor(
                    [
                        [left == right for right in pair_ids]
                        for left in pair_ids
                    ],
                    dtype=torch.bool,
                )

                def recall(positives: torch.Tensor, k: int) -> float:
                    selected = order[:, : min(k, len(records))]
                    return float(positives.gather(1, selected).any(dim=1).float().mean())

                exact_positive = torch.zeros_like(same_pair)
                exact_positive[exact, exact] = True
                pair_order: list[str] = []
                for pair_id in pair_ids:
                    if pair_id not in pair_order:
                        pair_order.append(pair_id)
                rgb_pair = torch.stack(
                    [rgb[[value == pair for value in pair_ids]].mean(dim=0) for pair in pair_order]
                )
                tir_pair = torch.stack(
                    [tir[[value == pair for value in pair_ids]].mean(dim=0) for pair in pair_order]
                )
                rgb_pair = F.normalize(rgb_pair, dim=1)
                tir_pair = F.normalize(tir_pair, dim=1)
                pair_ranking = (tir_pair @ rgb_pair.T).argsort(dim=1, descending=True)
                pair_target = torch.arange(len(pair_order))[:, None]
                layer_result[variant] = {
                    "record_exact": {
                        "r_at_1": recall(exact_positive, 1),
                        "r_at_5": recall(exact_positive, 5),
                    },
                    "pair_multi_positive": {
                        "r_at_1": recall(same_pair, 1),
                        "r_at_5": recall(same_pair, 5),
                    },
                    "pair_mean": {
                        "pair_count": len(pair_order),
                        "r_at_1": float(
                            (pair_ranking[:, :1] == pair_target).any(dim=1).float().mean()
                        ),
                        "r_at_5": float(
                            (pair_ranking[:, : min(5, len(pair_order))] == pair_target)
                            .any(dim=1)
                            .float()
                            .mean()
                        ),
                    },
                    "paired_cosine": float(similarity.diag().mean()),
                }
            result[str(layer)] = layer_result
        return result

    @staticmethod
    def spectrum_audit(
        records: Sequence[Mapping[str, Any]], *, layers: Sequence[str]
    ) -> dict[str, Any]:
        if not records:
            raise ValueError("spectrum audit requires records")
        pair_ids = [str(record["image_pair_key"]) for record in records]
        pair_order = list(dict.fromkeys(pair_ids))

        def metrics(matrix: torch.Tensor) -> dict[str, Any]:
            matrix = matrix.float()
            if not torch.isfinite(matrix).all():
                raise ValueError("spectrum input contains non-finite values")
            centered = matrix - matrix.mean(dim=0, keepdim=True)
            singular = torch.linalg.svdvals(centered)
            energy = singular.square()
            total = energy.sum()
            if float(total) <= 1e-12:
                effective_rank = 0.0
                participation = 0.0
                fractions = {str(k): 0.0 for k in (2, 8, 32, 64, 128)}
            else:
                probability = energy / total
                effective_rank = float(
                    torch.exp(
                        -(probability * probability.clamp_min(1e-12).log()).sum()
                    )
                )
                participation = float(
                    total.square() / energy.square().sum().clamp_min(1e-12)
                )
                fractions = {
                    str(k): float(energy[: min(k, energy.numel())].sum() / total)
                    for k in (2, 8, 32, 64, 128)
                }
            return {
                "sample_count": int(matrix.shape[0]),
                "feature_dimension": int(matrix.shape[1]),
                "effective_rank": effective_rank,
                "participation_ratio": participation,
                "top_energy_fraction": fractions,
            }

        result: dict[str, Any] = {}
        for layer in layers:
            layer_result: dict[str, Any] = {}
            for variant in ("rgb", "base", "adapted"):
                record_matrix = torch.stack(
                    [record["pooled"][variant][layer] for record in records]
                )
                pair_matrix = torch.stack(
                    [
                        record_matrix[
                            torch.tensor([value == pair for value in pair_ids])
                        ].mean(dim=0)
                        for pair in pair_order
                    ]
                )
                layer_result[variant] = {
                    "record_level": metrics(record_matrix),
                    "pair_mean_level": metrics(pair_matrix),
                }
            result[str(layer)] = layer_result
        return result

    @staticmethod
    def subgroup_excess_drift(
        rows: Sequence[Mapping[str, Any]],
        *,
        group_fields: Sequence[str],
        bootstrap_iterations: int,
        seed: int,
    ) -> dict[str, Any]:
        if not rows:
            raise ValueError("subgroup audit requires rows")
        prepared = [
            {
                **dict(row),
                "absolute_drift": float(row["adapted_alignment_loss"])
                - float(row["base_alignment_loss"]),
            }
            for row in rows
        ]
        pair_ids = list(dict.fromkeys(str(row["image_pair_key"]) for row in prepared))
        pair_rows = {
            pair_id: [row for row in prepared if str(row["image_pair_key"]) == pair_id]
            for pair_id in pair_ids
        }
        global_drift = float(np.mean([row["absolute_drift"] for row in prepared]))
        rng = np.random.default_rng(seed)
        sampled_pairs = [
            rng.choice(pair_ids, size=len(pair_ids), replace=True).tolist()
            for _ in range(int(bootstrap_iterations))
        ]
        groups: list[dict[str, Any]] = []
        for field in group_fields:
            values = sorted({str(row.get(field, "not_available")) for row in prepared})
            for value in values:
                group = [
                    row
                    for row in prepared
                    if str(row.get(field, "not_available")) == value
                ]
                group_drift = float(np.mean([row["absolute_drift"] for row in group]))
                bootstrap_values: list[float] = []
                for sample in sampled_pairs:
                    sampled_rows = [row for pair in sample for row in pair_rows[pair]]
                    sampled_group = [
                        row
                        for row in sampled_rows
                        if str(row.get(field, "not_available")) == value
                    ]
                    if not sampled_group:
                        continue
                    sampled_global = float(
                        np.mean([row["absolute_drift"] for row in sampled_rows])
                    )
                    sampled_group_drift = float(
                        np.mean([row["absolute_drift"] for row in sampled_group])
                    )
                    bootstrap_values.append(sampled_group_drift - sampled_global)
                if bootstrap_values:
                    low, high = np.quantile(bootstrap_values, [0.025, 0.975])
                else:
                    low = high = group_drift - global_drift
                groups.append(
                    {
                        "group_field": str(field),
                        "group_value": value,
                        "record_count": len(group),
                        "unique_pair_count": len(
                            {str(row["image_pair_key"]) for row in group}
                        ),
                        "absolute_drift": group_drift,
                        "excess_drift": group_drift - global_drift,
                        "excess_drift_ci95": [float(low), float(high)],
                    }
                )
        return {
            "record_count": len(prepared),
            "unique_pair_count": len(pair_ids),
            "global_absolute_drift": global_drift,
            "groups": groups,
        }

    def run(self) -> Phase18RAuditResult:
        cache_path = self.input_root / "official_val" / "embedding_cache.pt"
        rows_path = self.input_root / "official_val" / "per_record_metrics.jsonl"
        safety_path = self.input_root / "official_val" / "safety_equivalence.json"
        adapter_path = self.input_root / "selected_adapter" / "adapter.pt"
        required = (cache_path, rows_path, safety_path, adapter_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing Phase 1.8R audit inputs: " + ", ".join(missing))

        before = {str(path.relative_to(self.input_root)): self._sha256(path) for path in required}
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        records = list(cache["records"])
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if {str(row["record_id"]) for row in rows} != {
            str(record["record_id"]) for record in records
        }:
            raise ValueError("embedding cache and per-record metrics disagree")
        pair_count = len({str(record["image_pair_key"]) for record in records})
        complete = (
            len(records) == self.config.expected_records
            and len(rows) == self.config.expected_records
            and pair_count == self.config.expected_pairs
        )

        retrieval = self.retrieval_audit(records, layers=self.config.layers)
        spectrum = self.spectrum_audit(records, layers=self.config.layers)
        subgroups = self.subgroup_excess_drift(
            rows,
            group_fields=tuple(
                field for field in self.config.group_fields if any(field in row for row in rows)
            ),
            bootstrap_iterations=self.config.bootstrap_iterations,
            seed=self.config.seed,
        )
        safety = json.loads(safety_path.read_text(encoding="utf-8"))
        adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
        adapter_state = adapter.get("tir_adapter", adapter.get("adapter", adapter))
        adapter_finite = bool(adapter_state) and all(
            torch.is_tensor(value) and bool(torch.isfinite(value).all())
            for value in adapter_state.values()
        )
        historical_checks = {str(key): bool(value) for key, value in safety["checks"].items()}
        false_checks = sorted(key for key, value in historical_checks.items() if not value)
        repairable_false_checks = set(false_checks).issubset({"no_trainable_parameters"})
        safety_audit = {
            "historical_passed": bool(safety.get("passed", False)),
            "historical_false_checks": false_checks,
            "repairable_inference_freeze_issue": bool(false_checks)
            and repairable_false_checks,
            "adapter_tensor_count": len(adapter_state),
            "adapter_tensor_count_exact": len(adapter_state)
            == self.config.expected_adapter_tensors,
            "adapter_all_finite": adapter_finite,
            "functional_checks_other_than_freeze_passed": all(
                value
                for key, value in historical_checks.items()
                if key != "no_trainable_parameters"
            ),
        }
        gate_issue = bool(safety_audit["repairable_inference_freeze_issue"])

        rank_failures: list[str] = []
        rank_ratios: dict[str, Any] = {}
        for layer in self.config.layers:
            rank_ratios[layer] = {}
            for level in ("record_level", "pair_mean_level"):
                base_rank = float(spectrum[layer]["base"][level]["effective_rank"])
                adapted_rank = float(spectrum[layer]["adapted"][level]["effective_rank"])
                ratio = adapted_rank / max(base_rank, 1e-12)
                rank_ratios[layer][level] = ratio
                if ratio < self.config.minimum_effective_rank_vs_base:
                    rank_failures.append(f"layer_{layer}:{level}")
        global_drift = float(subgroups["global_absolute_drift"])
        model_issue = bool(rank_failures) or (
            global_drift > self.config.maximum_global_absolute_drift
        )
        findings: list[str] = []
        if not complete:
            findings.append(
                f"completion mismatch: records={len(records)}/{self.config.expected_records}, "
                f"pairs={pair_count}/{self.config.expected_pairs}"
            )
        if gate_issue:
            findings.append(
                "历史安全失败仅来自 official-val 前未关闭训练状态的冻结契约错误"
            )
        if rank_failures:
            findings.append("记录级有效秩未达门槛：" + ", ".join(rank_failures))
        if global_drift > self.config.maximum_global_absolute_drift:
            findings.append(
                f"全局绝对对齐漂移 {global_drift:.6f} 超过 "
                f"{self.config.maximum_global_absolute_drift:.6f} 门槛"
            )
        if not complete:
            decision = "ASSET_BLOCKED"
        elif gate_issue and model_issue:
            decision = "BOTH_REQUIRED"
        elif gate_issue:
            decision = "GATE_FIX"
        elif model_issue:
            decision = "MODEL_FIX"
        else:
            decision = "AUDIT_CLEAR"

        fingerprint = hashlib.sha256(
            json.dumps(before, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest().upper()
        decision_payload = {
            "decision": decision,
            "gate_issue": gate_issue,
            "model_issue": model_issue,
            "complete": complete,
            "rank_ratios_vs_base": rank_ratios,
            "rank_failures": rank_failures,
            "global_absolute_drift": global_drift,
            "findings": findings,
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._write_json(self.output_root / "retrieval_audit.json", retrieval)
        self._write_json(self.output_root / "spectrum_audit.json", spectrum)
        self._write_json(self.output_root / "subgroup_excess_drift.json", subgroups)
        self._write_json(self.output_root / "safety_contract_audit.json", safety_audit)
        self._write_json(self.output_root / "decision.json", decision_payload)
        self._write_csv(self.output_root / "subgroup_excess_drift.csv", subgroups["groups"])
        after = {str(path.relative_to(self.input_root)): self._sha256(path) for path in required}
        if after != before:
            raise RuntimeError("Phase 1.8R audit modified an input asset")
        result = Phase18RAuditResult(
            decision=decision,
            gate_issue=gate_issue,
            model_issue=model_issue,
            complete=complete,
            findings=tuple(findings),
            input_fingerprint=fingerprint,
        )
        self._write_json(
            self.output_root / "run_summary.json",
            {
                **result.as_dict(),
                "config": asdict(self.config),
                "input_sha256": before,
                "input_immutable": True,
            },
        )
        output_files = sorted(
            path
            for path in self.output_root.iterdir()
            if path.is_file() and path.name != "sha256_manifest.json"
        )
        self._write_json(
            self.output_root / "sha256_manifest.json",
            {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": self._sha256(path),
                }
                for path in output_files
            },
        )
        return result

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest().upper()

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8-sig")
            return
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
