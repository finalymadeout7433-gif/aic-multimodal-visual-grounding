from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .data import RGBTRecord


AUDIT_LAYERS = ("8", "16", "24")


@dataclass(frozen=True)
class LayerAuditFeatures:
    adapted: Mapping[str, torch.Tensor]
    base: Mapping[str, torch.Tensor]
    teacher: Mapping[str, torch.Tensor]


def pair_safe_shuffle_indices(
    records: Sequence[RGBTRecord], *, seed: int
) -> torch.Tensor:
    """Return deterministic negative indices that always cross image pairs."""

    if len({record.image_pair_key for record in records}) < 2:
        raise ValueError("pair-safe shuffle requires at least two image pairs")
    indices: list[int] = []
    for index, record in enumerate(records):
        candidates = [
            candidate
            for candidate, other in enumerate(records)
            if other.image_pair_key != record.image_pair_key
        ]
        digest = hashlib.sha256(
            f"{seed}:{record.record_id}:{index}".encode("utf-8")
        ).digest()
        indices.append(candidates[int.from_bytes(digest[:8], "big") % len(candidates)])
    return torch.tensor(indices, dtype=torch.long)


def _spectrum(features: torch.Tensor) -> dict[str, Any]:
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        probability = torch.zeros_like(energy)
        cumulative = torch.zeros_like(energy)
        effective_rank = 0.0
        participation_ratio = 0.0
    else:
        probability = energy / total
        cumulative = probability.cumsum(dim=0)
        effective_rank = float(
            torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum())
        )
        participation_ratio = float(1.0 / probability.square().sum())

    def components_for(threshold: float) -> int:
        if not cumulative.numel() or float(cumulative[-1]) <= 0.0:
            return 0
        return int(torch.searchsorted(cumulative, threshold).item()) + 1

    return {
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "singular_values": [float(value) for value in singular],
        "energy_probability": [float(value) for value in probability],
        "cumulative_energy": [float(value) for value in cumulative],
        "components_for_90pct": components_for(0.90),
        "components_for_95pct": components_for(0.95),
        "components_for_99pct": components_for(0.99),
    }


def _aggregate_by_pair(
    features: torch.Tensor, records: Sequence[RGBTRecord]
) -> torch.Tensor:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.image_pair_key, []).append(index)
    return torch.stack(
        [features[indices].float().mean(dim=0) for _, indices in sorted(groups.items())]
    )


def _retrieval(similarity: torch.Tensor) -> tuple[float, float]:
    order = similarity.argsort(dim=1, descending=True)
    target = torch.arange(similarity.shape[0])[:, None]
    r1 = float((order[:, :1] == target).any(dim=1).float().mean())
    r5 = float(
        (order[:, : min(5, order.shape[1])] == target)
        .any(dim=1)
        .float()
        .mean()
    )
    return r1, r5


class D2LayerAudit:
    """Compute lossless per-layer evidence for the four D2 full checkpoints."""

    def __init__(
        self,
        *,
        minimum_rank_vs_base: float = 0.85,
        seed: int = 20260817,
    ) -> None:
        self.minimum_rank_vs_base = float(minimum_rank_vs_base)
        self.seed = int(seed)

    @staticmethod
    def _validate(
        records: Sequence[RGBTRecord], features: LayerAuditFeatures
    ) -> None:
        if len(records) < 2:
            raise ValueError("layer audit requires at least two records")
        expected = set(AUDIT_LAYERS)
        for name, values in (
            ("adapted", features.adapted),
            ("base", features.base),
            ("teacher", features.teacher),
        ):
            if set(values) != expected:
                raise ValueError(f"{name} layer mismatch: {sorted(values)}")
            for layer, tensor in values.items():
                if tensor.ndim != 2 or tensor.shape[0] != len(records):
                    raise ValueError(
                        f"{name} feature shape mismatch at layer {layer}: {tensor.shape}"
                    )
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"non-finite {name} features at layer {layer}")
        for layer in AUDIT_LAYERS:
            shapes = {
                tuple(features.adapted[layer].shape),
                tuple(features.base[layer].shape),
                tuple(features.teacher[layer].shape),
            }
            if len(shapes) != 1:
                raise ValueError(f"feature shape disagreement at layer {layer}: {shapes}")

    @torch.inference_mode()
    def analyze(
        self,
        *,
        split: str,
        checkpoint_step: int,
        records: Sequence[RGBTRecord],
        features: LayerAuditFeatures,
        fingerprint: str,
    ) -> dict[str, Any]:
        self._validate(records, features)
        pair_keys = [record.image_pair_key for record in records]
        pair_safe = pair_safe_shuffle_indices(records, seed=self.seed)
        same_pair = torch.tensor(
            [[left == right for right in pair_keys] for left in pair_keys],
            dtype=torch.bool,
        )
        diagonal = torch.eye(len(records), dtype=torch.bool)
        layer_results: dict[str, Any] = {}

        for layer in AUDIT_LAYERS:
            adapted_raw = features.adapted[layer].detach().cpu().float()
            base_raw = features.base[layer].detach().cpu().float()
            teacher_raw = features.teacher[layer].detach().cpu().float()
            adapted = F.normalize(adapted_raw, dim=1)
            base = F.normalize(base_raw, dim=1)
            teacher = F.normalize(teacher_raw, dim=1)
            similarity = adapted @ teacher.T
            base_similarity = base @ teacher.T
            adapted_spectrum = _spectrum(adapted_raw)
            base_spectrum = _spectrum(base_raw)
            teacher_spectrum = _spectrum(teacher_raw)
            adapted_rank = float(adapted_spectrum["effective_rank"])
            base_rank = float(base_spectrum["effective_rank"])
            teacher_rank = float(teacher_spectrum["effective_rank"])
            r1, r5 = _retrieval(similarity)
            base_r1, base_r5 = _retrieval(base_similarity)
            nonpaired = similarity[~diagonal]
            base_nonpaired = base_similarity[~diagonal]
            cross_pair = similarity[~same_pair]
            base_cross_pair = base_similarity[~same_pair]
            paired = similarity.diag()
            base_paired = base_similarity.diag()
            rows = torch.arange(len(records))
            legacy_shuffle = torch.roll(rows, shifts=1)

            adapted_pair = _aggregate_by_pair(adapted_raw, records)
            base_pair = _aggregate_by_pair(base_raw, records)
            teacher_pair = _aggregate_by_pair(teacher_raw, records)
            adapted_pair_spectrum = _spectrum(adapted_pair)
            base_pair_spectrum = _spectrum(base_pair)
            teacher_pair_spectrum = _spectrum(teacher_pair)

            layer_results[layer] = {
                "effective_rank": adapted_rank,
                "base_effective_rank": base_rank,
                "teacher_effective_rank": teacher_rank,
                "effective_rank_ratio_vs_base": adapted_rank / max(base_rank, 1e-8),
                "effective_rank_ratio_vs_teacher": adapted_rank
                / max(teacher_rank, 1e-8),
                "participation_ratio": adapted_spectrum["participation_ratio"],
                "base_participation_ratio": base_spectrum["participation_ratio"],
                "teacher_participation_ratio": teacher_spectrum["participation_ratio"],
                "r_at_1": r1,
                "r_at_5": r5,
                "base_r_at_1": base_r1,
                "base_r_at_5": base_r5,
                "paired_cosine_mean": float(paired.mean()),
                "base_paired_cosine_mean": float(base_paired.mean()),
                "alignment_loss_mean": float((1.0 - paired).mean()),
                "base_alignment_loss_mean": float((1.0 - base_paired).mean()),
                "absolute_alignment_delta": float((base_paired - paired).mean()),
                "nonpaired_cosine_mean": float(nonpaired.mean()),
                "nonpaired_cosine_p95": float(torch.quantile(nonpaired, 0.95)),
                "base_nonpaired_cosine_mean": float(base_nonpaired.mean()),
                "base_nonpaired_cosine_p95": float(
                    torch.quantile(base_nonpaired, 0.95)
                ),
                "cross_pair_cosine_mean": float(cross_pair.mean()),
                "cross_pair_cosine_p95": float(torch.quantile(cross_pair, 0.95)),
                "base_cross_pair_cosine_mean": float(base_cross_pair.mean()),
                "base_cross_pair_cosine_p95": float(
                    torch.quantile(base_cross_pair, 0.95)
                ),
                "legacy_roll_margin_mean": float(
                    (paired - similarity[rows, legacy_shuffle]).mean()
                ),
                "pair_safe_margin_mean": float(
                    (paired - similarity[rows, pair_safe]).mean()
                ),
                "singular_spectrum": adapted_spectrum,
                "base_singular_spectrum": base_spectrum,
                "teacher_singular_spectrum": teacher_spectrum,
                "pair_aggregated": {
                    "count": int(adapted_pair.shape[0]),
                    "effective_rank": adapted_pair_spectrum["effective_rank"],
                    "base_effective_rank": base_pair_spectrum["effective_rank"],
                    "teacher_effective_rank": teacher_pair_spectrum["effective_rank"],
                    "effective_rank_ratio_vs_base": float(
                        adapted_pair_spectrum["effective_rank"]
                    )
                    / max(float(base_pair_spectrum["effective_rank"]), 1e-8),
                    "effective_rank_ratio_vs_teacher": float(
                        adapted_pair_spectrum["effective_rank"]
                    )
                    / max(float(teacher_pair_spectrum["effective_rank"]), 1e-8),
                },
            }

        bottleneck = min(
            AUDIT_LAYERS,
            key=lambda layer: float(
                layer_results[layer]["effective_rank_ratio_vs_base"]
            ),
        )
        minimum = float(layer_results[bottleneck]["effective_rank_ratio_vs_base"])
        return {
            "schema_version": 1,
            "split": str(split),
            "checkpoint_step": int(checkpoint_step),
            "fingerprint": str(fingerprint),
            "record_count": len(records),
            "pair_count": len(set(pair_keys)),
            "minimum_rank_vs_base_gate": self.minimum_rank_vs_base,
            "bottleneck_layer": bottleneck,
            "minimum_effective_rank_ratio_vs_base": minimum,
            "passed_absolute_rank": minimum >= self.minimum_rank_vs_base,
            "layers": layer_results,
        }


def summarize_checkpoint_audits(
    audits: Mapping[int, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    """Summarize four checkpoint audits without turning a reference into a release."""

    if not audits:
        raise ValueError("checkpoint audits are empty")
    rows: list[dict[str, Any]] = []
    for step, splits in sorted(audits.items()):
        if set(splits) != {"semantic", "multiquery"}:
            raise ValueError(f"checkpoint {step} does not contain both dev splits")
        rows.append(
            {
                "step": int(step),
                "semantic_bottleneck_layer": splits["semantic"]["bottleneck_layer"],
                "semantic_minimum_rank_vs_base": float(
                    splits["semantic"]["minimum_effective_rank_ratio_vs_base"]
                ),
                "multiquery_bottleneck_layer": splits["multiquery"]["bottleneck_layer"],
                "multiquery_minimum_rank_vs_base": float(
                    splits["multiquery"]["minimum_effective_rank_ratio_vs_base"]
                ),
            }
        )
    reference = max(
        rows,
        key=lambda row: (
            row["multiquery_minimum_rank_vs_base"],
            row["semantic_minimum_rank_vs_base"],
            -row["step"],
        ),
    )
    multiquery_layers = [str(row["multiquery_bottleneck_layer"]) for row in rows]
    unique = sorted(set(multiquery_layers), key=lambda layer: AUDIT_LAYERS.index(layer))
    return {
        "status": "PHASE_19_D2_LAYER_AUDIT_COMPLETE",
        "checkpoint_count": len(rows),
        "rows": rows,
        "diagnostic_reference_step": int(reference["step"]),
        "diagnostic_reference_is_release": False,
        "multiquery_bottleneck_layers": unique,
        "single_layer_bottleneck_across_checkpoints": len(unique) == 1,
    }
