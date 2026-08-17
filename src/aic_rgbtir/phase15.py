from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .data import RGBTRecord
from .modeling import RGBTTIRValidationFeatures
from .phase1 import build_roi_patch_weights


@dataclass(frozen=True)
class Phase15Config:
    expected_record_count: int = 2032
    expected_pair_count: int = 1115
    seed: int = 20260812
    bootstrap_iterations: int = 2000
    cache_checkpoint_pairs: int = 25
    cache_enabled: bool = True
    spatial_merge_size: int = 2
    roi_expansion: float = 0.10
    hard_negative_weight: float = 0.25
    hard_negative_margin: float = 0.15
    minimum_alignment_improvement: float = 0.02
    maximum_subgroup_degradation: float = 0.05
    subgroup_minimum_records: int = 100
    maximum_r5_drop: float = 0.005
    minimum_retrieval_gain: float = 0.01
    minimum_effective_rank_vs_base: float = 0.70
    minimum_effective_rank_vs_rgb: float = 0.50
    maximum_nonpaired_p95_increase: float = 0.10
    fingerprint: str = ""


@dataclass(frozen=True)
class Phase15Result:
    status: str
    completed_record_count: int
    completed_pair_count: int
    skipped_record_count: int
    alignment_summary: dict[str, Any]
    retrieval_metrics: dict[str, Any]
    collapse_metrics: dict[str, Any]
    subgroup_metrics: tuple[dict[str, Any], ...]
    safety_equivalence: dict[str, Any]
    gate_checks: dict[str, bool]
    gate_failures: tuple[str, ...]
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["subgroup_metrics"] = list(self.subgroup_metrics)
        payload["gate_failures"] = list(self.gate_failures)
        return payload


def _default_resource_probe() -> dict[str, int]:
    """Return cheap process telemetry without adding a runtime dependency."""
    rss_bytes = 0
    try:
        import resource

        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rss_bytes = raw if os.name == "nt" else raw * 1024
    except (ImportError, OSError, ValueError):
        pass
    return {
        "rss_bytes": rss_bytes,
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
    }


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl_dump(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def _csv_dump(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _safe_number(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric: {value}")
    return float(value)


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0}
    return {
        "mean": _safe_number(float(array.mean())),
        "median": _safe_number(float(np.median(array))),
        "p90": _safe_number(float(np.quantile(array, 0.90))),
        "p95": _safe_number(float(np.quantile(array, 0.95))),
    }


def _weighted_mean(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"expected [tokens, hidden], got {tuple(features.shape)}")
    if features.shape[0] != weights.numel():
        raise ValueError(
            f"feature/mask token mismatch: {features.shape[0]} vs {weights.numel()}"
        )
    weights = weights.to(device=features.device, dtype=torch.float32)
    denominator = weights.sum()
    if float(denominator) < 1.0:
        raise ValueError("empty ROI after TIR validity masking")
    return (features.float() * weights[:, None]).sum(dim=0) / denominator


def _alignment_loss(
    rgb: torch.Tensor,
    tir: torch.Tensor,
    foreground: torch.Tensor,
    background: torch.Tensor,
    *,
    margin: float,
    negative_weight: float,
) -> tuple[float, float]:
    rgb_fg = _weighted_mean(rgb, foreground)
    tir_fg = _weighted_mean(tir, foreground)
    positive = F.cosine_similarity(rgb_fg[None], tir_fg[None]).item()
    loss = 1.0 - positive
    if float(background.sum()) >= 1.0:
        rgb_bg = _weighted_mean(rgb, background)
        negative = F.cosine_similarity(rgb_bg[None], tir_fg[None]).item()
        loss += negative_weight * max(0.0, margin - positive + negative)
    return _safe_number(loss), _safe_number(positive)


def _normalise_rows(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=1, eps=1e-12)


def _bootstrap_ci(values: torch.Tensor, *, iterations: int, seed: int) -> tuple[float, float]:
    array = values.detach().cpu().numpy().astype(np.float64, copy=False)
    if array.size == 0:
        return 0.0, 0.0
    generator = np.random.default_rng(seed)
    sampled = generator.choice(array, size=(iterations, array.size), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return _safe_number(float(low)), _safe_number(float(high))


def _representation_metrics(features: torch.Tensor, pair_ids: Sequence[str]) -> dict[str, float]:
    matrix = features.float()
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    eigen = singular.square()
    total = eigen.sum()
    if float(total) <= 1e-12:
        effective_rank = 0.0
        participation_ratio = 0.0
    else:
        probability = eigen / total
        effective_rank = float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()))
        participation_ratio = float(total.square() / eigen.square().sum().clamp_min(1e-12))
    variance = matrix.var(dim=0, unbiased=False)
    normalised = _normalise_rows(matrix)
    similarity = normalised @ normalised.T
    group_index: dict[str, int] = {}
    encoded = []
    for pair_id in pair_ids:
        group_index.setdefault(pair_id, len(group_index))
        encoded.append(group_index[pair_id])
    groups = torch.tensor(encoded, dtype=torch.int64)
    nonpaired = similarity[groups[:, None] != groups[None, :]]
    return {
        "dimension_variance_mean": _safe_number(float(variance.mean())),
        "dimension_variance_min": _safe_number(float(variance.min())),
        "effective_rank": _safe_number(effective_rank),
        "participation_ratio": _safe_number(participation_ratio),
        "nonpaired_cosine_mean": _safe_number(float(nonpaired.mean())) if nonpaired.numel() else 0.0,
        "nonpaired_cosine_p95": _safe_number(float(torch.quantile(nonpaired, 0.95))) if nonpaired.numel() else 0.0,
    }


class Phase15Validator:
    """Full-val Phase 1.5 validator behind one deterministic public seam."""

    def __init__(
        self,
        *,
        processor: Any,
        encode_triplet: Callable[[Any], RGBTTIRValidationFeatures],
        output_root: Path | str,
        config: Phase15Config | None = None,
        safety_checks: Callable[[], Mapping[str, Any]] | None = None,
        resource_probe: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.processor = processor
        self.encode_triplet = encode_triplet
        self.output_root = Path(output_root).resolve()
        self.config = config or Phase15Config()
        self.safety_checks = safety_checks or (lambda: {"passed": False, "checks": {}})
        self.resource_probe = resource_probe or _default_resource_probe

    def _fingerprint(self, records: Sequence[RGBTRecord]) -> str:
        if self.config.fingerprint:
            return self.config.fingerprint
        payload = {
            "config": asdict(self.config),
            "records": [
                [record.record_id, record.image_pair_key, list(record.bbox_xyxy_normalized)]
                for record in records
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest().upper()

    @staticmethod
    def _black_border_severity(valid_ratio: float) -> str:
        if valid_ratio >= 0.999:
            return "none"
        if valid_ratio >= 0.95:
            return "mild"
        if valid_ratio >= 0.80:
            return "moderate"
        return "severe"

    @staticmethod
    def _weak_alignment_risk(record: RGBTRecord) -> str:
        flags = " ".join(record.quality_flags).lower()
        return "flagged" if "alignment" in flags or "misalign" in flags else "not_available"

    def _extract_pair(
        self,
        pair_records: Sequence[RGBTRecord],
        root: Path,
    ) -> list[dict[str, Any]]:
        first = pair_records[0]
        batch = self.processor.process(first, root=root)
        triplet = self.encode_triplet(batch)
        expected_layers = ("8", "16", "24", "final")
        if tuple(triplet.layer_names) != expected_layers:
            raise ValueError(f"unexpected validation layers: {triplet.layer_names}")
        variants = {
            "rgb": triplet.rgb_premerger,
            "base": triplet.tir_base_premerger,
            "adapted": triplet.tir_adapted_premerger,
        }
        for name, tensors in variants.items():
            if len(tensors) != len(expected_layers):
                raise ValueError(f"{name} path returned {len(tensors)} layers")
            if any(not torch.isfinite(tensor).all() for tensor in tensors):
                raise ValueError(f"{name} path contains non-finite values")

        rows: list[dict[str, Any]] = []
        ir_usable = bool(batch.ir_usable)
        valid = (
            batch.ir_patch_valid_mask.float().cpu()
            if ir_usable
            else torch.ones_like(batch.ir_patch_valid_mask, dtype=torch.float32).cpu()
        )
        for record in pair_records:
            roi = build_roi_patch_weights(
                normalized_bbox=record.bbox_xyxy_normalized,
                image_grid_thw=triplet.image_grid_thw.cpu(),
                spatial_merge_size=int(self.config.spatial_merge_size),
                expansion=float(self.config.roi_expansion),
                device="cpu",
            )
            foreground = roi * valid
            if float(foreground.sum()) < 1.0:
                raise ValueError(f"{record.record_id}: empty ROI after IR mask")
            background = (1.0 - roi) * valid
            pooled: dict[str, dict[str, torch.Tensor]] = {
                name: {} for name in variants
            }
            base_losses: list[float] = []
            adapted_losses: list[float] = []
            base_cosines: dict[str, float] = {}
            adapted_cosines: dict[str, float] = {}
            for layer_index, layer_name in enumerate(expected_layers):
                rgb = variants["rgb"][layer_index].detach().float().cpu()
                base = variants["base"][layer_index].detach().float().cpu()
                adapted = variants["adapted"][layer_index].detach().float().cpu()
                pooled["rgb"][layer_name] = _weighted_mean(rgb, foreground).half()
                pooled["base"][layer_name] = _weighted_mean(base, foreground).half()
                pooled["adapted"][layer_name] = _weighted_mean(adapted, foreground).half()
                base_loss, base_cosine = _alignment_loss(
                    rgb,
                    base,
                    foreground,
                    background,
                    margin=self.config.hard_negative_margin,
                    negative_weight=self.config.hard_negative_weight,
                )
                adapted_loss, adapted_cosine = _alignment_loss(
                    rgb,
                    adapted,
                    foreground,
                    background,
                    margin=self.config.hard_negative_margin,
                    negative_weight=self.config.hard_negative_weight,
                )
                base_losses.append(base_loss)
                adapted_losses.append(adapted_loss)
                base_cosines[layer_name] = base_cosine
                adapted_cosines[layer_name] = adapted_cosine
            row = {
                "record_id": record.record_id,
                "image_pair_key": record.image_pair_key,
                "source_dataset": record.source_dataset,
                "illumination": record.illumination,
                "weather": record.weather,
                "object_size": record.object_size,
                "occlusion": record.occlusion,
                "black_border_severity": self._black_border_severity(float(batch.ir_valid_ratio)),
                "weak_alignment_risk": self._weak_alignment_risk(record),
                "ir_usage_mode": "tir_adapter" if ir_usable else "rgb_only_degradation",
                "ir_valid_ratio": float(batch.ir_valid_ratio),
                "roi_token_count": int(foreground.sum().item()),
                "base_alignment_loss": float(np.mean(base_losses)),
                "adapted_alignment_loss": float(np.mean(adapted_losses)),
                "base_layer_cosines": base_cosines,
                "adapted_layer_cosines": adapted_cosines,
                "pooled": pooled,
            }
            rows.append(row)
        return rows

    def _retrieval(
        self, rows: Sequence[dict[str, Any]], layer_names: Sequence[str]
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        pair_ids = [str(row["image_pair_key"]) for row in rows]
        for layer_index, layer_name in enumerate(layer_names):
            rgb = _normalise_rows(torch.stack([row["pooled"]["rgb"][layer_name] for row in rows]))
            layer_result: dict[str, Any] = {}
            for variant_index, variant in enumerate(("base", "adapted")):
                tir = _normalise_rows(torch.stack([row["pooled"][variant][layer_name] for row in rows]))
                similarities = tir @ rgb.T
                order = similarities.argsort(dim=1, descending=True)
                target = torch.arange(len(rows))[:, None]
                r1 = float((order[:, :1] == target).any(dim=1).float().mean())
                r5 = float((order[:, : min(5, len(rows))] == target).any(dim=1).float().mean())
                paired = similarities.diag()
                rng = random.Random(
                    self.config.seed + layer_index * 101 + variant_index * 1009
                )
                shuffled_indexes: list[int] = []
                for index, pair_id in enumerate(pair_ids):
                    candidates = [
                        candidate
                        for candidate, candidate_pair in enumerate(pair_ids)
                        if candidate_pair != pair_id
                    ]
                    if not candidates:
                        raise ValueError("retrieval requires at least two unique image pairs")
                    shuffled_indexes.append(candidates[rng.randrange(len(candidates))])
                shuffled = similarities[
                    torch.arange(len(rows)), torch.tensor(shuffled_indexes)
                ]
                margins = paired - shuffled
                ci_low, ci_high = _bootstrap_ci(
                    margins,
                    iterations=self.config.bootstrap_iterations,
                    seed=self.config.seed + layer_index * 17 + variant_index,
                )
                layer_result[variant] = {
                    "r_at_1": _safe_number(r1),
                    "r_at_5": _safe_number(r5),
                    "paired_cosine": _safe_number(float(paired.mean())),
                    "shuffled_cosine": _safe_number(float(shuffled.mean())),
                    "paired_shuffled_margin": _safe_number(float(margins.mean())),
                    "paired_shuffled_margin_ci95": [ci_low, ci_high],
                }
            result[layer_name] = layer_result
        return result

    def _collapse(
        self, rows: Sequence[dict[str, Any]], layer_names: Sequence[str]
    ) -> dict[str, Any]:
        pair_ids = [str(row["image_pair_key"]) for row in rows]
        result: dict[str, Any] = {}
        for layer_name in layer_names:
            result[layer_name] = {
                variant: _representation_metrics(
                    torch.stack([row["pooled"][variant][layer_name] for row in rows]),
                    pair_ids,
                )
                for variant in ("rgb", "base", "adapted")
            }
        return result

    def _alignment_and_subgroups(
        self, rows: Sequence[dict[str, Any]]
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
        base = [float(row["base_alignment_loss"]) for row in rows]
        adapted = [float(row["adapted_alignment_loss"]) for row in rows]
        base_mean = float(np.mean(base)) if base else 0.0
        adapted_mean = float(np.mean(adapted)) if adapted else 0.0
        improvement = (base_mean - adapted_mean) / max(abs(base_mean), 1e-12)
        overall = {
            "record_count": len(rows),
            "base": _summary(base),
            "adapted": _summary(adapted),
            "relative_improvement": _safe_number(improvement),
        }
        subgroup_rows: list[dict[str, Any]] = []
        for field_name in (
            "source_dataset",
            "illumination",
            "weather",
            "object_size",
            "occlusion",
            "black_border_severity",
            "weak_alignment_risk",
        ):
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[str(row[field_name])].append(row)
            for value in sorted(grouped):
                group = grouped[value]
                group_base = float(np.mean([row["base_alignment_loss"] for row in group]))
                group_adapted = float(np.mean([row["adapted_alignment_loss"] for row in group]))
                relative = (group_base - group_adapted) / max(abs(group_base), 1e-12)
                subgroup_rows.append(
                    {
                        "group_field": field_name,
                        "group_value": value,
                        "record_count": len(group),
                        "base_alignment_loss_mean": _safe_number(group_base),
                        "adapted_alignment_loss_mean": _safe_number(group_adapted),
                        "relative_improvement": _safe_number(relative),
                    }
                )
        return overall, tuple(subgroup_rows)

    def _hard_gate(
        self,
        *,
        rows: Sequence[dict[str, Any]],
        pair_count: int,
        failures: Sequence[Mapping[str, Any]],
        alignment: Mapping[str, Any],
        retrieval: Mapping[str, Any],
        collapse: Mapping[str, Any],
        subgroups: Sequence[Mapping[str, Any]],
        safety: Mapping[str, Any],
    ) -> tuple[dict[str, bool], tuple[str, ...]]:
        cfg = self.config
        checks: dict[str, bool] = {}
        reasons: list[str] = []

        checks["complete"] = (
            len(rows) == cfg.expected_record_count
            and pair_count == cfg.expected_pair_count
            and not failures
        )
        if not checks["complete"]:
            reasons.append(
                "completion: "
                f"records={len(rows)}/{cfg.expected_record_count}, "
                f"pairs={pair_count}/{cfg.expected_pair_count}, failures={len(failures)}"
            )

        checks["alignment_improvement"] = (
            float(alignment["relative_improvement"])
            >= cfg.minimum_alignment_improvement
        )
        if not checks["alignment_improvement"]:
            reasons.append("alignment_improvement below 2%")

        bad_groups = [
            row
            for row in subgroups
            if int(row["record_count"]) >= cfg.subgroup_minimum_records
            and float(row["relative_improvement"]) < -cfg.maximum_subgroup_degradation
        ]
        checks["subgroup_degradation"] = not bad_groups
        if bad_groups:
            reasons.append(
                "subgroup_degradation: "
                + ", ".join(
                    f"{row['group_field']}={row['group_value']}"
                    for row in bad_groups
                )
            )

        early_layers = ("8", "16", "24")
        margin_bad = [
            layer
            for layer in early_layers
            if float(retrieval[layer]["adapted"]["paired_shuffled_margin"]) <= 0
            or float(retrieval[layer]["adapted"]["paired_shuffled_margin_ci95"][0]) <= 0
        ]
        checks["positive_margin"] = not margin_bad
        if margin_bad:
            reasons.append(f"paired_shuffled_margin failed at {margin_bad}")

        r5_bad = [
            layer
            for layer in ("8", "16")
            if float(retrieval[layer]["adapted"]["r_at_5"])
            < float(retrieval[layer]["base"]["r_at_5"]) - cfg.maximum_r5_drop
        ]
        checks["r5_preserved"] = not r5_bad
        if r5_bad:
            reasons.append(f"R@5 degraded at {r5_bad}")

        gains = [
            float(retrieval[layer]["adapted"][metric])
            - float(retrieval[layer]["base"][metric])
            for layer in ("8", "16")
            for metric in ("r_at_1", "r_at_5")
        ]
        checks["retrieval_gain"] = max(gains, default=-math.inf) >= cfg.minimum_retrieval_gain
        if not checks["retrieval_gain"]:
            reasons.append("layer 8/16 retrieval gain below 1 percentage point")

        rank_bad: list[str] = []
        for layer in early_layers:
            adapted_rank = float(collapse[layer]["adapted"]["effective_rank"])
            base_rank = float(collapse[layer]["base"]["effective_rank"])
            rgb_rank = float(collapse[layer]["rgb"]["effective_rank"])
            if adapted_rank < cfg.minimum_effective_rank_vs_base * base_rank or adapted_rank < cfg.minimum_effective_rank_vs_rgb * rgb_rank:
                rank_bad.append(layer)
        checks["effective_rank"] = not rank_bad
        if rank_bad:
            reasons.append(f"effective_rank collapse at {rank_bad}")

        p95_bad = [
            layer
            for layer in early_layers
            if float(collapse[layer]["adapted"]["nonpaired_cosine_p95"])
            - float(collapse[layer]["base"]["nonpaired_cosine_p95"])
            > cfg.maximum_nonpaired_p95_increase
        ]
        checks["nonpaired_cosine_p95"] = not p95_bad
        if p95_bad:
            reasons.append(f"nonpaired cosine P95 increase exceeded 0.10 at {p95_bad}")

        checks["safety_equivalence"] = bool(safety.get("passed"))
        if not checks["safety_equivalence"]:
            reasons.append("safety equivalence failed")
        return checks, tuple(reasons)

    def run(self, records: Sequence[RGBTRecord], root: Path | str) -> Phase15Result:
        ordered = sorted(records, key=lambda record: record.record_id)
        if len({record.record_id for record in ordered}) != len(ordered):
            raise ValueError("duplicate record_id in Phase 1.5 input")
        fingerprint = self._fingerprint(ordered)
        self.output_root.mkdir(parents=True, exist_ok=True)
        parts_root = self.output_root / "embedding_cache_parts"
        if self.config.cache_enabled:
            parts_root.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[RGBTRecord]] = defaultdict(list)
        for record in ordered:
            grouped[record.image_pair_key].append(record)

        part_paths: list[Path] = []
        rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        resource_events: list[dict[str, Any]] = []
        completed_pairs = 0
        for pair_key in sorted(grouped):
            part_name = hashlib.sha256(pair_key.encode("utf-8")).hexdigest() + ".pt"
            part_path = parts_root / part_name
            try:
                if self.config.cache_enabled and part_path.is_file():
                    part = torch.load(part_path, map_location="cpu", weights_only=False)
                    if part.get("fingerprint") != fingerprint or part.get("pair_key") != pair_key:
                        raise ValueError("cache fingerprint or pair key mismatch")
                    pair_rows = part["rows"]
                else:
                    pair_rows = self._extract_pair(grouped[pair_key], Path(root).resolve())
                    if self.config.cache_enabled:
                        temp = part_path.with_suffix(".tmp")
                        torch.save(
                            {"fingerprint": fingerprint, "pair_key": pair_key, "rows": pair_rows},
                            temp,
                        )
                        temp.replace(part_path)
                if self.config.cache_enabled:
                    part_paths.append(part_path)
                else:
                    rows.extend(pair_rows)
                del pair_rows
                completed_pairs += 1
                checkpoint = max(1, int(self.config.cache_checkpoint_pairs))
                if completed_pairs % checkpoint == 0 or completed_pairs == len(grouped):
                    resource_events.append(
                        {
                            "completed_pairs": completed_pairs,
                            "expected_pairs": len(grouped),
                            **dict(self.resource_probe()),
                        }
                    )
            except Exception as exc:
                failures.append(
                    {
                        "image_pair_key": pair_key,
                        "record_ids": [record.record_id for record in grouped[pair_key]],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        _jsonl_dump(self.output_root / "resource_events.jsonl", resource_events)
        if self.config.cache_enabled:
            for part_path in part_paths:
                part = torch.load(part_path, map_location="cpu", weights_only=False)
                if part.get("fingerprint") != fingerprint:
                    failures.append(
                        {
                            "image_pair_key": str(part.get("pair_key", "")),
                            "record_ids": [],
                            "error": "ValueError: cache fingerprint mismatch during aggregation",
                        }
                    )
                    continue
                rows.extend(part["rows"])
                del part
        rows.sort(key=lambda row: str(row["record_id"]))
        layer_names = ("8", "16", "24", "final")
        alignment, subgroups = self._alignment_and_subgroups(rows)
        retrieval = self._retrieval(rows, layer_names) if rows else {}
        collapse = self._collapse(rows, layer_names) if rows else {}
        safety = dict(self.safety_checks())
        if rows:
            gate_checks, gate_failures = self._hard_gate(
                rows=rows,
                pair_count=completed_pairs,
                failures=failures,
                alignment=alignment,
                retrieval=retrieval,
                collapse=collapse,
                subgroups=subgroups,
                safety=safety,
            )
        else:
            gate_checks = {"complete": False}
            gate_failures = ("no validation records completed",)
        status = "PHASE_15_GO" if gate_checks and all(gate_checks.values()) else "PHASE_15_NO_GO"
        result = Phase15Result(
            status=status,
            completed_record_count=len(rows),
            completed_pair_count=completed_pairs,
            skipped_record_count=len(ordered) - len(rows),
            alignment_summary=alignment,
            retrieval_metrics=retrieval,
            collapse_metrics=collapse,
            subgroup_metrics=subgroups,
            safety_equivalence=safety,
            gate_checks=gate_checks,
            gate_failures=gate_failures,
            fingerprint=fingerprint,
        )

        public_rows = [
            {key: value for key, value in row.items() if key != "pooled"}
            for row in rows
        ]
        if self.config.cache_enabled:
            torch.save(
                {
                    "fingerprint": fingerprint,
                    "records": [
                        {
                            "record_id": row["record_id"],
                            "image_pair_key": row["image_pair_key"],
                            "pooled": row["pooled"],
                        }
                        for row in rows
                    ],
                },
                self.output_root / "embedding_cache.pt",
            )
        _jsonl_dump(self.output_root / "per_record_metrics.jsonl", public_rows)
        _json_dump(self.output_root / "alignment_summary.json", alignment)
        _json_dump(self.output_root / "retrieval_metrics.json", retrieval)
        _json_dump(self.output_root / "collapse_metrics.json", collapse)
        _csv_dump(self.output_root / "subgroup_metrics.csv", subgroups)
        _json_dump(self.output_root / "safety_equivalence.json", safety)
        _csv_dump(self.output_root / "failure_cases.csv", failures)
        _json_dump(self.output_root / "run_summary.json", result.as_dict())
        return result
