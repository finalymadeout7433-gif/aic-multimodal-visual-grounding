from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .data import RGBTRecord


LAYER_NAMES = ("8", "16", "24", "final")
DISCRIMINATIVE_LAYERS = ("8", "16", "24")


@dataclass(frozen=True)
class RepairSplit:
    probe_train: tuple[RGBTRecord, ...]
    dev: tuple[RGBTRecord, ...]
    full_train: tuple[RGBTRecord, ...]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    pair_weight: float
    contrastive_weight: float
    relational_weight: float
    background_weight: float

    @classmethod
    def c0(cls) -> "CandidateSpec":
        return cls("C0", 1.0, 0.0, 0.0, 0.25)

    @classmethod
    def c1(cls) -> "CandidateSpec":
        return cls("C1", 0.25, 1.0, 0.0, 0.10)

    @classmethod
    def c2(cls) -> "CandidateSpec":
        return cls("C2", 0.25, 1.0, 0.50, 0.10)


@dataclass(frozen=True)
class Phase16Config:
    seed: int = 20260812
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0
    temperature: float = 0.07
    margin: float = 0.15
    probe_steps: int = 4096
    full_steps: int = 0
    negative_count: int = 256
    same_source_negatives: int = 128
    same_condition_negatives: int = 64
    relational_anchor_count: int = 64
    checkpoint_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    minimum_alignment_improvement: float = 0.02
    maximum_r5_drop: float = 0.005
    minimum_retrieval_gain: float = 0.01
    minimum_effective_rank_vs_base: float = 0.70
    minimum_effective_rank_vs_teacher: float = 0.50
    maximum_nonpaired_cosine_p95_increase: float = 0.10


@dataclass(frozen=True)
class RepairLossOutput:
    total: torch.Tensor
    pair: torch.Tensor
    contrastive: torch.Tensor
    relational: torch.Tensor
    background: torch.Tensor


@dataclass(frozen=True)
class TeacherEntry:
    record_id: str
    image_pair_key: str
    source_dataset: str
    illumination: str
    object_size: str
    foreground: Mapping[str, torch.Tensor]
    background: Mapping[str, torch.Tensor]
    base_tir: Mapping[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class Phase16Result:
    status: str
    selected_candidate: str
    completed_candidates: tuple[str, ...]
    probe_metrics: Mapping[str, Mapping[str, float]]
    selected_checkpoint: str


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _stratified_pair_order(
    representatives: Sequence[RGBTRecord], *, seed: int
) -> list[RGBTRecord]:
    strata: dict[tuple[str, str, str, str, str], list[RGBTRecord]] = {}
    for record in representatives:
        key = (
            record.source_dataset,
            record.illumination,
            record.weather,
            record.object_size,
            record.occlusion,
        )
        strata.setdefault(key, []).append(record)
    rng = random.Random(seed)
    for values in strata.values():
        values.sort(key=lambda item: (item.image_pair_key, item.record_id))
        rng.shuffle(values)
    keys = sorted(strata)
    rng.shuffle(keys)
    ordered: list[RGBTRecord] = []
    while keys:
        next_keys: list[tuple[str, str, str, str, str]] = []
        for key in keys:
            values = strata[key]
            if values:
                ordered.append(values.pop())
            if values:
                next_keys.append(key)
        keys = next_keys
    return ordered


def build_repair_split(
    records: Sequence[RGBTRecord],
    *,
    probe_pairs: int = 4096,
    dev_pairs: int = 1024,
    seed: int = 20260812,
) -> RepairSplit:
    """Build deterministic image-pair-isolated repair partitions.

    Probe/dev contain one representative record per pair. Full train retains
    every clean record whose image pair is not reserved by dev.
    """

    by_pair: dict[str, list[RGBTRecord]] = {}
    for record in records:
        by_pair.setdefault(record.image_pair_key, []).append(record)
    required = probe_pairs + dev_pairs
    if len(by_pair) < required:
        raise ValueError(
            f"need at least {required} unique image pairs, found {len(by_pair)}"
        )
    representatives = [
        sorted(values, key=lambda item: item.record_id)[0]
        for _, values in sorted(by_pair.items())
    ]
    ordered = _stratified_pair_order(representatives, seed=seed)
    dev = tuple(ordered[:dev_pairs])
    dev_pairs_set = {record.image_pair_key for record in dev}
    probe = tuple(ordered[dev_pairs : dev_pairs + probe_pairs])
    full = tuple(
        record for record in records if record.image_pair_key not in dev_pairs_set
    )
    return RepairSplit(probe_train=probe, dev=dev, full_train=full)


class DeterministicNegativeSampler:
    """Select reproducible cross-image negatives with domain-aware quotas."""

    def __init__(self, records: Sequence[RGBTRecord], *, seed: int) -> None:
        self.seed = int(seed)
        self.records = tuple(records)
        self.by_id = {record.record_id: record for record in self.records}
        if len(self.by_id) != len(self.records):
            raise ValueError("negative-sampling records must have unique record_id")

    @staticmethod
    def _condition(record: RGBTRecord) -> tuple[str, str]:
        return record.illumination, record.object_size

    def _choose(
        self,
        candidates: Sequence[RGBTRecord],
        *,
        count: int,
        rng: random.Random,
        selected: set[str],
    ) -> list[str]:
        values = [record for record in candidates if record.record_id not in selected]
        values.sort(key=lambda item: item.record_id)
        rng.shuffle(values)
        chosen = [record.record_id for record in values[:count]]
        selected.update(chosen)
        return chosen

    def sample(
        self,
        record: RGBTRecord,
        *,
        total: int,
        same_source: int,
        same_condition: int,
    ) -> tuple[str, ...]:
        if total < 1:
            raise ValueError("negative total must be positive")
        if same_source + same_condition > total:
            raise ValueError("negative bucket quotas exceed total")
        eligible = [
            value
            for value in self.records
            if value.image_pair_key != record.image_pair_key
        ]
        unique_pairs = {value.image_pair_key for value in eligible}
        if len(unique_pairs) < total:
            raise ValueError(
                f"need {total} cross-image negatives, found {len(unique_pairs)} pairs"
            )
        # One record per image pair prevents multi-query duplicates from
        # dominating the negative set.
        pair_representatives: dict[str, RGBTRecord] = {}
        for value in sorted(eligible, key=lambda item: item.record_id):
            pair_representatives.setdefault(value.image_pair_key, value)
        pool = list(pair_representatives.values())
        rng = random.Random(_stable_seed(self.seed, record.record_id))
        selected: set[str] = set()
        result: list[str] = []
        result.extend(
            self._choose(
                [value for value in pool if value.source_dataset == record.source_dataset],
                count=same_source,
                rng=rng,
                selected=selected,
            )
        )
        result.extend(
            self._choose(
                [value for value in pool if self._condition(value) == self._condition(record)],
                count=same_condition,
                rng=rng,
                selected=selected,
            )
        )
        result.extend(
            self._choose(
                pool,
                count=total - len(result),
                rng=rng,
                selected=selected,
            )
        )
        if len(result) != total:
            raise ValueError(
                f"negative sampler produced {len(result)}/{total}; dataset lacks bucket coverage"
            )
        return tuple(result)


class RGBTeacherBank:
    """Fingerprint-bound frozen RGB ROI reference features."""

    def __init__(self, *, fingerprint: str, entries: Mapping[str, TeacherEntry]) -> None:
        self.fingerprint = str(fingerprint)
        self.entries = dict(entries)

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(self.entries)

    def __getitem__(self, record_id: str) -> TeacherEntry:
        return self.entries[record_id]

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fingerprint": self.fingerprint,
                "entries": self.entries,
            },
            target,
        )

    @classmethod
    def load(
        cls, path: Path | str, *, expected_fingerprint: str
    ) -> "RGBTeacherBank":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        actual = str(payload.get("fingerprint", ""))
        if actual != expected_fingerprint:
            raise RuntimeError(
                f"teacher bank fingerprint mismatch: expected={expected_fingerprint}, actual={actual}"
            )
        entries = payload.get("entries")
        if not isinstance(entries, Mapping):
            raise TypeError("teacher bank entries are missing")
        return cls(fingerprint=actual, entries=entries)


class Phase16RepairObjective(nn.Module):
    """Pair alignment plus cross-image discrimination and relational geometry."""

    def __init__(
        self,
        candidate: CandidateSpec,
        *,
        temperature: float = 0.07,
        margin: float = 0.15,
        layer_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.candidate = candidate
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.layer_weights = dict(
            layer_weights or {"8": 1.0, "16": 1.0, "24": 0.5, "final": 0.05}
        )

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(a.float()[None], b.float()[None]).squeeze(0)

    def forward(
        self,
        tir_foreground: Mapping[str, torch.Tensor],
        rgb_positive: Mapping[str, torch.Tensor],
        rgb_background: Mapping[str, torch.Tensor],
        rgb_negatives: Mapping[str, torch.Tensor],
        rgb_anchors: Mapping[str, torch.Tensor],
        *,
        base_tir: Mapping[str, torch.Tensor] | None = None,
    ) -> RepairLossOutput:
        del base_tir
        parts: dict[str, list[torch.Tensor]] = {
            "pair": [],
            "contrastive": [],
            "relational": [],
            "background": [],
        }
        for layer, weight in self.layer_weights.items():
            query = tir_foreground[layer].float()
            device = query.device
            positive = rgb_positive[layer].float().detach().to(device)
            positive_cosine = self._cosine(query, positive)
            parts["pair"].append(weight * (1.0 - positive_cosine))
            negative_cosine = self._cosine(
                query, rgb_background[layer].detach().to(device)
            )
            parts["background"].append(
                weight * F.relu(self.margin - positive_cosine + negative_cosine)
            )
            if layer == "final":
                continue
            if self.candidate.contrastive_weight > 0:
                negatives = rgb_negatives[layer].float().detach().to(device)
                logits = torch.cat(
                    (
                        F.cosine_similarity(query[None], positive[None]).reshape(1),
                        F.cosine_similarity(query[None], negatives),
                    )
                )[None] / self.temperature
                parts["contrastive"].append(
                    weight
                    * F.cross_entropy(
                        logits, torch.zeros(1, dtype=torch.long, device=logits.device)
                    )
                )
            if self.candidate.relational_weight > 0:
                anchors = rgb_anchors[layer].float().detach().to(device)
                student_geometry = F.cosine_similarity(query[None], anchors)
                teacher_geometry = F.cosine_similarity(positive[None], anchors)
                parts["relational"].append(
                    weight * F.mse_loss(student_geometry, teacher_geometry)
                )

        zero = next(iter(tir_foreground.values())).float().new_zeros(())

        def mean(name: str) -> torch.Tensor:
            return torch.stack(parts[name]).mean() if parts[name] else zero

        pair = mean("pair")
        contrastive = mean("contrastive")
        relational = mean("relational")
        background = mean("background")
        total = (
            self.candidate.pair_weight * pair
            + self.candidate.contrastive_weight * contrastive
            + self.candidate.relational_weight * relational
            + self.candidate.background_weight * background
        )
        return RepairLossOutput(total, pair, contrastive, relational, background)


def _effective_rank(features: torch.Tensor) -> float:
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    if float(energy.sum()) <= 0:
        return 0.0
    probability = energy / energy.sum()
    return float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()))


def _candidate_metrics(
    students: Mapping[str, torch.Tensor],
    teachers: Mapping[str, torch.Tensor],
    base_tir: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    r1: list[float] = []
    r5: list[float] = []
    base_r5: list[float] = []
    rank_vs_teacher: list[float] = []
    rank_vs_base: list[float] = []
    p95_increases: list[float] = []
    alignment_improvements: list[float] = []
    margins: list[float] = []
    for layer in DISCRIMINATIVE_LAYERS:
        query = F.normalize(students[layer].float(), dim=1)
        gallery = F.normalize(teachers[layer].float(), dim=1)
        base = F.normalize(base_tir[layer].float(), dim=1)
        similarity = query @ gallery.T
        base_similarity = base @ gallery.T
        order = similarity.argsort(dim=1, descending=True)
        base_order = base_similarity.argsort(dim=1, descending=True)
        target = torch.arange(similarity.shape[0], device=similarity.device)[:, None]
        r1.append(float((order[:, :1] == target).any(dim=1).float().mean()))
        r5.append(float((order[:, : min(5, order.shape[1])] == target).any(dim=1).float().mean()))
        base_r5.append(
            float(
                (base_order[:, : min(5, base_order.shape[1])] == target)
                .any(dim=1)
                .float()
                .mean()
            )
        )
        student_rank = _effective_rank(students[layer])
        teacher_rank = _effective_rank(teachers[layer])
        base_rank = _effective_rank(base_tir[layer])
        rank_vs_teacher.append(student_rank / max(teacher_rank, 1e-8))
        rank_vs_base.append(student_rank / max(base_rank, 1e-8))
        mask = ~torch.eye(similarity.shape[0], dtype=torch.bool)
        nonpaired = similarity[mask]
        base_nonpaired = base_similarity[mask]
        adapted_p95 = float(torch.quantile(nonpaired, 0.95)) if nonpaired.numel() else 0.0
        base_p95 = (
            float(torch.quantile(base_nonpaired, 0.95)) if base_nonpaired.numel() else 0.0
        )
        p95_increases.append(adapted_p95 - base_p95)
        adapted_alignment = float((1.0 - similarity.diag()).mean())
        base_alignment = float((1.0 - base_similarity.diag()).mean())
        alignment_improvements.append(
            (base_alignment - adapted_alignment) / max(abs(base_alignment), 1e-8)
        )
        shuffled = torch.roll(torch.arange(similarity.shape[0]), shifts=1)
        margins.append(
            float((similarity.diag() - similarity[torch.arange(similarity.shape[0]), shuffled]).mean())
        )
    return {
        "mean_r_at_1": sum(r1) / len(r1),
        "mean_r_at_5": sum(r5) / len(r5),
        "mean_base_r_at_5": sum(base_r5) / len(base_r5),
        "maximum_r5_drop": max(base - adapted for base, adapted in zip(base_r5, r5)),
        "maximum_r5_gain": max(adapted - base for base, adapted in zip(base_r5, r5)),
        "minimum_effective_rank_ratio_vs_teacher": min(rank_vs_teacher),
        "minimum_effective_rank_ratio_vs_base": min(rank_vs_base),
        "maximum_nonpaired_cosine_p95_increase": max(p95_increases),
        "minimum_alignment_improvement": min(alignment_improvements),
        "minimum_paired_shuffled_margin": min(margins),
    }


def build_recovery_diagnostics(
    *,
    records: Sequence[RGBTRecord],
    students: Mapping[str, torch.Tensor],
    teachers: Mapping[str, torch.Tensor],
    base_tir: Mapping[str, torch.Tensor],
    top_k: int = 20,
) -> dict[str, Any]:
    """Build absolute, per-layer probe evidence without hiding small denominators."""

    count = len(records)
    if count < 2:
        raise ValueError("recovery diagnostics require at least two records")
    expected_ids = [record.record_id for record in records]
    per_record = [
        {
            "record_id": record.record_id,
            "image_pair_key": record.image_pair_key,
            "source_dataset": record.source_dataset,
            "illumination": record.illumination,
            "weather": record.weather,
            "object_size": record.object_size,
            "occlusion": record.occlusion,
            "query_original": record.query_original,
            "layers": {},
        }
        for record in records
    ]
    layers: dict[str, Any] = {}
    for layer in LAYER_NAMES:
        student = F.normalize(students[layer].float(), dim=1)
        teacher = F.normalize(teachers[layer].float(), dim=1)
        base = F.normalize(base_tir[layer].float(), dim=1)
        if student.shape != teacher.shape or base.shape != teacher.shape:
            raise ValueError(f"recovery feature shape mismatch at layer {layer}")
        if student.shape[0] != count:
            raise ValueError(f"recovery record/feature mismatch at layer {layer}")
        similarity = student @ teacher.T
        base_similarity = base @ teacher.T
        paired = similarity.diag()
        base_paired = base_similarity.diag()
        adapted_loss = 1.0 - paired
        base_loss = 1.0 - base_paired
        absolute_delta = adapted_loss - base_loss
        relative_improvement = (base_loss - adapted_loss) / base_loss.abs().clamp_min(1e-8)
        search = similarity.clone()
        search.fill_diagonal_(-math.inf)
        neighbor_count = min(max(1, int(top_k)), count - 1)
        neighbor_indices = search.topk(neighbor_count, dim=1).indices
        layers[layer] = {
            "base_alignment_loss_mean": float(base_loss.mean()),
            "adapted_alignment_loss_mean": float(adapted_loss.mean()),
            "absolute_alignment_delta_mean": float(absolute_delta.mean()),
            "absolute_alignment_delta_p95": float(torch.quantile(absolute_delta, 0.95)),
            "relative_alignment_improvement_mean": float(relative_improvement.mean()),
            "base_paired_cosine_mean": float(base_paired.mean()),
            "adapted_paired_cosine_mean": float(paired.mean()),
            "adapted_effective_rank": _effective_rank(students[layer]),
            "base_effective_rank": _effective_rank(base_tir[layer]),
            "teacher_effective_rank": _effective_rank(teachers[layer]),
        }
        for index, row in enumerate(per_record):
            row["layers"][layer] = {
                "base_alignment_loss": float(base_loss[index]),
                "adapted_alignment_loss": float(adapted_loss[index]),
                "absolute_alignment_delta": float(absolute_delta[index]),
                "relative_alignment_improvement": float(relative_improvement[index]),
                "base_paired_cosine": float(base_paired[index]),
                "adapted_paired_cosine": float(paired[index]),
                "top_neighbor_record_ids": [
                    expected_ids[int(value)] for value in neighbor_indices[index]
                ],
            }
    return {
        "record_count": count,
        "top_k": min(max(1, int(top_k)), count - 1),
        "layers": layers,
        "per_record": per_record,
    }


class Phase16RepairTrainer:
    """Run fair C0/C1/C2 repair probes and retrain the selected candidate."""

    def __init__(
        self,
        *,
        adapter_factory: Callable[[], nn.Module],
        encode_tir: Callable[[nn.Module, RGBTRecord], Mapping[str, torch.Tensor]],
        output_root: Path | str,
        config: Phase16Config | None = None,
        adapter_state: Callable[[nn.Module], Mapping[str, torch.Tensor]] | None = None,
        load_adapter_state: Callable[[nn.Module, Mapping[str, torch.Tensor]], Any] | None = None,
        resume: bool = False,
        candidate_specs: Sequence[CandidateSpec] | None = None,
        objective_factory: Callable[[CandidateSpec], nn.Module] | None = None,
        stop_after_probe: bool = False,
        on_candidate_complete: Callable[[str, Path], Any] | None = None,
    ) -> None:
        self.adapter_factory = adapter_factory
        self.encode_tir = encode_tir
        self.output_root = Path(output_root)
        self.config = config or Phase16Config()
        self.adapter_state = adapter_state or (
            lambda module: {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}
        )
        self.load_adapter_state = load_adapter_state or (
            lambda module, state: module.load_state_dict(state)
        )
        self.resume = bool(resume)
        self.candidate_specs = tuple(
            candidate_specs
            if candidate_specs is not None
            else (CandidateSpec.c0(), CandidateSpec.c1(), CandidateSpec.c2())
        )
        self.objective_factory = objective_factory or (
            lambda candidate: Phase16RepairObjective(
                candidate,
                temperature=self.config.temperature,
                margin=self.config.margin,
            )
        )
        if not self.candidate_specs:
            raise ValueError("at least one repair candidate is required")
        names = [candidate.name for candidate in self.candidate_specs]
        if len(set(names)) != len(names):
            raise ValueError(f"repair candidate names must be unique: {names}")
        self.stop_after_probe = bool(stop_after_probe)
        self.on_candidate_complete = on_candidate_complete

    def _negative_layers(
        self,
        bank: RGBTeacherBank,
        record_ids: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        return {
            layer: torch.stack([bank[record_id].foreground[layer] for record_id in record_ids])
            for layer in LAYER_NAMES
        }

    def _objective_kwargs(
        self,
        *,
        positive: TeacherEntry,
        bank: RGBTeacherBank,
        anchor_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Build optional objective inputs without changing legacy objectives."""

        del bank, anchor_ids
        return {"base_tir": positive.base_tir or positive.foreground}

    def _schedule(self, step: int, total: int) -> float:
        warmup = max(1, round(total * self.config.warmup_fraction))
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def _train(
        self,
        module: nn.Module,
        candidate: CandidateSpec,
        records: Sequence[RGBTRecord],
        bank: RGBTeacherBank,
        *,
        steps: int,
        checkpoint_dir: Path | None = None,
    ) -> None:
        if not records:
            raise ValueError("repair training records are empty")
        sampler = DeterministicNegativeSampler(records, seed=self.config.seed)
        objective = self.objective_factory(candidate)
        trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("repair adapter contains no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        initial_lrs = [group["lr"] for group in optimizer.param_groups]
        start_step = 0
        if self.resume and checkpoint_dir is not None and checkpoint_dir.is_dir():
            existing = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
            if existing:
                payload = torch.load(existing[-1], map_location="cpu", weights_only=False)
                if payload.get("candidate") != candidate.name:
                    raise RuntimeError("resume checkpoint candidate mismatch")
                if payload.get("teacher_bank_fingerprint") != bank.fingerprint:
                    raise RuntimeError("resume checkpoint teacher bank fingerprint mismatch")
                self.load_adapter_state(module, payload["adapter"])
                optimizer.load_state_dict(payload["optimizer"])
                start_step = int(payload["step"])
        order = list(records)
        random.Random(self.config.seed).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        accumulation = max(1, self.config.gradient_accumulation)
        checkpoints = set()
        for fraction in self.config.checkpoint_fractions:
            raw = max(1, min(steps, round(steps * fraction)))
            aligned = min(steps, int(math.ceil(raw / accumulation) * accumulation))
            checkpoints.add(aligned)
        for step in range(start_step, steps):
            record = order[step % len(order)]
            positive = bank[record.record_id]
            needs_negatives = (
                candidate.contrastive_weight > 0 or candidate.relational_weight > 0
            )
            negative_ids = (
                sampler.sample(
                    record,
                    total=self.config.negative_count,
                    same_source=self.config.same_source_negatives,
                    same_condition=self.config.same_condition_negatives,
                )
                if needs_negatives
                else ()
            )
            anchor_ids = (
                negative_ids[: self.config.relational_anchor_count]
                if candidate.relational_weight > 0
                else ()
            )
            output = objective(
                self.encode_tir(module, record),
                positive.foreground,
                positive.background,
                self._negative_layers(bank, negative_ids) if negative_ids else {},
                self._negative_layers(bank, anchor_ids) if anchor_ids else {},
                **self._objective_kwargs(
                    positive=positive,
                    bank=bank,
                    anchor_ids=anchor_ids,
                ),
            )
            (output.total / self.config.gradient_accumulation).backward()
            completed = step + 1
            if completed % self.config.gradient_accumulation == 0 or completed == steps:
                torch.nn.utils.clip_grad_norm_(trainable, self.config.max_grad_norm)
                scale = self._schedule(step, steps)
                for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
                    group["lr"] = initial_lr * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if checkpoint_dir is not None and completed in checkpoints:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "candidate": candidate.name,
                        "step": completed,
                        "teacher_bank_fingerprint": bank.fingerprint,
                        "adapter": self.adapter_state(module),
                        "optimizer": optimizer.state_dict(),
                    },
                    checkpoint_dir / f"checkpoint_{completed:08d}.pt",
                )

    @torch.inference_mode()
    def _evaluate(
        self,
        module: nn.Module,
        records: Sequence[RGBTRecord],
        bank: RGBTeacherBank,
    ) -> dict[str, float]:
        students = {layer: [] for layer in LAYER_NAMES}
        teachers = {layer: [] for layer in LAYER_NAMES}
        base_tir = {layer: [] for layer in LAYER_NAMES}
        for record in records:
            encoded = self.encode_tir(module, record)
            positive = bank[record.record_id]
            for layer in LAYER_NAMES:
                students[layer].append(encoded[layer].detach().cpu())
                teachers[layer].append(positive.foreground[layer].detach().cpu())
                base = positive.base_tir or positive.foreground
                base_tir[layer].append(base[layer].detach().cpu())
        return _candidate_metrics(
            {layer: torch.stack(values) for layer, values in students.items()},
            {layer: torch.stack(values) for layer, values in teachers.items()},
            {layer: torch.stack(values) for layer, values in base_tir.items()},
        )

    @staticmethod
    def _selection_key(metrics: Mapping[str, float]) -> tuple[float, float, float]:
        return (
            float(metrics["mean_r_at_5"]),
            float(metrics["mean_r_at_1"]),
            float(metrics["minimum_effective_rank_ratio_vs_base"]),
        )

    def _passes_probe_gate(self, metrics: Mapping[str, float]) -> bool:
        cfg = self.config
        return bool(
            metrics["minimum_alignment_improvement"] >= cfg.minimum_alignment_improvement
            and metrics["maximum_r5_drop"] <= cfg.maximum_r5_drop
            and metrics["maximum_r5_gain"] >= cfg.minimum_retrieval_gain
            and metrics["minimum_effective_rank_ratio_vs_base"]
            >= cfg.minimum_effective_rank_vs_base
            and metrics["minimum_effective_rank_ratio_vs_teacher"]
            >= cfg.minimum_effective_rank_vs_teacher
            and metrics["maximum_nonpaired_cosine_p95_increase"]
            <= cfg.maximum_nonpaired_cosine_p95_increase
            and metrics["minimum_paired_shuffled_margin"] > 0.0
        )

    def run(
        self,
        train_records: Sequence[RGBTRecord],
        dev_records: Sequence[RGBTRecord],
        rgb_teacher_bank: RGBTeacherBank,
        *,
        full_train_records: Sequence[RGBTRecord] | None = None,
    ) -> Phase16Result:
        full_records = train_records if full_train_records is None else full_train_records
        if not full_records:
            raise ValueError("full repair training records are empty")
        self.output_root.mkdir(parents=True, exist_ok=True)
        candidates = self.candidate_specs
        probe_metrics: dict[str, dict[str, float]] = {}
        for candidate in candidates:
            candidate_dir = self.output_root / "probe_candidates" / candidate.name
            summary_path = candidate_dir / "summary.json"
            adapter_path = candidate_dir / "adapter.pt"
            if self.resume and summary_path.is_file() and adapter_path.is_file():
                cached = json.loads(summary_path.read_text(encoding="utf-8"))
                if cached.get("teacher_bank_fingerprint") != rgb_teacher_bank.fingerprint:
                    raise RuntimeError(
                        f"probe cache teacher bank fingerprint mismatch: {candidate.name}"
                    )
                probe_metrics[candidate.name] = cached["metrics"]
                if self.on_candidate_complete is not None:
                    self.on_candidate_complete(candidate.name, candidate_dir)
                continue
            module = self.adapter_factory()
            self._train(module, candidate, train_records, rgb_teacher_bank, steps=self.config.probe_steps)
            probe_metrics[candidate.name] = self._evaluate(module, dev_records, rgb_teacher_bank)
            candidate_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.adapter_state(module), adapter_path)
            summary_path.write_text(
                json.dumps(
                    {
                        "candidate": candidate.name,
                        "teacher_bank_fingerprint": rgb_teacher_bank.fingerprint,
                        "metrics": probe_metrics[candidate.name],
                        "gate_passed": self._passes_probe_gate(probe_metrics[candidate.name]),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if self.on_candidate_complete is not None:
                self.on_candidate_complete(candidate.name, candidate_dir)
        if self.stop_after_probe:
            selected = max(
                candidates,
                key=lambda candidate: self._selection_key(probe_metrics[candidate.name]),
            )
            result = Phase16Result(
                status="PHASE_16_PROBES_RECOVERED",
                selected_candidate=selected.name,
                completed_candidates=tuple(candidate.name for candidate in candidates),
                probe_metrics=probe_metrics,
                selected_checkpoint="",
            )
            (self.output_root / "run_summary.json").write_text(
                json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return result
        eligible = [
            candidate
            for candidate in candidates
            if self._passes_probe_gate(probe_metrics[candidate.name])
        ]
        if not eligible:
            result = Phase16Result(
                status="PHASE_16_NO_GO",
                selected_candidate="",
                completed_candidates=tuple(candidate.name for candidate in candidates),
                probe_metrics=probe_metrics,
                selected_checkpoint="",
            )
            (self.output_root / "run_summary.json").write_text(
                json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return result
        selected = max(eligible, key=lambda candidate: self._selection_key(probe_metrics[candidate.name]))
        full_module = self.adapter_factory()
        full_steps = self.config.full_steps or len(full_records)
        checkpoint_dir = self.output_root / "full_training" / "checkpoints"
        self._train(
            full_module,
            selected,
            full_records,
            rgb_teacher_bank,
            steps=full_steps,
            checkpoint_dir=checkpoint_dir,
        )
        checkpoint_metrics: dict[Path, dict[str, float]] = {}
        for checkpoint_path in sorted(checkpoint_dir.glob("checkpoint_*.pt")):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.load_adapter_state(full_module, checkpoint["adapter"])
            checkpoint_metrics[checkpoint_path] = self._evaluate(
                full_module, dev_records, rgb_teacher_bank
            )
        eligible_checkpoints = [
            path
            for path, metrics in checkpoint_metrics.items()
            if self._passes_probe_gate(metrics)
        ]
        if not eligible_checkpoints:
            result = Phase16Result(
                status="PHASE_16_NO_GO",
                selected_candidate=selected.name,
                completed_candidates=tuple(candidate.name for candidate in candidates),
                probe_metrics=probe_metrics,
                selected_checkpoint="",
            )
            (self.output_root / "run_summary.json").write_text(
                json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return result
        best_checkpoint = max(
            eligible_checkpoints,
            key=lambda path: self._selection_key(checkpoint_metrics[path]),
        )
        best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
        self.load_adapter_state(full_module, best_payload["adapter"])
        selected_dir = self.output_root / "selected_adapter"
        selected_dir.mkdir(parents=True, exist_ok=True)
        selected_path = selected_dir / "adapter.pt"
        torch.save(
            {
                "candidate": selected.name,
                "adapter": self.adapter_state(full_module),
                "teacher_bank_fingerprint": rgb_teacher_bank.fingerprint,
                "source_checkpoint": str(best_checkpoint),
                "dev_metrics": checkpoint_metrics[best_checkpoint],
            },
            selected_path,
        )
        result = Phase16Result(
            status="PHASE_16_REPAIR_TRAINED",
            selected_candidate=selected.name,
            completed_candidates=tuple(candidate.name for candidate in candidates),
            probe_metrics=probe_metrics,
            selected_checkpoint=str(best_checkpoint),
        )
        (self.output_root / "run_summary.json").write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result
