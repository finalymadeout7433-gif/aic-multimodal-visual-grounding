from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from aic_rgbtir.data import RGBTRecord
from aic_rgbtir.modeling import Qwen3VLRGBTAdapter
from aic_rgbtir.phase16 import (
    CandidateSpec,
    DeterministicNegativeSampler,
    Phase16Config,
    Phase16RepairObjective,
    Phase16RepairTrainer,
    RGBTeacherBank,
    TeacherEntry,
    build_repair_split,
    build_recovery_diagnostics,
)


LAYERS = ("8", "16", "24", "final")


class _ModeBlock(nn.Module):
    def __init__(self, owner: "_RepairVisual") -> None:
        super().__init__()
        self.owner = owner
        self.weight = nn.Parameter(torch.eye(4))

    def forward(self, hidden: torch.Tensor, **_: object) -> torch.Tensor:
        delta = {"rgb": 0.0, "tir": 1.0, "base": 2.0}[self.owner.modality]
        return hidden @ self.weight + delta


class _RepairVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.modality = "rgb"
        self.calls: list[str] = []
        self.config = SimpleNamespace(hidden_size=4)
        self.blocks = nn.ModuleList([_ModeBlock(self) for _ in range(25)])
        self.deepstack_visual_indexes = [8, 16, 24]
        self.deepstack_merger_list = nn.ModuleList([nn.Identity() for _ in range(3)])
        self.merger = nn.Identity()
        self.spatial_merge_size = 2
        self.patch_size = 16

    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor, **_: object):
        del grid_thw
        self.calls.append(self.modality)
        deep = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if index in self.deepstack_visual_indexes:
                deep.append(hidden)
        return hidden, deep


class _ModeRouter:
    def __init__(self, visual: _RepairVisual) -> None:
        self.visual = visual

    @contextmanager
    def use(self, modality: str):
        previous = self.visual.modality
        self.visual.modality = modality
        try:
            yield
        finally:
            self.visual.modality = previous


def test_repair_feature_seam_encodes_rgb_teacher_once_and_keeps_only_tir_gradients() -> None:
    visual = _RepairVisual()
    adapter = Qwen3VLRGBTAdapter(
        visual,
        hidden_size=4,
        adapter_router=_ModeRouter(visual),  # type: ignore[arg-type]
    )
    grid = torch.tensor([[1, 2, 2]])
    rgb = torch.zeros(4, 4, requires_grad=True)
    tir = torch.zeros(4, 4, requires_grad=True)

    teacher = adapter.encode_rgb_teacher_features(
        rgb_pixel_values=rgb,
        image_grid_thw=grid,
    )
    repaired = adapter.encode_tir_repair_features(
        tir_pixel_values=tir,
        image_grid_thw=grid,
    )

    assert visual.calls == ["rgb", "tir"]
    assert teacher.layer_names == LAYERS
    assert repaired.layer_names == LAYERS
    assert all(not tensor.requires_grad for tensor in teacher.premerger)
    assert all(tensor.requires_grad for tensor in repaired.premerger)
    repaired.premerger[0].sum().backward()
    assert tir.grad is not None
    assert rgb.grad is None


def _record(
    index: int,
    *,
    source: str = "flir",
    split: str = "train",
    illumination: str = "NL",
    object_size: str = "SS",
) -> RGBTRecord:
    return RGBTRecord(
        record_id=f"{source}:{split}:{index}",
        source_dataset=source,
        split=split,
        rgb_relpath=f"image_data/{source}/rgb/pair_{index:05d}.png",
        tir_relpath=f"image_data/{source}/ir/pair_{index:05d}.png",
        query_original="the target",
        bbox_xywh_pixel=(10.0, 10.0, 20.0, 20.0),
        bbox_xyxy_normalized=(0.1, 0.1, 0.3, 0.3),
        width=100,
        height=100,
        illumination=illumination,
        weather="FY",
        object_size=object_size,
        occlusion="NO",
        crowded="NC",
        scene="UB",
    )


def _entry(record: RGBTRecord, vector: torch.Tensor) -> TeacherEntry:
    layers = {name: vector.clone() for name in LAYERS}
    return TeacherEntry(
        record_id=record.record_id,
        image_pair_key=record.image_pair_key,
        source_dataset=record.source_dataset,
        illumination=record.illumination,
        object_size=record.object_size,
        foreground=layers,
        background={name: -vector.clone() for name in LAYERS},
        base_tir={name: torch.roll(vector.clone(), shifts=1) for name in LAYERS},
    )


def test_repair_split_is_deterministic_pair_isolated_and_stratified() -> None:
    records: list[RGBTRecord] = []
    for index in range(80):
        source = ("flir", "m3fd", "mfad")[index % 3]
        illumination = ("NL", "SL")[index % 2]
        object_size = ("SS", "NS")[(index // 2) % 2]
        records.append(
            _record(
                index,
                source=source,
                illumination=illumination,
                object_size=object_size,
            )
        )
        records.append(replace(records[-1], record_id=f"{source}:train:{index}:second"))

    first = build_repair_split(records, probe_pairs=24, dev_pairs=16, seed=20260812)
    second = build_repair_split(records, probe_pairs=24, dev_pairs=16, seed=20260812)

    assert first == second
    assert len({record.image_pair_key for record in first.probe_train}) == 24
    assert len({record.image_pair_key for record in first.dev}) == 16
    assert not (
        {record.image_pair_key for record in first.probe_train}
        & {record.image_pair_key for record in first.dev}
    )
    assert not (
        {record.image_pair_key for record in first.full_train}
        & {record.image_pair_key for record in first.dev}
    )
    assert {record.source_dataset for record in first.dev} == {"flir", "m3fd", "mfad"}


def test_negative_sampler_is_deterministic_cross_pair_and_uses_requested_buckets() -> None:
    records = [
        _record(
            index,
            source=("flir", "m3fd", "mfad")[index % 3],
            illumination=("NL", "SL")[index % 2],
            object_size=("SS", "NS")[(index // 2) % 2],
        )
        for index in range(600)
    ]
    sampler = DeterministicNegativeSampler(records, seed=20260812)
    first = sampler.sample(records[0], total=256, same_source=128, same_condition=64)
    second = sampler.sample(records[0], total=256, same_source=128, same_condition=64)

    assert first == second
    assert len(first) == 256
    assert len(set(first)) == 256
    by_id = {record.record_id: record for record in records}
    assert all(by_id[record_id].image_pair_key != records[0].image_pair_key for record_id in first)
    assert sum(by_id[record_id].source_dataset == records[0].source_dataset for record_id in first) >= 128
    assert sum(
        by_id[record_id].illumination == records[0].illumination
        and by_id[record_id].object_size == records[0].object_size
        for record_id in first
    ) >= 64


def test_contrastive_and_relational_objectives_reward_correct_discriminative_geometry() -> None:
    positive = torch.tensor([1.0, 0.0, 0.0])
    negatives = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    anchors = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    background = torch.tensor([-1.0, 0.0, 0.0])
    matching = {name: positive.clone().requires_grad_(True) for name in LAYERS}
    conflicting = {
        name: torch.tensor([0.0, 1.0, 0.0], requires_grad=True) for name in LAYERS
    }
    teacher = {name: positive for name in LAYERS}
    negative_layers = {name: negatives for name in LAYERS}
    anchor_layers = {name: anchors for name in LAYERS}
    background_layers = {name: background for name in LAYERS}

    contrastive = Phase16RepairObjective(CandidateSpec.c1())
    relational = Phase16RepairObjective(CandidateSpec.c2())
    baseline = Phase16RepairObjective(CandidateSpec.c0())
    baseline_output = baseline(
        matching,
        teacher,
        background_layers,
        {},
        {},
    )
    good = contrastive(
        matching,
        teacher,
        background_layers,
        negative_layers,
        anchor_layers,
    )
    bad = contrastive(
        conflicting,
        teacher,
        background_layers,
        negative_layers,
        anchor_layers,
    )
    relational_good = relational(
        matching,
        teacher,
        background_layers,
        negative_layers,
        anchor_layers,
    )
    relational_bad = relational(
        conflicting,
        teacher,
        background_layers,
        negative_layers,
        anchor_layers,
    )

    assert good.total < bad.total
    assert relational_good.relational < relational_bad.relational
    assert torch.isfinite(baseline_output.total)
    assert baseline_output.contrastive == 0
    assert baseline_output.relational == 0
    good.total.backward()
    assert all(tensor.grad is not None for tensor in matching.values())


def test_teacher_bank_roundtrip_enforces_fingerprint(tmp_path: Path) -> None:
    records = [_record(0), _record(1)]
    bank = RGBTeacherBank(
        fingerprint="MODEL-PROCESSOR-MANIFEST-CODE",
        entries={
            record.record_id: _entry(record, torch.eye(2)[index])
            for index, record in enumerate(records)
        },
    )
    path = tmp_path / "teacher_bank.pt"
    bank.save(path)

    restored = RGBTeacherBank.load(path, expected_fingerprint=bank.fingerprint)
    assert restored.record_ids == tuple(record.record_id for record in records)
    with pytest.raises(RuntimeError, match="fingerprint"):
        RGBTeacherBank.load(path, expected_fingerprint="DIFFERENT")


class _ScaleAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, record_id: str) -> dict[str, torch.Tensor]:
        index = int(record_id.rsplit(":", 1)[-1])
        target = torch.eye(4)[index % 4]
        return {name: self.scale * target for name in LAYERS}


def test_trainer_restarts_candidates_from_identical_adapter_and_selects_by_dev_metrics(
    tmp_path: Path,
) -> None:
    records = [_record(index) for index in range(4)]
    full_records = [_record(index) for index in range(4, 10)]
    all_records = records + full_records
    bank = RGBTeacherBank(
        fingerprint="FIXED",
        entries={
            record.record_id: _entry(record, torch.eye(4)[index % 4])
            for index, record in enumerate(all_records)
        },
    )
    initial = _ScaleAdapter().state_dict()
    starts: list[dict[str, torch.Tensor]] = []
    modules: list[_ScaleAdapter] = []
    seen: dict[int, list[str]] = {}

    def factory() -> _ScaleAdapter:
        module = _ScaleAdapter()
        module.load_state_dict(initial)
        modules.append(module)
        seen[id(module)] = []
        starts.append({name: value.clone() for name, value in module.state_dict().items()})
        return module

    def encode(module: _ScaleAdapter, record: RGBTRecord) -> dict[str, torch.Tensor]:
        seen[id(module)].append(record.record_id)
        return module(record.record_id)

    trainer = Phase16RepairTrainer(
        adapter_factory=factory,
        encode_tir=encode,
        output_root=tmp_path,
        config=Phase16Config(
            probe_steps=4,
            full_steps=0,
            gradient_accumulation=1,
            negative_count=3,
            same_source_negatives=1,
            same_condition_negatives=1,
            relational_anchor_count=2,
            checkpoint_fractions=(0.5, 1.0),
            minimum_alignment_improvement=-1.0,
            maximum_r5_drop=1.0,
            minimum_retrieval_gain=-1.0,
            minimum_effective_rank_vs_base=0.0,
            minimum_effective_rank_vs_teacher=0.0,
            maximum_nonpaired_cosine_p95_increase=1.0,
        ),
    )
    result = trainer.run(
        records,
        records,
        bank,
        full_train_records=full_records,
    )

    assert len(starts) == 4  # C0/C1/C2 probes plus the full winner retrain.
    assert all(
        all(torch.equal(first[name], candidate[name]) for name in first)
        for first, candidate in zip(starts, starts[1:])
    )
    assert result.selected_candidate in {"C0", "C1", "C2"}
    assert result.completed_candidates == ("C0", "C1", "C2")
    assert set(seen[id(modules[-1])][: len(full_records)]) == {
        record.record_id for record in full_records
    }
    assert (tmp_path / "run_summary.json").is_file()
    assert (tmp_path / "selected_adapter" / "adapter.pt").is_file()


def test_recovery_probe_runs_only_c1_c2_and_never_enters_full_training(
    tmp_path: Path,
) -> None:
    records = [_record(index) for index in range(4)]
    full_records = [_record(index) for index in range(4, 10)]
    all_records = records + full_records
    bank = RGBTeacherBank(
        fingerprint="RECOVERY-FIXED",
        entries={
            record.record_id: _entry(record, torch.eye(4)[index % 4])
            for index, record in enumerate(all_records)
        },
    )
    calls: list[str] = []
    persisted: list[str] = []

    def factory() -> _ScaleAdapter:
        calls.append("fresh")
        return _ScaleAdapter()

    trainer = Phase16RepairTrainer(
        adapter_factory=factory,
        encode_tir=lambda module, record: module(record.record_id),
        output_root=tmp_path,
        config=Phase16Config(
            probe_steps=4,
            gradient_accumulation=1,
            negative_count=3,
            same_source_negatives=1,
            same_condition_negatives=1,
            relational_anchor_count=2,
        ),
        candidate_specs=(CandidateSpec.c1(), CandidateSpec.c2()),
        stop_after_probe=True,
        on_candidate_complete=lambda candidate, _directory: persisted.append(candidate),
    )
    result = trainer.run(records, records, bank, full_train_records=full_records)

    assert calls == ["fresh", "fresh"]
    assert persisted == ["C1", "C2"]
    assert result.status == "PHASE_16_PROBES_RECOVERED"
    assert result.completed_candidates == ("C1", "C2")
    assert result.selected_candidate in {"C1", "C2"}
    assert not (tmp_path / "full_training").exists()
    assert not (tmp_path / "selected_adapter").exists()
    assert (tmp_path / "probe_candidates" / "C1" / "adapter.pt").is_file()
    assert (tmp_path / "probe_candidates" / "C2" / "adapter.pt").is_file()


def test_recovery_diagnostics_preserve_absolute_layer_losses_and_top_neighbors() -> None:
    records = [_record(index) for index in range(3)]
    teachers = {
        layer: torch.eye(3) for layer in LAYERS
    }
    base = {
        layer: torch.tensor(
            [[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.0, 0.2, 0.8]]
        )
        for layer in LAYERS
    }
    adapted = {
        layer: torch.tensor(
            [[0.95, 0.05, 0.0], [0.0, 1.0, 0.0], [0.0, 0.05, 0.95]]
        )
        for layer in LAYERS
    }

    result = build_recovery_diagnostics(
        records=records,
        students=adapted,
        teachers=teachers,
        base_tir=base,
        top_k=2,
    )

    assert result["record_count"] == 3
    assert set(result["layers"]) == set(LAYERS)
    assert result["layers"]["8"]["adapted_alignment_loss_mean"] < result["layers"]["8"]["base_alignment_loss_mean"]
    assert len(result["per_record"]) == 3
    row = result["per_record"][0]
    assert row["layers"]["8"]["absolute_alignment_delta"] < 0
    assert row["layers"]["8"]["top_neighbor_record_ids"][0] != row["record_id"]


def test_checkpoint_resume_matches_continuous_training(tmp_path: Path) -> None:
    records = [_record(index) for index in range(4)]
    bank = RGBTeacherBank(
        fingerprint="RESUME-FIXED",
        entries={
            record.record_id: _entry(record, torch.eye(4)[index])
            for index, record in enumerate(records)
        },
    )
    config = Phase16Config(
        probe_steps=4,
        full_steps=4,
        gradient_accumulation=1,
        negative_count=3,
        same_source_negatives=1,
        same_condition_negatives=1,
        relational_anchor_count=2,
        checkpoint_fractions=(0.5, 1.0),
    )

    continuous = _ScaleAdapter()
    continuous_trainer = Phase16RepairTrainer(
        adapter_factory=_ScaleAdapter,
        encode_tir=lambda module, record: module(record.record_id),
        output_root=tmp_path / "continuous",
        config=config,
    )
    continuous_trainer._train(
        continuous,
        CandidateSpec.c1(),
        records,
        bank,
        steps=4,
        checkpoint_dir=tmp_path / "continuous-checkpoints",
    )

    interrupted = _ScaleAdapter()
    calls = 0

    def fail_after_checkpoint(
        module: _ScaleAdapter, record: RGBTRecord
    ) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return module(record.record_id)

    interrupted_dir = tmp_path / "interrupted-checkpoints"
    interrupted_trainer = Phase16RepairTrainer(
        adapter_factory=_ScaleAdapter,
        encode_tir=fail_after_checkpoint,
        output_root=tmp_path / "interrupted",
        config=config,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted_trainer._train(
            interrupted,
            CandidateSpec.c1(),
            records,
            bank,
            steps=4,
            checkpoint_dir=interrupted_dir,
        )

    resumed = _ScaleAdapter()
    resumed_trainer = Phase16RepairTrainer(
        adapter_factory=_ScaleAdapter,
        encode_tir=lambda module, record: module(record.record_id),
        output_root=tmp_path / "resumed",
        config=config,
        resume=True,
    )
    resumed_trainer._train(
        resumed,
        CandidateSpec.c1(),
        records,
        bank,
        steps=4,
        checkpoint_dir=interrupted_dir,
    )
    assert all(
        torch.equal(continuous.state_dict()[name], resumed.state_dict()[name])
        for name in continuous.state_dict()
    )


def test_resume_rejects_stale_probe_fingerprint(tmp_path: Path) -> None:
    records = [_record(index) for index in range(4)]
    bank = RGBTeacherBank(
        fingerprint="CURRENT",
        entries={
            record.record_id: _entry(record, torch.eye(4)[index])
            for index, record in enumerate(records)
        },
    )
    cache = tmp_path / "probe_candidates" / "C0"
    cache.mkdir(parents=True)
    (cache / "adapter.pt").write_bytes(b"placeholder")
    (cache / "summary.json").write_text(
        json.dumps(
            {
                "candidate": "C0",
                "teacher_bank_fingerprint": "STALE",
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    trainer = Phase16RepairTrainer(
        adapter_factory=_ScaleAdapter,
        encode_tir=lambda module, record: module(record.record_id),
        output_root=tmp_path,
        config=Phase16Config(
            probe_steps=1,
            negative_count=3,
            same_source_negatives=1,
            same_condition_negatives=1,
            relational_anchor_count=2,
        ),
        resume=True,
    )
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        trainer.run(records, records, bank)


def test_probe_gate_rejects_low_rank_candidate() -> None:
    trainer = Phase16RepairTrainer(
        adapter_factory=_ScaleAdapter,
        encode_tir=lambda module, record: module(record.record_id),
        output_root=Path("unused"),
    )
    metrics = {
        "minimum_alignment_improvement": 0.50,
        "maximum_r5_drop": 0.0,
        "maximum_r5_gain": 0.20,
        "minimum_effective_rank_ratio_vs_base": 0.20,
        "minimum_effective_rank_ratio_vs_teacher": 0.20,
        "maximum_nonpaired_cosine_p95_increase": 0.0,
        "minimum_paired_shuffled_margin": 0.20,
    }
    assert not trainer._passes_probe_gate(metrics)
