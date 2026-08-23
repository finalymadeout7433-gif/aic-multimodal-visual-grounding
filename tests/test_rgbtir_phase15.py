from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn

from aic_rgbtir.data import RGBTRecord
from aic_rgbtir.modeling import Qwen3VLRGBTAdapter
from aic_rgbtir.modeling import RGBTTIRValidationFeatures
from aic_rgbtir.phase15 import Phase15Config, Phase15Validator


class _ModeAwareBlock(nn.Module):
    def __init__(self, owner: "_TripletVisual") -> None:
        super().__init__()
        self.owner = owner

    def forward(self, hidden: torch.Tensor, **_: object) -> torch.Tensor:
        delta = {"rgb": 0.0, "base": 1.0, "tir": 2.0}[self.owner.active_modality]
        return hidden + delta


class _TripletVisual(nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.active_modality = "rgb"
        self.forward_modalities: list[str] = []
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.blocks = nn.ModuleList([_ModeAwareBlock(self) for _ in range(25)])
        self.deepstack_visual_indexes = [8, 16, 24]
        self.deepstack_merger_list = nn.ModuleList([nn.Identity() for _ in range(3)])
        self.merger = nn.Identity()
        self.spatial_merge_size = 2
        self.patch_size = 16

    def forward(
        self, hidden: torch.Tensor, grid_thw: torch.Tensor, **_: object
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del grid_thw
        self.forward_modalities.append(self.active_modality)
        deep: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if index in self.deepstack_visual_indexes:
                deep.append(hidden)
        return hidden, deep


class _TripletRouter:
    def __init__(self, visual: _TripletVisual) -> None:
        self.visual = visual

    @contextmanager
    def use(self, modality: str):
        previous = self.visual.active_modality
        self.visual.active_modality = modality
        try:
            yield
        finally:
            self.visual.active_modality = previous


def test_encode_tir_validation_triplet_runs_isolated_frozen_paths() -> None:
    visual = _TripletVisual()
    adapter = Qwen3VLRGBTAdapter(
        visual,
        hidden_size=4,
        adapter_router=_TripletRouter(visual),  # type: ignore[arg-type]
    )
    rgb = torch.zeros(4, 4, requires_grad=True)
    tir = torch.zeros(4, 4, requires_grad=True)
    grid = torch.tensor([[1, 2, 2]])

    triplet = adapter.encode_tir_validation_triplet(
        rgb_pixel_values=rgb,
        tir_pixel_values=tir,
        image_grid_thw=grid,
    )

    assert visual.forward_modalities == ["rgb", "base", "tir"]
    assert triplet.layer_names == ("8", "16", "24", "final")
    assert len(triplet.rgb_premerger) == 4
    assert len(triplet.tir_base_premerger) == 4
    assert len(triplet.tir_adapted_premerger) == 4
    assert all(not value.requires_grad for value in triplet.rgb_premerger)
    assert all(not value.requires_grad for value in triplet.tir_base_premerger)
    assert all(not value.requires_grad for value in triplet.tir_adapted_premerger)
    assert not torch.equal(
        triplet.tir_base_premerger[0], triplet.tir_adapted_premerger[0]
    )
    assert torch.equal(triplet.image_grid_thw, grid)


def test_encode_tir_validation_triplet_degrades_to_rgb_when_ir_is_unusable() -> None:
    visual = _TripletVisual()
    adapter = Qwen3VLRGBTAdapter(
        visual,
        hidden_size=4,
        adapter_router=_TripletRouter(visual),  # type: ignore[arg-type]
    )
    rgb = torch.zeros(4, 4)
    tir = torch.ones(4, 4)
    grid = torch.tensor([[1, 2, 2]])

    triplet = adapter.encode_tir_validation_triplet(
        rgb_pixel_values=rgb,
        tir_pixel_values=tir,
        image_grid_thw=grid,
        ir_usable=False,
    )

    assert visual.forward_modalities == ["rgb"]
    assert all(
        torch.equal(rgb_layer, base_layer)
        and torch.equal(rgb_layer, adapted_layer)
        for rgb_layer, base_layer, adapted_layer in zip(
            triplet.rgb_premerger,
            triplet.tir_base_premerger,
            triplet.tir_adapted_premerger,
        )
    )


def _record(pair: int, target: int) -> RGBTRecord:
    boxes = (
        (0.00, 0.00, 0.49, 0.49),
        (0.51, 0.00, 1.00, 0.49),
        (0.00, 0.51, 0.49, 1.00),
        (0.51, 0.51, 1.00, 1.00),
    )
    box = boxes[target]
    return RGBTRecord(
        record_id=f"pair{pair}:target{target}",
        source_dataset="synthetic",
        split="val",
        rgb_relpath=f"rgb/pair{pair}.png",
        tir_relpath=f"tir/pair{pair}.png",
        query_original="synthetic target",
        bbox_xywh_pixel=(box[0] * 100, box[1] * 100, (box[2] - box[0]) * 100, (box[3] - box[1]) * 100),
        bbox_xyxy_normalized=box,
        width=100,
        height=100,
        illumination="WL",
        weather="FY",
        object_size="SS",
        occlusion="NO",
        crowded="NC",
        scene="UB",
    )


class _SyntheticProcessor:
    spatial_merge_size = 1

    def __init__(self, *, ir_usable: bool = True) -> None:
        self.calls: list[str] = []
        self.ir_usable = ir_usable

    def process(self, record: RGBTRecord, *, root: Path):
        del root
        self.calls.append(record.image_pair_key)
        pair = int(Path(record.rgb_relpath).stem.removeprefix("pair"))
        return SimpleNamespace(
            rgb_pixel_values=torch.tensor([[float(pair)]]),
            tir_pixel_values=torch.tensor([[float(pair)]]),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            ir_patch_valid_mask=torch.ones(4),
            ir_usable=self.ir_usable,
            ir_valid_ratio=1.0 if self.ir_usable else 0.2,
        )


class _SyntheticEncoder:
    def __init__(self, *, collapse: bool = False) -> None:
        self.calls = 0
        self.collapse = collapse

    def __call__(self, batch) -> RGBTTIRValidationFeatures:
        self.calls += 1
        pair = int(batch.rgb_pixel_values.item())
        rgb = torch.zeros(4, 8)
        for token in range(4):
            rgb[token, pair * 4 + token] = 1.0
        if batch.ir_usable:
            base = torch.roll(rgb, shifts=4, dims=1)
            adapted = torch.ones_like(rgb) if self.collapse else rgb.clone()
        else:
            base = rgb.clone()
            adapted = rgb.clone()
        return RGBTTIRValidationFeatures(
            layer_names=("8", "16", "24", "final"),
            rgb_premerger=(rgb, rgb, rgb, rgb),
            tir_base_premerger=(base, base, base, base),
            tir_adapted_premerger=(adapted, adapted, adapted, adapted),
            image_grid_thw=batch.image_grid_thw,
        )


def _synthetic_records() -> list[RGBTRecord]:
    return [_record(0, 0), _record(0, 1), _record(1, 2), _record(1, 3)]


def test_phase15_validator_retrieval_is_deterministic_and_encodes_pairs_once(
    tmp_path: Path,
) -> None:
    processor = _SyntheticProcessor()
    encoder = _SyntheticEncoder()
    config = Phase15Config(
        expected_record_count=4,
        expected_pair_count=2,
        bootstrap_iterations=50,
        cache_checkpoint_pairs=1,
    )
    first = Phase15Validator(
        processor=processor,
        encode_triplet=encoder,
        output_root=tmp_path / "first",
        config=config,
        safety_checks=lambda: {"passed": True, "checks": {}},
    ).run(_synthetic_records(), tmp_path)

    assert processor.calls == ["synthetic/pair0.png", "synthetic/pair1.png"]
    assert encoder.calls == 2
    assert first.completed_record_count == 4
    assert first.completed_pair_count == 2
    assert first.retrieval_metrics["8"]["adapted"]["r_at_1"] == 1.0
    assert first.retrieval_metrics["8"]["base"]["r_at_1"] < 1.0
    assert first.retrieval_metrics["8"]["adapted"]["paired_shuffled_margin"] > 0

    second = Phase15Validator(
        processor=_SyntheticProcessor(),
        encode_triplet=_SyntheticEncoder(),
        output_root=tmp_path / "second",
        config=config,
        safety_checks=lambda: {"passed": True, "checks": {}},
    ).run(_synthetic_records(), tmp_path)
    assert first.retrieval_metrics == second.retrieval_metrics
    assert first.collapse_metrics == second.collapse_metrics


def test_phase15_validator_counts_ir_unusable_as_same_model_rgb_degradation(
    tmp_path: Path,
) -> None:
    result = Phase15Validator(
        processor=_SyntheticProcessor(ir_usable=False),
        encode_triplet=_SyntheticEncoder(),
        output_root=tmp_path,
        config=Phase15Config(
            expected_record_count=4,
            expected_pair_count=2,
            bootstrap_iterations=20,
        ),
        safety_checks=lambda: {"passed": True, "checks": {}},
    ).run(_synthetic_records(), tmp_path)

    assert result.completed_record_count == 4
    assert result.completed_pair_count == 2
    assert result.skipped_record_count == 0
    rows = [
        __import__("json").loads(line)
        for line in (tmp_path / "per_record_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["ir_usage_mode"] for row in rows} == {"rgb_only_degradation"}


def test_phase15_validator_hard_gate_rejects_collapsed_adaptation(tmp_path: Path) -> None:
    result = Phase15Validator(
        processor=_SyntheticProcessor(),
        encode_triplet=_SyntheticEncoder(collapse=True),
        output_root=tmp_path,
        config=Phase15Config(
            expected_record_count=4,
            expected_pair_count=2,
            bootstrap_iterations=50,
        ),
        safety_checks=lambda: {"passed": True, "checks": {}},
    ).run(_synthetic_records(), tmp_path)

    assert result.status == "PHASE_15_NO_GO"
    assert any("effective_rank" in reason for reason in result.gate_failures)


def test_phase15_validator_resumes_matching_pair_cache_without_reencoding(
    tmp_path: Path,
) -> None:
    config = Phase15Config(
        expected_record_count=4,
        expected_pair_count=2,
        bootstrap_iterations=20,
        fingerprint="FIXED-FINGERPRINT",
    )
    Phase15Validator(
        processor=_SyntheticProcessor(),
        encode_triplet=_SyntheticEncoder(),
        output_root=tmp_path,
        config=config,
        safety_checks=lambda: {"passed": True, "checks": {}},
    ).run(_synthetic_records(), tmp_path)
    resumed_processor = _SyntheticProcessor()
    resumed_encoder = _SyntheticEncoder()
    resumed = Phase15Validator(
        processor=resumed_processor,
        encode_triplet=resumed_encoder,
        output_root=tmp_path,
        config=config,
        safety_checks=lambda: {"passed": True, "checks": {}},
    ).run(_synthetic_records(), tmp_path)

    assert resumed.completed_record_count == 4
    assert resumed_processor.calls == []
    assert resumed_encoder.calls == 0


def test_phase15_validator_refuses_cache_with_different_fingerprint(
    tmp_path: Path,
) -> None:
    first_config = Phase15Config(
        expected_record_count=4,
        expected_pair_count=2,
        bootstrap_iterations=20,
        fingerprint="FIRST",
    )
    Phase15Validator(
        processor=_SyntheticProcessor(),
        encode_triplet=_SyntheticEncoder(),
        output_root=tmp_path,
        config=first_config,
        safety_checks=lambda: {"passed": True, "checks": {}},
    ).run(_synthetic_records(), tmp_path)
    second_processor = _SyntheticProcessor()
    refused = Phase15Validator(
        processor=second_processor,
        encode_triplet=_SyntheticEncoder(),
        output_root=tmp_path,
        config=Phase15Config(
            expected_record_count=4,
            expected_pair_count=2,
            bootstrap_iterations=20,
            fingerprint="SECOND",
        ),
        safety_checks=lambda: {"passed": True, "checks": {}},
    ).run(_synthetic_records(), tmp_path)

    assert refused.status == "PHASE_15_NO_GO"
    assert refused.completed_record_count == 0
    assert second_processor.calls == []
    assert any("fingerprint" in row["error"] for row in _read_csv(tmp_path / "failure_cases.csv"))


def test_phase15_validator_emits_bounded_extraction_telemetry(tmp_path: Path) -> None:
    snapshots = iter(
        [
            {"rss_bytes": 100, "cuda_allocated_bytes": 10},
            {"rss_bytes": 120, "cuda_allocated_bytes": 10},
        ]
    )
    result = Phase15Validator(
        processor=_SyntheticProcessor(),
        encode_triplet=_SyntheticEncoder(),
        output_root=tmp_path,
        config=Phase15Config(
            expected_record_count=4,
            expected_pair_count=2,
            bootstrap_iterations=20,
            cache_checkpoint_pairs=1,
        ),
        safety_checks=lambda: {"passed": True, "checks": {}},
        resource_probe=lambda: next(snapshots),
    ).run(_synthetic_records(), tmp_path)

    events = [
        __import__("json").loads(line)
        for line in (tmp_path / "resource_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result.completed_pair_count == 2
    assert [event["completed_pairs"] for event in events] == [1, 2]
    assert [event["rss_bytes"] for event in events] == [100, 120]


def test_phase15_pair_cache_uses_compact_pooled_features(tmp_path: Path) -> None:
    Phase15Validator(
        processor=_SyntheticProcessor(),
        encode_triplet=_SyntheticEncoder(),
        output_root=tmp_path,
        config=Phase15Config(
            expected_record_count=4,
            expected_pair_count=2,
            bootstrap_iterations=20,
        ),
        safety_checks=lambda: {"passed": True, "checks": {}},
    ).run(_synthetic_records(), tmp_path)

    first_part = next((tmp_path / "embedding_cache_parts").glob("*.pt"))
    payload = torch.load(first_part, map_location="cpu", weights_only=False)
    tensors = [
        tensor
        for row in payload["rows"]
        for variant in row["pooled"].values()
        for tensor in variant.values()
    ]
    assert tensors
    assert {tensor.dtype for tensor in tensors} == {torch.float16}


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
