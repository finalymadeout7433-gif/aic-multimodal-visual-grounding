from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from aic_rgbtir.data import RGBTRecord
from aic_rgbtir.phase18_full import (
    CheckpointArtifact,
    Phase18FullConfig,
    Phase18FullRunner,
)


def _record(record_id: str, split: str) -> RGBTRecord:
    return RGBTRecord(
        record_id=record_id,
        source_dataset="flir",
        split=split,
        rgb_relpath=f"image_data/flir/rgb/{record_id}.jpg",
        tir_relpath=f"image_data/flir/ir/{record_id}.jpg",
        query_original="The person",
        bbox_xywh_pixel=(1.0, 1.0, 5.0, 5.0),
        bbox_xyxy_normalized=(0.1, 0.1, 0.2, 0.2),
        width=100,
        height=100,
        illumination="NL",
        weather="FY",
        object_size="SO",
        occlusion="NO",
        crowded="NC",
        scene="UB",
    )


def _metrics(*, r5: float, drift24: float, r1: float = 0.62) -> dict:
    return {
        "mean_r_at_1": r1,
        "mean_r_at_5": r5,
        "layer_r_at_5": {"8": r5, "16": r5 + 0.01, "24": r5 + 0.02},
        "layer_drift": {"8": 0.08, "16": 0.10, "24": drift24},
        "minimum_effective_rank_ratio_vs_base": 0.88,
        "minimum_effective_rank_ratio_vs_teacher": 0.80,
        "maximum_nonpaired_cosine_p95_increase": 0.0,
        "minimum_paired_shuffled_margin": 0.15,
    }


class _Backend:
    def __init__(self, root: Path, dev_metrics: dict[int, dict], official: dict | None = None):
        self.root = root
        self.dev_metrics = dev_metrics
        self.official = official or {
            **_metrics(r5=0.85, drift24=0.20),
            "completed_records": 2,
            "completed_pairs": 2,
            "skipped_records": 0,
            "paired_margin_bootstrap_lower": 0.10,
            "maximum_subgroup_degradation": 0.0,
            "safety_passed": True,
        }
        self.events: list[str] = []
        self.module: nn.Module | None = None

    def fresh_adapter(self) -> nn.Module:
        self.events.append("fresh")
        self.module = nn.Linear(2, 2, bias=False)
        return self.module

    def train_full(self, module, records, checkpoint_dir, fractions):
        self.events.append(f"train:{len(records)}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, fraction in enumerate(fractions, start=1):
            path = checkpoint_dir / f"checkpoint_{index}.pt"
            torch.save({"step": index, "adapter": module.state_dict()}, path)
            rows.append(CheckpointArtifact(step=index, fraction=fraction, path=path))
        return tuple(rows)

    def load_checkpoint(self, module, checkpoint):
        self.events.append(f"load:{checkpoint.step}")

    def evaluate(self, module, records):
        step = int(self.events[-1].split(":")[1])
        self.events.append(f"dev:{step}:{len(records)}")
        return self.dev_metrics[step]

    def release(self, module, checkpoint, output_path, metadata):
        self.events.append(f"release:{checkpoint.step}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": checkpoint.step, "metadata": dict(metadata)}, output_path)

    def validate_official(self, module, records):
        self.events.append(f"official:{len(records)}")
        return self.official


def test_full_runner_selects_only_from_dev_then_opens_official_once(tmp_path: Path) -> None:
    backend = _Backend(
        tmp_path,
        {
            1: _metrics(r5=0.84, drift24=0.23),
            2: _metrics(r5=0.85, drift24=0.22),
            3: _metrics(r5=0.85, drift24=0.19),
            4: _metrics(r5=0.85, drift24=0.19),
        },
    )
    runner = Phase18FullRunner(
        backend=backend,
        output_root=tmp_path / "out",
        config=Phase18FullConfig(
            expected_full_records=2,
            expected_dev_records=2,
            expected_official_records=2,
            expected_official_pairs=2,
        ),
    )

    result = runner.run(
        [_record("f1", "train"), _record("f2", "train")],
        [_record("d1", "dev"), _record("d2", "dev")],
        [_record("v1", "val"), _record("v2", "val")],
        fingerprint="FINGERPRINT-A",
    )

    assert result.status == "PHASE_18_FULL_GO"
    assert result.selected_checkpoint_step == 3
    assert backend.events.index("release:3") < backend.events.index("official:2")
    assert backend.events.count("official:2") == 1
    assert backend.events[0] == "fresh"


def test_full_runner_does_not_open_official_when_no_dev_checkpoint_passes(tmp_path: Path) -> None:
    failed = _metrics(r5=0.70, drift24=0.50, r1=0.50)
    backend = _Backend(tmp_path, {1: failed, 2: failed, 3: failed, 4: failed})
    runner = Phase18FullRunner(
        backend=backend,
        output_root=tmp_path / "out",
        config=Phase18FullConfig(
            expected_full_records=1,
            expected_dev_records=1,
            expected_official_records=1,
            expected_official_pairs=1,
        ),
    )

    result = runner.run(
        [_record("f", "train")],
        [_record("d", "dev")],
        [_record("v", "val")],
        fingerprint="FINGERPRINT-B",
    )

    assert result.status == "PHASE_18_FULL_NO_GO"
    assert result.selected_checkpoint_step == 0
    assert not any(event.startswith("official:") for event in backend.events)


def test_full_runner_rejects_fingerprint_drift_before_training(tmp_path: Path) -> None:
    output = tmp_path / "out"
    backend = _Backend(
        tmp_path,
        {step: _metrics(r5=0.85, drift24=0.2) for step in range(1, 5)},
    )
    config = Phase18FullConfig(
        expected_full_records=1,
        expected_dev_records=1,
        expected_official_records=1,
        expected_official_pairs=1,
    )
    runner = Phase18FullRunner(backend=backend, output_root=output, config=config)
    runner.write_fingerprint("FIRST")

    try:
        runner.run(
            [_record("f", "train")],
            [_record("d", "dev")],
            [_record("v", "val")],
            fingerprint="SECOND",
        )
    except RuntimeError as error:
        assert "fingerprint mismatch" in str(error)
    else:
        raise AssertionError("fingerprint drift must be rejected")
    assert backend.events == []


def test_checkpoint_selection_honours_half_point_r5_tolerance(tmp_path: Path) -> None:
    runner = Phase18FullRunner(
        backend=_Backend(tmp_path, {}),
        output_root=tmp_path / "out",
        config=Phase18FullConfig(
            expected_full_records=1,
            expected_dev_records=1,
            expected_official_records=1,
            expected_official_pairs=1,
        ),
    )
    early = CheckpointArtifact(10, 0.25, tmp_path / "early.pt")
    late = CheckpointArtifact(20, 0.50, tmp_path / "late.pt")
    early_metrics = _metrics(r5=0.840, drift24=0.20)
    late_within_tolerance = _metrics(r5=0.844, drift24=0.30)

    selected, _ = runner._select_checkpoint(
        [(early, early_metrics), (late, late_within_tolerance)]
    )
    assert selected == early

    late_clear_gain = _metrics(r5=0.846, drift24=0.30)
    selected, _ = runner._select_checkpoint(
        [(early, early_metrics), (late, late_clear_gain)]
    )
    assert selected == late


def test_train_select_stage_persists_adapter_without_opening_official(tmp_path: Path) -> None:
    backend = _Backend(
        tmp_path,
        {step: _metrics(r5=0.85, drift24=0.20) for step in range(1, 5)},
    )
    runner = Phase18FullRunner(
        backend=backend,
        output_root=tmp_path / "out",
        config=Phase18FullConfig(
            expected_full_records=2,
            expected_dev_records=2,
            expected_official_records=2,
            expected_official_pairs=2,
        ),
    )

    selection = runner.train_select(
        [_record("f1", "train"), _record("f2", "train")],
        [_record("d1", "dev"), _record("d2", "dev")],
        [_record("v1", "val"), _record("v2", "val")],
        fingerprint="STAGED-A",
    )

    assert selection.status == "PHASE_18_TRAIN_SELECT_READY"
    assert selection.selected_checkpoint_step > 0
    assert Path(selection.selected_adapter).is_file()
    assert (tmp_path / "out" / "train_selection_summary.json").is_file()
    assert not any(event.startswith("official:") for event in backend.events)


def test_official_stage_finalizes_from_persisted_selection(tmp_path: Path) -> None:
    backend = _Backend(
        tmp_path,
        {step: _metrics(r5=0.85, drift24=0.20) for step in range(1, 5)},
    )
    runner = Phase18FullRunner(
        backend=backend,
        output_root=tmp_path / "out",
        config=Phase18FullConfig(
            expected_full_records=1,
            expected_dev_records=1,
            expected_official_records=1,
            expected_official_pairs=1,
        ),
    )
    selection = runner.train_select(
        [_record("f", "train")],
        [_record("d", "dev")],
        [_record("v", "val")],
        fingerprint="STAGED-B",
    )
    official = {
        **_metrics(r5=0.85, drift24=0.20),
        "completed_records": 1,
        "completed_pairs": 1,
        "skipped_records": 0,
        "paired_margin_bootstrap_lower": 0.10,
        "maximum_subgroup_degradation": 0.0,
        "safety_passed": True,
    }

    result = runner.complete_official(selection, official)

    assert result.status == "PHASE_18_FULL_GO"
    assert result.selected_checkpoint_step == selection.selected_checkpoint_step
    assert (tmp_path / "out" / "run_summary.json").is_file()
