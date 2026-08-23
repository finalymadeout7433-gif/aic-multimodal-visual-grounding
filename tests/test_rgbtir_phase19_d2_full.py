from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from aic_rgbtir.data import RGBTRecord
from aic_rgbtir.phase18_full import CheckpointArtifact
from aic_rgbtir.phase19_d2_full import D2FullConfig, Phase19D2FullRunner


def _record(record_id: str, split: str, *, pair: str | None = None) -> RGBTRecord:
    stem = pair or record_id
    return RGBTRecord(
        record_id=record_id,
        source_dataset="flir",
        split=split,
        rgb_relpath=f"image_data/flir/rgb/{stem}.jpg",
        tir_relpath=f"image_data/flir/ir/{stem}.jpg",
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


BASELINE = {
    "semantic": {
        "mean_r_at_5": 0.878,
        "minimum_effective_rank_ratio_vs_base": 0.886,
        "maximum_nonpaired_cosine_p95_increase": -0.197,
    },
    "multiquery": {
        "mean_r_at_5": 0.896,
        "minimum_effective_rank_ratio_vs_base": 0.807,
        "maximum_nonpaired_cosine_p95_increase": -0.173,
    },
}


def _dev_metrics(*, r5: float = 0.90, rank: float = 0.87, p95: float = -0.18) -> dict:
    return {
        "mean_r_at_1": 0.72,
        "mean_r_at_5": r5,
        "minimum_effective_rank_ratio_vs_base": rank,
        "minimum_effective_rank_ratio_vs_teacher": 0.82,
        "maximum_nonpaired_cosine_p95_increase": p95,
        "minimum_paired_shuffled_margin": 0.16,
    }


def _official_metrics() -> dict:
    return {
        "completed_records": 2,
        "completed_pairs": 2,
        "skipped_records": 0,
        "layer_r_at_5": {"8": 0.92, "16": 0.91, "24": 0.76},
        "minimum_effective_rank_ratio_vs_base": 0.87,
        "minimum_effective_rank_ratio_vs_teacher": 0.80,
        "maximum_absolute_layer_drift": 0.018,
        "maximum_nonpaired_cosine_p95_increase": 0.01,
        "minimum_paired_shuffled_margin": 0.15,
        "paired_margin_bootstrap_lower": 0.05,
        "maximum_subgroup_degradation": 0.01,
        "safety_passed": True,
    }


class _Backend:
    def __init__(self, root: Path, metrics: dict[int, tuple[dict, dict]]) -> None:
        self.root = root
        self.metrics = metrics
        self.events: list[str] = []
        self.current_step = 0

    def fresh_adapter(self) -> nn.Module:
        self.events.append("fresh-from-d1")
        return nn.Linear(2, 2, bias=False)

    def train_full(self, module, records, checkpoint_dir, fractions):
        self.events.append(f"train:{len(records)}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result = []
        for step, fraction in enumerate(fractions, start=1):
            path = checkpoint_dir / f"checkpoint_{step}.pt"
            torch.save({"step": step, "adapter": module.state_dict()}, path)
            result.append(CheckpointArtifact(step, fraction, path))
        return tuple(result)

    def load_checkpoint(self, module, checkpoint):
        self.current_step = checkpoint.step
        self.events.append(f"load:{checkpoint.step}")

    def evaluate(self, module, records):
        split = "semantic" if len(records) == 1 else "multiquery"
        self.events.append(f"dev:{split}:{self.current_step}")
        return self.metrics[self.current_step][0 if split == "semantic" else 1]

    def release(self, module, checkpoint, output_path, metadata):
        self.events.append(f"release:{checkpoint.step}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": checkpoint.step, "metadata": dict(metadata)}, output_path)

    def validate_official(self, module, records):
        self.events.append(f"official:{len(records)}")
        return _official_metrics()


def _runner(tmp_path: Path, backend: _Backend) -> Phase19D2FullRunner:
    return Phase19D2FullRunner(
        backend=backend,
        output_root=tmp_path / "out",
        baseline_metrics=BASELINE,
        config=D2FullConfig(
            expected_full_records=2,
            expected_semantic_records=1,
            expected_semantic_pairs=1,
            expected_multiquery_records=2,
            expected_multiquery_pairs=1,
            expected_official_records=2,
            expected_official_pairs=2,
        ),
    )


def test_full_runner_opens_official_once_only_after_both_dev_gates_pass(tmp_path: Path) -> None:
    passing = (_dev_metrics(rank=0.94), _dev_metrics(rank=0.86))
    backend = _Backend(tmp_path, {step: passing for step in range(1, 5)})
    runner = _runner(tmp_path, backend)

    result = runner.run(
        [_record("f1", "train"), _record("f2", "train")],
        [_record("s1", "dev")],
        [_record("m1", "dev", pair="m"), _record("m2", "dev", pair="m")],
        [_record("v1", "val"), _record("v2", "val")],
        fingerprint="D2-FULL-A",
    )

    assert result.status == "PHASE_19_D2_FULL_GO"
    assert result.selected_checkpoint_step == 1
    assert backend.events.count("official:2") == 1
    assert backend.events.index("release:1") < backend.events.index("official:2")


def test_full_runner_keeps_official_sealed_when_multiquery_rank_is_below_absolute_gate(
    tmp_path: Path,
) -> None:
    semantic = _dev_metrics(rank=0.94)
    multiquery = _dev_metrics(rank=0.84)
    backend = _Backend(tmp_path, {step: (semantic, multiquery) for step in range(1, 5)})
    runner = _runner(tmp_path, backend)

    result = runner.run(
        [_record("f1", "train"), _record("f2", "train")],
        [_record("s1", "dev")],
        [_record("m1", "dev", pair="m"), _record("m2", "dev", pair="m")],
        [_record("v1", "val"), _record("v2", "val")],
        fingerprint="D2-FULL-B",
    )

    assert result.status == "PHASE_19_D2_FULL_NO_GO"
    assert result.official_decision is None
    assert not any(event.startswith("official:") for event in backend.events)


def test_full_runner_rejects_cross_split_pair_leakage_before_training(tmp_path: Path) -> None:
    passing = (_dev_metrics(rank=0.94), _dev_metrics(rank=0.86))
    backend = _Backend(tmp_path, {step: passing for step in range(1, 5)})
    runner = _runner(tmp_path, backend)

    try:
        runner.run(
            [_record("f1", "train", pair="shared"), _record("f2", "train")],
            [_record("s1", "dev", pair="shared")],
            [_record("m1", "dev", pair="m"), _record("m2", "dev", pair="m")],
            [_record("v1", "val"), _record("v2", "val")],
            fingerprint="D2-FULL-C",
        )
    except ValueError as error:
        assert "split leakage" in str(error)
    else:
        raise AssertionError("cross-split image-pair leakage must stop before training")
    assert backend.events == []


def test_official_gate_rejects_residual_drift_even_when_retrieval_passes(tmp_path: Path) -> None:
    passing = (_dev_metrics(rank=0.94), _dev_metrics(rank=0.86))
    backend = _Backend(tmp_path, {step: passing for step in range(1, 5)})
    runner = _runner(tmp_path, backend)
    metrics = _official_metrics()
    metrics["maximum_absolute_layer_drift"] = 0.021

    gate = runner.official_gate(metrics)

    assert gate["passed"] is False
    assert gate["checks"]["absolute_drift_bounded"] is False


def test_cloud_entrypoint_preserves_boundaries_before_opening_official() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "tools/cloud/run_rgbtir_phase19_d2_full_gated.sh"

    text = script.read_text(encoding="utf-8")

    assert "runtime_failure_state.json" in text
    assert "prepare_persistent_storage" in text
    assert "checkpoint_*.pt" in text
    assert "--minimum-checkpoints 4" in text
    assert "aic_rgbtir_phase19_d2_train_select_artifacts_20260816.zip" in text
    assert text.index("train-select") < text.index("official-val")
    assert "rm -rf" not in text
