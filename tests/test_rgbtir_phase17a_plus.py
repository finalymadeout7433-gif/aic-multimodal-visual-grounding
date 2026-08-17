from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aic_rgbtir.data import RGBTRecord
from aic_rgbtir.phase16 import RGBTeacherBank, TeacherEntry
from aic_rgbtir.phase17a_plus import (
    Phase17APlusAuditor,
    Phase17DecisionInputs,
    Phase17DecisionPolicy,
)
from tools.run_rgbtir_phase17a_plus import _report


def _record(index: int, *, query: str, pair: str | None = None) -> RGBTRecord:
    name = pair or f"pair_{index:04d}.png"
    return RGBTRecord(
        record_id=f"flir:train:{index}",
        source_dataset="flir",
        split="train",
        rgb_relpath=f"image_data/flir/rgb/{name}",
        tir_relpath=f"image_data/flir/ir/{name}",
        query_original=query,
        bbox_xywh_pixel=(4.0, 4.0, 8.0, 8.0),
        bbox_xyxy_normalized=(0.25, 0.25, 0.75, 0.75),
        width=16,
        height=16,
        illumination="WL",
        weather="FY",
        object_size="SS",
        occlusion="PO",
        crowded="NC",
        scene="UB",
    )


def test_decision_policy_routes_one_branch_from_registered_evidence() -> None:
    policy = Phase17DecisionPolicy()

    assert policy.decide(Phase17DecisionInputs(assets_ready=False)).branch == "ASSET_BLOCKED"
    assert policy.decide(
        Phase17DecisionInputs(
            assets_ready=True,
            real_alignment_drift=False,
            false_negative_rate=0.05,
            pareto_conflict=False,
        )
    ).branch == "C2_FULL_TRAIN"
    assert policy.decide(
        Phase17DecisionInputs(
            assets_ready=True,
            real_alignment_drift=True,
            rgb_quality_association=0.48,
            false_negative_rate=0.05,
            pareto_conflict=False,
        )
    ).branch == "D1_QUALITY_WEIGHTED"
    assert policy.decide(
        Phase17DecisionInputs(
            assets_ready=True,
            real_alignment_drift=True,
            rgb_quality_association=0.05,
            false_negative_rate=0.05,
            pareto_conflict=False,
        )
    ).branch == "D1_RETENTION"
    assert policy.decide(
        Phase17DecisionInputs(
            assets_ready=True,
            real_alignment_drift=False,
            false_negative_rate=0.30,
            pareto_conflict=False,
        )
    ).branch == "C2_FALSE_NEGATIVE_AWARE"
    assert policy.decide(
        Phase17DecisionInputs(
            assets_ready=True,
            real_alignment_drift=True,
            rgb_quality_association=0.10,
            false_negative_rate=0.05,
            pareto_conflict=True,
        )
    ).branch == "SHARED_COMPLEMENTARY_PROBE"


def test_auditor_emits_partial_evidence_and_asset_blocked_decision(tmp_path: Path) -> None:
    root = tmp_path / "rgbt"
    rgb_dir = root / "image_data" / "flir" / "rgb"
    tir_dir = root / "image_data" / "flir" / "ir"
    rgb_dir.mkdir(parents=True)
    tir_dir.mkdir(parents=True)
    records = [
        _record(0, query="the small person beside the car"),
        _record(1, query="the man near the vehicle"),
        _record(2, query="the red traffic light"),
    ]
    for index, record in enumerate(records):
        rgb = np.full((16, 16, 3), 30 + index * 50, dtype=np.uint8)
        rgb[4:12, 4:12] = 120 + index * 20
        tir = np.full((16, 16, 3), 80 + index * 20, dtype=np.uint8)
        tir[:, :2] = 0
        Image.fromarray(rgb).save(root / record.rgb_relpath)
        Image.fromarray(tir).save(root / record.tir_relpath)

    output = tmp_path / "outputs"
    required_assets = {
        "c1_adapter": tmp_path / "missing" / "C1" / "adapter.pt",
        "c2_adapter": tmp_path / "missing" / "C2" / "adapter.pt",
        "teacher_bank": tmp_path / "missing" / "teacher_bank.pt",
        "c1_summary": tmp_path / "missing" / "C1" / "summary.json",
        "c2_summary": tmp_path / "missing" / "C2" / "summary.json",
    }
    result = Phase17APlusAuditor(output_root=output, seed=20260812).run(
        records=records,
        rgbt_root=root,
        required_assets=required_assets,
    )

    assert result.decision.branch == "ASSET_BLOCKED"
    assert result.completed_records == 3
    assert set(result.missing_assets) == set(required_assets)
    assert (output / "asset_preflight.json").is_file()
    assert (output / "rgb_teacher_quality.csv").is_file()
    assert (output / "negative_sampling_audit.json").is_file()
    assert (output / "decision.json").is_file()
    assert (output / "sha256_manifest.json").is_file()
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    assert decision["branch"] == "ASSET_BLOCKED"
    assert decision["claim_boundary"]["c2_drift_confirmed"] is False
    rows = (output / "per_record_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3


def test_negative_audit_is_deterministic_and_never_uses_same_pair(tmp_path: Path) -> None:
    records = [
        _record(index, query=("the person" if index < 10 else "the vehicle"))
        for index in range(40)
    ]
    auditor = Phase17APlusAuditor(output_root=tmp_path, seed=20260812, negative_count=16)
    first = auditor.audit_negatives(records)
    second = auditor.audit_negatives(records)

    assert first == second
    assert first["sampled_anchor_count"] == 40
    assert first["same_pair_violation_count"] == 0
    assert first["same_target_head_rate"] > 0
    assert first["potential_false_negative_rate"] >= first["same_target_head_rate"]


def test_auditor_consumes_recovered_c2_metrics_and_routes_quality_weighting(tmp_path: Path) -> None:
    root = tmp_path / "rgbt"
    rgb_dir = root / "image_data" / "flir" / "rgb"
    tir_dir = root / "image_data" / "flir" / "ir"
    rgb_dir.mkdir(parents=True)
    tir_dir.mkdir(parents=True)
    records = [_record(index, query=f"the target object {index}") for index in range(6)]
    per_record = []
    for index, record in enumerate(records):
        level = 20 + index * 35
        Image.fromarray(np.full((16, 16, 3), level, dtype=np.uint8)).save(root / record.rgb_relpath)
        Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8)).save(root / record.tir_relpath)
        drift = 0.12 - index * 0.018
        per_record.append({
            "record_id": record.record_id,
            "layers": {
                layer: {
                    "absolute_alignment_delta": drift,
                    "top_neighbor_record_ids": [records[(index + 1) % len(records)].record_id],
                }
                for layer in ("8", "16", "24", "final")
            },
        })
    model_metrics = {
        "record_count": len(records),
        "layers": {
            layer: {"absolute_alignment_delta_mean": 0.04}
            for layer in ("8", "16", "24", "final")
        },
        "per_record": per_record,
    }
    metrics_path = tmp_path / "C2.json"
    metrics_path.write_text(json.dumps(model_metrics), encoding="utf-8")
    required_assets = {}

    result = Phase17APlusAuditor(output_root=tmp_path / "out").run(
        records=records,
        rgbt_root=root,
        required_assets=required_assets,
        model_diagnostics_path=metrics_path,
    )

    assert result.decision.branch == "D1_QUALITY_WEIGHTED"
    decision = json.loads((tmp_path / "out" / "decision.json").read_text(encoding="utf-8"))
    assert decision["claim_boundary"]["c2_drift_confirmed"] is True
    assert decision["evidence"]["rgb_quality_association"] >= 0.30


def test_recovered_diagnostics_are_joined_into_records_subgroups_and_teacher_audit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rgbt"
    rgb_dir = root / "image_data" / "flir" / "rgb"
    tir_dir = root / "image_data" / "flir" / "ir"
    rgb_dir.mkdir(parents=True)
    tir_dir.mkdir(parents=True)
    records = [_record(index, query=f"the person {index}") for index in range(6)]
    per_record = []
    entries = {}
    for index, record in enumerate(records):
        Image.fromarray(np.full((16, 16, 3), 40 + index * 20, dtype=np.uint8)).save(
            root / record.rgb_relpath
        )
        Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8)).save(
            root / record.tir_relpath
        )
        per_record.append(
            {
                "record_id": record.record_id,
                "layers": {
                    layer: {
                        "absolute_alignment_delta": 0.01 * (index + 1),
                        "base_paired_cosine": 0.8,
                        "adapted_paired_cosine": 0.79 - 0.01 * index,
                        "top_neighbor_record_ids": [
                            records[(index + 1) % len(records)].record_id
                        ],
                    }
                    for layer in ("8", "16", "24", "final")
                },
            }
        )
        foreground = {
            layer: torch.tensor([1.0, 0.0, float(index) / 10.0])
            for layer in ("8", "16", "24", "final")
        }
        background = {
            layer: torch.tensor([0.0, 1.0, float(index) / 10.0])
            for layer in ("8", "16", "24", "final")
        }
        entries[record.record_id] = TeacherEntry(
            record_id=record.record_id,
            image_pair_key=record.image_pair_key,
            source_dataset=record.source_dataset,
            illumination=record.illumination,
            object_size=record.object_size,
            foreground=foreground,
            background=background,
            base_tir=foreground,
        )

    diagnostics = {
        "record_count": len(records),
        "layers": {
            layer: {"absolute_alignment_delta_mean": 0.04}
            for layer in ("8", "16", "24", "final")
        },
        "per_record": per_record,
    }
    diagnostics_path = tmp_path / "C2.json"
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    bank_path = tmp_path / "teacher_bank.pt"
    RGBTeacherBank(fingerprint="TEST-BANK", entries=entries).save(bank_path)
    c2_summary = tmp_path / "c2_summary.json"
    c2_summary.write_text(
        json.dumps({"teacher_bank_fingerprint": "TEST-BANK"}), encoding="utf-8"
    )

    output = tmp_path / "out"
    Phase17APlusAuditor(output_root=output, seed=20260812).run(
        records=records,
        rgbt_root=root,
        required_assets={"teacher_bank": bank_path, "c2_summary": c2_summary},
        model_diagnostics_path=diagnostics_path,
    )

    rows = [
        json.loads(line)
        for line in (output / "per_record_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["c2_metrics_available"] is True for row in rows)
    assert all(row["c2_alignment_delta"] is not None for row in rows)
    assert all(row["rgb_target_background_teacher_margin"] is not None for row in rows)
    subgroup_text = (output / "subgroup_metrics.csv").read_text(encoding="utf-8-sig")
    assert ",True," in subgroup_text
    reliability = json.loads((output / "teacher_reliability.json").read_text(encoding="utf-8"))
    assert reliability["status"] == "MODEL_EVIDENCE_COMPLETE"
    assert "rgb_target_background_teacher_margin" in reliability["spearman"]


def test_complete_report_does_not_emit_asset_recovery_instructions(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "decision.json").write_text(
        json.dumps(
            {
                "evidence": {
                    "layer_absolute_alignment_delta_mean": {
                        "8": 0.12,
                        "16": 0.15,
                        "24": 0.38,
                    },
                    "rgb_quality_association": 0.15,
                    "model_top_neighbor_potential_false_negative_rate": 0.19,
                }
            }
        ),
        encoding="utf-8",
    )
    result = type(
        "Result",
        (),
        {
            "status": "PHASE_17A_PLUS_COMPLETE",
            "decision": type(
                "Decision",
                (),
                {
                    "branch": "D1_RETENTION",
                    "reason": "Material drift is present.",
                },
            )(),
            "completed_records": 1024,
        },
    )()
    report = tmp_path / "report.md"
    _report(
        report,
        result=result,
        output_root=output,
        negative={"potential_false_negative_rate": 0.08},
        asset_preflight={"status": "ASSETS_READY", "missing": []},
    )

    text = report.read_text(encoding="utf-8")
    assert "## 4. 已完成的模型证据" in text
    assert "尚未完成的核心问题" not in text
    assert "回收 `probe_candidates/C1`" not in text
    assert "D1_RETENTION" in text
