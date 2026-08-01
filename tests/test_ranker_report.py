from __future__ import annotations

from pathlib import Path

from tools.build_ranker_training_report import (
    _compact_submissions,
    _local_promotion_checks,
    _public_artifact_path,
    _verification_pass,
)


def test_public_artifact_path_removes_workspace_prefix(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / "gdino_spatial_ltr_v1"
    artifact = output_root / "submissions" / "S02" / "submission.zip"

    assert _public_artifact_path(str(artifact), output_root) == (
        "outputs/gdino_spatial_ltr_v1/submissions/S02/submission.zip"
    )


def test_compact_submissions_contains_only_public_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / "gdino_spatial_ltr_v1"
    control = output_root / "submissions" / "S02"
    ranker = output_root / "submissions" / "S03"
    source = {
        "selection_summary": {"records": 9555},
        "submission_audit": {
            "control": {
                "json_path": str(control / "predictions.json"),
                "zip_path": str(control / "predictions.zip"),
                "zip_sha256": "control-hash",
            },
            "ranker": {
                "json_path": str(ranker / "predictions.json"),
                "zip_path": str(ranker / "predictions.zip"),
                "zip_sha256": "ranker-hash",
            },
        },
        "selection_debug_path": str(
            output_root / "submissions" / "selection_debug.jsonl"
        ),
    }

    compact = _compact_submissions(source, output_root)

    assert compact is not None
    serialized_paths = [
        compact["submission_audit"]["control"]["json_path"],
        compact["submission_audit"]["control"]["zip_path"],
        compact["submission_audit"]["ranker"]["json_path"],
        compact["submission_audit"]["ranker"]["zip_path"],
        compact["selection_debug_path"],
    ]
    assert all(
        path.startswith("outputs/gdino_spatial_ltr_v1/")
        for path in serialized_paths
    )
    assert all(str(tmp_path) not in path for path in serialized_paths)


def test_local_promotion_requires_every_planned_gate() -> None:
    holdout = {
        "overall": {
            "baseline_acc_at_05": 0.50,
            "selected_acc_at_05": 0.53,
        },
        "spatial_ordinal_combined": {"gain": 0.05},
        "grouped": {
            "dataset": {
                "a": {
                    "baseline_acc_at_05": 0.50,
                    "selected_acc_at_05": 0.52,
                },
                "b": {
                    "baseline_acc_at_05": 0.60,
                    "selected_acc_at_05": 0.595,
                },
            }
        },
    }
    submissions = {"selection_summary": {"legal_bbox_rate": 1.0}}

    passing = _local_promotion_checks(holdout, submissions)
    assert passing["pass"] is True

    submissions["selection_summary"]["legal_bbox_rate"] = 0.999
    failing = _local_promotion_checks(holdout, submissions)
    assert failing["pass"] is False
    assert failing["checks"]["legal_bbox_pass"] is False


def test_verification_pass_requires_all_release_gates() -> None:
    verification = {
        "pytest": {"status": "passed"},
        "compileall": {"status": "passed"},
        "submission_audit": {"status": "passed"},
        "code_review": {"status": "passed"},
    }

    assert _verification_pass(verification) is True
    verification["code_review"]["status"] = "pending"
    assert _verification_pass(verification) is False
    assert _verification_pass(None) is False
