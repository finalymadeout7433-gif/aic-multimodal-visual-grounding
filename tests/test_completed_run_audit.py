from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from aic_baseline.completed_run_audit import verify_completed_run
from aic_baseline.aic_full_detection import sha256_file


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_completed_run(root: Path) -> None:
    submission = root / "submission"
    submission.mkdir(parents=True)
    payload = {"q1": {"query": "target", "bbox": [0.1, 0.2, 0.5, 0.8]}}
    json_path = submission / "predictions_submission.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    zip_path = submission / "predictions_submission.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("predictions_submission.json", json_path.read_bytes())
    _write_json(
        root / "run_summary.json",
        {"complete": True, "completed_records": 1},
    )
    _write_json(
        root / "run_fingerprint.json",
        {
            "model_name": "expected/model",
            "queries_sha256": "A" * 64,
            "settings": {"top_k": 20},
        },
    )
    _write_json(
        submission / "submission_audit.json",
        {
            "query_count": 1,
            "query_ids_exact": True,
            "modified_non_bbox_count": 0,
            "invalid_bbox_count": 0,
            "zip_entries": ["predictions_submission.json"],
            "zip_contains_only_predictions_submission_json": True,
            "predictions_json_sha256": sha256_file(json_path),
            "predictions_zip_sha256": sha256_file(zip_path),
        },
    )


def test_verify_completed_run_checks_fingerprint_and_submission_bytes(
    tmp_path: Path,
) -> None:
    _make_completed_run(tmp_path)

    result = verify_completed_run(
        run_directory=tmp_path,
        expected_query_count=1,
        expected_fingerprint={
            "model_name": "expected/model",
            "queries_sha256": "A" * 64,
            "settings": {"top_k": 20},
        },
    )

    assert result["verified"] is True
    assert result["query_count"] == 1


def test_verify_completed_run_rejects_stale_fingerprint(tmp_path: Path) -> None:
    _make_completed_run(tmp_path)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_completed_run(
            run_directory=tmp_path,
            expected_query_count=1,
            expected_fingerprint={"model_name": "different/model"},
        )
    with pytest.raises(ValueError, match="fingerprint file SHA-256"):
        verify_completed_run(
            run_directory=tmp_path,
            expected_query_count=1,
            expected_fingerprint={"model_name": "expected/model"},
            expected_fingerprint_sha256="0" * 64,
        )


def test_verify_completed_run_rejects_zip_different_from_json(tmp_path: Path) -> None:
    _make_completed_run(tmp_path)
    zip_path = tmp_path / "submission" / "predictions_submission.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("predictions_submission.json", b"{}")
    audit_path = tmp_path / "submission" / "submission_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["predictions_zip_sha256"] = sha256_file(zip_path)
    _write_json(audit_path, audit)

    with pytest.raises(ValueError, match="ZIP payload differs"):
        verify_completed_run(
            run_directory=tmp_path,
            expected_query_count=1,
            expected_fingerprint={"model_name": "expected/model"},
        )


def test_verify_completed_run_can_bind_post_run_provenance(tmp_path: Path) -> None:
    _make_completed_run(tmp_path)
    fingerprint_path = tmp_path / "run_fingerprint.json"
    provenance = {
        "original_run_fingerprint_sha256": sha256_file(fingerprint_path),
        "ape_repository": {
            "head": "abc123",
            "runtime_files_sha256": {"runtime.py": "B" * 64},
        },
    }
    _write_json(tmp_path / "post_run_provenance_audit.json", provenance)

    verify_completed_run(
        run_directory=tmp_path,
        expected_query_count=1,
        expected_fingerprint={"model_name": "expected/model"},
        expected_post_run_provenance_sha256=sha256_file(
            tmp_path / "post_run_provenance_audit.json"
        ),
        expected_post_run_provenance=provenance,
    )
    provenance["ape_repository"]["head"] = "different"
    with pytest.raises(ValueError, match="post_run_provenance"):
        verify_completed_run(
            run_directory=tmp_path,
            expected_query_count=1,
            expected_fingerprint={"model_name": "expected/model"},
            expected_post_run_provenance=provenance,
        )
