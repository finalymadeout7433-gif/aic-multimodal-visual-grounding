from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"required completed-run artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_expected_subset(
    *,
    actual: Any,
    expected: Any,
    field: str = "run_fingerprint",
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError(f"fingerprint mismatch at {field}")
        for key, expected_value in expected.items():
            if key not in actual:
                raise ValueError(f"fingerprint mismatch at {field}.{key}: missing")
            _assert_expected_subset(
                actual=actual[key],
                expected=expected_value,
                field=f"{field}.{key}",
            )
        return
    if actual != expected:
        raise ValueError(
            f"fingerprint mismatch at {field}: expected={expected!r}, actual={actual!r}"
        )


def verify_completed_run(
    *,
    run_directory: Path | str,
    expected_query_count: int,
    expected_fingerprint: Mapping[str, Any],
    expected_fingerprint_sha256: str | None = None,
    expected_post_run_provenance: Mapping[str, Any] | None = None,
    expected_post_run_provenance_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify that a reusable run matches the requested model and package bytes."""

    root = Path(run_directory)
    summary = _read_json(root / "run_summary.json")
    if not summary.get("complete"):
        raise ValueError("completed-run summary is not complete")
    if summary.get("completed_records") != expected_query_count:
        raise ValueError("completed-run query count mismatch")

    fingerprint_path = root / "run_fingerprint.json"
    fingerprint = _read_json(fingerprint_path)
    if (
        expected_fingerprint_sha256 is not None
        and _sha256_file(fingerprint_path) != expected_fingerprint_sha256.upper()
    ):
        raise ValueError("run fingerprint file SHA-256 mismatch")
    _assert_expected_subset(actual=fingerprint, expected=expected_fingerprint)
    if expected_post_run_provenance is not None:
        provenance_path = root / "post_run_provenance_audit.json"
        provenance = _read_json(provenance_path)
        if (
            expected_post_run_provenance_sha256 is not None
            and _sha256_file(provenance_path)
            != expected_post_run_provenance_sha256.upper()
        ):
            raise ValueError("post_run_provenance file SHA-256 mismatch")
        _assert_expected_subset(
            actual=provenance,
            expected=expected_post_run_provenance,
            field="post_run_provenance",
        )
        recorded_fingerprint_sha256 = provenance.get(
            "original_run_fingerprint_sha256"
        )
        if recorded_fingerprint_sha256 != _sha256_file(fingerprint_path):
            raise ValueError(
                "post_run_provenance original fingerprint SHA-256 mismatch"
            )

    submission = root / "submission"
    json_path = submission / "predictions_submission.json"
    zip_path = submission / "predictions_submission.zip"
    audit = _read_json(submission / "submission_audit.json")
    for path in (json_path, zip_path):
        if not path.is_file():
            raise ValueError(f"required completed-run artifact is missing: {path}")

    expected_audit = {
        "query_count": expected_query_count,
        "query_ids_exact": True,
        "modified_non_bbox_count": 0,
        "invalid_bbox_count": 0,
        "zip_entries": ["predictions_submission.json"],
        "zip_contains_only_predictions_submission_json": True,
        "predictions_json_sha256": _sha256_file(json_path),
        "predictions_zip_sha256": _sha256_file(zip_path),
    }
    for key, expected_value in expected_audit.items():
        if audit.get(key) != expected_value:
            raise ValueError(
                f"submission audit mismatch at {key}: "
                f"expected={expected_value!r}, actual={audit.get(key)!r}"
            )

    json_bytes = json_path.read_bytes()
    with zipfile.ZipFile(zip_path) as archive:
        entries = archive.namelist()
        if entries != ["predictions_submission.json"]:
            raise ValueError(f"invalid ZIP entries: {entries}")
        if archive.read(entries[0]) != json_bytes:
            raise ValueError("ZIP payload differs from predictions_submission.json")
    payload = json.loads(json_bytes.decode("utf-8"))
    if len(payload) != expected_query_count:
        raise ValueError("submission JSON query count mismatch")

    return {
        "verified": True,
        "query_count": expected_query_count,
        "model_name": fingerprint.get("model_name"),
        "predictions_json_sha256": expected_audit["predictions_json_sha256"],
        "predictions_zip_sha256": expected_audit["predictions_zip_sha256"],
    }
