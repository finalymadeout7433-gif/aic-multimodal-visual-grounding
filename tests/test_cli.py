from __future__ import annotations

from pathlib import Path

from aic_baseline.cli import _build_run_fingerprint


def _make_config(tmp_path: Path) -> dict[str, str]:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model.safetensors").write_bytes(b"weights")
    (model_path / "config.json").write_text('{"version": 1}', encoding="utf-8")
    queries_path = tmp_path / "queries.json"
    queries_path.write_text("{}", encoding="utf-8")
    return {
        "model_path": str(model_path),
        "queries_path": str(queries_path),
    }


def test_run_fingerprint_changes_when_non_weight_model_artifact_changes(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    first = _build_run_fingerprint(config, fallback_mode="center")

    model_path = Path(config["model_path"])
    (model_path / "config.json").write_text('{"version": 2}', encoding="utf-8")
    second = _build_run_fingerprint(config, fallback_mode="center")

    assert first["model_artifacts_sha256"] != second["model_artifacts_sha256"]
