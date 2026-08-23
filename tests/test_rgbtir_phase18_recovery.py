from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from aic_rgbtir.artifacts import create_stage_archive


def test_stage_archive_is_atomic_hashed_and_contains_required_assets(tmp_path: Path) -> None:
    output = tmp_path / "output"
    (output / "checkpoints").mkdir(parents=True)
    (output / "selected_adapter").mkdir(parents=True)
    (output / "checkpoints" / "checkpoint_1.pt").write_bytes(b"checkpoint")
    (output / "selected_adapter" / "adapter.pt").write_bytes(b"adapter")
    (output / "train_selection_summary.json").write_text("{}\n", encoding="utf-8")
    archive = tmp_path / "ready" / "stage_a.zip"

    receipt = create_stage_archive(
        output_root=output,
        archive_path=archive,
        stage="train-select",
        required=(
            "train_selection_summary.json",
            "selected_adapter/adapter.pt",
            "checkpoints/checkpoint_1.pt",
        ),
    )

    assert archive.is_file()
    assert not archive.with_suffix(".zip.tmp").exists()
    assert receipt["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest().upper()
    assert json.loads(archive.with_suffix(".receipt.json").read_text(encoding="utf-8")) == receipt
    with zipfile.ZipFile(archive) as handle:
        assert set(handle.namelist()) == {
            "checkpoints/checkpoint_1.pt",
            "selected_adapter/adapter.pt",
            "train_selection_summary.json",
        }


def test_stage_archive_refuses_missing_required_asset(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileNotFoundError, match="selected_adapter/adapter.pt"):
        create_stage_archive(
            output_root=output,
            archive_path=tmp_path / "stage_a.zip",
            stage="train-select",
            required=("selected_adapter/adapter.pt",),
        )
