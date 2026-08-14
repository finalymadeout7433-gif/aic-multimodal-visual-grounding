from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aic_baseline.provenance import (
    git_repository_state,
    hash_named_files,
    verify_file_sha256,
)


def test_verify_file_sha256_recomputes_and_rejects_stale_assertion(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"original")
    actual = verify_file_sha256(artifact)

    assert verify_file_sha256(artifact, expected_sha256=actual.lower()) == actual
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file_sha256(artifact, expected_sha256=actual)


def test_hash_named_files_and_git_state_cover_actual_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    tracked = repo / "module.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "module.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Codex Test",
            "-c",
            "user.email=codex@example.invalid",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        check=True,
    )
    clean = git_repository_state(repo)
    tracked.write_text("value = 2\n", encoding="utf-8")
    dirty = git_repository_state(repo)

    hashes = hash_named_files(repo, ["module.py"])
    assert set(hashes) == {"module.py"}
    assert clean["head"] == dirty["head"]
    assert clean["tracked_diff_sha256"] != dirty["tracked_diff_sha256"]
    assert dirty["tracked_status"] == [" M module.py"]
