from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: Path | str) -> str:
    artifact = Path(path)
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_file_sha256(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Always hash ``path`` and optionally compare it with an assertion."""

    expected = None if expected_sha256 is None else expected_sha256.strip().upper()
    if expected is not None and (
        len(expected) != 64
        or any(character not in "0123456789ABCDEF" for character in expected)
    ):
        raise ValueError("expected SHA-256 must be 64 hexadecimal characters")
    actual = sha256_file(path)
    if expected is None:
        return actual
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {Path(path)}: expected={expected}, actual={actual}"
        )
    return actual


def hash_named_files(
    root: Path | str,
    relative_paths: Iterable[str],
) -> dict[str, str]:
    base = Path(root).resolve()
    result: dict[str, str] = {}
    for raw_relative in relative_paths:
        relative = Path(raw_relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"runtime file must be relative to repository root: {raw_relative}")
        artifact = (base / relative).resolve()
        if not artifact.is_relative_to(base):
            raise ValueError(f"runtime file escapes repository root: {raw_relative}")
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        key = relative.as_posix()
        result[key] = sha256_file(artifact)
    return dict(sorted(result.items()))


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments],
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot fingerprint git repository {root}") from error


def git_repository_state(root: Path | str) -> dict[str, Any]:
    """Fingerprint the exact tracked worktree, not only its base commit."""

    repository = Path(root).resolve()
    head = _git_bytes(repository, "rev-parse", "HEAD").decode("ascii").strip()
    tracked_diff = _git_bytes(
        repository,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
    )
    status_text = _git_bytes(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    ).decode("utf-8", errors="replace")
    submodule_text = _git_bytes(
        repository,
        "submodule",
        "status",
        "--recursive",
    ).decode("utf-8", errors="replace")
    return {
        "head": head,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest().upper(),
        "tracked_diff_size_bytes": len(tracked_diff),
        "tracked_status": [line for line in status_text.splitlines() if line],
        "submodule_status": [line for line in submodule_text.splitlines() if line],
    }
