from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Sequence


def create_stage_archive(
    *,
    output_root: Path | str,
    archive_path: Path | str,
    stage: str,
    required: Sequence[str] = (),
    minimum_checkpoints: int = 0,
) -> dict[str, Any]:
    """Atomically archive a completed stage and emit a hash-bound receipt."""
    root = Path(output_root).resolve()
    archive = Path(archive_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"output root is missing: {root}")
    for relative in required:
        candidate = root / relative
        if not candidate.is_file():
            raise FileNotFoundError(f"required stage asset is missing: {relative}")
    checkpoints = tuple(sorted((root / "checkpoints").glob("checkpoint_*.pt")))
    if len(checkpoints) < int(minimum_checkpoints):
        raise FileNotFoundError(
            f"required checkpoint count is missing: {len(checkpoints)} < {minimum_checkpoints}"
        )

    files = tuple(path for path in sorted(root.rglob("*")) if path.is_file())
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(path, path.relative_to(root).as_posix())
    temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest().upper()
    receipt = {
        "status": "PHASE_18_STAGE_ARCHIVE_READY",
        "stage": str(stage),
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "file_count": len(files),
        "sha256": digest,
    }
    archive.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
