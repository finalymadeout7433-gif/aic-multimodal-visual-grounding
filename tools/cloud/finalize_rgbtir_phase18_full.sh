#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/featurize/aic_cloud/outputs/aic_rgbtir_phase18_full_v1}"
REPORT_ROOT="${REPORT_ROOT:-/home/featurize/aic_cloud/reports}"
LOG_ROOT="${LOG_ROOT:-/home/featurize/aic_cloud/logs}"
READY_ROOT="${READY_ROOT:-/home/featurize/aic_cloud/platform_upload_ready}"
ARCHIVE="$READY_ROOT/aic_rgbtir_phase18_full_artifacts_20260815.zip"

test -f "$OUTPUT_ROOT/run_summary.json"
test -f "$OUTPUT_ROOT/sha256_manifest.json"
mkdir -p "$READY_ROOT"

python - "$OUTPUT_ROOT" "$REPORT_ROOT" "$LOG_ROOT" "$ARCHIVE" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

output_root, report_root, log_root, archive = map(Path, sys.argv[1:])
temporary = archive.with_suffix(".zip.tmp")
temporary.unlink(missing_ok=True)
roots = (
    (output_root, "outputs/aic_rgbtir_phase18_full_v1"),
    (report_root, "reports"),
    (log_root, "logs"),
)
with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
    for root, prefix in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                handle.write(path, f"{prefix}/{path.relative_to(root).as_posix()}")
temporary.replace(archive)
digest = hashlib.sha256(archive.read_bytes()).hexdigest().upper()
receipt = {
    "status": "PHASE_18_FULL_ARCHIVE_READY",
    "path": str(archive),
    "bytes": archive.stat().st_size,
    "sha256": digest,
}
archive.with_suffix(".receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(receipt, indent=2))
PY
