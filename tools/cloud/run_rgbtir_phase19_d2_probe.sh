#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-/home/featurize/aic_rgbtir_phase19_d2_bundle}"
ENV_ROOT="${ENV_ROOT:-/home/featurize/envs/aic-rgbtir-phase19-d2}"
CONFIG="$BUNDLE_ROOT/configs/aic_rgbtir_phase19_d2.cloud.example.yaml"
OUTPUT_ROOT="/home/featurize/aic_cloud/outputs/aic_rgbtir_phase19_d2_probe_v1"
LOG_ROOT="/home/featurize/aic_cloud/logs"
ARCHIVE_ROOT="/home/featurize/aic_cloud/platform_upload_ready"
ARCHIVE="$ARCHIVE_ROOT/aic_rgbtir_phase19_d2_probe_artifacts_20260816.zip"

source "$ENV_ROOT/bin/activate"
cd "$BUNDLE_ROOT"
mkdir -p "$LOG_ROOT" "$ARCHIVE_ROOT"
export PYTHONPATH="$BUNDLE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q tests/test_rgbtir_phase19_d2.py tests/test_rgbtir_phase16.py tests/test_rgbtir_phase18.py
python tools/run_rgbtir_phase19_d2.py preflight --config "$CONFIG"
set +e
python tools/run_rgbtir_phase19_d2.py probe --config "$CONFIG" --allow-network \
  2>&1 | tee "$LOG_ROOT/rgbtir_phase19_d2_probe.log"
STATUS=${PIPESTATUS[0]}
set -e
if [[ "$STATUS" -ne 0 && "$STATUS" -ne 2 ]]; then
  exit "$STATUS"
fi
python - "$OUTPUT_ROOT" "$LOG_ROOT/rgbtir_phase19_d2_probe.log" "$ARCHIVE" <<'PY'
from __future__ import annotations
import hashlib, json, sys, zipfile
from pathlib import Path

output = Path(sys.argv[1])
log = Path(sys.argv[2])
archive = Path(sys.argv[3])
summary = output / "run_summary.json"
if not summary.is_file():
    raise SystemExit("PHASE_19_D2_MISSING_RUN_SUMMARY")
payload = json.loads(summary.read_text(encoding="utf-8"))
if payload.get("status") not in {"PHASE_19_D2_PROBE_GO", "PHASE_19_D2_PROBE_NO_GO"}:
    raise SystemExit("PHASE_19_D2_INVALID_TERMINAL_STATUS")
files = [path for path in output.rglob("*") if path.is_file()]
files.append(log)
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
    for path in sorted(files):
        if path.is_relative_to(output):
            name = "outputs/aic_rgbtir_phase19_d2_probe_v1/" + path.relative_to(output).as_posix()
        else:
            name = "logs/" + path.name
        handle.write(path, name)
digest = hashlib.sha256(archive.read_bytes()).hexdigest().upper()
receipt = archive.with_suffix(".receipt.json")
receipt.write_text(json.dumps({
    "status": payload["status"],
    "selected_candidate": payload.get("selected_candidate", ""),
    "archive": str(archive),
    "bytes": archive.stat().st_size,
    "sha256": digest,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(receipt.read_text(encoding="utf-8"))
PY
exit "$STATUS"
