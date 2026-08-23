#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-/home/featurize/aic_rgbtir_phase19_d2_layer_audit_bundle}"
ENV_ROOT="${ENV_ROOT:-/home/featurize/envs/aic-rgbtir-phase19-d2}"
CONFIG_PATH="${CONFIG_PATH:-${BUNDLE_ROOT}/configs/aic_rgbtir_phase19_d2_layer_audit.cloud.example.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/featurize/aic_cloud/outputs/aic_rgbtir_phase19_d2_layer_audit_v1}"
LOG_ROOT="${LOG_ROOT:-/home/featurize/aic_cloud/logs}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/home/featurize/aic_cloud/platform_upload_ready}"
STAMP="${STAMP:-20260817}"
LOG_PATH="${LOG_ROOT}/rgbtir_phase19_d2_layer_audit.log"
PID_PATH="${LOG_ROOT}/rgbtir_phase19_d2_layer_audit_master.pid"
FAILURE_PATH="${OUTPUT_ROOT}/runtime_failure_state.json"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${ARCHIVE_ROOT}"
echo "$$" > "${PID_PATH}"

on_failure() {
  local exit_code=$?
  python - "${FAILURE_PATH}" "${exit_code}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "status": "PHASE_19_D2_LAYER_AUDIT_RUNTIME_FAILURE",
    "exit_code": int(sys.argv[2]),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  exit "${exit_code}"
}
trap on_failure ERR
trap 'rm -f "${PID_PATH}"' EXIT

export PYTHONPATH="${BUNDLE_ROOT}/src:${BUNDLE_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"${ENV_ROOT}/bin/python" "${BUNDLE_ROOT}/tools/run_rgbtir_phase19_d2_layer_audit.py" \
  --config "${CONFIG_PATH}" \
  > "${OUTPUT_ROOT}/preflight_stdout.json"

"${ENV_ROOT}/bin/python" "${BUNDLE_ROOT}/tools/run_rgbtir_phase19_d2_layer_audit.py" \
  --config "${CONFIG_PATH}" \
  --execute \
  --resume \
  2>&1 | tee "${LOG_PATH}"

ARCHIVE_PATH="${ARCHIVE_ROOT}/aic_rgbtir_phase19_d2_layer_audit_artifacts_${STAMP}.zip"
RECEIPT_PATH="${ARCHIVE_PATH%.zip}.receipt.json"
rm -f "${ARCHIVE_PATH}" "${RECEIPT_PATH}"
python - "${OUTPUT_ROOT}" "${LOG_PATH}" "${ARCHIVE_PATH}" "${RECEIPT_PATH}" <<'PY'
import hashlib
import json
import pathlib
import sys
import zipfile

output = pathlib.Path(sys.argv[1])
log = pathlib.Path(sys.argv[2])
archive = pathlib.Path(sys.argv[3])
receipt = pathlib.Path(sys.argv[4])
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(output.rglob("*")):
        if path.is_file():
            zf.write(path, pathlib.PurePosixPath("outputs") / path.relative_to(output))
    if log.is_file():
        zf.write(log, pathlib.PurePosixPath("logs") / log.name)
digest = hashlib.sha256(archive.read_bytes()).hexdigest().upper()
receipt.write_text(json.dumps({
    "status": "PHASE_19_D2_LAYER_AUDIT_ARCHIVE_READY",
    "archive": str(archive),
    "bytes": archive.stat().st_size,
    "sha256": digest,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

trap - ERR
rm -f "${FAILURE_PATH}"
