#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/featurize/aic_rgbtir_phase16_recovery_bundle/repo}"
ENV_ROOT="${ENV_ROOT:-/home/featurize/work/envs/aic_rgbtir_phase16}"
if [[ ! -d "${ENV_ROOT}" && -d /home/featurize/work/envs/aic_rgbtir_phase1 ]]; then
  ENV_ROOT=/home/featurize/work/envs/aic_rgbtir_phase1
fi
PHASE16_CONFIG="${PHASE16_CONFIG:-${REPO_ROOT}/configs/aic_rgbtir_phase16_recovery.cloud.local.yaml}"
PHASE17_CONFIG="${PHASE17_CONFIG:-${REPO_ROOT}/configs/aic_rgbtir_phase17a_plus_recovery.cloud.local.yaml}"
LOG_ROOT="${LOG_ROOT:-/home/featurize/aic_cloud/logs/aic_rgbtir_phase16_recovery_v1}"
READY_ROOT="${READY_ROOT:-/home/featurize/aic_cloud/platform_upload_ready}"
OUTPUT16=/home/featurize/aic_cloud/outputs/aic_rgbtir_phase16_recovery_v1
OUTPUT17=/home/featurize/aic_cloud/outputs/aic_rgbtir_phase17a_plus_recovery_v1
REPORT16=/home/featurize/aic_cloud/reports/aic_rgbtir_phase16_recovery_2026_08_13.md
REPORT17=/home/featurize/aic_cloud/reports/aic_rgbtir_phase17a_plus_recovery_2026_08_13.md

mkdir -p "${LOG_ROOT}" "${READY_ROOT}"
source "${ENV_ROOT}/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/featurize/work/hf_cache}"

python tools/run_rgbtir_phase16.py build-manifests --config "${PHASE16_CONFIG}" \
  2>&1 | tee "${LOG_ROOT}/01_build_manifests.log"
python tools/run_rgbtir_phase16.py preflight --config "${PHASE16_CONFIG}" \
  2>&1 | tee "${LOG_ROOT}/02_preflight.log"
python tools/run_rgbtir_phase16.py train --config "${PHASE16_CONFIG}" --allow-network --resume \
  2>&1 | tee "${LOG_ROOT}/03_c1_c2_recovery.log"

test -s "${OUTPUT16}/probe_candidates/C1/adapter.pt"
test -s "${OUTPUT16}/probe_candidates/C2/adapter.pt"
test -s "${OUTPUT16}/rgb_teacher_bank/teacher_bank.pt"
test -s "${OUTPUT16}/recovery_diagnostics/C2.json"

python tools/run_rgbtir_phase17a_plus.py --config "${PHASE17_CONFIG}" --resume \
  2>&1 | tee "${LOG_ROOT}/04_phase17a_plus.log"

ARCHIVE="${READY_ROOT}/AIC_RGBT_Phase16R_C1_C2_Phase17APlus_20260813.tar.gz"
rm -f "${ARCHIVE}" "${ARCHIVE}.sha256"
tar -czf "${ARCHIVE}" \
  -C /home/featurize/aic_cloud \
  outputs/aic_rgbtir_phase16_recovery_v1 \
  outputs/aic_rgbtir_phase17a_plus_recovery_v1 \
  logs/aic_rgbtir_phase16_recovery_v1 \
  reports/"$(basename "${REPORT17}")"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"

python - "${OUTPUT17}/decision.json" "${ARCHIVE}" <<'PY'
import json, pathlib, sys
decision = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
allowed = {
    "C2_FULL_TRAIN", "D1_RETENTION", "D1_QUALITY_WEIGHTED",
    "C2_FALSE_NEGATIVE_AWARE", "SHARED_COMPLEMENTARY_PROBE",
}
if decision.get("branch") not in allowed:
    raise SystemExit(f"RECOVERY_DECISION_INVALID: {decision.get('branch')}")
archive = pathlib.Path(sys.argv[2])
if not archive.is_file() or archive.stat().st_size == 0:
    raise SystemExit("RECOVERY_ARCHIVE_MISSING")
print(json.dumps({"status": "RECOVERY_READY_FOR_DOWNLOAD", "decision": decision["branch"], "archive": str(archive)}))
PY

echo "PHASE16_RECOVERY_AND_PHASE17A_PLUS_FINISHED"
