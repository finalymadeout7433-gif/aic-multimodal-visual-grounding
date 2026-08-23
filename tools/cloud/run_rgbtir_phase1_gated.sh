#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/featurize/aic_rgbtir_phase1_bundle/repo}"
ENV_ROOT="${ENV_ROOT:-/home/featurize/work/envs/aic_rgbtir_phase1}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/aic_rgbtir_phase1.cloud.local.yaml}"
RUN_FULL="${RUN_FULL:-0}"
LOG_ROOT="${LOG_ROOT:-/home/featurize/aic_cloud/logs/aic_rgbtir_phase1_v1}"

mkdir -p "${LOG_ROOT}"
source "${ENV_ROOT}/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/featurize/work/hf_cache}"

python tools/run_rgbtir_phase1.py preflight --config "${CONFIG}" --allow-network \
  2>&1 | tee "${LOG_ROOT}/01_preflight.log"
python tools/run_rgbtir_phase1.py download-assets --config "${CONFIG}" \
  2>&1 | tee "${LOG_ROOT}/02_download_assets.log"
python tools/run_rgbtir_phase1.py smoke --config "${CONFIG}" \
  2>&1 | tee "${LOG_ROOT}/03_real_weight_smoke.log"
python tools/run_rgbtir_phase1.py warmup --config "${CONFIG}" --mode overfit100 --resume \
  2>&1 | tee "${LOG_ROOT}/04_overfit100.log"
python tools/run_rgbtir_phase1.py warmup --config "${CONFIG}" --mode tracer400 --resume \
  2>&1 | tee "${LOG_ROOT}/05_tracer400.log"

if [[ "${RUN_FULL}" == "1" ]]; then
  python tools/run_rgbtir_phase1.py warmup --config "${CONFIG}" --mode full --resume \
    2>&1 | tee "${LOG_ROOT}/06_full_warmup.log"
else
  echo "TRACER_COMPLETE_FULL_NOT_STARTED: set RUN_FULL=1 only after reviewing tracer400/run_summary.json"
fi
