#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/featurize/aic_rgbtir_phase16_bundle/repo}"
ENV_ROOT="${ENV_ROOT:-/home/featurize/work/envs/aic_rgbtir_phase16}"
if [[ ! -d "${ENV_ROOT}" && -d /home/featurize/work/envs/aic_rgbtir_phase1 ]]; then
  ENV_ROOT=/home/featurize/work/envs/aic_rgbtir_phase1
fi
CONFIG="${CONFIG:-${REPO_ROOT}/configs/aic_rgbtir_phase16.cloud.local.yaml}"
LOG_ROOT="${LOG_ROOT:-/home/featurize/aic_cloud/logs/aic_rgbtir_phase16_v1}"

mkdir -p "${LOG_ROOT}"
source "${ENV_ROOT}/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/featurize/work/hf_cache}"

python tools/run_rgbtir_phase16.py build-manifests --config "${CONFIG}" \
  2>&1 | tee "${LOG_ROOT}/01_build_manifests.log"
python tools/run_rgbtir_phase16.py preflight --config "${CONFIG}" \
  2>&1 | tee "${LOG_ROOT}/02_preflight.log"
python tools/run_rgbtir_phase16.py all --config "${CONFIG}" --allow-network --resume \
  2>&1 | tee "${LOG_ROOT}/03_phase16_all.log"

echo "PHASE16_PIPELINE_FINISHED"
