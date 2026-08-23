#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-/home/featurize/aic_rgbtir_phase18_bundle}"
ENV_ROOT="${ENV_ROOT:-/home/featurize/work/envs/aic-rgbtir-phase18}"
CONFIG="$BUNDLE_ROOT/configs/aic_rgbtir_phase18.cloud.example.yaml"
LOG_ROOT="/home/featurize/aic_cloud/logs"

source "$ENV_ROOT/bin/activate"
cd "$BUNDLE_ROOT"
mkdir -p "$LOG_ROOT"
export PYTHONPATH="$BUNDLE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q tests/test_rgbtir_phase18.py tests/test_rgbtir_phase18_bundle_contract.py tests/test_rgbtir_phase16.py
python tools/run_rgbtir_phase18.py preflight --config "$CONFIG"
python tools/run_rgbtir_phase18.py probe --config "$CONFIG" --resume --allow-network 2>&1 | tee "$LOG_ROOT/rgbtir_phase18_d1_probe.log"
