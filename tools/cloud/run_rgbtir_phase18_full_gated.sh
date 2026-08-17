#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-/home/featurize/aic_rgbtir_phase18_full_bundle}"
ENV_ROOT="${ENV_ROOT:-/home/featurize/work/envs/aic-rgbtir-phase18}"
CONFIG="$BUNDLE_ROOT/configs/aic_rgbtir_phase18_full.cloud.example.yaml"
Q0_CONFIG="$BUNDLE_ROOT/configs/aic_rgbtir_phase18_q0.cloud.yaml"
Q0_REAL_CONFIG="$BUNDLE_ROOT/configs/aic_rgbtir_phase18_q0_real.cloud.yaml"
LOG_ROOT="/home/featurize/aic_cloud/logs"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/featurize/aic_cloud/outputs/aic_rgbtir_phase18_full_v1}"
READY_ROOT="${READY_ROOT:-/home/featurize/aic_cloud/platform_upload_ready}"

preserve_failure_state() {
  exit_status="$?"
  if [[ "$exit_status" -eq 0 ]]; then
    return
  fi
  set +e
  mkdir -p "$OUTPUT_ROOT"
  python - "$OUTPUT_ROOT" "$exit_status" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
checkpoints = sorted((root / "checkpoints").glob("checkpoint_*.pt"))
teacher_partial = root / "rgb_teacher_bank" / "teacher_bank.partial.pt"
teacher_final = root / "rgb_teacher_bank" / "teacher_bank.pt"
payload = {
    "status": "PHASE_18_RUNTIME_FAILURE_PRESERVED",
    "exit_status": int(sys.argv[2]),
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "checkpoint_count": len(checkpoints),
    "checkpoints": [str(path) for path in checkpoints],
    "teacher_partial_exists": teacher_partial.is_file(),
    "teacher_final_exists": teacher_final.is_file(),
    "selected_adapter_exists": (root / "selected_adapter" / "adapter.pt").is_file(),
}
(root / "runtime_failure_state.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  sync
}

trap preserve_failure_state EXIT

prepare_persistent_stage_storage() {
  mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/rgb_teacher_bank"

  # A previous no-cache run may have left dangling tmpfs symlinks behind.
  # Replace them before any expensive model work so the first checkpoint can
  # never discover the storage problem hours into training.
  if [[ -L "$OUTPUT_ROOT/checkpoints" ]]; then
    rm -f "$OUTPUT_ROOT/checkpoints"
  elif [[ -e "$OUTPUT_ROOT/checkpoints" && ! -d "$OUTPUT_ROOT/checkpoints" ]]; then
    echo "PHASE_18_CHECKPOINT_PATH_INVALID: $OUTPUT_ROOT/checkpoints" >&2
    exit 71
  fi
  mkdir -p "$OUTPUT_ROOT/checkpoints"
  test -d "$OUTPUT_ROOT/checkpoints"
  test -w "$OUTPUT_ROOT/checkpoints"
  checkpoint_probe="$OUTPUT_ROOT/checkpoints/.phase18-write-probe-$$"
  printf 'phase18-checkpoint-storage-ok\n' > "$checkpoint_probe"
  test -s "$checkpoint_probe"
  rm -f "$checkpoint_probe"

  if [[ -L "$OUTPUT_ROOT/rgb_teacher_bank/teacher_bank.pt" ]]; then
    rm -f "$OUTPUT_ROOT/rgb_teacher_bank/teacher_bank.pt"
  elif [[ -e "$OUTPUT_ROOT/rgb_teacher_bank/teacher_bank.pt" && ! -f "$OUTPUT_ROOT/rgb_teacher_bank/teacher_bank.pt" ]]; then
    echo "PHASE_18_TEACHER_BANK_PATH_INVALID: $OUTPUT_ROOT/rgb_teacher_bank/teacher_bank.pt" >&2
    exit 72
  fi
}

source "$ENV_ROOT/bin/activate"
cd "$BUNDLE_ROOT"
mkdir -p "$LOG_ROOT"
export PYTHONPATH="$BUNDLE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q \
  tests/test_rgbtir_phase15.py \
  tests/test_rgbtir_phase18.py \
  tests/test_rgbtir_phase18_full.py \
  tests/test_rgbtir_phase18_recovery.py \
  tests/test_rgbtir_query_interface_smoke.py \
  tests/test_rgbtir_phase18_bundle_contract.py \
  tests/test_rgbtir_phase16.py
python tools/run_rgbtir_query_interface_smoke.py --config "$Q0_CONFIG"
python tools/run_rgbtir_query_real_smoke.py --config "$Q0_REAL_CONFIG" --allow-network
python tools/run_rgbtir_phase18_full.py preflight --config "$CONFIG"
# Keep the already verified Teacher Bank partial without repeatedly rewriting
# its full payload.  Final Teacher Bank, four training checkpoints, selected
# Adapter, and official-val pair shards remain durable until local transfer.
export AIC_DISABLE_TEACHER_BANK_CACHE="${AIC_DISABLE_TEACHER_BANK_CACHE:-1}"
export PYTHONPATH="$BUNDLE_ROOT/tools/cloud/nocache_site:$PYTHONPATH"
prepare_persistent_stage_storage
set +e
python tools/run_rgbtir_phase18_full.py train-select --config "$CONFIG" --resume --allow-network \
  2>&1 | tee "$LOG_ROOT/rgbtir_phase18_train_select.log"
train_status="${PIPESTATUS[0]}"
set -e
if [[ "$train_status" -ne 0 && "$train_status" -ne 2 ]]; then
  echo "PHASE_18_TRAIN_SELECT_RUNTIME_FAILURE: exit=$train_status" >&2
  exit "$train_status"
fi
archive_args=(
  --output-root "$OUTPUT_ROOT"
  --archive "$READY_ROOT/aic_rgbtir_phase18_train_select_artifacts_20260815.zip"
  --stage train-select
  --required train_selection_summary.json
  --minimum-checkpoints 4
)
if [[ "$train_status" -eq 0 ]]; then
  archive_args+=(--required selected_adapter/adapter.pt)
fi
python tools/archive_rgbtir_phase18_stage.py "${archive_args[@]}"
sync
if [[ "$train_status" -eq 2 ]]; then
  echo "PHASE_18_TRAIN_SELECT_NO_GO_ARCHIVED"
  exit 0
fi
set +e
python tools/run_rgbtir_phase18_full.py official-val --config "$CONFIG" --resume --allow-network \
  2>&1 | tee "$LOG_ROOT/rgbtir_phase18_official_val.log"
official_status="${PIPESTATUS[0]}"
set -e
if [[ "$official_status" -ne 0 && "$official_status" -ne 2 ]]; then
  echo "PHASE_18_OFFICIAL_VAL_RUNTIME_FAILURE: exit=$official_status" >&2
  exit "$official_status"
fi
bash "$BUNDLE_ROOT/tools/cloud/finalize_rgbtir_phase18_full.sh"
