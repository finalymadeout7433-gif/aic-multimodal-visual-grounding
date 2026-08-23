#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-/home/featurize/aic_rgbtir_phase19_d2_full_bundle}"
ENV_ROOT="${ENV_ROOT:-/home/featurize/envs/aic-rgbtir-phase19-d2}"
CONFIG="$BUNDLE_ROOT/configs/aic_rgbtir_phase19_d2_full.cloud.example.yaml"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/featurize/aic_cloud/outputs/aic_rgbtir_phase19_d2_full_v1}"
LOG_ROOT="${LOG_ROOT:-/home/featurize/aic_cloud/logs}"
REPORT_ROOT="${REPORT_ROOT:-/home/featurize/aic_cloud/reports}"
READY_ROOT="${READY_ROOT:-/home/featurize/aic_cloud/platform_upload_ready}"
STAGE_A_ARCHIVE="$READY_ROOT/aic_rgbtir_phase19_d2_train_select_artifacts_20260816.zip"

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
payload = {
    "status": "PHASE_19_D2_RUNTIME_FAILURE_PRESERVED",
    "exit_status": int(sys.argv[2]),
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "checkpoint_count": len(checkpoints),
    "checkpoints": [str(path) for path in checkpoints],
    "selected_adapter_exists": (root / "selected_adapter" / "adapter.pt").is_file(),
    "train_selection_exists": (root / "train_selection_summary.json").is_file(),
    "official_cache_exists": (root / "official_val" / "embedding_cache.pt").is_file(),
    "official_part_count": len(list((root / "official_val" / "embedding_cache_parts").glob("*.pt"))),
}
(root / "runtime_failure_state.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  sync
}

trap preserve_failure_state EXIT

prepare_persistent_storage() {
  mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/official_val"
  if [[ -L "$OUTPUT_ROOT/checkpoints" ]]; then
    rm -f "$OUTPUT_ROOT/checkpoints"
  elif [[ -e "$OUTPUT_ROOT/checkpoints" && ! -d "$OUTPUT_ROOT/checkpoints" ]]; then
    echo "PHASE_19_D2_CHECKPOINT_PATH_INVALID: $OUTPUT_ROOT/checkpoints" >&2
    exit 71
  fi
  mkdir -p "$OUTPUT_ROOT/checkpoints"
  test -d "$OUTPUT_ROOT/checkpoints"
  test -w "$OUTPUT_ROOT/checkpoints"
  probe="$OUTPUT_ROOT/checkpoints/.phase19-write-probe-$$"
  printf 'phase19-d2-persistent-checkpoint-storage-ok\n' > "$probe"
  test -s "$probe"
  rm -f "$probe"
  if [[ -L "$OUTPUT_ROOT/official_val/embedding_cache_parts" ]]; then
    rm -f "$OUTPUT_ROOT/official_val/embedding_cache_parts"
  fi
  mkdir -p "$OUTPUT_ROOT/official_val/embedding_cache_parts"
  test -w "$OUTPUT_ROOT/official_val/embedding_cache_parts"
}

source "$ENV_ROOT/bin/activate"
cd "$BUNDLE_ROOT"
mkdir -p "$LOG_ROOT" "$REPORT_ROOT" "$READY_ROOT"
export PYTHONPATH="$BUNDLE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q \
  tests/test_rgbtir_phase15.py \
  tests/test_rgbtir_phase16.py \
  tests/test_rgbtir_phase18_full.py \
  tests/test_rgbtir_phase18r_audit.py \
  tests/test_rgbtir_phase19_d2.py \
  tests/test_rgbtir_phase19_d2_full.py
python tools/run_rgbtir_phase19_d2_full.py preflight --config "$CONFIG"
prepare_persistent_storage

set +e
python tools/run_rgbtir_phase19_d2_full.py train-select \
  --config "$CONFIG" --resume --allow-network \
  2>&1 | tee "$LOG_ROOT/rgbtir_phase19_d2_train_select.log"
train_status="${PIPESTATUS[0]}"
set -e
if [[ "$train_status" -ne 0 && "$train_status" -ne 2 ]]; then
  echo "PHASE_19_D2_TRAIN_SELECT_RUNTIME_FAILURE: exit=$train_status" >&2
  exit "$train_status"
fi

archive_args=(
  --output-root "$OUTPUT_ROOT"
  --archive "$STAGE_A_ARCHIVE"
  --stage train-select
  --required train_selection_summary.json
  --required sha256_manifest.json
  --minimum-checkpoints 4
)
if [[ "$train_status" -eq 0 ]]; then
  archive_args+=(--required selected_adapter/adapter.pt)
fi
python tools/archive_rgbtir_phase18_stage.py "${archive_args[@]}"
sync

if [[ "$train_status" -eq 2 ]]; then
  echo "PHASE_19_D2_TRAIN_SELECT_NO_GO_ARCHIVED"
  exit 0
fi

set +e
python tools/run_rgbtir_phase19_d2_full.py official-val \
  --config "$CONFIG" --resume --allow-network \
  2>&1 | tee "$LOG_ROOT/rgbtir_phase19_d2_official_val.log"
official_status="${PIPESTATUS[0]}"
set -e
if [[ "$official_status" -ne 0 && "$official_status" -ne 2 ]]; then
  echo "PHASE_19_D2_OFFICIAL_VAL_RUNTIME_FAILURE: exit=$official_status" >&2
  exit "$official_status"
fi

bash "$BUNDLE_ROOT/tools/cloud/finalize_rgbtir_phase19_d2_full.sh"
