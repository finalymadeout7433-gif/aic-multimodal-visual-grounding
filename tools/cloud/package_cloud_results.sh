#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/aic-multimodal-visual-grounding}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/aic_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs}"
READY_DIR="${READY_DIR:-$OUTPUT_ROOT/platform_upload_ready}"

LOCATE_ZIP="$OUTPUT_ROOT/aic_locateanything_3b_zero_shot_v1/submission/predictions_submission.zip"
QWEN_ZIP="$OUTPUT_ROOT/aic_qwen3_vl_8b_zero_shot_v1/submission/predictions_submission.zip"

ARGS=(--queries "$DATASET_ROOT/queries/queries.json" --output-dir "$READY_DIR")
if [ -f "$LOCATE_ZIP" ]; then
  ARGS+=(--locateanything-zip "$LOCATE_ZIP")
fi
if [ -f "$QWEN_ZIP" ]; then
  ARGS+=(--qwen3vl-zip "$QWEN_ZIP")
fi

python "$PROJECT_ROOT/tools/cloud/build_two_model_platform_release.py" "${ARGS[@]}"
echo "Upload-ready files are in: $READY_DIR"
