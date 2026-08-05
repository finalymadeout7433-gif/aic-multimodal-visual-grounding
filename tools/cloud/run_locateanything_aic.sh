#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/aic-multimodal-visual-grounding}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/aic_data}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/LocateAnything-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/aic_locateanything_3b_zero_shot_v1}"
LIMIT_ARG="${LIMIT_ARG:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-aic-locateanything}"
export PYTHONPATH="$MODEL_DIR:${PYTHONPATH:-}"
export LA_FLASH_ATTN="${LA_FLASH_ATTN:-la_flash}"

python "$PROJECT_ROOT/tools/cloud/run_locateanything_aic.py" \
  --dataset-root "$DATASET_ROOT" \
  --queries "$DATASET_ROOT/queries/queries.json" \
  --model-path "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --use-batch-runtime \
  --attn la_flash \
  --vision-attn flash_attention_2 \
  --scheduler pipeline \
  --batch-size "${BATCH_SIZE:-1}" \
  --resume \
  ${LIMIT_ARG} \
  ${FINALIZE_ARG:-}
