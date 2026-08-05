#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/aic-multimodal-visual-grounding}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/aic_data}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/aic_qwen3_vl_8b_zero_shot_v1}"
LIMIT_ARG="${LIMIT_ARG:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-aic-qwen3vl}"

python "$PROJECT_ROOT/tools/cloud/run_qwen3vl_aic.py" \
  --dataset-root "$DATASET_ROOT" \
  --queries "$DATASET_ROOT/queries/queries.json" \
  --model-path "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --dtype "${QWEN_DTYPE:-bfloat16}" \
  --attn "${QWEN_ATTN:-flash_attention_2}" \
  --min-pixels "${QWEN_MIN_PIXELS:-262144}" \
  --max-pixels "${QWEN_MAX_PIXELS:-1310720}" \
  --max-new-tokens "${QWEN_MAX_NEW_TOKENS:-96}" \
  --resume \
  ${LIMIT_ARG} \
  ${FINALIZE_ARG:-}
