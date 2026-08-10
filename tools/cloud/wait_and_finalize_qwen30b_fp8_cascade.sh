#!/usr/bin/env bash
set -euo pipefail

CLOUD_ROOT="/home/featurize/aic_cloud"
REPO="$CLOUD_ROOT/aic-multimodal-visual-grounding"
DATA_ROOT="$CLOUD_ROOT/aic_data/aic_round1"
QUERIES="$DATA_ROOT/queries/queries.json"
PRIMARY_OUT="$CLOUD_ROOT/outputs/aic_qwen3_vl_30b_a3b_fp8_zero_shot_v1"
CASCADE_OUT="$CLOUD_ROOT/outputs/aic_qwen3_vl_30b_a3b_fp8_zero_shot_v1_cascade_8b_fallback"
FALLBACK_SUBMISSION="$CLOUD_ROOT/fallback_sources/qwen3_vl_8b_instruct_20260807/predictions_submission.json"
PLATFORM_ZIP="$CLOUD_ROOT/platform_upload_ready/AIC_Qwen3_VL_30B_A3B_FP8_8BInstructFallback_20260809.zip"
RUN_LOG="$CLOUD_ROOT/logs/qwen30b_fp8_full.log"

cd "$REPO"
source /environment/miniconda3/etc/profile.d/conda.sh
conda activate "$CLOUD_ROOT/envs/qwen30fp8"

python tools/cloud/run_qwen3vl_openai_aic.py \
  --dataset-root "$DATA_ROOT" \
  --queries "$QUERIES" \
  --output-dir "$PRIMARY_OUT" \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model qwen3vl30b_fp8 \
  --concurrency 1 \
  --timeout-seconds 180 \
  --max-retries 1 \
  --max-tokens 128 \
  --temperature 0 \
  --fallback center \
  --progress-every 25 \
  --resume \
  --finalize | tee -a "$RUN_LOG"

python tools/cloud/finalize_cascade_fallback_submission.py \
  --queries "$QUERIES" \
  --primary-output-dir "$PRIMARY_OUT" \
  --fallback-submission "$FALLBACK_SUBMISSION" \
  --output-dir "$CASCADE_OUT" \
  --platform-zip "$PLATFORM_ZIP" \
  --primary-name "qwen3_vl_30b_a3b_instruct_fp8" \
  --fallback-name "qwen3_vl_8b_instruct_platform_0_7582_20260807"
