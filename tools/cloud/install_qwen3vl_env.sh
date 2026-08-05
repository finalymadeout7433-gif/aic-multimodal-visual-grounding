#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-aic-qwen3vl}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/aic-multimodal-visual-grounding}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3-VL-8B-Instruct}"

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install -e "$PROJECT_ROOT"
python -m pip install "transformers>=4.57.0" accelerate qwen-vl-utils==0.0.14
python -m pip install "huggingface_hub[cli]" hf_transfer
python -m pip install flash-attn --no-build-isolation || true

mkdir -p "$(dirname "$MODEL_DIR")"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-VL-8B-Instruct --local-dir "$MODEL_DIR"
fi

python -c "from pathlib import Path; print('Qwen3-VL env ready'); print('model_dir', Path('$MODEL_DIR').resolve())"
