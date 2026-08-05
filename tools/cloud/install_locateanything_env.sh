#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-aic-locateanything}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/aic-multimodal-visual-grounding}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/LocateAnything-3B}"
EAGLE_DIR="${EAGLE_DIR:-/workspace/Eagle}"

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install -e "$PROJECT_ROOT"
python -m pip install "huggingface_hub[cli]" hf_transfer

if [ ! -d "$EAGLE_DIR/.git" ]; then
  git clone https://github.com/NVlabs/Eagle.git "$EAGLE_DIR"
fi
python -m pip install -e "$EAGLE_DIR/Embodied"

mkdir -p "$(dirname "$MODEL_DIR")"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download nvidia/LocateAnything-3B --local-dir "$MODEL_DIR"
fi

python -c "from pathlib import Path; print('LocateAnything env ready'); print('model_dir', Path('$MODEL_DIR').resolve())"
