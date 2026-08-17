#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-/home/featurize/aic_rgbtir_phase18_bundle}"
ENV_ROOT="${ENV_ROOT:-/home/featurize/work/envs/aic-rgbtir-phase18}"

python3 -m venv --system-site-packages "$ENV_ROOT"
source "$ENV_ROOT/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$BUNDLE_ROOT/requirements-rgbtir-phase1.txt"
python - <<'PY'
import torch, transformers, yaml
print({"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.cuda.is_available()})
PY
