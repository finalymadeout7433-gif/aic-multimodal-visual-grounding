#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_ROOT="${ENV_ROOT:-/home/featurize/work/envs/aic_rgbtir_phase16}"
PHASE1_ENV="${PHASE1_ENV:-/home/featurize/work/envs/aic_rgbtir_phase1}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -d "${ENV_ROOT}" && -d "${PHASE1_ENV}" ]]; then
  # The dependency set is intentionally identical to Phase 1. Reusing its
  # immutable environment avoids paid cloud time spent reinstalling CUDA torch.
  ENV_ROOT="${PHASE1_ENV}"
fi

if [[ ! -d "${ENV_ROOT}" ]]; then
  python - <<'PY'
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 12):
    raise SystemExit(f"Python 3.10/3.11 required, got {sys.version}")
PY
  python -m venv "${ENV_ROOT}"
  source "${ENV_ROOT}/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install "torch==2.6.0" \
    --index-url "${TORCH_INDEX_URL}" \
    --trusted-host download.pytorch.org
  python -m pip install -r "${REPO_ROOT}/requirements-rgbtir-phase1.txt" \
    --index-url "${PYPI_INDEX_URL}" \
    --trusted-host pypi.tuna.tsinghua.edu.cn
else
  source "${ENV_ROOT}/bin/activate"
  # Reused Phase 1 environments may predate the Phase 1.7 diagnostics, which
  # require headless OpenCV and SciPy. Installing the pinned lightweight
  # requirements is idempotent and avoids silently assuming they are present.
  python -m pip install -r "${REPO_ROOT}/requirements-rgbtir-phase1.txt" \
    --index-url "${PYPI_INDEX_URL}" \
    --trusted-host pypi.tuna.tsinghua.edu.cn
fi

python - <<'PY'
import cv2, scipy, torch, transformers
from packaging.version import Version
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("opencv", cv2.__version__)
print("scipy", scipy.__version__)
assert Version(torch.__version__.split("+")[0]) >= Version("2.6")
assert transformers.__version__ == "4.57.6"
assert torch.cuda.is_available()
props = torch.cuda.get_device_properties(0)
print("gpu", props.name, "vram_gib", props.total_memory / 1024**3)
assert props.total_memory >= 23 * 1024**3
PY

echo "PHASE16_ENV_READY=${ENV_ROOT}"
