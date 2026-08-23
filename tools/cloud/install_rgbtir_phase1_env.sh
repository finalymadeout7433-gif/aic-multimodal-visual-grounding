#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/featurize/aic_rgbtir_phase1_bundle/repo}"
ENV_ROOT="${ENV_ROOT:-/home/featurize/work/envs/aic_rgbtir_phase1}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-${PYPI_INDEX_URL}}"

python - <<'PY'
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 12):
    raise SystemExit(f"Python 3.10/3.11 required, got {sys.version}")
print("python", sys.version)
PY

command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -d "${ENV_ROOT}" ]]; then
  # Keep the platform image immutable.  Older Featurize images may ship
  # torch<2.6, so Phase 1 owns a fully isolated CUDA environment.
  python -m venv "${ENV_ROOT}"
fi
source "${ENV_ROOT}/bin/activate"
python -m pip install --upgrade pip
python -m pip install "packaging>=24,<26" \
  --index-url "${PYPI_INDEX_URL}" \
  --trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install "torch==${TORCH_VERSION}" \
  --index-url "${TORCH_INDEX_URL}" \
  --trusted-host download.pytorch.org \
  --trusted-host files.pythonhosted.org \
  --trusted-host pypi.org \
  --trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install -r "${REPO_ROOT}/requirements-rgbtir-phase1.txt" \
  --index-url "${PYPI_INDEX_URL}" \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --trusted-host files.pythonhosted.org

python - <<'PY'
import torch, transformers, safetensors, huggingface_hub
from packaging.version import Version
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("safetensors", safetensors.__version__)
print("huggingface_hub", huggingface_hub.__version__)
assert Version(torch.__version__.split("+")[0]) >= Version("2.6")
assert torch.cuda.is_available()
props = torch.cuda.get_device_properties(0)
print("gpu", props.name, "vram_gib", props.total_memory / 1024**3)
assert props.total_memory >= 23 * 1024**3
assert transformers.__version__ == "4.57.6"
PY

echo "PHASE1_ENV_READY=${ENV_ROOT}"
