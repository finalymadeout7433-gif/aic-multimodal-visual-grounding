#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-/home/featurize/aic_rgbtir_phase19_d2_bundle}"
ENV_ROOT="${ENV_ROOT:-/home/featurize/envs/aic-rgbtir-phase19-d2}"

python3 -m venv --system-site-packages "$ENV_ROOT"
source "$ENV_ROOT/bin/activate"
python -m pip install --upgrade pip

CAPABILITY="$(python - <<'PY'
import torch
print("".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else "none")
PY
)"
if [[ "$CAPABILITY" == "120" ]]; then
  HAS_SM120="$(python - <<'PY'
import torch
print("yes" if "sm_120" in torch.cuda.get_arch_list() else "no")
PY
)"
  if [[ "$HAS_SM120" != "yes" ]]; then
    python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 \
      'torch==2.7.1' 'torchvision==0.22.1'
  fi
fi
python -m pip install -r "$BUNDLE_ROOT/requirements-rgbtir-phase1.txt"
python - <<'PY'
import torch, transformers
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
# PyTorch wheels may execute a newer minor architecture through compatible
# cubins/PTX without listing that exact architecture (for example, an Ada
# RTX 4090 reports sm_89 while the cu121 wheel lists sm_86 and sm_90).  The
# real compatibility gate is a CUDA kernel smoke test, not exact membership
# in get_arch_list().  Keep the compiled list in the receipt for diagnosis.
compiled_arches = torch.cuda.get_arch_list()
x = torch.ones(8, device="cuda")
assert float(x.sum()) == 8.0
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "device": torch.cuda.get_device_name(0),
    "capability": cap,
    "device_arch": arch,
    "compiled_arches": compiled_arches,
    "exact_arch_listed": arch in compiled_arches,
    "cuda_kernel_smoke": "PASS",
})
PY
