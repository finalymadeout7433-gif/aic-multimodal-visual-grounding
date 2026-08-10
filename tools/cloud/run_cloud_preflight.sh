#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/workspace/aic_data}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/aic-multimodal-visual-grounding}"
QUERIES="${QUERIES:-$DATASET_ROOT/queries/queries.json}"
export DATASET_ROOT PROJECT_ROOT QUERIES

echo "[preflight] system"
date -Is
uname -a
pwd

echo "[preflight] gpu"
nvidia-smi

echo "[preflight] python"
python --version
python - <<'PY'
import json
import os
import shutil
from pathlib import Path

dataset = Path(os.environ["DATASET_ROOT"])
queries = Path(os.environ["QUERIES"])
summary = {
    "dataset_root": str(dataset),
    "queries": str(queries),
    "queries_exists": queries.is_file(),
    "visible_exists": (dataset / "Images" / "visible").is_dir(),
    "infrared_exists": (dataset / "Images" / "infrared").is_dir(),
    "depth_exists": (dataset / "Images" / "depth").is_dir(),
    "disk_free_gb": round(shutil.disk_usage("/workspace").free / (1024**3), 2),
}
if queries.is_file():
    payload = json.loads(queries.read_text(encoding="utf-8-sig"))
    summary["query_count"] = len(payload)
print(json.dumps(summary, ensure_ascii=False, indent=2))
if summary.get("query_count") != 9555:
    raise SystemExit("unexpected query count")
PY

echo "[preflight] project"
test -d "$PROJECT_ROOT"
test -f "$PROJECT_ROOT/pyproject.toml"
python -m pip --version
