from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the local APE runtime prerequisites.")
    parser.add_argument("--ape-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _language_branch_evidence(checkpoint_audit: dict) -> dict:
    prefix = "model_vision.model_language.net"
    prefix_summary = next(
        (
            row
            for row in checkpoint_audit.get("largest_prefixes", [])
            if row.get("prefix") == prefix
        ),
        None,
    )
    tensor_count = int(prefix_summary.get("tensor_count", 0)) if prefix_summary else 0
    return {
        "prefix": prefix,
        "tensor_count": tensor_count,
        "branch_present": tensor_count > 0,
        "strict_load_verified": False,
        "interpretation": (
            "the checkpoint contains a substantial language-model prefix, but "
            "architectural completeness requires a strict model load in the official runtime"
            if tensor_count > 0
            else "the expected language-model prefix was not found"
        ),
    }


def main() -> int:
    args = parse_args()
    repo = args.ape_repo.resolve()
    checkpoint_audit = json.loads(
        args.checkpoint_audit.read_text(encoding="utf-8")
    )
    compiled_extensions = sorted(
        str(path) for path in (repo / "ape").glob("_C*.pyd")
    ) + sorted(str(path) for path in (repo / "ape").glob("_C*.so"))
    dependencies = {
        name: _module_available(name)
        for name in (
            "detectron2",
            "detrex",
            "pycocotools",
            "omegaconf",
            "fvcore",
            "fairscale",
            "lvis",
        )
    }
    toolchain = {
        name: shutil.which(name)
        for name in ("cl", "nvcc", "cmake", "ninja", "wsl", "docker")
    }
    wsl_probe = {"available": toolchain["wsl"] is not None, "configured": False}
    if toolchain["wsl"] is not None:
        completed = subprocess.run(
            [toolchain["wsl"], "-l", "-q"],
            capture_output=True,
            check=False,
            timeout=10,
        )
        wsl_probe["returncode"] = completed.returncode
        wsl_probe["configured"] = (
            completed.returncode == 0 and bool(completed.stdout.replace(b"\x00", b"").strip())
        )
    system = platform.system()
    reasons: list[str] = []
    warnings: list[str] = []
    if system == "Windows":
        warnings.append("official APE/Detectron2 stack is Linux-first")
    for dependency in ("detectron2", "detrex"):
        if not dependencies[dependency]:
            reasons.append(f"missing Python dependency: {dependency}")
    if not compiled_extensions:
        reasons.append("APE custom C++/CUDA extension ape._C is not built")
        if system == "Windows" and (
            toolchain["cl"] is None or toolchain["nvcc"] is None
        ):
            reasons.append("MSVC and/or CUDA Toolkit compiler is unavailable")
        elif system == "Linux" and toolchain["nvcc"] is None:
            reasons.append("CUDA Toolkit compiler is unavailable")
    language_evidence = _language_branch_evidence(checkpoint_audit)

    result = {
        "status": (
            "blocked_native_runtime"
            if reasons
            else "prerequisites_detected_unverified"
        ),
        "strict_runtime_load_verified": False,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "ape_repo": str(repo),
        "dependencies": dependencies,
        "toolchain": toolchain,
        "wsl_probe": wsl_probe,
        "compiled_ape_extensions": compiled_extensions,
        "checkpoint_language_branch_evidence": language_evidence,
        "checkpoint_language_tensor_count": checkpoint_audit.get(
            "language_related_key_count"
        ),
        "external_eva_clip_bootstrap_can_be_skipped": None,
        "external_eva_clip_skip_status": "pending_official_runtime_strict_load",
        "runtime_blockers": reasons,
        "runtime_warnings": warnings,
        "recommended_runtime": (
            "Linux or WSL2 with the official pinned Detectron2, Detrex, "
            "PyTorch and CUDA build chain"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
