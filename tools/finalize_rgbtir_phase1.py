from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize a successful RGB-TIR Phase 1 run")
    parser.add_argument("--phase1-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    full_root = args.phase1_root / "full"
    summary_path = full_root / "run_summary.json"
    checkpoint_path = full_root / "checkpoint_last.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("full run summary or checkpoint is missing")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PHASE_1_WARMUP_GO":
        raise RuntimeError(f"full warmup is not GO: {summary.get('status')!r}")
    if summary.get("steps") != 26477 or summary.get("skipped") != 0:
        raise RuntimeError("full warmup count invariant failed")
    if not summary.get("base_unchanged"):
        raise RuntimeError("frozen Qwen base changed during warmup")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter = checkpoint.get("tir_adapter")
    if not isinstance(adapter, dict) or not adapter:
        raise RuntimeError("checkpoint has no TIR adapter state")
    unexpected = [name for name in adapter if ".paths.tir." not in name]
    nonfinite = [name for name, tensor in adapter.items() if not torch.isfinite(tensor).all()]
    if unexpected or nonfinite:
        raise RuntimeError(f"adapter validation failed: unexpected={unexpected}, nonfinite={nonfinite}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    adapter_path = args.output_root / "qwen3vl8b_tir_rank48_adapter_phase1.pt"
    release = {
        "format_version": 1,
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "phase": "AIC_RGBTIR_PHASE_1",
        "training_scope": "TIR qkv/proj rank-48 adapter only",
        "claim_boundary": summary["claim_boundary"],
        "config_sha256": checkpoint["config_sha256"],
        "base_sha256": checkpoint["base_sha256"],
        "step": checkpoint["step"],
        "validation": {
            "baseline": summary["baseline_evaluation"],
            "final": summary["final_evaluation"],
            "relative_loss_improvement": summary["relative_loss_improvement"],
        },
        "tir_adapter": adapter,
    }
    torch.save(release, adapter_path)

    reloaded = torch.load(adapter_path, map_location="cpu", weights_only=False)
    if set(reloaded["tir_adapter"]) != set(adapter):
        raise RuntimeError("adapter key set changed after export/reload")
    for name, tensor in adapter.items():
        if not torch.equal(tensor, reloaded["tir_adapter"][name]):
            raise RuntimeError(f"adapter tensor changed after export/reload: {name}")

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": command_output([sys.executable, "-c", "import transformers; print(transformers.__version__)"]),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_branch": command_output(["git", "branch", "--show-current"]),
    }
    write_json(args.output_root / "environment.json", environment)

    copied_metadata = {}
    for source in (
        args.phase1_root / "preflight.json",
        args.phase1_root / "vision_assets.json",
        args.phase1_root / "smoke_summary.json",
        args.phase1_root / "overfit100" / "run_summary.json",
        args.phase1_root / "tracer400" / "run_summary.json",
        full_root / "run_summary.json",
        full_root / "train_log.jsonl",
        args.config,
    ):
        if source.is_file():
            target = args.output_root / source.name
            if target.exists():
                target = args.output_root / f"{source.parent.name}_{source.name}"
            target.write_bytes(source.read_bytes())
            copied_metadata[target.name] = str(source)
    write_json(args.output_root / "source_files.json", copied_metadata)

    manifest = {}
    for path in sorted(args.output_root.iterdir()):
        if path.is_file() and path.name != "sha256_manifest.json":
            manifest[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(args.output_root / "sha256_manifest.json", manifest)

    final = {
        "status": "PHASE_1_RELEASE_READY",
        "adapter_path": str(adapter_path),
        "adapter_tensor_count": len(adapter),
        "adapter_parameter_count": sum(tensor.numel() for tensor in adapter.values()),
        "adapter_sha256": manifest[adapter_path.name]["sha256"],
        "full_steps": summary["steps"],
        "skipped": summary["skipped"],
        "relative_loss_improvement": summary["relative_loss_improvement"],
        "base_unchanged": summary["base_unchanged"],
        "next_phase": "Phase 2 query-aware RGB-TIR fusion; not an AIC submission checkpoint",
    }
    write_json(args.output_root / "release_summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
