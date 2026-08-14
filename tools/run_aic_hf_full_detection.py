from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import transformers

from aic_baseline.aic_full_detection import (
    finalize_aic_full_detection,
    run_aic_full_detection,
    sha256_file,
)
from aic_baseline.data import AICDataset
from aic_baseline.model_adapters import GroundingDinoExternalPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local Hugging Face GroundingDINO-compatible model over the "
            "full unlabeled AIC test set and optionally build a platform ZIP."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--custom-modeling-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--autocast-dtype",
        choices=("none", "float16", "bfloat16"),
        default="none",
    )
    parser.add_argument("--box-threshold", type=float, default=0.0)
    parser.add_argument("--text-threshold", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _directory_code_hash(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*.py") if item.is_file())
    for file_path in files:
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest().upper()


def _runtime_assets(model_path: Path) -> dict[str, str]:
    names = (
        "config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "conversion_manifest.json",
    )
    return {
        name: sha256_file(model_path / name)
        for name in names
        if (model_path / name).is_file()
    }


def _load_custom_model(path: Path | None) -> type[Any] | None:
    if path is None:
        return None
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    module = importlib.import_module("modeling_grounding_dino")
    return module.GroundingDinoForObjectDetection


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    for path in (args.dataset_root, args.queries, args.model_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.custom_modeling_path is not None and not args.custom_modeling_path.exists():
        raise FileNotFoundError(args.custom_modeling_path)

    model_path = args.model_path.resolve()
    weights_path = model_path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    dataset = AICDataset(
        dataset_root=args.dataset_root,
        queries_path=args.queries,
    )
    custom_model_class = _load_custom_model(args.custom_modeling_path)
    fingerprint = {
        "schema": "aic-zero-shot-full-v1",
        "model_name": args.model_name,
        "model_path_name": model_path.name,
        "model_weights_sha256": sha256_file(weights_path),
        "model_runtime_asset_sha256": _runtime_assets(model_path),
        "custom_modeling_sha256": _directory_code_hash(args.custom_modeling_path),
        "inference_code_sha256": {
            "runner": sha256_file(Path(__file__)),
            "core": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src"
                / "aic_baseline"
                / "aic_full_detection.py"
            ),
            "model_adapter": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src"
                / "aic_baseline"
                / "model_adapters.py"
            ),
        },
        "queries_sha256": sha256_file(args.queries),
        "query_count": len(dataset),
        "dtype": args.dtype,
        "autocast_dtype": args.autocast_dtype,
        "device": args.device,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "max_candidates": args.max_candidates,
        "batch_size": args.batch_size,
        "selection_policy": "native_highest_score_top1",
        "record_limit": args.limit,
        "training_on_aic": False,
        "modalities": ["visible_rgb", "query_original"],
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
    }

    load_started = time.perf_counter()
    predictor = GroundingDinoExternalPredictor(
        model_path=model_path,
        model_name=args.model_name,
        model_class=custom_model_class,
        device=args.device,
        dtype=_dtype(args.dtype),
        autocast_dtype=(
            None if args.autocast_dtype == "none" else _dtype(args.autocast_dtype)
        ),
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_candidates=args.max_candidates,
    )
    load_seconds = time.perf_counter() - load_started
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    progress_started = time.perf_counter()

    def progress(completed: int, total: int, query_id: str) -> None:
        if completed == total or completed % 25 == 0:
            elapsed = max(time.perf_counter() - progress_started, 1e-9)
            rate = completed / elapsed
            eta = (total - completed) / rate if rate > 0 else 0.0
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": total,
                        "query_id": query_id,
                        "queries_per_second": round(rate, 4),
                        "eta_seconds": round(eta, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    result = run_aic_full_detection(
        dataset=dataset,
        predictor=predictor,
        output_dir=args.output_dir,
        run_fingerprint=fingerprint,
        resume=args.resume,
        limit=args.limit,
        batch_size=args.batch_size,
        progress_callback=progress,
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": (
            torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None
        ),
        "model_load_seconds": load_seconds,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if args.device.startswith("cuda")
            else 0
        ),
    }
    _write_json(args.output_dir / "environment.json", environment)

    finalized = None
    if args.finalize:
        if args.limit is not None:
            raise ValueError("cannot finalize a limited probe run")
        finalized = finalize_aic_full_detection(
            dataset=dataset,
            detection_dir=args.output_dir,
            submission_dir=args.output_dir / "submission",
        )
    print(
        json.dumps(
            {
                "summary": result.summary,
                "submission_zip": (
                    None if finalized is None else str(finalized.zip_path)
                ),
                "submission_audit": (
                    None if finalized is None else finalized.audit
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
