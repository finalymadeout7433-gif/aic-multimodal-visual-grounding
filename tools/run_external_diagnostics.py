from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

from aic_baseline.diagnostics import read_jsonl
from aic_baseline.external_data import sha256_file
from aic_baseline.external_eval import run_external_evaluation
from aic_baseline.florence import FlorenceGrounder
from aic_baseline.model_adapters import (
    FlorenceExternalPredictor,
    FlorenceTileExternalPredictor,
    GroundingDinoExternalPredictor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed external visual-grounding diagnostic."
    )
    parser.add_argument(
        "--model",
        choices=("florence", "florence-tile", "grounding-dino"),
        required=True,
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "float32"))
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument(
        "--custom-modeling-path",
        type=Path,
        help=(
            "Optional official modeling_grounding_dino.py directory, used for "
            "LLMDet's Hugging Face conversion."
        ),
    )
    parser.add_argument(
        "--use-pytorch-bin",
        action="store_true",
        help="Force pytorch_model.bin instead of model.safetensors.",
    )
    return parser.parse_args()


def _build_predictor(args: argparse.Namespace, dtype: torch.dtype):
    if args.model in {"florence", "florence-tile"}:
        grounder = FlorenceGrounder(
            model_path=args.model_path,
            device="cuda",
            dtype=dtype,
            selection_strategy="first",
        )
        if args.model == "florence-tile":
            return FlorenceTileExternalPredictor(
                grounder,
                overlap_ratio=0.2,
                dedup_iou=0.85,
            )
        return FlorenceExternalPredictor(grounder)
    model_class = None
    if args.custom_modeling_path is not None:
        modeling_path = str(args.custom_modeling_path.resolve())
        if modeling_path not in sys.path:
            sys.path.insert(0, modeling_path)
        module = importlib.import_module("modeling_grounding_dino")
        model_class = module.GroundingDinoForObjectDetection
    return GroundingDinoExternalPredictor(
        model_path=args.model_path,
        model_class=model_class,
        model_load_kwargs=(
            {"use_safetensors": False} if args.use_pytorch_bin else None
        ),
        device="cuda",
        dtype=dtype,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_candidates=args.max_candidates,
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for external diagnostics")
    resolved_dtype = args.dtype or (
        "float32" if args.model == "grounding-dino" else "float16"
    )
    dtype = torch.float16 if resolved_dtype == "float16" else torch.float32
    records = read_jsonl(args.subset)
    if args.limit is not None:
        records = records[: args.limit]
    fingerprint = {
        "model": args.model,
        "model_path_name": args.model_path.name,
        "model_weights_sha256": sha256_file(
            args.model_path / "model.safetensors"
        ),
        "subset_sha256": sha256_file(args.subset),
        "record_limit": args.limit,
        "dtype": resolved_dtype,
        "box_threshold": (
            args.box_threshold if args.model == "grounding-dino" else None
        ),
        "text_threshold": (
            args.text_threshold if args.model == "grounding-dino" else None
        ),
        "max_candidates": (
            args.max_candidates if args.model == "grounding-dino" else None
        ),
        "custom_modeling_path": (
            str(args.custom_modeling_path.resolve())
            if args.custom_modeling_path is not None
            else None
        ),
        "custom_modeling_sha256": (
            sha256_file(args.custom_modeling_path / "modeling_grounding_dino.py")
            if args.custom_modeling_path is not None
            else None
        ),
        "use_pytorch_bin": args.use_pytorch_bin,
        "tile_overlap_ratio": 0.2 if args.model == "florence-tile" else None,
        "tile_dedup_iou": 0.85 if args.model == "florence-tile" else None,
    }
    predictor = _build_predictor(args, dtype)
    summary = run_external_evaluation(
        records=records,
        data_root=args.data_root,
        predictor=predictor,
        output_dir=args.output_dir,
        run_fingerprint=fingerprint,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
