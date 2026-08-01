from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from aic_baseline.bbox import intersection_over_union
from aic_baseline.model_adapters import GroundingDinoExternalPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test a local Hugging Face GroundingDINO-compatible model."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--custom-modeling-path",
        type=Path,
        help=(
            "Optional directory containing modeling_grounding_dino.py. "
            "Used by the official LLMDet Hugging Face conversion."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--box-threshold", type=float, default=0.05)
    parser.add_argument("--text-threshold", type=float, default=0.05)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument(
        "--use-pytorch-bin",
        action="store_true",
        help="Force pytorch_model.bin instead of model.safetensors.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _normalized_bbox(pixel_bbox: list[float], image: Image.Image) -> list[float]:
    width, height = image.size
    return [
        pixel_bbox[0] / width,
        pixel_bbox[1] / height,
        pixel_bbox[2] / width,
        pixel_bbox[3] / height,
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = json.loads(args.sample_json.read_text(encoding="utf-8"))
    if args.query_id not in records:
        raise KeyError(f"query_id not found: {args.query_id}")
    record = records[args.query_id]
    image_path = (args.dataset_root / record["visible"]).resolve()
    image = Image.open(image_path).convert("RGB")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    load_started = time.perf_counter()
    model_class = None
    if args.custom_modeling_path is not None:
        modeling_path = str(args.custom_modeling_path.resolve())
        if modeling_path not in sys.path:
            sys.path.insert(0, modeling_path)
        module = importlib.import_module("modeling_grounding_dino")
        model_class = module.GroundingDinoForObjectDetection

    predictor = GroundingDinoExternalPredictor(
        model_path=args.model_path,
        model_class=model_class,
        model_load_kwargs=(
            {"use_safetensors": False} if args.use_pytorch_bin else None
        ),
        device=args.device,
        dtype=_dtype(args.dtype),
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_candidates=args.max_candidates,
    )
    load_seconds = time.perf_counter() - load_started

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    inference_started = time.perf_counter()
    prediction = predictor.predict(image=image, query=record["query"])
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    gt_bbox = record.get("bbox")
    candidates: list[dict[str, Any]] = []
    for rank, candidate in enumerate(prediction.candidates, start=1):
        pixel_bbox = [float(value) for value in candidate.pixel_bbox]
        normalized_bbox = _normalized_bbox(pixel_bbox, image)
        candidate_iou = (
            intersection_over_union(normalized_bbox, gt_bbox)
            if gt_bbox is not None
            else None
        )
        candidates.append(
            {
                "rank": rank,
                "pixel_bbox": [round(value, 4) for value in pixel_bbox],
                "normalized_bbox": [round(value, 8) for value in normalized_bbox],
                "score": (
                    None if candidate.score is None else round(float(candidate.score), 8)
                ),
                "label": candidate.label,
                "iou": None if candidate_iou is None else round(candidate_iou, 8),
            }
        )

    ious = [row["iou"] for row in candidates if row["iou"] is not None]
    result = {
        "status": "ok",
        "model": args.model_name,
        "model_path": str(args.model_path.resolve()),
        "custom_modeling_path": (
            None
            if args.custom_modeling_path is None
            else str(args.custom_modeling_path.resolve())
        ),
        "query_id": args.query_id,
        "query": record["query"],
        "image_path": str(image_path),
        "image_size": list(image.size),
        "device": args.device,
        "dtype": args.dtype,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "use_pytorch_bin": args.use_pytorch_bin,
        "load_seconds": round(load_seconds, 4),
        "inference_seconds": round(inference_seconds, 4),
        "peak_vram_mb": (
            round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
            if args.device.startswith("cuda")
            else None
        ),
        "candidate_count": len(candidates),
        "top1_iou": ious[0] if ious else None,
        "best_iou": max(ious) if ious else None,
        "candidates": candidates,
    }
    return result


def main() -> int:
    args = parse_args()
    result = run(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
