from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_baseline.aic_full_detection import (  # noqa: E402
    finalize_aic_full_detection,
    run_aic_full_detection,
    sha256_file,
)
from aic_baseline.data import AICDataset  # noqa: E402
from aic_baseline.external_eval import (  # noqa: E402
    ExternalPrediction,
    ModelCandidate,
)


BOX_PATTERN = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
REF_PATTERN = re.compile(r"<ref>(.*?)</ref>\s*<box>")
FALLBACK_CENTER_PIXEL_BOX = [0.25, 0.25, 0.75, 0.75]


def _fallback_pixel_bbox(width: int, height: int) -> list[float]:
    return [
        FALLBACK_CENTER_PIXEL_BOX[0] * width,
        FALLBACK_CENTER_PIXEL_BOX[1] * height,
        FALLBACK_CENTER_PIXEL_BOX[2] * width,
        FALLBACK_CENTER_PIXEL_BOX[3] * height,
    ]


def _parsed_box_to_pixel(
    values: list[int],
    *,
    width: int,
    height: int,
) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in values]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    left = max(0.0, min(1000.0, left))
    right = max(0.0, min(1000.0, right))
    top = max(0.0, min(1000.0, top))
    bottom = max(0.0, min(1000.0, bottom))
    if right - left < 1.0 or bottom - top < 1.0:
        return None
    return [
        left / 1000.0 * width,
        top / 1000.0 * height,
        right / 1000.0 * width,
        bottom / 1000.0 * height,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LocateAnything-3B over the unlabeled AIC test set."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--model-path", default="nvidia/LocateAnything-3B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-batch-runtime", action="store_true")
    parser.add_argument("--attn", choices=("sdpa", "la_flash"), default="la_flash")
    parser.add_argument(
        "--vision-attn",
        choices=("sdpa", "flash_attention_2"),
        default="flash_attention_2",
    )
    parser.add_argument("--scheduler", default="pipeline")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=None,
        help=(
            "Resize the image fed to LocateAnything so the longest side is at most "
            "this value. Parsed 0-1000 coordinates are still mapped back to the "
            "original image size for the AIC submission."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_hash() -> str:
    return sha256_file(Path(__file__))


class LocateAnythingPredictor:
    model_name = "nvidia/LocateAnything-3B"

    def __init__(
        self,
        *,
        model_path: str,
        use_batch_runtime: bool,
        attn: str,
        vision_attn: str,
        scheduler: str,
        max_image_side: int | None,
    ) -> None:
        from locateanything_worker import LocateAnythingWorker

        kwargs: dict[str, Any] = {}
        if use_batch_runtime:
            kwargs.update(
                {
                    "use_batch_runtime": True,
                    "attn": attn,
                    "vision_attn": vision_attn,
                    "scheduler": scheduler,
                }
            )
        self.model_name = model_path
        self.worker = LocateAnythingWorker(model_path, **kwargs)
        self.max_image_side = max_image_side

    def _model_image(self, image: Image.Image) -> Image.Image:
        if self.max_image_side is None:
            return image
        if self.max_image_side <= 0:
            raise ValueError("max_image_side must be positive")
        width, height = image.size
        longest = max(width, height)
        if longest <= self.max_image_side:
            return image
        scale = self.max_image_side / float(longest)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
        return resized

    @staticmethod
    def _label_for_box(answer: str, box_start: int, default: str) -> str:
        prefix = answer[:box_start]
        matches = list(REF_PATTERN.finditer(prefix + "<box>"))
        if not matches:
            return default
        label = matches[-1].group(1).strip()
        return label or default

    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        width, height = image.size
        model_image = self._model_image(image)
        raw = self.worker.ground_single(model_image, query)
        answer_probe = raw.get("answer") if isinstance(raw, dict) else str(raw)
        if not BOX_PATTERN.search(str(answer_probe or "")):
            raw = self.worker.ground_gui(model_image, query, output_type="box")
            answer_probe = raw.get("answer") if isinstance(raw, dict) else str(raw)
        if not BOX_PATTERN.search(str(answer_probe or "")):
            raw = self.worker.ground_multi(model_image, query)
        answer = raw.get("answer") if isinstance(raw, dict) else str(raw)
        if answer is None:
            answer = ""
        answer = str(answer)
        candidates: list[ModelCandidate] = []
        for index, match in enumerate(BOX_PATTERN.finditer(answer)):
            pixel_bbox = _parsed_box_to_pixel(
                [int(value) for value in match.groups()],
                width=width,
                height=height,
            )
            if pixel_bbox is None:
                continue
            candidates.append(
                ModelCandidate(
                    pixel_bbox=pixel_bbox,
                    label=self._label_for_box(answer, match.start(), query),
                    score=float(index * -1),
                    source="locateanything",
                )
            )
        if not candidates:
            candidates.append(
                ModelCandidate(
                    pixel_bbox=_fallback_pixel_bbox(width, height),
                    label=query,
                    score=-999.0,
                    source="locateanything_fallback_center",
                )
            )
        return ExternalPrediction(raw_output=raw, candidates=candidates)


def main() -> int:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)
    if not args.queries.exists():
        raise FileNotFoundError(args.queries)

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    dataset = AICDataset(dataset_root=args.dataset_root, queries_path=args.queries)
    fingerprint = {
        "schema": "aic-locateanything-zero-shot-v1",
        "model_name": args.model_path,
        "queries_sha256": sha256_file(args.queries),
        "query_count": len(dataset),
        "runner_sha256": _source_hash(),
        "selection_policy": "first_parsed_locateanything_box",
        "training_on_aic": False,
        "modalities": ["visible_rgb", "query_original"],
        "batch_size": args.batch_size,
        "use_batch_runtime": args.use_batch_runtime,
        "attn": args.attn,
        "vision_attn": args.vision_attn,
        "scheduler": args.scheduler,
        "record_limit": args.limit,
        "max_image_side": args.max_image_side,
    }

    load_started = time.perf_counter()
    predictor = LocateAnythingPredictor(
        model_path=args.model_path,
        use_batch_runtime=args.use_batch_runtime,
        attn=args.attn,
        vision_attn=args.vision_attn,
        scheduler=args.scheduler,
        max_image_side=args.max_image_side,
    )
    load_seconds = time.perf_counter() - load_started
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()

    def progress(completed: int, total: int, query_id: str) -> None:
        if completed == total or completed % 25 == 0:
            elapsed = max(time.perf_counter() - started, 1e-9)
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
        "model_load_seconds": load_seconds,
        "torch": None if torch is None else torch.__version__,
        "torch_cuda": None if torch is None else torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch is not None and torch.cuda.is_available()
            else None
        ),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch is not None and torch.cuda.is_available()
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
                "submission_audit": None if finalized is None else finalized.audit,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
