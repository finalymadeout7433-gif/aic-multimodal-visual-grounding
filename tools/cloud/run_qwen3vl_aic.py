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


JSON_ARRAY_PATTERN = re.compile(r"\[[^\[\]]+\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL-8B-Instruct over the unlabeled AIC test set."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--attn", choices=("auto", "flash_attention_2", "sdpa"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--retry-max-new-tokens",
        type=int,
        default=0,
        help=(
            "If greater than max-new-tokens, retry only parser-failed single-sample "
            "generations with this larger token budget before using fallback."
        ),
    )
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=1310720)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--enable-thinking",
        choices=("auto", "true", "false"),
        default="auto",
        help=(
            "Qwen3 chat-template thinking switch. Use false to force the "
            "Thinking checkpoints into non-thinking mode for structured bbox output."
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


def _dtype_value(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _extract_box(text: str) -> list[float]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            for key in ("bbox", "box", "answer"):
                value = payload.get(key)
                if isinstance(value, list) and len(value) == 4:
                    return [float(item) for item in value]
        if isinstance(payload, list) and len(payload) == 4:
            return [float(item) for item in payload]
    except json.JSONDecodeError:
        pass
    for match in JSON_ARRAY_PATTERN.finditer(stripped):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and len(value) == 4:
            return [float(item) for item in value]
    raise ValueError(f"no bbox found in model output: {text[:200]}")


def _to_pixel_bbox(box: list[float], *, width: int, height: int) -> list[float]:
    if max(abs(value) for value in box) <= 1.5:
        return [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
    if max(abs(value) for value in box) <= 1000.0:
        return [
            box[0] / 1000.0 * width,
            box[1] / 1000.0 * height,
            box[2] / 1000.0 * width,
            box[3] / 1000.0 * height,
        ]
    return box


def _sanitize_pixel_bbox(box: list[float], *, width: int, height: int) -> list[float] | None:
    if len(box) != 4:
        return None
    values = [float(value) for value in box]
    if any(value != value for value in values):
        return None
    x1, y1, x2, y2 = values
    left, right = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    top, bottom = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if right - left < 1.0 or bottom - top < 1.0:
        return None
    return [left, top, right, bottom]


def _fallback_pixel_bbox(*, width: int, height: int) -> list[float]:
    return [
        width * 0.25,
        height * 0.25,
        width * 0.75,
        height * 0.75,
    ]


class Qwen3VLPredictor:
    def __init__(
        self,
        *,
        model_path: str,
        dtype: str,
        attn: str,
        min_pixels: int,
        max_pixels: int,
        max_new_tokens: int,
        retry_max_new_tokens: int,
        enable_thinking: str,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        kwargs: dict[str, Any] = {
            "dtype": _dtype_value(torch, dtype),
            "device_map": "auto",
        }
        if attn != "auto":
            kwargs["attn_implementation"] = attn
        self.model_name = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.size = {
                "shortest_edge": min_pixels,
                "longest_edge": max_pixels,
            }
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.retry_max_new_tokens = retry_max_new_tokens
        self.enable_thinking = enable_thinking

    @staticmethod
    def _prompt(query: str) -> str:
        return (
            "Locate the target described by the query in the image. "
            "Return only one JSON object in this exact format: "
            "{\"bbox\":[x1,y1,x2,y2]}. "
            "Use normalized coordinates from 0 to 1. "
            "Even if the target is hard to see or you are uncertain, choose the best approximate region. "
            "Never answer that the object is absent. "
            "Do not include explanations. Query: "
            + query
        )

    def _messages(self, *, image: Image.Image, query: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._prompt(query)},
                ],
            }
        ]

    def _prepare_batch_inputs(self, messages_batch: list[list[dict[str, Any]]]) -> Any:
        from qwen_vl_utils import process_vision_info

        texts = []
        for messages in messages_batch:
            chat_template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.enable_thinking != "auto":
                chat_template_kwargs["enable_thinking"] = self.enable_thinking == "true"
            text = self.processor.apply_chat_template(
                messages,
                **chat_template_kwargs,
            )
            texts.append(text)
        image_inputs, video_inputs = process_vision_info(
            [message for messages in messages_batch for message in messages]
        )
        return self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    def _prediction_from_text(
        self,
        *,
        text: str,
        image: Image.Image,
        query: str,
        success_source: str = "qwen3vl",
    ) -> ExternalPrediction:
        width, height = image.size
        source = success_source
        score = 1.0
        try:
            bbox = _sanitize_pixel_bbox(
                _to_pixel_bbox(_extract_box(text), width=width, height=height),
                width=width,
                height=height,
            )
        except Exception as exc:  # noqa: BLE001 - preserve full-run progress on bad generations.
            bbox = None
            source = "qwen3vl_fallback_center"
            score = 0.0
            text = text + f"\n[PARSER_FALLBACK] {type(exc).__name__}: {exc}"
        if bbox is None:
            bbox = _fallback_pixel_bbox(width=width, height=height)
            source = "qwen3vl_fallback_center"
            score = 0.0
        return ExternalPrediction(
            raw_output={"text": text},
            candidates=[
                ModelCandidate(
                    pixel_bbox=bbox,
                    label=query,
                    score=score,
                    source=source,
                )
            ],
        )

    def _generate_single_text(
        self,
        *,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
    ) -> str:
        if self.enable_thinking != "auto":
            inputs = self._prepare_batch_inputs([messages])
        else:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self.model.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        messages = self._messages(image=image, query=query)
        text = self._generate_single_text(
            messages=messages,
            max_new_tokens=self.max_new_tokens,
        )
        prediction = self._prediction_from_text(text=text, image=image, query=query)
        if (
            prediction.candidates[0].source == "qwen3vl_fallback_center"
            and self.retry_max_new_tokens > self.max_new_tokens
        ):
            retry_text = self._generate_single_text(
                messages=messages,
                max_new_tokens=self.retry_max_new_tokens,
            )
            retry_prediction = self._prediction_from_text(
                text=retry_text,
                image=image,
                query=query,
                success_source="qwen3vl_retry",
            )
            if retry_prediction.candidates[0].source != "qwen3vl_fallback_center":
                retry_prediction.raw_output["initial_text"] = text
                retry_prediction.raw_output["retry_max_new_tokens"] = (
                    self.retry_max_new_tokens
                )
                return retry_prediction
        return prediction

    def predict_batch(
        self,
        *,
        images: list[Image.Image],
        queries: list[str],
    ) -> list[ExternalPrediction]:
        messages_batch = [
            self._messages(image=image, query=query)
            for image, query in zip(images, queries)
        ]
        inputs = self._prepare_batch_inputs(messages_batch).to(self.model.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        texts = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [
            self._prediction_from_text(text=text, image=image, query=query)
            for text, image, query in zip(texts, images, queries)
        ]


def main() -> int:
    args = parse_args()
    import torch

    dataset = AICDataset(dataset_root=args.dataset_root, queries_path=args.queries)
    fingerprint = {
        "schema": "aic-qwen3vl-zero-shot-v1",
        "model_name": args.model_path,
        "queries_sha256": sha256_file(args.queries),
        "query_count": len(dataset),
        "runner_sha256": sha256_file(Path(__file__)),
        "selection_policy": "single_generated_bbox",
        "training_on_aic": False,
        "modalities": ["visible_rgb", "query_original"],
        "dtype": args.dtype,
        "attn": args.attn,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "retry_max_new_tokens": args.retry_max_new_tokens,
        "enable_thinking": args.enable_thinking,
        "batch_size": args.batch_size,
        "record_limit": args.limit,
    }
    load_started = time.perf_counter()
    predictor = Qwen3VLPredictor(
        model_path=args.model_path,
        dtype=args.dtype,
        attn=args.attn,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        retry_max_new_tokens=args.retry_max_new_tokens,
        enable_thinking=args.enable_thinking,
    )
    load_seconds = time.perf_counter() - load_started
    if torch.cuda.is_available():
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
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_load_seconds": load_seconds,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
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
