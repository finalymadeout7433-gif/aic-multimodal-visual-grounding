from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from PIL import Image
import torch
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from .bbox import intersection_over_union
from .diagnostics import normalize_candidate_label, tile_windows
from .external_eval import (
    ExternalPrediction,
    ModelCandidate,
    normalize_grounding_query,
)
from .florence import FlorenceGrounder, FlorenceOutputError


class FlorenceExternalPredictor:
    model_name = "microsoft/Florence-2-large-ft"

    def __init__(self, grounder: FlorenceGrounder) -> None:
        self.grounder = grounder

    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        try:
            prediction = self.grounder.predict(image=image, query=query)
        except FlorenceOutputError as exc:
            return ExternalPrediction(
                raw_output={"florence_output_error": str(exc)},
                candidates=[],
            )
        return ExternalPrediction(
            raw_output=prediction.raw_output,
            candidates=[
                ModelCandidate(
                    pixel_bbox=candidate.pixel_bbox,
                    label=candidate.label,
                    score=None,
                    source="full",
                )
                for candidate in prediction.candidates
            ],
        )


class FlorenceTileExternalPredictor:
    model_name = "microsoft/Florence-2-large-ft/full-plus-tiles"

    def __init__(
        self,
        grounder: FlorenceGrounder,
        *,
        overlap_ratio: float = 0.2,
        dedup_iou: float = 0.85,
    ) -> None:
        self.full_predictor = FlorenceExternalPredictor(grounder)
        self.overlap_ratio = float(overlap_ratio)
        self.dedup_iou = float(dedup_iou)

    @staticmethod
    def _clip_global_candidate(
        candidate: ModelCandidate,
        *,
        x_offset: int,
        y_offset: int,
        width: int,
        height: int,
        source: str,
    ) -> ModelCandidate | None:
        x1, y1, x2, y2 = candidate.pixel_bbox
        clipped = [
            max(0.0, min(float(width), x1 + x_offset)),
            max(0.0, min(float(height), y1 + y_offset)),
            max(0.0, min(float(width), x2 + x_offset)),
            max(0.0, min(float(height), y2 + y_offset)),
        ]
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            return None
        return ModelCandidate(
            pixel_bbox=clipped,
            label=candidate.label,
            score=None,
            source=source,
        )

    def _deduplicate(
        self,
        candidates: list[ModelCandidate],
    ) -> list[ModelCandidate]:
        retained: list[ModelCandidate] = []
        for candidate in candidates:
            duplicate = any(
                normalize_candidate_label(candidate.label)
                == normalize_candidate_label(existing.label)
                and intersection_over_union(
                    candidate.pixel_bbox,
                    existing.pixel_bbox,
                )
                >= self.dedup_iou
                for existing in retained
            )
            if not duplicate:
                retained.append(candidate)
        return retained

    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        raw_outputs: list[dict[str, Any]] = []
        candidates: list[ModelCandidate] = []
        full = self.full_predictor.predict(image=image, query=query)
        raw_outputs.append({"source": "full", "output": full.raw_output})
        candidates.extend(full.candidates)

        for index, window in enumerate(
            tile_windows(
                width=image.width,
                height=image.height,
                overlap_ratio=self.overlap_ratio,
            )
        ):
            tile = image.crop((window.x1, window.y1, window.x2, window.y2))
            prediction = self.full_predictor.predict(image=tile, query=query)
            source = f"tile_{index}"
            raw_outputs.append(
                {
                    "source": source,
                    "window": window.as_box(),
                    "output": prediction.raw_output,
                }
            )
            for candidate in prediction.candidates:
                mapped = self._clip_global_candidate(
                    candidate,
                    x_offset=window.x1,
                    y_offset=window.y1,
                    width=image.width,
                    height=image.height,
                    source=source,
                )
                if mapped is not None:
                    candidates.append(mapped)
        return ExternalPrediction(
            raw_output=raw_outputs,
            candidates=self._deduplicate(candidates),
        )


def extract_grounding_dino_candidates(
    result: dict[str, Any],
    *,
    max_candidates: int,
) -> list[ModelCandidate]:
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("text_labels", result.get("labels", []))
    candidates: list[ModelCandidate] = []
    for index in range(min(len(boxes), len(scores))):
        score = float(scores[index])
        if not math.isfinite(score):
            raise ValueError("GroundingDINO score must be finite")
        box = [float(value) for value in boxes[index]]
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError("GroundingDINO bbox must contain four finite values")
        label = str(labels[index]) if index < len(labels) else ""
        candidates.append(
            ModelCandidate(
                pixel_bbox=box,
                label=label,
                score=score,
                source="groundingdino",
            )
        )
    candidates.sort(
        key=lambda candidate: (
            candidate.score if candidate.score is not None else float("-inf")
        ),
        reverse=True,
    )
    return candidates[:max_candidates]


class GroundingDinoExternalPredictor:
    model_name = "IDEA-Research/grounding-dino-tiny"

    def __init__(
        self,
        *,
        model_path: Path | str,
        model_class: type[Any] | None = None,
        model_load_kwargs: Mapping[str, Any] | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        max_candidates: int = 20,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.device = torch.device(device)
        self.dtype = dtype
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.max_candidates = int(max_candidates)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        model_loader = model_class or AutoModelForZeroShotObjectDetection
        extra_model_load_kwargs = dict(model_load_kwargs or {})
        self.model = model_loader.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=self.dtype,
            **extra_model_load_kwargs,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, *, image: Image.Image, query: str) -> ExternalPrediction:
        normalized_query = normalize_grounding_query(query)
        inputs = self.processor(
            images=image,
            text=normalized_query,
            return_tensors="pt",
        )
        device_inputs: dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if key == "pixel_values":
                device_inputs[key] = value.to(self.device, dtype=self.dtype)
            else:
                device_inputs[key] = value.to(self.device)
        outputs = self.model(**device_inputs)
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=device_inputs.get("input_ids"),
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        candidates = extract_grounding_dino_candidates(
            processed,
            max_candidates=self.max_candidates,
        )
        return ExternalPrediction(
            raw_output={
                "normalized_query": normalized_query,
                "candidate_count": len(candidates),
            },
            candidates=candidates,
        )
