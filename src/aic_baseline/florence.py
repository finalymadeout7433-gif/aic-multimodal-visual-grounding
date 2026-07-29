from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"


class FlorenceOutputError(ValueError):
    """Florence 输出中没有可用候选框。"""


class FlorenceConfigurationError(ValueError):
    """Florence baseline 配置不受支持。"""


@dataclass(frozen=True)
class FlorenceCandidate:
    pixel_bbox: list[float]
    label: str


@dataclass(frozen=True)
class FlorencePrediction:
    raw_output: str
    candidates: list[FlorenceCandidate]
    selected: FlorenceCandidate


def extract_candidates(parsed: dict[str, Any]) -> list[FlorenceCandidate]:
    result = parsed.get(TASK_PROMPT)
    if not isinstance(result, dict):
        raise FlorenceOutputError("Florence 输出缺少 phrase grounding 结果")
    boxes = result.get("bboxes")
    labels = result.get("labels", [])
    if not isinstance(boxes, list) or not boxes:
        raise FlorenceOutputError("Florence 没有返回候选框")

    candidates: list[FlorenceCandidate] = []
    for index, box in enumerate(boxes):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise FlorenceOutputError(f"第 {index + 1} 个候选框格式错误")
        label = str(labels[index]) if index < len(labels) else ""
        try:
            pixel_bbox = [float(value) for value in box]
        except (TypeError, ValueError) as exc:
            raise FlorenceOutputError(
                f"第 {index + 1} 个候选框坐标不是数字"
            ) from exc
        candidates.append(FlorenceCandidate(pixel_bbox=pixel_bbox, label=label))
    return candidates


def validate_selection_strategy(strategy: str) -> None:
    if strategy != "first":
        raise FlorenceConfigurationError(
            f"baseline v0 不支持候选选择策略: {strategy}"
        )


def select_candidate(
    candidates: list[FlorenceCandidate], *, strategy: str
) -> FlorenceCandidate:
    if not candidates:
        raise FlorenceOutputError("候选框列表为空")
    validate_selection_strategy(strategy)
    return candidates[0]


class FlorenceGrounder:
    """本地 Florence-2 的确定性 RGB phrase grounding 封装。"""

    def __init__(
        self,
        *,
        model_path: Path | str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        selection_strategy: str = "first",
    ):
        self.model_path = Path(model_path).resolve()
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.selection_strategy = selection_strategy
        validate_selection_strategy(self.selection_strategy)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=self.dtype,
            attn_implementation="eager",
        ).to(self.device)
        self.model.generation_config.early_stopping = False
        self.model.config.early_stopping = False
        self.model.eval()

    @torch.inference_mode()
    def predict(self, *, image: Image.Image, query: str) -> FlorencePrediction:
        prompt = TASK_PROMPT + query
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        device_inputs: dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if key == "pixel_values":
                device_inputs[key] = value.to(self.device, dtype=self.dtype)
            else:
                device_inputs[key] = value.to(self.device)

        generated_ids = self.model.generate(
            **device_inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            do_sample=False,
            use_cache=False,
        )
        raw_output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        parsed = self.processor.post_process_generation(
            raw_output,
            task=TASK_PROMPT,
            image_size=image.size,
        )
        candidates = extract_candidates(parsed)
        selected = select_candidate(candidates, strategy=self.selection_strategy)
        return FlorencePrediction(
            raw_output=raw_output,
            candidates=candidates,
            selected=selected,
        )
