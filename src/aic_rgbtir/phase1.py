from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from .modeling import RGBTTIRWarmupFeatures


@dataclass(frozen=True)
class QwenVisionAssetPlan:
    model_id: str
    revision: str
    config_file: str
    preprocessor_file: str
    index_file: str
    vision_shards: tuple[str, ...]
    visual_tensor_count: int
    total_model_bytes: int | None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["vision_shards"] = list(self.vision_shards)
        return payload


def read_qwen_vision_asset_plan(
    *,
    model_id: str,
    revision: str,
    cache_dir: Path | str | None = None,
    local_files_only: bool = False,
) -> QwenVisionAssetPlan:
    from huggingface_hub import hf_hub_download

    common = {
        "repo_id": model_id,
        "revision": revision,
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "local_files_only": local_files_only,
    }
    config_path = hf_hub_download(filename="config.json", **common)
    preprocessor_path = hf_hub_download(filename="preprocessor_config.json", **common)
    index_path = hf_hub_download(filename="model.safetensors.index.json", **common)
    del config_path, preprocessor_path
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    visual_keys = [key for key in weight_map if key.startswith("model.visual.")]
    if not visual_keys:
        raise ValueError("checkpoint index contains no model.visual tensors")
    shards = tuple(sorted({str(weight_map[key]) for key in visual_keys}))
    return QwenVisionAssetPlan(
        model_id=model_id,
        revision=revision,
        config_file="config.json",
        preprocessor_file="preprocessor_config.json",
        index_file="model.safetensors.index.json",
        vision_shards=shards,
        visual_tensor_count=len(visual_keys),
        total_model_bytes=index.get("metadata", {}).get("total_size"),
    )


def download_qwen_vision_assets(
    plan: QwenVisionAssetPlan,
    *,
    cache_dir: Path | str | None = None,
) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    files = (
        plan.config_file,
        plan.preprocessor_file,
        plan.index_file,
        *plan.vision_shards,
    )
    return {
        name: Path(
            hf_hub_download(
                repo_id=plan.model_id,
                filename=name,
                revision=plan.revision,
                cache_dir=None if cache_dir is None else str(cache_dir),
            )
        )
        for name in files
    }


def load_qwen3vl_vision_only(
    *,
    assets: dict[str, Path],
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load the official Qwen3-VL visual tower without materialising the LLM."""
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = Qwen3VLConfig.from_json_file(str(assets["config.json"]))
    # Instantiate only the visual tower on CPU.  A meta-device construction
    # followed by assign=True materialises parameters from safetensors, but
    # leaves non-persistent buffers (for example rotary-frequency buffers) on
    # meta; calling .to() then fails on PyTorch 2.6 and to_empty() would erase
    # the loaded parameters.  The vision-only CPU module is small enough for
    # the Phase 1 host and preserves the official buffer initialisation.
    visual = Qwen3VLVisionModel(config.vision_config)
    state: dict[str, torch.Tensor] = {}
    prefix = "model.visual."
    for name, path in assets.items():
        if not name.endswith(".safetensors"):
            continue
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key.removeprefix(prefix)] = handle.get_tensor(key)
    missing, unexpected = visual.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"visual-only checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    visual.requires_grad_(False)
    visual.eval()
    return visual.to(device=device, dtype=dtype)


def build_roi_patch_weights(
    *,
    normalized_bbox: Sequence[float],
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
    expansion: float = 0.10,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    grid = image_grid_thw.reshape(-1, 3)
    if grid.shape[0] != 1 or int(grid[0, 0]) != 1:
        raise ValueError("Phase 1 warmup currently supports one still image per step")
    height, width = int(grid[0, 1]), int(grid[0, 2])
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError("visual grid is not divisible by spatial merge size")
    x1, y1, x2, y2 = (float(value) for value in normalized_bbox)
    box_width, box_height = x2 - x1, y2 - y1
    x1 = max(0.0, x1 - expansion * box_width)
    y1 = max(0.0, y1 - expansion * box_height)
    x2 = min(1.0, x2 + expansion * box_width)
    y2 = min(1.0, y2 + expansion * box_height)
    ys = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    row_major = ((ys[:, None] >= y1) & (ys[:, None] <= y2)) & (
        (xs[None, :] >= x1) & (xs[None, :] <= x2)
    )
    if not row_major.any():
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        row = min(height - 1, max(0, int(center_y * height)))
        col = min(width - 1, max(0, int(center_x * width)))
        row_major[row, col] = True
    merge = spatial_merge_size
    ordered = (
        row_major.reshape(height // merge, merge, width // merge, merge)
        .permute(0, 2, 1, 3)
        .reshape(-1)
        .to(dtype=torch.float32, device=device)
    )
    return ordered


@dataclass(frozen=True)
class TIRWarmupLossOutput:
    loss: torch.Tensor
    foreground_alignment: torch.Tensor
    hard_negative_margin: torch.Tensor
    layer_cosines: tuple[float, ...]
    roi_token_count: int


class BBoxAwareTIRAlignmentLoss(nn.Module):
    """Align target-region TIR features to frozen RGB features without Query claims."""

    def __init__(self, *, hard_negative_weight: float = 0.25, margin: float = 0.15) -> None:
        super().__init__()
        self.hard_negative_weight = float(hard_negative_weight)
        self.margin = float(margin)

    @staticmethod
    def _weighted_mean(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(device=features.device, dtype=torch.float32)
        denominator = weights.sum().clamp_min(1.0)
        return (features.float() * weights[:, None]).sum(dim=0) / denominator

    def forward(
        self,
        features: RGBTTIRWarmupFeatures,
        *,
        roi_patch_weights: torch.Tensor,
        ir_valid_mask: torch.Tensor,
    ) -> TIRWarmupLossOutput:
        roi = roi_patch_weights.to(dtype=torch.float32)
        valid = ir_valid_mask.to(dtype=torch.float32)
        foreground = roi * valid
        if foreground.sum() < 1:
            raise ValueError("ROI contains no valid TIR patch after black-border masking")
        background = (1.0 - roi) * valid
        alignment_losses: list[torch.Tensor] = []
        margin_losses: list[torch.Tensor] = []
        cosines: list[float] = []
        for rgb_layer, tir_layer in zip(features.rgb_premerger, features.tir_premerger):
            rgb_fg = self._weighted_mean(rgb_layer, foreground)
            tir_fg = self._weighted_mean(tir_layer, foreground)
            positive = F.cosine_similarity(rgb_fg[None], tir_fg[None]).squeeze(0)
            alignment_losses.append(1.0 - positive)
            cosines.append(float(positive.detach().cpu()))
            if background.sum() >= 1:
                rgb_bg = self._weighted_mean(rgb_layer, background)
                negative = F.cosine_similarity(rgb_bg[None], tir_fg[None]).squeeze(0)
                margin_losses.append(F.relu(self.margin - positive + negative))
        alignment = torch.stack(alignment_losses).mean()
        margin_loss = (
            torch.stack(margin_losses).mean()
            if margin_losses
            else alignment.new_zeros(())
        )
        loss = alignment + self.hard_negative_weight * margin_loss
        return TIRWarmupLossOutput(
            loss=loss,
            foreground_alignment=alignment.detach(),
            hard_negative_margin=margin_loss.detach(),
            layer_cosines=tuple(cosines),
            roi_token_count=int(foreground.sum().item()),
        )


def parameter_sha256(parameters: Iterable[tuple[str, nn.Parameter]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(parameters, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        raw = parameter.detach().cpu().contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest().upper()


def trainable_parameter_summary(module: nn.Module) -> dict[str, Any]:
    rows = [
        (name, parameter.numel())
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "tensor_count": len(rows),
        "parameter_count": sum(count for _, count in rows),
        "names": [name for name, _ in rows],
    }


def freeze_for_tir_adapter_warmup(module: nn.Module) -> dict[str, Any]:
    """Keep only TIR LoRA tensors trainable during Phase 1.

    The visual base, RGB path, residual fusion and any future bbox/language
    components stay frozen. This makes Phase 1 an IR-domain adaptation stage,
    not a fusion or grounding fine-tune.
    """
    trainable_names: list[str] = []
    for name, parameter in module.named_parameters():
        trainable = ".paths.tir." in name
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError("no TIR LoRA parameters found after warmup freeze")
    forbidden = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and ".paths.tir." not in name
    ]
    if forbidden:
        raise RuntimeError(f"non-TIR parameters remain trainable: {forbidden}")
    return trainable_parameter_summary(module)


def freeze_for_rgbtir_inference(module: nn.Module) -> nn.Module:
    """Close the inference contract after training without changing weights."""
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def tir_adapter_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
        if ".paths.tir." in name
    }
    if not state:
        raise ValueError("module contains no TIR adapter state")
    return state


def load_tir_adapter_state_dict(
    module: nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    expected = {
        name for name in module.state_dict() if ".paths.tir." in name
    }
    supplied = set(state)
    if supplied != expected:
        raise ValueError(
            "TIR adapter checkpoint mismatch: "
            f"missing={sorted(expected - supplied)}, unexpected={sorted(supplied - expected)}"
        )
    current = module.state_dict()
    current.update({name: value for name, value in state.items()})
    module.load_state_dict(current, strict=True)
