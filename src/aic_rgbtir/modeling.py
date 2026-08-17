from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch import nn


class _LoRAPath(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.rank = int(rank)
        self.scale = float(alpha) / max(1, rank)
        self.a = nn.Linear(in_features, rank, bias=False)
        self.b = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
        nn.init.zeros_(self.b.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(value)) * self.scale


class AsymmetricLoRALinear(nn.Module):
    """One frozen linear base with isolated RGB and TIR low-rank paths."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rgb_rank: int = 0,
        tir_rank: int = 48,
        alpha: float = 16.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
        self.paths = nn.ModuleDict()
        if rgb_rank > 0:
            self.paths["rgb"] = _LoRAPath(
                base.in_features, base.out_features, rgb_rank, alpha
            )
        if tir_rank > 0:
            self.paths["tir"] = _LoRAPath(
                base.in_features, base.out_features, tir_rank, alpha
            )
        self._active_modality = "rgb"

    @property
    def active_modality(self) -> str:
        return self._active_modality

    def set_modality(self, modality: str) -> None:
        if modality not in {"rgb", "tir", "base"}:
            raise ValueError(f"unsupported modality {modality!r}")
        self._active_modality = modality

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.base(value)
        path = self.paths[self._active_modality] if self._active_modality in self.paths else None
        return result if path is None else result + path(value)


class ModalityAdapterRouter:
    def __init__(self, modules: Iterable[AsymmetricLoRALinear]) -> None:
        self.modules = tuple(modules)

    @contextmanager
    def use(self, modality: str) -> Iterator[None]:
        previous = [module.active_modality for module in self.modules]
        try:
            for module in self.modules:
                module.set_modality(modality)
            yield
        finally:
            for module, old in zip(self.modules, previous):
                module.set_modality(old)


def _resolve_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def install_asymmetric_lora(
    visual_encoder: nn.Module,
    *,
    target_suffixes: Sequence[str] = ("attn.qkv", "attn.proj"),
    rgb_rank: int = 0,
    tir_rank: int = 48,
    alpha: float = 16.0,
) -> ModalityAdapterRouter:
    names = [
        name
        for name, module in visual_encoder.named_modules()
        if isinstance(module, nn.Linear)
        and any(name.endswith(suffix) for suffix in target_suffixes)
    ]
    installed: list[AsymmetricLoRALinear] = []
    for name in names:
        parent, leaf = _resolve_parent(visual_encoder, name)
        base = getattr(parent, leaf)
        wrapped = AsymmetricLoRALinear(
            base,
            rgb_rank=rgb_rank,
            tir_rank=tir_rank,
            alpha=alpha,
        )
        setattr(parent, leaf, wrapped)
        installed.append(wrapped)
    if not installed:
        raise ValueError(f"no visual linear modules matched {tuple(target_suffixes)!r}")
    return ModalityAdapterRouter(installed)


class ZeroInitResidualFusion(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.project = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fusion_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        rgb_hidden: torch.Tensor,
        tir_hidden: torch.Tensor,
        valid_mask: torch.Tensor,
        grid_thw: torch.Tensor,
        query_state: torch.Tensor | None = None,
        quality_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del grid_thw, query_state, quality_state
        if rgb_hidden.shape != tir_hidden.shape:
            raise ValueError(
                f"RGB/TIR hidden shapes differ: {rgb_hidden.shape} vs {tir_hidden.shape}"
            )
        mask = valid_mask.to(device=rgb_hidden.device, dtype=rgb_hidden.dtype)
        if mask.numel() != rgb_hidden.shape[0]:
            raise ValueError(
                f"valid mask has {mask.numel()} tokens, expected {rgb_hidden.shape[0]}"
            )
        while mask.ndim < rgb_hidden.ndim:
            mask = mask.unsqueeze(-1)
        return rgb_hidden + torch.tanh(self.fusion_scale) * mask * self.project(tir_hidden)


class NormalizedBBoxHead(nn.Module):
    """Produce a strictly valid normalized xyxy box without text parsing."""

    def __init__(self, hidden_size: int, intermediate_size: int | None = None) -> None:
        super().__init__()
        middle = intermediate_size or max(32, hidden_size // 2)
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, middle),
            nn.GELU(),
            nn.Linear(middle, 4),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.network(hidden_state))
        left = raw[..., 0] * (1.0 - 1e-4)
        top = raw[..., 1] * (1.0 - 1e-4)
        width_fraction = 1e-4 + (1.0 - 1e-4) * raw[..., 2]
        height_fraction = 1e-4 + (1.0 - 1e-4) * raw[..., 3]
        right = left + (1.0 - left) * width_fraction
        bottom = top + (1.0 - top) * height_fraction
        return torch.stack((left, top, right, bottom), dim=-1)


@dataclass(frozen=True)
class RGBTVisionOutput:
    final_hidden: torch.Tensor
    deepstack_hidden: tuple[torch.Tensor, ...]
    rgb_native_final: torch.Tensor
    rgb_native_deepstack: tuple[torch.Tensor, ...]
    image_grid_thw: torch.Tensor
    used_tir: bool


@dataclass(frozen=True)
class RGBTTIRWarmupFeatures:
    layer_names: tuple[str, ...]
    rgb_premerger: tuple[torch.Tensor, ...]
    tir_premerger: tuple[torch.Tensor, ...]
    image_grid_thw: torch.Tensor


@dataclass(frozen=True)
class RGBTTIRValidationFeatures:
    """Frozen RGB teacher, unadapted TIR and adapted TIR feature triplet."""

    layer_names: tuple[str, ...]
    rgb_premerger: tuple[torch.Tensor, ...]
    tir_base_premerger: tuple[torch.Tensor, ...]
    tir_adapted_premerger: tuple[torch.Tensor, ...]
    image_grid_thw: torch.Tensor


@dataclass(frozen=True)
class RGBTTIRRepairFeatures:
    """One modality's pre-merger features at the Phase 1.6 training seam."""

    layer_names: tuple[str, ...]
    premerger: tuple[torch.Tensor, ...]
    image_grid_thw: torch.Tensor


def validate_qwen3vl_visual_contract(
    visual_encoder: nn.Module,
    *,
    expected_hidden_size: int | None = 1152,
    expected_indexes: Sequence[int] = (8, 16, 24),
) -> dict[str, Any]:
    required = ("blocks", "merger", "deepstack_merger_list", "deepstack_visual_indexes")
    missing = [name for name in required if not hasattr(visual_encoder, name)]
    if missing:
        raise ValueError(f"visual encoder is missing required attributes: {missing}")
    indexes = tuple(int(value) for value in visual_encoder.deepstack_visual_indexes)
    if indexes != tuple(expected_indexes):
        raise ValueError(f"unexpected DeepStack indexes: {indexes}")
    if len(visual_encoder.blocks) <= max(indexes):
        raise ValueError("visual encoder does not contain all DeepStack layers")
    config = getattr(visual_encoder, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if expected_hidden_size is not None and hidden_size not in {None, expected_hidden_size}:
        raise ValueError(f"unexpected visual hidden size: {hidden_size}")
    return {
        "depth": len(visual_encoder.blocks),
        "hidden_size": hidden_size,
        "deepstack_visual_indexes": list(indexes),
        "spatial_merge_size": int(getattr(visual_encoder, "spatial_merge_size", 0)),
        "patch_size": int(getattr(visual_encoder, "patch_size", 0)),
    }


class Qwen3VLRGBTAdapter(nn.Module):
    """Capture and fuse Qwen pre-merger features without patching Transformers."""

    def __init__(
        self,
        visual_encoder: nn.Module,
        *,
        hidden_size: int,
        adapter_router: ModalityAdapterRouter | None = None,
        enforce_qwen_contract: bool = True,
    ) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.adapter_router = adapter_router
        self.deepstack_indexes = tuple(
            int(value) for value in visual_encoder.deepstack_visual_indexes
        )
        if enforce_qwen_contract:
            validate_qwen3vl_visual_contract(
                visual_encoder, expected_hidden_size=hidden_size
            )
        keys = [str(index) for index in self.deepstack_indexes] + ["final"]
        self.fusions = nn.ModuleDict(
            {key: ZeroInitResidualFusion(hidden_size) for key in keys}
        )

    @contextmanager
    def _modality(self, modality: str) -> Iterator[None]:
        if self.adapter_router is None:
            yield
        else:
            with self.adapter_router.use(modality):
                yield

    def _run_capture(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, modality: str
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], Mapping[int, torch.Tensor]]:
        captured: dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(index: int):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                captured[index] = output[0] if isinstance(output, tuple) else output

            return hook

        capture_indexes = set(self.deepstack_indexes) | {len(self.visual_encoder.blocks) - 1}
        for index in sorted(capture_indexes):
            hooks.append(
                self.visual_encoder.blocks[index].register_forward_hook(make_hook(index))
            )
        try:
            with self._modality(modality):
                native_final, native_deep = self.visual_encoder(
                    pixel_values, grid_thw=grid_thw
                )
        finally:
            for handle in hooks:
                handle.remove()
        if set(captured) != capture_indexes:
            raise RuntimeError(
                f"failed to capture visual layers: expected={capture_indexes}, got={set(captured)}"
            )
        return native_final, tuple(native_deep), captured

    def encode_visual_pair(
        self,
        *,
        rgb_pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        tir_pixel_values: torch.Tensor | None = None,
        ir_patch_valid_mask: torch.Tensor | None = None,
        ir_usable: bool = True,
    ) -> RGBTVisionOutput:
        rgb_final, rgb_deep, rgb_captured = self._run_capture(
            rgb_pixel_values, image_grid_thw, "rgb"
        )
        if tir_pixel_values is None or not ir_usable:
            return RGBTVisionOutput(
                final_hidden=rgb_final,
                deepstack_hidden=rgb_deep,
                rgb_native_final=rgb_final,
                rgb_native_deepstack=rgb_deep,
                image_grid_thw=image_grid_thw,
                used_tir=False,
            )
        if ir_patch_valid_mask is None:
            raise ValueError("ir_patch_valid_mask is required when TIR is used")
        if tir_pixel_values.shape != rgb_pixel_values.shape:
            raise ValueError("RGB/TIR flattened patch tensors must have equal shape")
        _, _, tir_captured = self._run_capture(
            tir_pixel_values, image_grid_thw, "tir"
        )
        fused: dict[int, torch.Tensor] = {}
        for index in self.deepstack_indexes:
            fused[index] = self.fusions[str(index)](
                rgb_captured[index],
                tir_captured[index],
                ir_patch_valid_mask,
                image_grid_thw,
            )
        final_index = len(self.visual_encoder.blocks) - 1
        fused[final_index] = self.fusions["final"](
            rgb_captured[final_index],
            tir_captured[final_index],
            ir_patch_valid_mask,
            image_grid_thw,
        )
        deepstack = tuple(
            merger(fused[index])
            for index, merger in zip(
                self.deepstack_indexes, self.visual_encoder.deepstack_merger_list
            )
        )
        final = self.visual_encoder.merger(fused[final_index])
        return RGBTVisionOutput(
            final_hidden=final,
            deepstack_hidden=deepstack,
            rgb_native_final=rgb_final,
            rgb_native_deepstack=rgb_deep,
            image_grid_thw=image_grid_thw,
            used_tir=True,
        )

    def encode_for_tir_warmup(
        self,
        *,
        rgb_pixel_values: torch.Tensor,
        tir_pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> RGBTTIRWarmupFeatures:
        """Return paired pre-merger features for ROI-level modality alignment.

        The RGB path is a frozen teacher. Callers should keep the visual base frozen;
        gradients are intended to reach only the active TIR adapter path.
        """
        if rgb_pixel_values.shape != tir_pixel_values.shape:
            raise ValueError("RGB/TIR flattened patch tensors must have equal shape")
        with torch.no_grad():
            _, _, rgb_captured = self._run_capture(
                rgb_pixel_values, image_grid_thw, "rgb"
            )
        _, _, tir_captured = self._run_capture(
            tir_pixel_values, image_grid_thw, "tir"
        )
        final_index = len(self.visual_encoder.blocks) - 1
        indexes = self.deepstack_indexes + (final_index,)
        return RGBTTIRWarmupFeatures(
            layer_names=tuple(
                [str(index) for index in self.deepstack_indexes] + ["final"]
            ),
            rgb_premerger=tuple(rgb_captured[index].detach() for index in indexes),
            tir_premerger=tuple(tir_captured[index] for index in indexes),
            image_grid_thw=image_grid_thw,
        )

    @torch.inference_mode()
    def encode_rgb_teacher_features(
        self,
        *,
        rgb_pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> RGBTTIRRepairFeatures:
        """Encode the frozen RGB teacher once for a Phase 1.6 feature bank."""

        _, _, captured = self._run_capture(
            rgb_pixel_values, image_grid_thw, "rgb"
        )
        final_index = len(self.visual_encoder.blocks) - 1
        indexes = self.deepstack_indexes + (final_index,)
        return RGBTTIRRepairFeatures(
            layer_names=tuple(
                [str(index) for index in self.deepstack_indexes] + ["final"]
            ),
            premerger=tuple(captured[index].detach() for index in indexes),
            image_grid_thw=image_grid_thw.detach(),
        )

    def encode_tir_repair_features(
        self,
        *,
        tir_pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> RGBTTIRRepairFeatures:
        """Encode TIR while preserving gradients only through the active TIR path."""

        _, _, captured = self._run_capture(
            tir_pixel_values, image_grid_thw, "tir"
        )
        final_index = len(self.visual_encoder.blocks) - 1
        indexes = self.deepstack_indexes + (final_index,)
        return RGBTTIRRepairFeatures(
            layer_names=tuple(
                [str(index) for index in self.deepstack_indexes] + ["final"]
            ),
            premerger=tuple(captured[index] for index in indexes),
            image_grid_thw=image_grid_thw,
        )

    @torch.inference_mode()
    def encode_base_tir_features(
        self,
        *,
        tir_pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> RGBTTIRRepairFeatures:
        """Encode unadapted TIR once for Phase 1.6 gate baselines."""

        _, _, captured = self._run_capture(
            tir_pixel_values, image_grid_thw, "base"
        )
        final_index = len(self.visual_encoder.blocks) - 1
        indexes = self.deepstack_indexes + (final_index,)
        return RGBTTIRRepairFeatures(
            layer_names=tuple(
                [str(index) for index in self.deepstack_indexes] + ["final"]
            ),
            premerger=tuple(captured[index].detach() for index in indexes),
            image_grid_thw=image_grid_thw.detach(),
        )

    @torch.inference_mode()
    def encode_tir_validation_triplet(
        self,
        *,
        rgb_pixel_values: torch.Tensor,
        tir_pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        ir_usable: bool = True,
    ) -> RGBTTIRValidationFeatures:
        """Extract the three frozen feature paths needed by Phase 1.5.

        RGB is evaluated exactly once.  ``base`` bypasses every LoRA path and
        ``tir`` activates only the loaded TIR adapter.  No fusion module, query
        token or bbox head participates in this validation seam.
        """
        if ir_usable and rgb_pixel_values.shape != tir_pixel_values.shape:
            raise ValueError("RGB/TIR flattened patch tensors must have equal shape")
        _, _, rgb_captured = self._run_capture(
            rgb_pixel_values, image_grid_thw, "rgb"
        )
        final_index = len(self.visual_encoder.blocks) - 1
        indexes = self.deepstack_indexes + (final_index,)
        layer_names = tuple(
            [str(index) for index in self.deepstack_indexes] + ["final"]
        )
        rgb_layers = tuple(rgb_captured[index].detach() for index in indexes)
        if not ir_usable:
            return RGBTTIRValidationFeatures(
                layer_names=layer_names,
                rgb_premerger=rgb_layers,
                tir_base_premerger=rgb_layers,
                tir_adapted_premerger=rgb_layers,
                image_grid_thw=image_grid_thw.detach(),
            )
        _, _, tir_base_captured = self._run_capture(
            tir_pixel_values, image_grid_thw, "base"
        )
        _, _, tir_adapted_captured = self._run_capture(
            tir_pixel_values, image_grid_thw, "tir"
        )
        return RGBTTIRValidationFeatures(
            layer_names=layer_names,
            rgb_premerger=rgb_layers,
            tir_base_premerger=tuple(
                tir_base_captured[index].detach() for index in indexes
            ),
            tir_adapted_premerger=tuple(
                tir_adapted_captured[index].detach() for index in indexes
            ),
            image_grid_thw=image_grid_thw.detach(),
        )
