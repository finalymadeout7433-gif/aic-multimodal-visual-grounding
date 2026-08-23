from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from aic_rgbtir.modeling import RGBTTIRWarmupFeatures, install_asymmetric_lora
from aic_rgbtir.phase1 import (
    BBoxAwareTIRAlignmentLoss,
    build_roi_patch_weights,
    freeze_for_tir_adapter_warmup,
    load_tir_adapter_state_dict,
    parameter_sha256,
    read_qwen_vision_asset_plan,
    tir_adapter_state_dict,
)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(8, 24, bias=False)
        self.proj = nn.Linear(8, 8, bias=False)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()


class _TinyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])


def test_asset_plan_downloads_only_the_visual_shard(monkeypatch, tmp_path: Path) -> None:
    files = {
        "config.json": {},
        "preprocessor_config.json": {},
        "model.safetensors.index.json": {
            "metadata": {"total_size": 1234},
            "weight_map": {
                "model.visual.a": "model-00004-of-00004.safetensors",
                "model.visual.b": "model-00004-of-00004.safetensors",
                "model.language_model.a": "model-00001-of-00004.safetensors",
            },
        },
    }
    for name, payload in files.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    def fake_download(*, filename: str, **_kwargs):
        return str(tmp_path / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    plan = read_qwen_vision_asset_plan(
        model_id="Qwen/test", revision="fixed", local_files_only=True
    )
    assert plan.visual_tensor_count == 2
    assert plan.vision_shards == ("model-00004-of-00004.safetensors",)
    assert plan.total_model_bytes == 1234


def test_roi_patch_weights_follow_qwen_merge_order_and_never_empty() -> None:
    grid = torch.tensor([[1, 4, 6]])
    weights = build_roi_patch_weights(
        normalized_bbox=(0.0, 0.0, 0.20, 0.20),
        image_grid_thw=grid,
        spatial_merge_size=2,
        expansion=0.0,
    )
    assert weights.numel() == 24
    assert weights.sum() >= 1
    tiny = build_roi_patch_weights(
        normalized_bbox=(0.499, 0.499, 0.501, 0.501),
        image_grid_thw=grid,
        spatial_merge_size=2,
        expansion=0.0,
    )
    assert tiny.sum() == 1


def test_alignment_loss_prefers_matching_roi_features() -> None:
    rgb = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        requires_grad=False,
    )
    matching = rgb.clone().requires_grad_(True)
    conflicting = torch.flip(rgb, dims=[1]).requires_grad_(True)
    roi = torch.tensor([1.0, 1.0, 0.0, 0.0])
    valid = torch.ones(4)
    criterion = BBoxAwareTIRAlignmentLoss(hard_negative_weight=0.0)
    matching_output = criterion(
        RGBTTIRWarmupFeatures(("final",), (rgb,), (matching,), torch.tensor([[1, 2, 2]])),
        roi_patch_weights=roi,
        ir_valid_mask=valid,
    )
    conflicting_output = criterion(
        RGBTTIRWarmupFeatures(("final",), (rgb,), (conflicting,), torch.tensor([[1, 2, 2]])),
        roi_patch_weights=roi,
        ir_valid_mask=valid,
    )
    assert matching_output.loss < conflicting_output.loss


def test_warmup_freezes_everything_except_tir_lora_and_roundtrips_state() -> None:
    visual = _TinyVisual()
    install_asymmetric_lora(
        visual,
        target_suffixes=("attn.qkv", "attn.proj"),
        rgb_rank=0,
        tir_rank=4,
        alpha=4,
    )
    summary = freeze_for_tir_adapter_warmup(visual)
    assert summary["parameter_count"] > 0
    assert all(".paths.tir." in name for name in summary["names"])
    base_before = parameter_sha256(
        (name, parameter)
        for name, parameter in visual.named_parameters()
        if ".paths.tir." not in name
    )
    state = tir_adapter_state_dict(visual)
    for parameter in visual.parameters():
        if parameter.requires_grad:
            parameter.data.add_(1)
    load_tir_adapter_state_dict(visual, state)
    restored = tir_adapter_state_dict(visual)
    assert all(torch.equal(state[name], restored[name]) for name in state)
    base_after = parameter_sha256(
        (name, parameter)
        for name, parameter in visual.named_parameters()
        if ".paths.tir." not in name
    )
    assert base_before == base_after


def test_parameter_hash_supports_bfloat16() -> None:
    parameter = nn.Parameter(torch.arange(8, dtype=torch.bfloat16))
    first = parameter_sha256([("value", parameter)])
    second = parameter_sha256([("value", parameter)])
    assert first == second
    assert len(first) == 64
