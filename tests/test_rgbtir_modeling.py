from __future__ import annotations

import inspect

import torch
from torch import nn

from aic_rgbtir.modeling import (
    AsymmetricLoRALinear,
    NormalizedBBoxHead,
    ZeroInitResidualFusion,
)
from aic_rgbtir.validation import validate_zero_gate_equivalence


def test_zero_initialized_fusion_is_exact_rgb_identity() -> None:
    fusion = ZeroInitResidualFusion(hidden_size=8)
    rgb = torch.randn(12, 8)
    tir = torch.randn(12, 8)
    mask = torch.rand(12)
    grid = torch.tensor([[1, 3, 4]])
    output = fusion(rgb, tir, mask, grid)
    assert torch.equal(output, rgb)


def test_qwen_bridge_zero_gate_and_missing_tir_are_rgb_equivalent() -> None:
    summary = validate_zero_gate_equivalence(device="cpu")
    assert summary["passed"]
    assert summary["used_tir"]
    assert not summary["rgb_only_used_tir"]
    assert summary["final_max_abs_diff"] <= 1e-6
    assert summary["deepstack_max_abs_diff"] <= 1e-6


def test_qwen_bridge_zero_gate_is_bf16_equivalent() -> None:
    summary = validate_zero_gate_equivalence(device="cpu", dtype=torch.bfloat16)
    assert summary["passed"]
    assert summary["final_max_abs_diff"] <= 1e-3
    assert summary["deepstack_max_abs_diff"] <= 1e-3


def test_tir_backward_does_not_update_rgb_adapter_or_frozen_base() -> None:
    wrapped = AsymmetricLoRALinear(
        nn.Linear(6, 5), rgb_rank=2, tir_rank=3, alpha=4, freeze_base=True
    )
    wrapped.set_modality("tir")
    output = wrapped(torch.randn(7, 6)).sum()
    output.backward()
    assert all(parameter.grad is None for parameter in wrapped.base.parameters())
    assert all(parameter.grad is None for parameter in wrapped.paths["rgb"].parameters())
    assert any(parameter.grad is not None for parameter in wrapped.paths["tir"].parameters())


def test_bbox_head_always_returns_finite_strict_normalized_xyxy() -> None:
    head = NormalizedBBoxHead(hidden_size=16)
    boxes = head(torch.randn(64, 16) * 1000)
    assert torch.isfinite(boxes).all()
    assert (boxes >= 0).all() and (boxes <= 1).all()
    assert (boxes[:, 0] < boxes[:, 2]).all()
    assert (boxes[:, 1] < boxes[:, 3]).all()


def test_phase0_modeling_has_no_second_model_fallback_interface() -> None:
    import aic_rgbtir.modeling as modeling

    source = inspect.getsource(modeling).lower()
    assert "fallback_model" not in source
    assert "second_model" not in source
