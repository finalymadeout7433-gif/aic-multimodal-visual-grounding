from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .data import ManifestBundle, RGBTRecord
from .modeling import Qwen3VLRGBTAdapter
from .processing import inspect_pair_paths, qwen_grid_for_size


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path | str, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0])
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _audit_one_pair(
    *,
    pair_id: str,
    source: str,
    rgb_path: Path,
    tir_path: Path,
    min_pixels: int,
    max_pixels: int,
    black_threshold: int,
) -> dict[str, Any]:
    base = {
        "pair_id": pair_id,
        "source": source,
        "rgb_path": str(rgb_path),
        "tir_path": str(tir_path),
    }
    try:
        inspection = inspect_pair_paths(
            rgb_path, tir_path, black_threshold=black_threshold
        )
        rgb_grid = qwen_grid_for_size(
            inspection.rgb_width,
            inspection.rgb_height,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        tir_grid = qwen_grid_for_size(
            inspection.tir_width,
            inspection.tir_height,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return {
            **base,
            **inspection.as_dict(),
            "rgb_grid_thw": "x".join(map(str, rgb_grid)),
            "tir_grid_thw": "x".join(map(str, tir_grid)),
            "grid_equal": rgb_grid == tir_grid,
            "status": "ok",
            "error": "",
        }
    except Exception as exc:  # keep a complete audit rather than losing later rows
        return {
            **base,
            "rgb_width": "",
            "rgb_height": "",
            "tir_width": "",
            "tir_height": "",
            "same_spatial_dimensions": False,
            "ir_valid_ratio": "",
            "ir_intensity_p1": "",
            "ir_intensity_p99": "",
            "ir_intensity_std": "",
            "ir_low_information": "",
            "ir_usable": False,
            "rgb_grid_thw": "",
            "tir_grid_thw": "",
            "grid_equal": False,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _parallel_audit(
    pairs: Sequence[dict[str, Any]],
    *,
    min_pixels: int,
    max_pixels: int,
    black_threshold: int,
    workers: int,
) -> list[dict[str, Any]]:
    def work(pair: dict[str, Any]) -> dict[str, Any]:
        return _audit_one_pair(
            pair_id=str(pair["pair_id"]),
            source=str(pair["source"]),
            rgb_path=Path(pair["rgb_path"]),
            tir_path=Path(pair["tir_path"]),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            black_threshold=black_threshold,
        )

    ordered = sorted(pairs, key=lambda pair: str(pair["pair_id"]))
    if workers <= 1:
        return [work(pair) for pair in ordered]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(work, ordered))


def audit_rgbt_pairs(
    records: Iterable[RGBTRecord],
    *,
    root: Path | str,
    min_pixels: int,
    max_pixels: int,
    black_threshold: int,
    workers: int = 4,
) -> list[dict[str, Any]]:
    base = Path(root).resolve()
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(
            record.image_pair_key,
            {
                "pair_id": record.image_pair_key,
                "source": record.source_dataset,
                "rgb_path": base / record.rgb_relpath,
                "tir_path": base / record.tir_relpath,
            },
        )
    return _parallel_audit(
        list(unique.values()),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        black_threshold=black_threshold,
        workers=workers,
    )


def audit_aic_pairs(
    *,
    dataset_root: Path | str,
    queries_path: Path | str,
    min_pixels: int,
    max_pixels: int,
    black_threshold: int,
    workers: int = 4,
) -> list[dict[str, Any]]:
    root = Path(dataset_root).resolve()
    with Path(queries_path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("AIC queries JSON must contain an object")
    unique: dict[str, dict[str, Any]] = {}
    for query_id, record in payload.items():
        visible, infrared = record.get("visible"), record.get("infrared")
        if not isinstance(visible, str) or not isinstance(infrared, str):
            raise ValueError(f"{query_id}: invalid visible/infrared path")
        key = f"{visible}|{infrared}"
        unique.setdefault(
            key,
            {
                "pair_id": Path(visible).stem,
                "source": "aic",
                "rgb_path": root / visible,
                "tir_path": root / infrared,
            },
        )
    return _parallel_audit(
        list(unique.values()),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        black_threshold=black_threshold,
        workers=workers,
    )


def summarize_pair_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if row["status"] == "ok"]
    valid_ratios = [float(row["ir_valid_ratio"]) for row in success]
    return {
        "pair_count": len(rows),
        "success_count": len(success),
        "failure_count": len(rows) - len(success),
        "same_spatial_dimensions_count": sum(
            bool(row["same_spatial_dimensions"]) for row in success
        ),
        "grid_equal_count": sum(bool(row["grid_equal"]) for row in success),
        "ir_usable_count": sum(bool(row["ir_usable"]) for row in success),
        "ir_low_information_count": sum(
            bool(row["ir_low_information"]) for row in success
        ),
        "ir_black_border_count": sum(
            float(row["ir_valid_ratio"]) < 0.999 for row in success
        ),
        "ir_valid_ratio_median": (
            float(np.median(valid_ratios)) if valid_ratios else None
        ),
        "failures": [
            {"pair_id": row["pair_id"], "error": row["error"]}
            for row in rows
            if row["status"] != "ok"
        ],
    }


class _FakeBlock(nn.Module):
    def __init__(self, hidden_size: int, index: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden_size) * (1.0 + index / 100.0))

    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.linear(hidden_states)


class _FakeVisual(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.blocks = nn.ModuleList(
            [_FakeBlock(hidden_size, index) for index in range(25)]
        )
        self.deepstack_visual_indexes = [8, 16, 24]
        self.deepstack_merger_list = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(3)]
        )
        self.merger = nn.Linear(hidden_size, hidden_size, bias=False)
        self.spatial_merge_size = 2
        self.patch_size = 16
        with torch.no_grad():
            self.merger.weight.copy_(torch.eye(hidden_size))
            for merger in self.deepstack_merger_list:
                merger.weight.copy_(torch.eye(hidden_size))

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **_: Any
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del grid_thw
        deep: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)
            if index in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(index)
                ]
                deep.append(merger(hidden_states))
        return self.merger(hidden_states), deep


def validate_zero_gate_equivalence(
    *, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> dict[str, Any]:
    torch.manual_seed(20260812)
    visual = _FakeVisual().to(device=device, dtype=dtype)
    adapter = Qwen3VLRGBTAdapter(
        visual, hidden_size=8, enforce_qwen_contract=True
    ).to(device=device, dtype=dtype)
    rgb = torch.randn(4, 8, device=device, dtype=dtype)
    tir = torch.randn(4, 8, device=device, dtype=dtype)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.int64, device=device)
    mask = torch.ones(4, device=device)
    output = adapter.encode_visual_pair(
        rgb_pixel_values=rgb,
        tir_pixel_values=tir,
        image_grid_thw=grid,
        ir_patch_valid_mask=mask,
    )
    final_diff = float(
        (output.final_hidden - output.rgb_native_final).abs().max().item()
    )
    deep_diff = max(
        float((value - native).abs().max().item())
        for value, native in zip(output.deepstack_hidden, output.rgb_native_deepstack)
    )
    rgb_only = adapter.encode_visual_pair(
        rgb_pixel_values=rgb,
        image_grid_thw=grid,
        tir_pixel_values=None,
    )
    rgb_only_diff = float(
        (rgb_only.final_hidden - output.rgb_native_final).abs().max().item()
    )
    threshold = 1e-3 if dtype == torch.bfloat16 else 1e-6
    return {
        "device": device,
        "dtype": str(rgb.dtype),
        "final_max_abs_diff": final_diff,
        "deepstack_max_abs_diff": deep_diff,
        "rgb_only_max_abs_diff": rgb_only_diff,
        "used_tir": output.used_tir,
        "rgb_only_used_tir": rgb_only.used_tir,
        "threshold": threshold,
        "passed": max(final_diff, deep_diff, rgb_only_diff) <= threshold,
    }


def run_phase0_pytest(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    tests = sorted((root / "tests").glob("test_rgbtir_*.py"))
    command = [sys.executable, "-m", "pytest", "-q", *map(str, tests)]
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    stdout = re.sub(r"\bin\s+\d+(?:\.\d+)?s\b", "in <elapsed>", completed.stdout)
    stderr = re.sub(r"\bin\s+\d+(?:\.\d+)?s\b", "in <elapsed>", completed.stderr)
    return {
        "command": command,
        "test_file_count": len(tests),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": stdout,
        "stderr": stderr,
    }


def exclusion_reason_counts(bundle: ManifestBundle) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in bundle.excluded_train:
        counter.update(record.exclusion_reasons)
    return dict(sorted(counter.items()))


def build_readiness_report(
    *,
    manifest_summary: Mapping[str, Any],
    rgbt_summary: Mapping[str, Any],
    aic_summary: Mapping[str, Any],
    processor_summary: Mapping[str, Any],
    equivalence_summary: Mapping[str, Any],
    tracer_summary: Mapping[str, Any],
    pytest_summary: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    blockers: list[str] = []
    expected = {
        "raw_count": 38760,
        "train_all_count": 26604,
        "train_clean_count": 26477,
        "val_count": 2032,
        "test_count": 10124,
        "excluded_train_count": 127,
        "unique_pair_count": 21535,
    }
    for field, value in expected.items():
        if manifest_summary.get(field) != value:
            blockers.append(
                f"manifest {field}={manifest_summary.get(field)!r}, expected {value}"
            )
    for name, summary in (("RGBT", rgbt_summary), ("AIC", aic_summary)):
        if summary.get("failure_count"):
            blockers.append(f"{name} pair audit has {summary['failure_count']} failures")
        if summary.get("same_spatial_dimensions_count") != summary.get("pair_count"):
            blockers.append(f"{name} contains RGB/TIR source-size mismatches")
        if summary.get("grid_equal_count") != summary.get("pair_count"):
            blockers.append(f"{name} contains RGB/TIR grid_thw mismatches")
    if not equivalence_summary.get("passed"):
        blockers.append("zero-gate RGB equivalence failed")
    if tracer_summary.get("combined", {}).get("unique_pair_count") != 500:
        blockers.append("tracer does not contain 500 unique image pairs")
    if tracer_summary.get("combined", {}).get("unmet_minima"):
        blockers.append("tracer condition minima were not met")
    if not pytest_summary.get("passed"):
        blockers.append("Phase 0 pytest suite failed")
    if processor_summary.get("failure_count"):
        blockers.append("paired processor tracer validation failed")
    status = "PHASE_0_GO" if not blockers else "PHASE_0_NO_GO"
    lines = [
        "# AIC RGB–TIR Phase 0 可训练性验收",
        "",
        f"**最终状态：`{status}`**",
        "",
        "本轮仅完成数据、成对预处理和模型接口验证；未训练、未生成 checkpoint、未生成 AIC 提交。",
        "",
        "## Manifest",
        "",
        f"- 原始实例：{manifest_summary.get('raw_count')}；唯一图像对：{manifest_summary.get('unique_pair_count')}；",
        f"- train all / clean：{manifest_summary.get('train_all_count')} / {manifest_summary.get('train_clean_count')}；",
        f"- val / test：{manifest_summary.get('val_count')} / {manifest_summary.get('test_count')}；",
        f"- 排除训练记录：{manifest_summary.get('excluded_train_count')}；原因：`{manifest_summary.get('exclusion_reason_counts')}`；",
        f"- 跨 split 图像对：`{manifest_summary.get('cross_split_pairs')}`，对应 train 记录已排除。",
        "",
        "## RGB/TIR 图像与 Grid",
        "",
        f"- RGBT：{rgbt_summary.get('success_count')}/{rgbt_summary.get('pair_count')} 成功，黑边 {rgbt_summary.get('ir_black_border_count')}，低信息 IR {rgbt_summary.get('ir_low_information_count')}，可用 IR {rgbt_summary.get('ir_usable_count')}；",
        f"- AIC：{aic_summary.get('success_count')}/{aic_summary.get('pair_count')} 成功，黑边 {aic_summary.get('ir_black_border_count')}，低信息 IR {aic_summary.get('ir_low_information_count')}，可用 IR {aic_summary.get('ir_usable_count')}；",
        f"- RGBT/AIC grid 相等：{rgbt_summary.get('grid_equal_count')}/{rgbt_summary.get('pair_count')}、{aic_summary.get('grid_equal_count')}/{aic_summary.get('pair_count')}。",
        "",
        "## Processor 与模型等价性",
        "",
        f"- tracer processor：{processor_summary.get('success_count')}/{processor_summary.get('record_count')}；",
        f"- gate=0 final 最大绝对误差：{equivalence_summary.get('final_max_abs_diff')}；",
        f"- gate=0 DeepStack 最大绝对误差：{equivalence_summary.get('deepstack_max_abs_diff')}；",
        f"- RGB-only 缺失 TIR 最大绝对误差：{equivalence_summary.get('rgb_only_max_abs_diff')}；",
        "- 包装器不包含跨模型 fallback；IR 不可用时走同一个模型的 RGB-only 路径。",
        "",
        "## Tracer",
        "",
        f"- train / val：{tracer_summary.get('train', {}).get('total')} / {tracer_summary.get('val', {}).get('total')}；",
        f"- 合并条件统计：`{tracer_summary.get('combined', {}).get('condition_counts')}`；",
        f"- 来源统计：`{tracer_summary.get('combined', {}).get('source_counts')}`。",
        "",
        "## 测试",
        "",
        f"- pytest：{'PASS' if pytest_summary.get('passed') else 'FAIL'}；测试文件 {pytest_summary.get('test_file_count')} 个；",
        f"- paired processor failure：{processor_summary.get('failure_count')}。",
        "",
        "## Phase 1 进入条件",
        "",
    ]
    if blockers:
        lines.append("当前不得进入 Phase 1，阻塞项：")
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.extend(
            [
                "所有硬门槛通过，可以进入 Qwen3-VL-8B BF16 的 TIR adapter warmup。",
                "进入训练前仍需在真实 8B BF16 视觉权重上完成一次 10 条无训练 smoke。",
            ]
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- RGBT 的天气、尺寸和遮挡字段是外部数据 GT；",
            "- AIC 的黑边、低信息和 grid 结果是输入审计事实；",
            "- 本轮没有 AIC bbox GT，因此没有多模态精度提升结论。",
            "",
        ]
    )
    return "\n".join(lines), status, blockers


def build_sha256_manifest(output_root: Path | str) -> dict[str, Any]:
    root = Path(output_root).resolve()
    outputs: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            outputs[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {"schema_version": 1, "output_root": str(root), "outputs": outputs}
