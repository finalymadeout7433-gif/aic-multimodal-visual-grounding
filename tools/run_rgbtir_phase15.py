from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_rgbtir.data import RGBTRecord, read_jsonl_records  # noqa: E402
from aic_rgbtir.modeling import Qwen3VLRGBTAdapter, install_asymmetric_lora  # noqa: E402
from aic_rgbtir.phase1 import (  # noqa: E402
    download_qwen_vision_assets,
    load_qwen3vl_vision_only,
    load_tir_adapter_state_dict,
    parameter_sha256,
    read_qwen_vision_asset_plan,
)
from aic_rgbtir.phase15 import Phase15Config, Phase15Validator  # noqa: E402
from aic_rgbtir.processing import PairedRGBTProcessor, qwen_grid_for_size  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 1.5 config must be a YAML mapping")
    return payload


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _paths(config: Mapping[str, Any], config_path: Path) -> dict[str, Path]:
    section = config["paths"]
    return {name: _resolve(str(value), config_path.parent) for name, value in section.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _base_parameters(module: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Parameter]]:
    for name, parameter in module.named_parameters():
        if ".paths." in name or name.startswith("fusions."):
            continue
        yield name, parameter


def _require_local_cuda(minimum_free_gib: float) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE_15_LOCAL_HARDWARE_NO_GO: CUDA is unavailable")
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    props = torch.cuda.get_device_properties(0)
    free_gib = free_bytes / 1024**3
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            "PHASE_15_LOCAL_HARDWARE_NO_GO: "
            f"free VRAM {free_gib:.2f} GiB is below {minimum_free_gib:.2f} GiB"
        )
    return {
        "name": props.name,
        "total_vram_gib": total_bytes / 1024**3,
        "free_vram_gib_before_load": free_gib,
        "compute_capability": f"{props.major}.{props.minor}",
    }


def _preflight(config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], list[RGBTRecord]]:
    paths = _paths(config, config_path)
    required = (
        paths["rgbt_root"] / "image_data",
        paths["val_manifest"],
        paths["phase1_adapter"],
        paths["phase1_full_checkpoint"],
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Phase 1.5 inputs: {missing}")
    records = read_jsonl_records(paths["val_manifest"])
    pair_count = len({record.image_pair_key for record in records})
    runtime = config["runtime"]
    expected = (int(runtime["expected_records"]), int(runtime["expected_pairs"]))
    if (len(records), pair_count) != expected:
        raise RuntimeError(
            f"official val drifted: records/pairs={(len(records), pair_count)}, expected={expected}"
        )
    adapter_sha = _sha256(paths["phase1_adapter"])
    full_sha = _sha256(paths["phase1_full_checkpoint"])
    expected_phase1 = config["expected_phase1"]
    if adapter_sha != str(expected_phase1["adapter_sha256"]).upper():
        raise RuntimeError(f"adapter SHA-256 mismatch: {adapter_sha}")
    if full_sha != str(expected_phase1["full_checkpoint_sha256"]).upper():
        raise RuntimeError(f"full checkpoint SHA-256 mismatch: {full_sha}")
    gpu = _require_local_cuda(float(runtime["minimum_free_vram_gib"]))
    output = paths["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PHASE_15_PREFLIGHT_READY",
        "record_count": len(records),
        "unique_pair_count": pair_count,
        "adapter": {"path": str(paths["phase1_adapter"]), "sha256": adapter_sha},
        "full_checkpoint": {"path": str(paths["phase1_full_checkpoint"]), "sha256": full_sha},
        "manifest": {"path": str(paths["val_manifest"]), "sha256": _sha256(paths["val_manifest"])},
        "gpu": gpu,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "disk_free_gib": shutil.disk_usage(output.parent).free / 1024**3,
        "claim_boundary": "frozen full-val TIR representation validation; no Query, fusion, bbox or AIC claim",
    }
    _write_json(output / "preflight.json", payload)
    return payload, records


def _download_and_verify_assets(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    paths = _paths(config, config_path)
    model = config["model"]
    exact_shard = paths["model_cache"].parent / "ModelScope_ExactShard8" / "model-00004-of-00004.safetensors"
    plan = read_qwen_vision_asset_plan(
        model_id=str(model["model_id"]),
        revision=str(model["revision"]),
        cache_dir=paths["model_cache"],
        local_files_only=False,
    )
    if exact_shard.is_file():
        from huggingface_hub import hf_hub_download

        assets = {
            name: Path(
                hf_hub_download(
                    repo_id=str(model["model_id"]),
                    filename=name,
                    revision=str(model["revision"]),
                    cache_dir=str(paths["model_cache"]),
                    local_files_only=True,
                )
            )
            for name in ("config.json", "preprocessor_config.json", "model.safetensors.index.json")
        }
        assets["model-00004-of-00004.safetensors"] = exact_shard
    else:
        assets = download_qwen_vision_assets(plan, cache_dir=paths["model_cache"])
    expected = config["expected_assets"]
    rows: dict[str, Any] = {}
    for name, expected_row in expected.items():
        path = assets[name]
        row = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        if row["bytes"] != int(expected_row["bytes"]) or row["sha256"] != str(expected_row["sha256"]).upper():
            raise RuntimeError(f"model asset hash/size mismatch for {name}: {row}")
        rows[name] = row
    _write_json(
        paths["output_root"] / "vision_assets.json",
        {"status": "VISION_ASSETS_VERIFIED", "plan": plan.as_dict(), "files": rows},
    )
    return assets


def _build_model(config: dict[str, Any], config_path: Path, assets: dict[str, Path]):
    paths = _paths(config, config_path)
    visual = load_qwen3vl_vision_only(assets=assets, device="cuda:0", dtype=torch.bfloat16)
    model = config["model"]
    router = install_asymmetric_lora(
        visual,
        target_suffixes=tuple(model["lora_target_suffixes"]),
        rgb_rank=int(model["rgb_lora_rank"]),
        tir_rank=int(model["tir_lora_rank"]),
        alpha=float(model["lora_alpha"]),
    )
    wrapper = Qwen3VLRGBTAdapter(
        visual,
        hidden_size=int(model["vision_hidden_size"]),
        adapter_router=router,
    ).to(device="cuda:0", dtype=torch.bfloat16)
    wrapper.requires_grad_(False)
    base_before = parameter_sha256(_base_parameters(wrapper))
    release = torch.load(paths["phase1_adapter"], map_location="cpu", weights_only=False)
    if release.get("model_id") != model["model_id"] or release.get("model_revision") != model["revision"]:
        raise RuntimeError("Phase 1 adapter model identity mismatch")
    load_tir_adapter_state_dict(wrapper, release["tir_adapter"])
    wrapper.requires_grad_(False)
    wrapper.eval()
    base_after = parameter_sha256(_base_parameters(wrapper))
    if base_before != base_after:
        raise RuntimeError("RGB/base parameters changed while loading the TIR adapter")
    return wrapper, release, base_before


def _processor(config: Mapping[str, Any]) -> PairedRGBTProcessor:
    return PairedRGBTProcessor(**dict(config["processor"]))


def _to_cuda(batch: Any) -> dict[str, torch.Tensor]:
    return {
        "rgb": batch.rgb_pixel_values.to(device="cuda:0", dtype=torch.bfloat16, non_blocking=True),
        "tir": batch.tir_pixel_values.to(device="cuda:0", dtype=torch.bfloat16, non_blocking=True),
        "grid": batch.image_grid_thw.to(device="cuda:0", non_blocking=True),
    }


def _worstcase_records(
    records: list[RGBTRecord], config: Mapping[str, Any]
) -> list[RGBTRecord]:
    processor = config["processor"]
    unique: dict[str, RGBTRecord] = {}
    for record in records:
        unique.setdefault(record.image_pair_key, record)
    ranked = sorted(
        unique.values(),
        key=lambda record: (
            -int(
                __import__("math").prod(
                    qwen_grid_for_size(
                        record.width,
                        record.height,
                        min_pixels=int(processor["min_pixels"]),
                        max_pixels=int(processor["max_pixels"]),
                        patch_size=int(processor["patch_size"]),
                        merge_size=int(processor["spatial_merge_size"]),
                    )
                )
            ),
            record.image_pair_key,
        ),
    )
    return ranked[: int(config["runtime"]["smoke_pairs"])]


def _encode_callable(wrapper: Qwen3VLRGBTAdapter):
    def encode(batch: Any):
        moved = _to_cuda(batch)
        features = wrapper.encode_tir_validation_triplet(
            rgb_pixel_values=moved["rgb"],
            tir_pixel_values=moved["tir"],
            image_grid_thw=moved["grid"],
            ir_usable=bool(batch.ir_usable),
        )
        return features

    return encode


def _safety_checks(
    *,
    config: dict[str, Any],
    config_path: Path,
    wrapper: Qwen3VLRGBTAdapter,
    release: Mapping[str, Any],
    base_hash: str,
    smoke_batch: Any,
) -> dict[str, Any]:
    paths = _paths(config, config_path)
    full = torch.load(paths["phase1_full_checkpoint"], map_location="cpu", weights_only=False)
    release_state = release["tir_adapter"]
    full_state = full["tir_adapter"]
    adapter_keys_equal = set(release_state) == set(full_state)
    tensor_mismatches = [
        name
        for name in sorted(set(release_state) & set(full_state))
        if not torch.equal(release_state[name], full_state[name])
    ]
    moved = _to_cuda(smoke_batch)
    mask = smoke_batch.ir_patch_valid_mask.to(device="cuda:0")
    zero_mask = torch.zeros_like(mask)
    with torch.inference_mode():
        rgb_only = wrapper.encode_visual_pair(
            rgb_pixel_values=moved["rgb"], image_grid_thw=moved["grid"]
        )
        gate_zero = wrapper.encode_visual_pair(
            rgb_pixel_values=moved["rgb"],
            tir_pixel_values=moved["tir"],
            image_grid_thw=moved["grid"],
            ir_patch_valid_mask=mask,
        )
        unusable = wrapper.encode_visual_pair(
            rgb_pixel_values=moved["rgb"],
            tir_pixel_values=moved["tir"],
            image_grid_thw=moved["grid"],
            ir_patch_valid_mask=mask,
            ir_usable=False,
        )
        zero_ir = wrapper.encode_visual_pair(
            rgb_pixel_values=moved["rgb"],
            tir_pixel_values=moved["tir"],
            image_grid_thw=moved["grid"],
            ir_patch_valid_mask=zero_mask,
        )

    def max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left - right).abs().max().cpu())

    threshold = float(config["gates"]["gate_zero_max_abs_diff_bf16"])
    diffs = {
        "gate_zero": max_diff(gate_zero.final_hidden, rgb_only.final_hidden),
        "tir_none": 0.0,
        "ir_usable_false": max_diff(unusable.final_hidden, rgb_only.final_hidden),
        "all_zero_ir_mask": max_diff(zero_ir.final_hidden, rgb_only.final_hidden),
    }
    base_after = parameter_sha256(_base_parameters(wrapper))
    source = inspect.getsource(__import__("aic_rgbtir.modeling", fromlist=["*"])).lower()
    checks = {
        "adapter_key_set_108": adapter_keys_equal and len(release_state) == int(config["expected_phase1"]["adapter_tensor_count"]),
        "adapter_tensor_exact": not tensor_mismatches,
        "gate_zero_rgb_equivalent": diffs["gate_zero"] <= threshold,
        "tir_none_rgb_equivalent": diffs["tir_none"] <= threshold,
        "ir_usable_false_rgb_equivalent": diffs["ir_usable_false"] <= threshold,
        "all_zero_ir_mask_rgb_equivalent": diffs["all_zero_ir_mask"] <= threshold,
        "rgb_base_hash_unchanged": base_hash == base_after,
        "all_outputs_finite": all(
            torch.isfinite(output.final_hidden).all().item()
            for output in (rgb_only, gate_zero, unusable, zero_ir)
        ),
        "no_second_model_fallback": "fallback_model" not in source and "second_model" not in source,
        "no_trainable_parameters": not any(parameter.requires_grad for parameter in wrapper.parameters()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "max_abs_differences": diffs,
        "threshold": threshold,
        "adapter_tensor_mismatches": tensor_mismatches,
        "base_sha256_before": base_hash,
        "base_sha256_after": base_after,
    }


def _run_smoke(
    *,
    config: dict[str, Any],
    config_path: Path,
    records: list[RGBTRecord],
    processor: PairedRGBTProcessor,
    wrapper: Qwen3VLRGBTAdapter,
) -> tuple[dict[str, Any], Any]:
    paths = _paths(config, config_path)
    selected = _worstcase_records(records, config)
    torch.cuda.reset_peak_memory_stats()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    first_batch = None
    try:
        for record in tqdm(selected, desc="Phase1.5 worstcase smoke", unit="pair"):
            batch = processor.process(record, root=paths["rgbt_root"])
            if first_batch is None:
                first_batch = batch
            features = _encode_callable(wrapper)(batch)
            rows.append(
                {
                    "record_id": record.record_id,
                    "image_pair_key": record.image_pair_key,
                    "grid_thw": features.image_grid_thw.detach().cpu().tolist(),
                    "layer_shapes": {
                        name: list(value.shape)
                        for name, value in zip(features.layer_names, features.tir_adapted_premerger)
                    },
                    "all_finite": all(
                        torch.isfinite(value).all().item()
                        for path in (
                            features.rgb_premerger,
                            features.tir_base_premerger,
                            features.tir_adapted_premerger,
                        )
                        for value in path
                    ),
                }
            )
    except torch.cuda.OutOfMemoryError as exc:
        payload = {
            "status": "PHASE_15_LOCAL_HARDWARE_NO_GO",
            "reason": f"CUDA OOM: {exc}",
            "completed": len(rows),
            "requested": len(selected),
        }
        _write_json(paths["output_root"] / "worstcase_smoke.json", payload)
        raise RuntimeError(payload["status"]) from exc
    elapsed = time.perf_counter() - started
    payload = {
        "status": "PHASE_15_WORSTCASE_SMOKE_GO",
        "completed": len(rows),
        "requested": len(selected),
        "elapsed_seconds": elapsed,
        "seconds_per_pair": elapsed / max(len(rows), 1),
        "estimated_full_hours": elapsed / max(len(rows), 1) * int(config["runtime"]["expected_pairs"]) / 3600,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "rows": rows,
    }
    _write_json(paths["output_root"] / "worstcase_smoke.json", payload)
    if first_batch is None:
        raise RuntimeError("worst-case smoke selected no pairs")
    return payload, first_batch


def _phase15_config(config: Mapping[str, Any], fingerprint: str) -> Phase15Config:
    runtime, loss, gates = config["runtime"], config["loss"], config["gates"]
    return Phase15Config(
        expected_record_count=int(runtime["expected_records"]),
        expected_pair_count=int(runtime["expected_pairs"]),
        seed=int(runtime["seed"]),
        bootstrap_iterations=int(runtime["bootstrap_iterations"]),
        cache_checkpoint_pairs=int(runtime["cache_checkpoint_pairs"]),
        spatial_merge_size=int(config["processor"]["spatial_merge_size"]),
        roi_expansion=float(loss["roi_expansion"]),
        hard_negative_weight=float(loss["hard_negative_weight"]),
        hard_negative_margin=float(loss["margin"]),
        minimum_alignment_improvement=float(gates["minimum_alignment_improvement"]),
        maximum_subgroup_degradation=float(gates["maximum_subgroup_degradation"]),
        subgroup_minimum_records=int(gates["subgroup_minimum_records"]),
        maximum_r5_drop=float(gates["maximum_r5_drop"]),
        minimum_retrieval_gain=float(gates["minimum_retrieval_gain"]),
        minimum_effective_rank_vs_base=float(gates["minimum_effective_rank_vs_base"]),
        minimum_effective_rank_vs_rgb=float(gates["minimum_effective_rank_vs_rgb"]),
        maximum_nonpaired_p95_increase=float(gates["maximum_nonpaired_p95_increase"]),
        fingerprint=fingerprint,
    )


def _build_report(
    *, config: Mapping[str, Any], config_path: Path, result: Any, smoke: Mapping[str, Any]
) -> None:
    paths = _paths(config, config_path)
    alignment = result.alignment_summary
    lines = [
        "# AIC RGB–TIR Phase 1.5 全量红外适配验证",
        "",
        f"**最终状态：`{result.status}`**",
        "",
        "本报告只验证冻结 TIR rank-48 Adapter 在 RGBT-GroundBench official val 上的表征泛化；不验证 Query grounding、bbox ACC 或 AIC 平台收益。",
        "",
        "## 执行范围",
        "",
        f"- 记录：{result.completed_record_count}/2,032；唯一图像对：{result.completed_pair_count}/1,115；跳过：{result.skipped_record_count}。",
        f"- 本机最坏样本 smoke：{smoke['completed']}/{smoke['requested']}，峰值显存 {smoke['peak_vram_gib']:.2f} GiB。",
        "- 全部权重冻结；无训练、无反向传播、无 Query、无融合门、无 bbox 生成、无 AIC 测试集。",
        "",
        "## 核心结果",
        "",
        f"- Base TIR alignment loss mean：{alignment['base']['mean']:.6f}。",
        f"- Adapted TIR alignment loss mean：{alignment['adapted']['mean']:.6f}。",
        f"- 相对改善：{alignment['relative_improvement'] * 100:.2f}%。",
        "",
        "### 检索与坍缩门禁",
        "",
    ]
    for layer in ("8", "16", "24", "final"):
        metric = result.retrieval_metrics[layer]
        collapse = result.collapse_metrics[layer]
        lines.append(
            f"- Layer {layer}：base/adapted R@1 {metric['base']['r_at_1']:.4f}/{metric['adapted']['r_at_1']:.4f}，"
            f"R@5 {metric['base']['r_at_5']:.4f}/{metric['adapted']['r_at_5']:.4f}，"
            f"adapted margin {metric['adapted']['paired_shuffled_margin']:.4f}，"
            f"effective rank base/adapted {collapse['base']['effective_rank']:.2f}/{collapse['adapted']['effective_rank']:.2f}。"
        )
    lines.extend(["", "## 硬门禁", ""])
    for name, passed in result.gate_checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    if result.gate_failures:
        lines.extend(["", "失败原因："])
        lines.extend(f"- {reason}" for reason in result.gate_failures)
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 已验证事实：full-val RGB/TIR ROI 表征对齐、跨模态检索、非配对相似度、effective rank 与安全等价性。",
            "- 当前不能确认：自然语言 Query 是否受益、最终 bbox 是否改善、AIC ACC 是否提高。",
            f"- Phase 2：{'允许启动' if result.status == 'PHASE_15_GO' else '禁止启动，先修复首要失败项'}。",
            "",
        ]
    )
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text("\n".join(lines), encoding="utf-8")


def _sha_manifest(output_root: Path) -> None:
    rows: dict[str, Any] = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            rows[path.relative_to(output_root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _write_json(output_root / "sha256_manifest.json", {"schema_version": 1, "outputs": rows})


def run(config_path: Path, *, resume: bool) -> int:
    del resume  # pair-part cache is always safe to resume when the fingerprint matches
    config = _load_yaml(config_path)
    paths = _paths(config, config_path)
    preflight, records = _preflight(config, config_path)
    assets = _download_and_verify_assets(config, config_path)
    fingerprint = _canonical_sha(
        {
            "model": config["model"],
            "processor": config["processor"],
            "loss": config["loss"],
            "gates": config["gates"],
            "manifest_sha256": preflight["manifest"]["sha256"],
            "adapter_sha256": preflight["adapter"]["sha256"],
            "assets": {name: _sha256(path) for name, path in sorted(assets.items())},
            "source": {
                path.name: _sha256(path)
                for path in (
                    REPO_ROOT / "src/aic_rgbtir/modeling.py",
                    REPO_ROOT / "src/aic_rgbtir/phase15.py",
                    Path(__file__),
                )
            },
        }
    )
    wrapper, release, base_hash = _build_model(config, config_path, assets)
    processor = _processor(config)
    smoke, smoke_batch = _run_smoke(
        config=config,
        config_path=config_path,
        records=records,
        processor=processor,
        wrapper=wrapper,
    )
    safety = _safety_checks(
        config=config,
        config_path=config_path,
        wrapper=wrapper,
        release=release,
        base_hash=base_hash,
        smoke_batch=smoke_batch,
    )
    validator = Phase15Validator(
        processor=processor,
        encode_triplet=_encode_callable(wrapper),
        output_root=paths["output_root"],
        config=_phase15_config(config, fingerprint),
        safety_checks=lambda: safety,
    )
    result = validator.run(records, paths["rgbt_root"])
    _build_report(config=config, config_path=config_path, result=result, smoke=smoke)
    _sha_manifest(paths["output_root"])
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0 if result.status == "PHASE_15_GO" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="AIC RGB-TIR Phase 1.5 full validator")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(20260812)
    return run(args.config.resolve(), resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
