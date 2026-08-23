from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_rgbtir.data import RGBTRecord, read_jsonl_records  # noqa: E402
from aic_rgbtir.modeling import (  # noqa: E402
    Qwen3VLRGBTAdapter,
    install_asymmetric_lora,
)
from aic_rgbtir.phase1 import (  # noqa: E402
    BBoxAwareTIRAlignmentLoss,
    build_roi_patch_weights,
    download_qwen_vision_assets,
    freeze_for_tir_adapter_warmup,
    load_qwen3vl_vision_only,
    load_tir_adapter_state_dict,
    parameter_sha256,
    read_qwen_vision_asset_plan,
    tir_adapter_state_dict,
)
from aic_rgbtir.processing import PairedRGBTProcessor  # noqa: E402


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 1 config must be a YAML mapping")
    return payload


def _resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _phase0_paths(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    section = config["paths"]
    manifest_root = _resolve_path(section["phase0_manifest_root"], base=config_path.parent)
    return {
        "rgbt_root": _resolve_path(section["rgbt_root"], base=config_path.parent),
        "manifest_root": manifest_root,
        "train_clean": manifest_root / "train_clean.jsonl",
        "tracer_train": manifest_root / "tracer_train_400.jsonl",
        "tracer_val": manifest_root / "tracer_val_100.jsonl",
        "phase0_summary": _resolve_path(section["phase0_summary"], base=config_path.parent),
        "output_root": _resolve_path(section["output_root"], base=config_path.parent),
        "cache_dir": _resolve_path(section["model_cache"], base=config_path.parent),
    }


def _processor(config: dict[str, Any]) -> PairedRGBTProcessor:
    return PairedRGBTProcessor(**config["processor"])


def _require_cuda(min_vram_gib: float) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real Qwen3-VL Phase 1 run")
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
    if total_gib < min_vram_gib:
        raise RuntimeError(
            f"GPU has {total_gib:.2f} GiB, Phase 1 requires at least {min_vram_gib:.2f} GiB"
        )
    return {
        "name": props.name,
        "total_vram_gib": total_gib,
        "compute_capability": f"{props.major}.{props.minor}",
    }


def _read_phase0_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PHASE_0_GO":
        raise RuntimeError(f"Phase 0 is not GO: {payload.get('status')!r}")
    if payload.get("manifest_summary", {}).get("train_clean_count") != 26477:
        raise RuntimeError("Phase 0 clean-train count is not the frozen 26,477")
    return payload


def _base_parameters(module: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Parameter]]:
    for name, parameter in module.named_parameters():
        if ".paths." in name or name.startswith("fusions."):
            continue
        yield name, parameter


def _asset_plan(config: dict[str, Any], paths: dict[str, Path], *, local_only: bool):
    model = config["model"]
    return read_qwen_vision_asset_plan(
        model_id=model["model_id"],
        revision=model["revision"],
        cache_dir=paths["cache_dir"],
        local_files_only=local_only,
    )


def command_preflight(config: dict[str, Any], config_path: Path, *, allow_network: bool) -> None:
    paths = _phase0_paths(config, config_path)
    required = [
        paths["rgbt_root"] / "image_data",
        paths["train_clean"],
        paths["tracer_train"],
        paths["tracer_val"],
        paths["phase0_summary"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Phase 1 inputs: {missing}")
    phase0 = _read_phase0_summary(paths["phase0_summary"])
    counts = {
        "train_clean": len(read_jsonl_records(paths["train_clean"])),
        "tracer_train": len(read_jsonl_records(paths["tracer_train"])),
        "tracer_val": len(read_jsonl_records(paths["tracer_val"])),
    }
    if counts != {"train_clean": 26477, "tracer_train": 400, "tracer_val": 100}:
        raise RuntimeError(f"manifest counts drifted: {counts}")
    plan = _asset_plan(config, paths, local_only=not allow_network)
    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_root.parent)
    payload = {
        "status": "PHASE_1_PREFLIGHT_READY",
        "purpose": "vision-only TIR adapter warmup; no language grounding claim",
        "phase0_status": phase0["status"],
        "manifest_counts": counts,
        "model_asset_plan": plan.as_dict(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "disk_free_gib": disk.free / 1024**3,
        "config_sha256": _canonical_hash(config),
        "input_sha256": {
            path.name: _sha256(path)
            for path in (paths["train_clean"], paths["tracer_train"], paths["tracer_val"])
        },
    }
    _write_json(output_root / "preflight.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_download_assets(config: dict[str, Any], config_path: Path) -> None:
    paths = _phase0_paths(config, config_path)
    plan = _asset_plan(config, paths, local_only=False)
    assets = download_qwen_vision_assets(plan, cache_dir=paths["cache_dir"])
    payload = {
        "status": "VISION_ASSETS_READY",
        "plan": plan.as_dict(),
        "files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in assets.items()
        },
    }
    _write_json(paths["output_root"] / "vision_assets.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_model(config: dict[str, Any], paths: dict[str, Path]):
    gpu = _require_cuda(float(config["runtime"]["minimum_vram_gib"]))
    plan = _asset_plan(config, paths, local_only=True)
    assets = download_qwen_vision_assets(plan, cache_dir=paths["cache_dir"])
    visual = load_qwen3vl_vision_only(
        assets=assets, device="cuda:0", dtype=torch.bfloat16
    )
    model_cfg = config["model"]
    router = install_asymmetric_lora(
        visual,
        target_suffixes=tuple(model_cfg["lora_target_suffixes"]),
        rgb_rank=int(model_cfg["rgb_lora_rank"]),
        tir_rank=int(model_cfg["tir_lora_rank"]),
        alpha=float(model_cfg["lora_alpha"]),
    )
    wrapper = Qwen3VLRGBTAdapter(
        visual,
        hidden_size=int(model_cfg["vision_hidden_size"]),
        adapter_router=router,
    ).to(device="cuda:0", dtype=torch.bfloat16)
    trainable = freeze_for_tir_adapter_warmup(wrapper)
    wrapper.eval()
    return wrapper, trainable, gpu, assets


def _move_batch(batch, device: str = "cuda:0") -> dict[str, Any]:
    return {
        "rgb": batch.rgb_pixel_values.to(device=device, dtype=torch.bfloat16),
        "tir": batch.tir_pixel_values.to(device=device, dtype=torch.bfloat16),
        "grid": batch.image_grid_thw.to(device=device),
        "valid": batch.ir_patch_valid_mask.to(device=device),
    }


def _one_loss(wrapper, criterion, processor, record: RGBTRecord, root: Path):
    batch = processor.process(record, root=root)
    if batch.normalized_bbox is None or not batch.ir_usable:
        raise ValueError(f"record is not usable for TIR warmup: {record.record_id}")
    moved = _move_batch(batch)
    features = wrapper.encode_for_tir_warmup(
        rgb_pixel_values=moved["rgb"],
        tir_pixel_values=moved["tir"],
        image_grid_thw=moved["grid"],
    )
    roi = build_roi_patch_weights(
        normalized_bbox=batch.normalized_bbox,
        image_grid_thw=moved["grid"],
        spatial_merge_size=int(wrapper.visual_encoder.spatial_merge_size),
        device="cuda:0",
    )
    output = criterion(features, roi_patch_weights=roi, ir_valid_mask=moved["valid"])
    return output, batch, features


def command_smoke(config: dict[str, Any], config_path: Path) -> None:
    paths = _phase0_paths(config, config_path)
    _read_phase0_summary(paths["phase0_summary"])
    count = int(config["runtime"]["smoke_records"])
    records = read_jsonl_records(paths["tracer_val"])[:count]
    wrapper, trainable, gpu, assets = _build_model(config, paths)
    processor = _processor(config)
    criterion = BBoxAwareTIRAlignmentLoss(**config["loss"])
    base_hash_before = parameter_sha256(_base_parameters(wrapper))
    torch.cuda.reset_peak_memory_stats()
    max_gate_zero_diff = 0.0
    loss_rows: list[dict[str, Any]] = []
    hook_shapes: dict[str, list[int]] | None = None
    gradient_names: list[str] = []
    for index, record in enumerate(records):
        batch = processor.process(record, root=paths["rgbt_root"])
        moved = _move_batch(batch)
        with torch.no_grad():
            rgb_only = wrapper.encode_visual_pair(
                rgb_pixel_values=moved["rgb"], image_grid_thw=moved["grid"]
            )
            gate_zero = wrapper.encode_visual_pair(
                rgb_pixel_values=moved["rgb"],
                tir_pixel_values=moved["tir"],
                image_grid_thw=moved["grid"],
                ir_patch_valid_mask=moved["valid"],
                ir_usable=batch.ir_usable,
            )
        max_gate_zero_diff = max(
            max_gate_zero_diff,
            float((rgb_only.final_hidden - gate_zero.final_hidden).abs().max().cpu()),
        )
        wrapper.zero_grad(set_to_none=True)
        output, _, features = _one_loss(
            wrapper, criterion, processor, record, paths["rgbt_root"]
        )
        if index == 0:
            output.loss.backward()
            gradient_names = [
                name
                for name, parameter in wrapper.named_parameters()
                if parameter.grad is not None
            ]
            forbidden = [name for name in gradient_names if ".paths.tir." not in name]
            if forbidden:
                raise RuntimeError(f"smoke gradient isolation failed: {forbidden}")
            hook_shapes = {
                name: list(tensor.shape)
                for name, tensor in zip(features.layer_names, features.tir_premerger)
            }
        loss_rows.append(
            {
                "record_id": record.record_id,
                "loss": float(output.loss.detach().cpu()),
                "foreground_alignment": float(output.foreground_alignment.cpu()),
                "hard_negative_margin": float(output.hard_negative_margin.cpu()),
                "roi_token_count": output.roi_token_count,
            }
        )
    base_hash_after = parameter_sha256(_base_parameters(wrapper))
    threshold = float(config["gates"]["gate_zero_max_abs_diff_bf16"])
    payload = {
        "status": "PHASE_1_SMOKE_GO"
        if max_gate_zero_diff <= threshold and base_hash_before == base_hash_after
        else "PHASE_1_SMOKE_NO_GO",
        "record_count": len(records),
        "gpu": gpu,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "trainable": trainable,
        "gate_zero_max_abs_diff": max_gate_zero_diff,
        "gate_zero_threshold": threshold,
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "base_unchanged": base_hash_before == base_hash_after,
        "gradient_tensor_count": len(gradient_names),
        "all_gradients_tir_only": all(".paths.tir." in name for name in gradient_names),
        "hook_shapes": hook_shapes,
        "loss_rows": loss_rows,
        "asset_paths": {name: str(path) for name, path in assets.items()},
        "claim_boundary": "tests real visual hooks and TIR adaptation path only; does not establish Query-to-bbox grounding",
    }
    _write_json(paths["output_root"] / "smoke_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "PHASE_1_SMOKE_GO":
        raise RuntimeError(payload["status"])


@torch.no_grad()
def _evaluate(wrapper, criterion, processor, records, root: Path) -> dict[str, Any]:
    losses: list[float] = []
    cosines: list[list[float]] = []
    skipped = 0
    for record in records:
        try:
            output, _, _ = _one_loss(wrapper, criterion, processor, record, root)
        except ValueError:
            skipped += 1
            continue
        losses.append(float(output.loss.cpu()))
        cosines.append(list(output.layer_cosines))
    if not losses:
        raise RuntimeError("evaluation produced no valid samples")
    layer_count = len(cosines[0])
    return {
        "record_count": len(losses),
        "skipped": skipped,
        "mean_loss": sum(losses) / len(losses),
        "mean_layer_cosines": [
            sum(row[index] for row in cosines) / len(cosines)
            for index in range(layer_count)
        ],
    }


def _mode_records(config, paths, mode: str):
    runtime = config["runtime"]
    if mode == "overfit100":
        records = read_jsonl_records(paths["tracer_train"])[:100]
        return records, int(runtime["overfit_steps"])
    if mode == "tracer400":
        records = read_jsonl_records(paths["tracer_train"])
        return records, int(runtime["tracer_steps"])
    if mode == "full":
        records = read_jsonl_records(paths["train_clean"])
        steps = math.ceil(len(records) * float(runtime["full_epochs"]))
        return records, steps
    raise ValueError(f"unsupported warmup mode: {mode}")


def command_warmup(config: dict[str, Any], config_path: Path, *, mode: str, resume: bool) -> None:
    paths = _phase0_paths(config, config_path)
    smoke_path = paths["output_root"] / "smoke_summary.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "PHASE_1_SMOKE_GO":
        raise RuntimeError("real-weight smoke must be GO before warmup")
    records, max_steps = _mode_records(config, paths, mode)
    val_records = read_jsonl_records(paths["tracer_val"])
    wrapper, trainable, gpu, _assets = _build_model(config, paths)
    processor = _processor(config)
    criterion = BBoxAwareTIRAlignmentLoss(**config["loss"])
    opt_cfg = config["optimization"]
    trainable_params = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    mode_root = paths["output_root"] / mode
    checkpoint_path = mode_root / "checkpoint_last.pt"
    log_path = mode_root / "train_log.jsonl"
    start_step = 0
    checkpoint: dict[str, Any] | None = None
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["config_sha256"] != _canonical_hash(config):
            raise RuntimeError("resume checkpoint config fingerprint differs")
        load_tir_adapter_state_dict(wrapper, checkpoint["tir_adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
    elif not resume and checkpoint_path.exists():
        raise FileExistsError(f"checkpoint exists; use --resume: {checkpoint_path}")

    base_hash_before = parameter_sha256(_base_parameters(wrapper))
    metric_records = records if mode == "overfit100" else val_records
    baseline_eval = (
        checkpoint["baseline_evaluation"]
        if checkpoint is not None and "baseline_evaluation" in checkpoint
        else _evaluate(wrapper, criterion, processor, metric_records, paths["rgbt_root"])
    )
    rng = random.Random(int(config["runtime"]["seed"]))
    order = list(range(len(records)))
    rng.shuffle(order)
    grad_accum = int(opt_cfg["gradient_accumulation"])
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    skipped = 0
    for step in range(start_step, max_steps):
        record = records[order[step % len(order)]]
        try:
            output, _, _ = _one_loss(wrapper, criterion, processor, record, paths["rgbt_root"])
        except ValueError as exc:
            skipped += 1
            _append_jsonl(log_path, {"step": step + 1, "record_id": record.record_id, "skipped": str(exc)})
            continue
        (output.loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0 or step + 1 == max_steps:
            torch.nn.utils.clip_grad_norm_(trainable_params, float(opt_cfg["max_grad_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if (step + 1) % int(opt_cfg["log_every_steps"]) == 0 or step == start_step:
            elapsed = time.time() - started
            _append_jsonl(
                log_path,
                {
                    "step": step + 1,
                    "record_id": record.record_id,
                    "loss": float(output.loss.detach().cpu()),
                    "layer_cosines": list(output.layer_cosines),
                    "elapsed_seconds": elapsed,
                    "samples_per_second": (step + 1 - start_step) / max(elapsed, 1e-6),
                },
            )
        if (step + 1) % int(opt_cfg["save_every_steps"]) == 0 or step + 1 == max_steps:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step + 1,
                    "mode": mode,
                    "config_sha256": _canonical_hash(config),
                    "tir_adapter": tir_adapter_state_dict(wrapper),
                    "optimizer": optimizer.state_dict(),
                    "base_sha256": base_hash_before,
                    "baseline_evaluation": baseline_eval,
                },
                checkpoint_path,
            )
    final_eval = _evaluate(wrapper, criterion, processor, metric_records, paths["rgbt_root"])
    base_hash_after = parameter_sha256(_base_parameters(wrapper))
    relative_loss_improvement = (
        baseline_eval["mean_loss"] - final_eval["mean_loss"]
    ) / max(baseline_eval["mean_loss"], 1e-8)
    gate_name = (
        "minimum_overfit_loss_improvement"
        if mode == "overfit100"
        else "minimum_validation_loss_improvement"
    )
    gate = float(config["gates"][gate_name])
    payload = {
        "status": "PHASE_1_WARMUP_GO"
        if relative_loss_improvement >= gate and base_hash_before == base_hash_after
        else "PHASE_1_WARMUP_NO_GO",
        "mode": mode,
        "steps": max_steps,
        "skipped": skipped,
        "gpu": gpu,
        "trainable": trainable,
        "evaluation_scope": "overfit_train_100" if mode == "overfit100" else "official_val_100",
        "baseline_evaluation": baseline_eval,
        "final_evaluation": final_eval,
        "relative_loss_improvement": relative_loss_improvement,
        "gate_name": gate_name,
        "required_improvement": gate,
        "base_unchanged": base_hash_before == base_hash_after,
        "checkpoint": str(checkpoint_path),
        "claim_boundary": "TIR-to-frozen-RGB target-region alignment only; fusion and language grounding remain untrained",
    }
    _write_json(mode_root / "run_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "PHASE_1_WARMUP_GO":
        raise RuntimeError(payload["status"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIC RGB-TIR Phase 1 runner")
    parser.add_argument("command", choices=("preflight", "download-assets", "smoke", "warmup"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--mode", choices=("overfit100", "tracer400", "full"), default="overfit100")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(int(config["runtime"]["seed"]))
    if args.command == "preflight":
        command_preflight(config, config_path, allow_network=args.allow_network)
    elif args.command == "download-assets":
        command_download_assets(config, config_path)
    elif args.command == "smoke":
        command_smoke(config, config_path)
    else:
        command_warmup(config, config_path, mode=args.mode, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
