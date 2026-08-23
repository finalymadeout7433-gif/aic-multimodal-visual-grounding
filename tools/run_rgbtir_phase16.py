from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import shutil
import sys
import zipfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_rgbtir.data import RGBTRecord, read_jsonl_records, write_jsonl  # noqa: E402
from aic_rgbtir.modeling import Qwen3VLRGBTAdapter, install_asymmetric_lora  # noqa: E402
from aic_rgbtir.phase1 import (  # noqa: E402
    build_roi_patch_weights,
    download_qwen_vision_assets,
    freeze_for_tir_adapter_warmup,
    load_qwen3vl_vision_only,
    load_tir_adapter_state_dict,
    parameter_sha256,
    read_qwen_vision_asset_plan,
    tir_adapter_state_dict,
)
from aic_rgbtir.phase15 import Phase15Config, Phase15Validator  # noqa: E402
from aic_rgbtir.phase16 import (  # noqa: E402
    CandidateSpec,
    LAYER_NAMES,
    Phase16Config,
    Phase16RepairTrainer,
    RGBTeacherBank,
    TeacherEntry,
    build_repair_split,
    build_recovery_diagnostics,
)
from aic_rgbtir.processing import PairedRGBTProcessor  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 1.6 config must be a YAML mapping")
    return payload


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _paths(config: Mapping[str, Any], config_path: Path) -> dict[str, Path]:
    return {
        name: _resolve(str(value), config_path.parent)
        for name, value in config["paths"].items()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


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


def _require_cuda(minimum_vram_gib: float) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE_16_HARDWARE_NO_GO: CUDA is unavailable")
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1024**3
    if total < minimum_vram_gib:
        raise RuntimeError(
            f"PHASE_16_HARDWARE_NO_GO: GPU has {total:.2f} GiB, requires {minimum_vram_gib:.2f} GiB"
        )
    return {
        "name": props.name,
        "total_vram_gib": total,
        "compute_capability": f"{props.major}.{props.minor}",
    }


def _processor(config: Mapping[str, Any]) -> PairedRGBTProcessor:
    return PairedRGBTProcessor(**dict(config["processor"]))


def _to_cuda(batch: Any) -> dict[str, torch.Tensor]:
    return {
        "rgb": batch.rgb_pixel_values.to("cuda:0", dtype=torch.bfloat16),
        "tir": batch.tir_pixel_values.to("cuda:0", dtype=torch.bfloat16),
        "grid": batch.image_grid_thw.to("cuda:0"),
        "valid": batch.ir_patch_valid_mask.to("cuda:0"),
    }


def _weighted_mean(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=features.device, dtype=torch.float32)
    denominator = weights.sum().clamp_min(1.0)
    return (features.float() * weights[:, None]).sum(dim=0) / denominator


def _build_manifests(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    paths = _paths(config, config_path)
    train = read_jsonl_records(paths["train_clean_manifest"])
    official_val = read_jsonl_records(paths["official_val_manifest"])
    split_cfg = config["split"]
    split = build_repair_split(
        train,
        probe_pairs=int(split_cfg["probe_pairs"]),
        dev_pairs=int(split_cfg["dev_pairs"]),
        seed=int(config["runtime"]["seed"]),
    )
    manifest_root = paths["manifest_root"]
    write_jsonl(manifest_root / "repair_probe_train.jsonl", split.probe_train)
    write_jsonl(manifest_root / "repair_dev.jsonl", split.dev)
    write_jsonl(manifest_root / "repair_full_train.jsonl", split.full_train)
    write_jsonl(manifest_root / "official_val.jsonl", official_val)
    pair_sets = {
        "probe": {record.image_pair_key for record in split.probe_train},
        "dev": {record.image_pair_key for record in split.dev},
        "full": {record.image_pair_key for record in split.full_train},
        "official_val": {record.image_pair_key for record in official_val},
    }
    overlaps = {
        "probe_dev": sorted(pair_sets["probe"] & pair_sets["dev"]),
        "full_dev": sorted(pair_sets["full"] & pair_sets["dev"]),
        "train_official_val": sorted(pair_sets["full"] & pair_sets["official_val"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(
            f"Phase 1.6 split leakage detected: "
            f"{ {name: len(values) for name, values in overlaps.items()} }"
        )
    payload = {
        "status": "PHASE_16_MANIFESTS_READY",
        "seed": int(config["runtime"]["seed"]),
        "counts": {
            "probe_records": len(split.probe_train),
            "probe_pairs": len(pair_sets["probe"]),
            "dev_records": len(split.dev),
            "dev_pairs": len(pair_sets["dev"]),
            "full_records": len(split.full_train),
            "full_pairs": len(pair_sets["full"]),
            "official_val_records": len(official_val),
            "official_val_pairs": len(pair_sets["official_val"]),
        },
        "overlaps": {name: len(values) for name, values in overlaps.items()},
        "files": {
            path.name: _sha256(path)
            for path in sorted(manifest_root.glob("*.jsonl"))
        },
    }
    _write_json(manifest_root / "split_summary.json", payload)
    return payload


def _preflight(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    paths = _paths(config, config_path)
    required = [
        paths["rgbt_root"] / "image_data",
        paths["manifest_root"] / "repair_probe_train.jsonl",
        paths["manifest_root"] / "repair_dev.jsonl",
        paths["manifest_root"] / "repair_full_train.jsonl",
        paths["manifest_root"] / "official_val.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Phase 1.6 inputs: {missing}")
    manifests = {
        name: read_jsonl_records(paths["manifest_root"] / file_name)
        for name, file_name in {
            "probe": "repair_probe_train.jsonl",
            "dev": "repair_dev.jsonl",
            "full": "repair_full_train.jsonl",
            "official_val": "official_val.jsonl",
        }.items()
    }
    expected = config["expected"]
    actual = {
        "probe_pairs": len({record.image_pair_key for record in manifests["probe"]}),
        "dev_pairs": len({record.image_pair_key for record in manifests["dev"]}),
        "official_val_records": len(manifests["official_val"]),
        "official_val_pairs": len(
            {record.image_pair_key for record in manifests["official_val"]}
        ),
    }
    drift = {
        key: [actual[key], int(expected[key])]
        for key in actual
        if actual[key] != int(expected[key])
    }
    if drift:
        raise RuntimeError(f"Phase 1.6 manifest drift: {drift}")
    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PHASE_16_PREFLIGHT_READY",
        "counts": {name: len(records) for name, records in manifests.items()},
        "unique_pairs": {
            name: len({record.image_pair_key for record in records})
            for name, records in manifests.items()
        },
        "manifest_sha256": {
            name: _sha256(paths["manifest_root"] / file_name)
            for name, file_name in {
                "probe": "repair_probe_train.jsonl",
                "dev": "repair_dev.jsonl",
                "full": "repair_full_train.jsonl",
                "official_val": "official_val.jsonl",
            }.items()
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "disk_free_gib": shutil.disk_usage(output_root).free / 1024**3,
        "claim_boundary": "TIR representation repair only; no Query, bbox, fusion, AIC training or submission",
    }
    _write_json(output_root / "preflight.json", payload)
    return payload


def _model_assets(
    config: Mapping[str, Any], config_path: Path, *, allow_network: bool
) -> dict[str, Path]:
    paths = _paths(config, config_path)
    model = config["model"]
    plan = read_qwen_vision_asset_plan(
        model_id=str(model["model_id"]),
        revision=str(model["revision"]),
        cache_dir=paths["model_cache"],
        local_files_only=not allow_network,
    )
    assets = download_qwen_vision_assets(plan, cache_dir=paths["model_cache"])
    expected_assets = config.get("expected_assets", {})
    for name, expected in expected_assets.items():
        path = assets[name]
        if path.stat().st_size != int(expected["bytes"]) or _sha256(path) != str(
            expected["sha256"]
        ).upper():
            raise RuntimeError(f"model asset mismatch: {name}")
    return assets


def _build_model(
    config: Mapping[str, Any], assets: Mapping[str, Path]
) -> tuple[Qwen3VLRGBTAdapter, str]:
    model = config["model"]
    torch.manual_seed(int(config["runtime"]["seed"]))
    visual = load_qwen3vl_vision_only(
        assets=assets, device="cuda:0", dtype=torch.bfloat16
    )
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
    ).to("cuda:0", dtype=torch.bfloat16)
    freeze_for_tir_adapter_warmup(wrapper)
    wrapper.eval()
    return wrapper, parameter_sha256(_base_parameters(wrapper))


def _fingerprint(
    config: Mapping[str, Any], preflight: Mapping[str, Any], assets: Mapping[str, Path]
) -> str:
    return _canonical_sha(
        {
            "model": config["model"],
            "processor": config["processor"],
            "loss": config["loss"],
            "optimization": config["optimization"],
            "manifests": preflight["manifest_sha256"],
            "assets": {name: _sha256(path) for name, path in sorted(assets.items())},
            "source": {
                path.name: _sha256(path)
                for path in (
                    REPO_ROOT / "src/aic_rgbtir/modeling.py",
                    REPO_ROOT / "src/aic_rgbtir/phase16.py",
                    Path(__file__),
                )
            },
        }
    )


def _reference_records(
    paths: Mapping[str, Path], config: Mapping[str, Any]
) -> list[RGBTRecord]:
    by_id: dict[str, RGBTRecord] = {}
    names = tuple(
        str(value)
        for value in config.get("runtime", {}).get(
            "teacher_bank_manifests",
            ("repair_full_train.jsonl", "repair_dev.jsonl", "official_val.jsonl"),
        )
    )
    if not names:
        raise ValueError("teacher_bank_manifests cannot be empty")
    for name in names:
        for record in read_jsonl_records(paths["manifest_root"] / name):
            by_id.setdefault(record.record_id, record)
    return [by_id[key] for key in sorted(by_id)]


def _build_teacher_bank(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    wrapper: Qwen3VLRGBTAdapter,
    fingerprint: str,
    resume: bool,
) -> RGBTeacherBank:
    paths = _paths(config, config_path)
    output = paths["output_root"] / "rgb_teacher_bank"
    final_path = output / "teacher_bank.pt"
    partial_path = output / "teacher_bank.partial.pt"
    if final_path.is_file():
        return RGBTeacherBank.load(final_path, expected_fingerprint=fingerprint)
    entries: dict[str, TeacherEntry] = {}
    if resume and partial_path.is_file():
        entries.update(
            RGBTeacherBank.load(
                partial_path, expected_fingerprint=fingerprint
            ).entries
        )
    records = _reference_records(paths, config)
    by_pair: dict[str, list[RGBTRecord]] = defaultdict(list)
    for record in records:
        by_pair[record.image_pair_key].append(record)
    completed_pairs = {entry.image_pair_key for entry in entries.values()}
    processor = _processor(config)
    checkpoint_pairs = int(config["runtime"]["teacher_bank_checkpoint_pairs"])
    for pair_index, pair_key in enumerate(
        tqdm(sorted(by_pair), desc="RGB/Base-TIR teacher bank"), start=1
    ):
        if pair_key in completed_pairs:
            continue
        pair_records = by_pair[pair_key]
        batch = processor.process(pair_records[0], root=paths["rgbt_root"])
        moved = _to_cuda(batch)
        rgb = wrapper.encode_rgb_teacher_features(
            rgb_pixel_values=moved["rgb"], image_grid_thw=moved["grid"]
        )
        base = (
            wrapper.encode_base_tir_features(
                tir_pixel_values=moved["tir"], image_grid_thw=moved["grid"]
            )
            if batch.ir_usable
            else rgb
        )
        valid = moved["valid"] if batch.ir_usable else torch.ones_like(moved["valid"])
        for record in pair_records:
            roi = build_roi_patch_weights(
                normalized_bbox=record.bbox_xyxy_normalized,
                image_grid_thw=moved["grid"],
                spatial_merge_size=int(config["processor"]["spatial_merge_size"]),
                expansion=float(config["loss"]["roi_expansion"]),
                device="cuda:0",
            )
            foreground = roi * valid
            if float(foreground.sum()) < 1:
                raise RuntimeError(f"empty valid ROI: {record.record_id}")
            background = (1.0 - roi) * valid
            entries[record.record_id] = TeacherEntry(
                record_id=record.record_id,
                image_pair_key=record.image_pair_key,
                source_dataset=record.source_dataset,
                illumination=record.illumination,
                object_size=record.object_size,
                foreground={
                    name: _weighted_mean(tensor, foreground).detach().cpu().half()
                    for name, tensor in zip(rgb.layer_names, rgb.premerger)
                },
                background={
                    name: _weighted_mean(tensor, background).detach().cpu().half()
                    for name, tensor in zip(rgb.layer_names, rgb.premerger)
                },
                base_tir={
                    name: _weighted_mean(tensor, foreground).detach().cpu().half()
                    for name, tensor in zip(base.layer_names, base.premerger)
                },
            )
        if pair_index % checkpoint_pairs == 0:
            RGBTeacherBank(fingerprint=fingerprint, entries=entries).save(partial_path)
    bank = RGBTeacherBank(fingerprint=fingerprint, entries=entries)
    bank.save(final_path)
    bank.save(partial_path)
    _write_json(
        output / "summary.json",
        {
            "status": "PHASE_16_TEACHER_BANK_READY",
            "fingerprint": fingerprint,
            "record_count": len(entries),
            "pair_count": len({entry.image_pair_key for entry in entries.values()}),
            "path": str(final_path),
            "sha256": _sha256(final_path),
        },
    )
    return bank


def _phase16_config(config: Mapping[str, Any]) -> Phase16Config:
    runtime = config["runtime"]
    optimization = config["optimization"]
    loss = config["loss"]
    gates = config["gates"]
    return Phase16Config(
        seed=int(runtime["seed"]),
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        warmup_fraction=float(optimization["warmup_fraction"]),
        gradient_accumulation=int(optimization["gradient_accumulation"]),
        max_grad_norm=float(optimization["max_grad_norm"]),
        temperature=float(loss["temperature"]),
        margin=float(loss["margin"]),
        probe_steps=int(runtime["probe_steps"]),
        full_steps=int(runtime.get("full_steps", 0)),
        negative_count=int(loss["negative_count"]),
        same_source_negatives=int(loss["same_source_negatives"]),
        same_condition_negatives=int(loss["same_condition_negatives"]),
        relational_anchor_count=int(loss["relational_anchor_count"]),
        checkpoint_fractions=tuple(float(value) for value in runtime["checkpoint_fractions"]),
        minimum_alignment_improvement=float(gates["minimum_alignment_improvement"]),
        maximum_r5_drop=float(gates["maximum_r5_drop"]),
        minimum_retrieval_gain=float(gates["minimum_retrieval_gain"]),
        minimum_effective_rank_vs_base=float(gates["minimum_effective_rank_vs_base"]),
        minimum_effective_rank_vs_teacher=float(gates["minimum_effective_rank_vs_rgb"]),
        maximum_nonpaired_cosine_p95_increase=float(
            gates["maximum_nonpaired_p95_increase"]
        ),
    )


class _QwenRepairEncoder:
    def __init__(
        self,
        *,
        wrapper: Qwen3VLRGBTAdapter,
        processor: PairedRGBTProcessor,
        root: Path,
        roi_expansion: float,
    ) -> None:
        self.wrapper = wrapper
        self.processor = processor
        self.root = root
        self.roi_expansion = float(roi_expansion)

    def __call__(
        self, _module: torch.nn.Module, record: RGBTRecord
    ) -> Mapping[str, torch.Tensor]:
        batch = self.processor.process(record, root=self.root)
        if not batch.ir_usable:
            raise RuntimeError(f"unusable TIR cannot train repair adapter: {record.record_id}")
        moved = _to_cuda(batch)
        features = self.wrapper.encode_tir_repair_features(
            tir_pixel_values=moved["tir"], image_grid_thw=moved["grid"]
        )
        roi = build_roi_patch_weights(
            normalized_bbox=record.bbox_xyxy_normalized,
            image_grid_thw=moved["grid"],
            spatial_merge_size=int(self.processor.spatial_merge_size),
            expansion=self.roi_expansion,
            device="cuda:0",
        )
        foreground = roi * moved["valid"]
        if float(foreground.sum()) < 1:
            raise RuntimeError(f"empty valid TIR ROI: {record.record_id}")
        return {
            name: _weighted_mean(tensor, foreground)
            for name, tensor in zip(features.layer_names, features.premerger)
        }


@torch.inference_mode()
def _export_recovery_diagnostics(
    *,
    paths: Mapping[str, Path],
    wrapper: Qwen3VLRGBTAdapter,
    encoder: _QwenRepairEncoder,
    bank: RGBTeacherBank,
    dev: Sequence[RGBTRecord],
    candidate_names: Sequence[str],
) -> None:
    diagnostics_root = paths["output_root"] / "recovery_diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    teachers = {
        layer: torch.stack([bank[record.record_id].foreground[layer].float() for record in dev])
        for layer in LAYER_NAMES
    }
    base_tir = {
        layer: torch.stack([
            (bank[record.record_id].base_tir or bank[record.record_id].foreground)[layer].float()
            for record in dev
        ])
        for layer in LAYER_NAMES
    }
    combined: dict[str, Any] = {
        "status": "PHASE_16_RECOVERY_DIAGNOSTICS_READY",
        "teacher_bank_fingerprint": bank.fingerprint,
        "record_count": len(dev),
        "candidates": {},
    }
    for candidate in candidate_names:
        state = torch.load(
            paths["output_root"] / "probe_candidates" / candidate / "adapter.pt",
            map_location="cpu",
            weights_only=False,
        )
        load_tir_adapter_state_dict(wrapper, state)
        freeze_for_tir_adapter_warmup(wrapper)
        wrapper.eval()
        values = {layer: [] for layer in LAYER_NAMES}
        for record in tqdm(dev, desc=f"{candidate} recovery diagnostics"):
            encoded = encoder(wrapper, record)
            for layer in LAYER_NAMES:
                values[layer].append(encoded[layer].detach().cpu().half())
        result = build_recovery_diagnostics(
            records=dev,
            students={layer: torch.stack(items) for layer, items in values.items()},
            teachers=teachers,
            base_tir=base_tir,
            top_k=20,
        )
        candidate_path = diagnostics_root / f"{candidate}.json"
        _write_json(candidate_path, result)
        combined["candidates"][candidate] = {
            "path": str(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "sha256": _sha256(candidate_path),
            "layers": result["layers"],
        }
    _write_json(diagnostics_root / "summary.json", combined)


def _train_repair(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    wrapper: Qwen3VLRGBTAdapter,
    bank: RGBTeacherBank,
    base_hash: str,
    resume: bool,
) -> Any:
    paths = _paths(config, config_path)
    probe = read_jsonl_records(paths["manifest_root"] / "repair_probe_train.jsonl")
    dev = read_jsonl_records(paths["manifest_root"] / "repair_dev.jsonl")
    full = read_jsonl_records(paths["manifest_root"] / "repair_full_train.jsonl")
    initial = tir_adapter_state_dict(wrapper)
    encoder = _QwenRepairEncoder(
        wrapper=wrapper,
        processor=_processor(config),
        root=paths["rgbt_root"],
        roi_expansion=float(config["loss"]["roi_expansion"]),
    )

    def factory() -> torch.nn.Module:
        load_tir_adapter_state_dict(wrapper, initial)
        freeze_for_tir_adapter_warmup(wrapper)
        wrapper.eval()
        return wrapper

    recovery_mode = bool(config.get("recovery", {}).get("enabled", False))
    candidate_lookup = {
        "C0": CandidateSpec.c0(),
        "C1": CandidateSpec.c1(),
        "C2": CandidateSpec.c2(),
    }
    candidate_names = tuple(
        str(value).upper()
        for value in config.get("recovery", {}).get("candidates", ("C0", "C1", "C2"))
    )
    try:
        candidate_specs = tuple(candidate_lookup[name] for name in candidate_names)
    except KeyError as error:
        raise ValueError(f"unknown recovery candidate: {error.args[0]}") from error

    def persist_candidate(candidate: str, directory: Path) -> None:
        archive_root = paths["output_root"] / "recovery_archives"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive = archive_root / f"{candidate}.zip"
        temporary = archive.with_suffix(".zip.tmp")
        if temporary.exists():
            temporary.unlink()
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for source in sorted(directory.rglob("*")):
                if source.is_file():
                    handle.write(source, f"probe_candidates/{candidate}/{source.relative_to(directory).as_posix()}")
        temporary.replace(archive)
        _write_json(
            archive_root / f"{candidate}.json",
            {"candidate": candidate, "path": str(archive), "bytes": archive.stat().st_size, "sha256": _sha256(archive)},
        )

    trainer = Phase16RepairTrainer(
        adapter_factory=factory,
        encode_tir=encoder,
        output_root=paths["output_root"],
        config=_phase16_config(config),
        adapter_state=tir_adapter_state_dict,
        load_adapter_state=lambda module, state: load_tir_adapter_state_dict(module, state),
        resume=resume,
        candidate_specs=candidate_specs,
        stop_after_probe=recovery_mode,
        on_candidate_complete=persist_candidate if recovery_mode else None,
    )
    # Probes use one record per pair; only the winning objective is retrained
    # over the complete non-dev clean set from a fresh identical adapter.
    result = trainer.run(
        probe,
        dev,
        bank,
        full_train_records=full,
    )
    if result.status == "PHASE_16_PROBES_RECOVERED":
        _export_recovery_diagnostics(
            paths=paths,
            wrapper=wrapper,
            encoder=encoder,
            bank=bank,
            dev=dev,
            candidate_names=candidate_names,
        )
    if result.status == "PHASE_16_REPAIR_TRAINED":
        selected_payload = torch.load(
            paths["output_root"] / "selected_adapter" / "adapter.pt",
            map_location="cpu",
            weights_only=False,
        )
        release = {
            "status": "PHASE_16_REPAIR_READY_FOR_OFFICIAL_VAL",
            "model_id": config["model"]["model_id"],
            "model_revision": config["model"]["revision"],
            "tir_adapter_rank": int(config["model"]["tir_lora_rank"]),
            "selected_candidate": result.selected_candidate,
            "source_checkpoint": selected_payload["source_checkpoint"],
            "dev_metrics": selected_payload["dev_metrics"],
            "teacher_bank_fingerprint": bank.fingerprint,
            "repair_full_manifest": str(paths["manifest_root"] / "repair_full_train.jsonl"),
            "repair_full_record_count": len(full),
            "tir_adapter": selected_payload["adapter"],
        }
        release_path = (
            paths["output_root"]
            / "selected_adapter"
            / "qwen3vl8b_tir_rank48_adapter_phase16.pt"
        )
        torch.save(release, release_path)
        _write_json(
            paths["output_root"] / "selected_adapter" / "release_summary.json",
            {
                key: value
                for key, value in release.items()
                if key != "tir_adapter"
            }
            | {"path": str(release_path), "sha256": _sha256(release_path)},
        )
    if parameter_sha256(_base_parameters(wrapper)) != base_hash:
        raise RuntimeError("frozen RGB/base parameters changed during Phase 1.6")
    return result


def _official_validate(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    wrapper: Qwen3VLRGBTAdapter,
    base_hash: str,
    fingerprint: str,
) -> dict[str, Any]:
    paths = _paths(config, config_path)
    release_path = (
        paths["output_root"]
        / "selected_adapter"
        / "qwen3vl8b_tir_rank48_adapter_phase16.pt"
    )
    release = torch.load(release_path, map_location="cpu", weights_only=False)
    load_tir_adapter_state_dict(wrapper, release["tir_adapter"])
    wrapper.requires_grad_(False)
    wrapper.eval()
    records = read_jsonl_records(paths["manifest_root"] / "official_val.jsonl")
    processor = _processor(config)
    smoke_batch = processor.process(records[0], root=paths["rgbt_root"])
    safety = _phase16_safety_checks(
        config=config,
        wrapper=wrapper,
        release=release,
        base_hash=base_hash,
        smoke_batch=smoke_batch,
    )

    def encode(batch: Any):
        moved = _to_cuda(batch)
        return wrapper.encode_tir_validation_triplet(
            rgb_pixel_values=moved["rgb"],
            tir_pixel_values=moved["tir"],
            image_grid_thw=moved["grid"],
            ir_usable=batch.ir_usable,
        )

    gates = config["gates"]
    validator = Phase15Validator(
        processor=processor,
        encode_triplet=encode,
        output_root=paths["output_root"] / "official_val",
        config=Phase15Config(
            expected_record_count=int(config["expected"]["official_val_records"]),
            expected_pair_count=int(config["expected"]["official_val_pairs"]),
            seed=int(config["runtime"]["seed"]),
            bootstrap_iterations=int(config["runtime"]["bootstrap_iterations"]),
            cache_checkpoint_pairs=int(config["runtime"]["validation_checkpoint_pairs"]),
            spatial_merge_size=int(config["processor"]["spatial_merge_size"]),
            roi_expansion=float(config["loss"]["roi_expansion"]),
            hard_negative_weight=0.25,
            hard_negative_margin=float(config["loss"]["margin"]),
            minimum_alignment_improvement=float(gates["minimum_alignment_improvement"]),
            maximum_subgroup_degradation=float(gates["maximum_subgroup_degradation"]),
            subgroup_minimum_records=int(gates["subgroup_minimum_records"]),
            maximum_r5_drop=float(gates["maximum_r5_drop"]),
            minimum_retrieval_gain=float(gates["minimum_retrieval_gain"]),
            minimum_effective_rank_vs_base=float(gates["minimum_effective_rank_vs_base"]),
            minimum_effective_rank_vs_rgb=float(gates["minimum_effective_rank_vs_rgb"]),
            maximum_nonpaired_p95_increase=float(gates["maximum_nonpaired_p95_increase"]),
            fingerprint=fingerprint + "-OFFICIAL-VAL",
        ),
        safety_checks=lambda: safety,
    )
    result = validator.run(records, paths["rgbt_root"])
    retention = config["r5_retention"]
    retention_checks = {
        layer: result.retrieval_metrics[layer]["adapted"]["r_at_5"]
        >= float(threshold)
        for layer, threshold in retention.items()
    }
    status = (
        "PHASE_16_GO"
        if result.status == "PHASE_15_GO" and all(retention_checks.values())
        else "PHASE_16_NO_GO"
    )
    payload = {
        "status": status,
        "phase15_status": result.status,
        "retention_checks": retention_checks,
        "retention_thresholds": retention,
        "gate_failures": list(result.gate_failures),
        "release_sha256": _sha256(release_path),
        "claim_boundary": "official RGBT representation validation only; AIC Query/bbox/platform gain remains unverified",
    }
    _write_json(paths["output_root"] / "official_val" / "phase16_gate.json", payload)
    return payload


def _phase16_safety_checks(
    *,
    config: Mapping[str, Any],
    wrapper: Qwen3VLRGBTAdapter,
    release: Mapping[str, Any],
    base_hash: str,
    smoke_batch: Any,
) -> dict[str, Any]:
    release_state = release["tir_adapter"]
    checkpoint_path = Path(str(release["source_checkpoint"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"selected full checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint["adapter"]
    adapter_keys_equal = set(release_state) == set(checkpoint_state)
    tensor_mismatches = [
        name
        for name in sorted(set(release_state) & set(checkpoint_state))
        if not torch.equal(release_state[name], checkpoint_state[name])
    ]
    moved = _to_cuda(smoke_batch)
    valid_mask = smoke_batch.ir_patch_valid_mask.to("cuda:0")
    zero_mask = torch.zeros_like(valid_mask)
    with torch.inference_mode():
        rgb_only = wrapper.encode_visual_pair(
            rgb_pixel_values=moved["rgb"], image_grid_thw=moved["grid"]
        )
        gate_zero = wrapper.encode_visual_pair(
            rgb_pixel_values=moved["rgb"],
            tir_pixel_values=moved["tir"],
            image_grid_thw=moved["grid"],
            ir_patch_valid_mask=valid_mask,
        )
        tir_none = wrapper.encode_visual_pair(
            rgb_pixel_values=moved["rgb"],
            tir_pixel_values=None,
            image_grid_thw=moved["grid"],
        )
        unusable = wrapper.encode_visual_pair(
            rgb_pixel_values=moved["rgb"],
            tir_pixel_values=moved["tir"],
            image_grid_thw=moved["grid"],
            ir_patch_valid_mask=valid_mask,
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
        "tir_none": max_diff(tir_none.final_hidden, rgb_only.final_hidden),
        "ir_usable_false": max_diff(unusable.final_hidden, rgb_only.final_hidden),
        "all_zero_ir_mask": max_diff(zero_ir.final_hidden, rgb_only.final_hidden),
    }
    base_after = parameter_sha256(_base_parameters(wrapper))
    source = inspect.getsource(
        __import__("aic_rgbtir.modeling", fromlist=["*"])
    ).lower()
    expected_count = int(config["expected"]["adapter_tensor_count"])
    checks = {
        "adapter_key_set_exact": adapter_keys_equal
        and len(release_state) == expected_count,
        "adapter_tensor_exact": not tensor_mismatches,
        "gate_zero_rgb_equivalent": diffs["gate_zero"] <= threshold,
        "tir_none_rgb_equivalent": diffs["tir_none"] <= threshold,
        "ir_usable_false_rgb_equivalent": diffs["ir_usable_false"] <= threshold,
        "all_zero_ir_mask_rgb_equivalent": diffs["all_zero_ir_mask"] <= threshold,
        "rgb_base_hash_unchanged": base_hash == base_after,
        "all_outputs_finite": all(
            torch.isfinite(output.final_hidden).all().item()
            for output in (rgb_only, gate_zero, tir_none, unusable, zero_ir)
        ),
        "no_second_model_fallback": "fallback_model" not in source
        and "second_model" not in source,
        "no_trainable_parameters": not any(
            parameter.requires_grad for parameter in wrapper.parameters()
        ),
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


def _report(config: Mapping[str, Any], config_path: Path, gate: Mapping[str, Any]) -> None:
    paths = _paths(config, config_path)
    run_summary = json.loads(
        (paths["output_root"] / "run_summary.json").read_text(encoding="utf-8")
    )
    lines = [
        "# AIC RGB–TIR Phase 1.6 去坍缩修复验证",
        "",
        f"**最终状态：`{gate['status']}`**",
        "",
        "## 受控实验",
        "",
        "- C0：原始配对对齐与同图背景 margin。",
        "- C1：C0 加跨图 InfoNCE。",
        "- C2：C1 加 RGB teacher 关系结构蒸馏。",
        f"- 选中候选：`{run_summary.get('selected_candidate') or 'none'}`。",
        "- RGB/Qwen 主路径冻结；无 Query、bbox、融合、Depth 或 AIC 测试集训练。",
        "",
        "## Official val 门禁",
        "",
        f"- Phase 1.5 兼容门禁：`{gate['phase15_status']}`。",
    ]
    for layer, passed in gate["retention_checks"].items():
        lines.append(
            f"- Layer {layer} 原 Phase 1 R@5 增益保留：{'PASS' if passed else 'FAIL'}。"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"- Phase 2：{'允许启动' if gate['status'] == 'PHASE_16_GO' else '禁止启动'}。",
            "- 本阶段不能证明 AIC Query grounding、bbox ACC 或平台分数提升。",
            "- Phase 2 首次平台门槛：8B RGB–TIR 至少达到 0.7632。",
            "",
        ]
    )
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text("\n".join(lines), encoding="utf-8")


def _sha_manifest(output_root: Path) -> None:
    rows = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            rows[path.relative_to(output_root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _write_json(output_root / "sha256_manifest.json", {"schema_version": 1, "outputs": rows})


def run(
    command: str,
    *,
    config_path: Path,
    resume: bool,
    allow_network: bool,
) -> int:
    config = _load_yaml(config_path)
    paths = _paths(config, config_path)
    recovery_mode = bool(config.get("recovery", {}).get("enabled", False))
    if recovery_mode and command == "all":
        raise RuntimeError(
            "PHASE_16_RECOVERY_SCOPE_VIOLATION: 'all' would enter official validation; use 'train'"
        )
    if recovery_mode:
        recovery = config["recovery"]
        if not bool(recovery.get("forbid_full_training", False)) or not bool(
            recovery.get("forbid_official_val", False)
        ):
            raise RuntimeError("PHASE_16_RECOVERY_GUARDS_NOT_ENABLED")
    if command == "build-manifests":
        print(json.dumps(_build_manifests(config, config_path), ensure_ascii=False, indent=2))
        return 0
    preflight = _preflight(config, config_path)
    if command == "preflight":
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    _require_cuda(float(config["runtime"]["minimum_vram_gib"]))
    assets = _model_assets(config, config_path, allow_network=allow_network)
    fingerprint = _fingerprint(config, preflight, assets)
    wrapper, base_hash = _build_model(config, assets)
    bank = _build_teacher_bank(
        config=config,
        config_path=config_path,
        wrapper=wrapper,
        fingerprint=fingerprint,
        resume=resume,
    )
    if command == "build-teacher-bank":
        return 0
    result = _train_repair(
        config=config,
        config_path=config_path,
        wrapper=wrapper,
        bank=bank,
        base_hash=base_hash,
        resume=resume,
    )
    if result.status == "PHASE_16_NO_GO":
        _sha_manifest(paths["output_root"])
        return 2
    if command == "train":
        _sha_manifest(paths["output_root"])
        return 0
    gate = _official_validate(
        config=config,
        config_path=config_path,
        wrapper=wrapper,
        base_hash=base_hash,
        fingerprint=fingerprint,
    )
    _report(config, config_path, gate)
    _sha_manifest(paths["output_root"])
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0 if gate["status"] == "PHASE_16_GO" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="AIC RGB-TIR Phase 1.6 repair runner")
    parser.add_argument(
        "command",
        choices=("build-manifests", "preflight", "build-teacher-bank", "train", "all"),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(20260812)
    return run(
        args.command,
        config_path=args.config.resolve(),
        resume=args.resume,
        allow_network=args.allow_network,
    )


if __name__ == "__main__":
    raise SystemExit(main())
