from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for value in (REPO_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from aic_rgbtir.data import RGBTRecord, read_jsonl_records  # noqa: E402
from aic_rgbtir.phase1 import (  # noqa: E402
    freeze_for_rgbtir_inference,
    freeze_for_tir_adapter_warmup,
    load_tir_adapter_state_dict,
    parameter_sha256,
    tir_adapter_state_dict,
)
from aic_rgbtir.phase15 import Phase15Config, Phase15Validator  # noqa: E402
from aic_rgbtir.phase16 import Phase16Config, RGBTeacherBank  # noqa: E402
from aic_rgbtir.phase18 import (  # noqa: E402
    D1CandidateSpec,
    D1RetentionObjective,
    Phase18Config,
    _D1Trainer,
)
from aic_rgbtir.phase18_full import (  # noqa: E402
    CheckpointArtifact,
    Phase18FullConfig,
    Phase18FullRunner,
    Phase18TrainSelectionResult,
)
from tools.run_rgbtir_phase16 import (  # noqa: E402
    _QwenRepairEncoder,
    _base_parameters,
    _build_model,
    _build_teacher_bank,
    _fingerprint,
    _model_assets,
    _paths,
    _phase16_config,
    _phase16_safety_checks,
    _preflight,
    _processor,
    _require_cuda,
    _sha256,
    _to_cuda,
    _write_json,
)


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Phase 1.8 Full config must be a mapping")
    return payload


def _total_host_memory_gib() -> float:
    if os.name == "nt":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return float(status.ullTotalPhys) / (1024**3)
    pages = int(os.sysconf("SC_PHYS_PAGES"))
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    return float(pages * page_size) / (1024**3)


def _require_host_memory(minimum_gib: float) -> float:
    total = _total_host_memory_gib()
    if total + 1e-6 < float(minimum_gib):
        raise RuntimeError(
            f"PHASE_18_HOST_RAM_NO_GO: total={total:.2f}GiB < required={minimum_gib:.2f}GiB"
        )
    return total


def _scope_guard(config: Mapping[str, Any]) -> None:
    expected = {
        "full_training": True,
        "sealed_dev_selection": True,
        "one_shot_official_val": True,
        "forbid_query_training": True,
        "forbid_fusion": True,
        "forbid_depth": True,
        "forbid_aic_test": True,
    }
    if dict(config.get("scope", {})) != expected:
        raise RuntimeError("PHASE_18_FULL_SCOPE_GUARD_MISMATCH")
    if dict(config.get("candidate", {})) != {
        "name": "D1_L050",
        "retention_lambda": 0.50,
    }:
        raise RuntimeError("PHASE_18_FULL_CANDIDATE_DRIFT")
    if [float(value) for value in config["runtime"]["checkpoint_fractions"]] != [
        0.25,
        0.50,
        0.75,
        1.0,
    ]:
        raise RuntimeError("PHASE_18_FULL_CHECKPOINT_DRIFT")


def _records(config: Mapping[str, Any], config_path: Path) -> tuple[list[RGBTRecord], ...]:
    root = _paths(config, config_path)["manifest_root"]
    return tuple(
        read_jsonl_records(root / name)
        for name in ("repair_full_train.jsonl", "repair_dev.jsonl", "official_val.jsonl")
    )


def _full_config(config: Mapping[str, Any]) -> Phase18FullConfig:
    gates = config["gates"]
    reference = config["c2_reference"]
    d1 = config["d1_probe_reference"]
    expected = config["expected"]
    return Phase18FullConfig(
        checkpoint_fractions=tuple(float(x) for x in config["runtime"]["checkpoint_fractions"]),
        expected_full_records=int(expected["full_records"]),
        expected_dev_records=int(expected["dev_records"]),
        expected_official_records=int(expected["official_val_records"]),
        expected_official_pairs=int(expected["official_val_pairs"]),
        c2_layer_drift={str(k): float(v) for k, v in reference["layer_drift"].items()},
        d1_probe_r1=float(d1["mean_r_at_1"]),
        d1_probe_r5=float(d1["mean_r_at_5"]),
        maximum_r1_drop=float(gates["maximum_r1_drop"]),
        maximum_r5_drop=float(gates["maximum_r5_drop"]),
        minimum_layer24_drift_reduction=float(gates["minimum_layer24_drift_reduction"]),
        minimum_effective_rank_vs_base=float(gates["minimum_effective_rank_vs_base"]),
        minimum_effective_rank_vs_teacher=float(gates["minimum_effective_rank_vs_rgb"]),
        maximum_nonpaired_p95_increase=float(gates["maximum_nonpaired_p95_increase"]),
        maximum_subgroup_degradation=float(gates["maximum_subgroup_degradation"]),
    )


def _phase18_config(config: Mapping[str, Any]) -> Phase18Config:
    gates = config["gates"]
    reference = config["c2_reference"]
    d1 = config["d1_probe_reference"]
    return Phase18Config(
        seed=int(config["runtime"]["seed"]),
        c2_layer_drift={str(k): float(v) for k, v in reference["layer_drift"].items()},
        c2_mean_r_at_1=float(d1["mean_r_at_1"]),
        c2_mean_r_at_5=float(d1["mean_r_at_5"]),
        maximum_r1_drop=float(gates["maximum_r1_drop"]),
        maximum_r5_drop=float(gates["maximum_r5_drop"]),
        minimum_layer24_drift_reduction=float(gates["minimum_layer24_drift_reduction"]),
        minimum_effective_rank_vs_base=float(gates["minimum_effective_rank_vs_base"]),
        minimum_effective_rank_vs_teacher=float(gates["minimum_effective_rank_vs_rgb"]),
        maximum_nonpaired_cosine_p95_increase=float(gates["maximum_nonpaired_p95_increase"]),
        retention_tolerance={str(k): float(v) for k, v in config["retention"]["tolerance"].items()},
    )


def _preflight_full(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    _scope_guard(config)
    shared = _preflight(config, config_path)
    full, dev, official = _records(config, config_path)
    runner = Phase18FullRunner(
        backend=None,  # type: ignore[arg-type]
        output_root=_paths(config, config_path)["output_root"],
        config=_full_config(config),
    )
    runner._validate_records(full, dev, official)
    payload = {
        "status": "PHASE_18_FULL_PREFLIGHT_GO",
        "scope": config["scope"],
        "candidate": config["candidate"],
        "counts": {"full": len(full), "dev": len(dev), "official": len(official)},
        "shared_preflight": shared,
        "claim_boundary": (
            "D1_L050 TIR representation full training only; no Query training, fusion, "
            "Depth, AIC test inference, fallback or submission"
        ),
    }
    _write_json(_paths(config, config_path)["output_root"] / "full_preflight.json", payload)
    return payload


class _QwenFullBackend:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        config_path: Path,
        wrapper: torch.nn.Module,
        encoder: _QwenRepairEncoder,
        bank: RGBTeacherBank,
        initial: Mapping[str, torch.Tensor],
        base_hash: str,
        fingerprint: str,
        resume: bool,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.wrapper = wrapper
        self.encoder = encoder
        self.bank = bank
        self.initial = {k: v.detach().cpu().clone() for k, v in initial.items()}
        self.base_hash = base_hash
        self.fingerprint = fingerprint
        self.resume = bool(resume)
        self.selected_checkpoint: Path | None = None
        phase16: Phase16Config = _phase16_config(config)
        self.trainer = _D1Trainer(
            phase18_config=_phase18_config(config),
            adapter_factory=lambda: wrapper,
            encode_tir=encoder,
            output_root=_paths(config, config_path)["output_root"],
            config=phase16,
            adapter_state=tir_adapter_state_dict,
            load_adapter_state=lambda module, state: load_tir_adapter_state_dict(module, state),
            resume=self.resume,
            candidate_specs=(D1CandidateSpec.d1(0.50),),
            objective_factory=lambda candidate: D1RetentionObjective(
                candidate,  # type: ignore[arg-type]
                temperature=phase16.temperature,
                margin=phase16.margin,
                retention_tolerance=_phase18_config(config).retention_tolerance,
            ),
            stop_after_probe=False,
        )

    def fresh_adapter(self) -> torch.nn.Module:
        load_tir_adapter_state_dict(self.wrapper, self.initial)
        freeze_for_tir_adapter_warmup(self.wrapper)
        self.wrapper.eval()
        return self.wrapper

    def train_full(
        self,
        module: torch.nn.Module,
        records: Sequence[RGBTRecord],
        checkpoint_dir: Path,
        fractions: Sequence[float],
    ) -> Sequence[CheckpointArtifact]:
        self.trainer._train(
            module,
            D1CandidateSpec.d1(0.50),
            records,
            self.bank,
            steps=len(records),
            checkpoint_dir=checkpoint_dir,
        )
        result = []
        for path in sorted(checkpoint_dir.glob("checkpoint_*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            step = int(payload["step"])
            result.append(CheckpointArtifact(step=step, fraction=step / len(records), path=path))
        return result

    def load_checkpoint(self, module: torch.nn.Module, checkpoint: CheckpointArtifact) -> None:
        payload = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
        if payload.get("teacher_bank_fingerprint") != self.bank.fingerprint:
            raise RuntimeError("PHASE_18_FULL_CHECKPOINT_BANK_MISMATCH")
        load_tir_adapter_state_dict(module, payload["adapter"])
        freeze_for_tir_adapter_warmup(module)
        module.eval()
        self.selected_checkpoint = checkpoint.path

    def evaluate(self, module: torch.nn.Module, records: Sequence[RGBTRecord]) -> Mapping[str, Any]:
        return self.trainer._evaluate(module, records, self.bank)

    def release(
        self,
        module: torch.nn.Module,
        checkpoint: CheckpointArtifact,
        output_path: Path,
        metadata: Mapping[str, Any],
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "status": "PHASE_18_FULL_SELECTED_ADAPTER",
                "source_checkpoint": str(checkpoint.path),
                "teacher_bank_fingerprint": self.bank.fingerprint,
                "metadata": dict(metadata),
                "tir_adapter": tir_adapter_state_dict(module),
            },
            output_path,
        )

    def validate_official(
        self, module: torch.nn.Module, records: Sequence[RGBTRecord]
    ) -> Mapping[str, Any]:
        if self.selected_checkpoint is None:
            raise RuntimeError("official val requires a dev-selected checkpoint")
        return _validate_official_model(
            config=self.config,
            config_path=self.config_path,
            module=module,
            records=records,
            base_hash=self.base_hash,
            fingerprint=self.fingerprint,
            selected_checkpoint=str(self.selected_checkpoint),
        )


def _validate_official_model(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    module: torch.nn.Module,
    records: Sequence[RGBTRecord],
    base_hash: str,
    fingerprint: str,
    selected_checkpoint: str,
) -> Mapping[str, Any]:
    """Run sealed official validation without retaining the training backend."""
    paths = _paths(config, config_path)
    processor = _processor(config)
    release = {
        "source_checkpoint": selected_checkpoint,
        "tir_adapter": tir_adapter_state_dict(module),
    }
    freeze_for_rgbtir_inference(module)
    safety = _phase16_safety_checks(
        config=config,
        wrapper=module,  # type: ignore[arg-type]
        release=release,
        base_hash=base_hash,
        smoke_batch=processor.process(records[0], root=paths["rgbt_root"]),
    )

    def encode(batch: Any):
        moved = _to_cuda(batch)
        return module.encode_tir_validation_triplet(  # type: ignore[attr-defined]
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
            maximum_subgroup_degradation=float(gates["maximum_subgroup_degradation"]),
            subgroup_minimum_records=int(gates["subgroup_minimum_records"]),
            maximum_nonpaired_p95_increase=float(gates["maximum_nonpaired_p95_increase"]),
            fingerprint=fingerprint + "-OFFICIAL",
        ),
        safety_checks=lambda: safety,
    )
    result = validator.run(records, paths["rgbt_root"])
    layers = ("8", "16", "24")
    retrieval = result.retrieval_metrics
    collapse = result.collapse_metrics
    layer_drift = {
        layer: float(retrieval[layer]["base"]["paired_cosine"])
        - float(retrieval[layer]["adapted"]["paired_cosine"])
        for layer in layers
    }
    subgroup_degradation = max(
        [
            max(0.0, -float(row["relative_improvement"]))
            for row in result.subgroup_metrics
            if int(row["record_count"]) >= int(gates["subgroup_minimum_records"])
        ]
        or [0.0]
    )
    return {
        "completed_records": result.completed_record_count,
        "completed_pairs": result.completed_pair_count,
        "skipped_records": result.skipped_record_count,
        "mean_r_at_1": sum(float(retrieval[x]["adapted"]["r_at_1"]) for x in layers) / 3,
        "mean_r_at_5": sum(float(retrieval[x]["adapted"]["r_at_5"]) for x in layers) / 3,
        "layer_r_at_5": {x: float(retrieval[x]["adapted"]["r_at_5"]) for x in layers},
        "layer_drift": layer_drift,
        "minimum_effective_rank_ratio_vs_base": min(
            float(collapse[x]["adapted"]["effective_rank"])
            / max(float(collapse[x]["base"]["effective_rank"]), 1e-8)
            for x in layers
        ),
        "minimum_effective_rank_ratio_vs_teacher": min(
            float(collapse[x]["adapted"]["effective_rank"])
            / max(float(collapse[x]["rgb"]["effective_rank"]), 1e-8)
            for x in layers
        ),
        "maximum_nonpaired_cosine_p95_increase": max(
            float(collapse[x]["adapted"]["nonpaired_cosine_p95"])
            - float(collapse[x]["base"]["nonpaired_cosine_p95"])
            for x in layers
        ),
        "minimum_paired_shuffled_margin": min(
            float(retrieval[x]["adapted"]["paired_shuffled_margin"]) for x in layers
        ),
        "paired_margin_bootstrap_lower": min(
            float(retrieval[x]["adapted"]["paired_shuffled_margin_ci95"][0]) for x in layers
        ),
        "maximum_subgroup_degradation": subgroup_degradation,
        "safety_passed": bool(safety["passed"]),
        "phase15_status": result.status,
    }


def _sha_manifest(root: Path) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            files[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _write_json(root / "sha256_manifest.json", {"schema_version": 1, "files": files})


def run(command: str, *, config_path: Path, resume: bool, allow_network: bool) -> int:
    config = _load(config_path)
    preflight = _preflight_full(config, config_path)
    if command == "preflight":
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    hardware = _require_cuda(float(config["runtime"]["minimum_vram_gib"]))
    host_ram_gib = _require_host_memory(
        float(config["runtime"].get("minimum_host_ram_gib", 0.0))
    )
    assets = _model_assets(config, config_path, allow_network=allow_network)
    wrapper, base_hash = _build_model(config, assets)
    initial = tir_adapter_state_dict(wrapper)
    fingerprint = _fingerprint(config, preflight["shared_preflight"], assets)
    fingerprint = hashlib.sha256(
        (fingerprint + _sha256(Path(__file__)) + _sha256(config_path)).encode("ascii")
    ).hexdigest().upper()
    paths = _paths(config, config_path)
    full, dev, official = _records(config, config_path)

    if command == "official-val":
        selection_path = paths["output_root"] / "train_selection_summary.json"
        if not selection_path.is_file():
            raise RuntimeError("PHASE_18_TRAIN_SELECTION_MISSING")
        selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
        selection = Phase18TrainSelectionResult(**selection_payload)
        if selection.fingerprint != fingerprint:
            raise RuntimeError("PHASE_18_TRAIN_SELECTION_FINGERPRINT_MISMATCH")
        adapter_path = Path(selection.selected_adapter)
        if not adapter_path.is_file():
            raise RuntimeError("PHASE_18_SELECTED_ADAPTER_MISSING")
        release = torch.load(adapter_path, map_location="cpu", weights_only=False)
        metadata = dict(release.get("metadata", {}))
        if metadata.get("fingerprint") != fingerprint:
            raise RuntimeError("PHASE_18_SELECTED_ADAPTER_FINGERPRINT_MISMATCH")
        load_tir_adapter_state_dict(wrapper, release["tir_adapter"])
        freeze_for_tir_adapter_warmup(wrapper)
        wrapper.eval()
        official_metrics = _validate_official_model(
            config=config,
            config_path=config_path,
            module=wrapper,
            records=official,
            base_hash=base_hash,
            fingerprint=fingerprint,
            selected_checkpoint=selection.selected_checkpoint,
        )
        result = Phase18FullRunner(
            backend=None,  # type: ignore[arg-type]
            output_root=paths["output_root"],
            config=_full_config(config),
        ).complete_official(selection, official_metrics)
        if parameter_sha256(_base_parameters(wrapper)) != base_hash:
            raise RuntimeError("PHASE_18_FULL_FROZEN_BASE_CHANGED")
        _write_json(
            paths["output_root"] / "environment_official_val.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "hardware": hardware,
                "host_ram_gib": host_ram_gib,
                "model_revision": config["model"]["revision"],
                "fingerprint": fingerprint,
                "stage": "official-val",
                "second_model_fallback": False,
            },
        )
        _sha_manifest(paths["output_root"])
        print(json.dumps({"status": result.status, "selected": result.selected_checkpoint}, indent=2))
        return 0 if result.status == "PHASE_18_FULL_GO" else 2

    bank = _build_teacher_bank(
        config=config,
        config_path=config_path,
        wrapper=wrapper,
        fingerprint=fingerprint,
        resume=resume,
    )
    encoder = _QwenRepairEncoder(
        wrapper=wrapper,
        processor=_processor(config),
        root=paths["rgbt_root"],
        roi_expansion=float(config["loss"]["roi_expansion"]),
    )
    backend = _QwenFullBackend(
        config=config,
        config_path=config_path,
        wrapper=wrapper,
        encoder=encoder,
        bank=bank,
        initial=initial,
        base_hash=base_hash,
        fingerprint=fingerprint,
        resume=resume,
    )
    runner = Phase18FullRunner(
        backend=backend,
        output_root=paths["output_root"],
        config=_full_config(config),
    )
    if command == "train-select":
        selection = runner.train_select(full, dev, official, fingerprint=fingerprint)
        if parameter_sha256(_base_parameters(wrapper)) != base_hash:
            raise RuntimeError("PHASE_18_FULL_FROZEN_BASE_CHANGED")
        _write_json(
            paths["output_root"] / "environment_train_select.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "hardware": hardware,
                "host_ram_gib": host_ram_gib,
                "model_revision": config["model"]["revision"],
                "fingerprint": fingerprint,
                "stage": "train-select",
                "second_model_fallback": False,
            },
        )
        _sha_manifest(paths["output_root"])
        print(
            json.dumps(
                {"status": selection.status, "selected": selection.selected_checkpoint},
                indent=2,
            )
        )
        return 0 if selection.status == "PHASE_18_TRAIN_SELECT_READY" else 2

    result = runner.run(full, dev, official, fingerprint=fingerprint)
    if parameter_sha256(_base_parameters(wrapper)) != base_hash:
        raise RuntimeError("PHASE_18_FULL_FROZEN_BASE_CHANGED")
    _write_json(
        paths["output_root"] / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "hardware": hardware,
            "host_ram_gib": host_ram_gib,
            "model_revision": config["model"]["revision"],
            "fingerprint": fingerprint,
            "second_model_fallback": False,
        },
    )
    _sha_manifest(paths["output_root"])
    print(json.dumps({"status": result.status, "selected": result.selected_checkpoint}, indent=2))
    return 0 if result.status == "PHASE_18_FULL_GO" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="AIC RGB-TIR Phase 1.8 full D1 training")
    parser.add_argument(
        "command", choices=("preflight", "train-select", "official-val", "run")
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
