from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for value in (REPO_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from aic_rgbtir.data import RGBTRecord, read_jsonl_records  # noqa: E402
from aic_rgbtir.phase1 import (  # noqa: E402
    freeze_for_tir_adapter_warmup,
    load_tir_adapter_state_dict,
    parameter_sha256,
    tir_adapter_state_dict,
)
from aic_rgbtir.phase16 import Phase16Config, RGBTeacherBank  # noqa: E402
from aic_rgbtir.phase18_full import CheckpointArtifact  # noqa: E402
from aic_rgbtir.phase19_d2 import (  # noqa: E402
    D2CandidateSpec,
    D2RepairTrainer,
)
from aic_rgbtir.phase19_d2_full import (  # noqa: E402
    D2FullConfig,
    D2FullSelectionResult,
    Phase19D2FullRunner,
)
from tools.run_rgbtir_phase16 import (  # noqa: E402
    _QwenRepairEncoder,
    _base_parameters,
    _build_model,
    _model_assets,
    _processor,
    _require_cuda,
    _sha256,
    _write_json,
)
from tools.run_rgbtir_phase18_full import (  # noqa: E402
    _require_host_memory,
    _sha_manifest,
    _validate_official_model,
)
from tools.run_rgbtir_phase19_d2 import (  # noqa: E402
    _load_config,
    _mapping_hash,
    _resolve,
)


MANIFEST_FILES = {
    "full": "d2_remaining_full_train.jsonl",
    "semantic": "d2_semantic_dev.jsonl",
    "multiquery": "d2_multiquery_dev.jsonl",
    "official_val": "official_val.jsonl",
}


def _scope_guard(config: Mapping[str, Any]) -> None:
    expected_scope = {
        "full_training": True,
        "sealed_dual_dev_selection": True,
        "one_shot_official_val": True,
        "forbid_query_training": True,
        "forbid_fusion": True,
        "forbid_depth": True,
        "forbid_aic_test": True,
        "forbid_fallback": True,
    }
    if dict(config.get("scope", {})) != expected_scope:
        raise RuntimeError("PHASE_19_D2_FULL_SCOPE_DRIFT")
    if dict(config.get("candidate", {})) != {
        "name": "D2_G025",
        "geometry_lambda": 0.25,
        "initialize_from": "D1_L050",
        "continue_from_probe_adapter": False,
    }:
        raise RuntimeError("PHASE_19_D2_FULL_CANDIDATE_DRIFT")
    fractions = [float(value) for value in config["runtime"]["checkpoint_fractions"]]
    if fractions != [0.25, 0.50, 0.75, 1.0]:
        raise RuntimeError("PHASE_19_D2_FULL_CHECKPOINT_DRIFT")


def _paths(config: Mapping[str, Any], config_path: Path) -> dict[str, Path]:
    return {
        name: _resolve(config_path, str(value))
        for name, value in config["paths"].items()
    }


def _records(
    config: Mapping[str, Any], config_path: Path
) -> tuple[list[RGBTRecord], list[RGBTRecord], list[RGBTRecord], list[RGBTRecord]]:
    root = _paths(config, config_path)["manifest_root"]
    return tuple(  # type: ignore[return-value]
        read_jsonl_records(root / MANIFEST_FILES[name])
        for name in ("full", "semantic", "multiquery", "official_val")
    )


def _phase16_config(config: Mapping[str, Any]) -> Phase16Config:
    runtime = config["runtime"]
    loss = config["loss"]
    optimization = config["optimization"]
    return Phase16Config(
        seed=int(runtime["seed"]),
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        warmup_fraction=float(optimization["warmup_fraction"]),
        gradient_accumulation=int(optimization["gradient_accumulation"]),
        max_grad_norm=float(optimization["max_grad_norm"]),
        temperature=float(loss["temperature"]),
        margin=float(loss["margin"]),
        probe_steps=0,
        full_steps=int(runtime["full_steps"]),
        negative_count=int(loss["negative_count"]),
        same_source_negatives=int(loss["same_source_negatives"]),
        same_condition_negatives=int(loss["same_condition_negatives"]),
        relational_anchor_count=int(loss["relational_anchor_count"]),
        checkpoint_fractions=tuple(float(x) for x in runtime["checkpoint_fractions"]),
    )


def _full_config(config: Mapping[str, Any]) -> D2FullConfig:
    gates = config["gates"]
    expected = config["expected"]
    return D2FullConfig(
        checkpoint_fractions=tuple(
            float(value) for value in config["runtime"]["checkpoint_fractions"]
        ),
        expected_full_records=int(expected["full_records"]),
        expected_semantic_records=int(expected["semantic_records"]),
        expected_semantic_pairs=int(expected["semantic_pairs"]),
        expected_multiquery_records=int(expected["multiquery_records"]),
        expected_multiquery_pairs=int(expected["multiquery_pairs"]),
        expected_official_records=int(expected["official_val_records"]),
        expected_official_pairs=int(expected["official_val_pairs"]),
        minimum_dev_rank_vs_base=float(gates["minimum_dev_rank_vs_base"]),
        minimum_rank_gain_each_dev=float(gates["minimum_rank_gain_each_dev"]),
        maximum_semantic_r5_drop=float(gates["maximum_semantic_r5_drop"]),
        maximum_multiquery_r5_drop=float(gates["maximum_multiquery_r5_drop"]),
        maximum_nonpaired_p95_delta=float(gates["maximum_nonpaired_p95_delta"]),
        maximum_official_absolute_drift=float(
            gates["maximum_official_absolute_drift"]
        ),
        minimum_official_rank_vs_base=float(gates["minimum_effective_rank_vs_base"]),
        minimum_official_rank_vs_teacher=float(gates["minimum_effective_rank_vs_rgb"]),
        maximum_official_nonpaired_p95_increase=float(
            gates["maximum_nonpaired_p95_increase"]
        ),
        maximum_subgroup_degradation=float(gates["maximum_subgroup_degradation"]),
        official_layer_r5_baseline={
            str(key): float(value)
            for key, value in gates["official_layer_r5_baseline"].items()
        },
        official_layer_r5_maximum_drop={
            str(key): float(value)
            for key, value in gates["official_layer_r5_maximum_drop"].items()
        },
    )


def _load_probe_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PHASE_19_D2_PROBE_GO":
        raise RuntimeError("PHASE_19_D2_PROBE_EVIDENCE_NOT_GO")
    if payload.get("selected_candidate") != "D2_G025":
        raise RuntimeError("PHASE_19_D2_PROBE_WINNER_DRIFT")
    baseline = payload.get("baseline_metrics")
    if not isinstance(baseline, Mapping) or set(baseline) != {"semantic", "multiquery"}:
        raise RuntimeError("PHASE_19_D2_PROBE_BASELINE_MISSING")
    return payload


def _preflight(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    _scope_guard(config)
    paths = _paths(config, config_path)
    expected = config["expected"]
    if not (paths["rgbt_root"] / "image_data").is_dir():
        raise FileNotFoundError(paths["rgbt_root"] / "image_data")
    records = _records(config, config_path)
    names = ("full", "semantic", "multiquery", "official_val")
    counts: dict[str, dict[str, int]] = {}
    for name, rows in zip(names, records):
        path = paths["manifest_root"] / MANIFEST_FILES[name]
        counts[name] = {
            "records": len(rows),
            "pairs": len({row.image_pair_key for row in rows}),
        }
        if counts[name]["records"] != int(expected[f"{name}_records"]):
            raise RuntimeError(f"PHASE_19_D2_FULL_{name.upper()}_RECORD_DRIFT")
        if counts[name]["pairs"] != int(expected[f"{name}_pairs"]):
            raise RuntimeError(f"PHASE_19_D2_FULL_{name.upper()}_PAIR_DRIFT")
        if _sha256(path) != str(expected[f"{name}_sha256"]).upper():
            raise RuntimeError(f"PHASE_19_D2_FULL_{name.upper()}_SHA_DRIFT")
    probe = _load_probe_evidence(paths["probe_summary"])
    for name in ("teacher_bank", "d1_adapter", "probe_summary"):
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != str(expected[f"{name}_sha256"]).upper():
            raise RuntimeError(f"PHASE_19_D2_FULL_{name.upper()}_SHA_DRIFT")
    adapter_payload = torch.load(paths["d1_adapter"], map_location="cpu", weights_only=False)
    adapter = adapter_payload.get("tir_adapter")
    if not isinstance(adapter, Mapping) or len(adapter) != int(expected["adapter_tensors"]):
        raise RuntimeError("PHASE_19_D2_FULL_D1_ADAPTER_CONTRACT_FAILED")
    if adapter_payload.get("teacher_bank_fingerprint") != expected["teacher_bank_fingerprint"]:
        raise RuntimeError("PHASE_19_D2_FULL_ADAPTER_BANK_FINGERPRINT_DRIFT")
    bank = RGBTeacherBank.load(
        paths["teacher_bank"],
        expected_fingerprint=str(expected["teacher_bank_fingerprint"]),
    )
    required = {record.record_id for rows in records for record in rows}
    missing = required - set(bank.record_ids)
    if missing:
        raise RuntimeError(f"PHASE_19_D2_FULL_BANK_MISSING: {sorted(missing)[:3]}")
    runner = Phase19D2FullRunner(
        backend=None,  # type: ignore[arg-type]
        output_root=paths["output_root"],
        baseline_metrics=probe["baseline_metrics"],
        config=_full_config(config),
    )
    runner.validate_records(*records)
    payload = {
        "status": "PHASE_19_D2_FULL_PREFLIGHT_GO",
        "scope": config["scope"],
        "candidate": config["candidate"],
        "counts": counts,
        "teacher_bank_records": len(bank.record_ids),
        "teacher_bank_fingerprint": bank.fingerprint,
        "probe_status": probe["status"],
        "probe_selected_candidate": probe["selected_candidate"],
        "claim_boundary": (
            "D2_G025 TIR representation full training and sealed official val only; "
            "no Query, fusion, Depth, AIC test, fallback, submission or platform claim"
        ),
    }
    _write_json(paths["output_root"] / "full_preflight.json", payload)
    return payload


def _fingerprint(
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    assets: Mapping[str, Path],
    config_path: Path,
) -> str:
    payload = {
        "scope": config["scope"],
        "candidate": config["candidate"],
        "model": config["model"],
        "processor": config["processor"],
        "loss": config["loss"],
        "optimization": config["optimization"],
        "counts": preflight["counts"],
        "teacher_bank_fingerprint": preflight["teacher_bank_fingerprint"],
        "assets": {name: _sha256(path) for name, path in sorted(assets.items())},
        "sources": {
            path.name: _sha256(path)
            for path in (
                REPO_ROOT / "src/aic_rgbtir/phase19_d2.py",
                REPO_ROOT / "src/aic_rgbtir/phase19_d2_full.py",
                Path(__file__),
                config_path,
            )
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


class _D2FullBackend:
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
        self.initial = {key: value.detach().cpu().clone() for key, value in initial.items()}
        self.base_hash = base_hash
        self.fingerprint = fingerprint
        self.selected_checkpoint: Path | None = None
        phase16 = _phase16_config(config)
        self.candidate = D2CandidateSpec.d2(0.25)
        self.trainer = D2RepairTrainer(
            adapter_factory=lambda: wrapper,
            encode_tir=encoder,
            output_root=_paths(config, config_path)["output_root"],
            config=phase16,
            candidate_specs=(self.candidate,),
            adapter_state=tir_adapter_state_dict,
            load_adapter_state=lambda module, state: load_tir_adapter_state_dict(module, state),
            resume=resume,
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
        del fractions
        self.trainer._train(
            module,
            self.candidate,
            records,
            self.bank,
            steps=len(records),
            checkpoint_dir=checkpoint_dir,
        )
        artifacts = []
        for path in sorted(checkpoint_dir.glob("checkpoint_*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("candidate") != "D2_G025":
                raise RuntimeError("PHASE_19_D2_FULL_CHECKPOINT_CANDIDATE_DRIFT")
            step = int(payload["step"])
            artifacts.append(
                CheckpointArtifact(step=step, fraction=step / len(records), path=path)
            )
        return tuple(artifacts)

    def load_checkpoint(
        self, module: torch.nn.Module, checkpoint: CheckpointArtifact
    ) -> None:
        payload = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
        if payload.get("teacher_bank_fingerprint") != self.bank.fingerprint:
            raise RuntimeError("PHASE_19_D2_FULL_CHECKPOINT_BANK_MISMATCH")
        load_tir_adapter_state_dict(module, payload["adapter"])
        freeze_for_tir_adapter_warmup(module)
        module.eval()
        self.selected_checkpoint = checkpoint.path

    def evaluate(
        self, module: torch.nn.Module, records: Sequence[RGBTRecord]
    ) -> Mapping[str, Any]:
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
                "status": "PHASE_19_D2_FULL_SELECTED_ADAPTER",
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
            raise RuntimeError("D2 official val requires a dev-selected checkpoint")
        metrics = dict(
            _validate_official_model(
                config=self.config,
                config_path=self.config_path,
                module=module,
                records=records,
                base_hash=self.base_hash,
                fingerprint=self.fingerprint,
                selected_checkpoint=str(self.selected_checkpoint),
            )
        )
        drift = metrics.get("layer_drift", {})
        metrics["maximum_absolute_layer_drift"] = max(
            abs(float(drift.get(layer, float("inf")))) for layer in ("8", "16", "24")
        )
        return metrics


def _environment_payload(
    *, stage: str, hardware: Mapping[str, Any], host_ram_gib: float, fingerprint: str
) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "hardware": dict(hardware),
        "host_ram_gib": host_ram_gib,
        "fingerprint": fingerprint,
        "stage": stage,
        "candidate": "D2_G025",
        "geometry_lambda": 0.25,
        "initialized_from": "D1_L050",
        "continued_from_probe_adapter": False,
        "second_model_fallback": False,
    }


def run(
    command: str,
    *,
    config_path: Path,
    resume: bool,
    allow_network: bool,
) -> int:
    config = _load_config(config_path)
    preflight = _preflight(config, config_path)
    if command == "preflight":
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    hardware = _require_cuda(float(config["runtime"]["minimum_vram_gib"]))
    host_ram_gib = _require_host_memory(float(config["runtime"]["minimum_host_ram_gib"]))
    assets = _model_assets(config, config_path, allow_network=allow_network)
    wrapper, base_hash = _build_model(config, assets)
    paths = _paths(config, config_path)
    probe = _load_probe_evidence(paths["probe_summary"])
    fingerprint = _fingerprint(config, preflight, assets, config_path)
    full, semantic, multiquery, official = _records(config, config_path)

    if command == "official-val":
        selection_path = paths["output_root"] / "train_selection_summary.json"
        if not selection_path.is_file():
            raise RuntimeError("PHASE_19_D2_TRAIN_SELECTION_MISSING")
        selection = D2FullSelectionResult(
            **json.loads(selection_path.read_text(encoding="utf-8"))
        )
        if selection.fingerprint != fingerprint:
            raise RuntimeError("PHASE_19_D2_TRAIN_SELECTION_FINGERPRINT_MISMATCH")
        adapter_path = Path(selection.selected_adapter)
        if not adapter_path.is_file():
            raise RuntimeError("PHASE_19_D2_SELECTED_ADAPTER_MISSING")
        release = torch.load(adapter_path, map_location="cpu", weights_only=False)
        if release.get("metadata", {}).get("fingerprint") != fingerprint:
            raise RuntimeError("PHASE_19_D2_SELECTED_ADAPTER_FINGERPRINT_MISMATCH")
        load_tir_adapter_state_dict(wrapper, release["tir_adapter"])
        freeze_for_tir_adapter_warmup(wrapper)
        wrapper.eval()
        metrics = dict(
            _validate_official_model(
                config=config,
                config_path=config_path,
                module=wrapper,
                records=official,
                base_hash=base_hash,
                fingerprint=fingerprint,
                selected_checkpoint=selection.selected_checkpoint,
            )
        )
        metrics["maximum_absolute_layer_drift"] = max(
            abs(float(metrics["layer_drift"][layer])) for layer in ("8", "16", "24")
        )
        result = Phase19D2FullRunner(
            backend=None,  # type: ignore[arg-type]
            output_root=paths["output_root"],
            baseline_metrics=probe["baseline_metrics"],
            config=_full_config(config),
        ).complete_official(selection, metrics)
        if parameter_sha256(_base_parameters(wrapper)) != base_hash:
            raise RuntimeError("PHASE_19_D2_FULL_FROZEN_BASE_CHANGED")
        _write_json(
            paths["output_root"] / "environment_official_val.json",
            _environment_payload(
                stage="official-val",
                hardware=hardware,
                host_ram_gib=host_ram_gib,
                fingerprint=fingerprint,
            ),
        )
        _sha_manifest(paths["output_root"])
        print(json.dumps({"status": result.status}, indent=2))
        return 0 if result.status == "PHASE_19_D2_FULL_GO" else 2

    d1_payload = torch.load(paths["d1_adapter"], map_location="cpu", weights_only=False)
    initial = d1_payload["tir_adapter"]
    initial_hash = _mapping_hash(initial)
    load_tir_adapter_state_dict(wrapper, initial)
    bank = RGBTeacherBank.load(
        paths["teacher_bank"],
        expected_fingerprint=str(config["expected"]["teacher_bank_fingerprint"]),
    )
    encoder = _QwenRepairEncoder(
        wrapper=wrapper,
        processor=_processor(config),
        root=paths["rgbt_root"],
        roi_expansion=float(config["loss"]["roi_expansion"]),
    )
    backend = _D2FullBackend(
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
    runner = Phase19D2FullRunner(
        backend=backend,
        output_root=paths["output_root"],
        baseline_metrics=probe["baseline_metrics"],
        config=_full_config(config),
    )
    if command == "train-select":
        selection = runner.train_select(
            full, semantic, multiquery, official, fingerprint=fingerprint
        )
        if parameter_sha256(_base_parameters(wrapper)) != base_hash:
            raise RuntimeError("PHASE_19_D2_FULL_FROZEN_BASE_CHANGED")
        if _mapping_hash(initial) != initial_hash:
            raise RuntimeError("PHASE_19_D2_FULL_D1_INITIAL_ASSET_CHANGED")
        _write_json(
            paths["output_root"] / "environment_train_select.json",
            _environment_payload(
                stage="train-select",
                hardware=hardware,
                host_ram_gib=host_ram_gib,
                fingerprint=fingerprint,
            ),
        )
        _sha_manifest(paths["output_root"])
        print(json.dumps({"status": selection.status}, indent=2))
        return 0 if selection.status == "PHASE_19_D2_TRAIN_SELECT_READY" else 2
    result = runner.run(full, semantic, multiquery, official, fingerprint=fingerprint)
    if parameter_sha256(_base_parameters(wrapper)) != base_hash:
        raise RuntimeError("PHASE_19_D2_FULL_FROZEN_BASE_CHANGED")
    _write_json(
        paths["output_root"] / "environment.json",
        _environment_payload(
            stage="all",
            hardware=hardware,
            host_ram_gib=host_ram_gib,
            fingerprint=fingerprint,
        ),
    )
    _sha_manifest(paths["output_root"])
    print(json.dumps({"status": result.status}, indent=2))
    return 0 if result.status == "PHASE_19_D2_FULL_GO" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="AIC RGB-TIR Phase 1.9-D2 full train")
    parser.add_argument(
        "command", choices=("preflight", "train-select", "official-val", "run")
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(20260816)
    return run(
        args.command,
        config_path=args.config.resolve(),
        resume=args.resume,
        allow_network=args.allow_network,
    )


if __name__ == "__main__":
    raise SystemExit(main())
