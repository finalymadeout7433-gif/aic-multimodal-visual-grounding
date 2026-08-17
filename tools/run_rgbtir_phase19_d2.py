from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aic_rgbtir.data import read_jsonl_records  # noqa: E402
from aic_rgbtir.phase1 import (  # noqa: E402
    freeze_for_tir_adapter_warmup,
    load_tir_adapter_state_dict,
    parameter_sha256,
    tir_adapter_state_dict,
)
from aic_rgbtir.phase16 import Phase16Config, RGBTeacherBank  # noqa: E402
from aic_rgbtir.phase19_d2 import D2RepairTrainer  # noqa: E402
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


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Phase 1.9-D2 config must be a mapping")
    parent_name = payload.pop("extends", None)
    if parent_name is None:
        return payload
    parent = _load_config((path.parent / str(parent_name)).resolve())

    def merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(base)
        for key, value in override.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = merge(dict(result[key]), value)
            else:
                result[key] = value
        return result

    return merge(parent, payload)


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _mapping_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().float().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def _scope_guard(config: Mapping[str, Any]) -> None:
    expected = {
        "probe_only": True,
        "forbid_full_training": True,
        "forbid_official_val": True,
        "forbid_query": True,
        "forbid_fusion": True,
        "forbid_depth": True,
        "forbid_aic_test": True,
        "forbid_fallback": True,
    }
    if dict(config.get("scope", {})) != expected:
        raise RuntimeError("PHASE_19_D2_SCOPE_DRIFT")
    candidates = config.get("candidates", [])
    actual = [(row["name"], float(row["geometry_lambda"])) for row in candidates]
    if actual != [("D2_G010", 0.10), ("D2_G025", 0.25), ("D2_G050", 0.50)]:
        raise RuntimeError(f"PHASE_19_D2_CANDIDATE_DRIFT: {actual}")
    if int(config["runtime"].get("full_steps", -1)) != 0:
        raise RuntimeError("PHASE_19_D2_FULL_TRAIN_FORBIDDEN")


def _paths(config: Mapping[str, Any], config_path: Path) -> dict[str, Path]:
    paths = config["paths"]
    return {
        name: _resolve(config_path, str(paths[name]))
        for name in (
            "rgbt_root",
            "manifest_root",
            "teacher_bank",
            "d1_adapter",
            "output_root",
            "model_cache",
            "report",
        )
    }


def _preflight(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    _scope_guard(config)
    paths = _paths(config, config_path)
    expected = config["expected"]
    manifests = {
        "probe": paths["manifest_root"] / "d2_probe_train.jsonl",
        "semantic": paths["manifest_root"] / "d2_semantic_dev.jsonl",
        "multiquery": paths["manifest_root"] / "d2_multiquery_dev.jsonl",
    }
    records = {name: read_jsonl_records(path) for name, path in manifests.items()}
    actual_counts = {
        name: {
            "records": len(rows),
            "pairs": len({row.image_pair_key for row in rows}),
        }
        for name, rows in records.items()
    }
    for name, counts in actual_counts.items():
        for unit, value in counts.items():
            if value != int(expected[f"{name}_{unit}"]):
                raise RuntimeError(f"PHASE_19_D2_{name.upper()}_{unit.upper()}_DRIFT")
        expected_hash = str(expected[f"{name}_sha256"]).upper()
        if _sha256(manifests[name]) != expected_hash:
            raise RuntimeError(f"PHASE_19_D2_{name.upper()}_SHA_DRIFT")
    pair_sets = {
        name: {row.image_pair_key for row in rows} for name, rows in records.items()
    }
    if any(
        pair_sets[left] & pair_sets[right]
        for left, right in (("probe", "semantic"), ("probe", "multiquery"), ("semantic", "multiquery"))
    ):
        raise RuntimeError("PHASE_19_D2_SPLIT_LEAKAGE")
    for name in ("teacher_bank", "d1_adapter"):
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != str(expected[f"{name}_sha256"]).upper():
            raise RuntimeError(f"PHASE_19_D2_{name.upper()}_SHA_DRIFT")
    adapter_payload = torch.load(paths["d1_adapter"], map_location="cpu", weights_only=False)
    adapter = adapter_payload.get("tir_adapter")
    if not isinstance(adapter, Mapping) or len(adapter) != int(expected["adapter_tensors"]):
        raise RuntimeError("PHASE_19_D2_D1_ADAPTER_CONTRACT_FAILED")
    if adapter_payload.get("teacher_bank_fingerprint") != expected["teacher_bank_fingerprint"]:
        raise RuntimeError("PHASE_19_D2_ADAPTER_BANK_FINGERPRINT_DRIFT")
    bank = RGBTeacherBank.load(
        paths["teacher_bank"],
        expected_fingerprint=str(expected["teacher_bank_fingerprint"]),
    )
    required = {row.record_id for rows in records.values() for row in rows}
    missing = required - set(bank.record_ids)
    if missing:
        raise RuntimeError(f"PHASE_19_D2_TEACHER_BANK_MISSING: {sorted(missing)[:3]}")
    result = {
        "status": "PHASE_19_D2_PREFLIGHT_GO",
        "scope": config["scope"],
        "counts": actual_counts,
        "teacher_bank_fingerprint": bank.fingerprint,
        "teacher_bank_records": len(bank.record_ids),
        "d1_adapter_tensors": len(adapter),
        "candidates": config["candidates"],
        "claim_boundary": "D2 probe only; no official val or platform claim.",
    }
    _write_json(paths["output_root"] / "preflight.json", result)
    return result


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
        probe_steps=int(runtime["probe_steps"]),
        full_steps=0,
        negative_count=int(loss["negative_count"]),
        same_source_negatives=int(loss["same_source_negatives"]),
        same_condition_negatives=int(loss["same_condition_negatives"]),
        relational_anchor_count=int(loss["relational_anchor_count"]),
    )


def run(command: str, config_path: Path, *, allow_network: bool) -> int:
    config = _load_config(config_path)
    paths = _paths(config, config_path)
    preflight = _preflight(config, config_path)
    if command == "preflight":
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    hardware = _require_cuda(float(config["runtime"]["minimum_vram_gib"]))
    assets = _model_assets(config, config_path, allow_network=allow_network)
    wrapper, base_hash = _build_model(config, assets)
    payload = torch.load(paths["d1_adapter"], map_location="cpu", weights_only=False)
    initial = payload["tir_adapter"]
    load_tir_adapter_state_dict(wrapper, initial)
    initial_hash = _mapping_hash(initial)
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

    def factory() -> torch.nn.Module:
        load_tir_adapter_state_dict(wrapper, initial)
        freeze_for_tir_adapter_warmup(wrapper)
        wrapper.eval()
        if _mapping_hash(tir_adapter_state_dict(wrapper)) != initial_hash:
            raise RuntimeError("PHASE_19_D2_IDENTICAL_D1_START_FAILED")
        return wrapper

    trainer = D2RepairTrainer(
        adapter_factory=factory,
        encode_tir=encoder,
        output_root=paths["output_root"],
        config=_phase16_config(config),
        adapter_state=tir_adapter_state_dict,
        load_adapter_state=lambda module, state: load_tir_adapter_state_dict(module, state),
    )
    root = paths["manifest_root"]
    result = trainer.run_dual_dev(
        read_jsonl_records(root / "d2_probe_train.jsonl"),
        read_jsonl_records(root / "d2_semantic_dev.jsonl"),
        read_jsonl_records(root / "d2_multiquery_dev.jsonl"),
        bank,
    )
    if parameter_sha256(_base_parameters(wrapper)) != base_hash:
        raise RuntimeError("PHASE_19_D2_FROZEN_RGB_BASE_CHANGED")
    _write_json(
        paths["output_root"] / "safety_equivalence.json",
        {
            "status": "PASS",
            "hardware": hardware,
            "base_hash": base_hash,
            "d1_initial_adapter_hash": initial_hash,
            "second_model_fallback": False,
            "official_val_executed": False,
        },
    )
    print(json.dumps({"status": result.status, "selected": result.selected_candidate}, indent=2))
    return 0 if result.status == "PHASE_19_D2_PROBE_GO" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="AIC RGB-TIR Phase 1.9-D2 probe")
    parser.add_argument("command", choices=("preflight", "probe"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(20260816)
    return run(args.command, args.config.resolve(), allow_network=args.allow_network)


if __name__ == "__main__":
    raise SystemExit(main())
