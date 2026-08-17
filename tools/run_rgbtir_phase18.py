from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_rgbtir.data import read_jsonl_records
from aic_rgbtir.phase1 import (
    freeze_for_tir_adapter_warmup,
    load_tir_adapter_state_dict,
    parameter_sha256,
    tir_adapter_state_dict,
)
from aic_rgbtir.phase16 import Phase16Config, RGBTeacherBank
from aic_rgbtir.phase18 import Phase18Config, Phase18RetentionProbe
from tools.run_rgbtir_phase16 import (
    _QwenRepairEncoder,
    _base_parameters,
    _build_model,
    _model_assets,
    _paths,
    _phase16_config,
    _preflight,
    _processor,
    _require_cuda,
    _sha256,
    _write_json,
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 1.8 config must be a YAML mapping")
    parent_name = payload.pop("extends", None)
    if parent_name is None:
        return payload
    parent_path = (path.parent / str(parent_name)).resolve()
    parent = _load_config(parent_path)

    def merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(base)
        for key, value in override.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = merge(dict(result[key]), value)
            else:
                result[key] = value
        return result

    return merge(parent, payload)


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
        "forbid_aic_test": True,
    }
    if dict(config.get("scope", {})) != expected:
        raise RuntimeError("PHASE_18_SCOPE_GUARD_MISMATCH")
    candidates = [
        (str(row["name"]), float(row["retention_lambda"]))
        for row in config.get("candidates", [])
    ]
    if candidates != [("D1_L010", 0.10), ("D1_L025", 0.25), ("D1_L050", 0.50)]:
        raise RuntimeError(f"PHASE_18_CANDIDATE_DRIFT: {candidates}")
    if int(config["runtime"].get("full_steps", -1)) != 0:
        raise RuntimeError("PHASE_18_FULL_TRAIN_FORBIDDEN")


def _phase18_config(config: Mapping[str, Any]) -> Phase18Config:
    reference = config["c2_reference"]
    gates = config["gates"]
    return Phase18Config(
        seed=int(config["runtime"]["seed"]),
        c2_layer_drift={str(k): float(v) for k, v in reference["layer_drift"].items()},
        c2_mean_r_at_1=float(reference["mean_r_at_1"]),
        c2_mean_r_at_5=float(reference["mean_r_at_5"]),
        maximum_r1_drop=float(gates["maximum_r1_drop"]),
        maximum_r5_drop=float(gates["maximum_r5_drop"]),
        minimum_layer24_drift_reduction=float(gates["minimum_layer24_drift_reduction"]),
        minimum_effective_rank_vs_base=float(gates["minimum_effective_rank_vs_base"]),
        minimum_effective_rank_vs_teacher=float(gates["minimum_effective_rank_vs_rgb"]),
        maximum_nonpaired_cosine_p95_increase=float(gates["maximum_nonpaired_p95_increase"]),
        retention_tolerance={
            str(k): float(v) for k, v in config["retention"]["tolerance"].items()
        },
    )


def _validate_inputs(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    _scope_guard(config)
    paths = _paths(config, config_path)
    phase16 = _preflight(config, config_path)
    expected = config["expected"]
    probe = read_jsonl_records(paths["manifest_root"] / "repair_probe_train.jsonl")
    dev = read_jsonl_records(paths["manifest_root"] / "repair_dev.jsonl")
    if len(probe) != int(expected["probe_records"]) or len(dev) != int(expected["dev_records"]):
        raise RuntimeError("PHASE_18_RECORD_COUNT_DRIFT")
    bank_path = paths["teacher_bank"]
    if not bank_path.is_file():
        raise FileNotFoundError(bank_path)
    actual_bank_sha = _sha256(bank_path)
    if actual_bank_sha != str(expected["teacher_bank_sha256"]).upper():
        raise RuntimeError(
            f"PHASE_18_TEACHER_BANK_SHA_MISMATCH: {actual_bank_sha}"
        )
    bank = RGBTeacherBank.load(
        bank_path,
        expected_fingerprint=str(expected["teacher_bank_fingerprint"]),
    )
    required_ids = {record.record_id for record in probe + dev}
    missing = sorted(required_ids - set(bank.record_ids))
    if missing:
        raise RuntimeError(f"PHASE_18_TEACHER_BANK_MISSING_RECORDS: {missing[:5]}")
    if len(required_ids) != int(expected["teacher_bank_records"]):
        raise RuntimeError("PHASE_18_TEACHER_BANK_RECORD_COUNT_DRIFT")
    for key in ("c2_reference", "c2_probe_summary"):
        path = paths[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_sha = str(expected[f"{key}_sha256"]).upper()
        if _sha256(path) != expected_sha:
            raise RuntimeError(f"PHASE_18_{key.upper()}_SHA_MISMATCH")
    c2_diagnostics = json.loads(paths["c2_reference"].read_text(encoding="utf-8"))
    c2_probe = json.loads(paths["c2_probe_summary"].read_text(encoding="utf-8"))
    configured_reference = config["c2_reference"]
    recovered_drift = {
        layer: float(c2_diagnostics["candidates"]["C2"]["layers"][layer]["absolute_alignment_delta_mean"])
        for layer in ("8", "16", "24")
    }
    configured_drift = {
        str(layer): float(value) for layer, value in configured_reference["layer_drift"].items()
    }
    if any(abs(recovered_drift[layer] - configured_drift[layer]) > 1e-9 for layer in recovered_drift):
        raise RuntimeError("PHASE_18_C2_DRIFT_REFERENCE_MISMATCH")
    for metric_key, config_key in (("mean_r_at_1", "mean_r_at_1"), ("mean_r_at_5", "mean_r_at_5")):
        if abs(float(c2_probe["metrics"][metric_key]) - float(configured_reference[config_key])) > 1e-9:
            raise RuntimeError(f"PHASE_18_C2_{metric_key.upper()}_REFERENCE_MISMATCH")
    payload = {
        "status": "PHASE_18_LOCAL_OR_CLOUD_PREFLIGHT_GO",
        "scope": config["scope"],
        "phase16_manifest_preflight": phase16,
        "teacher_bank": {
            "path": str(bank_path),
            "sha256": actual_bank_sha,
            "fingerprint": bank.fingerprint,
            "required_record_count": len(required_ids),
        },
        "c2_reference": {
            "diagnostics_sha256": _sha256(paths["c2_reference"]),
            "probe_summary_sha256": _sha256(paths["c2_probe_summary"]),
            "layer_drift": recovered_drift,
            "mean_r_at_1": float(c2_probe["metrics"]["mean_r_at_1"]),
            "mean_r_at_5": float(c2_probe["metrics"]["mean_r_at_5"]),
        },
        "candidates": config["candidates"],
        "claim_boundary": (
            "D1 retention probe only; no full training, official val, Query, bbox, "
            "fusion, Depth, AIC test inference, fallback or submission"
        ),
    }
    _write_json(paths["output_root"] / "phase18_preflight.json", payload)
    return payload


def _write_report(config: Mapping[str, Any], config_path: Path, result: Any) -> None:
    paths = _paths(config, config_path)
    lines = [
        "# AIC RGB–TIR Phase 1.8A D1 Retention Probe",
        "",
        f"- 状态：`{result.status}`",
        f"- 选中候选：`{result.selected_candidate or 'none'}`",
        "- 范围：仅 4,096/1,024 固定 probe/dev；未运行全量训练或 official val。",
        "",
        "## 候选结果",
        "",
        "| 候选 | lambda | 门禁 | R@1 | R@5 | L8 drift | L16 drift | L24 drift |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    lambdas = {row["name"]: row["retention_lambda"] for row in config["candidates"]}
    for name in result.completed_candidates:
        metrics = result.probe_metrics[name]
        drift = metrics["layer_drift"]
        decision = result.candidate_decisions[name]
        lines.append(
            f"| {name} | {lambdas[name]:.2f} | {decision['status']} | "
            f"{metrics['mean_r_at_1']:.4f} | {metrics['mean_r_at_5']:.4f} | "
            f"{drift['8']:.4f} | {drift['16']:.4f} | {drift['24']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 决策边界",
            "",
            "本报告只能决定 D1 是否值得进入 full train。它不能证明 Query grounding、bbox ACC 或 AIC 平台提升。",
            "",
        ]
    )
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text("\n".join(lines), encoding="utf-8")


def _sha_manifest(root: Path) -> None:
    rows = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            rows[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _write_json(root / "sha256_manifest.json", {"schema_version": 1, "outputs": rows})


def run(command: str, *, config_path: Path, resume: bool, allow_network: bool) -> int:
    config = _load_config(config_path)
    paths = _paths(config, config_path)
    preflight = _validate_inputs(config, config_path)
    if command == "preflight":
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    hardware = _require_cuda(float(config["runtime"]["minimum_vram_gib"]))
    assets = _model_assets(config, config_path, allow_network=allow_network)
    wrapper, base_hash = _build_model(config, assets)
    initial = tir_adapter_state_dict(wrapper)
    initial_hash = _mapping_hash(initial)
    bank = RGBTeacherBank.load(
        paths["teacher_bank"],
        expected_fingerprint=str(config["expected"]["teacher_bank_fingerprint"]),
    )
    probe_records = read_jsonl_records(paths["manifest_root"] / "repair_probe_train.jsonl")
    dev_records = read_jsonl_records(paths["manifest_root"] / "repair_dev.jsonl")
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
            raise RuntimeError("PHASE_18_IDENTICAL_INITIALIZATION_FAILED")
        return wrapper

    phase16_cfg: Phase16Config = _phase16_config(config)
    probe = Phase18RetentionProbe(
        output_root=paths["output_root"],
        config=_phase18_config(config),
        adapter_factory=factory,
        encode_tir=encoder,
        phase16_config=phase16_cfg,
        adapter_state=tir_adapter_state_dict,
        load_adapter_state=lambda module, state: load_tir_adapter_state_dict(module, state),
        resume=resume,
    )
    result = probe.run(probe_records, dev_records, bank)
    final_base_hash = parameter_sha256(_base_parameters(wrapper))
    if final_base_hash != base_hash:
        raise RuntimeError("PHASE_18_FROZEN_RGB_BASE_CHANGED")
    _write_json(
        paths["output_root"] / "safety_equivalence.json",
        {
            "status": "PASS",
            "base_hash_before": base_hash,
            "base_hash_after": final_base_hash,
            "fresh_adapter_hash": initial_hash,
            "second_model_fallback": False,
            "full_training_executed": False,
            "official_val_executed": False,
        },
    )
    _write_json(
        paths["output_root"] / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "hardware": hardware,
            "model_revision": config["model"]["revision"],
            "assets": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in assets.items()},
        },
    )
    _write_report(config, config_path, result)
    _sha_manifest(paths["output_root"])
    print(json.dumps({"status": result.status, "selected": result.selected_candidate}, indent=2))
    return 0 if result.status == "PHASE_18_D1_GO" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="AIC RGB-TIR Phase 1.8 D1 retention probe")
    parser.add_argument("command", choices=("preflight", "probe"))
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
