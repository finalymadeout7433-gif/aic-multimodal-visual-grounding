from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for value in (REPO_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from aic_rgbtir.data import RGBTRecord, read_jsonl_records  # noqa: E402
from aic_rgbtir.phase1 import (  # noqa: E402
    build_roi_patch_weights,
    freeze_for_tir_adapter_warmup,
    load_tir_adapter_state_dict,
    parameter_sha256,
)
from aic_rgbtir.phase16 import RGBTeacherBank  # noqa: E402
from aic_rgbtir.phase19_d2_layer_audit import (  # noqa: E402
    AUDIT_LAYERS,
    D2LayerAudit,
    LayerAuditFeatures,
    summarize_checkpoint_audits,
)
from tools.run_rgbtir_phase16 import (  # noqa: E402
    _base_parameters,
    _build_model,
    _model_assets,
    _processor,
    _require_cuda,
    _sha256,
    _to_cuda,
    _weighted_mean,
    _write_json,
)
from tools.run_rgbtir_phase18_full import _require_host_memory, _sha_manifest  # noqa: E402
from tools.run_rgbtir_phase19_d2 import _load_config, _resolve  # noqa: E402


MANIFESTS = {
    "semantic": "d2_semantic_dev.jsonl",
    "multiquery": "d2_multiquery_dev.jsonl",
}


def _paths(config: Mapping[str, Any], config_path: Path) -> dict[str, Path]:
    return {
        name: _resolve(config_path, str(value))
        for name, value in config["paths"].items()
    }


def _scope_guard(config: Mapping[str, Any]) -> None:
    scope = dict(config.get("scope", {}))
    required = {
        "layer_audit": True,
        "full_training": False,
        "sealed_dual_dev_selection": False,
        "one_shot_official_val": False,
        "forbid_query_training": True,
        "forbid_fusion": True,
        "forbid_depth": True,
        "forbid_aic_test": True,
        "forbid_fallback": True,
    }
    if scope != required:
        raise RuntimeError(f"PHASE_19_D2_LAYER_AUDIT_SCOPE_DRIFT: {scope}")
    if tuple(str(value) for value in config["audit"]["layers"]) != AUDIT_LAYERS:
        raise RuntimeError("PHASE_19_D2_LAYER_AUDIT_LAYER_DRIFT")
    if tuple(int(value) for value in config["audit"]["checkpoint_steps"]) != (
        5848,
        11696,
        17544,
        23391,
    ):
        raise RuntimeError("PHASE_19_D2_LAYER_AUDIT_CHECKPOINT_DRIFT")


def _records(
    config: Mapping[str, Any], config_path: Path
) -> dict[str, list[RGBTRecord]]:
    root = _paths(config, config_path)["manifest_root"]
    return {
        split: read_jsonl_records(root / name) for split, name in MANIFESTS.items()
    }


def _checkpoint_path(paths: Mapping[str, Path], step: int) -> Path:
    return paths["checkpoints_dir"] / f"checkpoint_{step:08d}.pt"


def _preflight(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    _scope_guard(config)
    paths = _paths(config, config_path)
    expected = config["expected"]
    if not (paths["rgbt_root"] / "image_data").is_dir():
        raise FileNotFoundError(paths["rgbt_root"] / "image_data")
    records = _records(config, config_path)
    counts: dict[str, Any] = {}
    for split, rows in records.items():
        manifest = paths["manifest_root"] / MANIFESTS[split]
        counts[split] = {
            "records": len(rows),
            "pairs": len({row.image_pair_key for row in rows}),
            "sha256": _sha256(manifest),
        }
        if counts[split]["records"] != int(expected[f"{split}_records"]):
            raise RuntimeError(f"PHASE_19_D2_LAYER_AUDIT_{split.upper()}_COUNT_DRIFT")
        if counts[split]["pairs"] != int(expected[f"{split}_pairs"]):
            raise RuntimeError(f"PHASE_19_D2_LAYER_AUDIT_{split.upper()}_PAIR_DRIFT")
        if counts[split]["sha256"] != str(expected[f"{split}_sha256"]).upper():
            raise RuntimeError(f"PHASE_19_D2_LAYER_AUDIT_{split.upper()}_SHA_DRIFT")

    if _sha256(paths["teacher_bank"]) != str(expected["teacher_bank_sha256"]).upper():
        raise RuntimeError("PHASE_19_D2_LAYER_AUDIT_BANK_SHA_DRIFT")
    bank = RGBTeacherBank.load(
        paths["teacher_bank"],
        expected_fingerprint=str(expected["teacher_bank_fingerprint"]),
    )
    required_ids = {record.record_id for rows in records.values() for record in rows}
    if required_ids - set(bank.record_ids):
        raise RuntimeError("PHASE_19_D2_LAYER_AUDIT_BANK_COVERAGE_FAILED")

    checkpoint_hashes: dict[str, str] = {}
    expected_hashes = {
        int(step): str(value).upper()
        for step, value in expected["checkpoint_sha256"].items()
    }
    for step in (int(value) for value in config["audit"]["checkpoint_steps"]):
        path = _checkpoint_path(paths, step)
        actual = _sha256(path)
        if actual != expected_hashes[step]:
            raise RuntimeError(f"PHASE_19_D2_LAYER_AUDIT_CHECKPOINT_{step}_SHA_DRIFT")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("step", -1)) != step or payload.get("candidate") != "D2_G025":
            raise RuntimeError(f"PHASE_19_D2_LAYER_AUDIT_CHECKPOINT_{step}_CONTRACT")
        if payload.get("teacher_bank_fingerprint") != bank.fingerprint:
            raise RuntimeError(f"PHASE_19_D2_LAYER_AUDIT_CHECKPOINT_{step}_BANK_DRIFT")
        checkpoint_hashes[str(step)] = actual

    payload = {
        "status": "PHASE_19_D2_LAYER_AUDIT_PREFLIGHT_GO",
        "scope": config["scope"],
        "counts": counts,
        "checkpoint_sha256": checkpoint_hashes,
        "teacher_bank_fingerprint": bank.fingerprint,
        "claim_boundary": (
            "no-training dual-dev per-layer representation audit only; no selected "
            "adapter, official val, Query, fusion, Depth, AIC test or platform claim"
        ),
    }
    paths["output_root"].mkdir(parents=True, exist_ok=True)
    _write_json(paths["output_root"] / "preflight.json", payload)
    return payload


def _fingerprint(
    config: Mapping[str, Any],
    config_path: Path,
    preflight: Mapping[str, Any],
    assets: Mapping[str, Path],
) -> str:
    payload = {
        "scope": config["scope"],
        "model": config["model"],
        "processor": config["processor"],
        "audit": config["audit"],
        "counts": preflight["counts"],
        "checkpoints": preflight["checkpoint_sha256"],
        "teacher_bank_fingerprint": preflight["teacher_bank_fingerprint"],
        "model_assets": {
            name: _sha256(path) for name, path in sorted(assets.items())
        },
        "sources": {
            path.name: _sha256(path)
            for path in (
                REPO_ROOT / "src/aic_rgbtir/phase19_d2_layer_audit.py",
                Path(__file__),
                config_path,
            )
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def _encode_split(
    *,
    wrapper: torch.nn.Module,
    processor: Any,
    root: Path,
    records: Sequence[RGBTRecord],
    checkpoint_step: int,
    checkpoint_sha256: str,
    split: str,
    fingerprint: str,
    cache_path: Path,
    resume: bool,
    checkpoint_pairs: int,
    roi_expansion: float,
) -> dict[str, torch.Tensor]:
    expected_ids = [record.record_id for record in records]

    def validate(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if payload.get("fingerprint") != fingerprint:
            raise RuntimeError("PHASE_19_D2_LAYER_AUDIT_CACHE_FINGERPRINT_DRIFT")
        if payload.get("checkpoint_sha256") != checkpoint_sha256:
            raise RuntimeError("PHASE_19_D2_LAYER_AUDIT_CACHE_CHECKPOINT_DRIFT")
        if payload.get("record_ids") != expected_ids:
            raise RuntimeError("PHASE_19_D2_LAYER_AUDIT_CACHE_RECORD_DRIFT")
        adapted = payload.get("adapted")
        if not isinstance(adapted, Mapping) or set(adapted) != set(AUDIT_LAYERS):
            raise RuntimeError("PHASE_19_D2_LAYER_AUDIT_CACHE_LAYER_DRIFT")
        return {layer: adapted[layer].float() for layer in AUDIT_LAYERS}

    if resume and cache_path.is_file():
        return validate(torch.load(cache_path, map_location="cpu", weights_only=False))

    partial = cache_path.with_name(cache_path.stem + ".partial.pt")
    entries: dict[str, dict[str, torch.Tensor]] = {}
    completed_pairs: set[str] = set()
    if resume and partial.is_file():
        payload = torch.load(partial, map_location="cpu", weights_only=False)
        if payload.get("fingerprint") != fingerprint:
            raise RuntimeError("PHASE_19_D2_LAYER_AUDIT_PARTIAL_FINGERPRINT_DRIFT")
        if payload.get("checkpoint_sha256") != checkpoint_sha256:
            raise RuntimeError("PHASE_19_D2_LAYER_AUDIT_PARTIAL_CHECKPOINT_DRIFT")
        entries.update(payload.get("entries", {}))
        completed_pairs.update(payload.get("completed_pairs", ()))

    by_pair: dict[str, list[RGBTRecord]] = defaultdict(list)
    for record in records:
        by_pair[record.image_pair_key].append(record)
    for pair_index, pair_key in enumerate(
        tqdm(sorted(by_pair), desc=f"{checkpoint_step}:{split}"), start=1
    ):
        if pair_key in completed_pairs:
            continue
        pair_records = by_pair[pair_key]
        batch = processor.process(pair_records[0], root=root)
        if not batch.ir_usable:
            raise RuntimeError(f"unusable TIR in audit: {pair_key}")
        moved = _to_cuda(batch)
        encoded = wrapper.encode_tir_repair_features(
            tir_pixel_values=moved["tir"], image_grid_thw=moved["grid"]
        )
        layer_maps = dict(zip(encoded.layer_names, encoded.premerger))
        for record in pair_records:
            roi = build_roi_patch_weights(
                normalized_bbox=record.bbox_xyxy_normalized,
                image_grid_thw=moved["grid"],
                spatial_merge_size=int(processor.spatial_merge_size),
                expansion=float(roi_expansion),
                device="cuda:0",
            )
            foreground = roi * moved["valid"]
            if float(foreground.sum()) < 1.0:
                raise RuntimeError(f"empty valid TIR ROI in audit: {record.record_id}")
            entries[record.record_id] = {
                layer: _weighted_mean(layer_maps[layer], foreground).detach().cpu().half()
                for layer in AUDIT_LAYERS
            }
        completed_pairs.add(pair_key)
        if pair_index % max(1, checkpoint_pairs) == 0:
            _atomic_torch_save(
                {
                    "fingerprint": fingerprint,
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_sha256": checkpoint_sha256,
                    "split": split,
                    "completed_pairs": sorted(completed_pairs),
                    "entries": entries,
                },
                partial,
            )

    adapted = {
        layer: torch.stack([entries[record_id][layer] for record_id in expected_ids])
        for layer in AUDIT_LAYERS
    }
    final = {
        "fingerprint": fingerprint,
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_sha256,
        "split": split,
        "record_ids": expected_ids,
        "adapted": adapted,
    }
    _atomic_torch_save(final, cache_path)
    partial.unlink(missing_ok=True)
    return {layer: tensor.float() for layer, tensor in adapted.items()}


def run(
    config_path: Path, *, resume: bool, allow_network: bool, execute: bool
) -> int:
    config = _load_config(config_path)
    preflight = _preflight(config, config_path)
    paths = _paths(config, config_path)
    if not execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    _require_cuda(float(config["runtime"]["minimum_vram_gib"]))
    _require_host_memory(float(config["runtime"]["minimum_host_ram_gib"]))
    assets = _model_assets(config, config_path, allow_network=allow_network)
    wrapper, base_hash = _build_model(config, assets)
    processor = _processor(config)
    bank = RGBTeacherBank.load(
        paths["teacher_bank"],
        expected_fingerprint=str(config["expected"]["teacher_bank_fingerprint"]),
    )
    records = _records(config, config_path)
    fingerprint = _fingerprint(config, config_path, preflight, assets)
    _write_json(paths["output_root"] / "run_fingerprint.json", {"fingerprint": fingerprint})
    audit = D2LayerAudit(
        minimum_rank_vs_base=float(config["audit"]["minimum_rank_vs_base"]),
        seed=int(config["runtime"]["seed"]),
    )
    results: dict[int, dict[str, Any]] = {}
    checkpoint_hashes = preflight["checkpoint_sha256"]
    for step in (int(value) for value in config["audit"]["checkpoint_steps"]):
        checkpoint = _checkpoint_path(paths, step)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        load_tir_adapter_state_dict(wrapper, payload["adapter"])
        freeze_for_tir_adapter_warmup(wrapper)
        wrapper.eval()
        results[step] = {}
        for split in ("semantic", "multiquery"):
            rows = records[split]
            adapted = _encode_split(
                wrapper=wrapper,
                processor=processor,
                root=paths["rgbt_root"],
                records=rows,
                checkpoint_step=step,
                checkpoint_sha256=checkpoint_hashes[str(step)],
                split=split,
                fingerprint=fingerprint,
                cache_path=paths["output_root"]
                / "feature_cache"
                / f"checkpoint_{step:08d}_{split}.pt",
                resume=resume,
                checkpoint_pairs=int(config["runtime"]["cache_checkpoint_pairs"]),
                roi_expansion=float(config["loss"]["roi_expansion"]),
            )
            teacher = {
                layer: torch.stack(
                    [bank[record.record_id].foreground[layer].float() for record in rows]
                )
                for layer in AUDIT_LAYERS
            }
            base = {
                layer: torch.stack(
                    [
                        (bank[record.record_id].base_tir or bank[record.record_id].foreground)[
                            layer
                        ].float()
                        for record in rows
                    ]
                )
                for layer in AUDIT_LAYERS
            }
            result = audit.analyze(
                split=split,
                checkpoint_step=step,
                records=rows,
                features=LayerAuditFeatures(adapted=adapted, base=base, teacher=teacher),
                fingerprint=fingerprint,
            )
            results[step][split] = result
            _write_json(
                paths["output_root"]
                / "metrics"
                / f"checkpoint_{step:08d}_{split}.json",
                result,
            )

    summary = summarize_checkpoint_audits(results)
    summary["fingerprint"] = fingerprint
    summary["official_val_opened"] = False
    summary["selected_adapter_created"] = False
    _write_json(paths["output_root"] / "layer_audit_summary.json", summary)
    if parameter_sha256(_base_parameters(wrapper)) != base_hash:
        raise RuntimeError("PHASE_19_D2_LAYER_AUDIT_FROZEN_BASE_CHANGED")
    _sha_manifest(paths["output_root"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the no-training Phase 1.9-D2 layer audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="load Qwen and encode both fixed dev sets; otherwise preflight only",
    )
    args = parser.parse_args()
    return run(
        args.config.resolve(),
        resume=args.resume,
        allow_network=args.allow_network,
        execute=args.execute,
    )


if __name__ == "__main__":
    raise SystemExit(main())
