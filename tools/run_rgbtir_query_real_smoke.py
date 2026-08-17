from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from transformers import AutoModelForImageTextToText, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_rgbtir.data import read_jsonl_records  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("real Q0 config must be a mapping")
    return payload


def run(config_path: Path, *, allow_network: bool) -> int:
    config = _load(config_path)
    expected_scope = {
        "real_query_hidden_smoke_only": True,
        "forbid_training": True,
        "forbid_visual_fusion": True,
        "forbid_official_val": True,
        "forbid_aic_test": True,
    }
    if dict(config.get("scope", {})) != expected_scope:
        raise RuntimeError("PHASE_18_Q0_REAL_SCOPE_GUARD_MISMATCH")
    paths = config["paths"]
    manifest = Path(paths["repair_dev"])
    if _sha256(manifest) != str(config["expected"]["repair_dev_sha256"]).upper():
        raise RuntimeError("PHASE_18_Q0_REAL_MANIFEST_DRIFT")
    records = read_jsonl_records(manifest)
    if len(records) != int(config["expected"]["repair_dev_records"]):
        raise RuntimeError("PHASE_18_Q0_REAL_RECORD_COUNT_DRIFT")
    maximum = int(config["runtime"]["maximum_records"])
    selected = sorted(records, key=lambda row: row.record_id)[:maximum]
    model_cfg = config["model"]
    cache_dir = Path(paths["model_cache"])
    common = {
        "revision": str(model_cfg["revision"]),
        "cache_dir": cache_dir,
        "local_files_only": not allow_network,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(str(model_cfg["model_id"]), **common)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_cfg["model_id"]),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        **common,
    )
    model.requires_grad_(False)
    model.eval()
    batch = tokenizer(
        [record.query_original for record in selected],
        padding=True,
        return_tensors="pt",
    )
    batch = {key: value.to("cuda:0") for key, value in batch.items()}

    def encode() -> torch.Tensor:
        with torch.inference_mode():
            output = model(
                **batch,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            if not output.hidden_states:
                raise RuntimeError("Qwen real Q0 returned no hidden states")
            hidden = output.hidden_states[-1].float()
            mask = batch["attention_mask"].float().unsqueeze(-1)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    first = encode().cpu()
    second = encode().cpu()
    if not torch.equal(first, second):
        raise RuntimeError("PHASE_18_Q0_REAL_NONDETERMINISTIC_HIDDEN")
    if not torch.isfinite(first).all():
        raise RuntimeError("PHASE_18_Q0_REAL_NONFINITE_HIDDEN")
    generator = torch.Generator(device="cpu").manual_seed(int(config["runtime"]["seed"]))
    projection = torch.randn(
        first.shape[1],
        int(config["interface"]["projected_dimension"]),
        generator=generator,
        dtype=torch.float32,
    ) / max(first.shape[1], 1) ** 0.5
    projected = first @ projection
    if not torch.isfinite(projected).all():
        raise RuntimeError("PHASE_18_Q0_REAL_NONFINITE_PROJECTION")
    output_root = Path(paths["output_root"])
    summary = {
        "status": "PHASE_18_Q0_REAL_READY",
        "record_count": len(selected),
        "query_hidden_dimension": int(first.shape[1]),
        "projected_dimension": int(projected.shape[1]),
        "model_revision": model_cfg["revision"],
        "all_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
        "outputs_finite": True,
        "repeat_exact": True,
        "claim_boundary": (
            "Real Qwen text hidden-state and 1152-D bridge smoke only. The projection is "
            "fixed and untrained; this does not measure Query grounding, D1 quality, bbox ACC or fusion gain."
        ),
    }
    _write(output_root / "run_summary.json", summary)
    _write(
        output_root / "records.json",
        {"record_ids": [row.record_id for row in selected]},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Real Qwen3-VL Query hidden-state smoke")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    return run(args.config.resolve(), allow_network=args.allow_network)


if __name__ == "__main__":
    raise SystemExit(main())
