from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_rgbtir.data import RGBTRecord, read_jsonl_records  # noqa: E402
from aic_rgbtir.query_probe import QueryCandidate, QueryInterfaceSmoke  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Q0 config must be a mapping")
    return payload


def _path(config: Mapping[str, Any], key: str) -> Path:
    value = Path(str(config["paths"][key]))
    return value if value.is_absolute() else (REPO_ROOT / value).resolve()


class _TokenizerSketch:
    """Deterministic tokenizer contract; not a learned Query representation."""

    def __init__(self, tokenizer_path: Path, *, dimension: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=False
        )
        self.dimension = int(dimension)

    def __call__(self, query: str) -> torch.Tensor:
        ids = self.tokenizer(query, add_special_tokens=True, return_tensors="pt")[
            "input_ids"
        ][0]
        vector = torch.zeros(self.dimension, dtype=torch.float32)
        for position, token in enumerate(ids.tolist(), start=1):
            index = (int(token) * 131 + position * 17) % self.dimension
            vector[index] += 1.0 + (int(token) % 997) / 997.0
        return torch.nn.functional.normalize(vector, dim=0)


class _ROISketch:
    """Deterministic geometry contract; not a learned visual representation."""

    def __init__(self, *, dimension: int) -> None:
        self.dimension = int(dimension)

    def __call__(self, _record: RGBTRecord, candidate: QueryCandidate) -> torch.Tensor:
        vector = torch.zeros(self.dimension, dtype=torch.float32)
        box = candidate.bbox_xyxy_normalized
        vector[:4] = torch.tensor(box, dtype=torch.float32)
        digest = hashlib.sha256(
            f"{candidate.kind}:{candidate.target_head}".encode("utf-8")
        ).digest()
        for offset, byte in enumerate(digest):
            vector[4 + offset % max(1, self.dimension - 4)] += byte / 255.0
        return torch.nn.functional.normalize(vector, dim=0)


def _verify_file(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
    if int(expected["bytes"]) != actual["bytes"] or str(expected["sha256"]).upper() != actual[
        "sha256"
    ]:
        raise RuntimeError(f"Q0 asset mismatch: {path}")
    return actual


def _manifest_rows(records: list[RGBTRecord], selected_ids: set[str]) -> list[dict[str, Any]]:
    return [record.as_dict() for record in records if record.record_id in selected_ids]


def _sha_manifest(output_root: Path) -> None:
    rows = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            rows[path.relative_to(output_root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _write_json(output_root / "sha256_manifest.json", {"schema_version": 1, "outputs": rows})


def run(config_path: Path) -> int:
    config = _load(config_path)
    expected_scope = {
        "query_interface_contract_only": True,
        "forbid_training": True,
        "forbid_official_val": True,
        "forbid_aic_test": True,
        "forbid_fusion": True,
    }
    if dict(config.get("scope", {})) != expected_scope:
        raise RuntimeError("PHASE_18_Q0_SCOPE_GUARD_MISMATCH")
    output_root = _path(config, "output_root")
    assets = {
        name: _verify_file(_path(config, name), expected)
        for name, expected in config["expected_assets"].items()
    }
    dev = read_jsonl_records(_path(config, "repair_dev"))
    clean = read_jsonl_records(_path(config, "train_clean"))
    expected = config["expected"]
    if len(dev) != int(expected["repair_dev_records"]):
        raise RuntimeError("Q0 repair-dev count drift")
    if len(clean) != int(expected["train_clean_records"]):
        raise RuntimeError("Q0 train-clean count drift")
    dimension = int(config["interface"]["embedding_dimension"])
    smoke = QueryInterfaceSmoke(
        query_encoder=_TokenizerSketch(_path(config, "tokenizer_root"), dimension=dimension),
        roi_encoder=_ROISketch(dimension=dimension),
        maximum_records=int(config["runtime"]["maximum_records"]),
        seed=int(config["runtime"]["seed"]),
    )
    result = smoke.run(dev, candidate_pool=clean)
    selected_ids = {row["record_id"] for row in result.role_audit}
    _write_jsonl(
        output_root / "smoke_manifest.jsonl", _manifest_rows(dev, selected_ids)
    )
    _write_jsonl(output_root / "candidate_audit.jsonl", list(result.candidate_audit))
    _write_json(
        output_root / "interface_checks.json",
        {
            "status": result.status,
            "checks": result.checks,
            "query_dimension": result.query_dimension,
            "roi_dimension": result.roi_dimension,
            "role_audit": result.role_audit,
        },
    )
    _write_json(
        output_root / "run_summary.json",
        {
            "status": result.status,
            "record_count": result.record_count,
            "candidate_count": result.candidate_count,
            "backend": "TOKENIZER_AND_GEOMETRY_CONTRACT_ONLY",
            "claim_boundary": (
                "Local Q0 validates tokenizer/candidate/interface determinism only; "
                "it does not validate learned Query semantics, D1 visual features, bbox ACC, fusion, or AIC gain."
            ),
            "assets": assets,
        },
    )
    _sha_manifest(output_root)
    print(json.dumps(json.loads((output_root / "run_summary.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    return 0 if result.status == "PHASE_18_Q0_READY" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="AIC RGB-TIR Phase 1.8A-Q0 interface smoke")
    parser.add_argument("--config", required=True, type=Path)
    return run(parser.parse_args().config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
