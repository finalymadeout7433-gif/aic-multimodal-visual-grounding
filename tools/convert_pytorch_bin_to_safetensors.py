from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an official PyTorch state dict to a non-pickle safetensors "
            "directory while preserving duplicated/tied keys."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_bin = source_dir / "pytorch_model.bin"
    actual_sha256 = sha256_file(source_bin)
    if actual_sha256 != args.expected_sha256.upper():
        raise ValueError(
            f"source SHA-256 mismatch: {actual_sha256} != "
            f"{args.expected_sha256.upper()}"
        )

    state = torch.load(source_bin, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise TypeError("pytorch_model.bin does not contain a non-empty state dict")
    safe_state: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"non-tensor state entry: {name}")
        safe_state[str(name)] = tensor.detach().cpu().contiguous().clone()
    state_entry_count = len(safe_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ):
        shutil.copy2(source_dir / filename, output_dir / filename)
    output_weights = output_dir / "model.safetensors"
    save_file(
        safe_state,
        output_weights,
        metadata={
            "format": "pt",
            "source_pytorch_model_bin_sha256": actual_sha256,
        },
    )
    del safe_state
    del state

    manifest = {
        "status": "ok",
        "artifact_kind": "derived_safe_serialization",
        "source_dir": str(source_dir),
        "source_pytorch_model_bin": str(source_bin),
        "source_sha256": actual_sha256,
        "output_dir": str(output_dir),
        "output_model_safetensors": str(output_weights),
        "output_sha256": sha256_file(output_weights),
        "output_size_bytes": output_weights.stat().st_size,
        "state_entry_count": state_entry_count,
        "note": (
            "Weights are unchanged; tensors were cloned only to preserve all "
            "official duplicated keys in a non-pickle serialization."
        ),
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
