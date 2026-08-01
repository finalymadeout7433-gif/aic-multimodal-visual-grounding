from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a PyTorch checkpoint without constructing its model."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefix-depth", type=int, default=3)
    parser.add_argument(
        "--trusted-source",
        action="store_true",
        help=(
            "Allow legacy pickle metadata when weights_only loading is impossible. "
            "Use only after verifying the checkpoint came from its official source."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _state_dict(checkpoint: Any) -> tuple[Mapping[str, Any], str]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {type(checkpoint).__name__}")
    for key in ("model", "state_dict", "module"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value, key
    return checkpoint, "<root>"


def summarize_checkpoint(
    checkpoint: Any,
    *,
    checkpoint_path: Path,
    prefix_depth: int,
) -> dict[str, Any]:
    state, state_key = _state_dict(checkpoint)
    tensor_entries = {
        str(key): value for key, value in state.items() if torch.is_tensor(value)
    }
    prefix_counts: Counter[str] = Counter()
    prefix_bytes: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    total_tensor_bytes = 0
    total_parameters = 0
    for name, tensor in tensor_entries.items():
        parts = name.split(".")
        prefix = ".".join(parts[:prefix_depth])
        size_bytes = tensor.numel() * tensor.element_size()
        prefix_counts[prefix] += 1
        prefix_bytes[prefix] += size_bytes
        dtype_counts[str(tensor.dtype)] += 1
        total_tensor_bytes += size_bytes
        total_parameters += tensor.numel()

    language_keys = sorted(
        name
        for name in tensor_entries
        if "language" in name.lower()
        or "text" in name.lower()
        or "token_embedding" in name.lower()
    )
    return {
        "status": "ok",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "file_size_bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path),
        "top_level_keys": sorted(str(key) for key in checkpoint.keys()),
        "state_dict_container": state_key,
        "state_entry_count": len(state),
        "tensor_entry_count": len(tensor_entries),
        "total_parameters": total_parameters,
        "total_tensor_bytes": total_tensor_bytes,
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "prefix_depth": prefix_depth,
        "largest_prefixes": [
            {
                "prefix": prefix,
                "tensor_count": prefix_counts[prefix],
                "tensor_bytes": size_bytes,
            }
            for prefix, size_bytes in prefix_bytes.most_common(30)
        ],
        "language_related_key_count": len(language_keys),
        "language_related_key_examples": language_keys[:50],
    }


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        safe_load_mode = "weights_only"
    except Exception as error:
        if not args.trusted_source:
            raise RuntimeError(
                "safe weights-only loading failed; verify the official source and rerun "
                "with --trusted-source if legacy metadata inspection is required"
            ) from error
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        safe_load_mode = "trusted_legacy_pickle"
    summary = summarize_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
        prefix_depth=args.prefix_depth,
    )
    summary["load_mode"] = safe_load_mode
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
