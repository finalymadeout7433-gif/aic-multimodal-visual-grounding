from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import transformers

from aic_baseline.external_data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record local model hashes and diagnostic environment."
    )
    parser.add_argument("--florence-path", type=Path, required=True)
    parser.add_argument("--gdino-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _gdino_revision(model_path: Path) -> str | None:
    metadata = (
        model_path
        / ".cache"
        / "huggingface"
        / "download"
        / "model.safetensors.metadata"
    )
    if not metadata.exists():
        return None
    lines = metadata.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines else None


def _weight_record(path: Path) -> dict[str, object]:
    return {
        "file_name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    args = parse_args()
    florence_weights = args.florence_path / "model.safetensors"
    gdino_weights = args.gdino_path / "model.safetensors"
    payload = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "florence": {
            "model_id": "microsoft/Florence-2-large-ft",
            "source": "local ModelScope snapshot",
            "revision": None,
            "revision_note": (
                "The local snapshot does not expose a source commit; the "
                "weight SHA-256 is the immutable identifier for this run."
            ),
            "weights": _weight_record(florence_weights),
        },
        "grounding_dino": {
            "model_id": "IDEA-Research/grounding-dino-tiny",
            "source": "official Hugging Face repository",
            "revision": _gdino_revision(args.gdino_path),
            "weights": _weight_record(gdino_weights),
            "config": {
                "dtype": "float32",
                "box_threshold": 0.15,
                "text_threshold": 0.15,
                "max_candidates": 20,
                "query_policy": "lowercase and terminal period",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
