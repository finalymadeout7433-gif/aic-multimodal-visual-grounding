from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from aic_baseline.testset_profile_pipeline import run_profile_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the read-only AIC unlabeled test-domain profile."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed scale-probe work when its frozen fingerprint matches.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("profile config must contain a YAML object")
    return value


def main() -> int:
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    summary = run_profile_pipeline(
        _load_config(config_path.resolve()),
        repo_root=REPO_ROOT,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
