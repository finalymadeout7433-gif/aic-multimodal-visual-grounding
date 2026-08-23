from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_rgbtir.artifacts import create_stage_archive  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive one durable Phase 1.8 stage")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--required", action="append", default=[])
    parser.add_argument("--minimum-checkpoints", type=int, default=0)
    args = parser.parse_args()
    receipt = create_stage_archive(
        output_root=args.output_root,
        archive_path=args.archive,
        stage=args.stage,
        required=tuple(args.required),
        minimum_checkpoints=args.minimum_checkpoints,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
