from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_baseline.completed_run_audit import verify_completed_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a completed AIC run before reusing its platform ZIP."
    )
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--run-key", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expectations = json.loads(args.expectations.read_text(encoding="utf-8"))
    if args.run_key not in expectations:
        raise KeyError(f"missing run expectation: {args.run_key}")
    expected = expectations[args.run_key]
    result = verify_completed_run(
        run_directory=args.run_directory,
        expected_query_count=int(expected["expected_query_count"]),
        expected_fingerprint=dict(expected["fingerprint"]),
        expected_fingerprint_sha256=expected.get("fingerprint_sha256"),
        expected_post_run_provenance=expected.get("post_run_provenance"),
        expected_post_run_provenance_sha256=expected.get(
            "post_run_provenance_sha256"
        ),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
