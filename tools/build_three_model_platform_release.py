from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_baseline.platform_release import build_platform_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and collect the three zero-shot AIC model ZIPs."
    )
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--ape-zip", type=Path, required=True)
    parser.add_argument("--mm-zip", type=Path, required=True)
    parser.add_argument("--llmdet-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original_records = json.loads(args.queries.read_text(encoding="utf-8-sig"))
    manifest = build_platform_release(
        original_records=original_records,
        packages={
            "AIC_APE_Ti_zero_shot_v1.zip": args.ape_zip,
            "AIC_MM_Grounding_DINO_T_zero_shot_v1.zip": args.mm_zip,
            "AIC_LLMDet_Swin_T_zero_shot_v1.zip": args.llmdet_zip,
        },
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

