from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from aic_baseline.aic_full_detection import sha256_file
from aic_baseline.platform_release import _verify_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and collect LocateAnything/Qwen3-VL AIC ZIPs."
    )
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--locateanything-zip", type=Path)
    parser.add_argument("--qwen3vl-zip", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> int:
    args = parse_args()
    original_records = json.loads(args.queries.read_text(encoding="utf-8-sig"))
    packages: dict[str, Path] = {}
    if args.locateanything_zip:
        packages["AIC_LocateAnything_3B_zero_shot_v1.zip"] = args.locateanything_zip
    if args.qwen3vl_zip:
        packages["AIC_Qwen3_VL_8B_Instruct_zero_shot_v1.zip"] = args.qwen3vl_zip
    if not packages:
        raise ValueError("at least one package must be provided")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest_packages = {}
    checksum_lines = []
    for name, source in packages.items():
        audit = _verify_package(path=source, original_records=original_records)
        target = output / name
        shutil.copyfile(source, target)
        if sha256_file(target) != audit["sha256"]:
            raise RuntimeError(f"copied package hash mismatch: {name}")
        manifest_packages[name] = {
            **audit,
            "source": str(source.resolve()),
            "output": str(target.resolve()),
        }
        checksum_lines.append(f"{audit['sha256']}  {name}")

    manifest = {
        "package_count": len(manifest_packages),
        "query_count_per_package": len(original_records),
        "packages": manifest_packages,
    }
    _write_text(
        output / "release_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text(output / "SHA256SUMS.txt", "\n".join(checksum_lines) + "\n")
    _write_text(
        output / "UPLOAD_GUIDE.md",
        "\n".join(
            [
                "# AIC LocateAnything/Qwen3-VL Upload Guide",
                "",
                "Upload the ZIP files in this directory directly. Do not unzip or recompress them.",
                "",
                "| File | SHA-256 |",
                "|---|---|",
                *[
                    f"| `{name}` | `{audit['sha256']}` |"
                    for name, audit in manifest_packages.items()
                ],
                "",
                "Record the platform submission time, submission ID, and ACC@0.5 after upload.",
                "",
            ]
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
