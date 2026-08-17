from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


FILES = (
    "ann_flir.tar",
    "ann_m3fd.tar",
    "ann_mfad.tar",
    "data_flir.tar",
    "data_m3fd.tar",
    "data_mfad.tar",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset archive: {output}")
    missing = [name for name in FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing official archives: {missing}")
    manifest = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in FILES:
            path = source / name
            archive.write(path, arcname=name)
            manifest.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
        archive.writestr(
            "DATASET_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
    print(json.dumps({
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "file_count": len(FILES),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
