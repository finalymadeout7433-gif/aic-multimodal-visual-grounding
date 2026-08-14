from __future__ import annotations

import json
import zipfile
from pathlib import Path

from aic_baseline.platform_release import build_platform_release


def _package(path: Path, bbox: list[float]) -> None:
    payload = {
        "q1": {
            "visible": "Images/visible/a.png",
            "infrared": "Images/infrared/a.png",
            "depth": "Images/depth/a.png",
            "query": "target",
            "bbox": bbox,
        }
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "predictions_submission.json",
            json.dumps(payload, ensure_ascii=False),
        )


def test_build_platform_release_verifies_and_copies_three_packages(
    tmp_path: Path,
) -> None:
    original = {
        "q1": {
            "visible": "Images/visible/a.png",
            "infrared": "Images/infrared/a.png",
            "depth": "Images/depth/a.png",
            "query": "target",
        }
    }
    packages = {}
    for index, name in enumerate(("ape.zip", "mm.zip", "llm.zip"), start=1):
        source = tmp_path / name
        _package(source, [0.1, 0.1, 0.4 + index * 0.01, 0.5])
        packages[name] = source

    manifest = build_platform_release(
        original_records=original,
        packages=packages,
        output_dir=tmp_path / "ready",
    )

    assert manifest["package_count"] == 3
    assert set(manifest["packages"]) == set(packages)
    assert (tmp_path / "ready" / "SHA256SUMS.txt").is_file()
    guide = (tmp_path / "ready" / "UPLOAD_GUIDE.md").read_text(encoding="utf-8")
    for name in packages:
        assert (tmp_path / "ready" / name).read_bytes() == packages[name].read_bytes()
        assert name in guide
        assert manifest["packages"][name]["sha256"] in guide
