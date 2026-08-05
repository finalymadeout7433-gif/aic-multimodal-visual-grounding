from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from aic_baseline.aic_full_detection import sha256_file
from tools.cloud.build_two_model_platform_release import main as build_release_main
from tools.cloud.run_locateanything_aic import LocateAnythingPredictor
from tools.cloud.run_qwen3vl_aic import _extract_box, _to_pixel_bbox


def _submission_zip(path: Path, bbox: list[float]) -> None:
    payload = {
        "q1": {
            "visible": "Images/visible/a.png",
            "infrared": "Images/infrared/a.png",
            "depth": "Images/depth/a.png",
            "query": "the target",
            "bbox": bbox,
        }
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "predictions_submission.json",
            json.dumps(payload, ensure_ascii=False),
        )


def test_qwen3vl_extracts_json_bbox_variants() -> None:
    assert _extract_box('{"bbox":[0.1,0.2,0.3,0.4]}') == [0.1, 0.2, 0.3, 0.4]
    assert _extract_box("[10,20,30,40]") == [10.0, 20.0, 30.0, 40.0]
    assert _extract_box("answer: [100, 200, 300, 400]") == [
        100.0,
        200.0,
        300.0,
        400.0,
    ]
    with pytest.raises(ValueError, match="no bbox found"):
        _extract_box("there is no valid box here")


def test_qwen3vl_to_pixel_bbox_handles_normalized_locate_and_pixel_scales() -> None:
    assert _to_pixel_bbox([0.1, 0.2, 0.3, 0.4], width=1000, height=500) == [
        100.0,
        100.0,
        300.0,
        200.0,
    ]
    assert _to_pixel_bbox([100, 200, 300, 400], width=1000, height=500) == [
        100.0,
        100.0,
        300.0,
        200.0,
    ]
    assert _to_pixel_bbox([1200, 200, 1800, 600], width=2000, height=1000) == [
        1200,
        200,
        1800,
        600,
    ]


def test_locateanything_parses_first_locate_box_without_model_init() -> None:
    class Worker:
        def ground_multi(self, image, query: str):
            return {
                "answer": (
                    "<ref>camera</ref><box><100><200><300><400></box>"
                    "<ref>sign</ref><box><500><600><700><800></box>"
                )
            }

    predictor = object.__new__(LocateAnythingPredictor)
    predictor.worker = Worker()
    prediction = predictor.predict(image=type("Img", (), {"size": (2000, 1000)})(), query="target")

    assert len(prediction.candidates) == 2
    assert prediction.candidates[0].pixel_bbox == [200.0, 200.0, 600.0, 400.0]
    assert prediction.candidates[0].label == "camera"
    assert prediction.candidates[0].score == 0.0
    assert prediction.candidates[1].pixel_bbox == [1000.0, 600.0, 1400.0, 800.0]
    assert prediction.candidates[1].label == "sign"
    assert prediction.candidates[1].score == -1.0


def test_two_model_release_collects_one_or_two_submission_zips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            {
                "q1": {
                    "visible": "Images/visible/a.png",
                    "infrared": "Images/infrared/a.png",
                    "depth": "Images/depth/a.png",
                    "query": "the target",
                }
            }
        ),
        encoding="utf-8",
    )
    locate_zip = tmp_path / "locate.zip"
    qwen_zip = tmp_path / "qwen.zip"
    _submission_zip(locate_zip, [0.1, 0.2, 0.3, 0.4])
    _submission_zip(qwen_zip, [0.2, 0.3, 0.4, 0.5])
    output = tmp_path / "ready"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_two_model_platform_release.py",
            "--queries",
            str(queries),
            "--locateanything-zip",
            str(locate_zip),
            "--qwen3vl-zip",
            str(qwen_zip),
            "--output-dir",
            str(output),
        ],
    )

    assert build_release_main() == 0
    assert (output / "AIC_LocateAnything_3B_zero_shot_v1.zip").read_bytes() == locate_zip.read_bytes()
    assert (output / "AIC_Qwen3_VL_8B_Instruct_zero_shot_v1.zip").read_bytes() == qwen_zip.read_bytes()
    assert sha256_file(output / "AIC_LocateAnything_3B_zero_shot_v1.zip") in (
        output / "SHA256SUMS.txt"
    ).read_text(encoding="utf-8")
    assert "AIC_Qwen3_VL_8B_Instruct_zero_shot_v1.zip" in (
        output / "UPLOAD_GUIDE.md"
    ).read_text(encoding="utf-8")


def test_cloud_shell_scripts_parse_with_bash() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = sorted((root / "tools" / "cloud").glob("*.sh"))
    for script in scripts:
        if os.name == "nt":
            wsl = shutil.which("wsl.exe")
            if wsl is None:
                pytest.skip("WSL is not available on this Windows environment")
            resolved = script.resolve()
            drive = resolved.drive.rstrip(":").lower()
            wsl_path = f"/mnt/{drive}/{resolved.relative_to(resolved.anchor).as_posix()}"
            completed = subprocess.run(
                [wsl, "--", "bash", "-lc", f"bash -n {shlex.quote(wsl_path)}"],
                check=False,
                capture_output=True,
                text=True,
            )
        else:
            bash = shutil.which("bash")
            if bash is None:
                pytest.skip("bash is not available")
            completed = subprocess.run(
                [bash, "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
        assert completed.returncode == 0, f"{script}: {completed.stderr}"
