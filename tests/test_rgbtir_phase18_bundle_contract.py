from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_phase18_cloud_config_is_probe_only_and_preregistered() -> None:
    path = ROOT / "configs" / "aic_rgbtir_phase18.cloud.example.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["scope"] == {
        "probe_only": True,
        "forbid_full_training": True,
        "forbid_official_val": True,
        "forbid_aic_test": True,
    }
    assert config["candidates"] == [
        {"name": "D1_L010", "retention_lambda": 0.10},
        {"name": "D1_L025", "retention_lambda": 0.25},
        {"name": "D1_L050", "retention_lambda": 0.50},
    ]
    assert config["retention"]["tolerance"] == {
        "8": 0.02,
        "16": 0.02,
        "24": 0.01,
        "final": 0.0,
    }
    assert config["expected"]["probe_records"] == 4096
    assert config["expected"]["dev_records"] == 1024


def test_cloud_launcher_runs_tests_and_probe_but_never_full_train() -> None:
    script = (ROOT / "tools" / "cloud" / "run_rgbtir_phase18_probe.sh").read_text(
        encoding="utf-8"
    )

    assert "pytest" in script
    assert "run_rgbtir_phase18.py preflight" in script
    assert "run_rgbtir_phase18.py probe" in script
    assert "official" not in script.lower()
    assert "full-train" not in script.lower()


def test_candidate_overlays_vary_only_retention_lambda() -> None:
    root = ROOT / "configs" / "phase18_candidates"
    payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.yaml"))]

    assert [payload["retention_lambda"] for payload in payloads] == [0.10, 0.25, 0.50]
    common = [
        {key: value for key, value in payload.items() if key not in {"name", "retention_lambda"}}
        for payload in payloads
    ]
    assert common[0] == common[1] == common[2]


def test_phase18_full_cloud_config_is_single_candidate_and_sealed() -> None:
    path = ROOT / "configs" / "aic_rgbtir_phase18_full.cloud.example.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["scope"] == {
        "full_training": True,
        "sealed_dev_selection": True,
        "one_shot_official_val": True,
        "forbid_query_training": True,
        "forbid_fusion": True,
        "forbid_depth": True,
        "forbid_aic_test": True,
    }
    assert config["candidate"] == {
        "name": "D1_L050",
        "retention_lambda": 0.50,
    }
    assert config["runtime"]["checkpoint_fractions"] == [0.25, 0.50, 0.75, 1.0]
    assert config["runtime"]["minimum_host_ram_gib"] >= 48.0
    assert config["expected"]["full_records"] == 24612
    assert config["expected"]["dev_records"] == 1024
    assert config["expected"]["official_val_records"] == 2032
    assert config["expected"]["official_val_pairs"] == 1115


def test_phase18_full_launcher_is_gated_and_resumable() -> None:
    script = (ROOT / "tools" / "cloud" / "run_rgbtir_phase18_full_gated.sh").read_text(
        encoding="utf-8"
    )

    assert "pytest" in script
    assert "tests/test_rgbtir_phase15.py" in script
    assert "run_rgbtir_query_interface_smoke.py" in script
    assert "run_rgbtir_query_real_smoke.py" in script
    assert "run_rgbtir_phase18_full.py preflight" in script
    assert "run_rgbtir_phase18_full.py train-select" in script
    assert 'train_status="${PIPESTATUS[0]}"' in script
    assert 'PHASE_18_TRAIN_SELECT_NO_GO_ARCHIVED' in script
    assert "archive_rgbtir_phase18_stage.py" in script
    assert "run_rgbtir_phase18_full.py official-val" in script
    assert 'official_status="${PIPESTATUS[0]}"' in script
    assert '"$official_status" -ne 0 && "$official_status" -ne 2' in script
    assert "--resume" in script
    assert "finalize_rgbtir_phase18_full.sh" in script
    assert "submission" not in script.lower()


def test_phase18_full_launcher_persists_stage_boundaries_before_training() -> None:
    script = (ROOT / "tools" / "cloud" / "run_rgbtir_phase18_full_gated.sh").read_text(
        encoding="utf-8"
    )

    checkpoint_setup = 'mkdir -p "$OUTPUT_ROOT/checkpoints"'
    checkpoint_probe = 'test -w "$OUTPUT_ROOT/checkpoints"'
    train_command = "run_rgbtir_phase18_full.py train-select"

    assert checkpoint_setup in script
    assert checkpoint_probe in script
    assert script.index(checkpoint_setup) < script.index(train_command)
    assert script.index(checkpoint_probe) < script.index(train_command)
    assert "/dev/shm/aic_rgbtir_phase18_checkpoints" not in script
    assert "AIC_DISABLE_VALIDATION_CACHE=1" not in script
    assert "export AIC_DISABLE_VALIDATION_CACHE" not in script
    assert 'rm -rf "$checkpoint_target"' not in script
    assert "--minimum-checkpoints 4" in script


def test_phase18_full_launcher_records_failure_without_deleting_assets() -> None:
    script = (ROOT / "tools" / "cloud" / "run_rgbtir_phase18_full_gated.sh").read_text(
        encoding="utf-8"
    )

    assert "preserve_failure_state" in script
    assert "trap preserve_failure_state EXIT" in script
    assert "runtime_failure_state.json" in script
    assert 'rm -rf "$OUTPUT_ROOT/checkpoints"' not in script
    assert 'rm -f "$OUTPUT_ROOT/rgb_teacher_bank/teacher_bank.partial.pt"' not in script


def test_phase18_bundle_manifest_is_written_as_utf8_without_bom() -> None:
    manifest = ROOT / "BUNDLE_MANIFEST.json"
    if manifest.is_file():
        raw = manifest.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        yaml.safe_load(raw.decode("utf-8"))
        return

    script = (ROOT / "tools" / "prepare_rgbtir_phase18_full_cloud_bundle.ps1").read_text(
        encoding="utf-8"
    )
    assert "System.Text.UTF8Encoding($false)" in script
    assert "WriteAllText" in script
    assert "BUNDLE_MANIFEST.json\") -Encoding utf8" not in script


def test_phase18_bundle_contains_the_verified_rgbt_extractor() -> None:
    extractor = ROOT / "tools" / "cloud" / "extract_rgbt_groundbench.sh"
    if (ROOT / "BUNDLE_MANIFEST.json").is_file():
        assert extractor.is_file()
        assert "sha256sum" in extractor.read_text(encoding="utf-8")
        return

    script = (ROOT / "tools" / "prepare_rgbtir_phase18_full_cloud_bundle.ps1").read_text(
        encoding="utf-8"
    )
    assert '"tools\\cloud\\extract_rgbt_groundbench.sh"' in script
