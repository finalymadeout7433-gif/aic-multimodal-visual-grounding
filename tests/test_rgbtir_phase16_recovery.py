from __future__ import annotations

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]


def test_recovery_config_is_c1_c2_only_and_uses_compact_teacher_bank() -> None:
    config = yaml.safe_load(
        (REPO / "configs" / "aic_rgbtir_phase16_recovery.cloud.example.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["recovery"] == {
        "enabled": True,
        "candidates": ["C1", "C2"],
        "forbid_full_training": True,
        "forbid_official_val": True,
    }
    assert config["runtime"]["teacher_bank_manifests"] == [
        "repair_probe_train.jsonl",
        "repair_dev.jsonl",
    ]
    assert config["runtime"]["full_steps"] == 0


def test_cloud_recovery_script_cannot_enter_full_or_official_validation() -> None:
    script = (REPO / "tools" / "cloud" / "run_rgbtir_phase16_recovery.sh").read_text(
        encoding="utf-8"
    )

    assert "run_rgbtir_phase16.py train" in script
    assert "run_rgbtir_phase16.py all" not in script
    assert "probe_candidates/C1/adapter.pt" in script
    assert "probe_candidates/C2/adapter.pt" in script
    assert "rgb_teacher_bank/teacher_bank.pt" in script
    assert "recovery_diagnostics/C2.json" in script
    assert "run_rgbtir_phase17a_plus.py" in script
    assert "sha256sum" in script
