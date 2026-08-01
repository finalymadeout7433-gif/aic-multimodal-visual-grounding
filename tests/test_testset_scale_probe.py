from __future__ import annotations

from aic_baseline.testset_scale_probe import (
    candidate_generation_fingerprint,
    grid_windows,
    scale_record_metrics,
    summarize_scale_probe,
)


def test_candidate_fingerprint_excludes_thermal_cooldown() -> None:
    base = {
        "model_path": "model",
        "model_revision": "revision",
        "box_threshold": 0.15,
        "text_threshold": 0.15,
        "inter_query_cooldown_seconds": 0.0,
        "stage2_inter_query_cooldown_seconds": 0.0,
    }
    cooled = {
        **base,
        "inter_query_cooldown_seconds": 2.0,
        "stage2_inter_query_cooldown_seconds": 5.0,
    }

    assert candidate_generation_fingerprint(base) == candidate_generation_fingerprint(
        cooled
    )


def test_grid_windows_builds_deterministic_three_by_three_cover() -> None:
    windows = grid_windows(width=1000, height=500, rows=3, columns=3, overlap=0.2)

    assert len(windows) == 9
    assert windows[0].as_box() == [0, 0, 385, 192]
    assert windows[-1].as_box() == [615, 308, 1000, 500]
    assert min(window.x1 for window in windows) == 0
    assert max(window.x2 for window in windows) == 1000
    assert min(window.y1 for window in windows) == 0
    assert max(window.y2 for window in windows) == 500


def test_scale_record_requires_semantics_and_independent_position_support() -> None:
    record = scale_record_metrics(
        target_phrase="security camera",
        florence_bbox=[0.10, 0.10, 0.14, 0.14],
        full_candidates=[
            {
                "bbox": [0.7, 0.7, 0.75, 0.75],
                "label": "security camera",
                "source": "full",
            }
        ],
        high_resolution_candidates=[
            {
                "bbox": [0.105, 0.105, 0.145, 0.145],
                "label": "security camera",
                "source": "highres",
            }
        ],
        tile_candidates=[
            {
                "bbox": [0.11, 0.11, 0.15, 0.15],
                "label": "security camera",
                "source": "tile_0",
            }
        ],
    )

    assert record["full_stable_target_candidate"] is False
    assert record["enhanced_stable_target_candidate"] is True
    assert record["enhanced_small_support"] is True
    assert record["full_miss_tile_only_target_candidate"] is False


def test_scale_summary_reports_proxy_gain_not_accuracy() -> None:
    summary = summarize_scale_probe(
        [
            {
                "probe_group": "high_confidence_small",
                "stage1_complete": True,
                "full_stable_target_candidate": False,
                "enhanced_stable_target_candidate": True,
                "tile_candidate_count": 3,
                "full_candidate_count": 1,
            },
            {
                "probe_group": "medium_confidence_small",
                "stage1_complete": True,
                "full_stable_target_candidate": True,
                "enhanced_stable_target_candidate": True,
                "tile_candidate_count": 2,
                "full_candidate_count": 2,
            },
            {
                "probe_group": "ordinary_control",
                "stage1_complete": True,
                "full_stable_target_candidate": True,
                "enhanced_stable_target_candidate": True,
                "tile_candidate_count": 6,
                "full_candidate_count": 2,
            },
        ]
    )

    assert summary["status"] == "complete"
    assert summary["small_stable_candidate_gain_pp"] == 50.0
    assert summary["ordinary_control_candidate_inflation"] == 3.0
    assert "acc" not in summary
    assert summary["evidence_kind"] == "model_derived_proxy"
