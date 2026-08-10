from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aic_baseline.testset_profile_pipeline import (
    _build_small_proxies,
    build_optimization_triggers,
    resolve_frozen_trigger_thresholds,
    run_profile_pipeline,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_small_proxy_does_not_count_s03_s04_as_independent_models() -> None:
    query_profiles = [
        {
            "query_id": "q1",
            "image_group_id": "1",
            "instance_small_prior": True,
        }
    ]
    model_profiles = [
        {
            "query_id": "q1",
            "areas": {"s01": 0.20, "s02": 0.004, "s03": 0.004, "s04": 0.004},
            "pairwise_iou": {"s01_s02": 0.0, "s02_s03": 1.0, "s02_s04": 1.0},
            "topk_target_compatible_areas": [],
        }
    ]

    [proxy] = _build_small_proxies(query_profiles, model_profiles)

    assert proxy["small_target_proxy"] == "medium_confidence_small"
    assert proxy["small_model_signals"] == ["gdino_s02_top1"]
    assert proxy["dependent_selector_areas_excluded"] == ["s03", "s04"]


def _make_fixture(tmp_path: Path) -> dict[str, object]:
    dataset_root = tmp_path / "dataset"
    for group, suffix in (("000001", "png"), ("000002", "jpg")):
        visible = np.full((12, 16, 3), 60 if suffix == "png" else 160, dtype=np.uint8)
        infrared = np.full((12, 16, 3), 90, dtype=np.uint8)
        if suffix == "png":
            depth = np.full((12, 16), 1500, dtype=np.uint16)
        else:
            depth = np.full((12, 16, 3), 80, dtype=np.uint8)
        for modality, value in (
            ("visible", visible),
            ("infrared", infrared),
            ("depth", depth),
        ):
            path = dataset_root / "Images" / modality / f"{group}.{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(value).save(path)

    queries = {
        "000001_001": {
            "visible": "Images/visible/000001.png",
            "infrared": "Images/infrared/000001.png",
            "depth": "Images/depth/000001.png",
            "query": "The leftmost security camera",
        },
        "000001_002": {
            "visible": "Images/visible/000001.png",
            "infrared": "Images/infrared/000001.png",
            "depth": "Images/depth/000001.png",
            "query": "The passage beside the building",
        },
        "000002_001": {
            "visible": "Images/visible/000002.jpg",
            "infrared": "Images/infrared/000002.jpg",
            "depth": "Images/depth/000002.jpg",
            "query": "The red car",
        },
    }
    queries_path = dataset_root / "queries" / "queries.json"
    _write_json(queries_path, queries)

    boxes = {
        "s01": {
            "000001_001": [0.01, 0.01, 0.05, 0.05],
            "000001_002": [0.1, 0.1, 0.8, 0.8],
            "000002_001": [0.2, 0.2, 0.6, 0.6],
        },
        "s02": {
            "000001_001": {"bbox": [0.012, 0.012, 0.052, 0.052]},
            "000001_002": {"bbox": [0.12, 0.12, 0.82, 0.82]},
            "000002_001": {"bbox": [0.21, 0.21, 0.61, 0.61]},
        },
        "s03": {
            "000001_001": {"bbox": [0.0, 0.0, 0.5, 0.5]},
            "000001_002": {"bbox": [0.0, 0.0, 0.95, 0.95]},
            "000002_001": {"bbox": [0.22, 0.22, 0.62, 0.62]},
        },
        "s04": {
            "000001_001": {"bbox": [0.012, 0.012, 0.052, 0.052]},
            "000001_002": {"bbox": [0.12, 0.12, 0.82, 0.82]},
            "000002_001": {"bbox": [0.21, 0.21, 0.61, 0.61]},
        },
    }
    prediction_paths: dict[str, Path] = {}
    for name, payload in boxes.items():
        path = tmp_path / f"{name}.json"
        _write_json(path, payload)
        prediction_paths[name] = path

    candidate_path = tmp_path / "candidates.jsonl"
    _write_jsonl(
        candidate_path,
        [
            {
                "query_id": query_id,
                "query": record["query"],
                "image_relpath": record["visible"],
                "candidates": [
                    {
                        "bbox": boxes["s02"][query_id]["bbox"],
                        "label": (
                            "security camera"
                            if "camera" in record["query"]
                            else "object"
                        ),
                        "score": 0.8,
                    },
                    {
                        "bbox": boxes["s03"][query_id]["bbox"],
                        "label": (
                            "building" if "passage" in record["query"] else "object"
                        ),
                        "score": 0.5,
                    },
                ],
            }
            for query_id, record in queries.items()
        ],
    )
    s03_debug = tmp_path / "s03_debug.jsonl"
    s04_debug = tmp_path / "s04_debug.jsonl"
    _write_jsonl(
        s03_debug,
        [{"query_id": query_id, "ranker_index": 1} for query_id in queries],
    )
    _write_jsonl(
        s04_debug,
        [
            {"query_id": query_id, "selected_index": 0, "switched": False}
            for query_id in queries
        ],
    )

    external = tmp_path / "refcoco_validation.jsonl"
    _write_jsonl(
        external,
        [
            {
                "query_id": "external_1",
                "query": "the red car",
                "image_relpath": "external.jpg",
                "bbox_xyxy_normalized": [0.1, 0.1, 0.4, 0.4],
            }
        ],
    )
    return {
        "paths": {
            "dataset_root": str(dataset_root),
            "queries": str(queries_path),
            "s01_predictions": str(prediction_paths["s01"]),
            "s02_predictions": str(prediction_paths["s02"]),
            "s03_predictions": str(prediction_paths["s03"]),
            "s04_predictions": str(prediction_paths["s04"]),
            "candidate_cache": str(candidate_path),
            "s03_debug": str(s03_debug),
            "s04_debug": str(s04_debug),
            "output_root": str(tmp_path / "outputs"),
            "report_root": str(tmp_path / "reports"),
        },
        "external_manifests": {
            "refcoco": str(external),
            "sorec": None,
            "rgbt_groundbench": None,
        },
        "expected_counts": {"queries": 3, "image_groups": 2},
        "probe": {"per_group": 1, "seed": 20260801},
        "scale_probe": {"enabled": False},
    }


def test_pipeline_builds_deterministic_read_only_profile(tmp_path: Path) -> None:
    config = _make_fixture(tmp_path)
    query_path = Path(config["paths"]["queries"])
    before = query_path.read_bytes()

    first = run_profile_pipeline(config, repo_root=tmp_path)
    machine_files = [
        "query_profile.jsonl",
        "image_group_profile.csv",
        "model_behavior_profile.jsonl",
        "small_target_proxy.csv",
        "domain_shift_metrics.json",
        "run_summary.json",
        "sha256_manifest.json",
    ]
    snapshots = {
        name: (tmp_path / "outputs" / name).read_bytes() for name in machine_files
    }
    second = run_profile_pipeline(config, repo_root=tmp_path, resume=True)

    assert first["query_count"] == second["query_count"] == 3
    assert first["image_group_count"] == 2
    assert query_path.read_bytes() == before
    assert all(
        (tmp_path / "outputs" / name).read_bytes() == value
        for name, value in snapshots.items()
    )
    assert not list((tmp_path / "outputs").rglob("*.zip"))
    assert not list((tmp_path / "outputs").rglob("*.pth"))
    assert (tmp_path / "reports" / "aic_testset_full_profile.md").is_file()
    assert (tmp_path / "reports" / "optimization_trigger_matrix.md").is_file()
    next_experiments = (tmp_path / "reports" / "next_three_experiments.md").read_text(
        encoding="utf-8"
    )
    assert "APE-Ti 单模型对照" in next_experiments
    assert "PIZA/SOREC 小目标离线专项" in next_experiments
    assert "Depth late-fusion tracer" in next_experiments
    assert "MM-Grounding-DINO-T 多域验证 tracer" not in next_experiments
    contact_manifest = json.loads(
        (tmp_path / "outputs" / "contact_sheet_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert contact_manifest["warning"] == "MODEL PREDICTIONS — NOT GROUND TRUTH"
    assert all(
        len(cluster["selected"]) <= 25
        for cluster in contact_manifest["clusters"].values()
    )

    lines = (
        (tmp_path / "outputs" / "query_profile.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 3
    depth_rows = (tmp_path / "outputs" / "image_group_profile.csv").read_text(
        encoding="utf-8-sig"
    )
    assert "jpeg_uint8_unknown" in depth_rows


def test_pipeline_rejects_incomplete_prediction_ids(tmp_path: Path) -> None:
    config = _make_fixture(tmp_path)
    s04 = Path(config["paths"]["s04_predictions"])
    payload = json.loads(s04.read_text(encoding="utf-8"))
    payload.pop("000002_001")
    _write_json(s04, payload)

    with pytest.raises(ValueError, match="s04_predictions.*Query ID"):
        run_profile_pipeline(config, repo_root=tmp_path)


def test_trigger_matrix_uses_frozen_thresholds() -> None:
    triggers = build_optimization_triggers(
        {
            "query_count": 100,
            "high_confidence_small_ratio": 0.18,
            "medium_confidence_small_ratio": 0.16,
            "region_structure_ratio": 0.04,
            "region_all_disagree_ratio": 0.2,
            "low_light_group_ratio": 0.08,
            "usable_ir_within_low_light_ratio": 0.9,
            "depth_relation_ratio": 0.18,
        },
        {"status": "complete", "small_stable_candidate_gain_pp": 12.0},
    )

    assert triggers["recommended_next_model"] == "PIZA"
    assert triggers["piza"]["triggered"] is True
    assert triggers["ape_ti"]["triggered"] is False
    assert triggers["rgbt_audit"]["triggered"] is False


def test_trigger_matrix_keeps_scale_dependent_decision_pending() -> None:
    triggers = build_optimization_triggers(
        {
            "query_count": 100,
            "high_confidence_small_ratio": 0.2,
            "medium_confidence_small_ratio": 0.2,
            "region_structure_ratio": 0.12,
            "region_all_disagree_ratio": 0.5,
            "low_light_group_ratio": 0.2,
            "usable_ir_within_low_light_ratio": 0.9,
            "depth_relation_ratio": 0.18,
        },
        {"status": "not_run", "small_stable_candidate_gain_pp": None},
    )

    assert triggers["piza"]["status"] == "pending_scale_probe"
    assert triggers["ape_ti"]["triggered"] is True
    assert triggers["rgbt_audit"]["triggered"] is True
    assert triggers["recommended_next_model"] == "APE-Ti"


def test_pipeline_rejects_post_hoc_trigger_threshold_drift() -> None:
    with pytest.raises(ValueError, match="differ from the pre-S04 policy"):
        resolve_frozen_trigger_thresholds(
            {"frozen_trigger_thresholds": {"piza_high_small_ratio": 0.99}}
        )
