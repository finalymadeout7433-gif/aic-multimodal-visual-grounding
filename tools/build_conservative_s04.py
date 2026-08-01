from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from aic_baseline.bbox import validate_normalized_bbox
from aic_baseline.conservative_selector import (
    ConservativeSelectionResult,
    ConservativeSwitchPolicy,
    evaluate_conservative_policy,
    is_depth_query,
    select_conservative_predictions,
    write_conservative_submission,
)
from aic_baseline.diagnostics import write_jsonl
from aic_baseline.external_data import sha256_file
from aic_baseline.ranker_cache import read_candidate_cache
from aic_baseline.ranker_inference import load_fallback_predictions
from aic_baseline.ranker_training import load_ranker_model


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "verify-inputs",
    "evaluate-validation",
    "freeze-policy",
    "evaluate-holdout-once",
    "infer-aic-from-cache",
    "audit-submission",
    "build-report",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen conservative S04 AIC submission."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S04 config must be a YAML object")
    return payload


def _path(value: Any, *, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"config path is required: {name}")
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    values = config["paths"]
    return {name: _path(value, name=name) for name, value in values.items()}


def _json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dump(value), encoding="utf-8")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _public_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def _code_fingerprints() -> dict[str, str]:
    paths = (
        REPO_ROOT / "src/aic_baseline/bbox.py",
        REPO_ROOT / "src/aic_baseline/conservative_selector.py",
        REPO_ROOT / "src/aic_baseline/ranker_features.py",
        REPO_ROOT / "src/aic_baseline/ranker_training.py",
        REPO_ROOT / "src/aic_baseline/submission.py",
        Path(__file__).resolve(),
    )
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in paths
    }


def _verify_inputs(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    expected_hashes = config["expected_sha256"]
    fingerprints: dict[str, Any] = {}
    for name in (
        "validation_cache",
        "holdout_cache",
        "aic_candidate_cache",
        "aic_cache_fingerprint",
        "ranker_model",
        "aic_queries",
        "florence_fallback_predictions",
        "s02_submission_json",
    ):
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(f"required S04 input is missing: {name}")
        expected = str(expected_hashes.get(name, "")).upper()
        if not expected:
            raise ValueError(f"expected SHA-256 is required: {name}")
        actual = sha256_file(path).upper()
        if actual != expected:
            raise ValueError(
                f"input SHA-256 mismatch for {name}: "
                f"expected={expected}, actual={actual}"
            )
        fingerprints[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "verified": True,
        }

    actual_fingerprint = json.loads(
        paths["aic_cache_fingerprint"].read_text(encoding="utf-8")
    )
    expected_fingerprint = dict(config["expected_candidate_fingerprint"])
    mismatches = {
        key: {"expected": expected, "actual": actual_fingerprint.get(key)}
        for key, expected in expected_fingerprint.items()
        if actual_fingerprint.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "AIC candidate fingerprint fields do not match: "
            f"{json.dumps(mismatches, ensure_ascii=False)}"
        )
    code = _code_fingerprints()
    return {
        "inputs": fingerprints,
        "candidate_fingerprint": actual_fingerprint,
        "candidate_fingerprint_verified_fields": sorted(
            expected_fingerprint
        ),
        "code_files": code,
        "code_sha256": _canonical_sha256(code),
    }


def _policy(config: Mapping[str, Any]) -> ConservativeSwitchPolicy:
    return ConservativeSwitchPolicy(**dict(config["policy"]))


def _assert_at_least(name: str, value: float, minimum: float) -> None:
    if value + 1e-12 < minimum:
        raise ValueError(f"{name} failed: {value} < {minimum}")


def _validate_evaluation(
    split: str,
    evaluation: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "selected_acc_at_05": {
            "actual": float(evaluation["selected_acc_at_05"]),
            "minimum": float(thresholds["selected_acc_at_05_min"]),
        },
        "gain_pp": {
            "actual": float(evaluation["gain_pp"]),
            "minimum": float(thresholds["gain_pp_min"]),
        },
        "rescue_harm_ratio": {
            "actual": float(evaluation["rescue_harm_ratio"]),
            "minimum": float(thresholds["rescue_harm_ratio_min"]),
        },
    }
    for name, check in checks.items():
        _assert_at_least(
            f"{split}.{name}", check["actual"], check["minimum"]
        )
        check["passed"] = True
    return {"passed": True, "checks": checks}


def _read_aic_queries(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("AIC queries JSON must contain an object")
    records: dict[str, dict[str, Any]] = {}
    for query_id, record in payload.items():
        if not isinstance(record, dict):
            raise ValueError(f"AIC query is not an object: {query_id}")
        records[str(query_id)] = dict(record)
    return records


def _load_submission(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("submission JSON must contain an object")
    return {str(key): dict(value) for key, value in payload.items()}


def _same_non_bbox(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return {k: v for k, v in left.items() if k != "bbox"} == {
        k: v for k, v in right.items() if k != "bbox"
    }


def _audit_aic(
    *,
    original: Mapping[str, Mapping[str, Any]],
    s02: Mapping[str, Mapping[str, Any]],
    s04: Mapping[str, Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    selection: ConservativeSelectionResult,
    acceptance: Mapping[str, Any],
    submission_audit: Mapping[str, Any],
) -> dict[str, Any]:
    expected_ids = set(original)
    if set(s02) != expected_ids or set(s04) != expected_ids:
        raise ValueError("AIC/S02/S04 query ID sets differ")
    cache_by_id = {str(record["query_id"]): record for record in cache_records}
    if set(cache_by_id) != expected_ids:
        raise ValueError("AIC candidate cache ID set differs from queries")

    non_bbox_differences = 0
    s02_control_differences = 0
    bbox_differences: list[str] = []
    invalid_bbox_count = 0
    for query_id in original:
        if not _same_non_bbox(original[query_id], s04[query_id]):
            non_bbox_differences += 1
        if not _same_non_bbox(s02[query_id], s04[query_id]):
            non_bbox_differences += 1
        try:
            s02_bbox = validate_normalized_bbox(s02[query_id]["bbox"])
            s04_bbox = validate_normalized_bbox(s04[query_id]["bbox"])
        except (KeyError, TypeError, ValueError):
            invalid_bbox_count += 1
            continue
        if selection.top1_predictions[query_id] != s02_bbox:
            s02_control_differences += 1
        if s02_bbox != s04_bbox:
            bbox_differences.append(query_id)

    debug_by_id = {
        str(row["query_id"]): row for row in selection.debug_records
    }
    depth_switches = sum(
        bool(debug_by_id[query_id]["switched"])
        and is_depth_query(cache_by_id[query_id])
        for query_id in expected_ids
    )
    switch_categories = Counter(
        str(row["query_category"])
        for row in selection.debug_records
        if row["switched"]
    )
    audit = {
        "records": len(original),
        "cache_records": len(cache_records),
        "switch_count": int(selection.summary["switch_count"]),
        "switch_rate": float(selection.summary["switch_rate"]),
        "bbox_differences_vs_s02": len(bbox_differences),
        "bbox_difference_query_ids_sha256": _canonical_sha256(
            bbox_differences
        ),
        "switches_by_query_category": dict(sorted(switch_categories.items())),
        "depth_switch_count": depth_switches,
        "cross_canonical_label_switch_count": int(
            selection.summary["cross_canonical_label_switch_count"]
        ),
        "reference_dominant_switch_count": int(
            selection.summary["reference_dominant_switch_count"]
        ),
        "max_switched_area_ratio": float(
            selection.summary["max_switched_area_ratio"]
        ),
        "fallback_count": int(selection.summary["fallback_count"]),
        "invalid_bbox_count": invalid_bbox_count,
        "missing_id_count": len(expected_ids - set(s04)),
        "non_bbox_difference_count": non_bbox_differences,
        "s02_control_bbox_difference_count": s02_control_differences,
        "zip_entries": list(submission_audit["zip_entries"]),
        "zip_bad_member": submission_audit["zip_bad_member"],
    }

    expected_records = int(acceptance["records"])
    expected_differences = int(acceptance["bbox_differences_vs_s02"])
    exact_checks = {
        "records": (audit["records"], expected_records),
        "bbox_differences_vs_s02": (
            audit["bbox_differences_vs_s02"],
            expected_differences,
        ),
        "depth_switch_count": (
            audit["depth_switch_count"],
            int(acceptance["depth_switch_count"]),
        ),
        "cross_canonical_label_switch_count": (
            audit["cross_canonical_label_switch_count"],
            int(acceptance["cross_canonical_label_switch_count"]),
        ),
        "reference_dominant_switch_count": (
            audit["reference_dominant_switch_count"],
            int(acceptance["reference_dominant_switch_count"]),
        ),
        "invalid_bbox_count": (audit["invalid_bbox_count"], 0),
        "missing_id_count": (audit["missing_id_count"], 0),
        "non_bbox_difference_count": (
            audit["non_bbox_difference_count"], 0
        ),
        "s02_control_bbox_difference_count": (
            audit["s02_control_bbox_difference_count"], 0
        ),
    }
    for name, (actual, expected) in exact_checks.items():
        if actual != expected:
            raise ValueError(
                f"AIC audit failed for {name}: {actual} != {expected}"
            )
    if audit["switch_count"] != audit["bbox_differences_vs_s02"]:
        raise ValueError("AIC switches and changed bboxes do not match")
    if audit["switch_rate"] > float(acceptance["switch_rate_max"]):
        raise ValueError("AIC switch rate exceeds the frozen maximum")
    if audit["max_switched_area_ratio"] > float(
        acceptance["max_switched_area_ratio"]
    ) + 1e-12:
        raise ValueError("AIC switched area ratio exceeds the frozen maximum")
    if audit["zip_entries"] != ["predictions_submission.json"]:
        raise ValueError("S04 ZIP must contain exactly predictions_submission.json")
    audit["acceptance_passed"] = True
    return audit


def _build_report(
    *,
    config: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    validation: Mapping[str, Any],
    holdout: Mapping[str, Any],
    aic_audit: Mapping[str, Any],
    submission: Mapping[str, Any],
    platform_zip: Path,
) -> str:
    policy = config["policy"]
    baselines = config["platform_baselines"]
    return f"""# AIC S04 GroundingDINO 保守排序恢复实验报告

## 1. 执行结论

本轮已完成一个严格单变量、可手动上传平台的恢复性实验：保持 S02 GroundingDINO Top-1 为默认输出，仅在非 Top-1 候选同时通过全部安全门时允许 LightGBM Ranker 改选。

- 平台上传包：`{_public_path(platform_zip)}`
- ZIP SHA-256：`{submission['zip_sha256']}`
- AIC 记录数：{aic_audit['records']:,}
- 相对 S02 实际改框：{aic_audit['bbox_differences_vs_s02']:,} 条（{aic_audit['switch_rate']:.4%}）
- 自动上传：未执行；必须由用户手动上传。

这只能证明策略在外部 RefCOCO 系列隔离数据上更安全，不能证明 AIC 平台一定提升。真实收益必须由 `S04 - 0.4938` 计算。

## 2. 为什么要做恢复性 S04

已知平台结果为：Florence S01 `{baselines['florence_s01_acc_at_05']:.4f}`、GroundingDINO Top-1 S02 `{baselines['gdino_s02_acc_at_05']:.4f}`、无保护 LightGBM Ranker S03 `{baselines['unsafe_ranker_s03_acc_at_05']:.4f}`。S03 在 AIC 上大幅负迁移，主要风险是切换过多、训练域面积先验、目标/参照物反转和跨标签改选。

S04 没有重新训练任何模型，也没有修改候选生成。它只缩小 Ranker 的决策权限，因此是对 S02 的保守增量，而不是新的多模态模型。

## 3. 冻结输入与可追溯性

- AIC 候选缓存 SHA-256：`{fingerprints['inputs']['aic_candidate_cache']['actual_sha256']}`
- LightGBM 文本模型 SHA-256：`{fingerprints['inputs']['ranker_model']['actual_sha256']}`
- AIC Queries SHA-256：`{fingerprints['inputs']['aic_queries']['actual_sha256']}`
- S02 JSON SHA-256：`{fingerprints['inputs']['s02_submission_json']['actual_sha256']}`
- 本轮代码闭包 SHA-256：`{fingerprints['code_sha256']}`

候选模型固定为 `IDEA-Research/grounding-dino-tiny`，revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`，float32、box/text threshold 均为 0.15、Top-K=10。没有重跑候选，也没有使用 AIC 图像、伪标签或人工 bbox 训练。

## 4. 保守安全门

候选按 Ranker 分数降序检查，只有首个同时满足下列条件的候选才可替换 Top-1：

1. Ranker 相对 Top-1 margin ≥ `{policy['ranker_margin_min']}`；
2. GroundingDINO score drop ≤ `{policy['gdino_score_drop_max']}`；
3. canonical label 非空且与 Top-1 完全相同；
4. reference overlap 不得高于 target overlap；
5. 面积不得超过 Top-1 的 `{policy['max_area_ratio']}×`；
6. parser 或 query category 判为 depth 时禁止切换；
7. bbox 必须有限、合法且位于 `[0,1]`；
8. 任一字段异常时保持 Top-1。

没有启用 largest/group/region 面积例外，也没有加入 IR、Depth、Tile、PIZA、APE 或新候选模型。

## 5. 外部 validation 结果

| 指标 | 数值 |
|---|---:|
| 可评估 Query | {validation['evaluated_queries']:,} |
| Top-1 ACC@0.5 | {validation['baseline_acc_at_05']:.6f} |
| S04 ACC@0.5 | {validation['selected_acc_at_05']:.6f} |
| 增益 | {validation['gain_pp']:+.4f} pp |
| 切换 | {validation['switch_count']:,} |
| Rescue / Harm | {validation['rescues']:,} / {validation['harms']:,} |
| Rescue/Harm | {validation['rescue_harm_ratio']:.4f} |

严格实现同时禁止 `query_category=depth`，因此结果可能比早期仅按 parser depth 的只读草案更保守；本报告以最终代码和验收结果为准。

## 6. 外部 holdout 单次结果

| 指标 | 数值 |
|---|---:|
| 可评估 Query | {holdout['evaluated_queries']:,} |
| Top-1 ACC@0.5 | {holdout['baseline_acc_at_05']:.6f} |
| S04 ACC@0.5 | {holdout['selected_acc_at_05']:.6f} |
| 增益 | {holdout['gain_pp']:+.4f} pp |
| 切换 | {holdout['switch_count']:,} |
| Rescue / Harm | {holdout['rescues']:,} / {holdout['harms']:,} |
| Rescue/Harm | {holdout['rescue_harm_ratio']:.4f} |

本轮没有使用 holdout 搜索阈值；只对预先冻结的唯一策略执行一次检查。

## 7. AIC 无标签审计

| 审计项 | 结果 |
|---|---:|
| 总记录 | {aic_audit['records']:,} |
| 相对 S02 改框 | {aic_audit['bbox_differences_vs_s02']:,} |
| 切换率 | {aic_audit['switch_rate']:.4%} |
| Depth 切换 | {aic_audit['depth_switch_count']} |
| 跨 canonical label | {aic_audit['cross_canonical_label_switch_count']} |
| reference-dominant | {aic_audit['reference_dominant_switch_count']} |
| 最大面积倍率 | {aic_audit['max_switched_area_ratio']:.6f}× |
| 非法框 | {aic_audit['invalid_bbox_count']} |
| 非 bbox 字段变化 | {aic_audit['non_bbox_difference_count']} |
| S02 Top-1 不一致 | {aic_audit['s02_control_bbox_difference_count']} |

切换类别分布：`{json.dumps(aic_audit['switches_by_query_category'], ensure_ascii=False, sort_keys=True)}`。

## 8. 平台上传与结果解释

只上传 `{platform_zip.name}`，不要上传目录、模型文本或调试 JSONL。平台上传后记录提交时间、提交 ID 和 ACC@0.5：

```text
保守 Ranker 真实收益 = S04 - 0.4938
相对 Florence 收益   = S04 - 0.4980
```

- `S04 > 0.4980`：将 S04 作为新稳定基线，下一轮做视觉语义 selector 与 PIZA 小目标分支；
- `0.4938 < S04 <= 0.4980`：安全 Ranker 有迁移价值，但 Florence 仍是最佳单提交；
- `S04 <= 0.4938`：停止当前 LightGBM 主线，下一次使用新单模型。

## 9. 结论边界

S04 是恢复性排序策略，不是 GroundingDINO 微调模型，也没有使用 RGB 以外模态。外部分数来自 RefCOCO 系列，平台结果未知；不得把外部增益写成 AIC 已提升。
"""


def _write_upload_files(
    *,
    config: Mapping[str, Any],
    submission: Mapping[str, Any],
    output_root: Path,
) -> tuple[Path, Path, Path]:
    artifact_name = str(config["artifact_name"])
    source_zip = Path(str(submission["zip_path"]))
    ready = output_root / "platform_upload_ready"
    ready.mkdir(parents=True, exist_ok=True)
    platform_zip = ready / f"{artifact_name}.zip"
    shutil.copyfile(source_zip, platform_zip)
    if source_zip.read_bytes() != platform_zip.read_bytes():
        raise ValueError("platform ZIP copy is not byte-identical")
    digest = sha256_file(platform_zip)
    sums = ready / "SHA256SUMS.txt"
    sums.write_text(f"{digest}  {platform_zip.name}\n", encoding="utf-8")
    guide = ready / "UPLOAD_GUIDE.md"
    baselines = config["platform_baselines"]
    guide.write_text(
        f"""# S04 平台手动上传指南

1. 上传文件：`{platform_zip.name}`
2. 上传前 SHA-256：`{digest}`
3. ZIP 内必须且只包含：`predictions_submission.json`
4. 不要上传 `predictions_submission.zip` 以外的调试文件或模型文件。
5. 本工具没有自动登录或上传比赛平台。

平台返回后请填写：

```text
提交时间：
提交 ID：
S04 ACC@0.5：
S04 - S02 ({baselines['gdino_s02_acc_at_05']:.4f})：
S04 - Florence ({baselines['florence_s01_acc_at_05']:.4f})：
```
""",
        encoding="utf-8",
    )
    return platform_zip, sums, guide


def _write_manifest(
    output_root: Path, report_paths: Sequence[Path]
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        if path.name == "sha256_manifest.json":
            continue
        key = path.relative_to(output_root).as_posix()
        manifest[key] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    for path in report_paths:
        manifest[_public_path(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    _write_json(output_root / "sha256_manifest.json", manifest)
    return manifest


def run(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path.resolve())
    paths = _paths(config)
    output_root = paths["output_root"]
    stage_results: dict[str, Any] = {}

    fingerprints = _verify_inputs(config, paths)
    stage_results["verify-inputs"] = {"status": "completed"}
    output_root.mkdir(parents=True, exist_ok=True)
    policy = _policy(config)
    policy_path = output_root / "configuration/policy.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        yaml.safe_dump(
            asdict(policy), sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
    _write_json(
        output_root / "configuration/input_fingerprints.json",
        fingerprints,
    )

    ranker = load_ranker_model(paths["ranker_model"])
    validation_records = read_candidate_cache(paths["validation_cache"])
    validation = evaluate_conservative_policy(
        records=validation_records, ranker=ranker, policy=policy
    )
    validation["acceptance"] = _validate_evaluation(
        "validation",
        validation,
        config["acceptance"]["validation"],
    )
    validation_path = output_root / "evaluations/validation.json"
    _write_json(validation_path, validation)
    stage_results["evaluate-validation"] = {"status": "completed"}

    freeze_payload = {
        "schema_version": 1,
        "policy": asdict(policy),
        "policy_sha256": _canonical_sha256(asdict(policy)),
        "ranker_model_sha256": fingerprints["inputs"]["ranker_model"][
            "actual_sha256"
        ],
        "validation_cache_sha256": fingerprints["inputs"][
            "validation_cache"
        ]["actual_sha256"],
        "validation_evaluation_sha256": sha256_file(validation_path),
        "code_sha256": fingerprints["code_sha256"],
        "holdout_used_for_policy_search": False,
    }
    freeze_payload["freeze_sha256"] = _canonical_sha256(freeze_payload)
    _write_json(
        output_root / "model/conservative_policy.json", freeze_payload
    )
    _write_json(
        output_root / "model/ranker_reference.json",
        {
            "format": "LightGBM text model",
            "source_path": str(paths["ranker_model"]),
            "sha256": fingerprints["inputs"]["ranker_model"][
                "actual_sha256"
            ],
            "copied_into_s04": False,
            "retrained_for_s04": False,
        },
    )
    stage_results["freeze-policy"] = {"status": "completed"}

    holdout_records = read_candidate_cache(paths["holdout_cache"])
    holdout = evaluate_conservative_policy(
        records=holdout_records, ranker=ranker, policy=policy
    )
    holdout["acceptance"] = _validate_evaluation(
        "holdout", holdout, config["acceptance"]["holdout"]
    )
    _write_json(output_root / "evaluations/holdout.json", holdout)
    stage_results["evaluate-holdout-once"] = {
        "status": "completed",
        "policy_search_performed": False,
    }

    original = _read_aic_queries(paths["aic_queries"])
    cache_records = read_candidate_cache(paths["aic_candidate_cache"])
    if len(cache_records) != len(original):
        raise ValueError(
            f"AIC cache incomplete: {len(cache_records)}/{len(original)}"
        )
    fallback = load_fallback_predictions(
        paths["florence_fallback_predictions"]
    )
    selection = select_conservative_predictions(
        records=cache_records,
        ranker=ranker,
        policy=policy,
        fallback_predictions=fallback,
    )
    debug_path = output_root / "aic/selection_debug.jsonl"
    write_jsonl(debug_path, selection.debug_records)
    stage_results["infer-aic-from-cache"] = {
        "status": "completed",
        "candidate_generation_performed": False,
        "ranker_training_performed": False,
    }

    submission_dir = (
        output_root
        / "submissions"
        / str(config["artifact_name"])
    )
    submission = write_conservative_submission(
        original_records=original,
        selection=selection,
        output_dir=submission_dir,
    )
    s02 = _load_submission(paths["s02_submission_json"])
    s04 = _load_submission(Path(str(submission["json_path"])))
    aic_audit = _audit_aic(
        original=original,
        s02=s02,
        s04=s04,
        cache_records=cache_records,
        selection=selection,
        acceptance=config["acceptance"]["aic"],
        submission_audit=submission,
    )
    _write_json(output_root / "aic/selection_audit.json", aic_audit)
    platform_zip, sums_path, upload_guide = _write_upload_files(
        config=config, submission=submission, output_root=output_root
    )
    if sha256_file(platform_zip) != submission["zip_sha256"]:
        raise ValueError("platform ZIP hash differs from audited submission ZIP")
    stage_results["audit-submission"] = {"status": "completed"}

    report = _build_report(
        config=config,
        fingerprints=fingerprints,
        validation=validation,
        holdout=holdout,
        aic_audit=aic_audit,
        submission=submission,
        platform_zip=platform_zip,
    )
    paths["report_markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_markdown"].write_text(report, encoding="utf-8")
    report_summary = {
        "experiment": str(config["experiment_name"]),
        "status": "platform_upload_ready",
        "platform_scores": {
            "florence_s01": config["platform_baselines"][
                "florence_s01_acc_at_05"
            ],
            "gdino_s02": config["platform_baselines"][
                "gdino_s02_acc_at_05"
            ],
            "unsafe_ranker_s03": config["platform_baselines"][
                "unsafe_ranker_s03_acc_at_05"
            ],
            "conservative_s04": None,
        },
        "validation": validation,
        "holdout": holdout,
        "aic_audit": aic_audit,
        "platform_zip": _public_path(platform_zip),
        "platform_zip_sha256": sha256_file(platform_zip),
        "platform_uploaded": False,
    }
    _write_json(paths["report_summary_json"], report_summary)
    stage_results["build-report"] = {"status": "completed"}

    run_summary = {
        "schema_version": 1,
        "experiment": str(config["experiment_name"]),
        "artifact_name": str(config["artifact_name"]),
        "status": "platform_upload_ready",
        "stages": {
            stage: stage_results[stage] for stage in STAGES
        },
        "policy": asdict(policy),
        "input_fingerprints_path": _public_path(
            output_root / "configuration/input_fingerprints.json"
        ),
        "validation": {
            key: validation[key]
            for key in (
                "baseline_acc_at_05",
                "selected_acc_at_05",
                "gain_pp",
                "rescues",
                "harms",
                "rescue_harm_ratio",
            )
        },
        "holdout": {
            key: holdout[key]
            for key in (
                "baseline_acc_at_05",
                "selected_acc_at_05",
                "gain_pp",
                "rescues",
                "harms",
                "rescue_harm_ratio",
            )
        },
        "aic": aic_audit,
        "submission": {
            **submission,
            "platform_zip": str(platform_zip),
            "platform_zip_sha256": sha256_file(platform_zip),
            "sha256sums_path": str(sums_path),
            "upload_guide_path": str(upload_guide),
        },
        "platform_uploaded": False,
    }
    _write_json(output_root / "run_summary.json", run_summary)
    manifest = _write_manifest(
        output_root,
        (paths["report_markdown"], paths["report_summary_json"]),
    )
    run_summary["sha256_manifest_path"] = _public_path(
        output_root / "sha256_manifest.json"
    )
    run_summary["manifest_artifact_count"] = len(manifest)
    _write_json(output_root / "run_summary.json", run_summary)
    # Rebuild once because run_summary is itself covered by the manifest.
    _write_manifest(
        output_root,
        (paths["report_markdown"], paths["report_summary_json"]),
    )
    return run_summary


def main() -> None:
    args = parse_args()
    summary = run(args.config)
    print(_json_dump(summary), end="")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"S04 build failed: {error}", file=sys.stderr)
        raise
