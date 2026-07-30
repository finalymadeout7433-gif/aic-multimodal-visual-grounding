from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from aic_baseline.diagnostic_report import decide_next_actions
from aic_baseline.diagnostics import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the unattended diagnostic round report."
    )
    parser.add_argument("--subset-summary", type=Path, required=True)
    parser.add_argument("--florence-dir", type=Path, required=True)
    parser.add_argument("--florence-analysis", type=Path, required=True)
    parser.add_argument("--tile-dir", type=Path, required=True)
    parser.add_argument("--gdino-dir", type=Path)
    parser.add_argument("--aic-smoke-summary", type=Path)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _rate(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(bool(record[key]) for record in records) / len(records)


def _tile_comparison(
    full_records: list[dict[str, Any]],
    tile_records: list[dict[str, Any]],
) -> dict[str, Any]:
    full = {str(record["query_id"]): record for record in full_records}
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for tile_record in tile_records:
        query_id = str(tile_record["query_id"])
        if query_id not in full:
            raise ValueError(f"tile query missing from full evaluation: {query_id}")
        group = str(tile_record.get("probe_group"))
        groups[group].append((full[query_id], tile_record))

    output: dict[str, Any] = {}
    for group, pairs in sorted(groups.items()):
        full_group = [pair[0] for pair in pairs]
        tile_group = [pair[1] for pair in pairs]
        output[group] = {
            "count": len(pairs),
            "full_first_acc_at_05": _rate(
                full_group, "selected_acc_at_05"
            ),
            "full_oracle_acc_at_05": _rate(
                full_group, "oracle_acc_at_05"
            ),
            "full_plus_tile_oracle_acc_at_05": _rate(
                tile_group, "oracle_acc_at_05"
            ),
            "oracle_gain": (
                _rate(tile_group, "oracle_acc_at_05")
                - _rate(full_group, "oracle_acc_at_05")
            ),
            "full_candidate_count_mean": sum(
                len(record["candidates"]) for record in full_group
            )
            / len(full_group),
            "full_plus_tile_candidate_count_mean": sum(
                len(record["candidates"]) for record in tile_group
            )
            / len(tile_group),
            "latency_multiplier": (
                sum(record["latency_ms"] for record in tile_group)
                / sum(record["latency_ms"] for record in full_group)
            ),
        }
    return output


def main() -> int:
    args = parse_args()
    subset = json.loads(args.subset_summary.read_text(encoding="utf-8"))
    florence_summary = json.loads(
        (args.florence_dir / "summary.json").read_text(encoding="utf-8")
    )
    florence_analysis = json.loads(
        args.florence_analysis.read_text(encoding="utf-8")
    )
    tile_summary = json.loads(
        (args.tile_dir / "summary.json").read_text(encoding="utf-8")
    )
    full_records = read_jsonl(args.florence_dir / "predictions.jsonl")
    tile_records = read_jsonl(args.tile_dir / "predictions.jsonl")
    tile_comparison = _tile_comparison(full_records, tile_records)
    models = json.loads(args.model_manifest.read_text(encoding="utf-8"))

    gdino_summary = None
    gdino_complete = False
    if args.gdino_dir and (args.gdino_dir / "summary.json").exists():
        gdino_summary = json.loads(
            (args.gdino_dir / "summary.json").read_text(encoding="utf-8")
        )
        gdino_complete = (
            gdino_summary["overall"]["count"]
            == florence_summary["overall"]["count"]
        )
    aic_smoke = None
    if args.aic_smoke_summary and args.aic_smoke_summary.exists():
        aic_smoke = json.loads(
            args.aic_smoke_summary.read_text(encoding="utf-8")
        )

    gdino_top1 = (
        gdino_summary["overall"]["selected_acc_at_05"]
        if gdino_summary and gdino_complete
        else None
    )
    gdino_top10 = (
        gdino_summary["overall"]["top10_oracle_acc_at_05"]
        if gdino_summary and gdino_complete
        else None
    )
    decisions = decide_next_actions(
        florence_first=florence_summary["overall"]["selected_acc_at_05"],
        florence_oracle=florence_summary["overall"]["oracle_acc_at_05"],
        tile_small_full_oracle=tile_comparison["small"][
            "full_oracle_acc_at_05"
        ],
        tile_small_combined_oracle=tile_comparison["small"][
            "full_plus_tile_oracle_acc_at_05"
        ],
        tile_control_full_oracle=tile_comparison["control"][
            "full_oracle_acc_at_05"
        ],
        tile_control_combined_oracle=tile_comparison["control"][
            "full_plus_tile_oracle_acc_at_05"
        ],
        gdino_top1=gdino_top1,
        gdino_top10_oracle=gdino_top10,
    )

    round_summary = {
        "scope": {
            "training_performed": False,
            "aic_submission_generated": False,
            "aic_formal_labels_used": False,
            "holdout_used": False,
        },
        "subsets": subset,
        "models": models,
        "florence": florence_summary,
        "florence_candidate_analysis": {
            key: florence_analysis[key]
            for key in (
                "record_count",
                "outcome_counts",
                "candidate_semantic_type_counts",
                "candidate_label_role_counts",
            )
        },
        "tile": {
            "summary": tile_summary,
            "comparison": tile_comparison,
        },
        "grounding_dino": {
            "completed_full_core": gdino_complete,
            "summary": gdino_summary,
        },
        "aic_smoke": aic_smoke,
        "decisions": decisions,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(round_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    f = florence_summary["overall"]
    t_small = tile_comparison["small"]
    t_control = tile_comparison["control"]
    lines = [
        "# Florence 候选上限、Tile 与 GroundingDINO 零样本诊断报告",
        "",
        "## 1. 目的与边界",
        "",
        "本轮只做外部 validation 诊断与无标签 AIC smoke；没有训练模型、"
        "没有生成正式提交、没有使用 AIC 测试标签，也没有触碰 holdout。",
        "",
        "## 2. 固定数据清单",
        "",
        f"- `core_eval_1500`：{subset['core']['records']} 条，"
        f"{subset['core']['unique_image_keys']} 张互不重复图像；"
        f"选择哈希 `{subset['core']['selection_sha256']}`。",
        f"- `tile_probe_600`：{subset['tile_probe']['records']} 条；"
        f"选择哈希 `{subset['tile_probe']['selection_sha256']}`。",
        f"- 源 validation 清单哈希："
        f"`{subset['source_manifest_sha256']}`。",
        "",
        "## 3. Florence full-image first 与 oracle",
        "",
        f"- first ACC@0.5：{f['selected_acc_at_05']:.4f}",
        f"- candidate oracle ACC@0.5：{f['oracle_acc_at_05']:.4f}",
        f"- oracle 差值：{f['oracle_gap']:.4f}",
        f"- first mean IoU：{f['selected_iou_mean']:.4f}",
        f"- oracle mean IoU：{f['best_candidate_iou_mean']:.4f}",
        f"- 无候选率：{f['no_candidate_rate']:.4f}",
        f"- 平均候选数：{f['candidate_count_mean']:.3f}",
        f"- 平均延迟：{f['latency_ms_mean']:.1f} ms",
        f"- GPU 峰值分配显存："
        f"{florence_summary['execution']['cuda_peak_allocated_bytes'] / 2**30:.2f} GiB",
        "",
        "候选结果计数：",
        "",
    ]
    lines.extend(
        f"- `{key}`：{value}"
        for key, value in florence_analysis["outcome_counts"].items()
    )
    lines.extend(
        [
            "",
            "## 4. 候选短语角色",
            "",
        ]
    )
    lines.extend(
        f"- `{key}`：{value}"
        for key, value in florence_analysis[
            "candidate_label_role_counts"
        ].items()
    )
    lines.extend(
        [
            "",
            "详细案例见 `reports/florence_candidate_analysis.md` 和本地 "
            "`outputs/diagnostics/analysis/visualizations/`。",
            "",
            "## 5. Tile pilot",
            "",
            "| 组别 | full first | full oracle | full+tile oracle | oracle 增益 | "
            "候选数增长 | 延迟倍率 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| 小目标 | {t_small['full_first_acc_at_05']:.4f} | "
            f"{t_small['full_oracle_acc_at_05']:.4f} | "
            f"{t_small['full_plus_tile_oracle_acc_at_05']:.4f} | "
            f"{t_small['oracle_gain']:+.4f} | "
            f"{t_small['full_candidate_count_mean']:.2f}→"
            f"{t_small['full_plus_tile_candidate_count_mean']:.2f} | "
            f"{t_small['latency_multiplier']:.2f}× |",
            f"| 中/大目标对照 | {t_control['full_first_acc_at_05']:.4f} | "
            f"{t_control['full_oracle_acc_at_05']:.4f} | "
            f"{t_control['full_plus_tile_oracle_acc_at_05']:.4f} | "
            f"{t_control['oracle_gain']:+.4f} | "
            f"{t_control['full_candidate_count_mean']:.2f}→"
            f"{t_control['full_plus_tile_candidate_count_mean']:.2f} | "
            f"{t_control['latency_multiplier']:.2f}× |",
            "",
            "## 6. GroundingDINO-Tiny",
            "",
        ]
    )
    if gdino_summary:
        g = gdino_summary["overall"]
        qualification = (
            "完整 1,500 条，可与 Florence 受控比较"
            if gdino_complete
            else "仅 pilot，不与 Florence 完整结果作等量结论"
        )
        lines.extend(
            [
                f"- 样本数：{g['count']}（{qualification}）",
                f"- top-1 ACC@0.5：{g['selected_acc_at_05']:.4f}",
                f"- top-5 oracle ACC@0.5："
                f"{g['top5_oracle_acc_at_05']:.4f}",
                f"- top-10 oracle ACC@0.5："
                f"{g['top10_oracle_acc_at_05']:.4f}",
                f"- 无候选率：{g['no_candidate_rate']:.4f}",
                f"- 平均候选数：{g['candidate_count_mean']:.3f}",
                f"- 平均延迟：{g['latency_ms_mean']:.1f} ms",
            ]
        )
    else:
        lines.append("- 未完成；详见运行异常记录。")
    if aic_smoke:
        lines.extend(
            [
                "",
                "AIC 100 条无标签 smoke：",
                "",
                f"- 合法框率：{aic_smoke['valid_bbox_rate']:.4f}",
                f"- 无候选率：{aic_smoke['no_candidate_rate']:.4f}",
                f"- PNG/JPG 数量：{aic_smoke['domain_counts']}",
                f"- 平均延迟：{aic_smoke['latency_ms_mean']:.1f} ms",
            ]
        )
    lines.extend(
        [
            "",
            "## 7. 固定阈值决策",
            "",
            f"- Florence：`{decisions['florence']}`；oracle 差值 "
            f"{decisions['florence_oracle_gap']:+.4f}。",
            f"- Tile：`{decisions['tile']}`；小目标增益 "
            f"{decisions['tile_small_oracle_gain']:+.4f}，对照组下降 "
            f"{decisions['tile_control_oracle_drop']:+.4f}。",
            f"- GroundingDINO top-1："
            f"`{decisions['grounding_dino_top1']}`。",
            f"- GroundingDINO 候选："
            f"`{decisions['grounding_dino_candidates']}`。",
            "",
            "## 8. 下一步训练与提交建议",
            "",
            "下一回合应严格按上面的固定阈值选择单变量实验。本地外部数据可能"
            "与模型预训练集重叠，因此绝对分数只用于回归；AIC 平台分数仍是"
            "目标域证据。建议一次只提交一种变化，保留 Florence RGB-only "
            "0.4980 作为平台基线。",
            "",
            "## 9. 可复现性与产物",
            "",
            "- 机器可读汇总：`outputs/diagnostics/round_summary.json`",
            "- 完整预测：本地 `outputs/diagnostics/`（已被 Git 忽略）",
            "- 候选分析：`reports/florence_candidate_analysis.md`",
            "- 模型 revision 与哈希：见机器可读汇总中的 `models`。",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decisions, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
