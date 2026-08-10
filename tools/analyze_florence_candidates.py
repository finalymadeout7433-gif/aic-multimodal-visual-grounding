from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from aic_baseline.diagnostics import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Florence first-vs-oracle candidate behavior."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _tokens(text: str) -> set[str]:
    ignored = {
        "a",
        "an",
        "the",
        "of",
        "on",
        "in",
        "at",
        "to",
        "with",
        "and",
        "by",
        "from",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in ignored
    }


def _label_role(label: str, query: str) -> str:
    normalized_label = " ".join(label.lower().split()).strip(" .,:;!?")
    normalized_query = " ".join(query.lower().split()).strip(" .,:;!?")
    if not normalized_label:
        return "empty"
    if normalized_label == normalized_query:
        return "whole_query"
    if normalized_label in normalized_query:
        return "query_phrase"
    if normalized_query in normalized_label:
        return "expanded_query"
    label_tokens = _tokens(normalized_label)
    query_tokens = _tokens(normalized_query)
    if not label_tokens:
        return "empty"
    overlap = len(label_tokens & query_tokens) / len(label_tokens)
    if overlap >= 0.5:
        return "partial_query_phrase"
    return "divergent_or_reference"


def _select_visual_cases(
    records: list[dict[str, Any]],
    *,
    minimum: int = 12,
) -> list[tuple[str, dict[str, Any]]]:
    selectors = (
        (
            "first_correct",
            lambda record: bool(record["selected_acc_at_05"]),
        ),
        (
            "oracle_rescue",
            lambda record: (
                not record["selected_acc_at_05"]
                and record["oracle_acc_at_05"]
            ),
        ),
        (
            "multi_distinct_label",
            lambda record: (
                record["candidate_semantic_type"] == "multi_distinct_label"
            ),
        ),
        (
            "multi_same_label",
            lambda record: (
                record["candidate_semantic_type"] == "multi_same_label"
            ),
        ),
        (
            "small_target",
            lambda record: record["area_bin"] == "small_lt_1pct",
        ),
        (
            "large_region",
            lambda record: record["area_bin"] == "large_ge_10pct",
        ),
        (
            "plural_group",
            lambda record: record["query_category"] == "plural_group",
        ),
    )
    selected: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()
    for bucket, predicate in selectors:
        matches = sorted(
            (record for record in records if predicate(record)),
            key=lambda record: str(record["query_id"]),
        )
        for record in matches[:2]:
            query_id = str(record["query_id"])
            if query_id not in used:
                selected.append((bucket, record))
                used.add(query_id)
    if len(selected) < minimum:
        for record in sorted(records, key=lambda item: str(item["query_id"])):
            query_id = str(record["query_id"])
            if query_id in used:
                continue
            selected.append(("additional", record))
            used.add(query_id)
            if len(selected) >= minimum:
                break
    return selected


def _draw_case(
    *,
    record: dict[str, Any],
    data_root: Path,
    output_path: Path,
) -> None:
    image_path = data_root / record["image_relpath"]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def rectangle(box: list[float], color: tuple[int, int, int], line: int) -> None:
        x1, y1, x2, y2 = box
        draw.rectangle(
            (x1 * width, y1 * height, x2 * width, y2 * height),
            outline=color,
            width=line,
        )

    for candidate in record["candidates"]:
        rectangle(candidate["bbox"], (255, 215, 0), 2)
    rectangle(record["gt_bbox"], (0, 255, 0), 4)
    if record["selected_bbox"] is not None:
        rectangle(record["selected_bbox"], (255, 0, 0), 4)
    best_index = record.get("best_candidate_index")
    if best_index is not None:
        rectangle(record["candidates"][best_index]["bbox"], (0, 128, 255), 3)

    caption = (
        f"{record['query_id']} | first IoU={record['selected_iou']:.3f} | "
        f"oracle IoU={record['best_candidate_iou']:.3f} | "
        f"{record['query']}"
    )
    caption = caption.encode("ascii", errors="replace").decode("ascii")[:180]
    draw.rectangle((0, 0, width, 28), fill=(0, 0, 0))
    draw.text((5, 6), caption, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.predictions)
    semantic_counts = Counter(
        str(record["candidate_semantic_type"]) for record in records
    )
    outcome_counts = Counter()
    role_counts = Counter()
    for record in records:
        selected_ok = bool(record["selected_acc_at_05"])
        oracle_ok = bool(record["oracle_acc_at_05"])
        if selected_ok:
            outcome_counts["first_correct"] += 1
        elif oracle_ok:
            outcome_counts["oracle_rescue"] += 1
        else:
            outcome_counts["all_candidates_fail"] += 1
        for candidate in record["candidates"]:
            role_counts[_label_role(candidate["label"], record["query"])] += 1

    selected_cases = _select_visual_cases(records)
    visuals: list[dict[str, str]] = []
    visuals_dir = args.output_dir / "visualizations"
    for index, (bucket, record) in enumerate(selected_cases, start=1):
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record["query_id"]))
        output_path = visuals_dir / f"{index:02d}_{bucket}_{safe_id}.jpg"
        _draw_case(
            record=record,
            data_root=args.data_root,
            output_path=output_path,
        )
        visuals.append(
            {
                "bucket": bucket,
                "query_id": str(record["query_id"]),
                "file": output_path.name,
            }
        )

    summary_path = args.predictions.parent / "summary.json"
    metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = {
        "record_count": len(records),
        "outcome_counts": dict(outcome_counts),
        "candidate_semantic_type_counts": dict(semantic_counts),
        "candidate_label_role_counts": dict(role_counts),
        "visualizations": visuals,
        "metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    overall = metrics["overall"]
    lines = [
        "# Florence 候选角色与 Oracle 分析",
        "",
        "本报告只使用外部 RefCOCO 系列 validation；未使用 AIC 正式标签或 holdout。",
        "",
        "## 核心结果",
        "",
        f"- 样本数：{len(records)}",
        f"- first ACC@0.5：{overall['selected_acc_at_05']:.4f}",
        f"- candidate oracle ACC@0.5：{overall['oracle_acc_at_05']:.4f}",
        f"- oracle 差值：{overall['oracle_gap']:.4f}",
        f"- first 正确：{outcome_counts['first_correct']}",
        f"- first 错但其他候选可救：{outcome_counts['oracle_rescue']}",
        f"- 所有候选均失败：{outcome_counts['all_candidates_fail']}",
        "",
        "## 候选语义类型",
        "",
    ]
    lines.extend(
        f"- `{key}`：{value}"
        for key, value in sorted(semantic_counts.items())
    )
    lines.extend(["", "## 候选 label 与 Query 的粗粒度角色", ""])
    lines.extend(
        f"- `{key}`：{value}" for key, value in sorted(role_counts.items())
    )
    lines.extend(
        [
            "",
            "## 可视化",
            "",
            "颜色：GT 绿色、first 红色、oracle 蓝色、其他候选黄色。",
            "",
        ]
    )
    lines.extend(
        f"- `{item['bucket']}` / `{item['query_id']}` / "
        f"`outputs/diagnostics/analysis/visualizations/{item['file']}`"
        for item in visuals
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
