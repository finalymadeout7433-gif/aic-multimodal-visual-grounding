from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_rgbtir.phase18r_audit import Phase18RAuditConfig, Phase18RAuditor  # noqa: E402


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _write_report(report: Path, output: Path) -> None:
    run = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    retrieval = json.loads((output / "retrieval_audit.json").read_text(encoding="utf-8"))
    subgroup = json.loads((output / "subgroup_excess_drift.json").read_text(encoding="utf-8"))
    safety = json.loads((output / "safety_contract_audit.json").read_text(encoding="utf-8"))
    lines = [
        "# AIC RGB–TIR Phase 1.8R-Audit 本地复核报告",
        "",
        f"**最终决策：`{run['decision']}`**",
        "",
        "## 结论",
        "",
        *[f"- {finding}" for finding in run["findings"]],
        "",
        "## 完整性与安全契约",
        "",
        f"- 输入完整：`{run['complete']}`；输入复核前后保持不变：`{run['input_immutable']}`。",
        f"- 历史安全失败项：`{', '.join(safety['historical_false_checks']) or '无'}`。",
        f"- 是否属于可修复的推理冻结契约问题：`{safety['repairable_inference_freeze_issue']}`。",
        f"- Adapter：`{safety['adapter_tensor_count']}` 个张量，全部有限：`{safety['adapter_all_finite']}`。",
        "",
        "## 三套检索口径",
        "",
        "| 层 | 严格记录 R@5 | 同图像对多正例 R@5 | 图像对聚合 R@5 |",
        "|---|---:|---:|---:|",
    ]
    for layer in ("8", "16", "24"):
        adapted = retrieval[layer]["adapted"]
        lines.append(
            f"| {layer} | {adapted['record_exact']['r_at_5']:.4f} | "
            f"{adapted['pair_multi_positive']['r_at_5']:.4f} | "
            f"{adapted['pair_mean']['r_at_5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 有效秩复核",
            "",
            "| 层 | 记录级 Adapted/Base | 图像对聚合 Adapted/Base |",
            "|---|---:|---:|",
        ]
    )
    for layer in ("8", "16", "24"):
        record_ratio = decision["rank_ratios_vs_base"][layer]["record_level"]
        pair_ratio = decision["rank_ratios_vs_base"][layer]["pair_mean_level"]
        lines.append(f"| {layer} | {record_ratio:.4f} | {pair_ratio:.4f} |")
    worst_groups = sorted(
        subgroup["groups"], key=lambda row: float(row["excess_drift"]), reverse=True
    )[:8]
    lines.extend(
        [
            "",
            "## ExcessDrift 最大的分组",
            "",
            f"全局绝对漂移：`{subgroup['global_absolute_drift']:.6f}`。正值表示 Adapted 比 Base 更偏离 RGB Teacher。",
            "",
            "| 分组 | 记录/图像对 | ExcessDrift | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in worst_groups:
        low, high = row["excess_drift_ci95"]
        lines.append(
            f"| {row['group_field']}={row['group_value']} | "
            f"{row['record_count']}/{row['unique_pair_count']} | "
            f"{row['excess_drift']:.6f} | [{low:.6f}, {high:.6f}] |"
        )
    next_step = {
        "GATE_FIX": "只修验证器冻结与统计口径，然后运行冻结 Query–TIR probe。",
        "MODEL_FIX": "保持修正后的门禁，先运行单变量 D2 谱保持修复。",
        "BOTH_REQUIRED": "先固定验证器，再运行单变量 D2 谱保持修复；通过后才进入冻结 Query–TIR probe。",
        "AUDIT_CLEAR": "直接进入冻结 Query–TIR probe。",
        "ASSET_BLOCKED": "补齐或修复审计输入，禁止继续训练。",
    }[run["decision"]]
    lines.extend(
        [
            "",
            "## 下一轮唯一主分支",
            "",
            next_step,
            "",
            "## 结论边界",
            "",
            "本报告只复核 RGBT-GroundBench official-val 表征和历史安全门禁；不证明 Query grounding、bbox ACC、AIC 平台分数或 UniRGB-IR 融合收益。",
            "",
            f"输入指纹：`{run['input_fingerprint']}`。",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit returned Phase 1.8R assets locally")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = raw["paths"]
    expected = raw["expected"]
    runtime = raw["runtime"]
    gates = raw["gates"]
    auditor = Phase18RAuditor(
        input_root=_resolve(config_path, paths["input_root"]),
        output_root=_resolve(config_path, paths["output_root"]),
        config=Phase18RAuditConfig(
            expected_records=int(expected["records"]),
            expected_pairs=int(expected["pairs"]),
            expected_adapter_tensors=int(expected["adapter_tensors"]),
            seed=int(runtime["seed"]),
            bootstrap_iterations=int(runtime["bootstrap_iterations"]),
            minimum_effective_rank_vs_base=float(gates["minimum_effective_rank_vs_base"]),
            maximum_global_absolute_drift=float(gates["maximum_global_absolute_drift"]),
        ),
    )
    result = auditor.run()
    _write_report(
        _resolve(config_path, paths["report"]),
        _resolve(config_path, paths["output_root"]),
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
