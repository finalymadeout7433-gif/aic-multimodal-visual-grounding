from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_rgbtir.data import read_jsonl_records  # noqa: E402
from aic_rgbtir.phase17a_plus import (  # noqa: E402
    Phase17APlusAuditor,
    Phase17DecisionPolicy,
)


def _resolve(raw: str, base: Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 1.7A+ config must be a YAML mapping")
    return payload


def _report(
    path: Path,
    *,
    result: Any,
    output_root: Path,
    negative: Mapping[str, Any],
    asset_preflight: Mapping[str, Any],
) -> None:
    missing = asset_preflight.get("missing", [])
    decision_path = output_root / "decision.json"
    decision_payload = (
        json.loads(decision_path.read_text(encoding="utf-8"))
        if decision_path.is_file()
        else {}
    )
    evidence = decision_payload.get("evidence", {})
    complete = result.status == "PHASE_17A_PLUS_COMPLETE" and not missing and bool(evidence)
    lines = [
        "# AIC RGB–TIR Phase 1.7A+ 无训练证据审计",
        "",
        "> AS_OF: 2026-08-13  ",
        "> 本轮不训练、不使用 AIC 测试集、不生成提交。",
        "",
        "## 1. 执行结论",
        "",
        f"- 状态：`{result.status}`",
        f"- 唯一后续分支：`{result.decision.branch}`",
        f"- 原因：{result.decision.reason}",
        f"- 已画像 repair-dev：`{result.completed_records}` 条。",
        "",
        "## 2. 资产门禁",
        "",
        f"- 资产状态：`{asset_preflight.get('status')}`",
        f"- 缺失资产：`{', '.join(missing) if missing else '无'}`",
        "",
        "缺失 C1/C2 与 Teacher Bank 时，不能计算逐层绝对对齐、Teacher margin 或模型空间 top-20 假负例；输入画像不会被冒充为模型诊断。",
        "",
        "## 3. 已完成的输入事实与代理",
        "",
        "- repair-dev RGB 全图与 ROI 亮度、对比度、模糊度、熵；",
        "- TIR 边界连通黑边有效视场比例；",
        "- source、illumination、weather、size、occlusion 分组；",
        "- 固定 256 跨图负例的 target-head、superclass 与 Query 词集相似代理。",
        "",
        f"潜在假负例代理率：`{float(negative.get('potential_false_negative_rate', 0.0)):.4f}`。它是词法启发式代理，不是模型空间已确认假负例率。",
        "",
    ]
    if complete:
        layer_deltas = evidence.get("layer_absolute_alignment_delta_mean", {})
        lines.extend([
            "## 4. 已完成的模型证据",
            "",
            f"- Layer 8 绝对 alignment delta：`{float(layer_deltas.get('8', 0.0)):.4f}`；",
            f"- Layer 16 绝对 alignment delta：`{float(layer_deltas.get('16', 0.0)):.4f}`；",
            f"- Layer 24 绝对 alignment delta：`{float(layer_deltas.get('24', 0.0)):.4f}`；",
            f"- RGB 质量风险与漂移 Spearman 关联：`{float(evidence.get('rgb_quality_association', 0.0)):.4f}`；",
            f"- 模型 top-neighbor 潜在假负例代理率：`{float(evidence.get('model_top_neighbor_potential_false_negative_rate', 0.0)):.4f}`。",
            "",
            "绝对逐层指标确认 C2 存在真实漂移；质量关联和假负例指标属于诊断代理，不是因果结论或人工真值。",
            "",
            "## 5. 下一操作",
            "",
            f"只进入 `{result.decision.branch}`。保持 C2 训练口径，仅新增预注册的 per-layer Base-TIR retention；先做 repair probe，通过门禁后才允许 full train。",
            "",
            "当前仍不能确认 Query grounding、bbox ACC 或 AIC 平台收益；本轮没有训练、没有使用 AIC 测试数据。",
            "",
            "## 6. 产物",
            "",
        ])
    else:
        lines.extend([
            "## 4. 尚未完成的核心问题",
            "",
            "1. C2 的最差层负百分比是否由小 Base loss 分母放大；",
            "2. 真实漂移是否与低质量 RGB 显著相关；",
            "3. Teacher 相似度 top-20 中是否存在模型空间假负例；",
            "4. 哪个风险簇造成 C2 的实际层级漂移。",
            "",
            "这些问题必须恢复原始 C1/C2 Adapter 与共同 Teacher Bank 后复算，不能从聚合报告反推。",
            "",
            "## 5. 下一操作",
            "",
            "从 Phase 1.6 云端持久盘或归档回收 `probe_candidates/C1`、`probe_candidates/C2` 和 `rgb_teacher_bank/teacher_bank.pt`，保持原 fingerprint；不得重训或改变 repair-dev。",
            "",
            "预期云端根目录：`/home/featurize/aic_cloud/outputs/aic_rgbtir_phase16_v1/`。建议只打包上述目录以及 `run_summary.json`，回传到本地现有 Phase 1.6 输出根目录。资产到齐后进入 Phase 1.7A+ 的模型特征复算切片。",
            "",
            "## 6. 产物",
            "",
        ])
    for artifact in sorted(output_root.glob("*")):
        if artifact.is_file():
            lines.append(f"- `{artifact.name}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> int:
    config = _load_config(config_path)
    base = config_path.parent
    paths = {name: _resolve(str(value), base) for name, value in config["paths"].items()}
    records = read_jsonl_records(paths["repair_dev_manifest"])
    runtime = config["runtime"]
    actual_pairs = len({record.image_pair_key for record in records})
    expected = (int(runtime["expected_records"]), int(runtime["expected_pairs"]))
    actual = (len(records), actual_pairs)
    if actual != expected:
        raise RuntimeError(f"repair-dev drift: expected={expected}, actual={actual}")
    phase16_root = paths["phase16_output_root"]
    required_assets = {
        name: phase16_root / str(relative)
        for name, relative in config["assets"].items()
    }
    policy = Phase17DecisionPolicy(
        quality_association_threshold=float(config["decision"]["quality_association_threshold"]),
        false_negative_rate_threshold=float(config["decision"]["false_negative_rate_threshold"]),
    )
    auditor = Phase17APlusAuditor(
        output_root=paths["output_root"],
        seed=int(runtime["seed"]),
        negative_count=int(runtime["negative_count"]),
        policy=policy,
    )
    result = auditor.run(
        records=records,
        rgbt_root=paths["rgbt_root"],
        required_assets=required_assets,
        model_diagnostics_path=paths.get("model_diagnostics"),
    )
    negative = json.loads((paths["output_root"] / "negative_sampling_audit.json").read_text(encoding="utf-8"))
    preflight = json.loads((paths["output_root"] / "asset_preflight.json").read_text(encoding="utf-8"))
    _report(paths["report"], result=result, output_root=paths["output_root"], negative=negative, asset_preflight=preflight)
    print(json.dumps({"status": result.status, "decision": result.decision.branch, "report": str(paths["report"])}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AIC RGB-TIR Phase 1.7A+ no-training evidence audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Accepted for a stable rerun interface; outputs are deterministic.")
    args = parser.parse_args()
    return run(args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
