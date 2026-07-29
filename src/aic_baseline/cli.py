from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from .bbox import acc_at_05, intersection_over_union
from .data import AICDataset
from .florence import TASK_PROMPT, FlorenceGrounder
from .inference import run_inference
from .submission import build_submission
from .visualization import draw_comparison


def load_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是对象")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _directory_manifest_sha256(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"模型目录中没有文件: {root}")
    for path in files:
        relative_path = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def _build_run_fingerprint(
    config: dict[str, Any], *, fallback_mode: str
) -> dict[str, Any]:
    model_path = Path(config["model_path"])
    weights = model_path / "model.safetensors"
    queries_path = Path(config["queries_path"])
    return {
        "schema_version": 2,
        "task_prompt": TASK_PROMPT,
        "queries_path": str(queries_path.resolve()),
        "queries_sha256": _sha256(queries_path),
        "model_path": str(model_path.resolve()),
        "model_weights_sha256": _sha256(weights),
        "model_artifacts_sha256": _directory_manifest_sha256(model_path),
        "device": config.get("device", "cuda"),
        "dtype": "float16",
        "seed": int(config.get("seed", 20260729)),
        "max_new_tokens": int(config.get("max_new_tokens", 256)),
        "num_beams": int(config.get("num_beams", 1)),
        "selection_strategy": config.get("selection_strategy", "first"),
        "fallback_mode": fallback_mode,
        "attention_implementation": "eager",
        "generation_use_cache": False,
    }


def _record_environment(
    config: dict[str, Any],
    output_dir: Path,
    *,
    run_fingerprint: dict[str, Any],
) -> None:
    model_path = Path(config["model_path"])
    weights = model_path / "model.safetensors"
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "model_path": str(model_path.resolve()),
        "model_weights_bytes": weights.stat().st_size if weights.exists() else None,
        "model_weights_sha256": run_fingerprint["model_weights_sha256"],
        "model_artifacts_sha256": run_fingerprint["model_artifacts_sha256"],
        "queries_path": run_fingerprint["queries_path"],
        "queries_sha256": run_fingerprint["queries_sha256"],
        "attention_implementation": "eager",
        "generation_use_cache": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _make_dataset(config: dict[str, Any]) -> AICDataset:
    return AICDataset(
        dataset_root=config["dataset_root"],
        queries_path=config["queries_path"],
    )


def _make_grounder(config: dict[str, Any]) -> FlorenceGrounder:
    return FlorenceGrounder(
        model_path=config["model_path"],
        device=config.get("device", "cuda"),
        max_new_tokens=int(config.get("max_new_tokens", 256)),
        num_beams=int(config.get("num_beams", 1)),
        selection_strategy=config.get("selection_strategy", "first"),
    )


def command_validate(config: dict[str, Any]) -> None:
    dataset = _make_dataset(config)
    missing: list[str] = []
    suffixes: Counter[str] = Counter()
    unique_visible: set[Path] = set()
    for record in dataset:
        unique_visible.add(record.visible_path)
        suffixes[record.visible_path.suffix.lower()] += 1
        for path in (
            record.visible_path,
            record.infrared_path,
            record.depth_path,
        ):
            if not path.is_file():
                missing.append(str(path))
    result = {
        "query_count": len(dataset),
        "unique_visible_count": len(unique_visible),
        "visible_query_suffix_counts": dict(suffixes),
        "missing_file_count": len(missing),
        "missing_files_preview": missing[:20],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(2)


def command_infer(
    config: dict[str, Any],
    *,
    limit: int | None,
    output_override: str | None,
    build_zip: bool,
    resume: bool,
) -> None:
    random.seed(int(config.get("seed", 20260729)))
    torch.manual_seed(int(config.get("seed", 20260729)))
    dataset = _make_dataset(config)
    output_dir = Path(output_override or config["output_dir"])
    fallback_mode = config.get("fallback_mode", "error")
    run_fingerprint = _build_run_fingerprint(
        config,
        fallback_mode=fallback_mode,
    )
    _record_environment(
        config,
        output_dir,
        run_fingerprint=run_fingerprint,
    )
    grounder = _make_grounder(config)
    result = run_inference(
        dataset=dataset,
        grounder=grounder,
        output_dir=output_dir,
        limit=limit,
        fallback_mode=fallback_mode,
        resume=resume,
        run_fingerprint=run_fingerprint,
    )
    if build_zip:
        if not result.summary["is_full_dataset_run"]:
            raise ValueError("有限样本 smoke run 不能生成正式提交 ZIP")
        build_submission(
            original_records=dataset.raw_records,
            predictions=result.predictions,
            output_json=output_dir / "predictions_submission.json",
            output_zip=output_dir / "predictions_submission.zip",
        )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))


def command_sample(config: dict[str, Any], *, output_override: str | None) -> None:
    dataset = _make_dataset(config)
    if len(dataset) != 1 or dataset[0].bbox is None:
        raise ValueError("官方 sanity 配置必须指向唯一且带 bbox 的样例")
    output_dir = Path(output_override or config["output_dir"])
    run_fingerprint = _build_run_fingerprint(config, fallback_mode="error")
    _record_environment(
        config,
        output_dir,
        run_fingerprint=run_fingerprint,
    )
    grounder = _make_grounder(config)
    result = run_inference(
        dataset=dataset,
        grounder=grounder,
        output_dir=output_dir,
        fallback_mode="error",
        run_fingerprint=run_fingerprint,
    )
    record = dataset[0]
    prediction = result.predictions[record.query_id]
    assert record.bbox is not None
    evaluation = {
        "query_id": record.query_id,
        "query": record.query,
        "official_bbox": record.bbox,
        "prediction_bbox": prediction,
        "iou": intersection_over_union(prediction, record.bbox),
        "acc_at_05": acc_at_05(prediction, record.bbox),
        "interpretation": "单条官方样例仅用于 sanity check，不代表模型总体准确率。",
    }
    (output_dir / "sample_evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    draw_comparison(
        image=dataset.load_visible(record),
        ground_truth=record.bbox,
        prediction=prediction,
        output_path=output_dir / "gt_vs_prediction.png",
    )
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2026 AIC RGB-only baseline v0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="只读核验全部数据路径")
    validate.add_argument("--config", required=True)
    infer = subparsers.add_parser("infer", help="运行 Florence RGB-only 推理")
    infer.add_argument("--config", required=True)
    infer.add_argument("--limit", type=int)
    infer.add_argument("--output-dir")
    infer.add_argument("--build-submission", action="store_true")
    infer.add_argument("--resume", action="store_true")
    sample = subparsers.add_parser("sample", help="运行官方单样例 sanity test")
    sample.add_argument("--config", required=True)
    sample.add_argument("--output-dir")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "validate":
        command_validate(config)
    elif args.command == "infer":
        command_infer(
            config,
            limit=args.limit,
            output_override=args.output_dir,
            build_zip=args.build_submission,
            resume=args.resume,
        )
    elif args.command == "sample":
        command_sample(config, output_override=args.output_dir)


if __name__ == "__main__":
    main()
