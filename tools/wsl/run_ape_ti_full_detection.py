from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run APE-Ti native Top-1 over the full unlabeled AIC test set."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--ape-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-sha256",
        help=(
            "Expected SHA-256 assertion. The checkpoint is always re-hashed "
            "and the run stops if the assertion is stale."
        ),
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def trim_cpu_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


class AuditedDetectionCheckpointer(DetectionCheckpointer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_incompatible = None

    def _load_model(self, checkpoint: dict[str, Any]) -> Any:
        incompatible = super()._load_model(checkpoint)
        self.last_incompatible = incompatible
        return incompatible


def skip_unused_eva_visual_bootstrap() -> None:
    from ape.modeling.text.eva02_clip import model as eva02_clip_model

    class DisposableVisionTower(torch.nn.Identity):
        image_size = 224

    eva02_clip_model._build_vision_tower = (
        lambda *args, **kwargs: DisposableVisionTower()
    )


@dataclass(frozen=True)
class FastAICRecord:
    query_id: str
    query: str
    visible_path: Path
    infrared_path: Path
    depth_path: Path
    bbox: list[float] | None
    source: dict[str, Any]


class FastAICDataset:
    """Read the pre-audited AIC manifest without 28k slow WSL stat calls."""

    def __init__(self, *, dataset_root: Path, queries_path: Path) -> None:
        self.dataset_root = dataset_root
        self.queries_path = queries_path
        self.raw_records = json.loads(queries_path.read_text(encoding="utf-8-sig"))
        if not isinstance(self.raw_records, dict):
            raise ValueError("queries JSON must be an object")
        self._records: list[FastAICRecord] = []
        for query_id, source in self.raw_records.items():
            if not isinstance(source, dict):
                raise ValueError(f"invalid record: {query_id}")
            for field in ("visible", "infrared", "depth", "query"):
                if field not in source:
                    raise ValueError(f"{query_id} missing field: {field}")
            paths: dict[str, Path] = {}
            for field in ("visible", "infrared", "depth"):
                relative = Path(str(source[field]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe relative path for {query_id}: {field}")
                paths[field] = dataset_root / relative
            query = str(source["query"])
            if not query.strip():
                raise ValueError(f"empty query: {query_id}")
            self._records.append(
                FastAICRecord(
                    query_id=str(query_id),
                    query=query,
                    visible_path=paths["visible"],
                    infrared_path=paths["infrared"],
                    depth_path=paths["depth"],
                    bbox=None,
                    source=source,
                )
            )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> FastAICRecord:
        return self._records[index]

    def load_visible(self, record: FastAICRecord) -> Image.Image:
        with Image.open(record.visible_path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")


class ApeTiPredictor:
    model_name = "shenyunhang/APE-Ti"

    def __init__(
        self,
        *,
        ape_root: Path,
        config: Path,
        checkpoint: Path,
        score_threshold: float,
        max_candidates: int,
    ) -> None:
        from aic_baseline.external_eval import ExternalPrediction, ModelCandidate

        self._prediction_class = ExternalPrediction
        self._candidate_class = ModelCandidate
        self.max_candidates = max_candidates
        self._cached_image: Image.Image | None = None
        self._cached_prepared_image: tuple[torch.Tensor, int, int] | None = None
        skip_unused_eva_visual_bootstrap()
        cfg = LazyConfig.load(str(config))
        cfg.model.model_language.cache_dir = None
        cfg.train.device = "cuda"
        cfg.model.model_vision.test_score_thresh = score_threshold
        cfg.model.model_vision.backbone.net.xattn = False
        # This experiment evaluates native bbox Top-1 only. Disabling unused
        # semantic/panoptic/mask outputs leaves box logits unchanged and avoids
        # full-resolution mask allocation on an 8 GiB GPU.
        cfg.model.model_vision.semantic_on = False
        cfg.model.model_vision.panoptic_on = False
        cfg.model.model_vision.test_mask_on = False
        self.cfg = cfg
        self.augmentation = instantiate(cfg.dataloader.test.mapper.augmentations[0])

        build_started = time.perf_counter()
        self.model = instantiate(cfg.model)
        self.build_seconds = time.perf_counter() - build_started
        trim_cpu_memory()
        self.model.to("cuda")
        self.model.eval()
        trim_cpu_memory()

        checkpointer = AuditedDetectionCheckpointer(self.model)
        load_started = time.perf_counter()
        checkpointer.load(str(checkpoint), checkpointables=[])
        self.checkpoint_load_seconds = time.perf_counter() - load_started
        incompatible = checkpointer.last_incompatible
        if incompatible is None:
            raise RuntimeError("checkpoint compatibility information was not captured")
        self.checkpoint_compatibility = {
            "missing_keys": sorted(incompatible.missing_keys),
            "unexpected_keys": sorted(incompatible.unexpected_keys),
            "incorrect_shapes": [
                {
                    "key": key,
                    "checkpoint_shape": list(checkpoint_shape),
                    "model_shape": list(model_shape),
                }
                for key, checkpoint_shape, model_shape in incompatible.incorrect_shapes
            ],
        }
        self.checkpoint_compatibility["strict_equivalent"] = not any(
            self.checkpoint_compatibility[key]
            for key in ("missing_keys", "unexpected_keys", "incorrect_shapes")
        )
        if not self.checkpoint_compatibility["strict_equivalent"]:
            raise RuntimeError(
                f"APE checkpoint is not strict-equivalent: {self.checkpoint_compatibility}"
            )
        trim_cpu_memory()

    @torch.inference_mode()
    def predict(self, *, image: Image.Image, query: str) -> Any:
        return self.predict_batch(images=[image], queries=[query])[0]

    @torch.inference_mode()
    def predict_batch(
        self, *, images: list[Image.Image], queries: list[str]
    ) -> list[Any]:
        if len(images) != len(queries):
            raise ValueError("images and queries must have the same length")
        if not images:
            return []
        prepared: dict[int, tuple[torch.Tensor, int, int]] = {}
        inputs: list[dict[str, Any]] = []
        for image, query in zip(images, queries):
            image_key = id(image)
            if (
                image is self._cached_image
                and self._cached_prepared_image is not None
            ):
                prepared[image_key] = self._cached_prepared_image
            elif image_key not in prepared:
                rgb = np.asarray(image.convert("RGB"))
                original = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                height, width = original.shape[:2]
                input_format = self.cfg.model.model_vision.input_format
                model_image = (
                    original[:, :, ::-1] if input_format == "RGB" else original
                )
                transformed = self.augmentation.get_transform(
                    model_image
                ).apply_image(model_image)
                tensor = torch.as_tensor(
                    transformed.astype("float32").transpose(2, 0, 1)
                )
                prepared[image_key] = (tensor, height, width)
                self._cached_image = image
                self._cached_prepared_image = prepared[image_key]
            tensor, height, width = prepared[image_key]
            inputs.append(
                {
                    "image": tensor,
                    "height": height,
                    "width": width,
                    "prompt": "text",
                    "text_prompt": query,
                }
            )
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            model_predictions = self.model(inputs)
        predictions: list[Any] = []
        for prediction in model_predictions:
            instances = prediction.get("instances")
            if instances is None:
                predictions.append(
                    self._prediction_class(raw_output={}, candidates=[])
                )
                continue
            instances = instances.to("cpu")
            boxes = (
                instances.pred_boxes.tensor.tolist()
                if instances.has("pred_boxes")
                else []
            )
            scores = instances.scores.tolist() if instances.has("scores") else []
            classes = (
                instances.pred_classes.tolist()
                if instances.has("pred_classes")
                else []
            )
            order = sorted(
                range(len(boxes)), key=lambda index: scores[index], reverse=True
            )
            candidates = [
                self._candidate_class(
                    pixel_bbox=[float(value) for value in boxes[index]],
                    label=(str(classes[index]) if index < len(classes) else ""),
                    score=float(scores[index]),
                    source="ape_ti_native",
                )
                for index in order[: self.max_candidates]
            ]
            predictions.append(
                self._prediction_class(
                    raw_output={"instance_count": len(boxes)},
                    candidates=candidates,
                )
            )
        return predictions


def main() -> int:
    args = parse_args()
    for path in (
        args.project_root,
        args.ape_root,
        args.config,
        args.checkpoint,
        args.dataset_root,
        args.queries,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in WSL2")
    sys.path.insert(0, str((args.project_root / "src").resolve()))
    from aic_baseline.aic_full_detection import (
        finalize_aic_full_detection,
        run_aic_full_detection,
    )
    from aic_baseline.provenance import (
        git_repository_state,
        hash_named_files,
        sha256_file,
        verify_file_sha256,
    )
    dataset = FastAICDataset(
        dataset_root=args.dataset_root,
        queries_path=args.queries,
    )
    checkpoint_sha256 = verify_file_sha256(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
    )
    ape_repository_state = git_repository_state(args.ape_root)
    ape_runtime_files_sha256 = hash_named_files(
        args.ape_root,
        [
            "ape/modeling/ape_deta/deformable_transformer_vl.py",
            "ape/modeling/ape_deta/deformable_detr_segm_vl.py",
        ],
    )
    fingerprint = {
        "schema": "aic-zero-shot-full-v1",
        "model_name": "shenyunhang/APE-Ti",
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": sha256_file(args.config),
        "queries_sha256": sha256_file(args.queries),
        "query_count": len(dataset),
        "ape_commit": ape_repository_state["head"],
        "ape_repository_state": ape_repository_state,
        "ape_runtime_files_sha256": ape_runtime_files_sha256,
        "ape_runtime_patch_sha256": sha256_file(
            args.project_root / "tools" / "wsl" / "ape_ti_rtx4060_cuda116.patch"
        ),
        "inference_code_sha256": {
            "runner": sha256_file(Path(__file__)),
            "core": sha256_file(
                args.project_root
                / "src"
                / "aic_baseline"
                / "aic_full_detection.py"
            ),
        },
        "score_threshold": args.score_threshold,
        "max_candidates": args.max_candidates,
        "batch_size": args.batch_size,
        "selection_policy": "native_highest_score_top1",
        "record_limit": args.limit,
        "training_on_aic": False,
        "modalities": ["visible_rgb", "query_original"],
        "xformers_disabled": True,
        "bbox_only_inference": True,
        "autocast_dtype": "float16",
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    predictor = ApeTiPredictor(
        ape_root=args.ape_root,
        config=args.config,
        checkpoint=args.checkpoint,
        score_threshold=args.score_threshold,
        max_candidates=args.max_candidates,
    )
    torch.cuda.reset_peak_memory_stats()
    progress_started = time.perf_counter()

    def progress(completed: int, total: int, query_id: str) -> None:
        if completed == total or completed % 10 == 0:
            elapsed = max(time.perf_counter() - progress_started, 1e-9)
            rate = completed / elapsed
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": total,
                        "query_id": query_id,
                        "queries_per_second": round(rate, 4),
                        "eta_seconds": round((total - completed) / rate, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    result = run_aic_full_detection(
        dataset=dataset,
        predictor=predictor,
        output_dir=args.output_dir,
        run_fingerprint=fingerprint,
        resume=args.resume,
        limit=args.limit,
        batch_size=args.batch_size,
        progress_callback=progress,
    )
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "build_seconds": predictor.build_seconds,
        "checkpoint_load_seconds": predictor.checkpoint_load_seconds,
        "checkpoint_compatibility": predictor.checkpoint_compatibility,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    finalized = None
    if args.finalize:
        if args.limit is not None:
            raise ValueError("cannot finalize a limited probe run")
        finalized = finalize_aic_full_detection(
            dataset=dataset,
            detection_dir=args.output_dir,
            submission_dir=args.output_dir / "submission",
        )
    print(
        json.dumps(
            {
                "summary": result.summary,
                "submission_zip": (
                    None if finalized is None else str(finalized.zip_path)
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
