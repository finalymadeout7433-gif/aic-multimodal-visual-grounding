from __future__ import annotations

import argparse
import ctypes
import gc
import json
import time
from pathlib import Path

import cv2
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate


def trim_cpu_memory() -> None:
    """Return large, already-freed temporary model buffers to Linux promptly."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


class AuditedDetectionCheckpointer(DetectionCheckpointer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_incompatible = None

    def _load_model(self, checkpoint):
        incompatible = super()._load_model(checkpoint)
        self.last_incompatible = incompatible
        return incompatible


def skip_unused_eva_visual_bootstrap() -> None:
    """Avoid constructing the EVA-CLIP image tower that APE deletes immediately."""
    from ape.modeling.text.eva02_clip import model as eva02_clip_model

    class DisposableVisionTower(torch.nn.Identity):
        image_size = 224

    eva02_clip_model._build_vision_tower = (
        lambda *args, **kwargs: DisposableVisionTower()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build APE-Ti, inspect checkpoint compatibility, and run one inference."
    )
    parser.add_argument("--ape-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.ape_root, args.config, args.checkpoint, args.image):
        if not path.exists():
            raise FileNotFoundError(path)

    skip_unused_eva_visual_bootstrap()
    cfg = LazyConfig.load(str(args.config))
    # The released final checkpoint already contains the complete EVA02 text branch.
    # Avoid downloading a separate bootstrap checkpoint before compatibility is known.
    cfg.model.model_language.cache_dir = None
    cfg.train.device = "cuda"
    cfg.model.model_vision.test_score_thresh = args.score_threshold
    # Official APE supports a pure-PyTorch fallback when xFormers is unavailable.
    cfg.model.model_vision.backbone.net.xattn = False

    started = time.perf_counter()
    model = instantiate(cfg.model)
    build_seconds = time.perf_counter() - started
    trim_cpu_memory()
    model.to(cfg.train.device)
    trim_cpu_memory()
    model.eval()

    checkpointer = AuditedDetectionCheckpointer(model)
    checkpointer.load(str(args.checkpoint), checkpointables=[])
    incompatible = checkpointer.last_incompatible
    if incompatible is None:
        raise RuntimeError("Checkpoint compatibility information was not captured")
    trim_cpu_memory()
    missing_keys = sorted(incompatible.missing_keys)
    unexpected_keys = sorted(incompatible.unexpected_keys)
    incorrect_shapes = [
        {
            "key": key,
            "checkpoint_shape": list(checkpoint_shape),
            "model_shape": list(model_shape),
        }
        for key, checkpoint_shape, model_shape in incompatible.incorrect_shapes
    ]
    load_seconds = time.perf_counter() - started - build_seconds

    original_image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if original_image is None:
        raise RuntimeError(f"Unable to decode image: {args.image}")
    height, width = original_image.shape[:2]
    input_format = cfg.model.model_vision.input_format
    model_image = original_image[:, :, ::-1] if input_format == "RGB" else original_image
    augmentation = instantiate(cfg.dataloader.test.mapper.augmentations[0])
    image = augmentation.get_transform(model_image).apply_image(model_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = {
        "image": image,
        "height": height,
        "width": width,
        "prompt": "text",
        "text_prompt": args.query,
    }

    torch.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    with torch.inference_mode(), torch.cuda.amp.autocast(
        enabled=True, dtype=torch.float16
    ):
        prediction = model([inputs])[0]
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    instances = prediction.get("instances")
    boxes: list[list[float]] = []
    scores: list[float] = []
    if instances is not None:
        instances = instances.to("cpu")
        if instances.has("pred_boxes"):
            boxes = instances.pred_boxes.tensor.tolist()
        if instances.has("scores"):
            scores = instances.scores.tolist()

    result = {
        "status": "ok",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "image": str(args.image),
        "query": args.query,
        "input_size": [width, height],
        "checkpoint_compatibility": {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "incorrect_shapes": incorrect_shapes,
            "strict_equivalent": not missing_keys
            and not unexpected_keys
            and not incorrect_shapes,
        },
        "prediction": {
            "instance_count": len(boxes),
            "boxes_xyxy": boxes,
            "scores": scores,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "build_seconds": build_seconds,
            "checkpoint_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "xformers_disabled": True,
            "unused_eva_visual_bootstrap_skipped": True,
            "autocast_dtype": "float16",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
