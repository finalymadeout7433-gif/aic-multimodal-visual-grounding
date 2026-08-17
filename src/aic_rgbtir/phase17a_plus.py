from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import rankdata

from .data import RGBTRecord
from .phase16 import RGBTeacherBank


DECISION_BRANCHES = (
    "C2_FULL_TRAIN",
    "D1_RETENTION",
    "D1_QUALITY_WEIGHTED",
    "C2_FALSE_NEGATIVE_AWARE",
    "SHARED_COMPLEMENTARY_PROBE",
    "ASSET_BLOCKED",
)

_DISCRIMINATIVE_LAYERS = ("8", "16", "24")


@dataclass(frozen=True)
class Phase17DecisionInputs:
    assets_ready: bool
    real_alignment_drift: bool | None = None
    rgb_quality_association: float | None = None
    false_negative_rate: float | None = None
    pareto_conflict: bool | None = None


@dataclass(frozen=True)
class Phase17Decision:
    branch: str
    reason: str


@dataclass(frozen=True)
class Phase17APlusResult:
    status: str
    decision: Phase17Decision
    completed_records: int
    missing_assets: tuple[str, ...]
    claim_boundary: Mapping[str, bool]


class Phase17DecisionPolicy:
    """Convert pre-registered evidence into exactly one next-stage branch."""

    def __init__(
        self,
        *,
        quality_association_threshold: float = 0.30,
        false_negative_rate_threshold: float = 0.20,
    ) -> None:
        self.quality_association_threshold = float(quality_association_threshold)
        self.false_negative_rate_threshold = float(false_negative_rate_threshold)

    def decide(self, evidence: Phase17DecisionInputs) -> Phase17Decision:
        if not evidence.assets_ready:
            return Phase17Decision(
                "ASSET_BLOCKED",
                "C1/C2 adapters, summaries, or the shared Teacher Bank are unavailable; "
                "model-derived drift cannot be recomputed.",
            )
        if (
            evidence.false_negative_rate is not None
            and evidence.false_negative_rate >= self.false_negative_rate_threshold
        ):
            return Phase17Decision(
                "C2_FALSE_NEGATIVE_AWARE",
                "The registered semantic audit found a high potential false-negative rate.",
            )
        if evidence.pareto_conflict is True:
            return Phase17Decision(
                "SHARED_COMPLEMENTARY_PROBE",
                "Alignment, retrieval, and representation capacity remain in a structural conflict.",
            )
        if evidence.real_alignment_drift is False:
            return Phase17Decision(
                "C2_FULL_TRAIN",
                "Absolute layerwise losses do not confirm material drift; the relative gate was unstable.",
            )
        if evidence.real_alignment_drift is True:
            association = abs(float(evidence.rgb_quality_association or 0.0))
            if association >= self.quality_association_threshold:
                return Phase17Decision(
                    "D1_QUALITY_WEIGHTED",
                    "Material drift is concentrated in samples with unreliable RGB teacher evidence.",
                )
            return Phase17Decision(
                "D1_RETENTION",
                "Material drift is present but is not sufficiently associated with RGB quality.",
            )
        return Phase17Decision(
            "ASSET_BLOCKED",
            "Required model-derived evidence is incomplete, so no training branch is authorized.",
        )


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.IGNORECASE)
_RELATIONS = {
    "above", "below", "behind", "beside", "between", "inside", "near",
    "under", "with", "wearing", "holding", "carrying", "mounted", "next",
    "left", "right", "front", "over", "on", "at", "by", "from",
}
_IGNORED = {
    "a", "an", "the", "this", "that", "one", "two", "three", "four",
    "first", "second", "third", "leftmost", "rightmost", "nearest", "farthest",
    "small", "large", "big", "tiny", "red", "green", "blue", "white", "black",
    "yellow", "silver", "gray", "grey", "brown", "orange", "visible", "standing",
}
_SUPERCLASS: dict[str, str] = {
    "man": "person", "woman": "person", "person": "person", "pedestrian": "person",
    "boy": "person", "girl": "person", "worker": "person", "people": "person",
    "car": "vehicle", "vehicle": "vehicle", "truck": "vehicle", "bus": "vehicle",
    "motorcycle": "vehicle", "bike": "vehicle", "bicycle": "vehicle", "van": "vehicle",
    "dog": "animal", "cat": "animal", "bird": "animal", "animal": "animal",
    "light": "traffic_or_light", "lamp": "traffic_or_light", "signal": "traffic_or_light",
    "camera": "small_device", "phone": "small_device", "drone": "small_device",
    "sign": "sign_or_text", "logo": "sign_or_text", "billboard": "sign_or_text",
}


def _tokens(query: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_RE.findall(query))


def _target_head(query: str) -> tuple[str, str, float]:
    tokens = list(_tokens(query))
    if not tokens:
        return "unknown", "unknown", 0.0
    cut = next((index for index, token in enumerate(tokens) if token in _RELATIONS), len(tokens))
    phrase = [token for token in tokens[:cut] if token not in _IGNORED]
    if not phrase:
        phrase = [token for token in tokens if token not in _IGNORED and token not in _RELATIONS]
    head = phrase[-1] if phrase else "unknown"
    superclass = _SUPERCLASS.get(head, head if head != "unknown" else "unknown")
    confidence = 1.0 if head in _SUPERCLASS else (0.65 if head != "unknown" else 0.0)
    return head, superclass, confidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    probability = hist[hist > 0] / hist.sum()
    return float(-(probability * np.log2(probability)).sum())


def _image_quality(path: Path, bbox: tuple[float, float, float, float]) -> dict[str, float]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    x1, y1, x2, y2 = bbox
    left = max(0, min(width - 1, int(math.floor(x1 * width))))
    top = max(0, min(height - 1, int(math.floor(y1 * height))))
    right = max(left + 1, min(width, int(math.ceil(x2 * width))))
    bottom = max(top + 1, min(height, int(math.ceil(y2 * height))))
    roi = gray[top:bottom, left:right]
    return {
        "global_rgb_brightness": float(gray.mean() / 255.0),
        "global_rgb_contrast": float(gray.std() / 255.0),
        "global_rgb_blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "global_rgb_entropy": _entropy(gray),
        "roi_rgb_brightness": float(roi.mean() / 255.0),
        "roi_rgb_contrast": float(roi.std() / 255.0),
        "roi_rgb_entropy": _entropy(roi),
    }


def _border_valid_ratio(path: Path, threshold: int = 3) -> float:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    near_black = (gray <= threshold).astype(np.uint8)
    count, labels = cv2.connectedComponents(near_black, connectivity=8)
    border_labels = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    border_labels.discard(0)
    invalid = np.isin(labels, list(border_labels)) if count > 1 and border_labels else np.zeros_like(near_black, dtype=bool)
    return float(1.0 - invalid.mean())


class Phase17APlusAuditor:
    """Run the no-training Phase 1.7A+ evidence audit behind one interface."""

    def __init__(
        self,
        *,
        output_root: Path | str,
        seed: int = 20260812,
        negative_count: int = 256,
        policy: Phase17DecisionPolicy | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.seed = int(seed)
        self.negative_count = int(negative_count)
        self.policy = policy or Phase17DecisionPolicy()

    def audit_negatives(self, records: Sequence[RGBTRecord]) -> dict[str, Any]:
        representatives: dict[str, RGBTRecord] = {}
        for record in sorted(records, key=lambda value: value.record_id):
            representatives.setdefault(record.image_pair_key, record)
        pool = tuple(representatives.values())
        rows: list[dict[str, Any]] = []
        same_pair_violations = 0
        for anchor in records:
            eligible = [record for record in pool if record.image_pair_key != anchor.image_pair_key]
            rng = random.Random(int.from_bytes(hashlib.sha256(f"{self.seed}:{anchor.record_id}".encode()).digest()[:8], "big"))
            eligible.sort(key=lambda value: value.record_id)
            rng.shuffle(eligible)
            negatives = eligible[: min(self.negative_count, len(eligible))]
            head, superclass, confidence = _target_head(anchor.query_original)
            anchor_tokens = set(_tokens(anchor.query_original))
            same_head = 0
            same_superclass = 0
            high_query_similarity = 0
            potential = 0
            for negative in negatives:
                if negative.image_pair_key == anchor.image_pair_key:
                    same_pair_violations += 1
                other_head, other_superclass, _ = _target_head(negative.query_original)
                other_tokens = set(_tokens(negative.query_original))
                union = anchor_tokens | other_tokens
                jaccard = len(anchor_tokens & other_tokens) / len(union) if union else 0.0
                head_match = head != "unknown" and head == other_head
                superclass_match = superclass != "unknown" and superclass == other_superclass
                semantic_match = jaccard >= 0.50
                same_head += int(head_match)
                same_superclass += int(superclass_match)
                high_query_similarity += int(semantic_match)
                potential += int(head_match or superclass_match or semantic_match)
            denominator = max(1, len(negatives))
            rows.append({
                "record_id": anchor.record_id,
                "image_pair_key": anchor.image_pair_key,
                "target_head": head,
                "target_superclass": superclass,
                "parser_confidence": confidence,
                "sampled_negative_count": len(negatives),
                "same_target_head_rate": same_head / denominator,
                "same_superclass_rate": same_superclass / denominator,
                "high_query_similarity_rate": high_query_similarity / denominator,
                "potential_false_negative_rate": potential / denominator,
            })
        total_negatives = sum(row["sampled_negative_count"] for row in rows)
        def weighted(field: str) -> float:
            if total_negatives == 0:
                return 0.0
            return float(sum(row[field] * row["sampled_negative_count"] for row in rows) / total_negatives)
        return {
            "status": "INPUT_ONLY_PROXY_COMPLETE",
            "sampled_anchor_count": len(rows),
            "sampled_negative_count": total_negatives,
            "same_pair_violation_count": same_pair_violations,
            "same_target_head_rate": weighted("same_target_head_rate"),
            "same_superclass_rate": weighted("same_superclass_rate"),
            "high_query_similarity_rate": weighted("high_query_similarity_rate"),
            "potential_false_negative_rate": weighted("potential_false_negative_rate"),
            "claim_boundary": "Lexical heuristic proxy; not a confirmed model-space false-negative rate.",
            "per_record": rows,
        }

    @staticmethod
    def _asset_preflight(required_assets: Mapping[str, Path | str]) -> dict[str, Any]:
        assets: dict[str, Any] = {}
        for name, raw_path in sorted(required_assets.items()):
            path = Path(raw_path).resolve()
            present = path.is_file()
            assets[name] = {
                "path": str(path),
                "present": present,
                "bytes": path.stat().st_size if present else 0,
                "sha256": _sha256(path) if present else "",
            }
        missing = [name for name, item in assets.items() if not item["present"]]
        return {
            "status": "ASSETS_READY" if not missing else "ASSETS_MISSING",
            "assets": assets,
            "missing": missing,
        }

    def _quality_rows(self, records: Sequence[RGBTRecord], root: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record in records:
            quality = _image_quality(root / record.rgb_relpath, record.bbox_xyxy_normalized)
            head, superclass, confidence = _target_head(record.query_original)
            rows.append({
                "record_id": record.record_id,
                "image_pair_key": record.image_pair_key,
                "source_dataset": record.source_dataset,
                "illumination": record.illumination,
                "weather": record.weather,
                "object_size": record.object_size,
                "occlusion": record.occlusion,
                "target_head": head,
                "target_superclass": superclass,
                "parser_confidence": confidence,
                "ir_valid_fov_ratio": _border_valid_ratio(root / record.tir_relpath),
                **quality,
                "rgb_target_background_teacher_margin": None,
                "c2_alignment_delta": None,
                "c2_retrieval_delta": None,
            })
        return rows

    @staticmethod
    def _subgroups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for field in ("source_dataset", "illumination", "weather", "object_size", "occlusion"):
            groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in rows:
                groups[str(row[field])].append(row)
            for value, members in sorted(groups.items()):
                model_members = [row for row in members if row.get("c2_metrics_available")]
                output.append({
                    "group_field": field,
                    "group_value": value,
                    "record_count": len(members),
                    "rgb_brightness_mean": float(np.mean([row["global_rgb_brightness"] for row in members])),
                    "rgb_contrast_mean": float(np.mean([row["global_rgb_contrast"] for row in members])),
                    "ir_valid_fov_mean": float(np.mean([row["ir_valid_fov_ratio"] for row in members])),
                    "c2_metrics_available": bool(model_members),
                    "c2_alignment_delta_mean": (
                        float(np.mean([row["c2_alignment_delta"] for row in model_members]))
                        if model_members
                        else None
                    ),
                    "c2_alignment_delta_p95": (
                        float(np.quantile([row["c2_alignment_delta"] for row in model_members], 0.95))
                        if model_members
                        else None
                    ),
                    "c2_minus_base_paired_cosine_delta_mean": (
                        float(np.mean([
                            row["c2_minus_base_paired_cosine_delta"]
                            for row in model_members
                        ]))
                        if model_members
                        else None
                    ),
                    "rgb_target_background_teacher_margin_mean": (
                        float(np.mean([
                            row["rgb_target_background_teacher_margin"]
                            for row in model_members
                            if row.get("rgb_target_background_teacher_margin") is not None
                        ]))
                        if any(
                            row.get("rgb_target_background_teacher_margin") is not None
                            for row in model_members
                        )
                        else None
                    ),
                })
        return output

    @staticmethod
    def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right) or len(left) < 3:
            return 0.0
        x = rankdata(np.asarray(left, dtype=np.float64))
        y = rankdata(np.asarray(right, dtype=np.float64))
        if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
            return 0.0
        value = float(np.corrcoef(x, y)[0, 1])
        return value if math.isfinite(value) else 0.0

    def _bootstrap_spearman(
        self,
        left: Sequence[float],
        right: Sequence[float],
        *,
        iterations: int = 2000,
    ) -> tuple[float, float]:
        if len(left) != len(right) or len(left) < 3:
            return (0.0, 0.0)
        x = np.asarray(left, dtype=np.float64)
        y = np.asarray(right, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        values = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            sample = rng.integers(0, len(x), size=len(x))
            values[index] = self._spearman(x[sample], y[sample])
        return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))

    @staticmethod
    def _teacher_bank(
        required_assets: Mapping[str, Path | str],
    ) -> RGBTeacherBank | None:
        bank_path = required_assets.get("teacher_bank")
        summary_path = required_assets.get("c2_summary")
        if bank_path is None or summary_path is None:
            return None
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        fingerprint = str(summary.get("teacher_bank_fingerprint", ""))
        if not fingerprint:
            raise RuntimeError("C2 summary is missing teacher_bank_fingerprint")
        return RGBTeacherBank.load(bank_path, expected_fingerprint=fingerprint)

    @staticmethod
    def _enrich_model_rows(
        quality_rows: Sequence[Mapping[str, Any]],
        diagnostics: Mapping[str, Any],
        teacher_bank: RGBTeacherBank | None,
    ) -> list[dict[str, Any]]:
        diagnostics_by_id = {
            str(row["record_id"]): row for row in diagnostics["per_record"]
        }
        output: list[dict[str, Any]] = []
        for source in quality_rows:
            row = dict(source)
            record_id = str(row["record_id"])
            diagnostic = diagnostics_by_id.get(record_id)
            if diagnostic is None:
                raise RuntimeError(f"model diagnostics missing record: {record_id}")
            layers = diagnostic["layers"]
            absolute = {
                layer: float(layers[layer]["absolute_alignment_delta"])
                for layer in _DISCRIMINATIVE_LAYERS
            }
            paired_delta = {
                layer: (
                    float(layers[layer]["adapted_paired_cosine"])
                    - float(layers[layer]["base_paired_cosine"])
                    if "adapted_paired_cosine" in layers[layer]
                    and "base_paired_cosine" in layers[layer]
                    else -absolute[layer]
                )
                for layer in _DISCRIMINATIVE_LAYERS
            }
            worst_layer = max(absolute, key=absolute.get)
            row.update({
                "c2_metrics_available": True,
                "c2_alignment_delta": float(np.mean(list(absolute.values()))),
                "c2_worst_layer": worst_layer,
                "c2_worst_layer_alignment_delta": absolute[worst_layer],
                "c2_minus_base_paired_cosine_delta": float(
                    np.mean(list(paired_delta.values()))
                ),
            })
            for layer in _DISCRIMINATIVE_LAYERS:
                row[f"c2_alignment_delta_layer_{layer}"] = absolute[layer]
                row[f"c2_minus_base_paired_cosine_delta_layer_{layer}"] = paired_delta[layer]

            if teacher_bank is not None:
                entry = teacher_bank[record_id]
                margins: list[float] = []
                for layer in _DISCRIMINATIVE_LAYERS:
                    foreground = entry.foreground[layer].float()
                    background = entry.background[layer].float()
                    margin = float(
                        1.0
                        - F.cosine_similarity(
                            foreground.unsqueeze(0), background.unsqueeze(0)
                        ).item()
                    )
                    row[f"rgb_target_background_teacher_margin_layer_{layer}"] = margin
                    margins.append(margin)
                row["rgb_target_background_teacher_margin"] = float(np.mean(margins))
            output.append(row)
        return output

    def _teacher_reliability(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        model_rows = [row for row in rows if row.get("c2_metrics_available")]
        if not model_rows:
            return {
                "status": "MODEL_EVIDENCE_NOT_AVAILABLE",
                "record_count": 0,
                "spearman": {},
                "claim_boundary": "Input quality facts only; no C2 drift was joined.",
            }
        signals = (
            "global_rgb_brightness",
            "global_rgb_contrast",
            "global_rgb_blur",
            "global_rgb_entropy",
            "roi_rgb_brightness",
            "roi_rgb_contrast",
            "roi_rgb_entropy",
            "ir_valid_fov_ratio",
            "rgb_target_background_teacher_margin",
        )
        target = [float(row["c2_alignment_delta"]) for row in model_rows]
        correlations: dict[str, Any] = {}
        for signal in signals:
            paired = [
                (float(row[signal]), float(row["c2_alignment_delta"]))
                for row in model_rows
                if row.get(signal) is not None and math.isfinite(float(row[signal]))
            ]
            if len(paired) < 3:
                correlations[signal] = {
                    "available": False,
                    "record_count": len(paired),
                }
                continue
            left = [value[0] for value in paired]
            right = [value[1] for value in paired]
            lower, upper = self._bootstrap_spearman(left, right)
            correlations[signal] = {
                "available": True,
                "record_count": len(paired),
                "spearman": self._spearman(left, right),
                "bootstrap_95_ci": [lower, upper],
            }
        return {
            "status": "MODEL_EVIDENCE_COMPLETE",
            "record_count": len(model_rows),
            "c2_alignment_delta_mean": float(np.mean(target)),
            "spearman": correlations,
            "claim_boundary": (
                "Correlations are diagnostic associations, not causal evidence and not AIC facts."
            ),
        }

    def _model_evidence(
        self,
        diagnostics: Mapping[str, Any],
        quality_rows: Sequence[Mapping[str, Any]],
        records: Sequence[RGBTRecord],
    ) -> dict[str, Any]:
        layer_deltas = {
            layer: float(diagnostics["layers"][layer]["absolute_alignment_delta_mean"])
            for layer in _DISCRIMINATIVE_LAYERS
        }
        real_drift = max(layer_deltas.values()) > 0.02
        quality_by_id = {str(row["record_id"]): row for row in quality_rows}
        record_by_id = {record.record_id: record for record in records}
        risks: list[float] = []
        drifts: list[float] = []
        potential_neighbors = 0
        all_neighbors = 0
        for row in diagnostics["per_record"]:
            record_id = str(row["record_id"])
            quality = quality_by_id[record_id]
            brightness = float(quality["global_rgb_brightness"])
            contrast = float(quality["global_rgb_contrast"])
            blur = float(quality["global_rgb_blur"])
            risk = (1.0 - brightness) + (1.0 - min(1.0, contrast * 4.0)) + 1.0 / (1.0 + math.log1p(max(0.0, blur)))
            drift = float(np.mean([
                float(row["layers"][layer]["absolute_alignment_delta"])
                for layer in _DISCRIMINATIVE_LAYERS
            ]))
            risks.append(risk)
            drifts.append(drift)
            anchor = record_by_id[record_id]
            _, superclass, _ = _target_head(anchor.query_original)
            neighbor_ids: set[str] = set()
            for layer in _DISCRIMINATIVE_LAYERS:
                neighbor_ids.update(str(value) for value in row["layers"][layer]["top_neighbor_record_ids"])
            for neighbor_id in neighbor_ids:
                neighbor = record_by_id.get(neighbor_id)
                if neighbor is None:
                    continue
                _, other_superclass, _ = _target_head(neighbor.query_original)
                potential_neighbors += int(superclass != "unknown" and superclass == other_superclass)
                all_neighbors += 1
        association = self._spearman(risks, drifts)
        model_false_negative_rate = potential_neighbors / max(1, all_neighbors)
        return {
            "real_alignment_drift": real_drift,
            "layer_absolute_alignment_delta_mean": layer_deltas,
            "rgb_quality_association": association,
            "model_top_neighbor_potential_false_negative_rate": model_false_negative_rate,
            "model_top_neighbor_count": all_neighbors,
            "pareto_conflict": False,
            "drift_threshold": 0.02,
            "claim_boundary": "Target-superclass neighbor rate remains a semantic proxy; layerwise drift is model-derived fact.",
        }

    def run(
        self,
        *,
        records: Sequence[RGBTRecord],
        rgbt_root: Path | str,
        required_assets: Mapping[str, Path | str],
        model_diagnostics_path: Path | str | None = None,
    ) -> Phase17APlusResult:
        self.output_root.mkdir(parents=True, exist_ok=True)
        root = Path(rgbt_root).resolve()
        preflight = self._asset_preflight(required_assets)
        _write_json(self.output_root / "asset_preflight.json", preflight)

        quality_rows = self._quality_rows(records, root)
        negative = self.audit_negatives(records)
        negative_rows = {row["record_id"]: row for row in negative.pop("per_record")}

        assets_ready = preflight["status"] == "ASSETS_READY"
        diagnostics_file = Path(model_diagnostics_path).resolve() if model_diagnostics_path else None
        model_evidence: dict[str, Any] | None = None
        if assets_ready and diagnostics_file is not None and diagnostics_file.is_file():
            diagnostics = json.loads(diagnostics_file.read_text(encoding="utf-8"))
            if int(diagnostics.get("record_count", -1)) != len(records):
                raise RuntimeError("model diagnostics record count mismatch")
            layerwise = {"status": "COMPUTED", **diagnostics["layers"]}
            quality_rows = self._enrich_model_rows(
                quality_rows,
                diagnostics,
                self._teacher_bank(required_assets),
            )
            model_evidence = self._model_evidence(diagnostics, quality_rows, records)
        else:
            layerwise = {
                "status": "NOT_COMPUTED" if not assets_ready else "ASSETS_PRESENT_EXTRACTION_REQUIRED",
                "reason": "C1/C2 and Teacher Bank plus C2 per-record diagnostics are required.",
                "layers": ["8", "16", "24", "final"],
            }
        per_record = [
            {
                **row,
                **{
                    key: value
                    for key, value in negative_rows[row["record_id"]].items()
                    if key not in row
                },
            }
            for row in quality_rows
        ]
        _write_jsonl(self.output_root / "per_record_metrics.jsonl", per_record)
        _write_csv(self.output_root / "rgb_teacher_quality.csv", quality_rows)
        _write_json(self.output_root / "negative_sampling_audit.json", negative)
        _write_json(
            self.output_root / "teacher_reliability.json",
            self._teacher_reliability(quality_rows),
        )
        _write_csv(self.output_root / "subgroup_metrics.csv", self._subgroups(quality_rows))
        worst = sorted(
            quality_rows,
            key=lambda row: (
                -float(row.get("c2_worst_layer_alignment_delta") or -math.inf),
                float(row["global_rgb_brightness"]),
                str(row["record_id"]),
            ),
        )[:100]
        _write_csv(self.output_root / "worst_cases.csv", worst)
        _write_json(self.output_root / "layerwise_metrics.json", layerwise)
        decision_inputs = Phase17DecisionInputs(
            assets_ready=assets_ready and model_evidence is not None,
            real_alignment_drift=(model_evidence or {}).get("real_alignment_drift"),
            rgb_quality_association=(model_evidence or {}).get("rgb_quality_association"),
            false_negative_rate=(model_evidence or {}).get("model_top_neighbor_potential_false_negative_rate"),
            pareto_conflict=(model_evidence or {}).get("pareto_conflict"),
        )
        decision = self.policy.decide(decision_inputs)
        claim_boundary = {
            "input_quality_profile_complete": True,
            "lexical_negative_proxy_complete": True,
            "c2_drift_confirmed": model_evidence is not None,
            "rgb_quality_causality_confirmed": False,
            "model_space_false_negatives_confirmed": False,
        }
        decision_payload = {
            **asdict(decision),
            "allowed_branches": list(DECISION_BRANCHES),
            "claim_boundary": claim_boundary,
            "missing_assets": preflight["missing"],
            "evidence": model_evidence or {},
        }
        _write_json(self.output_root / "decision.json", decision_payload)
        result = Phase17APlusResult(
            status="PHASE_17A_PLUS_COMPLETE" if assets_ready else "PHASE_17A_PLUS_ASSET_BLOCKED",
            decision=decision,
            completed_records=len(records),
            missing_assets=tuple(preflight["missing"]),
            claim_boundary=claim_boundary,
        )
        _write_json(self.output_root / "run_summary.json", asdict(result))
        artifacts = [path for path in sorted(self.output_root.iterdir()) if path.is_file() and path.name != "sha256_manifest.json"]
        _write_json(self.output_root / "sha256_manifest.json", {
            "files": {path.name: _sha256(path) for path in artifacts},
            "record_count": len(records),
            "seed": self.seed,
        })
        return result
