from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .diagnostics import query_category


AIC_PRIMARY_CATEGORY_PROPORTIONS: dict[str, float] = {
    "spatial": 0.3031,
    "ordinal": 0.2209,
    "depth": 0.1799,
    "other": 0.1048,
    "attribute": 0.0975,
    "action": 0.0728,
    "plural_group": 0.0209,
}


def _stable_key(seed: int, *parts: object) -> str:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    cleaned = {
        str(key): max(0.0, float(value))
        for key, value in weights.items()
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        raise ValueError("sampling weights must contain a positive value")
    return {key: value / total for key, value in cleaned.items()}


def _allocate_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError("allocation total must be non-negative")
    normalized = _normalize_weights(weights)
    raw = {key: total * value for key, value in normalized.items()}
    allocated = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(allocated.values())
    order = sorted(
        normalized,
        key=lambda key: (-(raw[key] - allocated[key]), key),
    )
    for key in order[:remainder]:
        allocated[key] += 1
    return allocated


def _decorate_record(record: Mapping[str, Any], *, subset: str) -> dict[str, Any]:
    copied = dict(record)
    copied["query_category"] = query_category(str(record["query"]))
    copied["ranker_subset"] = subset
    return copied


def _cell_quotas(
    records: Sequence[Mapping[str, Any]],
    *,
    size: int,
    category_proportions: Mapping[str, float],
) -> dict[tuple[str, str], int]:
    category_quotas = _allocate_counts(size, category_proportions)
    by_category_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    global_datasets: Counter[str] = Counter()
    for record in records:
        category = query_category(str(record["query"]))
        dataset = str(record["dataset"])
        by_category_dataset[category][dataset] += 1
        global_datasets[dataset] += 1

    quotas: dict[tuple[str, str], int] = {}
    for category, category_count in category_quotas.items():
        available = by_category_dataset.get(category)
        weights: Mapping[str, float] = (
            available if available and sum(available.values()) else global_datasets
        )
        for dataset, count in _allocate_counts(category_count, weights).items():
            quotas[(category, dataset)] = count
    return quotas


def _sample_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    size: int,
    seed: int,
    subset_name: str,
    category_proportions: Mapping[str, float],
    image_caps: Sequence[int],
    required_records: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if size <= 0:
        raise ValueError("subset size must be positive")
    if size > len(records):
        raise ValueError(
            f"requested {size} records from only {len(records)} records"
        )
    if not image_caps or any(cap <= 0 for cap in image_caps):
        raise ValueError("image caps must be positive")

    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        query_id = str(record["query_id"])
        if query_id in by_id:
            raise ValueError(f"duplicate query_id in source records: {query_id}")
        by_id[query_id] = record

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    image_counts: Counter[str] = Counter()
    cell_counts: Counter[tuple[str, str]] = Counter()
    for required in required_records:
        query_id = str(required["query_id"])
        if query_id not in by_id:
            raise ValueError(f"required query_id is missing: {query_id}")
        if query_id in selected_ids:
            continue
        decorated = _decorate_record(by_id[query_id], subset=subset_name)
        selected.append(decorated)
        selected_ids.add(query_id)
        image_counts[str(decorated["image_key"])] += 1
        cell_counts[
            (
                str(decorated["query_category"]),
                str(decorated["dataset"]),
            )
        ] += 1
    if len(selected) > size:
        raise ValueError("required records exceed requested subset size")

    quotas = _cell_quotas(
        records,
        size=size,
        category_proportions=category_proportions,
    )
    pools: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        query_id = str(record["query_id"])
        if query_id in selected_ids:
            continue
        cell = (
            query_category(str(record["query"])),
            str(record["dataset"]),
        )
        pools[cell].append(record)
    for cell, pool in pools.items():
        pool.sort(
            key=lambda record: _stable_key(
                seed,
                subset_name,
                cell[0],
                cell[1],
                record["query_id"],
            )
        )

    cursors: Counter[tuple[str, str]] = Counter()
    cap_used = image_caps[0]
    for cap in image_caps:
        cap_used = cap
        made_progress = True
        while len(selected) < size and made_progress:
            made_progress = False
            cells = sorted(
                quotas,
                key=lambda cell: (
                    -(quotas[cell] - cell_counts[cell]),
                    cell[0],
                    cell[1],
                ),
            )
            for cell in cells:
                if len(selected) >= size:
                    break
                if cell_counts[cell] >= quotas[cell]:
                    continue
                pool = pools.get(cell, [])
                while cursors[cell] < len(pool):
                    candidate = pool[cursors[cell]]
                    cursors[cell] += 1
                    image_key = str(candidate["image_key"])
                    if image_counts[image_key] >= cap:
                        continue
                    decorated = _decorate_record(
                        candidate,
                        subset=subset_name,
                    )
                    selected.append(decorated)
                    selected_ids.add(str(candidate["query_id"]))
                    image_counts[image_key] += 1
                    cell_counts[cell] += 1
                    made_progress = True
                    break

        if len(selected) >= size:
            break

    if len(selected) < size:
        remaining = sorted(
            (
                record
                for record in records
                if str(record["query_id"]) not in selected_ids
            ),
            key=lambda record: _stable_key(
                seed,
                subset_name,
                "fill",
                record["query_id"],
            ),
        )
        for cap in image_caps:
            cap_used = max(cap_used, cap)
            for candidate in remaining:
                if len(selected) >= size:
                    break
                query_id = str(candidate["query_id"])
                if query_id in selected_ids:
                    continue
                image_key = str(candidate["image_key"])
                if image_counts[image_key] >= cap:
                    continue
                decorated = _decorate_record(
                    candidate,
                    subset=subset_name,
                )
                selected.append(decorated)
                selected_ids.add(query_id)
                image_counts[image_key] += 1
            if len(selected) >= size:
                break

    if len(selected) != size:
        raise ValueError(
            f"could not build {subset_name}: requested={size}, "
            f"selected={len(selected)}"
        )
    actual_category_counts = Counter(r["query_category"] for r in selected)
    requested_category_counts = _allocate_counts(size, category_proportions)
    summary = {
        "name": subset_name,
        "records": len(selected),
        "unique_images": len(image_counts),
        "maximum_queries_per_image": max(image_counts.values(), default=0),
        "image_cap_attempted": cap_used,
        "image_cap_relaxation_used": (
            max(image_counts.values(), default=0) > image_caps[0]
        ),
        "requested_category_counts": dict(
            sorted(requested_category_counts.items())
        ),
        "category_quota_deviation": {
            category: actual_category_counts.get(category, 0) - requested
            for category, requested in sorted(
                requested_category_counts.items()
            )
        },
        "category_counts": dict(
            sorted(actual_category_counts.items())
        ),
        "dataset_counts": dict(
            sorted(Counter(str(r["dataset"]) for r in selected).items())
        ),
        "query_ids_sha256": hashlib.sha256(
            "\n".join(str(r["query_id"]) for r in selected).encode("utf-8")
        ).hexdigest().upper(),
    }
    return selected, summary


def build_nested_ranker_subsets(
    records: Sequence[Mapping[str, Any]],
    *,
    pilot_size: int,
    full_size: int,
    seed: int,
    category_proportions: Mapping[str, float] = AIC_PRIMARY_CATEGORY_PROPORTIONS,
    pilot_image_cap: int = 2,
    full_image_cap: int = 3,
) -> dict[str, Any]:
    """Build deterministic nested pilot/full training subsets."""

    if pilot_size > full_size:
        raise ValueError("pilot_size must not exceed full_size")
    pilot, pilot_summary = _sample_subset(
        records,
        size=pilot_size,
        seed=seed,
        subset_name="rank_train_pilot",
        category_proportions=category_proportions,
        image_caps=(pilot_image_cap, pilot_image_cap + 1, pilot_image_cap + 2),
    )
    full, full_summary = _sample_subset(
        records,
        size=full_size,
        seed=seed,
        subset_name="rank_train_full",
        category_proportions=category_proportions,
        image_caps=(full_image_cap, full_image_cap + 1),
        required_records=pilot,
    )
    return {
        "pilot": pilot,
        "full": full,
        "summary": {"pilot": pilot_summary, "full": full_summary},
    }


def build_evaluation_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    size: int,
    seed: int,
    name: str,
    image_cap: int = 4,
    category_proportions: Mapping[str, float] = AIC_PRIMARY_CATEGORY_PROPORTIONS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _sample_subset(
        records,
        size=size,
        seed=seed,
        subset_name=name,
        category_proportions=category_proportions,
        image_caps=(image_cap, image_cap + 1),
    )


def split_by_image(
    records: Sequence[Mapping[str, Any]],
    *,
    dev_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")
    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_image[str(record["image_key"])].append(record)
    if len(by_image) < 2:
        raise ValueError("at least two image keys are required")
    image_keys = sorted(
        by_image,
        key=lambda image_key: _stable_key(seed, "rank-dev", image_key),
    )
    target_dev_records = max(1, round(len(records) * dev_fraction))
    dev_images: set[str] = set()
    dev_count = 0
    for image_key in image_keys:
        if dev_count >= target_dev_records and dev_images:
            break
        dev_images.add(image_key)
        dev_count += len(by_image[image_key])
    if len(dev_images) == len(by_image):
        dev_images.remove(image_keys[-1])
    train = [
        dict(record)
        for record in records
        if str(record["image_key"]) not in dev_images
    ]
    dev = [
        dict(record)
        for record in records
        if str(record["image_key"]) in dev_images
    ]
    return train, dev
