from __future__ import annotations

import torch

from aic_rgbtir.data import RGBTRecord
from aic_rgbtir.query_probe import QueryInterfaceSmoke


def _record(record_id: str, query: str, bbox: tuple[float, float, float, float]) -> RGBTRecord:
    return RGBTRecord(
        record_id=record_id,
        source_dataset="flir",
        split="train",
        rgb_relpath="image_data/flir/rgb/shared.jpg",
        tir_relpath="image_data/flir/ir/shared.jpg",
        query_original=query,
        bbox_xywh_pixel=(bbox[0] * 100, bbox[1] * 100, (bbox[2] - bbox[0]) * 100, (bbox[3] - bbox[1]) * 100),
        bbox_xyxy_normalized=bbox,
        width=100,
        height=100,
        illumination="NL",
        weather="FY",
        object_size="SO",
        occlusion="NO",
        crowded="C",
        scene="UB",
    )


def _query_encoder(query: str) -> torch.Tensor:
    values = [float(len(query)), float(sum(ord(ch) for ch in query) % 997)]
    return torch.tensor(values, dtype=torch.float32)


def _roi_encoder(_record: RGBTRecord, candidate) -> torch.Tensor:
    x1, y1, x2, y2 = candidate.bbox_xyxy_normalized
    return torch.tensor([x2 - x1, y2 - y1], dtype=torch.float32)


def test_query_interface_smoke_builds_role_and_geometry_candidates_deterministically() -> None:
    man = _record(
        "q_man",
        "A man holding a phone beside the bus",
        (0.10, 0.10, 0.30, 0.70),
    )
    bus = _record("q_bus", "The bus beside the man", (0.40, 0.20, 0.90, 0.80))
    phone = _record("q_phone", "The phone held by the man", (0.20, 0.30, 0.27, 0.40))
    smoke = QueryInterfaceSmoke(
        query_encoder=_query_encoder,
        roi_encoder=_roi_encoder,
        maximum_records=1,
        seed=20260812,
    )

    first = smoke.run([man], candidate_pool=[man, bus, phone])
    second = smoke.run([man], candidate_pool=[phone, man, bus])

    assert first.status == "PHASE_18_Q0_READY"
    assert first.as_dict() == second.as_dict()
    assert first.record_count == 1
    assert first.query_dimension == first.roi_dimension == 2
    kinds = {row["kind"] for row in first.candidate_audit}
    assert {"positive", "same_image_object", "reference", "part", "geometry_near_miss"} <= kinds
    assert first.role_audit[0]["target"] == "man"
    assert first.role_audit[0]["reference"] == "bus"
    assert first.role_audit[0]["part"] == "phone"


def test_query_interface_smoke_rejects_nonfinite_or_dimension_mismatch() -> None:
    record = _record("q1", "The person", (0.1, 0.1, 0.2, 0.2))
    bad = QueryInterfaceSmoke(
        query_encoder=lambda _query: torch.tensor([float("nan"), 1.0]),
        roi_encoder=_roi_encoder,
        maximum_records=1,
    )
    try:
        bad.run([record], candidate_pool=[record])
    except ValueError as error:
        assert "non-finite query embedding" in str(error)
    else:
        raise AssertionError("non-finite query embedding must fail Q0")


def test_query_interface_smoke_never_builds_cross_pair_same_image_candidates() -> None:
    anchor = _record("q_anchor", "The person beside the bus", (0.1, 0.1, 0.2, 0.5))
    other = RGBTRecord(
        **{
            **anchor.__dict__,
            "record_id": "q_other",
            "rgb_relpath": "image_data/flir/rgb/other.jpg",
            "tir_relpath": "image_data/flir/ir/other.jpg",
            "query_original": "The bus",
        }
    )
    result = QueryInterfaceSmoke(
        query_encoder=_query_encoder,
        roi_encoder=_roi_encoder,
        maximum_records=1,
    ).run([anchor], candidate_pool=[anchor, other])

    assert all(
        row["source_record_id"] != "q_other"
        for row in result.candidate_audit
        if row["kind"] in {"same_image_object", "reference", "part"}
    )
