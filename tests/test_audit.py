from __future__ import annotations

import json

import numpy as np
from PIL import Image

from tools.audit_dataset import audit


def test_audit_minimal_multimodal_dataset(tmp_path):
    for modality in ("visible", "infrared", "depth"):
        (tmp_path / "Images" / modality).mkdir(parents=True)

    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    depth = np.full((4, 6), 1_000, dtype=np.uint16)
    Image.fromarray(rgb, mode="RGB").save(tmp_path / "Images/visible/000001.png")
    Image.fromarray(rgb, mode="RGB").save(tmp_path / "Images/infrared/000001.png")
    Image.fromarray(depth).save(tmp_path / "Images/depth/000001.png")

    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        json.dumps(
            {
                "000001_001": {
                    "visible": "Images/visible/000001.png",
                    "infrared": "Images/infrared/000001.png",
                    "depth": "Images/depth/000001.png",
                    "query": "the object",
                }
            }
        ),
        encoding="utf-8",
    )

    result = audit(tmp_path, queries_path)

    assert result["query_count"] == 1
    assert result["image_group_count"] == 1
    assert result["missing_files"] == []
    assert result["decode_failures"] == []
    assert result["modality_size_mismatches"] == []
    assert result["unique_image_counts"] == {
        "visible": 1,
        "infrared": 1,
        "depth": 1,
    }
