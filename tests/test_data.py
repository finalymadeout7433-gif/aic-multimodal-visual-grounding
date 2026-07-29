from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from aic_baseline.data import AICDataset


def test_dataset_reads_records_and_visible_image_through_public_interface(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "Images" / "visible"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(image_dir / "a.png")
    records = {
        "q_001": {
            "visible": "Images/visible/a.png",
            "infrared": "Images/infrared/a.png",
            "depth": "Images/depth/a.png",
            "query": "the object",
        }
    }
    json_path = tmp_path / "queries.json"
    json_path.write_text(json.dumps(records), encoding="utf-8")

    dataset = AICDataset(dataset_root=tmp_path, queries_path=json_path)
    record = dataset[0]
    image = dataset.load_visible(record)

    assert len(dataset) == 1
    assert record.query_id == "q_001"
    assert record.query == "the object"
    assert record.visible_path == image_dir / "a.png"
    assert image.mode == "RGB"
    assert image.size == (20, 10)


def test_official_chinese_path_sample_is_read_without_copying() -> None:
    root = Path(r"D:\基于大模型的多模态视觉理解与推理-示例数据")
    if not (root / "sample.json").exists():
        return

    dataset = AICDataset(dataset_root=root, queries_path=root / "sample.json")
    record = dataset[0]

    assert record.query_id == "000108_001"
    assert record.bbox == [0.7718, 0.9249, 0.8129, 0.9762]
    assert dataset.load_visible(record).size == (1920, 1080)
