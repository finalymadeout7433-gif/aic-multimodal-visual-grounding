from __future__ import annotations

from argparse import Namespace

import pytest

from tools.run_external_diagnostics import (
    _model_runtime_asset_hashes,
    _selected_model_weight_path,
)


@pytest.mark.parametrize(
    ("use_pytorch_bin", "filename"),
    ((False, "model.safetensors"), (True, "pytorch_model.bin")),
)
def test_selected_model_weight_path_matches_loader_choice(
    tmp_path,
    use_pytorch_bin: bool,
    filename: str,
) -> None:
    expected = tmp_path / filename
    expected.write_bytes(b"weights")

    selected = _selected_model_weight_path(
        Namespace(model_path=tmp_path, use_pytorch_bin=use_pytorch_bin)
    )

    assert selected == expected


def test_selected_model_weight_path_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="selected model weight"):
        _selected_model_weight_path(
            Namespace(model_path=tmp_path, use_pytorch_bin=False)
        )


def test_model_runtime_asset_hashes_include_only_runtime_text_assets(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "vocab.txt").write_text("token", encoding="utf-8")
    (tmp_path / "tokenizer.model").write_bytes(b"tokenizer")
    (tmp_path / "README.md").write_text("docs", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    fingerprints = _model_runtime_asset_hashes(tmp_path)

    assert set(fingerprints) == {"config.json", "tokenizer.model", "vocab.txt"}
    assert all(len(value) == 64 for value in fingerprints.values())
