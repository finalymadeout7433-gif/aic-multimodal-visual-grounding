from __future__ import annotations

from PIL import Image
import pytest
import torch

from aic_baseline.external_eval import ExternalPrediction
from aic_baseline.florence import (
    FlorenceCandidate,
    FlorenceOutputError,
    FlorencePrediction,
)
from aic_baseline.model_adapters import (
    FlorenceExternalPredictor,
    FlorenceTileExternalPredictor,
    GroundingDinoExternalPredictor,
    extract_grounding_dino_candidates,
)


class FakeFlorence:
    def predict(self, *, image: Image.Image, query: str) -> FlorencePrediction:
        candidates = [
            FlorenceCandidate([1, 2, 10, 20], "first"),
            FlorenceCandidate([3, 4, 30, 40], "second"),
        ]
        return FlorencePrediction("raw", candidates, candidates[0])


class EmptyFlorence:
    def predict(self, *, image: Image.Image, query: str) -> FlorencePrediction:
        raise FlorenceOutputError("no candidate")


def test_florence_adapter_preserves_all_candidates_with_null_scores() -> None:
    predictor = FlorenceExternalPredictor(FakeFlorence())
    prediction = predictor.predict(
        image=Image.new("RGB", (100, 50)),
        query="target",
    )

    assert isinstance(prediction, ExternalPrediction)
    assert [candidate.label for candidate in prediction.candidates] == [
        "first",
        "second",
    ]
    assert [candidate.score for candidate in prediction.candidates] == [
        None,
        None,
    ]


def test_florence_adapter_turns_only_output_error_into_no_candidate() -> None:
    predictor = FlorenceExternalPredictor(EmptyFlorence())
    prediction = predictor.predict(
        image=Image.new("RGB", (100, 50)),
        query="target",
    )

    assert prediction.candidates == []
    assert prediction.raw_output == {"florence_output_error": "no candidate"}


class TileAwareFlorence:
    def predict(self, *, image: Image.Image, query: str) -> FlorencePrediction:
        candidate = FlorenceCandidate(
            [0.0, 0.0, float(image.width), float(image.height)],
            "target",
        )
        return FlorencePrediction("raw", [candidate], candidate)


def test_florence_tile_adapter_maps_candidates_and_preserves_full_first() -> None:
    predictor = FlorenceTileExternalPredictor(TileAwareFlorence())
    prediction = predictor.predict(
        image=Image.new("RGB", (1000, 800)),
        query="target",
    )

    assert prediction.candidates[0].source == "full"
    assert prediction.candidates[0].pixel_bbox == [0.0, 0.0, 1000.0, 800.0]
    assert {candidate.source for candidate in prediction.candidates[1:]} == {
        "tile_0",
        "tile_1",
        "tile_2",
        "tile_3",
    }
    assert prediction.candidates[-1].pixel_bbox == [
        444.0,
        356.0,
        1000.0,
        800.0,
    ]


def test_grounding_dino_candidate_parser_sorts_and_limits_scores() -> None:
    result = {
        "boxes": torch.tensor(
            [
                [10.0, 20.0, 30.0, 40.0],
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        ),
        "scores": torch.tensor([0.2, 0.9, 0.5]),
        "text_labels": ["low", "high", "middle"],
    }

    candidates = extract_grounding_dino_candidates(result, max_candidates=2)

    assert [candidate.label for candidate in candidates] == ["high", "middle"]
    assert [candidate.score for candidate in candidates] == pytest.approx(
        [0.9, 0.5]
    )
    assert candidates[0].pixel_bbox == [1.0, 2.0, 3.0, 4.0]


def test_grounding_dino_candidate_parser_rejects_non_finite_score() -> None:
    result = {
        "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "scores": torch.tensor([float("nan")]),
        "text_labels": ["bad"],
    }

    with pytest.raises(ValueError, match="finite"):
        extract_grounding_dino_candidates(result, max_candidates=20)


def test_grounding_dino_adapter_accepts_official_custom_model_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}

    class FakeProcessorLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["processor_path"] = path
            calls["processor_kwargs"] = kwargs
            return object()

    class FakeModel:
        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True

    class FakeModelClass:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model_path"] = path
            calls["model_kwargs"] = kwargs
            return FakeModel()

    monkeypatch.setattr(
        "aic_baseline.model_adapters.AutoProcessor",
        FakeProcessorLoader,
    )
    predictor = GroundingDinoExternalPredictor(
        model_path=tmp_path,
        model_class=FakeModelClass,
        model_load_kwargs={"use_safetensors": False},
        device="cpu",
        dtype=torch.float32,
    )

    assert predictor.model_path == tmp_path.resolve()
    assert calls["model_path"] == tmp_path.resolve()
    assert calls["model_kwargs"] == {
        "local_files_only": True,
        "dtype": torch.float32,
        "use_safetensors": False,
    }
    assert calls["eval"] is True
