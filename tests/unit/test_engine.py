from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from btd.constants import NO_TUMOR
from btd.inference.engine import ModelLoadError, SegmentationEngine, resolve_providers


def test_resolve_providers_always_has_cpu_fallback() -> None:
    avail = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert resolve_providers("auto", avail) == avail
    assert resolve_providers("cpu", avail) == ["CPUExecutionProvider"]
    assert resolve_providers("tensorrt,cuda", avail) == avail
    assert resolve_providers(["cuda"], ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


@pytest.mark.parametrize("fixture", ["dummy_model", "dummy_raw_model"])
def test_engine_decisions(
    fixture: str, request: pytest.FixtureRequest, bright_image: np.ndarray, black_image: np.ndarray
) -> None:
    engine = SegmentationEngine(request.getfixturevalue(fixture), providers="cpu")
    assert engine.class_names == ("glioma", "meningioma", "pituitary")
    assert engine.input_hw == (320, 320)
    assert engine.precision == "fp32"
    assert engine.sha256 is not None  # checksum verified against model.json

    healthy = engine.predict(black_image)
    assert healthy.decision.label == NO_TUMOR
    assert healthy.detections == []
    assert not healthy.decision.tumor_detected

    sick = engine.predict(bright_image)
    assert sick.decision.label == "meningioma"
    assert len(sick.detections) == 1  # raw layout: the overlapping duplicate is removed by NMS
    det = sick.detections[0]
    x1, y1, x2, y2 = det.box
    assert 0 <= x1 < x2 <= 320 and 0 <= y1 < y2 <= 240
    assert det.mask is not None and det.area_px > 1000
    full = det.full_mask(sick.image_hw)
    assert full.shape == (240, 320)
    ys, xs = np.nonzero(full)
    assert abs(xs.mean() - 160) < 5 and abs(ys.mean() - 120) < 5  # disc centred in the image
    assert set(sick.timings_ms) == {"preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"}


def test_conf_override_does_not_change_decision_threshold(dummy_model: Path) -> None:
    engine = SegmentationEngine(dummy_model, providers="cpu")
    img = np.full((240, 320, 3), 57, np.uint8)  # mean ≈ 0.224 → score ≈ sigmoid(0.7) ≈ 0.67
    base = engine.predict(img)
    assert base.decision.label == "meningioma"
    strict = engine.predict(img, conf=0.95)
    assert strict.detections == []  # hidden from the response…
    assert strict.decision.label == "meningioma"  # …but the calibrated decision is unchanged


def test_with_masks_false(dummy_model: Path, bright_image: np.ndarray) -> None:
    pred = SegmentationEngine(dummy_model, providers="cpu").predict(bright_image, with_masks=False)
    assert pred.detections[0].mask is None
    assert pred.detections[0].area_px > 0


def test_checksum_mismatch_is_rejected(dummy_model: Path, tmp_path: Path) -> None:
    shutil.copy(dummy_model, tmp_path / "model.onnx")
    meta = json.loads((dummy_model.parent / "model.json").read_text())
    meta["artifacts"]["fp32"]["sha256"] = "0" * 64
    (tmp_path / "model.json").write_text(json.dumps(meta))
    with pytest.raises(ModelLoadError, match="Checksum mismatch"):
        SegmentationEngine(tmp_path / "model.onnx", providers="cpu")
    SegmentationEngine(tmp_path / "model.onnx", providers="cpu", verify_checksum=False)


def test_metadata_fallback_without_model_json(
    dummy_model: Path, tmp_path: Path, bright_image: np.ndarray
) -> None:
    shutil.copy(dummy_model, tmp_path / "bare_int8.onnx")
    engine = SegmentationEngine(tmp_path / "bare_int8.onnx", providers="cpu")
    assert engine.class_names == ("glioma", "meningioma", "pituitary")  # from ONNX metadata
    assert engine.conf_threshold == 0.25  # default when no tuned threshold is available
    assert engine.info.as_dict()["file"] == "bare_int8.onnx"
    assert engine.predict(bright_image).decision.tumor_detected


def test_missing_model() -> None:
    with pytest.raises(ModelLoadError, match="not found"):
        SegmentationEngine("does/not/exist.onnx")


def test_warmup_and_info(dummy_model: Path) -> None:
    engine = SegmentationEngine(dummy_model, providers="cpu")
    engine.warmup(1)
    info = engine.info.as_dict()
    assert info["layout"] == "end2end"
    assert info["providers"][-1] == "CPUExecutionProvider"
    assert info["conf_threshold"] == 0.5
