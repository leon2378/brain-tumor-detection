from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import btd
from btd.inference.quality import MAX_COLOURFUL_FRACTION, colourful_fraction, input_warnings
from btd.utils import imread

SAMPLES = Path(btd.__file__).parent / "api" / "static" / "samples"


def test_greyscale_images_pass() -> None:
    grey = np.full((64, 64, 3), 90, np.uint8)
    assert colourful_fraction(grey) == 0.0
    assert input_warnings(grey) == []
    assert input_warnings(np.zeros((64, 64), np.uint8)) == []  # single channel
    # small channel differences, like JPEG chroma noise on a greyscale scan, are not colour
    noisy = grey.astype(np.int16) + np.random.default_rng(0).integers(-8, 9, grey.shape)
    assert input_warnings(noisy.clip(0, 255).astype(np.uint8)) == []


def test_colour_image_is_flagged() -> None:
    img = np.zeros((64, 64, 3), np.uint8)
    img[..., 2] = 200  # red (BGR)
    [warning] = input_warnings(img)
    assert warning.code == "not_greyscale"
    assert "greyscale" in warning.message


def test_small_colour_annotation_is_tolerated() -> None:
    img = np.full((100, 100, 3), 60, np.uint8)
    img[:5, :, 2] = 255  # a coloured label strip over 5% of the slice
    assert colourful_fraction(img) < MAX_COLOURFUL_FRACTION
    assert input_warnings(img) == []


@pytest.mark.parametrize("path", sorted(SAMPLES.glob("*.jpg")), ids=lambda p: p.stem)
def test_real_mri_slices_pass(path: Path) -> None:
    assert input_warnings(imread(path)) == []
