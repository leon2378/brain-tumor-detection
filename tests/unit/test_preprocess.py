from __future__ import annotations

import numpy as np
import pytest

from btd.inference.preprocess import PAD_VALUE, letterbox, preprocess, to_tensor


@pytest.mark.parametrize(("h", "w"), [(512, 512), (240, 320), (320, 240), (100, 700), (641, 639)])
def test_letterbox_shape_and_padding(h: int, w: int) -> None:
    img = np.full((h, w, 3), 200, np.uint8)
    out, info = letterbox(img, (640, 640))
    assert out.shape == (640, 640, 3)
    assert info.orig_hw == (h, w)
    r = min(640 / h, 640 / w)
    assert info.gain == pytest.approx(r)
    nh, nw = round(h * r), round(w * r)
    # content region is untouched by padding, padding is the Ultralytics grey
    assert (out[info.pad_top : info.pad_top + nh, info.pad_left : info.pad_left + nw] == 200).all()
    if info.pad_top:
        assert (out[: info.pad_top] == PAD_VALUE).all()
    if info.pad_left:
        assert (out[:, : info.pad_left] == PAD_VALUE).all()


def test_grayscale_input_is_promoted_to_bgr() -> None:
    out, _ = letterbox(np.zeros((64, 64), np.uint8), (128, 128))
    assert out.shape == (128, 128, 3)


def test_to_tensor_rgb_order_and_range() -> None:
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 0] = 255  # blue channel in BGR
    x = to_tensor(img)
    assert x.shape == (1, 3, 4, 4)
    assert x.dtype == np.float32
    assert x[0, 2].max() == pytest.approx(1.0)  # blue ends up last in RGB
    assert x[0, 0].max() == 0.0


def test_preprocess_matches_ultralytics_letterbox() -> None:
    lb_mod = pytest.importorskip("ultralytics.data.augment")
    rng = np.random.default_rng(0)
    for h, w in [(512, 512), (333, 517), (700, 120)]:
        img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        ours, _ = letterbox(img, (640, 640))
        theirs = lb_mod.LetterBox((640, 640), auto=False, stride=32)(image=img)
        np.testing.assert_array_equal(ours, theirs)
    x, _ = preprocess(img, (640, 640))
    assert x.shape == (1, 3, 640, 640)
