"""Shared fixtures. Nothing here needs torch: models are tiny hand-built ONNX graphs (see btd.testing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def dummy_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from btd.testing import make_dummy_model

    return make_dummy_model(tmp_path_factory.mktemp("model_e2e"), layout="end2end")


@pytest.fixture(scope="session")
def dummy_raw_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from btd.testing import make_dummy_model

    return make_dummy_model(tmp_path_factory.mktemp("model_raw"), layout="raw")


@pytest.fixture
def bright_image() -> np.ndarray:
    return np.full((240, 320, 3), 120, np.uint8)


@pytest.fixture
def black_image() -> np.ndarray:
    return np.zeros((240, 320, 3), np.uint8)


@pytest.fixture(scope="session")
def synthetic_brisc(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Small BRISC-shaped tree (path contains a space, like a Windows 'Projects/Brain Tumor Detection')."""
    from btd.data.synthetic import generate

    return generate(
        tmp_path_factory.mktemp("raw data"), n_train_per_class=12, n_test_per_class=4, size=128, seed=7
    )
