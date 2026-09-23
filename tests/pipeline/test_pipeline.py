"""End-to-end ML pipeline on synthetic BRISC-shaped data (CPU, a few minutes).

synthetic data → prepare (leakage-safe split) → train YOLO26n-seg → export FP32/FP16/INT8 (+ head selection,
threshold tuning) → numpy engine == Ultralytics on the same ONNX → evaluate → package → serve through the API.

Run with ``pytest -m pipeline``. Needs torch + ultralytics (``pip install -e ".[train,serve,cpu,dev]"``).
Set ``BTD_WEIGHTS_DIR`` to cache pretrained weights and ``BTD_PIPELINE_ARTIFACTS`` to keep the reports.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.pipeline
pytest.importorskip("torch")
pytest.importorskip("ultralytics")

REPO = Path(__file__).resolve().parents[2]
IMGSZ = 160


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    from btd.data.prepare import prepare_dataset
    from btd.data.synthetic import generate

    root = tmp_path_factory.mktemp("pipeline ws")  # space on purpose (Windows-style project paths)
    raw = generate(root / "raw", n_train_per_class=40, n_test_per_class=10, size=IMGSZ, seed=11)
    dataset = root / "processed"
    prepare_dataset(raw, dataset, val_fraction=0.2, seed=0, workers=2)
    weights_dir = Path(os.environ.get("BTD_WEIGHTS_DIR", root / "weights"))
    weights_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "dataset": dataset, "weights": weights_dir / "yolo26n-seg.pt"}


@pytest.fixture(scope="module")
def trained(workspace: dict[str, Path]) -> Path:
    from btd.training.train import load_config, train

    cfg = load_config(
        REPO / "configs" / "train.yaml",
        [
            f"model={workspace['weights']}",
            f"data={workspace['dataset'] / 'data.yaml'}",
            "epochs=12",
            "warmup_epochs=1",
            f"imgsz={IMGSZ}",
            "batch=16",
            "device=cpu",
            "workers=0",
            "amp=false",
            "plots=false",
            "close_mosaic=3",
            "deterministic=false",
            f"project={workspace['root'] / 'runs'}",
            "name=ci",
        ],
    )
    summary = train(cfg)
    best = Path(summary["best_weights"])
    assert best.is_file()
    assert json.loads((best.parent.parent / "run_summary.json").read_text())["epochs_completed"] == 12
    return best


@pytest.fixture(scope="module")
def exported(trained: Path, workspace: dict[str, Path]) -> Path:
    from btd.export.pipeline import export_model

    out = workspace["root"] / "model"
    meta = export_model(trained, workspace["dataset"], out, REPO / "configs" / "export.yaml", imgsz=IMGSZ)
    assert set(meta["artifacts"]) == {"fp32", "fp16", "int8"}
    assert meta["layout"] in {"end2end", "raw"}
    assert 0.05 <= meta["thresholds"]["conf"] <= 0.95
    sizes = {p: a["bytes"] for p, a in meta["artifacts"].items()}
    assert sizes["int8"] < sizes["fp16"] < sizes["fp32"]
    return out


def test_engine_matches_ultralytics_on_same_onnx(exported: Path, workspace: dict[str, Path]) -> None:
    """The torch-free serving pipeline must reproduce Ultralytics' own ONNX predictions."""
    from ultralytics import YOLO

    from btd.inference.engine import SegmentationEngine
    from btd.training.evaluate import load_split
    from btd.utils import imread

    onnx_path = exported / "model_fp32.onnx"
    conf = 0.01  # low on purpose: even a briefly-trained model yields plenty of detections to compare
    engine = SegmentationEngine(onnx_path, providers="cpu", conf=conf, agnostic=False, max_det=300)
    reference = YOLO(str(onnx_path), task="segment")
    compared = 0
    for sample in load_split(workspace["dataset"], "test"):
        img = imread(sample.image)
        ours = sorted(engine.predict(img).detections, key=lambda d: -d.confidence)
        ref = reference.predict(img, conf=conf, iou=0.7, retina_masks=True, verbose=False)[0]
        assert len(ours) == len(ref.boxes)
        if not ours:
            continue
        order = np.argsort(-ref.boxes.conf.numpy(), kind="stable")
        np.testing.assert_allclose([d.box for d in ours], ref.boxes.xyxy.numpy()[order], atol=0.05)
        np.testing.assert_allclose([d.confidence for d in ours], ref.boxes.conf.numpy()[order], atol=1e-4)
        ref_masks = ref.masks.data.numpy().astype(bool)[order]
        for det, mask in zip(ours, ref_masks, strict=True):
            full = det.full_mask(img.shape[:2])
            assert (full ^ mask).sum() <= max(5, 0.01 * mask.sum())
        compared += len(ours)
    assert compared > 0


def test_quantised_models_agree_with_fp32(exported: Path, workspace: dict[str, Path]) -> None:
    from btd.inference.engine import SegmentationEngine
    from btd.training.evaluate import load_split
    from btd.utils import imread

    samples = load_split(workspace["dataset"], "test")
    fp32 = SegmentationEngine(exported / "model_fp32.onnx", providers="cpu")
    for precision in ("fp16", "int8"):
        q = SegmentationEngine(exported / f"model_{precision}.onnx", providers="cpu")
        assert q.precision == precision
        agree = sum(
            fp32.predict(imread(s.image)).decision.label == q.predict(imread(s.image)).decision.label
            for s in samples
        )
        assert agree / len(samples) >= (0.95 if precision == "fp16" else 0.8), precision


def test_evaluate_package_and_serve(exported: Path, workspace: dict[str, Path], tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from btd.api.app import create_app
    from btd.api.settings import Settings
    from btd.cli import main
    from btd.training.evaluate import evaluate_engine, load_split, ultralytics_metrics
    from btd.utils import imread, read_json, write_json

    report = evaluate_engine(exported / "model_int8.onnx", workspace["dataset"], "test", providers="cpu")
    assert report["images"] == 40
    assert report["segmentation"]["images"] == 30
    assert 0.0 <= report["classification"]["macro_f1"] <= 1.0
    maps = ultralytics_metrics(
        exported / "model_int8.onnx", workspace["dataset"] / "data.yaml", "test", IMGSZ
    )
    assert 0.0 <= maps["mask_map50_95"] <= 1.0

    assert (
        main(
            [
                "evaluate",
                "--model-dir",
                str(exported),
                "--data",
                str(workspace["dataset"]),
                "--out",
                str(tmp_path / "reports"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "benchmark",
                "--model-dir",
                str(exported),
                "--data",
                str(workspace["dataset"]),
                "--runs",
                "5",
                "--warmup",
                "1",
                "--out",
                str(tmp_path / "reports"),
            ]
        )
        == 0
    )
    assert read_json(exported / "model.json")["metrics"]["test"]["int8"]["accuracy"] >= 0.0

    models = tmp_path / "models"
    assert main(["package", "--model-dir", str(exported), "--precision", "int8", "--dest", str(models)]) == 0
    settings = Settings(model_path=models / "model.onnx", providers="cpu", log_json=False)
    sample = next(s for s in load_split(workspace["dataset"], "test") if s.label != "no_tumor")
    ok, buf = cv2.imencode(".png", imread(sample.image))
    assert ok
    with TestClient(create_app(settings)) as client:
        assert client.get("/ready").status_code == 200
        r = client.post("/v1/predict", files={"file": ("s.png", buf.tobytes(), "image/png")})
        assert r.status_code == 200
        assert r.json()["model"]["precision"] == "int8"

    keep = os.environ.get("BTD_PIPELINE_ARTIFACTS")
    if keep:
        dest = Path(keep)
        dest.mkdir(parents=True, exist_ok=True)
        for f in (tmp_path / "reports").iterdir():
            shutil.copy2(f, dest / f.name)
        shutil.copy2(exported / "model.json", dest / "model.json")
        write_json(dest / "evaluation_int8_test.json", report)
