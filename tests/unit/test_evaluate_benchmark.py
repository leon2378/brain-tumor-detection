"""Evaluation + benchmark plumbing, driven by the dummy ONNX model (no torch needed)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from btd.cli import main
from btd.export.benchmark import render_markdown as bench_md
from btd.export.benchmark import run_benchmark
from btd.training.evaluate import evaluate_engine, load_split, plot_confusion_matrix, render_markdown
from btd.utils import JsonFormatter, git_commit, read_json, setup_logging


@pytest.fixture(scope="module")
def prepared(synthetic_brisc: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from btd.data.prepare import prepare_dataset

    out = tmp_path_factory.mktemp("prepared") / "yolo"
    prepare_dataset(synthetic_brisc, out, val_fraction=0.25, seed=0, workers=2)
    return out


@pytest.fixture
def model_dir(dummy_model: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "model"
    shutil.copytree(dummy_model.parent, dst)
    return dst


def test_load_split(prepared: Path) -> None:
    test = load_split(prepared, "test")
    assert len(test) == 16
    assert all(s.mask is not None for s in test if s.label != "no_tumor")
    assert len(load_split(prepared, "val", limit=3)) == 3
    with pytest.raises(ValueError, match="No samples"):
        load_split(prepared, "holdout")


def test_evaluate_engine_report(prepared: Path, dummy_model: Path, tmp_path: Path) -> None:
    rep = evaluate_engine(dummy_model, prepared, "val", tune=True, providers="cpu")
    assert rep["threshold_tuned_on_this_split"] is True
    assert len(rep["threshold_sweep"]) == 19
    assert rep["confusion_matrix"]["labels"] == ["glioma", "meningioma", "pituitary", "no_tumor"]
    assert sum(map(sum, rep["confusion_matrix"]["matrix"])) == rep["images"]
    seg = rep["segmentation"]
    assert seg["images"] == sum(1 for s in load_split(prepared, "val") if s.label != "no_tumor")
    assert 0.0 <= seg["mean_dice"] <= 1.0
    assert set(rep["latency"]) == {"total", "inference"}
    md = render_markdown([rep], "Eval")
    assert "| model |" in md and "Confusion matrix" in md
    png = plot_confusion_matrix(rep, tmp_path / "cm.png")
    assert png is None or png.is_file()

    fixed = evaluate_engine(dummy_model, prepared, "test", threshold=0.99, providers="cpu")
    assert fixed["threshold"] == 0.99
    assert fixed["classification"]["per_class"]["no_tumor"]["recall"] == 1.0  # nothing passes 0.99…


def test_benchmark_cpu_and_unavailable_provider(model_dir: Path, tmp_path: Path) -> None:
    report = run_benchmark(
        model_dir, None, providers=("cpu", "tensorrt"), runs=3, warmup=1, n_images=2, out_dir=tmp_path
    )
    rows = {r["provider"]: r for r in report["rows"]}
    assert rows["cpu"]["status"] == "ok" and rows["cpu"]["total"]["p50_ms"] > 0
    assert rows["tensorrt"]["status"].startswith("unavailable")
    assert "| precision |" in bench_md(report)
    assert (tmp_path / "benchmark.md").is_file()


def test_cli_evaluate_and_benchmark(model_dir: Path, prepared: Path, tmp_path: Path) -> None:
    out = tmp_path / "reports"
    assert main(["evaluate", "--model-dir", str(model_dir), "--data", str(prepared), "--out", str(out)]) == 0
    assert (out / "evaluation_test.md").is_file()
    assert "fp32" in read_json(model_dir / "model.json")["metrics"]["test"]
    assert (
        main(
            [
                "evaluate",
                "--model-dir",
                str(model_dir),
                "--data",
                str(prepared),
                "--split",
                "val",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "benchmark",
                "--model-dir",
                str(model_dir),
                "--data",
                "",
                "--runs",
                "2",
                "--warmup",
                "1",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads((out / "benchmark.json").read_text())["rows"][0]["status"] == "ok"


def test_json_logging(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO", json_logs=True)
    log = logging.getLogger("btd.test")
    log.info("hello", extra={"request_id": "r1"})
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("failed")
    lines = [json.loads(line) for line in capsys.readouterr().err.strip().splitlines()]
    assert lines[0]["msg"] == "hello" and lines[0]["request_id"] == "r1"
    assert "RuntimeError: boom" in lines[1]["exc"]
    assert isinstance(JsonFormatter().format(logging.makeLogRecord({"msg": "x"})), str)
    setup_logging("INFO", json_logs=False)


def test_git_commit_is_optional(tmp_path: Path) -> None:
    assert git_commit(tmp_path) is None
