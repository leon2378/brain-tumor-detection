from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from btd.cli import build_parser, main
from btd.utils import imwrite


def test_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("env", "data", "train", "export", "evaluate", "benchmark", "package", "predict", "serve"):
        assert cmd in out


def test_train_overrides_are_collected() -> None:
    args = build_parser().parse_args(["train", "epochs=3", "batch=8"])
    assert args.overrides == ["epochs=3", "batch=8"]


def test_serve_host_and_port_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BTD_HOST", raising=False)
    monkeypatch.delenv("BTD_PORT", raising=False)
    args = build_parser().parse_args(["serve"])
    assert (args.host, args.port) == ("127.0.0.1", 8000)
    monkeypatch.setenv("BTD_HOST", "0.0.0.0")  # what the Docker images set
    monkeypatch.setenv("BTD_PORT", "8080")  # what SageMaker needs
    args = build_parser().parse_args(["serve"])
    assert (args.host, args.port) == ("0.0.0.0", 8080)
    assert build_parser().parse_args(["serve", "--port", "9000"]).port == 9000  # a flag still wins


def test_predict_and_package(dummy_model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    img = tmp_path / "in put.png"
    imwrite(img, np.full((100, 120, 3), 130, np.uint8))
    assert (
        main(
            [
                "predict",
                "--model",
                str(dummy_model),
                str(img),
                "--providers",
                "cpu",
                "--save-dir",
                str(tmp_path / "o"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["decision"] == "meningioma"
    assert (tmp_path / "o" / "in put_overlay.png").is_file()

    assert (
        main(
            [
                "package",
                "--model-dir",
                str(dummy_model.parent),
                "--precision",
                "fp32",
                "--dest",
                str(tmp_path / "models"),
            ]
        )
        == 0
    )
    meta = json.loads((tmp_path / "models" / "model.json").read_text())
    assert list(meta["artifacts"]) == ["fp32"] and meta["artifacts"]["fp32"]["file"] == "model.onnx"
    assert (
        main(
            [
                "package",
                "--model-dir",
                str(dummy_model.parent),
                "--precision",
                "int8",
                "--dest",
                str(tmp_path / "m2"),
            ]
        )
        == 1
    )


def test_data_synthetic_and_audit_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "syn"
    assert (
        main(
            [
                "data",
                "synthetic",
                "--out",
                str(out),
                "--train-per-class",
                "3",
                "--test-per-class",
                "1",
                "--size",
                "96",
            ]
        )
        == 0
    )
    assert (
        main(["data", "audit", "--data", str(out), "--out", str(tmp_path / "audit"), "--workers", "2"]) == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["images"] == 16


def test_env_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["env"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert "python" in info and "hints" in info


def test_errors_return_code_2(tmp_path: Path) -> None:
    assert main(["data", "prepare", "--src", str(tmp_path), "--out", str(tmp_path / "o")]) == 2


def test_latest_weights_picks_newest(tmp_path: Path) -> None:
    import os
    import time

    from btd.cli import latest_weights

    with pytest.raises(FileNotFoundError):
        latest_weights(tmp_path)
    # the last one is nested, like runs made before `btd train` pinned `project` to an absolute path
    names = ["brisc-yolo26s", "brisc-yolo26s2", "runs/segment/brisc-yolo26s3"]
    for i, name in enumerate(names):
        w = tmp_path / name / "weights" / "best.pt"
        w.parent.mkdir(parents=True)
        w.write_bytes(b"x")
        os.utime(w, (time.time() + i, time.time() + i))
    assert latest_weights(tmp_path).parent.parent.name == "brisc-yolo26s3"
