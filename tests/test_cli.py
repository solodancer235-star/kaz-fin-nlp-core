"""Unit tests for CLI commands."""

import json
import os
import pytest
from kaz_fin_nlp.cli import main


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "0.1.0" in captured.out


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([])
    assert code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "kaz-fin-nlp" in captured.out


def test_cli_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["benchmark", "--iterations", "15"])
    assert code == 0
    captured = capsys.readouterr()
    assert "KAZAKHSTANI FINANCIAL NLP CORE - BENCHMARK" in captured.out
    assert "Articles Analyzed" in captured.out


def test_cli_analyze(capsys: pytest.CaptureFixture[str]) -> None:
    text = (
        "KASE: Kaspi.kz акциялары 15% өсім көрсетті, таза табыс 300 млрд теңгеге жетті."
    )
    code = main(["analyze", "--text", text])
    assert code == 0
    captured = capsys.readouterr()
    assert "Document Analysis Results" in captured.out
    assert "Detected Language" in captured.out
    assert "Kaspi.kz" in captured.out


def test_cli_stream_demo(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["stream-demo", "--articles", "2", "--delay", "0.01"])
    assert code == 0
    captured = capsys.readouterr()
    assert "REAL-TIME STREAMING HUD" in captured.out


def test_cli_export_report(tmp_path: pytest.TempPathFactory) -> None:
    out_file = str(tmp_path) + "/out_rep.json"
    code = main(["export-report", "--output", out_file])
    assert code == 0
    assert os.path.exists(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "report_id" in data
    assert "regulatory" in data
