import json
import subprocess
import sys

import pytest

from tsanomaly.cli import build_parser, main


def test_build_parser_requires_a_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_detect_command_end_to_end(tmp_path, capsys):
    csv_path = tmp_path / "series.csv"
    lines = ["value"]
    import math

    period = 8
    n = 64
    for i in range(n):
        v = 10.0 + 5.0 * math.sin(2 * math.pi * (i % period) / period)
        if i == 32:
            v += 25.0
        lines.append(f"{v}")
    csv_path.write_text("\n".join(lines) + "\n")

    rc = main(
        [
            "detect",
            "--input",
            str(csv_path),
            "--value-col",
            "value",
            "--period",
            str(period),
            "--method",
            "zscore",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["anomaly_count"] >= 1
    assert any(a["index"] == 32 for a in output["anomalies"])


def test_detect_command_text_format(tmp_path, capsys):
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("value\n" + "\n".join(str(float(x)) for x in range(1, 21)) + "\n")
    rc = main(
        [
            "detect",
            "--input",
            str(csv_path),
            "--value-col",
            "value",
            "--trend-window",
            "3",
            "--method",
            "iqr",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "method=iqr" in out


def test_rolling_detect_command(tmp_path, capsys):
    csv_path = tmp_path / "series.csv"
    values = [10.0] * 10 + [1000.0] + [10.0] * 5
    csv_path.write_text("value\n" + "\n".join(str(v) for v in values) + "\n")
    rc = main(
        [
            "rolling-detect",
            "--input",
            str(csv_path),
            "--value-col",
            "value",
            "--window",
            "5",
            "--threshold",
            "3.0",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert any(a["index"] == 10 for a in output["anomalies"])


def test_demo_command_reports_reasonable_recall(capsys):
    rc = main(
        [
            "demo",
            "--length",
            "200",
            "--period",
            "24",
            "--anomalies",
            "6",
            "--seed",
            "42",
            "--method",
            "zscore",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["n_points"] == 200
    assert len(output["true_anomaly_indices"]) == 6
    # z-score on a clean decomposition should catch most of the injected spikes
    assert output["recall"] >= 0.6
    assert 0.0 <= output["precision"] <= 1.0


def test_detect_command_handles_missing_column_gracefully(tmp_path, capsys):
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("value\n1\n2\n3\n")
    rc = main(
        [
            "detect",
            "--input",
            str(csv_path),
            "--value-col",
            "does-not-exist",
            "--trend-window",
            "2",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "does-not-exist" in err


def test_cli_runs_as_module_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "tsanomaly.cli", "demo", "--length", "120", "--period", "12", "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(result.stdout)
    assert output["n_points"] == 120
