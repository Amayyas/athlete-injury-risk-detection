"""Tests for the single entry point.

The CLI is what CI (#24) and the Docker image (#30) will call, so a broken command
would break the build in a much more confusing way than a failing unit test.
"""

from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from injury_risk.cli import app

runner = CliRunner()


def _text(result) -> str:
    """Typer sends parameter errors to stderr, which CliRunner captures separately."""
    out = result.stdout or ""
    try:
        return out + (result.stderr or "")
    except ValueError:  # stderr not captured separately
        return out


COMMANDS = ["download", "data", "train", "tune", "benchmark", "shap", "dashboard"]


def test_help_lists_every_pipeline_stage():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in _text(result)


def test_every_command_has_its_own_help():
    for command in COMMANDS:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, f"`{command} --help` failed"


def test_unknown_track_is_rejected():
    result = runner.invoke(app, ["train", "--track", "nonexistent"])
    assert result.exit_code != 0
    assert "unknown track" in _text(result)


def test_unknown_dataset_key_is_rejected():
    result = runner.invoke(app, ["download", "not-a-dataset"])
    assert result.exit_code != 0
    assert "unknown key" in _text(result)


def test_data_generates_a_usable_dataset(tmp_path):
    """The `data` command is the first step of the pipeline — it must actually work."""
    out = tmp_path / "synthetic.parquet"
    result = runner.invoke(app, ["data", "--athletes", "4", "--days", "60", "--output", str(out)])

    assert result.exit_code == 0, _text(result)
    assert out.exists()

    df = pd.read_parquet(out)
    assert len(df) == 4 * 60
    assert {"injury_onset", "is_injured", "injury_next_7d"} <= set(df.columns)
