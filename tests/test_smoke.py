"""Tests for the model-quality guard.

A guard that never fires is worse than none — it grants false confidence. So these
verify it actually *detects*, by breaking the model on purpose and asserting it says so.
"""

from __future__ import annotations

import json

import pytest

from injury_risk.config import (
    SMOKE_MAX_ROC_AUC,
    SMOKE_MIN_ROC_AUC,
    SMOKE_N_ATHLETES,
    TARGET_COL,
)
from injury_risk.models import smoke as smoke_module
from injury_risk.models.smoke import format_report, run_smoke_test, write_report

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def result():
    return run_smoke_test()


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_the_current_model_passes(result):
    assert result.passed, format_report(result)
    assert result.n_athletes == SMOKE_N_ATHLETES
    assert 0.0 < result.prevalence < 0.2


def test_the_run_is_deterministic():
    """The thresholds are only meaningful because the run does not wander."""
    a, b = run_smoke_test(), run_smoke_test()
    assert a.average_precision == b.average_precision
    assert a.roc_auc == b.roc_auc


def test_report_is_written_as_json(result, tmp_path):
    path = tmp_path / "smoke.json"
    write_report(result, path)
    body = json.loads(path.read_text())
    assert body["passed"] is True
    assert {c["name"] for c in body["checks"]} == {
        "average_precision",
        "roc_auc",
        "lift_over_chance",
        "roc_auc_not_suspicious",
    }


def test_the_report_names_every_failing_check(result):
    text = format_report(result)
    assert "RESULT: PASS" in text
    assert "drift vs recorded baseline" in text


# --------------------------------------------------------------------------- #
# It has to actually catch things
# --------------------------------------------------------------------------- #


def test_it_catches_a_collapse_in_signal(monkeypatch):
    """Destroy the relationship between features and target: the floors must fail."""
    original = smoke_module.build_features

    def shuffled_target(df):
        out = original(df)
        out[TARGET_COL] = out[TARGET_COL].sample(frac=1.0, random_state=0).to_numpy()
        return out

    monkeypatch.setattr(smoke_module, "build_features", shuffled_target)
    result = run_smoke_test()

    assert not result.passed
    failed = {c.name for c in result.checks if not c.passed}
    assert {"average_precision", "roc_auc", "lift_over_chance"} <= failed
    assert result.roc_auc < SMOKE_MIN_ROC_AUC


def test_it_catches_leakage(monkeypatch):
    """The ceiling exists for this: a leak makes the numbers *better*, not worse.

    This project shipped that failure twice — athletes spanning CV folds, then a
    calibrator silently dropping the grouping — and floors alone would have missed both.
    """
    monkeypatch.setattr(
        smoke_module,
        "SYNTHETIC_FEATURE_COLS",
        [*smoke_module.SYNTHETIC_FEATURE_COLS, TARGET_COL],
    )
    result = run_smoke_test()

    assert not result.passed
    assert result.roc_auc > SMOKE_MAX_ROC_AUC
    failed = {c.name for c in result.checks if not c.passed}
    assert failed == {"roc_auc_not_suspicious"}
