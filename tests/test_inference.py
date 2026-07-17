"""Tests for the serving seam.

`inference` is where the model is turned into a prediction and an explanation. The
dashboard and (soon) the API both go through it, so a bug here is a bug everywhere the
model is used.
"""

from __future__ import annotations

import pytest

from injury_risk.config import POSITION_TO_CODE
from injury_risk.features.engineering import SYNTHETIC_FEATURE_COLS
from injury_risk.inference import AthleteInputs, Prediction, load_predictor, predictor_path

pytestmark = pytest.mark.skipif(
    not predictor_path("synthetic").exists(),
    reason="needs a trained model; run `injury-risk train --track synthetic`",
)


# --------------------------------------------------------------------------- #
# Feature mapping (runs without a model)
# --------------------------------------------------------------------------- #


def test_to_features_produces_every_model_column():
    row = AthleteInputs().to_features()
    assert set(SYNTHETIC_FEATURE_COLS) <= set(row.columns)
    assert len(row) == 1


def test_to_features_encodes_the_position():
    row = AthleteInputs(position="forward").to_features()
    assert row["position_code"].iloc[0] == POSITION_TO_CODE["forward"]
    assert AthleteInputs(position="alien").to_features()["position_code"].iloc[0] == -1


def test_to_features_reflects_the_snapshot_assumptions():
    row = AthleteInputs(
        acute_load=120, chronic_load=60, resting_hr=70, baseline_hr=55
    ).to_features()
    r = row.iloc[0]
    assert r["acwr"] == pytest.approx(2.0)
    assert r["training_load_7d"] == 120  # the acute load
    assert r["training_load_28d"] == 60  # the chronic load
    assert 60 < r["training_load_14d"] < 120  # interpolated between them
    assert r["hr_delta"] == 15
    assert r["load_trend_7d"] == 0.0  # a snapshot has no direction


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


def test_prediction_is_a_probability_with_a_decision():
    pred = load_predictor("synthetic").predict(AthleteInputs())
    assert isinstance(pred, Prediction)
    assert 0.0 <= pred.probability <= 1.0
    assert pred.at_risk == (pred.probability >= pred.threshold)


def test_a_risky_athlete_scores_higher_than_a_healthy_one():
    p = load_predictor("synthetic")
    healthy = p.predict(
        AthleteInputs(acute_load=64, chronic_load=64, sleep_hours=7.8, soreness=1, resting_hr=52)
    )
    risky = p.predict(
        AthleteInputs(
            acute_load=120,
            chronic_load=60,
            sleep_hours=4.5,
            soreness=9,
            resting_hr=72,
            injury_prone=True,
            previous_injuries=5,
            days_since_injury=15,
        )
    )
    assert risky.probability > healthy.probability


def test_predictor_is_cached():
    assert load_predictor("synthetic") is load_predictor("synthetic")


def test_missing_model_raises_an_actionable_error():
    load_predictor.cache_clear()
    with pytest.raises(FileNotFoundError, match="train"):
        load_predictor("nonexistent_track")


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #


def test_explanation_has_one_contribution_per_feature():
    p = load_predictor("synthetic")
    explanation = p.explain(AthleteInputs(acute_load=120, chronic_load=60, soreness=8))
    assert len(explanation.values) == len(p.feature_cols)
    # The plot shows human-readable inputs, not standardised ones.
    assert explanation.data is not None
    assert len(explanation.data) == len(p.feature_cols)
