"""Tests for the synthetic data generator.

The generator is the backbone of the whole project — it is the only source of a
daily time series, and therefore the only thing that makes an ACWR possible.
"""

from __future__ import annotations

import pandas as pd
import pytest

from injury_risk.config import INJURY_PRONE_RATE, POSITIONS
from injury_risk.data.generate_synthetic import generate

N_ATHLETES = 12
N_DAYS = 90


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return generate(n_athletes=N_ATHLETES, n_days=N_DAYS, seed=42)


def test_shape_and_identifiers(df: pd.DataFrame):
    assert len(df) == N_ATHLETES * N_DAYS
    assert df["athlete_id"].nunique() == N_ATHLETES
    assert df.groupby("athlete_id")["day"].size().eq(N_DAYS).all()


def test_no_missing_values(df: pd.DataFrame):
    assert not df.isna().any().any()


def test_generation_is_deterministic_for_a_seed():
    a = generate(n_athletes=5, n_days=40, seed=7)
    b = generate(n_athletes=5, n_days=40, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_data():
    a = generate(n_athletes=5, n_days=40, seed=7)
    b = generate(n_athletes=5, n_days=40, seed=8)
    assert not a["training_load"].equals(b["training_load"])


def test_static_attributes_are_constant_per_athlete(df: pd.DataFrame):
    """Age, position and proneness describe the athlete, not the day."""
    for col in ("age", "position", "injury_prone", "previous_injuries", "baseline_hr"):
        assert df.groupby("athlete_id")[col].nunique().eq(1).all(), col


def test_values_stay_physiologically_plausible(df: pd.DataFrame):
    assert df["training_load"].ge(0).all()
    assert df["sleep_hours"].between(3.5, 10.0).all()
    assert df["soreness"].between(0, 10).all()
    assert df["resting_hr"].between(40, 95).all()
    assert set(df["position"]) <= set(POSITIONS)


def test_labels_cover_the_three_risk_levels(df: pd.DataFrame):
    assert set(df["risk_level"]) <= {0, 1, 2}


def test_class_distribution_is_imbalanced():
    """The imbalance is deliberate — it is what justifies SMOTE downstream."""
    big = generate(n_athletes=60, n_days=200, seed=42)
    shares = big["risk_level"].value_counts(normalize=True)
    assert shares[0] > shares.get(1, 0) > shares.get(2, 0)
    assert shares[0] > 0.5, "the low-risk class should dominate"
    assert shares.get(2, 0) < 0.2, "the high-risk class should stay rare"


def test_injury_prone_share_is_near_the_configured_rate():
    big = generate(n_athletes=200, n_days=30, seed=42)
    prone = big.groupby("athlete_id")["injury_prone"].first().mean()
    assert prone == pytest.approx(INJURY_PRONE_RATE, abs=0.10)


def test_dates_are_daily_and_ordered(df: pd.DataFrame):
    one = df[df["athlete_id"] == 0].sort_values("day")
    assert one["date"].is_monotonic_increasing
    assert one["date"].diff().dropna().eq(pd.Timedelta(days=1)).all()
