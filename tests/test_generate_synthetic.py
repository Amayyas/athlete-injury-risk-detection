"""Tests for the synthetic data generator.

The generator is the backbone of the whole project — it is the only source of a
daily time series, and therefore the only thing that makes an ACWR possible.
"""

from __future__ import annotations

import pandas as pd
import pytest

from injury_risk.config import INJURY_PRONE_RATE, POSITIONS, PREDICTION_HORIZON_DAYS, TARGET_COL
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
    """Age, position and proneness describe the athlete, not the day.

    `previous_injuries` is deliberately excluded: it now *grows* when an athlete
    gets injured, which is the point of simulating events.
    """
    for col in ("age", "position", "injury_prone", "baseline_hr"):
        assert df.groupby("athlete_id")[col].nunique().eq(1).all(), col


def test_values_stay_physiologically_plausible(df: pd.DataFrame):
    assert df["training_load"].ge(0).all()
    assert df["sleep_hours"].between(3.5, 10.0).all()
    assert df["soreness"].between(0, 10).all()
    assert df["resting_hr"].between(40, 95).all()
    assert set(df["position"]) <= set(POSITIONS)


def test_target_is_binary_and_rare(df: pd.DataFrame):
    """The imbalance is deliberate — it is what justifies SMOTE downstream."""
    assert set(df[TARGET_COL]) <= {0, 1}
    modelled = df[~df["is_injured"] & df["horizon_complete"]]
    assert 0.0 < modelled[TARGET_COL].mean() < 0.20, "injuries must stay rare"


def test_injury_prone_share_is_near_the_configured_rate():
    big = generate(n_athletes=200, n_days=30, seed=42)
    prone = big.groupby("athlete_id")["injury_prone"].first().mean()
    assert prone == pytest.approx(INJURY_PRONE_RATE, abs=0.10)


def test_dates_are_daily_and_ordered(df: pd.DataFrame):
    one = df[df["athlete_id"] == 0].sort_values("day")
    assert one["date"].is_monotonic_increasing
    assert one["date"].diff().dropna().eq(pd.Timedelta(days=1)).all()


# --------------------------------------------------------------------------- #
# The injury process (the heart of the redesign)
# --------------------------------------------------------------------------- #


def test_injuries_actually_happen(df: pd.DataFrame):
    assert df["injury_onset"].sum() > 0
    assert df["is_injured"].sum() > 0


def test_an_injured_athlete_cannot_get_injured_again_while_out(df: pd.DataFrame):
    """A sidelined athlete is not exposed to a *new* injury."""
    assert not (df["is_injured"] & df["injury_onset"]).any()


def test_an_injury_sidelines_the_athlete_the_following_days():
    d = generate(n_athletes=8, n_days=300, seed=3)
    for _, athlete in d.groupby("athlete_id"):
        athlete = athlete.sort_values("day").reset_index(drop=True)
        onsets = athlete.index[athlete["injury_onset"]].tolist()
        for i in onsets:
            if i + 1 < len(athlete):
                assert athlete.loc[i + 1, "is_injured"], "the day after an onset must be an absence"


def test_injury_history_grows_and_recency_resets():
    d = generate(n_athletes=8, n_days=300, seed=3)
    for _, athlete in d.groupby("athlete_id"):
        athlete = athlete.sort_values("day").reset_index(drop=True)
        assert athlete["previous_injuries"].is_monotonic_increasing
        onsets = athlete.index[athlete["injury_onset"]].tolist()
        for i in onsets:
            if i + 1 < len(athlete):
                assert athlete.loc[i + 1, "days_since_injury"] == 0.0


def test_target_looks_forward_only(df: pd.DataFrame):
    """`injury_next_7d` is 1 iff an onset occurs strictly within the next 7 days."""
    one = df[df["athlete_id"] == 0].sort_values("day").reset_index(drop=True)
    onsets = set(one.index[one["injury_onset"]])
    for i in one.index:
        expected = any(i < j <= i + PREDICTION_HORIZON_DAYS for j in onsets)
        if one.loc[i, "horizon_complete"]:
            assert bool(one.loc[i, TARGET_COL]) == expected, f"day {i}"


def test_the_censored_tail_is_flagged(df: pd.DataFrame):
    """The last days cannot know their own future and must be excluded."""
    last = df.groupby("athlete_id")["day"].transform("max")
    assert not df.loc[df["day"] > last - PREDICTION_HORIZON_DAYS, "horizon_complete"].any()
