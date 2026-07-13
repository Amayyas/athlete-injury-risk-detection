"""Unit tests for the feature engineering (core domain logic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from injury_risk.features.engineering import (
    POSITION_TO_CODE,
    acwr_zone,
    add_acwr,
    add_load_trend,
    add_rolling_features,
    build_features,
    composite_risk_score,
    encode_position,
    risk_score_to_level,
)


def _toy_df(n_days: int = 40) -> pd.DataFrame:
    """Two athletes with a constant load (expected ACWR ≈ 1)."""
    rows = []
    for aid in (0, 1):
        for d in range(n_days):
            rows.append(
                {
                    "athlete_id": aid,
                    "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=d),
                    "day": d,
                    "position": "midfielder",
                    "age": 25,
                    "previous_injuries": 0,
                    "days_since_injury": 300,
                    "training_load": 100.0,
                    "soreness": 3.0,
                    "sleep_hours": 8.0,
                    "resting_hr": 55.0,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# acwr_zone
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.5, "under"),
        (0.79, "under"),
        (0.8, "optimal"),
        (1.0, "optimal"),
        (1.29, "optimal"),
        (1.3, "elevated"),
        (1.49, "elevated"),
        (1.5, "danger"),
        (2.5, "danger"),
    ],
)
def test_acwr_zone_boundaries(value, expected):
    assert acwr_zone(value) == expected


def test_acwr_zone_nan():
    assert acwr_zone(float("nan")) == "unknown"


# --------------------------------------------------------------------------- #
# add_acwr
# --------------------------------------------------------------------------- #


def test_add_acwr_constant_load_is_one():
    df = build_features(_toy_df())
    # Constant load => ACWR ≈ 1 (optimal zone) outside the warmup.
    stable = df[df["day"] >= 28]
    assert np.allclose(stable["acwr"], 1.0, atol=1e-6)
    assert (stable["acwr_zone"] == "optimal").all()


def test_add_acwr_per_athlete_no_leak():
    """An athlete's ACWR must not be influenced by another."""
    df = _toy_df()
    df.loc[df["athlete_id"] == 1, "training_load"] = 999.0
    out = add_acwr(df)
    a0 = out[out["athlete_id"] == 0]
    assert np.allclose(a0["acwr"], 1.0, atol=1e-6)


def test_add_acwr_spike_raises_ratio():
    df = _toy_df(n_days=40)
    # Load spike on the last days of one athlete.
    mask = (df["athlete_id"] == 0) & (df["day"] >= 35)
    df.loc[mask, "training_load"] = 400.0
    out = add_acwr(df)
    last = out[(out["athlete_id"] == 0)].iloc[-1]
    assert last["acwr"] > 1.5
    assert last["acwr_zone"] == "danger"


# --------------------------------------------------------------------------- #
# rolling features & trend
# --------------------------------------------------------------------------- #


def test_rolling_features_columns_created():
    out = add_rolling_features(_toy_df())
    for col in ("training_load_7d", "training_load_28d", "soreness_7d", "sleep_hours_7d"):
        assert col in out.columns
    # Constant load => moving average == value.
    assert np.allclose(out["training_load_7d"], 100.0)


def test_load_trend_flat_is_zero():
    out = add_load_trend(_toy_df())
    assert np.allclose(out["load_trend_7d"], 0.0, atol=1e-9)


def test_load_trend_increasing_is_positive():
    df = _toy_df(n_days=10)
    df.loc[df["athlete_id"] == 0, "training_load"] = np.arange(10) * 10.0
    out = add_load_trend(df)
    last = out[out["athlete_id"] == 0].iloc[-1]
    assert last["load_trend_7d"] > 0


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #


def test_encode_position_known_and_unknown():
    df = pd.DataFrame({"position": ["Forward", "midfielder", "alien"]})
    out = encode_position(df)
    assert out["position_code"].tolist() == [
        POSITION_TO_CODE["forward"],
        POSITION_TO_CODE["midfielder"],
        -1,  # unknown position
    ]


# --------------------------------------------------------------------------- #
# composite risk score
# --------------------------------------------------------------------------- #


def test_composite_risk_score_bounds():
    s = composite_risk_score(acwr=1.0, soreness=3, sleep_hours=8, resting_hr=55)
    assert 0.0 <= s <= 1.0


def test_composite_risk_score_danger_higher_than_safe():
    safe = composite_risk_score(acwr=1.0, soreness=2, sleep_hours=8, resting_hr=52)
    risky = composite_risk_score(
        acwr=1.8,
        soreness=9,
        sleep_hours=4,
        resting_hr=80,
        injury_prone=True,
        previous_injuries=4,
        days_since_injury=10,
    )
    assert risky > safe
    assert risky >= 0.6


def test_risk_score_to_level_thresholds():
    assert risk_score_to_level(0.1) == 0  # < low_thr
    assert risk_score_to_level(0.2) == 1  # between low_thr and high_thr
    assert risk_score_to_level(0.5) == 2  # >= high_thr
    # Customizable thresholds.
    assert risk_score_to_level(0.4, low_thr=0.33, high_thr=0.55) == 1
