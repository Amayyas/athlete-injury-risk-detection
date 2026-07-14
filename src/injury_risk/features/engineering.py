"""Feature engineering for injury-risk detection.

This module gathers the project's "domain" logic:

- computation of the **ACWR** (Acute:Chronic Workload Ratio), a metric actually
  used by strength & conditioning staff (optimal zone 0.8–1.3, danger > 1.5);
- **rolling features** (7/14/28-day moving averages) on workload and soreness;
- **workload trend** over 7 days (slope);
- position encoding.

The rule-based scoring (composite score + risk factors) lives in
:mod:`injury_risk.features.risk_factors`.

Every constant it relies on lives in :mod:`injury_risk.config`.

Functions operating on time series expect a DataFrame sorted by athlete then by
date, with at least the columns: ``athlete_id``, ``date``, ``training_load``,
``soreness``, ``sleep_hours``, ``resting_hr``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from injury_risk.config import (
    ACUTE_WINDOW,
    ACWR_ZONES,
    CHRONIC_WINDOW,
    LOAD_TREND_WINDOW,
    POSITION_TO_CODE,
    ROLLING_WINDOWS,
)

# --------------------------------------------------------------------------- #
# ACWR & zones
# --------------------------------------------------------------------------- #


def acwr_zone(acwr: float) -> str:
    """Return the ACWR zone ('under' / 'optimal' / 'elevated' / 'danger')."""
    if acwr is None or (isinstance(acwr, float) and np.isnan(acwr)):
        return "unknown"
    for zone, (low, high) in ACWR_ZONES.items():
        if low <= acwr < high:
            return zone
    return "danger"  # acwr >= ACWR_DANGER


def add_acwr(df: pd.DataFrame, load_col: str = "training_load") -> pd.DataFrame:
    """Add the ``acute_load``, ``chronic_load``, ``acwr`` and ``acwr_zone`` columns.

    The ACWR is the ratio of acute load (7-day mean) to chronic load (28-day
    mean), computed **per athlete** to avoid any leakage between athletes.
    """
    df = df.copy()
    grp = df.groupby("athlete_id")[load_col]
    df["acute_load"] = grp.transform(lambda s: s.rolling(ACUTE_WINDOW, min_periods=1).mean())
    df["chronic_load"] = grp.transform(lambda s: s.rolling(CHRONIC_WINDOW, min_periods=1).mean())
    # Avoid division by ~0 (chronic load near zero at the very start of a series).
    df["acwr"] = df["acute_load"] / df["chronic_load"].replace(0, np.nan)
    df["acwr"] = df["acwr"].fillna(1.0)  # start of series: neutral ratio
    df["acwr_zone"] = df["acwr"].apply(acwr_zone)
    return df


# --------------------------------------------------------------------------- #
# Rolling features & trend
# --------------------------------------------------------------------------- #


def add_rolling_features(
    df: pd.DataFrame,
    cols: tuple[str, ...] = ("training_load", "soreness", "sleep_hours"),
    windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Add ``{col}_{window}d`` moving averages per athlete."""
    df = df.copy()
    for col in cols:
        for w in windows:
            name = f"{col}_{w}d"
            df[name] = df.groupby("athlete_id")[col].transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean()
            )
    return df


def _slope(values: np.ndarray) -> float:
    """Slope of a simple linear regression over a window of values."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n)
    # degree-1 polyfit -> [slope, intercept]
    return float(np.polyfit(x, values, 1)[0])


def add_load_trend(
    df: pd.DataFrame,
    load_col: str = "training_load",
    window: int = LOAD_TREND_WINDOW,
) -> pd.DataFrame:
    """Add ``load_trend_7d``: workload slope over the window (per athlete)."""
    df = df.copy()
    name = f"load_trend_{window}d"
    df[name] = df.groupby("athlete_id")[load_col].transform(
        lambda s: s.rolling(window, min_periods=2).apply(_slope, raw=True)
    )
    df[name] = df[name].fillna(0.0)
    return df


# --------------------------------------------------------------------------- #
# Categorical encoding
# --------------------------------------------------------------------------- #


def encode_position(df: pd.DataFrame, col: str = "position") -> pd.DataFrame:
    """Encode the position as an integer (``position_code``)."""
    df = df.copy()
    df["position_code"] = df[col].str.lower().map(POSITION_TO_CODE).fillna(-1).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Full pipeline (synthetic time series)
# --------------------------------------------------------------------------- #


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the whole temporal feature engineering on the daily data.

    Expects the columns: athlete_id, date, position, training_load, soreness,
    sleep_hours, resting_hr (+ any static columns, which are kept).
    """
    df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)
    df = add_acwr(df)
    df = add_rolling_features(df)
    df = add_load_trend(df)
    if "position" in df.columns:
        df = encode_position(df)
    return df


# Columns used as model features (synthetic track).
SYNTHETIC_FEATURE_COLS = [
    "age",
    "position_code",
    "previous_injuries",
    "days_since_injury",
    "training_load",
    "resting_hr",
    "sleep_hours",
    "soreness",
    "training_load_7d",
    "training_load_14d",
    "training_load_28d",
    "soreness_7d",
    "soreness_14d",
    "sleep_hours_7d",
    "acute_load",
    "chronic_load",
    "acwr",
    "load_trend_7d",
]
