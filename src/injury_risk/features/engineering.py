"""Feature engineering for injury-risk detection.

This module gathers the project's "domain" logic:

- computation of the **ACWR** (Acute:Chronic Workload Ratio), a metric actually
  used by strength & conditioning staff (optimal zone 0.8–1.3, danger > 1.5);
- **rolling features** (7/14/28-day moving averages) on workload and soreness;
- **workload trend** over 7 days (slope);
- position encoding;
- a **composite risk score** function based on business rules, shared between the
  synthetic generator (label creation) and the dashboard (real-time scoring
  without a trained model).

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
    ACWR_DANGER,
    ACWR_ELEVATED,
    ACWR_UNDER,
    ACWR_ZONES,
    CHRONIC_WINDOW,
    HR_DELTA_RANGE,
    LOAD_TREND_WINDOW,
    POSITION_TO_CODE,
    PREVIOUS_INJURIES_RANGE,
    RECENT_RETURN_DAYS,
    RISK_HIGH_THRESHOLD,
    RISK_LOW_THRESHOLD,
    ROLLING_WINDOWS,
    SLEEP_RANGE,
    SLEEP_TARGET,
    SORENESS_ONSET,
    SORENESS_RANGE,
    W_ACWR_DANGER,
    W_ACWR_ELEVATED,
    W_ACWR_UNDER,
    W_INJURY_PRONE,
    W_PREVIOUS_INJURIES,
    W_RECENT_RETURN,
    W_RESTING_HR,
    W_SLEEP,
    W_SORENESS,
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
# Composite risk score (business rules) — shared by generator/dashboard
# --------------------------------------------------------------------------- #


def composite_risk_score(
    *,
    acwr: float,
    soreness: float,
    sleep_hours: float,
    resting_hr: float,
    baseline_hr: float = 55.0,
    injury_prone: bool = False,
    previous_injuries: int = 0,
    days_since_injury: float = 365.0,
) -> float:
    """Continuous risk score (0 = low, ~1 = very high), based on rules.

    This function encodes the project's domain knowledge. It is used to:

    - generate the synthetic dataset labels (with noise added upstream);
    - compute a real-time score in the dashboard, **before** the ML model is
      trained/loaded.

    Contributions are deliberately additive and bounded to stay readable and
    explainable. Every weight and range comes from :mod:`injury_risk.config`.
    """
    score = 0.0

    # 1) ACWR: the danger zone is the main risk factor.
    if acwr >= ACWR_DANGER:
        score += W_ACWR_DANGER
    elif acwr >= ACWR_ELEVATED:
        score += W_ACWR_ELEVATED
    elif acwr < ACWR_UNDER:
        score += W_ACWR_UNDER  # under-loading: moderate risk (detraining)

    # 2) High soreness (0-10 scale).
    score += np.clip((soreness - SORENESS_ONSET) / SORENESS_RANGE, 0, 1) * W_SORENESS

    # 3) Lack of sleep (below the target is a recognised risk factor).
    score += np.clip((SLEEP_TARGET - sleep_hours) / SLEEP_RANGE, 0, 1) * W_SLEEP

    # 4) Resting heart rate elevated vs baseline (fatigue/stress).
    score += np.clip((resting_hr - baseline_hr) / HR_DELTA_RANGE, 0, 1) * W_RESTING_HR

    # 5) History: proneness + number of past injuries.
    if injury_prone:
        score += W_INJURY_PRONE
    score += np.clip(previous_injuries / PREVIOUS_INJURIES_RANGE, 0, 1) * W_PREVIOUS_INJURIES

    # 6) Recent return from injury: tissue still fragile.
    if days_since_injury < RECENT_RETURN_DAYS:
        score += W_RECENT_RETURN * (1 - days_since_injury / RECENT_RETURN_DAYS)

    return float(np.clip(score, 0.0, 1.0))


def risk_score_to_level(
    score: float,
    low_thr: float = RISK_LOW_THRESHOLD,
    high_thr: float = RISK_HIGH_THRESHOLD,
) -> int:
    """Convert a continuous score into a risk level: 0=low, 1=moderate, 2=high."""
    if score >= high_thr:
        return 2
    if score >= low_thr:
        return 1
    return 0


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
