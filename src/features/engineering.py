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

Functions operating on time series expect a DataFrame sorted by athlete then by
date, with at least the columns: ``athlete_id``, ``date``, ``training_load``,
``soreness``, ``sleep_hours``, ``resting_hr``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #

# Rolling windows (in days) used for the ACWR and the rolling features.
ACUTE_WINDOW = 7
CHRONIC_WINDOW = 28
ROLLING_WINDOWS = (7, 14, 28)

# ACWR zone bounds (from the training-load literature).
ACWR_ZONES = {
    "under": (0.0, 0.8),  # under-training
    "optimal": (0.8, 1.3),  # "sweet spot"
    "elevated": (1.3, 1.5),  # rising load, watch closely
    "danger": (1.5, np.inf),  # high-risk zone
}

# Possible positions -> integer code (simple, stable ordinal encoding).
POSITIONS = ("goalkeeper", "defender", "midfielder", "forward")
POSITION_TO_CODE = {pos: i for i, pos in enumerate(POSITIONS)}


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
    return "danger"  # acwr >= 1.5


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
    df: pd.DataFrame, load_col: str = "training_load", window: int = 7
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
    explainable.
    """
    score = 0.0

    # 1) ACWR: the danger zone (>1.5) is the main risk factor.
    if acwr >= 1.5:
        score += 0.35
    elif acwr >= 1.3:
        score += 0.18
    elif acwr < 0.8:
        score += 0.08  # under-loading: moderate risk (detraining)

    # 2) High soreness (0-10 scale).
    score += np.clip((soreness - 4) / 6, 0, 1) * 0.20

    # 3) Lack of sleep (< 7 h = recognized risk factor).
    score += np.clip((7 - sleep_hours) / 4, 0, 1) * 0.15

    # 4) Resting heart rate elevated vs baseline (fatigue/stress).
    score += np.clip((resting_hr - baseline_hr) / 20, 0, 1) * 0.12

    # 5) History: proneness + number of past injuries.
    if injury_prone:
        score += 0.10
    score += np.clip(previous_injuries / 5, 0, 1) * 0.08

    # 6) Recent return from injury (< 60 days): tissue still fragile.
    if days_since_injury < 60:
        score += 0.10 * (1 - days_since_injury / 60)

    return float(np.clip(score, 0.0, 1.0))


def risk_score_to_level(score: float, low_thr: float = 0.16, high_thr: float = 0.27) -> int:
    """Convert a continuous score into a risk level: 0=low, 1=moderate, 2=high."""
    if score >= high_thr:
        return 2
    if score >= low_thr:
        return 1
    return 0


RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High"}


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
