"""Tests for the athlete-grouped cross-validation splitters.

The point of these tests is to lock down the fix for the athlete-level leakage:
no athlete may ever appear in both sides of a split on the synthetic track.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from injury_risk.models.splits import (
    grouped_train_test_split,
    make_cv,
    make_groups,
    needs_grouping,
)


def _repeated_measures_df(n_athletes: int = 20, n_days: int = 10) -> pd.DataFrame:
    """Several daily rows per athlete, with both classes represented."""
    rng = np.random.default_rng(0)
    rows = [
        {
            "athlete_id": athlete,
            "feature": float(rng.normal()),
            "risk_level": int((athlete + day) % 2),
        }
        for athlete in range(n_athletes)
        for day in range(n_days)
    ]
    return pd.DataFrame(rows)


def _xy(df: pd.DataFrame):
    return df[["feature"]], df["risk_level"]


# --------------------------------------------------------------------------- #
# splitter selection
# --------------------------------------------------------------------------- #


def test_synthetic_track_is_grouped_real_is_not():
    assert needs_grouping("synthetic")
    assert not needs_grouping("real")
    assert isinstance(make_cv("synthetic"), StratifiedGroupKFold)
    assert isinstance(make_cv("real"), StratifiedKFold)


def test_make_groups_returns_athletes_for_synthetic_and_none_for_real():
    df = _repeated_measures_df()
    groups = make_groups("synthetic", df)
    assert groups is not None
    assert set(groups) == set(df["athlete_id"])
    assert make_groups("real", df) is None


def test_make_groups_requires_the_athlete_column():
    with pytest.raises(ValueError, match="athlete_id"):
        make_groups("synthetic", pd.DataFrame({"feature": [1.0]}))


# --------------------------------------------------------------------------- #
# the leakage guarantee
# --------------------------------------------------------------------------- #


def test_no_athlete_appears_in_both_folds():
    """The core regression guard: grouped CV never splits an athlete."""
    df = _repeated_measures_df()
    X, y = _xy(df)
    groups = make_groups("synthetic", df)
    cv = make_cv("synthetic", seed=42)

    for train_idx, test_idx in cv.split(X, y, groups):
        overlap = set(groups[train_idx]) & set(groups[test_idx])
        assert not overlap, f"athletes leaked across the split: {sorted(overlap)}"


def test_plain_stratified_kfold_would_leak():
    """Documents the bug this module fixes: ungrouped CV *does* split athletes."""
    df = _repeated_measures_df()
    X, y = _xy(df)
    athletes = df["athlete_id"].to_numpy()

    leaked = False
    for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        if set(athletes[train_idx]) & set(athletes[test_idx]):
            leaked = True
            break
    assert leaked, "expected the ungrouped splitter to leak athletes"


def test_grouped_holdout_keeps_athletes_intact():
    df = _repeated_measures_df()
    X, y = _xy(df)
    groups = make_groups("synthetic", df)

    X_tr, X_te, y_tr, y_te = grouped_train_test_split(X, y, groups, seed=42)

    train_athletes = set(df.loc[X_tr.index, "athlete_id"])
    test_athletes = set(df.loc[X_te.index, "athlete_id"])
    assert not (train_athletes & test_athletes)
    assert len(X_tr) + len(X_te) == len(X)
    assert 0.1 < len(X_te) / len(X) < 0.35  # roughly a 20% hold-out


def test_grouped_holdout_falls_back_to_stratified_without_groups():
    df = _repeated_measures_df()
    X, y = _xy(df)
    X_tr, X_te, _, _ = grouped_train_test_split(X, y, groups=None, seed=42)
    assert len(X_tr) + len(X_te) == len(X)
