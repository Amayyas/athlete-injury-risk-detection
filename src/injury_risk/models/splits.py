"""Cross-validation splitters that respect athlete grouping.

Why this module exists
----------------------
The synthetic dataset holds **many daily rows per athlete**. Splitting those rows
randomly (plain ``StratifiedKFold``) puts the *same athlete* in both the training
and the validation fold: the model can memorise an athlete's baseline profile
(their resting HR, their proneness, their typical load) and recognise them in
the test fold. That is **group leakage**, and it inflates every score.

Sub-sampling a few days per athlete — as the pipeline does — reduces the
autocorrelation between consecutive days, but it does **not** fix this: the
athlete is still present on both sides of the split.

The fix is to group by ``athlete_id`` so that an athlete lands entirely in the
training set or entirely in the validation set, never both.

The real SIRP-600 track is a **snapshot** (one row = one athlete), so grouping is
meaningless there and plain stratification stays correct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedGroupKFold, StratifiedKFold

# The column identifying an athlete in the synthetic track.
GROUP_COL = "athlete_id"

# Tracks whose rows are repeated measures of the same subject, and therefore
# require grouped splitting.
GROUPED_TRACKS = frozenset({"synthetic"})


def needs_grouping(track: str) -> bool:
    """Whether a track has several rows per athlete (and so needs grouped CV)."""
    return track in GROUPED_TRACKS


def make_cv(track: str, seed: int = 42, n_splits: int = 5) -> BaseCrossValidator:
    """Return the correct cross-validator for a track.

    - ``synthetic`` -> ``StratifiedGroupKFold`` (no athlete spans two folds);
    - ``real``      -> ``StratifiedKFold`` (snapshot: one row per athlete).
    """
    if needs_grouping(track):
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_groups(track: str, df: pd.DataFrame) -> np.ndarray | None:
    """Return the group labels to pass to the splitter, or ``None`` if not needed."""
    if not needs_grouping(track):
        return None
    if GROUP_COL not in df.columns:
        raise ValueError(f"track {track!r} requires a {GROUP_COL!r} column for grouped CV")
    return df[GROUP_COL].to_numpy()


def grouped_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray | None,
    seed: int = 42,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """A single stratified hold-out split that also keeps athletes intact.

    Taking the first fold of a 5-fold splitter yields a ~20% test set while
    preserving both stratification and grouping — which a plain
    ``train_test_split(stratify=y)`` cannot do.
    """
    splitter = make_cv("synthetic" if groups is not None else "real", seed=seed, n_splits=n_splits)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
