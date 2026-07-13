"""Cross-validation splitters that respect athlete grouping.

Why this module exists
----------------------
The synthetic dataset holds **many daily rows per athlete**. Splitting those rows
randomly (plain ``StratifiedKFold``) puts the *same athlete* in both the training
and the validation fold: the model can memorise an athlete's baseline profile
(their resting HR, their proneness, their typical load) and recognise them in the
test fold. That is **group leakage**, and it inflates every score.

Sub-sampling a few days per athlete — as the loaders do — reduces the
autocorrelation between consecutive days, but it does **not** fix this: the
athlete is still present on both sides of the split.

The fix is to group by athlete, so an athlete lands entirely in the training set
or entirely in the validation set, never both.

Which tracks need grouping is a property of the *data*, so it is defined in
:mod:`injury_risk.data.datasets`; this module only turns it into splitters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedGroupKFold, StratifiedKFold

from injury_risk.config import CV_N_SPLITS, DEFAULT_SEED
from injury_risk.data.datasets import needs_grouping


def make_cv(
    track: str,
    seed: int = DEFAULT_SEED,
    n_splits: int = CV_N_SPLITS,
) -> BaseCrossValidator:
    """Return the correct cross-validator for a track.

    - ``synthetic`` -> ``StratifiedGroupKFold`` (no athlete spans two folds);
    - ``real``      -> ``StratifiedKFold`` (snapshot: one row per athlete).
    """
    if needs_grouping(track):
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def grouped_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray | None,
    seed: int = DEFAULT_SEED,
    n_splits: int = CV_N_SPLITS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """A single stratified hold-out split that also keeps athletes intact.

    Taking the first fold of a 5-fold splitter yields a ~20% test set while
    preserving both stratification and grouping — which a plain
    ``train_test_split(stratify=y)`` cannot do.
    """
    track = "synthetic" if groups is not None else "real"
    splitter = make_cv(track, seed=seed, n_splits=n_splits)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
