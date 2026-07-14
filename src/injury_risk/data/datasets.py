"""Public dataset API — the single way to obtain a modelling-ready track.

Before this module, ``benchmark``, ``tune`` and ``shap_plots`` all reached into
``train`` for its **private** helpers (``_prepare_synthetic`` / ``_prepare_real``):
every module was coupled to the internals of another, and the sub-sampling size
was duplicated at four call sites.

Everything now goes through :func:`load_track`, which returns a :class:`Dataset`
carrying the features, the target, the underlying frame and the athlete groups
needed for leak-free cross-validation.

Two tracks:

- ``synthetic`` — the generated daily time series (many rows per athlete, so it
  **requires grouped CV**). The target is an observed event: "will this athlete get
  injured within the next 7 days?";
- ``real`` — SIRP-600, a snapshot (one row = one athlete, so grouping is
  meaningless).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from injury_risk.config import (
    DEFAULT_SEED,
    SAMPLE_PER_ATHLETE,
    SYNTHETIC_DATASET,
    TARGET_COL,
    WARMUP_DAYS,
)
from injury_risk.data.generate_synthetic import generate
from injury_risk.data.load_dataset import SIRP_FEATURE_COLS, SIRP_TARGET, load_sirp600
from injury_risk.features.engineering import SYNTHETIC_FEATURE_COLS, build_features

TRACKS = ("synthetic", "real")

# The column identifying an athlete in the synthetic track.
GROUP_COL = "athlete_id"

# Tracks holding repeated measures of the same subject, which therefore require
# grouped splitting (cf. injury_risk.models.splits).
GROUPED_TRACKS = frozenset({"synthetic"})


@dataclass(frozen=True)
class Dataset:
    """A modelling-ready track."""

    track: str
    X: pd.DataFrame
    y: pd.Series
    frame: pd.DataFrame
    groups: np.ndarray | None

    @property
    def n_classes(self) -> int:
        return int(self.y.nunique())

    def __len__(self) -> int:
        return len(self.X)


def needs_grouping(track: str) -> bool:
    """Whether a track has several rows per athlete (and so needs grouped CV)."""
    return track in GROUPED_TRACKS


def make_groups(track: str, frame: pd.DataFrame) -> np.ndarray | None:
    """Return the group labels for a track, or ``None`` when grouping is moot."""
    if not needs_grouping(track):
        return None
    if GROUP_COL not in frame.columns:
        raise ValueError(f"track {track!r} requires a {GROUP_COL!r} column for grouped CV")
    return frame[GROUP_COL].to_numpy()


def load_synthetic(
    sample_per_athlete: int | None = SAMPLE_PER_ATHLETE,
    seed: int = DEFAULT_SEED,
) -> Dataset:
    """Load (or generate) the synthetic track and apply the feature engineering.

    We drop the warm-up period, where the chronic load — and therefore the ACWR —
    is not yet stabilised, then thin each athlete's series: consecutive days are
    highly autocorrelated, so feeding 730 near-duplicate rows per athlete adds
    little signal.

    Thinning reduces autocorrelation; it does **not** prevent athlete leakage.
    That is what the grouped cross-validation is for — hence the ``groups``.
    """
    if SYNTHETIC_DATASET.exists():
        frame = pd.read_parquet(SYNTHETIC_DATASET)
    else:
        frame = generate(seed=seed)

    frame = build_features(frame)

    # Rows we cannot legitimately model:
    #  - the warm-up, where the chronic load (and so the ACWR) is not yet stable;
    #  - days the athlete is already sidelined — they are not exposed to a *new*
    #    injury, so asking "will they get injured?" is meaningless;
    #  - the censored tail, whose 7-day horizon runs past the end of the simulation,
    #    so its label would be a guess.
    frame = frame[
        (frame["day"] >= WARMUP_DAYS) & (~frame["is_injured"]) & (frame["horizon_complete"])
    ].copy()

    if sample_per_athlete:
        sampled_idx = (
            frame.groupby(GROUP_COL, group_keys=False)
            .sample(frac=1.0, random_state=seed)  # shuffle within each athlete
            .groupby(GROUP_COL)
            .head(sample_per_athlete)
            .index
        )
        frame = frame.loc[sampled_idx].reset_index(drop=True)

    return Dataset(
        track="synthetic",
        X=frame[SYNTHETIC_FEATURE_COLS],
        y=frame[TARGET_COL],
        frame=frame,
        groups=make_groups("synthetic", frame),
    )


def load_real(seed: int = DEFAULT_SEED) -> Dataset:
    """Load the real SIRP-600 track (a snapshot: one row per athlete)."""
    frame = load_sirp600()
    return Dataset(
        track="real",
        X=frame[SIRP_FEATURE_COLS],
        y=frame[SIRP_TARGET],
        frame=frame,
        groups=None,
    )


def load_track(
    track: str,
    seed: int = DEFAULT_SEED,
    sample_per_athlete: int | None = SAMPLE_PER_ATHLETE,
) -> Dataset:
    """Load a track by name — the entry point every model module should use."""
    if track == "synthetic":
        return load_synthetic(sample_per_athlete=sample_per_athlete, seed=seed)
    if track == "real":
        return load_real(seed=seed)
    raise ValueError(f"unknown track: {track!r} (expected one of {TRACKS})")
