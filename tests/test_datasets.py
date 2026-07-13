"""Tests for the public dataset API.

These lock the contract that every model module now depends on: `load_track`
returns a coherent `Dataset`, and it carries the athlete groups that the grouped
cross-validation needs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from injury_risk.config import SAMPLE_PER_ATHLETE, WARMUP_DAYS
from injury_risk.data.datasets import (
    GROUP_COL,
    TRACKS,
    Dataset,
    load_synthetic,
    load_track,
    make_groups,
    needs_grouping,
)
from injury_risk.features.engineering import SYNTHETIC_FEATURE_COLS


def test_unknown_track_is_rejected():
    with pytest.raises(ValueError, match="unknown track"):
        load_track("nonexistent")


def test_tracks_are_the_two_expected_ones():
    assert set(TRACKS) == {"synthetic", "real"}


def test_synthetic_track_needs_grouping_real_does_not():
    assert needs_grouping("synthetic")
    assert not needs_grouping("real")


def test_make_groups_requires_the_athlete_column():
    with pytest.raises(ValueError, match=GROUP_COL):
        make_groups("synthetic", pd.DataFrame({"feature": [1.0]}))


# --------------------------------------------------------------------------- #
# The synthetic track (generated on the fly, so it always runs in CI)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def synthetic() -> Dataset:
    # A small dataset keeps the test fast; the contract is what matters.
    return load_synthetic(sample_per_athlete=5, seed=42)


def test_synthetic_dataset_is_coherent(synthetic: Dataset):
    assert synthetic.track == "synthetic"
    assert list(synthetic.X.columns) == SYNTHETIC_FEATURE_COLS
    assert len(synthetic.X) == len(synthetic.y) == len(synthetic)
    assert synthetic.n_classes == synthetic.y.nunique()


def test_synthetic_dataset_carries_athlete_groups(synthetic: Dataset):
    """The whole point of the API: the groups travel with the data."""
    assert synthetic.groups is not None
    assert len(synthetic.groups) == len(synthetic)
    assert set(synthetic.groups) == set(synthetic.frame[GROUP_COL])


def test_warmup_period_is_dropped(synthetic: Dataset):
    """The ACWR is not stable before a full chronic window has elapsed."""
    assert synthetic.frame["day"].min() >= WARMUP_DAYS


def test_sampling_caps_the_rows_per_athlete(synthetic: Dataset):
    per_athlete = synthetic.frame.groupby(GROUP_COL).size()
    assert per_athlete.max() <= 5


def test_sample_per_athlete_defaults_to_the_config_value():
    data = load_synthetic(seed=42)
    assert data.frame.groupby(GROUP_COL).size().max() <= SAMPLE_PER_ATHLETE


def test_loading_is_deterministic_for_a_given_seed():
    a = load_synthetic(sample_per_athlete=5, seed=7)
    b = load_synthetic(sample_per_athlete=5, seed=7)
    pd.testing.assert_frame_equal(a.X, b.X)
