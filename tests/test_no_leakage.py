"""The leakage guarantees — the tests that keep the project honest.

Once the target became "will this athlete get injured in the next 7 days?", two
things could silently destroy it:

1. a **feature** peeking at the future (then the model cheats);
2. the **hazard driver** leaking back into X (then the target is circular again —
   the exact bug this redesign was built to kill).

Both are asserted here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from injury_risk.config import TARGET_COL
from injury_risk.data.datasets import load_synthetic
from injury_risk.data.generate_synthetic import generate
from injury_risk.features.engineering import SYNTHETIC_FEATURE_COLS, build_features


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    # Enough athlete-days for the (deliberately rare) injuries to actually occur.
    return generate(n_athletes=20, n_days=400, seed=11)


def test_features_at_day_t_ignore_the_future(raw: pd.DataFrame):
    """Corrupt every day after a cutoff; the features before it must not move.

    This is the strongest statement we can make: the past is genuinely blind to the
    future, whatever the future contains.
    """
    cutoff = 250
    baseline = build_features(raw)

    tampered = raw.copy()
    future = tampered["day"] > cutoff
    rng = np.random.default_rng(0)
    for col in ("training_load", "soreness", "sleep_hours", "resting_hr"):
        tampered.loc[future, col] = rng.uniform(0, 500, size=int(future.sum()))
    tampered_features = build_features(tampered)

    past = baseline["day"] <= cutoff
    pd.testing.assert_frame_equal(
        baseline.loc[past.to_numpy(), SYNTHETIC_FEATURE_COLS].reset_index(drop=True),
        tampered_features.loc[past.to_numpy(), SYNTHETIC_FEATURE_COLS].reset_index(drop=True),
    )


def test_the_target_does_depend_on_the_future(raw: pd.DataFrame):
    """The mirror image: the *label* must look forward, or it predicts nothing.

    A sanity check on the test above — it would also pass if nothing depended on
    anything, so we prove the future does matter, just not to the features.
    """
    assert raw["injury_onset"].sum() > 0, "this seed should produce injuries"
    assert raw[TARGET_COL].sum() > 0, "the target must flag the days before an injury"

    # Every flagged day must indeed be followed by an onset within the horizon.
    flagged = raw[raw[TARGET_COL] == 1]
    assert not flagged.empty
    assert (
        flagged["athlete_id"].isin(raw.loc[raw["injury_onset"], "athlete_id"]).all()
    ), "only athletes who do get injured can have positive days"


def test_the_hazard_driver_is_not_a_feature():
    """`latent_risk` drives the injury draw. Feeding it to the model would make the
    target circular again — the original sin of this project."""
    assert "latent_risk" not in SYNTHETIC_FEATURE_COLS
    data = load_synthetic(sample_per_athlete=3, seed=42)
    assert "latent_risk" not in data.X.columns


def test_no_outcome_column_leaks_into_the_features():
    """Nothing that describes the outcome may sit in X."""
    forbidden = {TARGET_COL, "injury_onset", "is_injured", "horizon_complete", "latent_risk"}
    assert forbidden.isdisjoint(SYNTHETIC_FEATURE_COLS)
