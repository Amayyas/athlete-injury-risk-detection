"""Tests for the rule-based scoring and its factor decomposition.

The central guarantee locked here: **the factor list is the score's decomposition**,
so the dashboard can never show a raised gauge with no factor explaining it — the
bug this module was extracted to kill.
"""

from __future__ import annotations

import numpy as np
import pytest

from injury_risk.config import (
    RISK_HIGH_THRESHOLD,
    RISK_LOW_THRESHOLD,
    SORENESS_ONSET,
    W_ACWR_DANGER,
)
from injury_risk.features.risk_factors import (
    all_factors,
    assess,
    composite_risk_score,
    risk_factors,
    risk_score_to_level,
)

HEALTHY = dict(acwr=1.0, soreness=2.0, sleep_hours=8.0, resting_hr=52.0, baseline_hr=55.0)


def _random_inputs(rng: np.random.Generator) -> dict:
    return dict(
        acwr=float(rng.uniform(0.3, 2.5)),
        soreness=float(rng.uniform(0, 10)),
        sleep_hours=float(rng.uniform(3, 10)),
        resting_hr=float(rng.uniform(40, 95)),
        baseline_hr=float(rng.uniform(42, 70)),
        injury_prone=bool(rng.random() < 0.3),
        previous_injuries=int(rng.integers(0, 8)),
        days_since_injury=float(rng.integers(0, 400)),
    )


# --------------------------------------------------------------------------- #
# The consistency guarantee (the reason this module exists)
# --------------------------------------------------------------------------- #


def test_factors_sum_to_the_score():
    """The score IS the sum of the displayed factors — on 2000 random athletes."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        kwargs = _random_inputs(rng)
        score = composite_risk_score(**kwargs)
        total = sum(f.contribution for f in risk_factors(**kwargs))
        assert total == pytest.approx(min(score, 1.0), abs=1e-9) or score == 1.0


def test_no_hidden_contribution():
    """A non-zero contribution can never be missing from the displayed list.

    This is exactly the bug that existed: soreness raised the score from 4/10 but
    was only displayed from 6/10, so the gauge could read 14% next to
    "no risk factor detected".
    """
    rng = np.random.default_rng(1)
    for _ in range(2000):
        kwargs = _random_inputs(rng)
        contributing = {f.code for f in all_factors(**kwargs) if f.contribution > 0}
        displayed = {f.code for f in risk_factors(**kwargs)}
        assert contributing == displayed


def test_the_old_bug_is_gone():
    """The precise case that used to show a raised gauge with an empty factor list."""
    kwargs = dict(
        acwr=1.0,
        soreness=5.5,
        sleep_hours=7.5,
        resting_hr=61.0,
        baseline_hr=55.0,
        previous_injuries=2,
    )
    score = composite_risk_score(**kwargs)
    factors = risk_factors(**kwargs)
    assert score > 0
    assert factors, "a positive score must be explained by at least one factor"
    assert {"soreness", "resting_hr", "injury_history"} <= {f.code for f in factors}


def test_zero_score_means_no_factors():
    assert composite_risk_score(**HEALTHY) == 0.0
    assert risk_factors(**HEALTHY) == []


# --------------------------------------------------------------------------- #
# Ordering, bounds and severity
# --------------------------------------------------------------------------- #


def test_factors_are_sorted_by_contribution():
    factors = risk_factors(
        acwr=1.8,
        soreness=9.0,
        sleep_hours=4.0,
        resting_hr=80.0,
        baseline_hr=55.0,
        injury_prone=True,
        previous_injuries=4,
        days_since_injury=10,
    )
    contributions = [f.contribution for f in factors]
    assert contributions == sorted(contributions, reverse=True)
    # The ACWR danger zone is the heaviest single rule in the model.
    assert factors[0].code == "acwr_danger"
    assert factors[0].contribution == W_ACWR_DANGER


def test_score_stays_bounded():
    rng = np.random.default_rng(2)
    for _ in range(1000):
        assert 0.0 <= composite_risk_score(**_random_inputs(rng)) <= 1.0


def test_acwr_optimal_contributes_nothing():
    optimal = [f for f in all_factors(**HEALTHY) if f.code.startswith("acwr")]
    assert len(optimal) == 1
    assert optimal[0].contribution == 0.0


def test_soreness_counts_from_the_configured_onset():
    below = composite_risk_score(**{**HEALTHY, "soreness": SORENESS_ONSET})
    above = composite_risk_score(**{**HEALTHY, "soreness": SORENESS_ONSET + 1})
    assert below == 0.0
    assert above > 0.0


def test_percent_is_the_contribution_in_points():
    factors = risk_factors(**{**HEALTHY, "acwr": 2.0})
    assert factors[0].percent == round(W_ACWR_DANGER * 100)


# --------------------------------------------------------------------------- #
# Risk levels
# --------------------------------------------------------------------------- #


def test_risk_score_to_level_thresholds():
    assert risk_score_to_level(RISK_LOW_THRESHOLD - 0.01) == 0
    assert risk_score_to_level(RISK_LOW_THRESHOLD) == 1
    assert risk_score_to_level(RISK_HIGH_THRESHOLD - 0.01) == 1
    assert risk_score_to_level(RISK_HIGH_THRESHOLD) == 2
    # Thresholds stay overridable.
    assert risk_score_to_level(0.4, low_thr=0.33, high_thr=0.55) == 1


def test_assess_agrees_with_the_individual_functions():
    """`assess` is what consumers call; it must not drift from the parts."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        kwargs = _random_inputs(rng)
        a = assess(**kwargs)
        assert a.score == composite_risk_score(**kwargs)
        assert a.level == risk_score_to_level(a.score)
        assert a.factors == risk_factors(**kwargs)


def test_danger_profile_scores_higher_than_safe_one():
    safe = composite_risk_score(**HEALTHY)
    risky = composite_risk_score(
        acwr=1.8,
        soreness=9.0,
        sleep_hours=4.0,
        resting_hr=80.0,
        baseline_hr=55.0,
        injury_prone=True,
        previous_injuries=4,
        days_since_injury=10,
    )
    assert risky > safe
    assert risk_score_to_level(risky) == 2
