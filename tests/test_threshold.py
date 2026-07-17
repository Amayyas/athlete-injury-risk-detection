"""Tests for the cost-based decision threshold.

What matters here is not that the arithmetic is right (it is trivial), but that the
*business posture* the README claims is actually encoded: a missed injury must cost
more than a false alarm, and the chosen threshold must reflect that.
"""

from __future__ import annotations

import numpy as np
import pytest

from injury_risk.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE
from injury_risk.models.threshold import (
    confusion_at,
    expected_cost,
    optimal_threshold,
)


def _separable(n: int = 400, prevalence: float = 0.1, seed: int = 0):
    """Probabilities that rank positives above negatives, with some overlap."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    proba = np.clip(rng.normal(np.where(y == 1, 0.65, 0.35), 0.15), 0.01, 0.99)
    return y, proba


# --------------------------------------------------------------------------- #
# Confusion & cost
# --------------------------------------------------------------------------- #


def test_confusion_counts_add_up():
    y, proba = _separable()
    tn, fp, fn, tp = confusion_at(y, proba, 0.5)
    assert tn + fp + fn + tp == len(y)
    assert tp + fn == int(y.sum())


def test_threshold_zero_flags_everyone():
    y, proba = _separable()
    tn, fp, fn, tp = confusion_at(y, proba, 0.0)
    assert fn == 0, "flagging everyone cannot miss an injury"
    assert tn == 0


def test_threshold_above_one_flags_nobody():
    y, proba = _separable()
    _, fp, fn, tp = confusion_at(y, proba, 1.01)
    assert tp == 0 and fp == 0
    assert fn == int(y.sum()), "flagging nobody misses every injury"


def test_expected_cost_weights_misses_more_than_false_alarms():
    """The asymmetry is the whole point of the module."""
    y = np.array([1, 0])
    one_miss = np.array([0.0, 0.0])  # misses the positive, flags nothing
    one_false_alarm = np.array([1.0, 1.0])  # catches the positive, flags the negative

    cost_miss = expected_cost(y, one_miss, 0.5)
    cost_alarm = expected_cost(y, one_false_alarm, 0.5)

    assert cost_miss == COST_FALSE_NEGATIVE
    assert cost_alarm == COST_FALSE_POSITIVE
    assert cost_miss > cost_alarm


# --------------------------------------------------------------------------- #
# The chosen operating point
# --------------------------------------------------------------------------- #


def test_optimal_threshold_minimises_the_cost():
    y, proba = _separable()
    point = optimal_threshold(y, proba)
    # No other threshold on a fine grid may do better.
    for candidate in np.linspace(proba.min(), proba.max(), 100):
        assert expected_cost(y, proba, candidate) >= point.expected_cost - 1e-9


def test_the_cost_based_point_beats_the_naive_half():
    """The claim is that 0.5 is an arbitrary default, and a derived point is better.

    "Better" means **cheaper**, not "higher recall": minimising cost can trade a
    little recall for a lot fewer false alarms, and that is the optimiser working as
    intended. Recall is bought only when its price is worth paying — which is exactly
    what makes the cost ratio, rather than a slogan, the thing that decides.
    """
    y, proba = _separable(prevalence=0.05, seed=3)
    point = optimal_threshold(y, proba)

    assert point.expected_cost <= expected_cost(y, proba, 0.5), "0.5 cannot be cheaper"


def test_a_higher_miss_cost_lowers_the_threshold():
    """The knob is a business statement: raise the cost of a miss, flag more."""
    y, proba = _separable(prevalence=0.08, seed=5)
    cheap = optimal_threshold(y, proba, cost_fn=2, cost_fp=1)
    dear = optimal_threshold(y, proba, cost_fn=50, cost_fp=1)
    assert dear.threshold <= cheap.threshold
    assert dear.recall >= cheap.recall


def test_symmetric_costs_do_not_favour_recall():
    y, proba = _separable(prevalence=0.5, seed=7)
    balanced = optimal_threshold(y, proba, cost_fn=1, cost_fp=1)
    assert 0.2 < balanced.threshold < 0.8


def test_operating_point_reports_what_the_staff_lives_with():
    y, proba = _separable()
    point = optimal_threshold(y, proba)
    assert 0.0 <= point.alert_rate <= 1.0
    assert 0.0 <= point.recall <= 1.0
    assert 0.0 <= point.precision <= 1.0
    assert set(point.as_dict()) == {
        "threshold",
        "expected_cost",
        "recall",
        "precision",
        "false_positives",
        "false_negatives",
        "alert_rate",
    }


def test_threshold_stays_inside_the_observed_range():
    """A calibrated model on a rare class may never output 0.5 — searching a fixed
    0..1 grid would waste its points where nothing happens."""
    y = np.array([0] * 95 + [1] * 5)
    proba = np.concatenate([np.full(95, 0.02), np.full(5, 0.08)])
    point = optimal_threshold(y, proba)
    assert proba.min() <= point.threshold <= proba.max()
    assert point.recall == pytest.approx(1.0)
