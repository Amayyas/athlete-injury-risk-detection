"""Choosing the decision threshold from the cost of being wrong.

Every classifier defaults to 0.5, and that default carries a hidden claim: that a
missed injury and a false alarm cost the same. In this project they obviously do not
— the README has claimed a "recall-first" posture from the start while the code
happily predicted at 0.5.

Worse, 0.5 is meaningless on an imbalanced problem: with ~5% positives, a
well-calibrated model may *never* output 0.5, and the classifier then predicts "no
injury" for everyone — 95% accuracy, zero usefulness.

So the threshold is derived, not assumed:

    expected cost(t) = COST_FN * false_negatives(t) + COST_FP * false_positives(t)

and we take the ``t`` minimising it. That turns a vague intention ("recall matters
more") into a number anyone can audit and disagree with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from injury_risk.config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    THRESHOLD_SEARCH_POINTS,
)


@dataclass(frozen=True)
class OperatingPoint:
    """A decision threshold and what it costs."""

    threshold: float
    expected_cost: float
    recall: float
    precision: float
    false_positives: int
    false_negatives: int
    alert_rate: float  # share of athlete-days flagged — what the staff lives with

    def as_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "expected_cost": self.expected_cost,
            "recall": self.recall,
            "precision": self.precision,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "alert_rate": self.alert_rate,
        }


def confusion_at(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    """(tn, fp, fn, tp) at a given threshold."""
    predicted = y_proba >= threshold
    actual = y_true.astype(bool)
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    return tn, fp, fn, tp


def expected_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> float:
    """Total cost of the mistakes made at ``threshold``."""
    _, fp, fn, _ = confusion_at(y_true, y_proba, threshold)
    return cost_fn * fn + cost_fp * fp


def optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
    n_points: int = THRESHOLD_SEARCH_POINTS,
) -> OperatingPoint:
    """The threshold minimising the expected cost.

    ``y_proba`` must come from data the model was **not** fitted on (out-of-fold
    predictions), otherwise the threshold is tuned on memorised answers.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Search inside the range the model actually produces: on an imbalanced problem
    # a calibrated model may never reach 0.5, and a fixed 0..1 grid would waste most
    # of its points where nothing happens.
    lo, hi = float(y_proba.min()), float(y_proba.max())
    candidates = np.linspace(lo, hi, n_points)

    best: OperatingPoint | None = None
    for threshold in candidates:
        _, fp, fn, tp = confusion_at(y_true, y_proba, threshold)
        cost = cost_fn * fn + cost_fp * fp
        if best is not None and cost >= best.expected_cost:
            continue
        flagged = tp + fp
        best = OperatingPoint(
            threshold=float(threshold),
            expected_cost=float(cost),
            recall=tp / (tp + fn) if (tp + fn) else 0.0,
            precision=tp / flagged if flagged else 0.0,
            false_positives=fp,
            false_negatives=fn,
            alert_rate=flagged / len(y_true),
        )

    assert best is not None  # the grid is never empty
    return best
