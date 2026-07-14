"""Reference points a model's score must be read against.

A PR-AUC of 0.28 means nothing on its own. It is excellent when the positive class
is 5% of the data, and catastrophic when it is 50%. Two reference points make the
number interpretable:

- **chance** — what a coin flip achieves. For PR-AUC that is simply the prevalence;
  for ROC-AUC it is 0.5.
- **the rule score** — what the *domain rules alone* achieve: ranking athletes by
  today's ``composite_risk_score``, the very quantity that drives the injury hazard.
  This is the baseline a machine-learning model has to justify itself against. If
  XGBoost cannot beat a hand-written formula, the formula wins — it is free, exact
  and explainable.

A note on why the rule score is **not** a ceiling
------------------------------------------------
It is tempting to call it an oracle, since the simulation draws injuries from it.
It is not. The target is "an injury *within the next 7 days*", and today's risk is
only the first of the seven daily hazards that decide it. A model that anticipates
where an athlete's risk is *heading* — from their proneness, their chronic load,
their history — can legitimately beat today's snapshot. Calling it a ceiling would
be plain wrong, and any model exceeding it would look like a bug rather than the
expected behaviour that it is.
"""

from __future__ import annotations

from sklearn.metrics import average_precision_score, roc_auc_score

from injury_risk.data.datasets import Dataset

# The rule-based composite score, recorded by the generator as the hazard driver.
RULE_SCORE_COL = "latent_risk"


def reference_points(data: Dataset) -> dict:
    """Chance and (when available) rule-score reference points for a track."""
    y = data.y.to_numpy()

    refs: dict[str, dict[str, float]] = {
        # The PR-AUC of a random ranker is the prevalence itself.
        "chance": {"average_precision": float(y.mean()), "roc_auc": 0.5},
    }

    if RULE_SCORE_COL in data.frame.columns:
        rule = data.frame[RULE_SCORE_COL].to_numpy()
        refs["rule_score"] = {
            "average_precision": float(average_precision_score(y, rule)),
            "roc_auc": float(roc_auc_score(y, rule)),
        }

    return refs


def lift_over(model_score: float, reference_score: float) -> float:
    """Relative improvement of the model over a reference (e.g. +18%)."""
    if reference_score <= 0:
        return 0.0
    return model_score / reference_score - 1.0
