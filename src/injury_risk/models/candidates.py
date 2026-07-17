"""The candidate models — defined once, used by train, tune and benchmark.

Each candidate is a full ``SMOTE + estimator`` pipeline, so that resampling always
happens *inside* the cross-validation fold rather than before it (resampling first
would leak synthetic neighbours of validation rows into training).

Note the ordering for logistic regression: **scaler → SMOTE → classifier**. SMOTE
interpolates between k-nearest neighbours using Euclidean distance, so on raw
features its synthetic samples are dominated by whichever variable happens to be
measured in the largest units.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from injury_risk.config import DEFAULT_SEED, XGB_DEFAULT_PARAMS

CANDIDATES = ("logistic_regression", "random_forest", "xgboost")

# The delivered model per track — decided by the *tuned* benchmark, not by reflex.
#
# synthetic: logistic regression (PR-AUC 0.367 vs 0.344 RF / 0.341 XGBoost). The
#   simulated hazard is logit-linear by construction, so the linear model is nearly
#   correctly specified; tuning narrowed the gap but did not close it, and it even
#   drove the other two towards simplicity (C=0.009, RF max_depth=3).
# real: random forest (PR-AUC 0.934 vs 0.919 XGBoost / 0.754 LogReg) — the ranking
#   flips on real data, which is the whole point of measuring rather than assuming.
DELIVERED_MODEL = {
    "synthetic": "logistic_regression",
    "real": "random_forest",
}
DEFAULT_MODEL = "xgboost"  # fallback for callers that do not name a track


def delivered_model(track: str) -> str:
    """The model that ships for a track, per the tuned benchmark."""
    return DELIVERED_MODEL.get(track, DEFAULT_MODEL)


def build_candidate(
    model: str,
    n_classes: int = 2,
    seed: int = DEFAULT_SEED,
    params: dict | None = None,
) -> ImbPipeline:
    """Build a candidate pipeline, optionally with tuned hyperparameters."""
    params = params or {}

    if model == "logistic_regression":
        defaults = {"C": 1.0, "l1_ratio": 0.0}
        return ImbPipeline(
            steps=[
                # Scale before resampling: SMOTE's neighbours are Euclidean.
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=seed)),
                (
                    "clf",
                    LogisticRegression(
                        **{**defaults, **params},
                        penalty="elasticnet",
                        solver="saga",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model == "random_forest":
        defaults = {"n_estimators": 300, "max_depth": 8}
        return ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=seed)),
                (
                    "clf",
                    RandomForestClassifier(
                        **{**defaults, **params},
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model == "xgboost":
        objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
        return ImbPipeline(
            steps=[
                # k_neighbors stays cautious in case a class is very rare.
                ("smote", SMOTE(random_state=seed, k_neighbors=5)),
                (
                    "clf",
                    XGBClassifier(
                        **{**XGB_DEFAULT_PARAMS, **params},
                        objective=objective,
                        eval_metric="mlogloss" if n_classes > 2 else "logloss",
                        tree_method="hist",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    raise ValueError(f"unknown model: {model!r} (expected one of {CANDIDATES})")
