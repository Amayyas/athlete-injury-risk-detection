"""Hyperparameter search for **every** candidate, not just the favourite.

Why every candidate
-------------------
This module used to tune XGBoost alone, then the benchmark compared a *tuned*
XGBoost against *untuned* baselines. That comparison cannot lose: it measures which
model received attention, not which model is better. Once the untuned benchmark
showed logistic regression winning on the synthetic track, settling the question
honestly required tuning all three under the same budget and the same protocol.

What it optimises
-----------------
``average_precision`` (PR-AUC), not recall. Recall alone is trivially maximised by
flagging everybody; PR-AUC scores the whole probability ranking, which is what we
then turn into a decision with a cost-based threshold
(:mod:`injury_risk.models.threshold`).

Searches are grouped by athlete, like every other evaluation in the project — tuning
against leaky folds would select whichever hyperparameters memorise athletes best.
"""

from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV

from injury_risk.config import DEFAULT_SEED, MODELS_DIR, TUNING_METRIC, TUNING_N_ITER
from injury_risk.data.datasets import load_track
from injury_risk.models.candidates import CANDIDATES, build_candidate
from injury_risk.models.splits import make_cv

# Search spaces, prefixed with ``clf__`` to target the classifier step of each
# pipeline. Deliberately modest: the point is a fair comparison under one budget,
# not squeezing the last decimal out of one model.
PARAM_DISTRIBUTIONS: dict[str, dict] = {
    "xgboost": {
        "clf__n_estimators": randint(150, 600),
        "clf__max_depth": randint(2, 8),
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__min_child_weight": randint(1, 12),
        "clf__gamma": uniform(0.0, 0.5),
        "clf__reg_lambda": loguniform(0.1, 20.0),
    },
    "random_forest": {
        "clf__n_estimators": randint(200, 800),
        "clf__max_depth": randint(3, 20),
        "clf__min_samples_leaf": randint(1, 40),
        "clf__min_samples_split": randint(2, 30),
        "clf__max_features": uniform(0.2, 0.8),
    },
    "logistic_regression": {
        "clf__C": loguniform(1e-3, 1e3),
        "clf__l1_ratio": uniform(0.0, 1.0),
    },
}


def best_params_path(track: str, model: str) -> Path:
    return MODELS_DIR / f"best_params_{track}_{model}.json"


def tune_candidate(
    track: str,
    model: str,
    n_iter: int = TUNING_N_ITER,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Search one candidate's hyperparameters on one track, and save the best."""
    data = load_track(track, seed=seed)

    search = RandomizedSearchCV(
        estimator=build_candidate(model, data.n_classes, seed),
        param_distributions=PARAM_DISTRIBUTIONS[model],
        n_iter=n_iter,
        scoring=TUNING_METRIC,
        cv=make_cv(track, seed=seed),
        random_state=seed,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(data.X, data.y, groups=data.groups)

    best = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = best_params_path(track, model)
    out.write_text(json.dumps(best, indent=2, default=str))

    print(f"  {model:<20} {TUNING_METRIC} = {search.best_score_:.4f}  -> {out.name}")
    return {"model": model, "score": float(search.best_score_), "params": best}


def tune_track(track: str, n_iter: int = TUNING_N_ITER, seed: int = DEFAULT_SEED) -> dict:
    """Tune every candidate on a track, under the same budget."""
    print(f"\n=== Tuning '{track}': {n_iter} configs per model, scoring={TUNING_METRIC} ===")
    results = {name: tune_candidate(track, name, n_iter, seed) for name in CANDIDATES}
    winner = max(results.values(), key=lambda r: r["score"])
    print(f"  -> best: {winner['model']} ({winner['score']:.4f})")
    return results


def load_best_params(track: str, model: str) -> dict | None:
    """Load a candidate's tuned parameters, if the search has been run."""
    path = best_params_path(track, model)
    if not path.exists():
        return None
    params = json.loads(path.read_text())
    # JSON does not distinguish int from float; the estimators do.
    for key in (
        "n_estimators",
        "max_depth",
        "min_child_weight",
        "min_samples_leaf",
        "min_samples_split",
    ):
        if key in params:
            params[key] = int(round(float(params[key])))
    return params
