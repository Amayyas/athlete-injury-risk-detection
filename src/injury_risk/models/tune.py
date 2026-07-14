"""XGBoost hyperparameter tuning (Week 4 of the plan).

Random search (``RandomizedSearchCV``) of the ``SMOTE + XGBoost`` pipeline
hyperparameters, **optimized on macro recall** — consistent with the business
priority: in a medical context, missing an injury (false negative) costs more
than a false alarm.

The best parameters are saved to ``models/best_params_{track}.json`` and
automatically reused by ``injury_risk.models.train`` when the ``--tuned`` option is passed.

"""

from __future__ import annotations

import json
from pathlib import Path

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from injury_risk.config import DEFAULT_SEED, MODELS_DIR
from injury_risk.data.datasets import load_track
from injury_risk.models.splits import make_cv

# Search space (prefixed with ``clf__`` to target the pipeline step).
PARAM_DISTRIBUTIONS = {
    "clf__n_estimators": randint(150, 500),
    "clf__max_depth": randint(3, 8),
    "clf__learning_rate": uniform(0.01, 0.2),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.6, 0.4),
    "clf__min_child_weight": randint(1, 8),
    "clf__gamma": uniform(0.0, 0.5),
}


def _base_pipeline(n_classes: int, seed: int) -> ImbPipeline:
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
    return ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=seed)),
            (
                "clf",
                XGBClassifier(
                    objective=objective,
                    eval_metric="mlogloss" if n_classes > 2 else "logloss",
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def best_params_path(track: str) -> Path:
    return MODELS_DIR / f"best_params_{track}.json"


def tune_track(track: str, n_iter: int = 30, seed: int = DEFAULT_SEED) -> dict:
    """Run the random search and save the best parameters."""
    data = load_track(track, seed=seed)
    X, y = data.X, data.y

    # Grouped CV here too: tuning against leaky scores would select the
    # hyperparameters that memorise athletes best.
    cv = make_cv(track, seed=seed)

    search = RandomizedSearchCV(
        estimator=_base_pipeline(data.n_classes, seed),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="recall_macro",  # business priority
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=1,
    )
    print(f"\n=== Tuning '{track}': {n_iter} configurations, scoring=recall_macro ===")
    search.fit(X, y, groups=data.groups)

    # Keep only the classifier hyperparameters (without the clf__ prefix).
    best = {
        k.replace("clf__", ""): v for k, v in search.best_params_.items() if k.startswith("clf__")
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = best_params_path(track)
    out.write_text(json.dumps(best, indent=2, default=str))

    print(f"Best recall_macro (CV): {search.best_score_:.4f}")
    print(f"Best parameters: {best}")
    print(f"Saved: {out}")
    return best


def load_best_params(track: str) -> dict | None:
    """Load the best parameters if tuning has already been performed."""
    path = best_params_path(track)
    if not path.exists():
        return None
    params = json.loads(path.read_text())
    # Recast types (JSON does not distinguish int/float).
    for key in ("n_estimators", "max_depth", "min_child_weight"):
        if key in params:
            params[key] = int(round(float(params[key])))
    return params
