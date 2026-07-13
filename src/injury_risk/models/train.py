"""Training of the injury-risk models (XGBoost + SMOTE).

Two independent "tracks" share the same pipeline skeleton:

- ``synthetic``: generated temporal dataset (3 classes low/moderate/high), after
  full feature engineering (ACWR, rolling, trend);
- ``real``: real SIRP-600 dataset (binary target, naturally imbalanced).

Common pipeline:
    SMOTE (rebalancing) -> XGBoostClassifier
evaluated with 5-fold cross-validation **grouped by athlete** on the synthetic
track (cf. :mod:`injury_risk.models.splits`), with **recall**-oriented metrics (in
a medical context, missing an injury costs more than a false alarm):
f1_macro, recall_macro, roc_auc (weighted OVR).

Usage:
    python -m injury_risk.models.train               # train both tracks
    python -m injury_risk.models.train --track real  # a single track
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier

from injury_risk.config import (
    CV_N_SPLITS,
    DEFAULT_SEED,
    MODELS_DIR,
    REPORTS_DIR,
    SCORING,
    XGB_DEFAULT_PARAMS,
)
from injury_risk.data.datasets import TRACKS, Dataset, load_track
from injury_risk.models.splits import grouped_train_test_split, make_cv


def build_pipeline(
    n_classes: int,
    seed: int = DEFAULT_SEED,
    params: dict | None = None,
) -> ImbPipeline:
    """SMOTE + XGBoost pipeline adapted to the number of classes.

    ``params`` allows injecting hyperparameters from tuning
    (cf. :mod:`injury_risk.models.tune`). Otherwise the defaults from the config
    are used.
    """
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
    xgb_params = {**XGB_DEFAULT_PARAMS, **(params or {})}
    clf = XGBClassifier(
        **xgb_params,
        objective=objective,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    # Cautious k_neighbors in case a minority class is very rare.
    return ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=seed, k_neighbors=5)),
            ("clf", clf),
        ]
    )


def _evaluate_cv(pipe: ImbPipeline, data: Dataset, seed: int = DEFAULT_SEED) -> dict:
    """Cross-validation, returns the mean scores.

    On the synthetic track the folds are **grouped by athlete** so that no athlete
    appears in both the training and the validation fold.
    """
    cv = make_cv(data.track, seed=seed)
    results = cross_validate(
        pipe, data.X, data.y, groups=data.groups, cv=cv, scoring=SCORING, n_jobs=-1
    )
    return {
        metric: {
            "mean": float(np.mean(results[f"test_{metric}"])),
            "std": float(np.std(results[f"test_{metric}"])),
        }
        for metric in SCORING
    }


def train_track(track: str, seed: int = DEFAULT_SEED, tuned: bool = False) -> dict:
    """Train and evaluate a track, save the model + the metrics.

    If ``tuned`` is true and tuning has been performed (cf. src.models.tune), the
    best hyperparameters are loaded and used.
    """
    data = load_track(track, seed=seed)
    X, y = data.X, data.y

    print(
        f"\n=== Track '{track}': {len(data)} rows, {X.shape[1]} features, "
        f"{data.n_classes} classes ==="
    )
    print(f"Target distribution: {y.value_counts(normalize=True).sort_index().round(3).to_dict()}")

    params = None
    if tuned:
        from injury_risk.models.tune import load_best_params

        params = load_best_params(track)
        if params:
            print(f"Tuned hyperparameters loaded: {params}")
        else:
            print("No tuned params found -> defaults. Run injury_risk.models.tune.")
    pipe = build_pipeline(data.n_classes, seed, params=params)

    # 1) Cross-validation (honest performance estimate), grouped by athlete.
    cv_summary = _evaluate_cv(pipe, data, seed)
    grouped = "grouped by athlete" if data.groups is not None else "stratified"
    print(f"Cross-validation ({CV_N_SPLITS} folds, {grouped}):")
    for metric, stats in cv_summary.items():
        print(f"  {metric:14s} = {stats['mean']:.3f} ± {stats['std']:.3f}")

    # 2) Hold-out for a readable classification report — also grouped, otherwise
    #    the same athlete would sit on both sides of the split.
    X_tr, X_te, y_tr, y_te = grouped_train_test_split(X, y, data.groups, seed=seed)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    print("Hold-out report (recall per class):")
    for label in sorted(set(y_te)):
        rec = report[str(label)]["recall"]
        print(f"  class {label}: recall={rec:.3f}")

    # 3) Retrain on the whole dataset for the delivered model.
    pipe.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"model_{track}.joblib"
    joblib.dump(
        {"pipeline": pipe, "feature_cols": list(X.columns), "classes": pipe.classes_.tolist()},
        model_path,
    )

    metrics = {
        "track": track,
        "n_samples": int(len(data)),
        "n_features": int(X.shape[1]),
        "n_classes": data.n_classes,
        "class_distribution": y.value_counts(normalize=True).sort_index().round(4).to_dict(),
        "cv_strategy": type(make_cv(track, seed=seed)).__name__,
        "n_groups": int(len(np.unique(data.groups))) if data.groups is not None else None,
        "cross_validation": cv_summary,
        "holdout_report": report,
    }
    metrics_path = REPORTS_DIR / f"metrics_{track}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"Model -> {model_path}")
    print(f"Metrics -> {metrics_path}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the risk models.")
    parser.add_argument("--track", choices=[*TRACKS, "both"], default="both")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--tuned", action="store_true", help="use tuned hyperparameters if available"
    )
    args = parser.parse_args()

    tracks = list(TRACKS) if args.track == "both" else [args.track]
    for track in tracks:
        train_track(track, seed=args.seed, tuned=args.tuned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
