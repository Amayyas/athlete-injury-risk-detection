"""Training of the injury-risk models (XGBoost + SMOTE).

Two independent "tracks" share the same pipeline skeleton:

- ``synthetic``: generated temporal dataset (3 classes low/moderate/high), after
  full feature engineering (ACWR, rolling, trend);
- ``real``: real SIRP-600 dataset (binary target, naturally imbalanced).

Common pipeline:
    SMOTE (rebalancing) -> XGBoostClassifier
evaluated with stratified 5-fold cross-validation, with **recall**-oriented
metrics (in a medical context, missing an injury costs more than a false alarm):
f1_macro, recall_macro, roc_auc (weighted OVR).

Usage:
    python -m injury_risk.models.train               # train both tracks
    python -m injury_risk.models.train --track real  # a single track
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier

from injury_risk.data.generate_synthetic import DEFAULT_OUTPUT, generate
from injury_risk.data.load_dataset import SIRP_FEATURE_COLS, SIRP_TARGET, load_sirp600
from injury_risk.features.engineering import SYNTHETIC_FEATURE_COLS, build_features
from injury_risk.models.splits import grouped_train_test_split, make_cv, make_groups

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SCORING = {
    "f1_macro": "f1_macro",
    "recall_macro": "recall_macro",
    "roc_auc": "roc_auc_ovr_weighted",
}


def _build_pipeline(n_classes: int, seed: int = 42, params: dict | None = None) -> ImbPipeline:
    """SMOTE + XGBoost pipeline adapted to the number of classes.

    ``params`` allows injecting hyperparameters from tuning
    (cf. :mod:`injury_risk.models.tune`). Otherwise sensible defaults are used.
    """
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
    default_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }
    xgb_params = {**default_params, **(params or {})}
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


def _evaluate_cv(
    pipe: ImbPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    track: str,
    groups: np.ndarray | None,
    seed: int = 42,
) -> dict:
    """Stratified 5-fold cross-validation, returns the mean scores.

    On the synthetic track the folds are **grouped by athlete** so that no athlete
    appears in both the training and the validation fold (cf. :mod:`.splits`).
    """
    cv = make_cv(track, seed=seed)
    results = cross_validate(pipe, X, y, groups=groups, cv=cv, scoring=SCORING, n_jobs=-1)
    summary = {
        metric: {
            "mean": float(np.mean(results[f"test_{metric}"])),
            "std": float(np.std(results[f"test_{metric}"])),
        }
        for metric in SCORING
    }
    return summary


def _prepare_synthetic(sample_per_athlete: int | None = None, seed: int = 42):
    """Load/generate the synthetic dataset and apply the feature engineering.

    We subsample a few days per athlete to thin out the strong autocorrelation
    between consecutive days, and we drop the 28-day "warmup" period where the
    chronic load (and therefore the ACWR) is not yet stabilized.

    Note that sub-sampling does **not** protect against athlete-level leakage —
    the same athlete would still land in several folds. That is handled by the
    grouped cross-validation in :mod:`.splits`.
    """
    if DEFAULT_OUTPUT.exists():
        df = pd.read_parquet(DEFAULT_OUTPUT)
    else:
        df = generate(seed=seed)

    df = build_features(df)
    # Warmup: drop the first 28 days (chronic load not yet reliable).
    df = df[df["day"] >= 28].copy()

    if sample_per_athlete:
        # Per-athlete stratified subsampling, without groupby.apply.
        sampled_idx = (
            df.groupby("athlete_id", group_keys=False)
            .sample(frac=1.0, random_state=seed)  # intra-athlete shuffle
            .groupby("athlete_id")
            .head(sample_per_athlete)
            .index
        )
        df = df.loc[sampled_idx].reset_index(drop=True)

    X = df[SYNTHETIC_FEATURE_COLS]
    y = df["risk_level"]
    return X, y, df


def _prepare_real(seed: int = 42):
    df = load_sirp600()
    X = df[SIRP_FEATURE_COLS]
    y = df[SIRP_TARGET]
    return X, y, df


def train_track(track: str, seed: int = 42, tuned: bool = False) -> dict:
    """Train and evaluate a track, save the model + the metrics.

    If ``tuned`` is true and tuning has been performed (cf. injury_risk.models.tune), the
    best hyperparameters are loaded and used.
    """
    if track == "synthetic":
        X, y, df = _prepare_synthetic(sample_per_athlete=40, seed=seed)
    elif track == "real":
        X, y, df = _prepare_real(seed=seed)
    else:
        raise ValueError(f"unknown track: {track!r} (expected 'synthetic' or 'real')")

    groups = make_groups(track, df)
    n_classes = int(y.nunique())
    print(f"\n=== Track '{track}': {len(X)} rows, {X.shape[1]} features, {n_classes} classes ===")
    print(f"Target distribution: {y.value_counts(normalize=True).sort_index().round(3).to_dict()}")

    params = None
    if tuned:
        from injury_risk.models.tune import load_best_params

        params = load_best_params(track)
        if params:
            print(f"Tuned hyperparameters loaded: {params}")
        else:
            print("No tuned params found -> defaults. Run injury_risk.models.tune.")
    pipe = _build_pipeline(n_classes, seed, params=params)

    # 1) Cross-validation (honest performance estimate), grouped by athlete.
    cv_summary = _evaluate_cv(pipe, X, y, track, groups, seed)
    grouped = "grouped by athlete" if groups is not None else "stratified"
    print(f"Cross-validation (5 folds, {grouped}):")
    for metric, stats in cv_summary.items():
        print(f"  {metric:14s} = {stats['mean']:.3f} ± {stats['std']:.3f}")

    # 2) Hold-out for a readable classification report — also grouped, otherwise
    #    the same athlete would sit on both sides of the split.
    X_tr, X_te, y_tr, y_te = grouped_train_test_split(X, y, groups, seed=seed)
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
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
        "class_distribution": y.value_counts(normalize=True).sort_index().round(4).to_dict(),
        "cv_strategy": type(make_cv(track, seed=seed)).__name__,
        "n_groups": int(len(np.unique(groups))) if groups is not None else None,
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
    parser.add_argument("--track", choices=["synthetic", "real", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tuned", action="store_true", help="use tuned hyperparameters if available"
    )
    args = parser.parse_args()

    tracks = ["synthetic", "real"] if args.track == "both" else [args.track]
    for track in tracks:
        train_track(track, seed=args.seed, tuned=args.tuned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
