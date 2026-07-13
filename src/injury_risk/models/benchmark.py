"""Benchmark of baseline models (Week 3 of the plan).

Compares three model families, **all preceded by SMOTE** and evaluated with the
same protocol (stratified 5-fold cross-validation, recall-oriented metrics):

- **Logistic Regression** — interpretable linear baseline;
- **Random Forest** — non-linear ensemble baseline;
- **XGBoost** — main model (gradient boosting).

The goal is to justify the choice of XGBoost with a quantified comparison rather
than by principle. Results are printed as a table and saved to
``reports/benchmark_{track}.json``.

Usage:
    python -m injury_risk.models.benchmark               # both tracks
    python -m injury_risk.models.benchmark --track real
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from injury_risk.models.train import (
    REPORTS_DIR,
    SCORING,
    _prepare_real,
    _prepare_synthetic,
)


def _candidate_models(n_classes: int, seed: int = 42) -> dict[str, ImbPipeline]:
    """Return the candidate pipelines (SMOTE + model)."""
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"

    models = {
        # Logistic regression needs scaling (sensitive to feature scale).
        "logistic_regression": ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=seed)),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight=None,  # SMOTE already handles the imbalance
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=seed)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "xgboost": ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=seed)),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective=objective,
                        eval_metric="mlogloss" if n_classes > 2 else "logloss",
                        tree_method="hist",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return models


def benchmark_track(track: str, seed: int = 42) -> pd.DataFrame:
    """Evaluate the 3 baselines on a track and return a comparison table."""
    if track == "synthetic":
        X, y, _ = _prepare_synthetic(sample_per_athlete=40, seed=seed)
    elif track == "real":
        X, y, _ = _prepare_real(seed=seed)
    else:
        raise ValueError(f"unknown track: {track!r}")

    n_classes = int(y.nunique())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    rows = []
    for name, pipe in _candidate_models(n_classes, seed).items():
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        rows.append(
            {
                "model": name,
                **{m: float(res[f"test_{m}"].mean()) for m in SCORING},
                **{f"{m}_std": float(res[f"test_{m}"].std()) for m in SCORING},
            }
        )

    table = pd.DataFrame(rows).set_index("model").round(4)
    table = table.sort_values("recall_macro", ascending=False)  # recall = priority

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"benchmark_{track}.json"
    out.write_text(json.dumps(table.reset_index().to_dict(orient="records"), indent=2))

    print(f"\n=== Benchmark '{track}' ({len(X)} rows, {n_classes} classes) ===")
    print(table[list(SCORING)].to_string())
    best = table.index[0]
    print(f"-> Best recall_macro: {best}")
    print(f"Saved: {out}")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline model benchmark.")
    parser.add_argument("--track", choices=["synthetic", "real", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tracks = ["synthetic", "real"] if args.track == "both" else [args.track]
    for track in tracks:
        benchmark_track(track, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
