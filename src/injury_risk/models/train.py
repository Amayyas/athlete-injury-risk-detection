"""Train, calibrate, and pick an operating point.

Pipeline per track:

    SMOTE -> estimator          (fitted inside every CV fold, never before)
    -> calibration              (isotonic, on out-of-fold predictions)
    -> cost-based threshold     (FN costs COST_FALSE_NEGATIVE times an FP)

Everything is evaluated with 5-fold cross-validation **grouped by athlete**, and the
delivered bundle carries its threshold: a model that ships without its operating
point silently reverts to 0.5, which on a ~5% positive class means predicting "no
injury" for everyone.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_validate

from injury_risk.config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    CV_N_SPLITS,
    DEFAULT_SEED,
    MODELS_DIR,
    REPORTS_DIR,
    SCORING,
)
from injury_risk.data.datasets import Dataset, load_track
from injury_risk.models.baselines import lift_over, reference_points
from injury_risk.models.candidates import build_candidate, delivered_model
from injury_risk.models.evaluate import (
    out_of_fold_calibrated_proba,
    out_of_fold_proba,
    plot_calibration,
    plot_confusion,
    plot_precision_recall,
    summarise,
)
from injury_risk.models.splits import make_cv, materialise_folds
from injury_risk.models.threshold import optimal_threshold
from injury_risk.models.tune import load_best_params


def build_pipeline(n_classes: int, seed: int = DEFAULT_SEED, params: dict | None = None):
    """A model pipeline for callers that just want *a* model (tests, smoke checks)."""
    from injury_risk.models.candidates import DEFAULT_MODEL

    return build_candidate(DEFAULT_MODEL, n_classes, seed, params=params)


def _calibrated(pipe, data: Dataset, seed: int = DEFAULT_SEED) -> CalibratedClassifierCV:
    """Wrap a pipeline in isotonic calibration, keeping the athlete grouping.

    The folds are materialised rather than passed as a splitter: CalibratedClassifierCV
    does not forward ``groups``, so a StratifiedGroupKFold would silently lose the
    grouping inside it (cf. injury_risk.models.splits.materialise_folds).
    """
    folds = materialise_folds(data.X, data.y, data.groups, data.track, seed=seed)
    return CalibratedClassifierCV(pipe, method="isotonic", cv=folds)


def _evaluate_cv(pipe, data: Dataset, seed: int = DEFAULT_SEED) -> dict:
    """Grouped cross-validation, returning mean and std per metric."""
    results = cross_validate(
        pipe,
        data.X,
        data.y,
        groups=data.groups,
        cv=make_cv(data.track, seed=seed),
        scoring=SCORING,
        n_jobs=-1,
    )
    return {
        metric: {
            "mean": float(np.mean(results[f"test_{metric}"])),
            "std": float(np.std(results[f"test_{metric}"])),
        }
        for metric in SCORING
    }


def train_track(
    track: str,
    seed: int = DEFAULT_SEED,
    tuned: bool = False,
    model: str | None = None,
    calibrate: bool = True,
) -> dict:
    """Train a model on a track: evaluate, calibrate, threshold, save.

    ``model`` defaults to whichever candidate won the *tuned* benchmark on that track.
    """
    model = model or delivered_model(track)
    data = load_track(track, seed=seed)

    params = load_best_params(track, model) if tuned else None
    print(f"\n=== Track '{track}' — {model} ({'tuned' if params else 'default params'}) ===")
    print(f"{len(data)} rows, {data.X.shape[1]} features, positive rate {data.y.mean():.2%}")

    pipe = build_candidate(model, data.n_classes, seed, params=params)

    # 1) Honest performance: grouped CV, so no athlete spans a fold.
    cv_summary = _evaluate_cv(pipe, data, seed)
    grouped = "grouped by athlete" if data.groups is not None else "stratified"
    print(f"Cross-validation ({CV_N_SPLITS} folds, {grouped}):")
    for metric, stats in cv_summary.items():
        print(f"  {metric:18s} = {stats['mean']:.3f} ± {stats['std']:.3f}")

    # 2) Is the model worth more than the hand-written rules?
    refs = reference_points(data)
    print("Reference points (same rows):")
    for name, scores in refs.items():
        print(
            f"  {name:11s} average_precision = {scores['average_precision']:.3f}"
            f"  roc_auc = {scores['roc_auc']:.3f}"
        )
    lift = None
    if "rule_score" in refs:
        lift = lift_over(
            cv_summary["average_precision"]["mean"], refs["rule_score"]["average_precision"]
        )
        print(f"  -> the model {'beats' if lift > 0 else 'does NOT beat'} the rules ({lift:+.0%})")

    # 3) Out-of-fold probabilities drive everything below: a threshold or a
    #    calibrator fitted on training predictions would be tuned on memorised answers.
    raw_proba = out_of_fold_proba(pipe, data, seed)

    calibrated_proba = raw_proba
    if calibrate:
        calibrated_proba = out_of_fold_calibrated_proba(pipe, data, seed)
        before, after = summarise(data.y.to_numpy(), raw_proba), summarise(
            data.y.to_numpy(), calibrated_proba
        )
        # Calibration is judged on *reliability* (Brier), not on ranking (PR-AUC).
        # SMOTE rebalances the classes to train, which inflates every probability:
        # the raw model claims ~32% risk on a population that gets injured ~5% of the
        # time. It ranks well and lies about magnitude. Isotonic regression fixes the
        # magnitude at a small cost in ranking — and a probability a medical staff can
        # read is the entire point of showing one.
        brier_before = brier_score_loss(data.y.to_numpy(), raw_proba)
        brier_after = brier_score_loss(data.y.to_numpy(), calibrated_proba)
        print("Calibration (isotonic):")
        print(
            f"  average_precision {before['average_precision']:.3f} -> "
            f"{after['average_precision']:.3f}   (ranking: slight cost)"
        )
        print(
            f"  brier             {brier_before:.4f} -> {brier_after:.4f}   (reliability: the point)"
        )
        print(
            f"  mean predicted    {raw_proba.mean():.3f} -> {calibrated_proba.mean():.3f}"
            f"   (actual rate {data.y.mean():.3f})"
        )

    # 4) The operating point, from the cost of being wrong.
    point = optimal_threshold(data.y.to_numpy(), calibrated_proba)
    print(f"Operating point (FN costs {COST_FALSE_NEGATIVE:.0f}x an FP):")
    print(f"  threshold = {point.threshold:.3f}")
    print(f"  recall    = {point.recall:.3f}   precision = {point.precision:.3f}")
    print(f"  flags {point.alert_rate:.1%} of athlete-days ({point.false_negatives} missed)")

    # 5) Figures.
    curves = {"Uncalibrated": raw_proba}
    if calibrate:
        curves["Isotonic"] = calibrated_proba
    plot_precision_recall(data.y.to_numpy(), calibrated_proba, track, point)
    plot_calibration(data.y.to_numpy(), curves, track)
    plot_confusion(data.y.to_numpy(), calibrated_proba, point, track)

    # 6) Fit the delivered artefact on everything.
    final = _calibrated(pipe, data, seed) if calibrate else pipe
    final.fit(data.X, data.y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"model_{track}.joblib"
    joblib.dump(
        {
            "pipeline": final,
            "model": model,
            "feature_cols": list(data.X.columns),
            "classes": [int(c) for c in np.unique(data.y)],
            "threshold": point.threshold,  # ships with the model, or 0.5 silently wins
            "calibrated": calibrate,
        },
        model_path,
    )

    metrics = {
        "track": track,
        "model": model,
        "tuned": bool(params),
        "calibrated": calibrate,
        "n_samples": int(len(data)),
        "n_features": int(data.X.shape[1]),
        "positive_rate": float(data.y.mean()),
        "cv_strategy": type(make_cv(track, seed=seed)).__name__,
        "n_groups": int(len(np.unique(data.groups))) if data.groups is not None else None,
        "cross_validation": cv_summary,
        "reference_points": refs,
        "lift_over_rules": lift,
        "out_of_fold": summarise(data.y.to_numpy(), calibrated_proba),
        "reliability": {
            "brier_uncalibrated": float(brier_score_loss(data.y.to_numpy(), raw_proba)),
            "brier_calibrated": float(brier_score_loss(data.y.to_numpy(), calibrated_proba)),
            "mean_predicted_uncalibrated": float(raw_proba.mean()),
            "mean_predicted_calibrated": float(calibrated_proba.mean()),
            "actual_rate": float(data.y.mean()),
        },
        "operating_point": point.as_dict(),
        "cost_ratio": COST_FALSE_NEGATIVE / COST_FALSE_POSITIVE,
    }
    metrics_path = REPORTS_DIR / f"metrics_{track}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"Model -> {model_path}")
    print(f"Metrics -> {metrics_path}")
    return metrics
