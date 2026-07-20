"""Model explainability via SHAP.

The project favours **explainability** over raw accuracy: a medical staff must
understand *why* an athlete is flagged. Two views:

- a global **summary plot**: importance and effect of each feature;
- an individual **waterfall plot**: how each feature moved one athlete's prediction.

The explainer is chosen from the model, not hardcoded. This used to assume
``TreeExplainer`` — which stopped working the moment the tuned benchmark put logistic
regression ahead of the tree models on the synthetic track. Picking the model on
evidence means the explainability has to follow the evidence too.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from injury_risk.config import DEFAULT_SEED, FIGURES_DIR
from injury_risk.data.datasets import load_track
from injury_risk.inference import load_predictor, make_explainer, to_model_space

matplotlib.use("Agg")  # non-interactive backend (file generation)
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402


def compute_shap(track: str, n: int = 500, seed: int = DEFAULT_SEED):
    """Return (explainer, shap_values, X) for a given track.

    The explainer selection and the transform-into-model-space logic are shared with
    :mod:`injury_risk.inference` (the dashboard and the API use the same), so a global
    summary and a live per-athlete explanation can never disagree about how the model
    reasons. SHAP is computed on the data the estimator sees (standardised, for the
    linear model) while the plots display the **original** values.
    """
    predictor = load_predictor(track)
    inner = predictor._inner
    estimator = inner.named_steps["clf"]

    X = load_track(track, seed=seed).X[predictor.feature_cols]
    if len(X) > n:
        X = X.sample(n, random_state=seed)
    X = X.reset_index(drop=True)

    X_model = to_model_space(inner, X)
    explainer = make_explainer(estimator, X_model)
    shap_values = explainer(X_model)
    # Show the human-readable values, keep the computed contributions.
    shap_values.data = X.to_numpy()
    return explainer, shap_values, X


def save_summary_plot(track: str, n: int = 500, seed: int = DEFAULT_SEED) -> Path:
    """Generate and save the global summary plot."""
    _, shap_values, X = compute_shap(track, n=n, seed=seed)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # In multiclass, shap_values has a class dimension: we aggregate the mean
    # absolute importance over the classes for the global view.
    values = shap_values.values
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        agg = np.abs(values).mean(axis=2)
        plt.figure()
        shap.summary_plot(agg, X, plot_type="bar", show=False)
    else:
        plt.figure()
        shap.summary_plot(values, X, show=False)

    out = FIGURES_DIR / f"shap_summary_{track}.png"
    plt.title(f"SHAP — global importance ({track})")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Summary plot -> {out}")
    return out


def save_waterfall_plot(
    track: str, index: int = 0, class_idx: int | None = None, n: int = 500, seed: int = DEFAULT_SEED
) -> Path:
    """Generate the waterfall plot for one athlete (row ``index``).

    In multiclass, ``class_idx`` selects the explained class (defaults to the
    highest-risk class).
    """
    _, shap_values, X = compute_shap(track, n=n, seed=seed)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if shap_values.values.ndim == 3:
        if class_idx is None:
            class_idx = shap_values.values.shape[2] - 1  # "high risk" class
        single = shap_values[index, :, class_idx]
    else:
        single = shap_values[index]

    plt.figure()
    shap.plots.waterfall(single, show=False)
    out = FIGURES_DIR / f"shap_waterfall_{track}_athlete{index}.png"
    plt.title(f"SHAP — individual explanation ({track}, athlete #{index})")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Waterfall plot -> {out}")
    return out
