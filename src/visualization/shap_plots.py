"""Model explainability via SHAP.

The project favors **explainability** over raw accuracy: a medical staff must
understand *why* an athlete is classified as at risk. We use a
``shap.TreeExplainer`` (fast and exact on XGBoost trees) to produce:

- a global **summary plot**: importance and effect of each feature;
- an individual **waterfall plot**: contribution of each feature to a given
  athlete's prediction.

Usage:
    python -m src.visualization.shap_plots --track synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend (file generation)
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"


def _load_model(track: str):
    path = MODELS_DIR / f"model_{track}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\nRun first: python -m src.models.train --track {track}"
        )
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["feature_cols"], bundle["classes"]


def _sample_data(track: str, feature_cols: list[str], n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Rebuild a feature sample to explain the model."""
    if track == "synthetic":
        from src.models.train import _prepare_synthetic

        X, _, _ = _prepare_synthetic(sample_per_athlete=40, seed=seed)
    else:
        from src.models.train import _prepare_real

        X, _, _ = _prepare_real(seed=seed)
    X = X[feature_cols]
    if len(X) > n:
        X = X.sample(n, random_state=seed)
    return X.reset_index(drop=True)


def compute_shap(track: str, n: int = 500, seed: int = 42):
    """Return (explainer, shap_values, X) for a given track."""
    pipe, feature_cols, _ = _load_model(track)
    model = pipe.named_steps["clf"]
    X = _sample_data(track, feature_cols, n=n, seed=seed)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return explainer, shap_values, X


def save_summary_plot(track: str, n: int = 500, seed: int = 42) -> Path:
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
    track: str, index: int = 0, class_idx: int | None = None, n: int = 500, seed: int = 42
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the SHAP plots.")
    parser.add_argument("--track", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--index", type=int, default=0, help="row for the waterfall")
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    save_summary_plot(args.track, n=args.n)
    save_waterfall_plot(args.track, index=args.index, n=args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
