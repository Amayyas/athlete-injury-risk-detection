"""Out-of-fold evaluation: PR curve, calibration, confusion matrix.

Everything here is computed from **out-of-fold** predictions (grouped by athlete), so
the curves, the calibration and the chosen threshold all describe how the model
behaves on athletes it has never seen — not on the ones it memorised.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (file generation)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict  # noqa: E402

from injury_risk.config import DEFAULT_SEED, FIGURES_DIR
from injury_risk.data.datasets import Dataset  # noqa: E402
from injury_risk.models.splits import make_cv, materialise_folds  # noqa: E402
from injury_risk.models.threshold import OperatingPoint, confusion_at  # noqa: E402


def out_of_fold_proba(pipe, data: Dataset, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Predicted probabilities for rows the model did not train on."""
    proba = cross_val_predict(
        pipe,
        data.X,
        data.y,
        groups=data.groups,
        cv=make_cv(data.track, seed=seed),
        method="predict_proba",
        n_jobs=-1,
    )
    return np.asarray(proba)[:, 1]


def out_of_fold_calibrated_proba(
    pipe,
    data: Dataset,
    seed: int = DEFAULT_SEED,
    inner_splits: int = 3,
) -> np.ndarray:
    """Out-of-fold probabilities of the **calibrated** model — nested, and grouped.

    This has to be written by hand rather than handed to ``cross_val_predict``.
    Calibration needs its own inner split of the training data, and
    ``CalibratedClassifierCV`` never forwards ``groups`` to its splitter. Passing it a
    ``StratifiedGroupKFold`` loses the grouping; passing it folds precomputed on the
    *full* dataset breaks the moment it receives a subset (the indices point past the
    end of it).

    So: for each grouped outer fold, calibration folds are recomputed **relative to
    that fold's training rows**, keeping athletes intact at both levels. Cost: one
    proper nested cross-validation. Benefit: the calibrated probabilities are honest,
    and the leakage fixed in #1 cannot creep back in through the calibrator.
    """
    X, y = data.X, data.y
    proba = np.zeros(len(y), dtype=float)

    outer = materialise_folds(X, y, data.groups, data.track, seed=seed)
    for train_idx, test_idx in outer:
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        groups_tr = data.groups[train_idx] if data.groups is not None else None

        inner = materialise_folds(
            X_tr, y_tr, groups_tr, data.track, seed=seed, n_splits=inner_splits
        )
        calibrator = CalibratedClassifierCV(pipe, method="isotonic", cv=inner)
        calibrator.fit(X_tr, y_tr)
        proba[test_idx] = calibrator.predict_proba(X.iloc[test_idx])[:, 1]

    return proba


def plot_precision_recall(
    y_true: np.ndarray, y_proba: np.ndarray, track: str, point: OperatingPoint | None = None
):
    """PR curve with the chance line and the chosen operating point."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    prevalence = float(np.mean(y_true))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#2563eb", label=f"Model (AP = {ap:.3f})")
    ax.axhline(
        prevalence,
        color="#9ca3af",
        linestyle="--",
        label=f"Chance (prevalence = {prevalence:.3f})",
    )
    if point is not None:
        ax.plot(
            point.recall,
            point.precision,
            "o",
            color="#ef4444",
            markersize=9,
            label=f"Operating point (t = {point.threshold:.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {track} (out-of-fold)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / f"pr_curve_{track}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_calibration(y_true: np.ndarray, curves: dict[str, np.ndarray], track: str):
    """Reliability diagram: does a predicted 20% actually mean 20%?"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="Perfectly calibrated")

    for label, proba in curves.items():
        # Quantile bins: with a rare positive class, uniform bins leave most of the
        # range empty and the diagram becomes unreadable.
        true_freq, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, true_freq, "o-", label=label)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration — {track} (out-of-fold)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / f"calibration_{track}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_confusion(y_true: np.ndarray, y_proba: np.ndarray, point: OperatingPoint, track: str):
    """Confusion matrix at the cost-based operating point."""
    tn, fp, fn, tp = confusion_at(y_true, y_proba, point.threshold)
    matrix = np.array([[tn, fp], [fn, tp]])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(matrix, cmap="Blues")
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{matrix[i, j]}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                fontsize=10,
            )
    ax.set_xticks([0, 1], ["Predicted safe", "Predicted at risk"])
    ax.set_yticks([0, 1], ["Actually safe", "Actually injured"])
    ax.set_title(f"Confusion at t = {point.threshold:.3f} — {track}")
    fig.tight_layout()

    out = FIGURES_DIR / f"confusion_{track}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def summarise(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """The headline out-of-fold scores."""
    return {
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }
