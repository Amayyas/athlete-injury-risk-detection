"""Serving a trained model: inputs in, prediction (and explanation) out.

This is the seam the delivered model is used through. The dashboard calls it today;
the FastAPI service (#21) will wrap the same functions tomorrow, so the number the
UI shows and the number the API returns come from one place and cannot diverge.

Two things live here that are easy to get wrong elsewhere:

- **The feature mapping.** The model was trained on 21 features derived from a daily
  time series (rolling loads, ACWR, trend…). A dashboard user provides a *snapshot*:
  a recent acute load, a chronic load, today's sleep and soreness. Turning one into
  the other requires assumptions, and they are stated in one place
  (:func:`AthleteInputs.to_features`) rather than scattered and implicit.

- **The explainer.** The delivered model differs per track (linear on synthetic, a
  forest on real), so the explainer is chosen from the model — never hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from injury_risk.config import DEFAULT_SEED, MODELS_DIR, POSITION_TO_CODE
from injury_risk.data.datasets import load_track


@dataclass(frozen=True)
class AthleteInputs:
    """The snapshot a user (or an API caller) provides for a single athlete."""

    age: int = 24
    position: str = "midfielder"
    # Daily training loads (RPE x duration), the scale the model was trained on —
    # an acute/chronic average of a few dozen, not a weekly total in the hundreds.
    acute_load: float = 66.0  # recent 7-day average
    chronic_load: float = 63.0  # longer 28-day average
    sleep_hours: float = 7.4
    soreness: float = 3.0
    resting_hr: float = 55.0
    baseline_hr: float = 55.0
    injury_prone: bool = False
    previous_injuries: int = 1
    days_since_injury: float = 200.0

    @property
    def acwr(self) -> float:
        return self.acute_load / self.chronic_load if self.chronic_load else 1.0

    def to_features(self) -> pd.DataFrame:
        """Build the model's feature row from the snapshot.

        The rolling windows the model expects cannot be recovered from a single day,
        so they are filled under a **steady-state assumption**: the athlete has been
        at these values recently. Concretely:

        - the 7-day load *is* the provided acute load, the 28-day load *is* the chronic
          load, and the 14-day load is interpolated between them;
        - soreness and sleep are taken as their own recent averages;
        - the load trend is 0 (a snapshot carries no direction).

        These are honest approximations for a what-if tool, not measurements. A caller
        that has the real series should build features with
        :func:`injury_risk.features.engineering.build_features` instead.
        """
        acute, chronic = self.acute_load, self.chronic_load
        row = {
            "age": self.age,
            "position_code": POSITION_TO_CODE.get(self.position.lower(), -1),
            "injury_prone": bool(self.injury_prone),
            "baseline_hr": self.baseline_hr,
            "previous_injuries": self.previous_injuries,
            "days_since_injury": self.days_since_injury,
            "training_load": acute,  # best single-day estimate = the recent average
            "resting_hr": self.resting_hr,
            "sleep_hours": self.sleep_hours,
            "soreness": self.soreness,
            "training_load_7d": acute,
            "training_load_14d": acute + (chronic - acute) / 3.0,  # between 7d and 28d
            "training_load_28d": chronic,
            "soreness_7d": self.soreness,
            "soreness_14d": self.soreness,
            "sleep_hours_7d": self.sleep_hours,
            "acute_load": acute,
            "chronic_load": chronic,
            "acwr": self.acwr,
            "load_trend_7d": 0.0,
            "hr_delta": self.resting_hr - self.baseline_hr,
        }
        return pd.DataFrame([row])


@dataclass(frozen=True)
class Prediction:
    """A model prediction, with the operating point that turned it into a decision."""

    probability: float
    at_risk: bool
    threshold: float
    model: str


def _inner_pipeline(bundle_pipeline):
    """The SMOTE+estimator pipeline, reached through the calibrator if present."""
    if hasattr(bundle_pipeline, "calibrated_classifiers_"):
        return bundle_pipeline.calibrated_classifiers_[0].estimator
    return bundle_pipeline


def to_model_space(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Apply everything before the classifier (e.g. the scaler), keep column names.

    Resamplers such as SMOTE are inert at transform time, so this is just the
    preprocessing the estimator actually sees.
    """
    transformed = X
    for _, step in pipe.steps[:-1]:
        if hasattr(step, "transform"):
            transformed = step.transform(transformed)
    return pd.DataFrame(np.asarray(transformed), columns=X.columns, index=X.index)


def make_explainer(estimator, background: pd.DataFrame):
    """The right SHAP explainer for the model at hand.

    Linear model -> exact LinearExplainer; tree ensemble -> TreeExplainer. Hardcoding
    one broke the day model selection stopped being "XGBoost by default".
    """
    import shap

    if isinstance(estimator, LogisticRegression):
        return shap.LinearExplainer(estimator, background)
    return shap.TreeExplainer(estimator)


class Predictor:
    """A loaded model plus everything needed to predict and explain with it."""

    def __init__(self, bundle: dict, track: str):
        self.track = track
        self.pipeline = bundle["pipeline"]
        self.model = bundle["model"]
        self.feature_cols = bundle["feature_cols"]
        self.threshold = float(bundle["threshold"])
        self._inner = _inner_pipeline(self.pipeline)
        self._explainer: Any = None  # a SHAP explainer; shap ships no stubs
        self._background: pd.DataFrame | None = None

    def _features(self, inputs: AthleteInputs) -> pd.DataFrame:
        return inputs.to_features()[self.feature_cols]

    def predict(self, inputs: AthleteInputs) -> Prediction:
        X = self._features(inputs)
        proba = float(self.pipeline.predict_proba(X)[0, 1])
        return Prediction(
            probability=proba,
            at_risk=proba >= self.threshold,
            threshold=self.threshold,
            model=self.model,
        )

    def _ensure_explainer(self, seed: int = DEFAULT_SEED) -> None:
        if self._explainer is not None:
            return
        # A modest background of real feature rows, in the space the estimator sees.
        sample = load_track(self.track, seed=seed).X[self.feature_cols]
        if len(sample) > 200:
            sample = sample.sample(200, random_state=seed)
        self._background = to_model_space(self._inner, sample)
        self._explainer = make_explainer(self._inner.named_steps["clf"], self._background)

    def explain(self, inputs: AthleteInputs, seed: int = DEFAULT_SEED):
        """A SHAP explanation for this single athlete (for a waterfall plot).

        Computed in the estimator's space but carrying the athlete's **original**
        values, so the plot reads "sleep = 4.8 h", not "sleep = -2.3 σ".
        """
        self._ensure_explainer(seed)
        X = self._features(inputs)
        X_model = to_model_space(self._inner, X)
        explanation = self._explainer(X_model)
        explanation.data = X.to_numpy()
        return explanation[0]


def predictor_path(track: str):
    return MODELS_DIR / f"model_{track}.joblib"


@lru_cache(maxsize=4)
def load_predictor(track: str = "synthetic") -> Predictor:
    """Load the delivered model for a track (cached across calls)."""
    path = predictor_path(track)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model for track {track!r}: {path}\n"
            f"Run first: injury-risk train --track {track}"
        )
    return Predictor(joblib.load(path), track)
