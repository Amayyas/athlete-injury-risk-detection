"""Tests for the model pipeline and the real-dataset loader.

The real SIRP-600 dataset is fetched from Kaggle and is **not** committed, so CI
cannot see it. Those tests skip cleanly instead of failing — the suite must pass
both with and without the private data present.
"""

from __future__ import annotations

import joblib
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

from injury_risk.config import SIRP_PATH, XGB_DEFAULT_PARAMS
from injury_risk.data.datasets import load_synthetic
from injury_risk.data.load_dataset import SIRP_FEATURE_COLS, SIRP_TARGET, load_sirp600
from injury_risk.models.train import build_pipeline

requires_real_data = pytest.mark.skipif(
    not SIRP_PATH.exists(),
    reason="SIRP-600 is a private Kaggle dataset; run `injury-risk download`",
)


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #


def test_pipeline_is_smote_then_classifier():
    pipe = build_pipeline(n_classes=3)
    assert isinstance(pipe, ImbPipeline)
    assert list(pipe.named_steps) == ["smote", "clf"]


def test_pipeline_objective_matches_the_class_count():
    assert build_pipeline(n_classes=3).named_steps["clf"].objective == "multi:softprob"
    assert build_pipeline(n_classes=2).named_steps["clf"].objective == "binary:logistic"


def test_pipeline_uses_the_config_defaults():
    clf = build_pipeline(n_classes=3).named_steps["clf"]
    assert clf.max_depth == XGB_DEFAULT_PARAMS["max_depth"]
    assert clf.n_estimators == XGB_DEFAULT_PARAMS["n_estimators"]


def test_tuned_params_override_the_defaults():
    clf = build_pipeline(n_classes=3, params={"max_depth": 7}).named_steps["clf"]
    assert clf.max_depth == 7


# --------------------------------------------------------------------------- #
# Fit / predict / persistence
# --------------------------------------------------------------------------- #


def test_pipeline_fits_predicts_and_round_trips_through_joblib(tmp_path):
    data = load_synthetic(sample_per_athlete=3, seed=42)
    pipe = build_pipeline(data.n_classes)
    pipe.fit(data.X, data.y)

    predictions = pipe.predict(data.X)
    assert len(predictions) == len(data)
    assert set(predictions) <= set(data.y.unique())

    path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": pipe, "feature_cols": list(data.X.columns)}, path)
    reloaded = joblib.load(path)

    assert reloaded["feature_cols"] == list(data.X.columns)
    assert (reloaded["pipeline"].predict(data.X) == predictions).all()


def test_predict_proba_rows_sum_to_one():
    data = load_synthetic(sample_per_athlete=3, seed=42)
    pipe = build_pipeline(data.n_classes)
    pipe.fit(data.X, data.y)
    proba = pipe.predict_proba(data.X)
    assert proba.shape == (len(data), data.n_classes)
    assert proba.sum(axis=1) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Real dataset loader (skipped when the Kaggle file is absent)
# --------------------------------------------------------------------------- #


def test_missing_dataset_raises_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        load_sirp600(path=tmp_path / "nope.xlsx")


@requires_real_data
def test_sirp600_has_the_expected_columns_and_target():
    df = load_sirp600()
    assert set(SIRP_FEATURE_COLS + [SIRP_TARGET]) == set(df.columns)
    assert set(df[SIRP_TARGET]) == {0, 1}
    assert not df.isna().any().any()


@requires_real_data
def test_sirp600_is_naturally_imbalanced():
    """~68/32 — the natural imbalance is why this dataset was chosen over the
    perfectly balanced 50/50 candidates."""
    df = load_sirp600()
    share = df[SIRP_TARGET].mean()
    assert 0.2 < share < 0.45
