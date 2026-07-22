"""Tests for the REST API.

Two things are worth asserting beyond "it returns 200":

- **validation actually rejects nonsense** — the point of a typed API is that a
  resting heart rate of 500 gets a 422, not a confident meaningless number;
- **the API and the dashboard agree** — both go through `injury_risk.inference`, and
  a test pins that down so the two readings cannot drift apart.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from injury_risk.api.main import app
from injury_risk.inference import AthleteInputs, load_predictor, predictor_path

client = TestClient(app)

RISKY = {
    "age": 24,
    "position": "midfielder",
    "acute_load": 124,
    "chronic_load": 61,
    "sleep_hours": 4.8,
    "soreness": 7.5,
    "resting_hr": 68,
    "baseline_hr": 55,
    "injury_prone": True,
    "previous_injuries": 4,
    "days_since_injury": 28,
}
HEALTHY = {
    "acute_load": 64,
    "chronic_load": 64,
    "sleep_hours": 7.8,
    "soreness": 1.0,
    "resting_hr": 52,
    "baseline_hr": 55,
    # Explicitly zero: the default of 1 past injury is itself a (small) contribution,
    # so leaving it out would not describe a genuinely risk-free athlete.
    "previous_injuries": 0,
    "injury_prone": False,
    "days_since_injury": 300,
}

needs_model = pytest.mark.skipif(
    not predictor_path("synthetic").exists(),
    reason="needs a trained model; run `injury-risk train --track synthetic`",
)


# --------------------------------------------------------------------------- #
# Service endpoints
# --------------------------------------------------------------------------- #


def test_health_never_fails_even_without_a_model():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)


def test_openapi_documents_every_endpoint():
    schema = client.get("/openapi.json").json()
    assert {"/health", "/model-info", "/assess", "/predict", "/explain"} <= set(schema["paths"])


# --------------------------------------------------------------------------- #
# /assess — the rule-based reading, which needs no model
# --------------------------------------------------------------------------- #


def test_assess_works_without_any_model():
    response = client.post("/assess", json=RISKY)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["level"] in (0, 1, 2)
    assert body["acwr"] == pytest.approx(124 / 61)
    assert body["acwr_zone"] == "danger"


def test_assess_factors_sum_to_the_score():
    """The API cannot report a risk it does not explain."""
    body = client.post("/assess", json=RISKY).json()
    total = sum(f["contribution"] for f in body["factors"])
    assert total == pytest.approx(body["score"], abs=1e-9)


def test_assess_healthy_athlete_has_no_factors():
    body = client.post("/assess", json=HEALTHY).json()
    assert body["score"] == 0.0
    assert body["factors"] == []


# --------------------------------------------------------------------------- #
# Validation — the reason for a typed API
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field,value",
    [
        ("resting_hr", 500),  # physiologically impossible
        ("sleep_hours", -3),  # negative time
        ("soreness", 42),  # scale is 0-10
        ("chronic_load", 0),  # would divide by zero in the ACWR
        ("age", 5),  # not an athlete in this population
        ("previous_injuries", -1),
    ],
)
def test_out_of_range_values_are_rejected(field, value):
    response = client.post("/assess", json={**RISKY, field: value})
    assert response.status_code == 422, f"{field}={value} should be rejected"


def test_unknown_field_types_are_rejected():
    assert client.post("/assess", json={**RISKY, "age": "young"}).status_code == 422


def test_defaults_make_the_body_optional():
    """Every field has a sensible default, so a bare {} is a valid request."""
    assert client.post("/assess", json={}).status_code == 200


# --------------------------------------------------------------------------- #
# /predict and /explain
# --------------------------------------------------------------------------- #


@needs_model
def test_predict_returns_a_decision_with_its_threshold():
    body = client.post("/predict", json=RISKY).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["at_risk"] == (body["probability"] >= body["threshold"])
    assert body["horizon_days"] == 7
    assert body["model"]


@needs_model
def test_predict_ranks_a_risky_athlete_above_a_healthy_one():
    risky = client.post("/predict", json=RISKY).json()["probability"]
    healthy = client.post("/predict", json=HEALTHY).json()["probability"]
    assert risky > healthy


@needs_model
def test_api_and_inference_agree():
    """The API must return exactly what the dashboard shows — same seam, same number."""
    api = client.post("/predict", json=RISKY).json()
    direct = load_predictor("synthetic").predict(AthleteInputs(**RISKY))
    assert api["probability"] == pytest.approx(direct.probability)
    assert api["at_risk"] is direct.at_risk


@needs_model
def test_explain_returns_one_contribution_per_feature():
    body = client.post("/explain", json=RISKY).json()
    predictor = load_predictor("synthetic")
    assert len(body["contributions"]) == len(predictor.feature_cols)
    assert {c["feature"] for c in body["contributions"]} == set(predictor.feature_cols)


@needs_model
def test_explain_is_sorted_by_impact():
    contributions = client.post("/explain", json=RISKY).json()["contributions"]
    magnitudes = [abs(c["contribution"]) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


@needs_model
def test_model_info_describes_what_is_deployed():
    body = client.get("/model-info").json()
    assert body["track"] == "synthetic"
    assert body["n_features"] == len(body["features"])
    assert body["horizon_days"] == 7
    assert body["cost_ratio"] == 10.0
