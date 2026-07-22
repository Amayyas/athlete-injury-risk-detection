"""Tests for the HTTP client and the local/remote resolution.

The property that matters is **substitutability**: the dashboard must get the same
answer whether it loaded the model in-process or asked a server over the network. If
that ever stops holding, the client/server split becomes two implementations quietly
disagreeing — so it is asserted, not assumed.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from injury_risk.api.client import ApiClient, ApiError
from injury_risk.api.main import app
from injury_risk.inference import AthleteInputs, load_predictor, predictor_path
from injury_risk.service import API_URL_ENV, resolve_service

RISKY = AthleteInputs(
    acute_load=124,
    chronic_load=61,
    sleep_hours=4.8,
    soreness=7.5,
    resting_hr=68,
    injury_prone=True,
    previous_injuries=4,
    days_since_injury=28,
)

needs_model = pytest.mark.skipif(
    not predictor_path("synthetic").exists(),
    reason="needs a trained model; run `injury-risk train --track synthetic`",
)


@pytest.fixture
def api_client() -> ApiClient:
    """The real client, wired to the real app through FastAPI's TestClient."""
    return ApiClient("http://testserver", client=TestClient(app))


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #


def test_health_round_trips(api_client: ApiClient):
    assert api_client.health()["status"] == "ok"


def test_assess_works_over_http_without_a_model(api_client: ApiClient):
    body = api_client.assess(RISKY)
    assert body["acwr_zone"] == "danger"
    assert body["factors"]


@needs_model
def test_predict_returns_the_same_type_as_the_local_predictor(api_client: ApiClient):
    remote = api_client.predict(RISKY)
    local = load_predictor("synthetic").predict(RISKY)
    assert type(remote) is type(local)
    assert remote.model == local.model


@needs_model
def test_remote_and_local_predictions_are_identical(api_client: ApiClient):
    """The whole justification for one shared inference seam."""
    remote = api_client.predict(RISKY)
    local = load_predictor("synthetic").predict(RISKY)
    assert remote.probability == pytest.approx(local.probability, abs=1e-12)
    assert remote.at_risk == local.at_risk
    assert remote.threshold == pytest.approx(local.threshold)


@needs_model
def test_explanation_is_rebuilt_as_a_shap_object(api_client: ApiClient):
    """The dashboard renders the same waterfall whichever side produced it."""
    explanation = api_client.explain(RISKY)
    predictor = load_predictor("synthetic")
    assert len(explanation.values) == len(predictor.feature_cols)
    assert set(explanation.feature_names) == set(predictor.feature_cols)
    assert float(explanation.base_values) == pytest.approx(
        float(predictor.explain(RISKY).base_values), abs=1e-9
    )


@needs_model
def test_model_info_is_cached(api_client: ApiClient):
    assert api_client.model_info() is api_client.model_info()
    assert api_client.model
    assert 0.0 < api_client.threshold < 1.0


def test_unreachable_api_raises_a_clear_error():
    client = ApiClient("http://127.0.0.1:1")  # nothing listens on port 1
    with pytest.raises(ApiError, match="unreachable"):
        client.health()


def test_injected_client_is_not_closed_by_us():
    """Closing a caller-supplied client would break their session."""
    supplied = TestClient(app)
    api = ApiClient("http://testserver", client=supplied)
    api.close()
    assert supplied.get("/health").status_code == 200  # still usable


def test_owned_client_is_closed():
    api = ApiClient("http://127.0.0.1:1")
    api.close()
    assert isinstance(api._client, httpx.Client)


# --------------------------------------------------------------------------- #
# Resolution: API first, local fallback, honest when neither
# --------------------------------------------------------------------------- #


@needs_model
def test_resolves_to_the_api_when_one_is_configured():
    resolution = resolve_service("http://testserver", client=TestClient(app))
    assert resolution.source == "api"
    assert resolution.service is not None
    assert resolution.service.source == "api"


@needs_model
def test_resolves_locally_when_no_api_is_configured(monkeypatch):
    monkeypatch.delenv(API_URL_ENV, raising=False)
    resolution = resolve_service()
    assert resolution.source == "local"
    assert resolution.service is not None
    assert resolution.service.source == "local"


@needs_model
def test_falls_back_to_local_when_the_api_is_down():
    """A dead API must degrade to in-process inference, not break the dashboard."""
    resolution = resolve_service("http://127.0.0.1:1")
    assert resolution.source == "local"
    assert "unreachable" in resolution.detail


@needs_model
def test_the_environment_variable_selects_the_api(monkeypatch):
    monkeypatch.setenv(API_URL_ENV, "http://127.0.0.1:1")
    resolution = resolve_service()  # unreachable -> falls back, but the env was read
    assert "127.0.0.1:1" in resolution.detail


@needs_model
def test_both_sources_are_interchangeable_for_the_dashboard():
    """Same call, same answer — regardless of which side of the network it came from."""
    remote = resolve_service("http://testserver", client=TestClient(app)).service
    local = resolve_service("").service
    assert remote is not None and local is not None
    assert remote.predict(RISKY).probability == pytest.approx(
        local.predict(RISKY).probability, abs=1e-12
    )
