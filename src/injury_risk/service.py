"""Choosing where predictions come from: the API, or this process.

The dashboard should demonstrate a real client/server split — but it also has to keep
working when there is no server, because the free hosting it is deployed to runs a
single Streamlit process and nothing else.

So the source is *resolved*, not hardcoded:

1. if ``INJURY_RISK_API_URL`` is set and the API answers, use it over HTTP;
2. otherwise load the model in-process;
3. if there is no model either, say so — and the rule-based score carries the app.

Both paths satisfy the same small interface and return the same types, so the caller
never branches on which one it got. That is the point: swapping the transport must not
change the product.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from injury_risk.inference import AthleteInputs, Prediction

API_URL_ENV = "INJURY_RISK_API_URL"


@runtime_checkable
class RiskService(Protocol):
    """What the dashboard needs, whichever side of the network it lives on."""

    source: str

    @property
    def model(self) -> str: ...

    @property
    def threshold(self) -> float: ...

    def predict(self, inputs: AthleteInputs) -> Prediction: ...

    def explain(self, inputs: AthleteInputs) -> Any: ...


@dataclass(frozen=True)
class Resolution:
    """Which service was obtained, and why — so the UI can be honest about it."""

    service: RiskService | None
    source: str  # "api" | "local" | "none"
    detail: str


def resolve_service(api_url: str | None = None, client: Any = None) -> Resolution:
    """Return the best available risk service, preferring the API when configured."""
    url = api_url if api_url is not None else os.environ.get(API_URL_ENV, "").strip()

    if url:
        from injury_risk.api.client import ApiClient

        api = ApiClient(url, client=client)
        try:
            if api.health().get("model_loaded"):
                return Resolution(api, "api", f"Served by the API at {url}")
            api.close()
            return Resolution(None, "none", f"The API at {url} is up but has no model deployed")
        except Exception as exc:
            # Any failure to reach the API — connection refused, timeout, bad
            # payload — means the same thing here: fall back rather than break.
            api.close()
            local = _local()
            if local is not None:
                return Resolution(
                    local,
                    "local",
                    f"API at {url} unreachable ({type(exc).__name__}); using the local model",
                )
            return Resolution(None, "none", f"API at {url} unreachable and no local model")

    local = _local()
    if local is not None:
        return Resolution(local, "local", "Model loaded in-process")
    return Resolution(None, "none", "No model available — train one to enable predictions")


def _local() -> RiskService | None:
    """The in-process predictor, or None when no model has been trained."""
    from injury_risk.inference import load_predictor

    try:
        return load_predictor("synthetic")
    except FileNotFoundError:
        return None
