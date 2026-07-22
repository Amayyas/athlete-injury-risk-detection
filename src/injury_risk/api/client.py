"""HTTP client for the injury-risk API.

Lets the dashboard (or any other consumer) talk to the service over the network
instead of loading the model in-process — the actual client/server split, rather
than two copies of the same logic pretending to be one.

The client deliberately returns the **same types** the in-process
:class:`injury_risk.inference.Predictor` returns: a :class:`Prediction`, and a
``shap.Explanation`` rebuilt from the API's per-feature contributions. That is what
lets the dashboard swap one for the other without branching on which is in use —
including rendering the identical SHAP waterfall either way.
"""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np

from injury_risk.inference import AthleteInputs, Prediction

DEFAULT_TIMEOUT = 10.0


class ApiError(RuntimeError):
    """The API could not answer (unreachable, or no model deployed)."""


class ApiClient:
    """Talks to a running injury-risk API.

    ``client`` exists for testing: FastAPI's ``TestClient`` *is* an ``httpx.Client``
    bound to the ASGI app, so injecting one exercises this exact client against the
    real API without a live server. (An ``ASGITransport`` cannot be used here — it is
    async-only.)
    """

    source = "api"

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)
        self._info: dict[str, Any] | None = None

    # -- service ---------------------------------------------------------- #

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def model_info(self) -> dict[str, Any]:
        if self._info is None:
            self._info = self._get("/model-info")
        return self._info

    @property
    def model(self) -> str:
        return str(self.model_info()["model"])

    @property
    def threshold(self) -> float:
        return float(self.model_info()["threshold"])

    @property
    def feature_cols(self) -> list[str]:
        return list(self.model_info()["features"])

    # -- risk ------------------------------------------------------------- #

    def predict(self, inputs: AthleteInputs) -> Prediction:
        body = self._post("/predict", inputs)
        return Prediction(
            probability=float(body["probability"]),
            at_risk=bool(body["at_risk"]),
            threshold=float(body["threshold"]),
            model=str(body["model"]),
        )

    def explain(self, inputs: AthleteInputs, seed: int | None = None) -> Any:
        """Rebuild a ``shap.Explanation`` from the API's contributions.

        ``seed`` is accepted and ignored, so this stays interchangeable with the
        in-process predictor's signature.
        """
        import shap

        body = self._post("/explain", inputs)
        contributions = body["contributions"]
        return shap.Explanation(
            values=np.array([c["contribution"] for c in contributions], dtype=float),
            base_values=float(body["base_value"]),
            data=np.array([c["value"] for c in contributions], dtype=float),
            feature_names=[c["feature"] for c in contributions],
        )

    def assess(self, inputs: AthleteInputs) -> dict[str, Any]:
        """The rule-based reading — served even when no model is deployed."""
        return self._post("/assess", inputs)

    # -- plumbing --------------------------------------------------------- #

    def _get(self, path: str) -> dict[str, Any]:
        try:
            response = self._client.get(path)
        except httpx.HTTPError as exc:
            raise ApiError(f"API unreachable at {self.base_url}: {exc}") from exc
        return self._unwrap(response)

    def _post(self, path: str, inputs: AthleteInputs) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=_payload(inputs))
        except httpx.HTTPError as exc:
            raise ApiError(f"API unreachable at {self.base_url}: {exc}") from exc
        return self._unwrap(response)

    @staticmethod
    def _unwrap(response: httpx.Response) -> dict[str, Any]:
        if response.status_code == 503:
            raise ApiError(response.json().get("detail", "no model deployed"))
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def close(self) -> None:
        """Close the underlying connection — unless the caller supplied it."""
        if self._owns_client:
            self._client.close()


def _payload(inputs: AthleteInputs) -> dict[str, Any]:
    """AthleteInputs -> the JSON body the API expects."""
    return {
        "age": inputs.age,
        "position": inputs.position,
        "acute_load": inputs.acute_load,
        "chronic_load": inputs.chronic_load,
        "sleep_hours": inputs.sleep_hours,
        "soreness": inputs.soreness,
        "resting_hr": inputs.resting_hr,
        "baseline_hr": inputs.baseline_hr,
        "injury_prone": inputs.injury_prone,
        "previous_injuries": inputs.previous_injuries,
        "days_since_injury": inputs.days_since_injury,
    }
