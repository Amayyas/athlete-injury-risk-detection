"""FastAPI service exposing the injury-risk model.

The API is a thin layer over :mod:`injury_risk.inference` — the same seam the
dashboard uses. That is deliberate: the probability this service returns and the one
the dashboard displays come from one implementation, so they cannot drift apart as
the model evolves.

Endpoints:

- ``GET  /health``     — liveness, and whether a model is loaded
- ``GET  /model-info`` — what is actually deployed (model, threshold, features)
- ``POST /assess``     — the rule-based score and its decomposition (**no model needed**)
- ``POST /predict``    — the model's calibrated probability and its decision
- ``POST /explain``    — per-feature SHAP contributions for one athlete

``/assess`` works with no trained model, exactly like the dashboard's rule-based
panel; the model endpoints answer **503** with an actionable message instead of
pretending.

Run:
    uvicorn injury_risk.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, status

from injury_risk.api.schemas import (
    AssessmentResponse,
    AthleteRequest,
    ExplanationResponse,
    FeatureContribution,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RiskFactorOut,
)
from injury_risk.config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    PREDICTION_HORIZON_DAYS,
    RISK_LABELS,
)
from injury_risk.features.engineering import acwr_zone
from injury_risk.features.risk_factors import assess as rule_assess
from injury_risk.inference import Predictor, load_predictor

TRACK = "synthetic"

app = FastAPI(
    title="Athlete Injury Risk API",
    description=(
        "Predicts the probability that an athlete suffers a muscle injury within the "
        "next 7 days, with a calibrated probability, a cost-based decision and a SHAP "
        "explanation. A transparent rule-based score is available without any model."
    ),
    version="0.1.0",
)


def _predictor() -> Predictor:
    """The loaded model, or a 503 explaining how to produce one."""
    try:
        return load_predictor(TRACK)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No trained model is available. Run `injury-risk train --track "
                f"{TRACK}` (or mount a models/ volume) and restart."
            ),
        ) from exc


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Liveness check. Reports whether a model is loaded, without failing if not."""
    try:
        load_predictor(TRACK)
        loaded = True
    except FileNotFoundError:
        loaded = False
    return HealthResponse(status="ok", model_loaded=loaded)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["service"])
def model_info() -> ModelInfoResponse:
    """What is deployed right now — so a caller can tell two versions apart."""
    predictor = _predictor()
    return ModelInfoResponse(
        track=predictor.track,
        model=predictor.model,
        calibrated=True,
        threshold=predictor.threshold,
        n_features=len(predictor.feature_cols),
        features=list(predictor.feature_cols),
        horizon_days=PREDICTION_HORIZON_DAYS,
        cost_ratio=COST_FALSE_NEGATIVE / COST_FALSE_POSITIVE,
    )


@app.post("/assess", response_model=AssessmentResponse, tags=["risk"])
def assess(request: AthleteRequest) -> AssessmentResponse:
    """Rule-based score and its decomposition. Needs no trained model.

    The contributions sum to the score, so this response can never claim a risk it
    does not explain.
    """
    inputs = request.to_inputs()
    result = rule_assess(
        acwr=inputs.acwr,
        soreness=inputs.soreness,
        sleep_hours=inputs.sleep_hours,
        resting_hr=inputs.resting_hr,
        baseline_hr=inputs.baseline_hr,
        injury_prone=inputs.injury_prone,
        previous_injuries=inputs.previous_injuries,
        days_since_injury=inputs.days_since_injury,
    )
    return AssessmentResponse(
        score=result.score,
        level=result.level,
        level_label=RISK_LABELS[result.level],
        acwr=inputs.acwr,
        acwr_zone=acwr_zone(inputs.acwr),
        factors=[
            RiskFactorOut(
                code=f.code,
                label=f.label,
                detail=f.detail,
                contribution=f.contribution,
                severity=f.severity,
            )
            for f in result.factors
        ],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["risk"])
def predict(request: AthleteRequest) -> PredictionResponse:
    """The model's calibrated probability of an injury within the next 7 days."""
    predictor = _predictor()
    prediction = predictor.predict(request.to_inputs())
    return PredictionResponse(
        probability=prediction.probability,
        at_risk=prediction.at_risk,
        threshold=prediction.threshold,
        model=prediction.model,
        horizon_days=PREDICTION_HORIZON_DAYS,
    )


@app.post("/explain", response_model=ExplanationResponse, tags=["risk"])
def explain(request: AthleteRequest) -> ExplanationResponse:
    """Per-feature SHAP contributions behind this athlete's prediction."""
    predictor = _predictor()
    explanation = predictor.explain(request.to_inputs())

    contributions = [
        FeatureContribution(
            feature=name,
            value=float(value),
            contribution=float(shap_value),
        )
        for name, value, shap_value in zip(
            predictor.feature_cols,
            explanation.data,
            explanation.values,
            strict=True,
        )
    ]
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)

    return ExplanationResponse(
        model=predictor.model,
        base_value=float(explanation.base_values),
        contributions=contributions,
    )
