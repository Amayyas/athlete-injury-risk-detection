"""Request and response contracts for the API.

The validation here is not decoration. The model was trained on a specific range of
athlete-days; a caller sending a resting heart rate of 500 or a negative sleep
duration would get a confident, meaningless number back. Bounds turn that into a
422 with a readable message instead.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from injury_risk.config import POSITIONS
from injury_risk.inference import AthleteInputs

# Bounds mirror the dashboard sliders, which mirror the training distribution.
Position = str


class AthleteRequest(BaseModel):
    """One athlete's snapshot.

    Loads are **daily** training loads (RPE x duration) averaged over the window —
    the scale the model was trained on, a few dozen, not a weekly total.
    """

    age: int = Field(24, ge=16, le=45, description="Age in years")
    position: Position = Field("midfielder", description=f"One of {list(POSITIONS)}")
    acute_load: float = Field(66, ge=0, le=400, description="7-day average daily load")
    chronic_load: float = Field(63, gt=0, le=400, description="28-day average daily load")
    sleep_hours: float = Field(7.4, ge=0, le=14, description="Hours of sleep per night")
    soreness: float = Field(3.0, ge=0, le=10, description="Subjective soreness, 0-10")
    resting_hr: float = Field(55, ge=30, le=120, description="Resting heart rate (bpm)")
    baseline_hr: float = Field(55, ge=30, le=120, description="The athlete's usual resting HR")
    injury_prone: bool = Field(False, description="Flagged as an at-risk profile")
    previous_injuries: int = Field(1, ge=0, le=50)
    days_since_injury: float = Field(200, ge=0, le=5000)

    def to_inputs(self) -> AthleteInputs:
        return AthleteInputs(**self.model_dump())

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """The model's answer, and the operating point that turned it into a decision."""

    probability: float = Field(..., description="Calibrated probability of injury within 7 days")
    at_risk: bool = Field(..., description="probability >= threshold")
    threshold: float = Field(..., description="Cost-based decision threshold")
    model: str = Field(..., description="The estimator that produced this")
    horizon_days: int = Field(..., description="Prediction horizon")


class RiskFactorOut(BaseModel):
    """One rule contributing to the rule-based score."""

    code: str
    label: str
    detail: str
    contribution: float = Field(..., description="Share of the 0-1 score this factor adds")
    severity: str


class AssessmentResponse(BaseModel):
    """The rule-based reading — available even with no trained model."""

    score: float = Field(..., description="Rule-based risk score, 0-1")
    level: int = Field(..., description="0 = low, 1 = moderate, 2 = high")
    level_label: str
    acwr: float
    acwr_zone: str
    factors: list[RiskFactorOut] = Field(
        ..., description="The score's decomposition: contributions sum to the score"
    )


class FeatureContribution(BaseModel):
    feature: str
    value: float = Field(..., description="The athlete's value for this feature")
    contribution: float = Field(..., description="SHAP value: how it moved the prediction")


class ExplanationResponse(BaseModel):
    """Why the model said what it said, for this athlete."""

    model: str
    base_value: float = Field(..., description="The model's average output before features")
    contributions: list[FeatureContribution] = Field(
        ..., description="Sorted by absolute impact, strongest first"
    )


class ModelInfoResponse(BaseModel):
    """What is actually deployed, so a caller can tell versions apart."""

    track: str
    model: str
    calibrated: bool
    threshold: float
    n_features: int
    features: list[str]
    horizon_days: int
    cost_ratio: float = Field(..., description="How many false alarms one missed injury is worth")
    version: str = Field(..., description="The deployed package version")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
