"""Rule-based risk scoring, decomposed into explainable factors.

This module holds the project's business rules. It powers two things:

- the **labels** of the synthetic dataset (via :func:`composite_risk_score`);
- the dashboard's **live score** and its "active risk factors" list, before any ML
  model is involved.

Why the score and the factors live together
-------------------------------------------
They used to be written twice: the score here, and a hand-rolled list of factors
inlined in the Streamlit script — with **different thresholds**. Soreness started
raising the score at 4/10 but was only *displayed* from 6/10; the same held for the
heart-rate delta and the injury history. The dashboard could therefore show a 14%
risk gauge next to "no major risk factor detected", which is incoherent.

The fix is structural, not cosmetic: **the factor list is the score's
decomposition**. The score is the sum of the factor contributions, so a non-zero
contribution cannot exist without its factor appearing. The two cannot drift apart
again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from injury_risk.config import (
    ACWR_DANGER,
    ACWR_ELEVATED,
    ACWR_UNDER,
    HR_DELTA_RANGE,
    PREVIOUS_INJURIES_RANGE,
    RECENT_RETURN_DAYS,
    RISK_HIGH_THRESHOLD,
    RISK_LOW_THRESHOLD,
    SLEEP_RANGE,
    SLEEP_TARGET,
    SORENESS_ONSET,
    SORENESS_RANGE,
    W_ACWR_DANGER,
    W_ACWR_ELEVATED,
    W_ACWR_UNDER,
    W_INJURY_PRONE,
    W_PREVIOUS_INJURIES,
    W_RECENT_RETURN,
    W_RESTING_HR,
    W_SLEEP,
    W_SORENESS,
)

# Severity drives the colour/icon in the dashboard.
HIGH = "high"
MODERATE = "moderate"
INFO = "info"


@dataclass(frozen=True)
class RiskFactor:
    """One rule contributing to the risk score."""

    code: str
    label: str
    detail: str
    contribution: float
    severity: str = MODERATE

    @property
    def percent(self) -> int:
        """Contribution expressed in risk points (the score is on a 0-1 scale)."""
        return int(round(self.contribution * 100))


def _acwr_factor(acwr: float) -> RiskFactor:
    """The ACWR factor — the single most important rule in the project."""
    if acwr >= ACWR_DANGER:
        return RiskFactor(
            "acwr_danger",
            f"ACWR in danger zone (> {ACWR_DANGER})",
            f"Acute load far above the athlete's habit (ACWR {acwr:.2f})",
            W_ACWR_DANGER,
            HIGH,
        )
    if acwr >= ACWR_ELEVATED:
        return RiskFactor(
            "acwr_elevated",
            f"Elevated ACWR ({ACWR_ELEVATED}–{ACWR_DANGER})",
            f"Rising load, worth monitoring (ACWR {acwr:.2f})",
            W_ACWR_ELEVATED,
            MODERATE,
        )
    if acwr < ACWR_UNDER:
        return RiskFactor(
            "acwr_under",
            f"Low ACWR (< {ACWR_UNDER})",
            f"Possible under-loading / detraining (ACWR {acwr:.2f})",
            W_ACWR_UNDER,
            INFO,
        )
    # Optimal zone: no contribution, hence no factor.
    return RiskFactor("acwr_optimal", "ACWR optimal", f"ACWR {acwr:.2f}", 0.0, INFO)


def all_factors(
    *,
    acwr: float,
    soreness: float,
    sleep_hours: float,
    resting_hr: float,
    baseline_hr: float = 55.0,
    injury_prone: bool = False,
    previous_injuries: int = 0,
    days_since_injury: float = 365.0,
) -> list[RiskFactor]:
    """Every factor, in canonical order, including the ones contributing zero.

    :func:`composite_risk_score` sums these in this exact order; :func:`risk_factors`
    keeps only the ones that actually contribute.
    """
    soreness_c = float(np.clip((soreness - SORENESS_ONSET) / SORENESS_RANGE, 0, 1) * W_SORENESS)
    sleep_c = float(np.clip((SLEEP_TARGET - sleep_hours) / SLEEP_RANGE, 0, 1) * W_SLEEP)
    hr_delta = resting_hr - baseline_hr
    hr_c = float(np.clip(hr_delta / HR_DELTA_RANGE, 0, 1) * W_RESTING_HR)
    prone_c = W_INJURY_PRONE if injury_prone else 0.0
    history_c = float(
        np.clip(previous_injuries / PREVIOUS_INJURIES_RANGE, 0, 1) * W_PREVIOUS_INJURIES
    )
    recent_c = (
        W_RECENT_RETURN * (1 - days_since_injury / RECENT_RETURN_DAYS)
        if days_since_injury < RECENT_RETURN_DAYS
        else 0.0
    )

    return [
        _acwr_factor(acwr),
        RiskFactor(
            "soreness",
            "High soreness",
            f"{soreness:.1f}/10 (counts above {SORENESS_ONSET:.0f})",
            soreness_c,
            HIGH if soreness >= 7 else MODERATE,
        ),
        RiskFactor(
            "sleep",
            "Insufficient sleep",
            f"{sleep_hours:.1f} h/night (target {SLEEP_TARGET:.0f} h)",
            sleep_c,
            HIGH if sleep_hours < 5 else MODERATE,
        ),
        RiskFactor(
            "resting_hr",
            "Elevated resting HR",
            f"+{hr_delta:.0f} bpm above baseline — sign of fatigue/stress",
            hr_c,
            MODERATE,
        ),
        RiskFactor(
            "injury_prone",
            "Injury proneness",
            "Athlete flagged as an at-risk profile",
            prone_c,
            MODERATE,
        ),
        RiskFactor(
            "injury_history",
            "Injury history",
            f"{previous_injuries} past injuries",
            history_c,
            MODERATE,
        ),
        RiskFactor(
            "recent_return",
            "Recent return from injury",
            f"{days_since_injury:.0f} days ago (< {RECENT_RETURN_DAYS}) — tissue still fragile",
            recent_c,
            HIGH,
        ),
    ]


def risk_factors(
    *,
    acwr: float,
    soreness: float,
    sleep_hours: float,
    resting_hr: float,
    baseline_hr: float = 55.0,
    injury_prone: bool = False,
    previous_injuries: int = 0,
    days_since_injury: float = 365.0,
) -> list[RiskFactor]:
    """The **active** risk factors, strongest first.

    A factor is active exactly when it contributes to the score — which is what
    makes the display and the score impossible to contradict.
    """
    factors = all_factors(
        acwr=acwr,
        soreness=soreness,
        sleep_hours=sleep_hours,
        resting_hr=resting_hr,
        baseline_hr=baseline_hr,
        injury_prone=injury_prone,
        previous_injuries=previous_injuries,
        days_since_injury=days_since_injury,
    )
    active = [f for f in factors if f.contribution > 0]
    return sorted(active, key=lambda f: f.contribution, reverse=True)


def composite_risk_score(
    *,
    acwr: float,
    soreness: float,
    sleep_hours: float,
    resting_hr: float,
    baseline_hr: float = 55.0,
    injury_prone: bool = False,
    previous_injuries: int = 0,
    days_since_injury: float = 365.0,
) -> float:
    """Continuous risk score (0 = low, ~1 = very high), from the business rules.

    It is exactly the sum of the factor contributions, bounded to [0, 1] — which is
    what guarantees the dashboard's factor list can never contradict the gauge.
    """
    factors = all_factors(
        acwr=acwr,
        soreness=soreness,
        sleep_hours=sleep_hours,
        resting_hr=resting_hr,
        baseline_hr=baseline_hr,
        injury_prone=injury_prone,
        previous_injuries=previous_injuries,
        days_since_injury=days_since_injury,
    )
    score = 0.0
    for factor in factors:
        score += factor.contribution
    return float(np.clip(score, 0.0, 1.0))


def risk_score_to_level(
    score: float,
    low_thr: float = RISK_LOW_THRESHOLD,
    high_thr: float = RISK_HIGH_THRESHOLD,
) -> int:
    """Convert a continuous score into a risk level: 0=low, 1=moderate, 2=high."""
    if score >= high_thr:
        return 2
    if score >= low_thr:
        return 1
    return 0


@dataclass(frozen=True)
class RiskAssessment:
    """A complete rule-based assessment: the score, its level and its explanation."""

    score: float
    level: int
    factors: list[RiskFactor]


def assess(
    *,
    acwr: float,
    soreness: float,
    sleep_hours: float,
    resting_hr: float,
    baseline_hr: float = 55.0,
    injury_prone: bool = False,
    previous_injuries: int = 0,
    days_since_injury: float = 365.0,
) -> RiskAssessment:
    """Score, level and active factors in one call.

    Consumers (the dashboard, and later the API) should use this rather than
    computing the score and the factors separately: it guarantees the number and
    its explanation are derived from the very same evaluation.
    """
    factors = all_factors(
        acwr=acwr,
        soreness=soreness,
        sleep_hours=sleep_hours,
        resting_hr=resting_hr,
        baseline_hr=baseline_hr,
        injury_prone=injury_prone,
        previous_injuries=previous_injuries,
        days_since_injury=days_since_injury,
    )
    score = 0.0
    for factor in factors:
        score += factor.contribution
    score = float(np.clip(score, 0.0, 1.0))

    active = sorted(
        (f for f in factors if f.contribution > 0),
        key=lambda f: f.contribution,
        reverse=True,
    )
    return RiskAssessment(score=score, level=risk_score_to_level(score), factors=active)
