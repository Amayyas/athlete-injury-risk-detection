"""A CI guard on **model quality**, not just on code.

The unit tests prove the pipeline runs. They say nothing about whether the model is
still any good — a silent modelling regression (a broken feature, a reverted leakage
fix, a resampler applied at the wrong moment) leaves every test green while the model
quietly degrades.

So this runs the whole pipeline on a small deterministic dataset and checks the
resulting metrics against committed thresholds.

Sizing it honestly
------------------
The temptation is to make it tiny for speed. Measured across seeds, at 60 athletes
PR-AUC swings ±0.093 — a threshold there would be testing sampling noise, and would
either fire on nothing or fire at random. At 120 athletes × 500 days it tightens to
±0.035 (ROC-AUC to ±0.010) and still runs in seconds. With a fixed seed the whole run
is reproducible to the sixth decimal, so the thresholds can be tight.

Checking both directions
------------------------
Floors catch degradation. The **ceiling** catches the opposite failure, which this
project has actually shipped twice: athletes spanning CV folds, then a calibrator
silently dropping the grouping. Both made the numbers *better*. Injury prediction
cannot legitimately reach ROC-AUC 0.95 on this data — if it does, something leaked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_validate

from injury_risk.config import (
    DEFAULT_SEED,
    SCORING,
    SMOKE_BASELINE_AVERAGE_PRECISION,
    SMOKE_BASELINE_ROC_AUC,
    SMOKE_MAX_ROC_AUC,
    SMOKE_MIN_AVERAGE_PRECISION,
    SMOKE_MIN_LIFT_OVER_CHANCE,
    SMOKE_MIN_ROC_AUC,
    SMOKE_N_ATHLETES,
    SMOKE_N_DAYS,
    SMOKE_SAMPLE_PER_ATHLETE,
    TARGET_COL,
    WARMUP_DAYS,
)
from injury_risk.data.generate_synthetic import generate
from injury_risk.features.engineering import SYNTHETIC_FEATURE_COLS, build_features
from injury_risk.models.candidates import build_candidate, delivered_model
from injury_risk.models.splits import make_cv


@dataclass
class Check:
    """One assertion about the model, and whether it held."""

    name: str
    value: float
    expected: str
    passed: bool


@dataclass
class SmokeResult:
    n_rows: int
    n_athletes: int
    prevalence: float
    average_precision: float
    roc_auc: float
    lift_over_chance: float
    model: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def as_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_athletes": self.n_athletes,
            "prevalence": self.prevalence,
            "average_precision": self.average_precision,
            "roc_auc": self.roc_auc,
            "lift_over_chance": self.lift_over_chance,
            "model": self.model,
            "passed": self.passed,
            "checks": [
                {"name": c.name, "value": c.value, "expected": c.expected, "passed": c.passed}
                for c in self.checks
            ],
        }


def run_smoke_test(seed: int = DEFAULT_SEED) -> SmokeResult:
    """Generate, train, evaluate, and check the metrics against the thresholds.

    Uses **default** hyperparameters on purpose: the tuned ones are not committed
    (``models/*.json`` is gitignored), so CI must be able to run this from a clean
    checkout with nothing but the source.
    """
    frame = build_features(generate(n_athletes=SMOKE_N_ATHLETES, n_days=SMOKE_N_DAYS, seed=seed))
    frame = frame[
        (frame["day"] >= WARMUP_DAYS) & (~frame["is_injured"]) & (frame["horizon_complete"])
    ]
    sampled = (
        frame.groupby("athlete_id", group_keys=False)
        .sample(frac=1.0, random_state=seed)
        .groupby("athlete_id")
        .head(SMOKE_SAMPLE_PER_ATHLETE)
        .index
    )
    frame = frame.loc[sampled]

    X = frame[SYNTHETIC_FEATURE_COLS]
    y = frame[TARGET_COL]
    groups = frame["athlete_id"].to_numpy()

    if y.nunique() < 2:
        raise RuntimeError("the smoke dataset contains a single class — check the generator")

    model = delivered_model("synthetic")
    scores = cross_validate(
        build_candidate(model, 2, seed),
        X,
        y,
        groups=groups,
        cv=make_cv("synthetic", seed=seed),
        scoring=SCORING,
        n_jobs=-1,
    )

    prevalence = float(y.mean())
    average_precision = float(np.mean(scores["test_average_precision"]))
    roc_auc = float(np.mean(scores["test_roc_auc"]))
    lift = average_precision / prevalence if prevalence else 0.0

    result = SmokeResult(
        n_rows=len(X),
        n_athletes=int(len(np.unique(groups))),
        prevalence=prevalence,
        average_precision=average_precision,
        roc_auc=roc_auc,
        lift_over_chance=lift,
        model=model,
    )
    result.checks = [
        Check(
            "average_precision",
            average_precision,
            f">= {SMOKE_MIN_AVERAGE_PRECISION}",
            average_precision >= SMOKE_MIN_AVERAGE_PRECISION,
        ),
        Check("roc_auc", roc_auc, f">= {SMOKE_MIN_ROC_AUC}", roc_auc >= SMOKE_MIN_ROC_AUC),
        Check(
            "lift_over_chance",
            lift,
            f">= {SMOKE_MIN_LIFT_OVER_CHANCE}x",
            lift >= SMOKE_MIN_LIFT_OVER_CHANCE,
        ),
        Check(
            "roc_auc_not_suspicious",
            roc_auc,
            f"<= {SMOKE_MAX_ROC_AUC} (higher suggests leakage)",
            roc_auc <= SMOKE_MAX_ROC_AUC,
        ),
    ]
    return result


def format_report(result: SmokeResult) -> str:
    """A readable summary — this is what a failing CI job shows."""
    lines = [
        f"ML smoke test — {result.model} on {result.n_rows} rows "
        f"({result.n_athletes} athletes, {result.prevalence:.2%} positive)",
        "",
    ]
    for check in result.checks:
        mark = "PASS" if check.passed else "FAIL"
        lines.append(f"  [{mark}] {check.name:24s} {check.value:.4f}  (expected {check.expected})")

    # The run is deterministic, so any movement here is a real change, not noise —
    # visible even when it is far too small to trip a floor.
    lines += [
        "",
        "  drift vs recorded baseline (deterministic run):",
        f"    average_precision  {result.average_precision:.4f}  "
        f"({result.average_precision - SMOKE_BASELINE_AVERAGE_PRECISION:+.4f})",
        f"    roc_auc            {result.roc_auc:.4f}  "
        f"({result.roc_auc - SMOKE_BASELINE_ROC_AUC:+.4f})",
    ]
    lines.append("")
    lines.append("RESULT: PASS" if result.passed else "RESULT: FAIL — model quality regressed")
    return "\n".join(lines)


def write_report(result: SmokeResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2))
