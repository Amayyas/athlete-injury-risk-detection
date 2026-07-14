"""Realistic synthetic data generator — with actual injury *events*.

Why this generator exists
-------------------------
No available real Kaggle dataset contains a **per-athlete daily time series**, which
is what an ACWR and rolling features need. So we simulate one.

Why the labels are events, not a score
--------------------------------------
The first version of this generator labelled each day by discretising the rule-based
``composite_risk_score`` of that same day — and the model was then trained on the
very variables that score was computed from. The model did not predict injuries: it
**re-learned the scoring function**. Its performance measured how well XGBoost can
imitate a formula.

Now the rules only drive a **hazard**. Each day an athlete may actually get injured:

    logit( P(injury on day t) ) = intercept + slope * latent_risk(t)

An injured athlete is sidelined for a recovery period (no real training load), and
comes back with ``previous_injuries`` incremented and ``days_since_injury`` reset —
which feeds back into their future risk, exactly as in real life.

The label becomes an **observed outcome**: "did this athlete get injured within the
next 7 days?". The link between features and target is now stochastic and has to be
genuinely learned, and the rule score becomes a *baseline to beat* rather than the
answer key.

Usage:
    python -m injury_risk.data.generate_synthetic            # 200 athletes, 730 days
    python -m injury_risk.data.generate_synthetic --athletes 50 --days 365
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from injury_risk.config import (
    ACUTE_WINDOW,
    CHRONIC_WINDOW,
    DEFAULT_SEED,
    HAZARD_INTERCEPT,
    HAZARD_SLOPE,
    INJURY_PRONE_RATE,
    INJURY_RECOVERY_DAYS,
    N_ATHLETES,
    N_DAYS,
    POSITION_BASE_LOAD,
    POSITIONS,
    PREDICTION_HORIZON_DAYS,
    SYNTHETIC_DATASET,
    TARGET_COL,
)
from injury_risk.features.risk_factors import composite_risk_score

DEFAULT_OUTPUT = SYNTHETIC_DATASET


def daily_hazard(latent_risk: float) -> float:
    """Probability of an injury occurring today, given the athlete's latent risk.

    A discrete-time logistic hazard: the intercept sets the base rate, the slope
    sets how strongly risk actually translates into injuries.
    """
    return float(1.0 / (1.0 + np.exp(-(HAZARD_INTERCEPT + HAZARD_SLOPE * latent_risk))))


def _simulate_athlete(athlete_id: int, n_days: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate one athlete's period, day by day, injuries included.

    The simulation has to be sequential: today's injury changes tomorrow's training
    load, injury history and time-since-injury, which change tomorrow's risk. The
    ACWR is therefore maintained online rather than computed afterwards.
    """
    position = str(rng.choice(POSITIONS))
    age = int(np.clip(rng.normal(24, 4), 17, 38))
    base_fitness = float(np.clip(rng.normal(0.7, 0.12), 0.3, 1.0))
    injury_prone = bool(rng.random() < INJURY_PRONE_RATE)
    baseline_hr = float(np.clip(rng.normal(55, 5), 42, 70))
    base_load = POSITION_BASE_LOAD[position] / 7.0

    # Mutable athlete state, updated as the period unfolds.
    previous_injuries = int(rng.poisson(2 if injury_prone else 0.6))
    days_since_injury = float(rng.integers(60, 400))
    days_out = 0  # remaining days of the current absence

    load_history: deque[float] = deque(maxlen=CHRONIC_WINDOW)
    records = []

    for day in range(n_days):
        weekday = day % 7
        season_phase = np.sin(2 * np.pi * day / 365.0)
        fitness = float(np.clip(base_fitness + 0.05 * season_phase, 0.2, 1.0))
        currently_injured = days_out > 0

        if currently_injured:
            # Sidelined: rehab only, elevated soreness, no exposure to a new injury.
            load = float(max(rng.normal(base_load * 0.15, 8), 0.0))
            soreness = float(np.clip(rng.normal(5.5, 1.2), 0, 10))
            sleep_hours = float(np.clip(rng.normal(7.2, 1.0), 3.5, 10.0))
            resting_hr = float(np.clip(baseline_hr + rng.normal(2, 3), 40, 95))
        else:
            if weekday == 6:  # Sunday: rest
                load = rng.normal(base_load * 0.25, 15)
            elif weekday in (2, 3):  # mid-week peaks
                load = rng.normal(base_load * 1.4, 40)
            else:
                load = rng.normal(base_load, 35)
            if rng.random() < 0.04:  # occasional overload (double session, camp)
                load *= rng.uniform(1.5, 2.2)
            load = float(max(load, 0.0))

            resting_hr = float(np.clip(baseline_hr + (1 - fitness) * 8 + rng.normal(0, 3), 40, 95))
            sleep_hours = float(np.clip(rng.normal(7.4, 1.0), 3.5, 10.0))
            soreness = float(np.clip(rng.normal(3, 1.4) + (load / base_load - 1) * 2.5, 0, 10))

        load_history.append(load)
        acute = float(np.mean(list(load_history)[-ACUTE_WINDOW:]))
        chronic = float(np.mean(load_history))
        acwr = acute / chronic if chronic > 0 else 1.0

        # The rules drive the *hazard*; they are no longer the label.
        latent_risk = composite_risk_score(
            acwr=acwr,
            soreness=soreness,
            sleep_hours=sleep_hours,
            resting_hr=resting_hr,
            baseline_hr=baseline_hr,
            injury_prone=injury_prone,
            previous_injuries=previous_injuries,
            days_since_injury=days_since_injury,
        )

        # An athlete already sidelined cannot suffer a *new* injury today.
        injury_today = (not currently_injured) and bool(rng.random() < daily_hazard(latent_risk))

        records.append(
            {
                "athlete_id": athlete_id,
                "day": day,
                "position": position,
                "age": age,
                "base_fitness": round(base_fitness, 3),
                "injury_prone": injury_prone,
                "previous_injuries": previous_injuries,
                "baseline_hr": round(baseline_hr, 1),
                "days_since_injury": days_since_injury,
                "training_load": round(load, 1),
                "resting_hr": round(resting_hr, 1),
                "sleep_hours": round(sleep_hours, 2),
                "soreness": round(soreness, 2),
                "is_injured": currently_injured,
                "injury_onset": injury_today,
                "latent_risk": round(latent_risk, 4),
            }
        )

        # --- advance the state to tomorrow ---
        if injury_today:
            days_out = int(rng.integers(*INJURY_RECOVERY_DAYS))
            previous_injuries += 1
            days_since_injury = 0.0
        elif currently_injured:
            days_out -= 1
            days_since_injury = 0.0 if days_out > 0 else 1.0
        else:
            days_since_injury += 1

    return pd.DataFrame(records)


def add_target(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON_DAYS) -> pd.DataFrame:
    """Add the model target: an injury **starts** within the next ``horizon`` days.

    Strictly forward-looking and strictly exclusive of today, so nothing about day
    ``t`` can leak into its own label.

    ``horizon_complete`` marks the rows whose horizon actually fits inside the
    simulated period: the final days of each series are censored (we cannot know what
    happens after the last simulated day) and must be dropped before modelling.
    """
    df = df.sort_values(["athlete_id", "day"]).reset_index(drop=True)

    def _future_onsets(s: pd.Series) -> pd.Series:
        # Onsets strictly after today, within the horizon: shift by one day, then
        # look forward by reversing the series around a backward-looking window.
        ahead = s.shift(-1).fillna(0)
        return ahead[::-1].rolling(horizon, min_periods=1).sum()[::-1]

    future = df.groupby("athlete_id")["injury_onset"].transform(_future_onsets)
    df[TARGET_COL] = (future > 0).astype(int)

    last_day = df.groupby("athlete_id")["day"].transform("max")
    df["horizon_complete"] = df["day"] <= (last_day - horizon)
    return df


def generate(
    n_athletes: int = N_ATHLETES, n_days: int = N_DAYS, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """Generate the full synthetic dataset, injuries and target included."""
    rng = np.random.default_rng(seed)
    frames = [_simulate_athlete(aid, n_days, rng) for aid in range(n_athletes)]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(df["day"], unit="D")
    return add_target(df)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic dataset.")
    parser.add_argument("--athletes", type=int, default=N_ATHLETES)
    parser.add_argument("--days", type=int, default=N_DAYS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = generate(args.athletes, args.days, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    injuries = int(df["injury_onset"].sum())
    per_athlete = injuries / df["athlete_id"].nunique()
    modelled = df[~df["is_injured"] & df["horizon_complete"]]

    print(f"Synthetic dataset: {len(df)} rows, {df['athlete_id'].nunique()} athletes")
    print(f"Injury events    : {injuries} ({per_athlete:.1f} per athlete over the period)")
    print(f"Days sidelined   : {df['is_injured'].mean():.1%} of all athlete-days")
    print(f"Target '{TARGET_COL}' : {modelled[TARGET_COL].mean():.2%} positive on modellable rows")
    print(f"Written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
