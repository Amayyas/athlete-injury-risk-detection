"""Realistic synthetic data generator.

None of the available real Kaggle datasets contains a **per-athlete daily time
series**, which is required to compute a meaningful ACWR and rolling features.
This generator therefore produces a demonstration dataset that is "deep" in time:

- 200 simulated athletes over 2 seasons (730 days);
- static characteristics (age, position, base fitness, injury proneness, history);
- daily measurements (training load, resting HR, sleep, soreness);
- composite risk score (cf. :mod:`injury_risk.features.engineering`) + noise, then
  discretization into 3 **deliberately imbalanced** classes (few "high"), which
  justifies the use of SMOTE downstream.

Usage:
    python -m injury_risk.data.generate_synthetic            # 200 athletes, 730 days
    python -m injury_risk.data.generate_synthetic --athletes 50 --days 365
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from injury_risk.config import (
    ACUTE_WINDOW,
    CHRONIC_WINDOW,
    DEFAULT_SEED,
    INJURY_PRONE_RATE,
    LABEL_NOISE_SIGMA,
    N_ATHLETES,
    N_DAYS,
    POSITION_BASE_LOAD,
    POSITIONS,
    SYNTHETIC_DATASET,
)
from injury_risk.features.risk_factors import composite_risk_score, risk_score_to_level

DEFAULT_OUTPUT = SYNTHETIC_DATASET


def _simulate_athlete(athlete_id: int, n_days: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate the daily series of one athlete."""
    position = rng.choice(POSITIONS)
    age = int(np.clip(rng.normal(24, 4), 17, 38))
    base_fitness = float(np.clip(rng.normal(0.7, 0.12), 0.3, 1.0))
    injury_prone = bool(rng.random() < INJURY_PRONE_RATE)
    previous_injuries = int(rng.poisson(2 if injury_prone else 0.6))
    baseline_hr = float(np.clip(rng.normal(55, 5), 42, 70))

    base_load = POSITION_BASE_LOAD[position] / 7.0  # base daily load

    # Counter of days since the last injury (evolves over time).
    days_since_injury = float(rng.integers(60, 400))

    records = []
    # Weekly seasonality: rest on Sunday, peaks mid-week.
    for day in range(n_days):
        weekday = day % 7
        season_phase = np.sin(2 * np.pi * day / 365.0)  # fitness shape over the year

        # --- Training load (RPE x duration), noisy ---
        if weekday == 6:  # Sunday: rest / recovery
            load = rng.normal(base_load * 0.25, 15)
        elif weekday in (2, 3):  # mid-week peaks
            load = rng.normal(base_load * 1.4, 40)
        else:
            load = rng.normal(base_load, 35)
        # Occasional overloads (pre-season, double session) -> future high ACWR
        if rng.random() < 0.04:
            load *= rng.uniform(1.5, 2.2)
        load = float(max(load, 0.0))

        # --- Physiological measurements ---
        fitness = np.clip(base_fitness + 0.05 * season_phase, 0.2, 1.0)
        resting_hr = float(np.clip(baseline_hr + (1 - fitness) * 8 + rng.normal(0, 3), 40, 95))
        sleep_hours = float(np.clip(rng.normal(7.4, 1.0), 3.5, 10.0))
        # Soreness rises with recent load.
        soreness = float(np.clip(rng.normal(3, 1.4) + (load / base_load - 1) * 2.5, 0, 10))

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
            }
        )

        days_since_injury += 1

    return pd.DataFrame(records)


def _label_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Compute the composite risk score (+ noise) and the risk level.

    The ACWR is not available here yet; it is approximated by
    ``acute_load / chronic_load`` computed on the fly. The full feature
    engineering (cf. build_features) recomputes a clean ACWR on the training side.
    """
    df = df.sort_values(["athlete_id", "day"]).reset_index(drop=True)

    acute = df.groupby("athlete_id")["training_load"].transform(
        lambda s: s.rolling(ACUTE_WINDOW, min_periods=1).mean()
    )
    chronic = df.groupby("athlete_id")["training_load"].transform(
        lambda s: s.rolling(CHRONIC_WINDOW, min_periods=1).mean()
    )
    df["acwr_raw"] = (acute / chronic.replace(0, np.nan)).fillna(1.0)

    scores: list[float] = []
    for row in df.itertuples(index=False):
        s = composite_risk_score(
            acwr=row.acwr_raw,
            soreness=row.soreness,
            sleep_hours=row.sleep_hours,
            resting_hr=row.resting_hr,
            baseline_hr=row.baseline_hr,
            injury_prone=row.injury_prone,
            previous_injuries=row.previous_injuries,
            days_since_injury=row.days_since_injury,
        )
        scores.append(s)

    raw_scores = np.asarray(scores)
    # Gaussian noise to avoid a perfect decision boundary (realism).
    noisy = np.clip(raw_scores + rng.normal(0, LABEL_NOISE_SIGMA, size=len(raw_scores)), 0, 1)
    df["risk_score"] = noisy.round(4)
    # Default thresholds (cf. risk_score_to_level), calibrated to obtain a
    # realistic imbalance (~70% low / ~22% moderate / ~8% high), shared with the
    # dashboard for consistency.
    df["risk_level"] = [risk_score_to_level(s) for s in noisy]
    return df.drop(columns=["acwr_raw"])


def generate(
    n_athletes: int = N_ATHLETES, n_days: int = N_DAYS, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """Generate the full synthetic dataset."""
    rng = np.random.default_rng(seed)
    frames = [_simulate_athlete(aid, n_days, rng) for aid in range(n_athletes)]
    df = pd.concat(frames, ignore_index=True)
    # Calendar date (useful for EDA and the dashboard).
    df["date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(df["day"], unit="D")
    df = _label_rows(df, rng)
    return df


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

    dist = df["risk_level"].value_counts(normalize=True).sort_index().round(3)
    print(f"Synthetic dataset: {df.shape[0]} rows, {df['athlete_id'].nunique()} athletes")
    print(f"risk_level distribution (0=low,1=moderate,2=high):\n{dist.to_string()}")
    print(f"Written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
