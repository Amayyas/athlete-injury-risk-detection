"""Compare the candidate models under one identical protocol.

Three families, all preceded by SMOTE, all evaluated with the same grouped
cross-validation and the same metrics:

- **Logistic Regression** — interpretable linear baseline;
- **Random Forest** — non-linear ensemble baseline;
- **XGBoost** — gradient boosting.

Ranked on ``average_precision`` (PR-AUC): with a ~5% positive class it is the metric
that discriminates, and recall alone would reward a model that flags everybody.

Pass ``tuned=True`` to compare the *tuned* versions. That distinction matters: a
tuned favourite against untuned baselines measures which model got attention, not
which model is better.
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import cross_validate

from injury_risk.config import DEFAULT_SEED, REPORTS_DIR, SCORING
from injury_risk.data.datasets import load_track
from injury_risk.models.baselines import reference_points
from injury_risk.models.candidates import CANDIDATES, build_candidate
from injury_risk.models.splits import make_cv
from injury_risk.models.tune import load_best_params


def benchmark_track(track: str, seed: int = DEFAULT_SEED, tuned: bool = False) -> pd.DataFrame:
    """Evaluate every candidate on a track and return a comparison table."""
    data = load_track(track, seed=seed)
    cv = make_cv(track, seed=seed)

    rows = []
    for name in CANDIDATES:
        params = load_best_params(track, name) if tuned else None
        pipe = build_candidate(name, data.n_classes, seed, params=params)
        res = cross_validate(
            pipe, data.X, data.y, groups=data.groups, cv=cv, scoring=SCORING, n_jobs=-1
        )
        rows.append(
            {
                "model": name,
                "tuned": bool(params),
                **{m: float(res[f"test_{m}"].mean()) for m in SCORING},
                **{f"{m}_std": float(res[f"test_{m}"].std()) for m in SCORING},
            }
        )

    table = pd.DataFrame(rows).set_index("model").round(4)
    table = table.sort_values("average_precision", ascending=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_tuned" if tuned else ""
    out = REPORTS_DIR / f"benchmark_{track}{suffix}.json"
    out.write_text(
        json.dumps(
            {
                "track": track,
                "tuned": tuned,
                "reference_points": reference_points(data),
                "models": table.reset_index().to_dict(orient="records"),
            },
            indent=2,
        )
    )

    label = "tuned" if tuned else "default params"
    print(f"\n=== Benchmark '{track}' ({len(data)} rows, {label}) ===")
    print(table[list(SCORING)].to_string())

    refs = reference_points(data)
    if "rule_score" in refs:
        print(
            f"  (domain rules: average_precision = {refs['rule_score']['average_precision']:.4f})"
        )
    print(f"-> Best average_precision (PR-AUC): {table.index[0]}")
    print(f"Saved: {out}")
    return table
