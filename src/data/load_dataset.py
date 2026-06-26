"""Loading & cleaning of the real SIRP-600 dataset.

**SIRP-600** ("Sports Injury Risk Prediction", 600 athletes) was retained among
the 4 audited Kaggle candidates because it is the only one that:

- is explicitly dedicated to *risk scoring* (``Injury_Risk`` target);
- is **naturally imbalanced** (~68% low risk / 32% risk) — which justifies the
  use of SMOTE, unlike the `mrsimple07` and `university-football` datasets that
  are perfectly balanced 50/50 (an artificial balance flagged as suspicious in
  the literature).

Accepted limitations (see README): no temporal dimension (snapshot, one row =
one athlete), so **no real ACWR possible** on this dataset; binary target (not
3 classes). It serves as a "real, imperfect data" validation track alongside the
temporal synthetic dataset.

Usage:
    from src.data.load_dataset import load_sirp600
    df = load_sirp600()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SIRP_PATH = RAW_DIR / "sirp-600" / "High_Accuracy_Sport_Injury_Dataset.xlsx"

# Feature columns (everything but the target) for the "real data" track.
SIRP_TARGET = "Injury_Risk"
SIRP_FEATURE_COLS = [
    "Age",
    "Gender",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Training_Frequency",
    "Training_Duration",
    "Warmup_Time",
    "Sleep_Hours",
    "Flexibility_Score",
    "Muscle_Asymmetry",
    "Recovery_Time",
    "Injury_History",
    "Stress_Level",
    "Training_Intensity",
]


def load_sirp600(path: Path = SIRP_PATH) -> pd.DataFrame:
    """Load and clean SIRP-600.

    Cleaning: drop duplicates, validate types, clip obvious outliers. The dataset
    is already clean (no missing values), so we deliberately stay light to avoid
    masking its nature.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"SIRP-600 dataset not found: {path}\n"
            "Run first: python -m src.data.download sirp-600"
        )

    df = pd.read_excel(path)

    # Keep only the expected columns (robust if the order changes).
    missing = [c for c in SIRP_FEATURE_COLS + [SIRP_TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in SIRP-600: {missing}")
    df = df[SIRP_FEATURE_COLS + [SIRP_TARGET]].copy()

    # Light cleaning.
    df = df.drop_duplicates().reset_index(drop=True)
    df[SIRP_TARGET] = df[SIRP_TARGET].astype(int)

    return df


def class_balance(df: pd.DataFrame, target: str = SIRP_TARGET) -> pd.Series:
    """Target distribution (proportions) — useful to justify SMOTE."""
    return df[target].value_counts(normalize=True).sort_index().round(3)


if __name__ == "__main__":
    data = load_sirp600()
    print(f"SIRP-600 loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Target '{SIRP_TARGET}' (proportions):\n{class_balance(data).to_string()}")
