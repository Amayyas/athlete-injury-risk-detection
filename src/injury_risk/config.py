"""Single source of truth for paths, domain constants and training settings.

Everything that used to be a magic number scattered across the codebase lives
here: the ACWR windows, the risk thresholds, the composite-score weights, the
generator settings and the cross-validation setup.

Deliberately kept as plain module constants: on a project this size, a YAML /
Hydra / pydantic-settings layer would be machinery for its own sake. The point is
that a value is defined **once** and changing it here changes it everywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# config.py lives at src/injury_risk/config.py -> the project root is 3 levels up.
ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SYNTHETIC_DATASET = PROCESSED_DIR / "synthetic_athletes.parquet"
SIRP_PATH = RAW_DIR / "sirp-600" / "High_Accuracy_Sport_Injury_Dataset.xlsx"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

DEFAULT_SEED = 42

# --------------------------------------------------------------------------- #
# Domain: workload windows & ACWR
# --------------------------------------------------------------------------- #

ACUTE_WINDOW = 7  # days — "acute" workload
CHRONIC_WINDOW = 28  # days — "chronic" workload
ROLLING_WINDOWS = (7, 14, 28)
LOAD_TREND_WINDOW = 7

# The chronic load is not meaningful before a full chronic window has elapsed, so
# the first days of each athlete's series are dropped before modelling.
WARMUP_DAYS = CHRONIC_WINDOW

# ACWR zone bounds, from the training-load literature.
ACWR_ZONES = {
    "under": (0.0, 0.8),  # under-training
    "optimal": (0.8, 1.3),  # "sweet spot"
    "elevated": (1.3, 1.5),  # rising load, watch closely
    "danger": (1.5, np.inf),  # high-risk zone
}
ACWR_DANGER = 1.5
ACWR_ELEVATED = 1.3
ACWR_UNDER = 0.8

# --------------------------------------------------------------------------- #
# Domain: positions
# --------------------------------------------------------------------------- #

POSITIONS = ("goalkeeper", "defender", "midfielder", "forward")
POSITION_TO_CODE = {pos: i for i, pos in enumerate(POSITIONS)}

# Average weekly workload per position (intensity proxy, used by the generator).
POSITION_BASE_LOAD = {
    "goalkeeper": 320.0,
    "defender": 420.0,
    "midfielder": 520.0,  # midfielders cover the most distance
    "forward": 470.0,
}

# --------------------------------------------------------------------------- #
# Domain: composite risk score (business rules)
# --------------------------------------------------------------------------- #

# Additive, bounded contributions — kept readable on purpose, since this function
# both generates the synthetic labels and powers the dashboard's live score.
W_ACWR_DANGER = 0.35  # ACWR >= 1.5
W_ACWR_ELEVATED = 0.18  # 1.3 <= ACWR < 1.5
W_ACWR_UNDER = 0.08  # ACWR < 0.8 (detraining)
W_SORENESS = 0.20
W_SLEEP = 0.15
W_RESTING_HR = 0.12
W_INJURY_PRONE = 0.10
W_PREVIOUS_INJURIES = 0.08
W_RECENT_RETURN = 0.10

# Normalisation ranges for the graded contributions.
SORENESS_ONSET = 4.0  # soreness starts counting above this (0-10 scale)
SORENESS_RANGE = 6.0
SLEEP_TARGET = 7.0  # hours; below this, sleep starts counting
SLEEP_RANGE = 4.0
HR_DELTA_RANGE = 20.0  # bpm above baseline for a full contribution
PREVIOUS_INJURIES_RANGE = 5.0
RECENT_RETURN_DAYS = 60  # returning from injury within this window is a risk

# Risk-level thresholds on the continuous score.
# These are *calibrated*, not arbitrary: they were chosen so the synthetic dataset
# lands on a realistic, imbalanced distribution (~70% low / ~22% moderate / ~8%
# high), which is what justifies using SMOTE downstream.
RISK_LOW_THRESHOLD = 0.16
RISK_HIGH_THRESHOLD = 0.27

RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High"}

# --------------------------------------------------------------------------- #
# Synthetic data generator
# --------------------------------------------------------------------------- #

N_ATHLETES = 200
N_DAYS = 730  # two seasons
INJURY_PRONE_RATE = 0.30  # share of athletes flagged as injury-prone
LABEL_NOISE_SIGMA = 0.06  # Gaussian noise on the score, to avoid a perfect boundary

# --------------------------------------------------------------------------- #
# Training & evaluation
# --------------------------------------------------------------------------- #

CV_N_SPLITS = 5

# Days kept per athlete. Consecutive days are highly autocorrelated, so we thin
# the series rather than feeding 730 near-duplicate rows per athlete.
# (This thins autocorrelation; it does NOT prevent athlete leakage — grouped CV
# does, see injury_risk.models.splits.)
SAMPLE_PER_ATHLETE = 40

# Recall-oriented metrics: in a medical context, missing an injury costs more than
# raising a false alarm.
SCORING = {
    "f1_macro": "f1_macro",
    "recall_macro": "recall_macro",
    "roc_auc": "roc_auc_ovr_weighted",
}

XGB_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}
