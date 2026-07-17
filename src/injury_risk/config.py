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

# --- Injury process -------------------------------------------------------- #
# Injuries are *events*, drawn day by day from a **discrete-time logistic hazard**
# driven by the latent risk (the rule-based composite score):
#
#   logit( P(injury on day t) ) = HAZARD_INTERCEPT + HAZARD_SLOPE * latent_risk(t)
#
# The label is therefore an observed outcome, not a deterministic function of the
# features — see MODEL TARGET below.
#
# The two parameters are deliberately independent, and that is the whole point:
#   - the INTERCEPT sets the *base rate*    -> ~1.3 injuries per athlete per season;
#   - the SLOPE sets the *contrast*         -> how much risk actually predicts injury.
# A purely multiplicative hazard could not separate the two: keeping a realistic
# injury rate forced a high base, so most injuries fell on low-risk athlete-days and
# the signal drowned (an oracle knowing the latent risk only reached ROC-AUC 0.57 —
# i.e. the task was unlearnable *by construction*).
#
# Calibrated so that an oracle with perfect knowledge of the latent risk reaches
# ROC-AUC ~0.80 / PR-AUC ~0.23. That is the ceiling of this dataset, and it matches
# the order of magnitude reported by the real injury-prediction literature: a real
# signal, far from perfect, with a large irreducible random component.
HAZARD_INTERCEPT = -7.4
HAZARD_SLOPE = 8.0
INJURY_RECOVERY_DAYS = (7, 45)  # sidelined for this many days (uniform draw)

# --- Model target ---------------------------------------------------------- #
# "Will this athlete get injured within the next N days?", predicted from what is
# known up to (and including) day t. This is what makes the task genuinely
# predictive rather than a re-fit of the scoring rules.
PREDICTION_HORIZON_DAYS = 7
TARGET_COL = "injury_next_7d"

# --------------------------------------------------------------------------- #
# Training & evaluation
# --------------------------------------------------------------------------- #

CV_N_SPLITS = 5

# Days kept per athlete. Consecutive days are highly autocorrelated, so we thin
# the series rather than feeding 730 near-duplicate rows per athlete.
# (This thins autocorrelation; it does NOT prevent athlete leakage — grouped CV
# does, see injury_risk.models.splits.)
SAMPLE_PER_ATHLETE = 100

# Both tracks are binary, and their positive class is rare (~4% of athlete-days for
# the synthetic injury target). ROC-AUC is over-optimistic under that kind of
# imbalance, so **average precision (PR-AUC) is the headline metric**; ROC-AUC is
# kept for comparability. Recall stays a priority: in a medical context, missing an
# injury costs more than raising a false alarm.
SCORING = {
    "average_precision": "average_precision",  # PR-AUC — the metric that matters here
    "roc_auc": "roc_auc",
    "recall": "recall",
    "f1": "f1",
}

XGB_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}

# The metric hyperparameter search optimises. PR-AUC ranks the *whole* probability
# ordering, which is what we then threshold — optimising recall directly would just
# reward a model that flags everybody.
TUNING_METRIC = "average_precision"
TUNING_N_ITER = 40

# --------------------------------------------------------------------------- #
# Decision threshold: the asymmetric cost of being wrong
# --------------------------------------------------------------------------- #

# The README has claimed a "recall-first" posture from day one, while the code
# happily predicted at 0.5 — the default of every classifier, which implicitly says a
# missed injury and a false alarm cost the same. They do not.
#
# A false negative is a **missed injury**: the athlete trains through it and the
# muscle tears — weeks out, and a medical failure.
# A false positive is a **needless caution**: one session adapted for someone who was
# fine. Mildly annoying at worst.
#
# COST_RATIO is how many false alarms one missed injury is worth. It is a *business*
# statement, not a tuning knob: change it and the operating point moves accordingly.
# 10 is deliberately conservative — the ratio is arguably higher in a professional
# setting, but a staff that gets flooded with alerts stops reading them.
COST_FALSE_NEGATIVE = 10.0
COST_FALSE_POSITIVE = 1.0

# Never trust a threshold picked on the data the model was fitted on.
THRESHOLD_SEARCH_POINTS = 200
