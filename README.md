# 🩺 Athlete Injury Risk Detection

[![CI](https://github.com/Amayyas/athlete-injury-risk-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Amayyas/athlete-injury-risk-detection/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Detecting **muscle injury risk** in athletes from training metrics (workload, RPE)
and physiological signals (sleep, heart rate, soreness), with **full explainability
via SHAP** so the output is usable by a sports medical staff — not just a black box.

![Streamlit dashboard preview](docs/dashboard.png)

> *Streamlit dashboard: risk score computed in real time (here an athlete in the
> danger zone), active risk factors and built-in SHAP explainability.*

---

## 🎯 Goal & design choices

Answer the question a medical staff actually asks:

> **Is this athlete about to get injured in the next 7 days?**

grounded in the real domain knowledge used by strength & conditioning staff. The
project rests on 3 pillars:

1. **Real domain knowledge** — use of the **ACWR** (*Acute:Chronic Workload
   Ratio*), a metric actually used by staff: optimal zone **0.8–1.3**, danger
   zone **> 1.5**.
2. **Explainability (SHAP)** over raw *accuracy* — in a medical context,
   **recall** (not missing an injury) matters more than overall precision.
3. **A visual, interactive deliverable** — a **Streamlit** dashboard, not just a notebook.

Alongside the model, a rule-based **composite risk score** (low / moderate / high)
powers the dashboard's live assessment — and doubles as the **baseline the model has
to beat**.

---

## 🗃️ Data: transparency & limitations

An **audit of the 4 candidate Kaggle datasets** was done before any processing:

| Dataset | Size | Target | Balance | Temporal? | Kept |
|---|---|---|---|---|---|
| `mrsimple07` injury-prediction | 1000 × 7 | binary | 50/50 (suspicious) | ❌ snapshot | no |
| university-football-injury | 800 × 19 | binary | 50/50 (suspicious) | ❌ snapshot | no |
| EPL player-injuries-impact | 656 × 42 | no risk target | — | match history | no |
| **SIRP-600** | **600 × 16** | binary | **68/32 (natural)** | ❌ snapshot | ✅ |

### ⚠️ Key finding

**No available real dataset contains a per-athlete daily time series.** Yet ACWR
and the 7/14/28-day *rolling features* need one. Hence a deliberate **dual-track**
strategy:

- **🧪 Synthetic track (primary)** — a generator simulates 200 athletes × 730 days
  with **actual injury events** (see below). The only basis allowing a realistic
  temporal ACWR, and a genuinely predictive task.
- **🌍 Real SIRP-600 track (validation)** — demonstrates the same approach on
  **real, imperfect data** that is naturally imbalanced. Limitations: *snapshot*
  (no ACWR possible) and a target that is a risk *label*, not an observed injury.

---

## 🎯 The target: a real predictive task

**This is the most important design decision in the project, and it was originally
wrong.**

The first version of the generator labelled each day by discretising the rule-based
`composite_risk_score` **of that same day**, then trained the model on the very
variables that score was computed from. The model was not predicting injuries — it
was **re-learning the scoring function**. Its accuracy measured how well XGBoost can
imitate a formula. The whole result was circular.

### What it does now

The rules no longer *are* the label; they drive a **discrete-time logistic hazard**:

```
logit( P(injury on day t) ) = intercept + slope · latent_risk(t)
```

Each day, an athlete may actually get injured. An injured athlete is **sidelined**
for a recovery period, comes back with `previous_injuries` incremented and
`days_since_injury` reset — which feeds back into their future risk, exactly as in
real life.

The target becomes an **observed outcome**:

> **`injury_next_7d`** — will this athlete get injured within the next 7 days,
> given everything known up to (and including) today?

| | before | now |
|---|---|---|
| Target | rule score of the same day | an **observed injury event** |
| Relationship to features | deterministic (+ noise) | **stochastic** |
| What the model learns | the formula it was given | a real, noisy signal |
| Positive rate | 8% (by threshold choice) | **~5%** (by simulation) |

Injuries land at **~1.3 per athlete per season**, athletes are sidelined ~13% of
days, and the target is positive on ~5% of modellable athlete-days.

### Calibrating the hazard honestly

The intercept and the slope are separated on purpose: the **intercept sets the base
rate**, the **slope sets how much risk actually predicts injury**. A naive
multiplicative hazard cannot separate the two — keeping a realistic injury rate
forces a high base, so most injuries fall on low-risk days and the signal drowns. In
that first attempt, even a model that *knew the true latent risk* only reached
ROC-AUC 0.57: **the task was unlearnable by construction**, and no algorithm could
have saved it.

The slope is now calibrated so the signal is real but far from perfect — which is
what injury prediction actually looks like.

**Rows that cannot be modelled are dropped**, not silently kept: the ACWR warm-up,
days the athlete is *already* injured (they are not exposed to a *new* injury), and
the censored tail whose 7-day horizon runs past the end of the simulation.

**Limitations to keep in mind:**
- The synthetic hazard is **logit-linear in the latent risk by construction**. That
  is a strong hint the data-generating process gives to linear models — and the
  benchmark below shows exactly that. It is a property of the simulation, not
  evidence about injury prediction in the real world.
- SIRP-600 yields very high scores (roc_auc ≈ 0.96), suggesting strongly separable
  features (dataset possibly partly generated) — flagged as a limitation rather than
  oversold. Its target is a risk *label*, not an observed injury.

---

## 🧠 ML pipeline

1. **Feature engineering** (`src/injury_risk/features/engineering.py`): 7/14/28-day rolling
   (workload, soreness, sleep), **ACWR + zone** (under/optimal/elevated/danger),
   **workload trend** over 7 days, position encoding.
2. **Modeling** (`src/injury_risk/models/train.py`): `XGBoostClassifier` + **SMOTE**,
   **5-fold cross-validation grouped by athlete** (see below). The positive class is
   rare (~5%), so **PR-AUC (average precision) is the headline metric** — ROC-AUC is
   over-optimistic under that kind of imbalance.
3. **Explainability** (`src/injury_risk/visualization/shap_plots.py`): `TreeExplainer`,
   global **summary plot** + per-athlete **waterfall plot**.
4. **Dashboard** (`dashboard/app.py`): live score, colored gauge, active risk
   factors, SHAP plots.

### ⚠️ Two leakage guarantees, both tested

**No athlete spans a split.** The synthetic dataset holds many daily rows per
athlete; splitting them randomly would put the same athlete in both the training and
the validation fold, letting the model memorise their baseline profile and recognise
them at test time. The synthetic track therefore uses **`StratifiedGroupKFold` grouped
on `athlete_id`** (200 groups); the hold-out split is grouped too. The real SIRP-600
track is a snapshot (one row = one athlete), so grouping is meaningless there.

**No feature sees the future.** Every feature at day *t* is computed from data up to
and including *t*; the target looks strictly forward. A test corrupts every day after
a cutoff and asserts that not one feature before it moves.

And the hazard driver (`latent_risk`) is **never** a feature — otherwise the target
would be circular again.

### Results (seed 42, grouped CV)

A PR-AUC means nothing without knowing what chance and what the domain rules already
achieve on the same rows. So the model is reported **between two reference points**:

**Synthetic track — "injury within 7 days", 4.9% positive**

| | PR-AUC | ROC-AUC | Recall |
|---|---|---|---|
| Chance (prevalence) | 0.049 | 0.500 | — |
| **Domain rules** (`composite_risk_score`) | 0.287 | 0.830 | — |
| XGBoost + SMOTE | 0.291 ± 0.072 | 0.799 ± 0.015 | 0.599 |
| **Logistic Regression + SMOTE** | **0.367 ± 0.034** | **0.841 ± 0.011** | **0.697** |

**Real SIRP-600 track — binary risk label, 31.5% positive**

| | PR-AUC | ROC-AUC | Recall |
|---|---|---|---|
| Chance (prevalence) | 0.315 | 0.500 | — |
| XGBoost + SMOTE | 0.919 ± 0.081 | 0.959 ± 0.025 | 0.931 |
| **Random Forest + SMOTE** | **0.934 ± 0.077** | **0.960 ± 0.026** | **0.947** |

### 🔎 An uncomfortable result, reported rather than buried

**On the synthetic track, logistic regression beats XGBoost — clearly.** PR-AUC
0.367 vs 0.291, ROC-AUC 0.841 vs 0.799 (a gap of ~3 standard deviations: not noise).
It also beats the domain rules by **+28% PR-AUC**, where XGBoost barely matches them
(+1%).

The reason is not mysterious, and it is a limitation of the *simulation*, not a
finding about injury prediction: the hazard is **logit-linear in the latent risk by
construction**, and that latent risk is itself close to additive in the features. The
logistic model is therefore nearly *correctly specified*. Gradient boosting spends its
capacity fitting noise around a relationship a linear model captures exactly.

On the **real** dataset, the ranking flips back: tree models win (RF 0.934 / XGBoost
0.919 vs LogReg 0.754). Which is the honest summary — **the best model depends on the
data, and reaching for XGBoost by default is a reflex, not a decision.**

> Model selection is deliberately *not* settled in this PR: tuning and calibration
> ([#16](https://github.com/Amayyas/athlete-injury-risk-detection/issues/16)) may
> still change the picture, and the delivered artefact should be the winner of a
> tuned comparison, not of an untuned one.

---

## 🧰 Tech stack

`Python 3.12` · `pandas` · `numpy` · `scikit-learn` · `xgboost` ·
`imbalanced-learn (SMOTE)` · `shap` · `streamlit` · `matplotlib` · `seaborn` ·
`pytest` · `black` · `ruff`

---

## 📁 Structure

The project is a proper installable package (`src` layout):

```
athlete-injury-risk-detection/
├── data/
│   ├── raw/                  # raw Kaggle datasets (gitignored)
│   └── processed/            # generated synthetic dataset (gitignored)
├── src/
│   └── injury_risk/          # the installable package
│       ├── config.py         # single source of truth (paths, thresholds, weights, CV)
│       ├── data/             # datasets.py (public API), generate_synthetic.py, load_dataset.py
│       ├── features/         # engineering.py (ACWR, rolling) + risk_factors.py (rule scoring)
│       ├── models/           # train.py, tune.py, benchmark.py, splits.py (grouped CV)
│       └── visualization/    # shap_plots.py
├── dashboard/                # app.py (Streamlit)
├── tests/                    # pytest tests
├── notebooks/                # 01_eda.ipynb
├── reports/figures/          # generated SHAP plots (gitignored)
├── models/                   # trained .joblib models (gitignored)
├── docs/                     # dashboard screenshot
├── pyproject.toml            # metadata, dependencies, tooling config
├── requirements.txt          # thin pointer to pyproject
├── LICENSE
└── README.md
```

---

## 🚀 Getting started

Everything hangs off **one command**, `injury-risk` (`make help` lists the shortcuts):

```bash
# 1) Environment — installs the `injury_risk` package in editable mode
python3.12 -m venv .venv && source .venv/bin/activate
make setup                    # == pip install -e ".[dev]"

# 2) The whole pipeline: data -> train -> SHAP plots
make pipeline

# 3) Launch the dashboard (works even without a trained model)
make run
```

Or stage by stage:

```bash
injury-risk --help                          # every command, self-documented
injury-risk download sirp-600               # real datasets (needs a Kaggle token)
injury-risk data                            # generate 200 athletes x 730 days
injury-risk benchmark                       # LogReg / RandomForest / XGBoost
injury-risk tune --track synthetic          # hyperparameter search
injury-risk train --tuned                   # train and write the metrics report
injury-risk shap --track synthetic          # explainability plots
injury-risk dashboard                       # Streamlit app
```

### Reproducible installs

`pyproject.toml` is the single source of truth for dependencies; `requirements.lock`
and `requirements-dev.lock` pin the exact resolved versions (regenerate with
`make lock`).

```bash
pip install -c requirements.lock -e .       # the exact, reproducible versions
```

CI installs **pinned on Python 3.12** — that is the reproducible build the Docker
image and the deployment will use — and **unpinned on 3.13**, on purpose: that leg is
the canary that catches an upstream release breaking us *before* we bump the lock.

### Code quality & tests

```bash
make check        # everything CI runs: lint + types + tests
make test         # tests with their coverage floor
make lint         # ruff + mypy
make format       # black + ruff --fix
```

All four run on every push and pull request via [GitHub Actions](.github/workflows/ci.yml),
on Python 3.12 and 3.13.

---

## 🔍 Dashboard details

The dashboard computes a **real-time risk score** from the business rules (ACWR,
sleep, soreness, HR, injury history) — so it stays functional **even before the
model is trained**. Once the SHAP plots are generated, the explainability section
shows up automatically.

**The "active risk factors" list is the score's own decomposition.** Each factor
carries the number of risk points it contributes, and the score is their sum. A
contribution therefore cannot exist without its factor being displayed — the gauge
and its explanation are mathematically unable to contradict each other. All of it
lives in `injury_risk.features.risk_factors`, where it is unit-tested; the Streamlit
file only renders.

---

## 📌 Possible improvements

- Obtain a real daily GPS/workload dataset (Catapult, StatSports) for a true ACWR.
- Probability calibration + recall-oriented threshold (asymmetric cost).
- Per-athlete temporal tracking in the dashboard (ACWR curve over the season).
- Hyperparameter tuning (Optuna) and baseline comparison (LogReg, RandomForest).

> Tracked as [GitHub issues](https://github.com/Amayyas/athlete-injury-risk-detection/issues).

---

## 📄 License

Released under the [MIT License](LICENSE).
```
