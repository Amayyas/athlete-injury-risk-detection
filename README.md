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

Predict a risk level (**low / moderate / high**) grounded in the real domain
knowledge used by strength & conditioning staff. The project rests on 3 pillars:

1. **Real domain knowledge** — use of the **ACWR** (*Acute:Chronic Workload
   Ratio*), a metric actually used by staff: optimal zone **0.8–1.3**, danger
   zone **> 1.5**.
2. **Explainability (SHAP)** over raw *accuracy* — in a medical context,
   **recall** (not missing an injury) matters more than overall precision.
3. **A visual, interactive deliverable** — a **Streamlit** dashboard, not just a notebook.

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

- **🧪 Synthetic track (primary)** — a generator produces 200 athletes × 730
  days, the only basis allowing a realistic temporal ACWR/rolling, **3 imbalanced
  classes** (~70 / 22 / 8) that justify the use of SMOTE, and which feeds the
  dashboard. Labels = composite risk score (business rules) + Gaussian noise.
- **🌍 Real SIRP-600 track (validation)** — demonstrates the same approach
  (SMOTE + XGBoost + SHAP) on **real, imperfect data** that is naturally
  imbalanced. Limitations: *snapshot* (no ACWR possible) and a **binary** target.

**Limitations to keep in mind:**
- The synthetic dataset has labels derived from a rule function: the model partly
  learns that function (interpret performance with caution; the noise prevents a
  perfect decision boundary — f1_macro ≈ 0.57 in CV, realistic).
- SIRP-600 yields very high scores (roc_auc ≈ 0.96), suggesting strongly separable
  features (dataset possibly partly generated) — to be flagged as a limitation
  rather than oversold.

---

## 🧠 ML pipeline

1. **Feature engineering** (`src/injury_risk/features/engineering.py`): 7/14/28-day rolling
   (workload, soreness, sleep), **ACWR + zone** (under/optimal/elevated/danger),
   **workload trend** over 7 days, position encoding.
2. **Modeling** (`src/injury_risk/models/train.py`): `XGBoostClassifier` + **SMOTE**,
   **stratified 5-fold cross-validation**, recall-oriented metrics —
   `f1_macro`, `recall_macro`, `roc_auc_ovr_weighted`.
3. **Explainability** (`src/injury_risk/visualization/shap_plots.py`): `TreeExplainer`,
   global **summary plot** + per-athlete **waterfall plot**.
4. **Dashboard** (`dashboard/app.py`): live score, colored gauge, active risk
   factors, SHAP plots.

### Results (seed 42)

| Track | f1_macro | recall_macro | roc_auc | Top SHAP feature |
|---|---|---|---|---|
| Synthetic (3 classes) | ~0.57 | ~0.62 | ~0.81 | **ACWR** ✅ |
| Real SIRP-600 (binary) | ~0.92 | ~0.93 | ~0.96 | Training_Intensity / Recovery |

> The SHAP summary plot confirms the domain narrative: **ACWR is the most
> decisive variable** of the synthetic model, ahead of injury history and soreness.

### Baseline benchmark & model choice

Before settling on XGBoost, three model families were compared under the **same
protocol** (SMOTE + stratified 5-fold CV), sorted by `recall_macro` (business
priority). Reproducible via `python -m injury_risk.models.benchmark`.

**Synthetic track (3 classes)**

| Model | f1_macro | recall_macro | roc_auc |
|---|---|---|---|
| Random Forest | 0.565 ± 0.010 | **0.627 ± 0.013** | 0.798 ± 0.012 |
| **XGBoost** | **0.569 ± 0.015** | 0.621 ± 0.018 | **0.805 ± 0.009** |
| Logistic Regression | 0.480 ± 0.006 | 0.550 ± 0.009 | 0.726 ± 0.009 |

**Real SIRP-600 track (binary)**

| Model | f1_macro | recall_macro | roc_auc |
|---|---|---|---|
| Random Forest | **0.950 ± 0.029** | **0.954 ± 0.027** | **0.960 ± 0.026** |
| **XGBoost** | 0.923 ± 0.033 | 0.931 ± 0.030 | 0.959 ± 0.025 |
| Logistic Regression | 0.764 ± 0.033 | 0.783 ± 0.036 | 0.848 ± 0.042 |

**Honest reading:**
- Logistic regression clearly trails → a **non-linear** model is justified.
- Random Forest and XGBoost are **statistically indistinguishable**: on
  `recall_macro`, the gap (0.006 synthetic, 0.023 real) is **smaller than the
  cross-validation standard deviation** in both cases — the intervals overlap
  heavily. Declaring a "winner" on that basis would be over-interpreting noise.
- **XGBoost is the chosen model**, not because it "wins" but as a tie-breaker at
  equivalent performance: (1) it beats RF on `f1_macro` *and* `roc_auc` of the
  **synthetic track** (the project's primary track); (2) its regularization
  (`min_child_weight`, `gamma`, `subsample`) is better suited to recall-oriented
  tuning (`src/injury_risk/models/tune.py`); (3) it integrates natively with
  `shap.TreeExplainer` for explainability.

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
│       ├── data/             # download.py, load_dataset.py, generate_synthetic.py
│       ├── features/         # engineering.py (ACWR, rolling, composite score)
│       ├── models/           # train.py, tune.py, benchmark.py
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

```bash
# 1) Environment — installs the `injury_risk` package in editable mode
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # runtime + tests, lint, notebooks
# (or, runtime + dashboard only:  pip install -r requirements.txt)

# 2) (Optional) Download the real datasets — requires a Kaggle token
python -m injury_risk.data.download                 # everything
python -m injury_risk.data.download sirp-600        # a single one

# 3) Generate the synthetic dataset (200 athletes × 730 days)
python -m injury_risk.data.generate_synthetic

# 4) (Optional) Compare baselines (LogReg / RandomForest / XGBoost)
python -m injury_risk.models.benchmark              # or --track synthetic / --track real

# 5) (Optional) Tune XGBoost hyperparameters (recall-first)
python -m injury_risk.models.tune --track synthetic --n-iter 30

# 6) Train the models (synthetic + real)
python -m injury_risk.models.train                  # add --tuned to use tuned params

# 7) Generate the SHAP plots
python -m injury_risk.visualization.shap_plots --track synthetic

# 8) Launch the dashboard (works even without a trained model)
streamlit run dashboard/app.py
```

### Code quality & tests

```bash
pytest            # unit tests + coverage floor
ruff check .      # lint
black .           # formatting
mypy              # static type checking
```

All four run on every push and pull request via [GitHub Actions](.github/workflows/ci.yml),
on Python 3.12 and 3.13.

---

## 🔍 Dashboard details

The dashboard computes a **real-time risk score** from the business rules (ACWR,
sleep, soreness, HR, injury history) — so it stays functional **even before the
model is trained**. Once the SHAP plots are generated, the explainability section
shows up automatically.

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
