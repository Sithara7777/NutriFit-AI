# NutriFit-AI

**AI-Based Personalised Nutrition, Meal Recommendation and Weight Management System for Gym Users**

CSE6035 Development Project (WRIT1) · U.L.S.R. Perera · CL.BSCSD.34.69

---

## What this is

A full-stack web application that predicts a gym user's daily calorie and
protein requirements from their physiology and training, recommends meals that
fit those targets, and generates a structured eight-week meal plan that adapts
as the user's weight changes.

| Layer | Technology |
|---|---|
| Frontend | React 18 (Vite), Tailwind CSS v4, React Router, Recharts |
| Backend | Node.js 18+, Express, Supabase JS, Zod, Helmet |
| Auth & Database | Supabase (Postgres + Auth + Row Level Security) |
| ML service | Python 3.10+, FastAPI, scikit-learn, joblib |
| Training | Google Colab (CPU runtime) |

```
React SPA  ──HTTPS/JWT──▶  Node/Express API  ──internal HTTP──▶  FastAPI ML service
     │                            │                                      │
     └────Supabase Auth───────────┴──────── Supabase Postgres (RLS) ──────┘
```

The ML service is **never** exposed to the internet. Every call is brokered by
the Node API, which authenticates the Supabase JWT first.

---

## Repository layout

```
NutriFit-AI/
├── data/
│   ├── raw/               <- download the Kaggle CSVs here (see raw/README.md)
│   ├── processed/         <- generated: labelled dataset, food catalogue, seed SQL
│   └── external/          <- optional USDA FoodData Central bulk download
├── ml/
│   ├── nutrifit/          <- SHARED PACKAGE (notebooks + FastAPI both import this)
│   │   ├── nutrition.py     BMR/TDEE/protein formulas - single source of truth
│   │   ├── labels.py        supervised target construction
│   │   ├── preprocessing.py sklearn ColumnTransformer + Pipeline
│   │   ├── training.py      LR + RF, tuning, CV, learning curves
│   │   ├── recommender.py   content-based engine
│   │   └── planner.py       eight-week plan generator
│   ├── notebooks/         <- 01_EDA … 05_export (run these in Colab)
│   ├── scripts/           <- prepare_data.py, train.py, export_models.py
│   ├── tests/             <- 136 pytest tests
│   ├── artifacts/         <- trained .pkl files + metrics + model card
│   └── reports/           <- figures and CSV tables for the dissertation
├── services/ml-service/   <- FastAPI app (37 tests)
├── backend/               <- Node/Express API (37 tests)
├── frontend/              <- React SPA
├── db/schema.sql          <- Postgres schema + RLS policies
└── docs/COLAB_GUIDE.md    <- step-by-step training guide
```

**Why `ml/nutrifit/` is shared:** the Colab notebooks and the FastAPI service
import the *same* package. The preprocessing pipeline is fitted inside the same
`sklearn.Pipeline` as the estimator and exported as one `.pkl`, so inference
applies byte-identical transforms to training. There is no train/serve skew by
construction, not by discipline.

---

## Quick start

### 0. Prerequisites
Node ≥ 18, Python ≥ 3.10, a free Supabase project.

### 1. Get the data
See [`data/raw/README.md`](data/raw/README.md) for exact download links.

### 2. Prepare data and train

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r ml/requirements.txt

python ml/scripts/prepare_data.py     # add --demo to test without the download
python ml/scripts/train.py
python ml/scripts/export_models.py    # copies + VERIFIES artefacts into the service
```

> For the dissertation, run the **Colab notebooks** instead of `train.py` — they
> produce all the figures and tables. See [`docs/COLAB_GUIDE.md`](docs/COLAB_GUIDE.md).
> The whole project trains on Colab's **free CPU runtime**: 0 compute units.

### 3. Set up the database
Supabase Dashboard → SQL Editor → run:
1. `db/schema.sql`
2. `data/processed/foods_seed.sql` (generated in step 2)

### 4. Run the three services

```bash
# Terminal 1 - ML service
cd services/ml-service
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000        # docs at http://localhost:8000/docs

# Terminal 2 - backend
cd backend
cp .env.example .env      # fill in your Supabase keys
npm install && npm run dev                        # http://localhost:4000

# Terminal 3 - frontend
cd frontend
cp .env.example .env      # fill in your Supabase URL + anon key
npm install && npm run dev                        # http://localhost:5173
```

---

## Running the tests

```bash
cd ml                  && python -m pytest tests -q       # 149 passed
cd services/ml-service && python -m pytest tests -q       #  37 passed
cd backend             && npm test                        #  37 passed
cd frontend            && npm run build                   # production build
```

The strict ±5% meal-plan guarantee is asserted in `TestRealCatalogue`, which
runs against `data/processed/food_catalogue.csv` and **skips automatically**
until `prepare_data.py` has been run — so the suite passes on a clean checkout
without silently dropping the guarantee.

---

## Key design decisions

### Labels are constructed, not found
No public dataset contains "this person's correct calorie target for their
goal". Targets are built from published formulas — **Katch–McArdle** BMR where
body fat is measured (the right estimator for a trained population),
**Mifflin–St Jeor** otherwise, scaled by a FAO/WHO activity multiplier and a
goal-specific factor sampled within ISSN-published ranges, plus Gaussian
residual noise. Full rationale in [`ml/nutrifit/labels.py`](ml/nutrifit/labels.py).

The noise gives a **theoretical R² ceiling of ≈0.97**, which is reported
alongside the model R² so the remaining error can be identified as irreducible.

### Random Forest wins — but only once there is enough data
On the real 973-record dataset Random Forest outperforms Linear Regression on
both targets, as the proposal predicted:

| Target | Model | CV MAE | CV R² | Test R² |
|---|---|---|---|---|
| Calories | Linear Regression | 114.15 ± 5.17 | 0.9514 | 0.9564 |
| Calories | **Random Forest** | **96.26 ± 4.02** | **0.9636** | 0.9564 |
| Protein | Linear Regression | 8.86 ± 0.42 | 0.9572 | 0.9552 |
| Protein | **Random Forest** | **8.72 ± 0.49** | 0.9569 | **0.9590** |

The learning curve makes the *mechanism* visible, and it is the more
interesting result: Random Forest is **worse than the linear baseline at small
sample sizes** and only overtakes as the data grows. On the protein target the
two are still converging at n=973 (LR 8.86 vs RF 8.87), which is why the
calorie model — where RF is decisively ahead — shows the larger gain.

This is a genuine bias–variance demonstration rather than a formality: an
axis-aligned ensemble approximating a smooth physiological relationship needs
substantially more data than a correctly-specified linear model, and 973
records sits right at the crossover. Both models are trained and reported, and
the deployed model is **selected on cross-validated MAE**. Evidence:
`ml/reports/learning_curve_*.csv` and notebook 04 §3.

> Reproducibility note: an earlier run against *synthetic* smoke-test data
> showed the opposite ordering, because that data was generated from smooth
> Gaussian distributions with no measured body-composition signal. Only results
> from `data/raw/` are reportable — every script prints a warning when the
> synthetic fallback is active.

### Protein match is weighted above calorie match
The nearest-neighbour metric weights protein 2.5× against calories 1.0×. This
is not a tuning knob picked by feel: calories are trivially recoverable via the
portion multiplier, whereas protein is the binding constraint. Sweeping the
weight over the real catalogue across all three goals:

| Protein weight | fat_loss | maintenance | muscle_gain | worst |
|---|---|---|---|---|
| 1.0 (unweighted) | 8.10% ✗ | 4.72% | 8.47% ✗ | 8.47% |
| 2.0 | 5.00% | 2.53% | 3.74% | 5.00% |
| **2.5 (shipped)** | **3.80%** | **1.96%** | **2.80%** | **3.80%** |
| 3.0 | 4.05% | 2.46% | 1.03% | 4.05% |

This removed the greedy repair pass entirely (47 repairs → 0 on fat-loss), so
plans now meet the ±5% tolerance *by construction* rather than by correction.

### A meal slot is composed of several foods
A catalogue entry is a *food*, not a *meal* — the median item in the real
catalogue is ~130 kcal against a ~600 kcal breakfast budget. One item per slot
under-delivered by **26%**. Each slot is now filled greedily with 1–3 items plus
a quarter-step portion multiplier, bringing weekly delivery to **within ±0.6%**
on calories and **±2.4%** on protein.

### The source CSV needed repairing before it would parse
Six rows of the published food dataset contain unquoted commas inside the food
name (`Milk (2%, 1 cup),Dairy,122,...`), producing 13 fields against a 12-field
header — pandas aborts the entire read. `read_csv_resilient()` parses those rows
from the right, since the trailing fields are positional, and re-joins the
excess leading fields into the name. The repair count is recorded in
`data/processed/data_quality_report.json` as data-quality evidence rather than
silently hidden.

The same loader auto-detects whether the food source is a consumption *log* or
an already-aggregated *catalogue* — the published dataset ships in both forms,
and applying a "require ≥2 observations" filter to the latter would discard
every row.

### Security is enforced by the database
Row Level Security is enabled on every user-scoped table with policies keyed on
`auth.uid()`. Application queries use a Supabase client built from the *user's*
JWT, so Postgres itself enforces isolation — a forgotten `.eq('user_id', …)` in
a handler still returns nothing. The service-role key is used only for catalogue
seeding, never to serve a user request.

### The system prompts; it never overwrites
When a new weight moves the calculated targets by more than 7%, the user is
*asked* whether to regenerate their plan. A plan someone is part-way through
following is their data.

---

## Requirements traceability

| FR | Requirement | Implementation |
|---|---|---|
| FR1 | Registration & Authentication | Supabase Auth · `frontend/src/pages/Login.jsx` |
| FR2 | Profile Management | `backend/src/routes/profile.js` · `profiles` table |
| FR3 | BMI Calculation | `ml/nutrifit/nutrition.py` · `backend/src/utils/nutrition.js` |
| FR4 | Calorie Prediction | `ml/nutrifit/training.py` · `POST /api/predict` |
| FR5 | Protein Prediction | same pipeline, second target |
| FR6 | Meal Recommendation | `ml/nutrifit/recommender.py` · `GET /api/recommendations` |
| FR7 | Two-Month Meal Plan | `ml/nutrifit/planner.py` · `POST /api/mealplan/generate` |
| FR8 | Progress Monitoring | `backend/src/routes/progress.js` · `weight_logs` |
| FR9 | Recommendation Adjustment | `/drift-check` · prompt banner in `Progress.jsx` |
| FR10 | Dashboard & Reporting | `GET /api/dashboard` · `frontend/src/pages/Dashboard.jsx` |

---

## Documentation

- [`docs/COLAB_GUIDE.md`](docs/COLAB_GUIDE.md) — training, runtime management, compute-unit budget
- [`data/raw/README.md`](data/raw/README.md) — dataset links, licensing, ethics
- `ml/artifacts/model_card.json` — provenance, intended use, known limitations
- `ml/reports/` — figures and CSV tables for the dissertation

---

## Disclaimer

NutriFit-AI provides general nutritional guidance from established
sports-nutrition formulas and machine-learning estimates. It is **not** a
medical device and does not replace advice from a qualified doctor, dietitian or
nutritionist.
