# Implementation Plan
## AI-Based Personalized Nutrition, Meal Recommendation & Weight Management System for Gym Users
**CSE6035 Development Project — Software Engineering Final Development Project (WRIT1, 80%)**

---

## 0. How this plan was built

This plan cross-checks three source documents you uploaded:

1. **`ICBT_CIS6035_S3SRI_WRIT1_May-2026_Main_2025-26.pdf`** — the assessment brief. Key facts pulled from it: this is a *Development Project* (CSE6035), 80% weight, 8000-word (equivalent) thesis, marked on **Achievement of Objectives (25%), Use of Literature (15%), Methodology (20%), Analysis/Discussion/Solution Design & Implementation (30%), Report Structure (10%)**. Learning outcomes explicitly say *"design and develop a project based on a software or hardware artefact"* — this is an artefact-centred module, not a pure research module. Milestones required: Proposal, Literature Review + SRS, Design Spec + Prototype, Final Thesis. Minimum 5 supervisor meetings with signed logs.
2. **`BSc_Hons_Literature_Review.pdf`** — confirms that because this is a **Software Engineering** project, the literature review and the whole project must be structured around the *software system* as the primary artefact, with AI/ML as an **integrated feature**, not the core research method. This changes how you frame everything: existing systems review (not just papers), SRS, architecture, design patterns, database design, AI-as-a-feature discussion, UI/UX, security, and software testing (unit/integration/system/UAT) — **not** a Data-Science-style "dataset → model comparison → research gap" narrative.
3. **`Final_Project_Proposal.docx`** — your actual proposal: *AI-Based Personalized Nutrition, Meal Recommendation and Weight Management System for Gym Users*. This plan implements **exactly** what's in it — every module, every deliverable, every requirement — at full depth, not a cut-down version. Nothing in the proposal is skipped.

**Your one stated deviation:** the proposal says "Jupyter Notebook"; you'll use **Google Colab (L4 GPU)** instead. This is a valid, easily-justified substitution (cloud GPU, zero local setup, easy sharing/versioning) — flag it explicitly in your Methodology/Tools chapter as a documented deviation from the proposal, with a one-line justification. Examiners like small, explained deviations far more than silent ones.

**Important technical honesty point (write this into your report — it will earn you marks in "Analysis and Discussion" and "Methodology" bands 60–100, which explicitly reward "consideration of alternative approaches" and "understanding of application and potential limitations"):**
Scikit-learn's `LinearRegression` and `RandomForestRegressor` run on **CPU only** — they do not use a GPU. The L4 GPU in Colab gives you *no* speed benefit for the models named in your proposal. This is fine and does not weaken your project; you should still use Colab+L4 (it's a good environment regardless), but state this clearly rather than implying GPU acceleration you're not getting. If you want a genuine, defensible use for the GPU, §5.6 below gives you an optional, low-risk way to use it (RAPIDS cuML) without changing your proposed algorithms.

---

## 1. System Architecture

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│   React.js Frontend (SPA)    │ HTTPS  │   Node.js / Express Backend   │
│  - Onboarding & Profile      │───────▶│  - REST API                   │
│  - Dashboard                 │◀───────│  - Business logic             │
│  - Meal Plan Viewer          │  JSON  │  - Orchestrates ML calls      │
│  - Progress Tracker          │        │  - Orchestrates recommender   │
└──────────────┬───────────────┘        └───────────┬───────────────────┘
               │                                     │
               │ Supabase JS client                  │ REST (internal)
               ▼                                     ▼
   ┌───────────────────────┐            ┌─────────────────────────────┐
   │       Supabase         │            │   Python ML Microservice     │
   │  - Auth (JWT)           │            │   (FastAPI)                  │
   │  - Postgres DB + RLS    │            │  - calorie_model.pkl          │
   │  - Storage (optional)   │            │  - protein_model.pkl          │
   └───────────────────────┘            │  - recommender.py             │
                                          │  - meal_plan_generator.py     │
                                          └─────────────────────────────┘
```

**Why a separate Python microservice (FastAPI) instead of shelling out from Node or rewriting models in JS?**
Your proposal states *"the trained model will be integrated with the Node.js backend through APIs"* — this is exactly that pattern, and it's the industry-standard way to serve scikit-learn models: train once in Colab → export with `joblib` → serve behind a small stateless FastAPI service → Node.js backend calls it over HTTP like any other internal API. This keeps your ML code in Python (where scikit-learn actually runs), keeps your web backend in Node (as specified), and is trivial to justify in your System Design chapter as a **microservice / service-oriented architecture** decision — a legitimate software-architecture topic for your literature review (§6.6 "System Architecture, Design Patterns" in the SE lit-review structure).

### Component responsibilities
| Layer | Tech | Responsibility |
|---|---|---|
| Frontend | React.js (Vite), Tailwind CSS, React Router, Recharts | UI, forms, dashboards, calling backend API |
| Backend | Node.js + Express | Auth guard, validation, orchestration, calling ML service, business rules (BMI calc, plan versioning) |
| Auth/DB | Supabase (Postgres + Auth + Row Level Security) | Users, profiles, predictions, meals, meal plans, progress logs |
| ML service | Python + FastAPI + scikit-learn + joblib | Calorie/protein prediction, recommendation engine, meal-plan generator |
| Dev/training env | Google Colab (L4 GPU) | EDA, preprocessing, training, evaluation, model export |
| Version control | Git + GitHub | Source control, milestone tagging, supervisor evidence |

---

## 2. Data strategy — this is the part most students get wrong

Your proposal lists generic categories ("Nutrition Dataset", "BMI Dataset", "Obesity Prediction Dataset", "Food Nutrition Dataset", "Calorie and Dietary Information Dataset"). Here is the concrete, best-available dataset for each, why it's the right pick, and — critically — **how the pieces fit together**, because no single public dataset contains "personalized calorie/protein *targets* by fitness goal" as a ready-made label column. You need to construct that label yourself using established sports-nutrition formulas, which is standard practice and gives you a strong, defensible Methodology section.

### 2.1 Recommended datasets

| Purpose | Dataset | Source | Why this one |
|---|---|---|---|
| User physiology + activity features (age, gender, height, weight, BMI, workout frequency, experience level, calories burned per session) | **Gym Members Exercise Dataset** (`valakhorasani/gym-members-exercise-dataset`) | Kaggle | 973 real-world-style gym records with exactly the input features your proposal names (age, gender, weight, height, BMI, workout frequency, experience level, calories burned). This is your primary **training-feature dataset**. |
| Canonical food & macro-nutrient database (for the recommendation engine's food catalogue) | **USDA FoodData Central** (Foundation Foods + SR Legacy) | `https://fdc.nal.usda.gov` (free API + bulk CSV download) | This is the authoritative, government-maintained nutrition database (already cited as reference [4] in your own proposal). Using it instead of a random Kaggle CSV materially strengthens your "Dataset Review" and "Ethical/Data Quality" discussion — it's peer-reviewed-grade data, not scraped/synthetic. Use the **bulk data download** (CSV) rather than hitting the API 1000s of times. |
| Ready-to-use meal-level dataset with pre-computed macros and meal-type tags (breakfast/lunch/dinner/snack) — speeds up building the recommendation engine without you having to build recipes from raw USDA rows | **Daily Food & Nutrition Dataset** (`adilshamim8/daily-food-and-nutrition-dataset`) | Kaggle | Has Calories, Protein, Carbs, Fat, Fiber, Sugar, Sodium, Cholesterol **and** a `Meal_Type` column — exactly the structure your content-based recommender needs, with no extra feature engineering. |
| BMI / obesity category cross-check (sanity-check your BMI classification logic, optional extra evaluation) | **Obesity Prediction Dataset** (UCI "Estimation of Obesity Levels" mirrored on Kaggle) | Kaggle / UCI ML Repository | Gives you an independent dataset to validate your BMI-category logic and optionally show a secondary classification experiment (nice-to-have, not required). |
| Domain-standard formulas used to *generate* your calorie/protein labels | Mifflin-St Jeor Equation (1990) for BMR; WHO/FAO activity multipliers; ISSN protein guidelines (1.6–2.2 g/kg bodyweight depending on goal) | Peer-reviewed sports nutrition literature | These are the formulas dietitians actually use. You cite them in your literature review as the domain ground truth your ML models are trained to approximate/improve on. |

### 2.2 The label-generation strategy (this is your key methodological decision — document it clearly)

No public dataset hands you "this person's correct daily calorie target given their goal." So:

1. Take the Gym Members Exercise Dataset (age, gender, height, weight, activity data).
2. Assign each row a **synthetic fitness goal** (muscle gain / fat loss / maintenance) — either randomly with realistic class balance, or derived from BMI/experience level (e.g., higher BMI → more likely fat-loss goal) to make the synthetic assignment *plausible*, not purely random.
3. Compute **BMR** via Mifflin-St Jeor:
   - Men: `BMR = 10×weight(kg) + 6.25×height(cm) − 5×age + 5`
   - Women: `BMR = 10×weight(kg) + 6.25×height(cm) − 5×age − 161`
4. Compute **TDEE** = BMR × activity multiplier (sedentary 1.2 → extra active 1.9, mapped from workout frequency/experience level).
5. Apply a **goal adjustment**: fat loss → TDEE × 0.80 (≈20% deficit), muscle gain → TDEE × 1.10–1.15 (surplus), maintenance → TDEE × 1.0.
6. Compute **protein target** = bodyweight(kg) × goal-specific coefficient (1.6 g/kg maintenance, 1.8–2.2 g/kg muscle gain, 2.0–2.4 g/kg fat loss to preserve lean mass — cite ISSN position stand).
7. Add small realistic Gaussian noise to both labels to avoid the ML models trivially "memorizing" a deterministic formula (which would make Linear Regression get R²≈1.0 and look suspicious to an examiner — you want your models to *learn* the relationship, and you want an honest, explainable gap between Linear Regression and Random Forest).
8. **This computed column is now your `calorie_target` / `protein_target` label.** Linear Regression and Random Forest are trained to predict it from the raw user features (age, gender, height, weight, BMI, activity level, workout frequency, goal).

This is exactly the workflow real nutrition apps (MacroFactor, MyFitnessPal's goal calculator, etc.) use as their non-ML baseline — and you are demonstrating that ML can **learn and generalize** this relationship from data, plus (with Random Forest) capture non-linear interactions (e.g., how activity level modifies the age–calorie relationship) that plain formulas can't. State this explicitly in your report: *"the models are trained to approximate an established physiological formula from raw features, and are evaluated on their ability to generalize this relationship, with Random Forest expected to outperform Linear Regression on non-linear interaction terms."* This single paragraph will satisfy the "Methodology" and "Use of Literature" criteria simultaneously.

### 2.3 Getting the data (practical steps)
- Kaggle datasets: `pip install kaggle`, get API token from Kaggle account settings, `kaggle datasets download -d valakhorasani/gym-members-exercise-dataset` and `kaggle datasets download -d adilshamim8/daily-food-and-nutrition-dataset`. Works fine inside Colab (`!pip install kaggle`, upload `kaggle.json`).
- USDA FoodData Central: download the **Foundation Foods CSV bulk file** from `fdc.nal.usda.gov/download-datasets` (no API key needed for bulk CSV; API key only needed if you query the live API).
- Store raw files in `/data/raw/`, cleaned files in `/data/processed/`, and keep the Colab notebook that produced each processed file — this becomes your "Dataset Review" evidence and your reproducibility appendix.

---

## 3. Machine Learning Plan (Calorie & Protein Prediction)

### 3.1 Features
`age, gender (encoded), height_cm, weight_kg, BMI (engineered), activity_level (ordinal-encoded), workout_frequency, experience_level, fitness_goal (one-hot: muscle_gain/fat_loss/maintenance)`

### 3.2 Preprocessing pipeline (build as a single `sklearn.pipeline.Pipeline` — examiners reward clean, reproducible pipelines)
1. Missing-value handling: `SimpleImputer` (median for numeric, most-frequent for categorical).
2. Outlier handling: IQR-based clipping on height/weight (protect against data-entry errors).
3. Encoding: `OneHotEncoder` for gender/goal, `OrdinalEncoder` for activity level.
4. Scaling: `StandardScaler` for numeric features (helps Linear Regression; harmless for Random Forest).
5. Wrap all of the above in a `ColumnTransformer` + `Pipeline` so the exact same preprocessing is applied at inference time inside the FastAPI service (no train/serve skew).

### 3.3 Models (exactly as proposed — do not substitute)
- **Linear Regression** — baseline, interpretable, coefficients discussable in your report (e.g., "protein requirement increases by X g per kg bodyweight, holding goal constant").
- **Random Forest Regression** — main model. Tune with `RandomizedSearchCV` over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`; 5-fold cross-validation; use `neg_mean_absolute_error` as the scoring metric (directly relevant to your evaluation metric).

### 3.4 Evaluation (exactly the four metrics your proposal names)
MAE, MSE, RMSE, R² — computed on a held-out test split (80/20, stratified by fitness_goal) **and** via 5-fold cross-validation (report mean ± std, not just a single split — this alone pushes you into the 60–100 "exceptional understanding" band for Methodology). Produce two separate model+metric sets: one for calorie target, one for protein target. Present results in a comparison table (Linear Regression vs Random Forest) plus a feature-importance plot from the Random Forest (this doubles as your "Analysis and Discussion" evidence — discuss *which* features drive predictions and whether that matches nutrition science).

### 3.5 Model export
`joblib.dump(pipeline, "calorie_model.pkl")` and same for protein — export directly from the Colab notebook, download, commit to the FastAPI service's `models/` folder (small files, fine for Git; use Git LFS only if >100MB, which they won't be).

### 3.6 Optional, low-risk GPU justification (only if you want to explicitly use the L4)
If you want the L4 GPU to do *something* real (nice-to-have, not required by the proposal): install **RAPIDS cuML** in the Colab notebook and train a `cuml.ensemble.RandomForestRegressor` as a *side-by-side benchmark* against scikit-learn's CPU version, reporting training-time speedup. This gives you one clean paragraph and one chart ("GPU-accelerated training reduced fit time from Xs to Ys") without touching your production model (you still ship the scikit-learn model in the FastAPI service, since it's what your architecture and proposal specify). Skip this entirely if you're short on time — it's a bonus, not a requirement.

---

## 4. Recommendation Engine (content-based, as specified)

1. Build a **feature vector per food/meal item**: `[calories, protein_g, carbs_g, fat_g, fiber_g]`, min-max normalized.
2. For a given user, compute **remaining macro budget** for the next meal slot (`daily_target − consumed_so_far`, split across breakfast/lunch/dinner/snack using standard ratios, e.g. 25/35/30/10%).
3. Use `sklearn.neighbors.NearestNeighbors` (cosine or Euclidean distance) to rank candidate meals against an **ideal target vector** for that slot, filtered by `Meal_Type` and, if available, dietary tags.
4. Apply goal-specific constraints as a post-filter: fat-loss → prefer higher protein-to-calorie ratio items; muscle-gain → prefer higher total calorie + protein items; maintenance → balanced macro ratio.
5. Return top-N candidates, then pick one with light randomization (seeded) so repeat calls don't always return an identical meal — this matters for the two-month planner (§5) so weeks don't look copy-pasted.

This is a legitimate, well-documented **content-based filtering** approach — cite standard recommender-systems literature (e.g., Ricci et al., *Recommender Systems Handbook*) in your lit review's "AI Integration Approaches" section.

## 5. Two-Month Meal Plan Generator

- Generate **8 weekly cycles**, each with 7 days × 4 meal slots = 28 recommendation-engine calls per week.
- **Variety constraint:** don't repeat the same specific meal within a rolling 5-day window (track a "recently used" set).
- **Weekly macro check:** after generating a week, sum actual calories/protein delivered vs target; if drift exceeds a tolerance (e.g., ±5%), swap the worst-fitting day's meals for better-fitting alternatives (simple greedy repair, not full re-optimization — keep this pragmatic and explain the trade-off in your Limitations chapter, echoing your proposal's own §11.5).
- Persist the generated plan (see DB schema §7) so it can be displayed, re-fetched, and versioned when the user's profile changes.

## 6. Progress Monitoring & Recommendation Adjustment

- User submits a new weight entry (with timestamp) → backend recomputes BMI → calls the ML service again with updated features → if the new calorie/protein target differs from the active plan by more than a threshold (e.g., >7%), prompt the user to regenerate their meal plan (don't silently overwrite — this is a good UX/ethics discussion point, tying back to your proposal's Legal & Ethical Feasibility section).
- Store every historical prediction and weight entry (`progress_logs` table) so the dashboard can chart trend lines (Recharts line chart: weight over time, predicted-vs-actual calorie adherence if you add manual logging — optional stretch goal).

---

## 7. Database Schema (Supabase / Postgres)

```sql
users               (id, email, created_at)                          -- from Supabase Auth
profiles            (user_id FK, age, gender, height_cm, activity_level,
                      workout_frequency, fitness_goal, created_at, updated_at)
weight_logs          (id, user_id FK, weight_kg, bmi, logged_at)
predictions          (id, user_id FK, calorie_target, protein_target,
                      model_version, created_at)
foods                (id, name, category, calories, protein_g, carbs_g,
                      fat_g, fiber_g, meal_type, source)               -- seeded from USDA + Kaggle
meal_plans           (id, user_id FK, week_number, start_date, status)
meal_plan_items      (id, meal_plan_id FK, day_of_week, meal_slot, food_id FK)
```

Enable **Row Level Security (RLS)** on every user-scoped table (`user_id = auth.uid()`) — this is a concrete, gradeable "Security" non-functional requirement fulfilment straight from your proposal's §6.2.

---

## 8. API Design (Node.js/Express)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/profile` | Create/update profile |
| POST | `/api/predict` | Calls ML service → returns calorie + protein targets, stores in `predictions` |
| GET | `/api/recommendations` | Returns meal suggestions for current targets |
| POST | `/api/mealplan/generate` | Triggers 8-week plan generation |
| GET | `/api/mealplan/:id` | Fetch a plan |
| POST | `/api/progress` | Log new weight, triggers recalculation |
| GET | `/api/dashboard` | Aggregated view for the dashboard |

Node backend authenticates requests via Supabase JWT middleware, then calls the FastAPI ML service over an internal HTTP call (`axios`), never exposing the ML service directly to the internet.

---

## 9. Environment Setup (matching your Colab + L4 choice)

1. **Colab notebook structure** (mirrors your proposal's Jupyter → now Colab substitution):
   - `01_EDA.ipynb` — exploratory analysis of Gym Members dataset + food datasets (distributions, correlations, missing values) — this becomes your "Exploratory Data Analysis" evidence.
   - `02_preprocessing.ipynb` — cleaning, label generation (§2.2), pipeline construction.
   - `03_model_training.ipynb` — Linear Regression + Random Forest training, tuning, cross-validation.
   - `04_evaluation.ipynb` — metric tables, feature importance, error analysis, exported charts (PNG) for your report.
   - `05_export.ipynb` — final pipeline fit on full data, `joblib.dump`, sanity-check reload.
2. Mount Google Drive in Colab for dataset persistence between sessions; mirror notebooks to GitHub at the end of each session (`!git add . && git commit && git push` from Colab, or download `.ipynb` and commit locally) — this gives you commit-history evidence of iterative work, which supervisors and examiners like to see.
3. Runtime: `Runtime → Change runtime type → L4 GPU`. Since scikit-learn won't use it (§0), your GPU is mainly buying you Colab's higher RAM/priority tier — still worth keeping for a smoother notebook experience, and required if you do the optional cuML benchmark (§3.6).
4. Local dev: VS Code for React/Node/FastAPI code, Node ≥18, Python ≥3.10 (`venv` or `poetry`), Git.

---

## 10. Testing & Evaluation Plan (feeds your "Methodology" + "Analysis" marks)

| Type | Tooling | Scope |
|---|---|---|
| Unit tests | Jest (Node), Pytest (Python) | BMI calc, TDEE calc, API validators, ML service response shape |
| Integration tests | Supertest (Node) | Backend ↔ ML service ↔ Supabase round-trips |
| Model evaluation | scikit-learn metrics + cross-validation | §3.4 |
| System / E2E tests | Manual + Playwright (optional) | Full user journey: register → profile → predict → recommend → plan → progress update |
| UAT | 3–5 real gym-going testers, structured feedback form | Confirms usability against the stated non-functional requirement |
| Security tests | Manual RLS policy checks, auth bypass attempts, input validation fuzzing | Confirms §6.2 "Security" and "Data Privacy" NFRs |
| Performance tests | k6 or Apache Bench on `/api/predict` and `/api/mealplan/generate` | Confirms "Performance" NFR under concurrent load |

Document each test with expected vs actual result in your Testing chapter — this is exactly what the "Report Structure and Use of Academic Writing" and "Analysis and Discussion" bands reward.

---

## 11. 12-Week Timeline (aligned to both your proposal's phases and CSE6035's milestone table)

| Week | Phase | Deliverable (maps to CSE6035 milestones) |
|---|---|---|
| 1–2 | Planning & research; finalize proposal | Project title, Project Proposal, Research Ethics Application |
| 3 | Literature review drafting (SE-track structure) + dataset acquisition | — |
| 4–5 | EDA + preprocessing + label generation (Colab) | Literature Review, Project Planning, SRS |
| 6–7 | Model training, tuning, evaluation; start recommendation engine | — |
| 8 | Database schema + Supabase setup + Auth; FastAPI service skeleton | Design Specification |
| 9 | React frontend (profile, dashboard) + Node API integration | Prototype |
| 10 | Meal plan generator + progress monitoring + polish UI | Working prototype demo |
| 11 | Full testing pass (§10), bug fixing, performance/security checks | Testing evidence |
| 12 | Final documentation, thesis assembly, proofreading, submission | Thesis document (WRIT1) |

---

## 12. Deliverables Checklist (verbatim from your proposal §7 — confirm each before submission)

- [ ] Intelligent Web Application (React.js + Node.js + Supabase)
- [ ] Calorie Prediction Model (Linear Regression + Random Forest)
- [ ] Protein Prediction Model (Linear Regression + Random Forest)
- [ ] Personalized Meal Recommendation Engine (content-based)
- [ ] Two-Month Meal Plan Generator
- [ ] Database System (schema in §7 above)
- [ ] Model Evaluation Results (MAE, MSE, RMSE, R²)
- [ ] Project Documentation (Proposal, Requirements, System Design, DB Design, ML Model Docs, Testing Reports, User Manual, Final Report)
- [ ] Research/analysis discussion of ML applied to personalized nutrition (framed as *system evaluation*, not a research contribution — SE track)

---

## 13. What NOT to do (common ways students accidentally under-deliver on this exact proposal)

1. Don't skip the **two-month plan generator** and only build daily recommendations — it's a named, separate deliverable (§7.4).
2. Don't build the recommendation engine as simple SQL filtering — the proposal explicitly says "content-based recommendation techniques," so a feature-vector + similarity/ranking approach (§4) is required to match what you proposed.
3. Don't train models directly on a formula output with zero noise (§2.2 step 7) — an examiner who checks your R² and sees a suspiciously perfect 0.999+ score with zero discussion will question it. Add noise, and *discuss* the gap between Linear Regression and Random Forest honestly.
4. Don't forget the **progress monitoring / recalculation** loop (§6) — it's a distinct functional requirement (§6.1 item 8–9 in your proposal), not just "let the user edit their profile."
5. Don't present this as a Data Science research project in your literature review — per the Literature Review guide, keep the emphasis on system architecture, requirements, design, AI-as-integrated-feature, and software testing.

---

## 14. Functional Requirements Traceability (proposal §6.1 → this plan)

| # | Functional Requirement (proposal, verbatim) | Where it's implemented in this plan |
|---|---|---|
| FR1 | User Registration and Authentication | §1 Supabase Auth (JWT); §7 `users` table |
| FR2 | User Profile Management | §7 `profiles` table; §8 `POST /api/profile` |
| FR3 | BMI Calculation | §3.1 engineered feature; §7 `weight_logs.bmi`; recomputed on every weight log (§6) |
| FR4 | Calorie Requirement Prediction | §3 (full ML pipeline); §7 `predictions.calorie_target`; §8 `POST /api/predict` |
| FR5 | Protein Requirement Prediction | §3 (same pipeline, second target); §7 `predictions.protein_target` |
| FR6 | Personalized Meal Recommendation | §4 content-based recommender; §8 `GET /api/recommendations` |
| FR7 | Two-Month Meal Plan Generation | §5; §7 `meal_plans` / `meal_plan_items`; §8 `POST /api/mealplan/generate` |
| FR8 | Progress Monitoring | §6; §7 `weight_logs`; §8 `POST /api/progress` |
| FR9 | Recommendation Adjustment | §6 (threshold-triggered recalculation + re-prompt logic) |
| FR10 | Dashboard and Reporting | §8 `GET /api/dashboard`; frontend Dashboard page (§1 component table) rendering BMI trend, calorie/protein targets, active meal plan summary, and progress-over-time charts (Recharts line/bar charts) — this was under-specified in the first draft of this plan and is now made explicit: the dashboard must visually report **(a)** current BMI + category, **(b)** calorie/protein targets vs. history, **(c)** this week's meal plan at a glance, **(d)** a weight trend line, per the proposal's "visual reports and summaries" wording. |

All 10 functional requirements from your proposal are now explicitly accounted for. None were missing in substance, but FR10 (Dashboard/Reporting) needed the extra detail above to match what the proposal actually describes ("view their BMI, daily calorie requirements, protein targets... monitor their progress and review previous recommendations through visual reports and summaries").

## 15. Non-Functional Requirements Implementation Matrix (proposal §6.2 → concrete measures)

The first draft of this plan only addressed 3 of the 10 listed NFRs explicitly (Security, Usability, Performance). Here is the complete set:

| NFR (proposal) | Concrete implementation measure |
|---|---|
| Performance | FastAPI async endpoints; DB indexes on `user_id`/`food.meal_type`; k6 load test target (§10) |
| Reliability | Try/catch + fallback responses in the ML service (e.g., if a model file fails to load, return a formula-only calorie estimate rather than crash); Node-level error middleware returning consistent error shapes |
| Security | Supabase RLS on every user-scoped table (§7); JWT-only access to `/api/*`; ML microservice never exposed publicly (§8) |
| Usability | UAT with 3–5 gym-going testers (§10); onboarding flow limited to the exact fields in FR2, no unnecessary friction |
| Scalability | Stateless FastAPI service (horizontally scalable); Supabase/Postgres scales independently of the app tier; meal-plan generation is a background-safe operation you can later move to a queue if needed (documented as a future-work note, not required now) |
| Availability | Deploy frontend (Vercel/Netlify free tier) and backend (Render/Railway free tier) as always-on managed services rather than a laptop-hosted server — cheap, realistic, and demoable at any time for your supervisor |
| Maintainability | Modular structure (separate `services/`, `routes/`, `models/` folders); ColumnTransformer pipeline (§3.2) keeps preprocessing in one place instead of scattered across the codebase |
| Compatibility | Responsive Tailwind layout tested on Chrome/Firefox/Edge + mobile viewport widths; REST/JSON API is client-agnostic |
| Data Privacy | RLS + Supabase Auth; no raw health data ever sent to the ML service beyond what's needed for the single prediction call (no persistent storage of PII inside the ML microservice itself — it stays in Supabase only) |
| Accuracy | §3.4 evaluation (MAE/MSE/RMSE/R², 5-fold CV) directly targets this; recommendation engine tolerance check in §5 ("weekly macro check") keeps delivered meal plans within ±5% of the predicted target |

## 16. Expected Outputs & Outcomes Traceability (proposal §9.1/§9.2)

Every bullet in your proposal's "Expected Outputs" (9.1) maps onto a deliverable already in §12, with one addition worth calling out explicitly: your proposal lists *"Nutritional information and calorie analysis for recommended meals"* as an expected output — make sure the meal recommendation UI shows the **per-meal macro breakdown** (not just the meal name), since that's the difference between a meal *name* list and a meal *recommendation with analysis* as promised. Everything else in 9.1/9.2 (trained models, BMI + weight category, profile management, evaluation metrics, documentation) is already covered in §3, §7, §12.

## 17. Limitations Carried Into the Implementation (proposal §11 → design response)

Your proposal already lists 8 limitations. A strong report doesn't just repeat them at the end — it shows the *design responded* to them where possible. This is the mapping to write into your Limitations chapter:

| Limitation (proposal) | How the implementation responds |
|---|---|
| 11.1 Dependence on dataset quality | §2 documents dataset provenance (USDA = authoritative) and the label-generation method transparently, so quality is auditable rather than hidden |
| 11.2 Limited personal health information (no allergies/medical conditions) | Explicitly out of scope — state this as a conscious design boundary, not an oversight; mention it as future work (an `allergies`/`dietary_restrictions` field would extend `profiles` and filter §4's candidate pool) |
| 11.3 User input accuracy | Basic client + server-side validation (range checks on age/height/weight in the API layer, §8) — reduces but doesn't eliminate this risk; say so honestly |
| 11.4 Limited food database coverage | Mitigated by using USDA FoodData Central (thousands of items) rather than a small hand-made list, but still finite — acknowledge regional/local food gaps as-is |
| 11.5 Model generalization limitations | Directly addressed by cross-validation reporting (§3.4) and by discussing Random Forest vs Linear Regression error patterns rather than claiming perfect generalization |
| 11.6 Internet dependency | Inherent to the chosen web-app architecture — acknowledge as-is, no mitigation needed for an academic project |
| 11.7 Not a replacement for professional medical advice | Add a visible disclaimer in the UI (footer/onboarding step) — a one-line implementation task that directly satisfies an ethical commitment you already made in the proposal's Legal & Ethical Feasibility section |
| 11.8 Future improvements | §17 and §4/§6's "future work" notes above already seed this discussion |

## 18. Resource Confirmation (proposal §8.1/§8.2 → status)

- **Software resources** (§8.1 of proposal): Python, VS Code, React.js, Node.js, Supabase, Pandas, NumPy, Scikit-learn, Git/GitHub — all used exactly as listed. **Jupyter Notebook → Google Colab** is the one documented substitution (§0). FastAPI is one addition not named in the proposal, needed purely as the integration layer the proposal itself calls for ("integrated with the Node.js backend through APIs") — worth a one-line justification in your Tools chapter.
- **Hardware resources** (§8.2 of proposal): the minimum spec listed (8GB RAM, i5-class CPU, internet) is for *local development* — since model training now happens on Colab's cloud hardware (with a free L4 GPU, which exceeds the proposal's minimum ask), your local machine only needs to run the web app in development, which is a lighter load than the proposal anticipated. Worth one sentence in your report noting this favorable deviation.

---

### Next step
This is the plan only — no code has been written yet, per your request. When you're ready, tell me which piece to start with (I'd suggest: Colab EDA + label generation first, since everything else depends on the trained models), and I'll build it with you step by step.
