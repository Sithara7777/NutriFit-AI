# Raw datasets — download instructions

Put the downloaded CSV files **directly in this folder** (`data/raw/`).
Do not rename them if you don't want to — the loader recognises the common
Kaggle filenames automatically. If you do rename them, use the names in the
"Save as" column below.

---

## 1. Gym Members Exercise Dataset — **REQUIRED**

This is the primary training-feature dataset (973 records).

| | |
|---|---|
| **Link** | https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset |
| **Download** | Click the **Download** button (top right) → you get `archive.zip` |
| **Unzip** | Extract `gym_members_exercise_tracking.csv` into this folder |
| **Save as** | `gym_members_exercise_tracking.csv` |
| **Size** | ~60 KB |

**Expected columns** (the loader tolerates spelling/spacing variants):

```
Age, Gender, Weight (kg), Height (m), Max_BPM, Avg_BPM, Resting_BPM,
Session_Duration (hours), Calories_Burned, Workout_Type, Fat_Percentage,
Water_Intake (liters), Workout_Frequency (days/week), Experience_Level, BMI
```

Why this one: it carries every input feature the proposal names (age, gender,
height, weight, BMI, workout frequency, experience level) **plus**
`Fat_Percentage`, which lets the system use the Katch–McArdle lean-mass BMR
equation — the estimator recommended for trained populations.

---

## 2. Daily Food & Nutrition Dataset — **REQUIRED**

This is the food catalogue behind the recommendation engine.

| | |
|---|---|
| **Link** | https://www.kaggle.com/datasets/adilshamim8/daily-food-and-nutrition-dataset |
| **Download** | Click **Download** → `archive.zip` |
| **Unzip** | Extract the CSV into this folder |
| **Save as** | `daily_food_nutrition_dataset.csv` |
| **Size** | ~1 MB |

**Expected columns:**

```
Date, User_ID, Food_Item, Category, Calories (kcal), Protein (g),
Carbohydrates (g), Fat (g), Fiber (g), Sugars (g), Sodium (mg),
Cholesterol (mg), Meal_Type, Water_Intake (ml)
```

Why this one: it already carries a `Meal_Type` column
(breakfast/lunch/dinner/snack), which is exactly what the content-based
recommender needs to filter candidates per meal slot.

> **Note:** this file is a *consumption log*, not a catalogue — the same food
> appears on many rows with different portion sizes. `prepare_data.py`
> aggregates it into one row per `(food, meal slot)` using the median of each
> macro. You don't need to do anything; just be aware the item count after
> processing is much smaller than the row count.

---

## 3. USDA FoodData Central — **RECOMMENDED (measured improvement)**

Adds ~104 whole foods to the catalogue (489 → 593 items). It also strengthens
the "Dataset Review" and data-quality discussion, because USDA is authoritative
government-maintained data rather than a scraped Kaggle upload.

**Measured effect** on the eight-week plan (worst weekly macro error across all
three goals, 8 weeks, seed 42):

| Catalogue | Worst error | Greedy repair passes needed |
|---|---|---|
| Kaggle only (489 items) | 3.80 % | 2 |
| **+ USDA (593 items)** | **2.88 %** | **0** |

Note the USDA rows are **per 100 g** while the Kaggle rows are per serving.
Mixing measurement bases was expected to *hurt*, and the initial design
disabled USDA for exactly that reason — but the added whole foods are
protein-dense, and protein (not calories) is the binding constraint in meal
composition, so it measurably helps. Toggle `USE_USDA` in notebook 02, or
`--with-usda` on `prepare_data.py`, to reproduce the comparison; it makes a
good ablation for the Analysis chapter.

| | |
|---|---|
| **Link** | https://fdc.nal.usda.gov/download-datasets |
| **Which file** | "Foundation Foods" → **Latest Downloads → CSV** (no API key needed) |
| **Unzip into** | `data/external/` (NOT `data/raw/`) |
| **Needs** | `food.csv` and `food_nutrient.csv` |
| **Size** | ~50 MB zipped |

The loader skips this silently if the files are absent, so the pipeline works
with or without it.

---

## 4. Obesity Prediction Dataset — **OPTIONAL**

Only used for an independent cross-check of the BMI-category logic. Nothing
depends on it.

| | |
|---|---|
| **Link** | https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition |
| **Kaggle mirror** | https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels |
| **Save as** | `obesity_prediction.csv` |

---

## Alternative: download inside Colab with the Kaggle API

If you'd rather not download manually, you can pull both required datasets
straight into Colab:

1. Kaggle → your avatar → **Settings** → **API** → **Create New Token**.
   This downloads `kaggle.json`.
2. In the Colab notebook, run the "Kaggle download" cell and upload
   `kaggle.json` when prompted.

The commands the notebook runs:

```bash
pip install -q kaggle
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d valakhorasani/gym-members-exercise-dataset -p data/raw --unzip
kaggle datasets download -d adilshamim8/daily-food-and-nutrition-dataset -p data/raw --unzip
```

---

## Verifying your download

From the project root:

```bash
python ml/scripts/prepare_data.py
```

You should see the real row count (973 for the gym dataset) and **no**
"RUNNING ON SYNTHETIC DEMO DATA" warning. If a column can't be resolved, the
error message names the missing field and lists the headers actually found —
add the real spelling to the alias table in `ml/nutrifit/schema.py`.

To test the pipeline *before* downloading anything:

```bash
python ml/scripts/prepare_data.py --demo
```

This uses synthetic stand-in data. **Results from `--demo` are not reportable**
and every script prints a loud warning when it is active.

---

## Licensing / ethics note for your report

* Both Kaggle datasets are published under open licences and contain **no
  personally identifying information** — records are anonymous physiological
  measurements.
* USDA FoodData Central is US Government public-domain data.
* No human participants were recruited for model training, so no ethics
  approval is required for the dataset component. (UAT with 3–5 gym-going
  testers *does* need a participant information sheet and consent form — see
  the assessment brief's Research Ethics Application milestone.)
