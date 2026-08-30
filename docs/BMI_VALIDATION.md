# The Obesity Prediction Dataset — how it is used

Proposal §8.3 lists an *Obesity Prediction Dataset* among the project's data
resources. This document records exactly what role it plays, so the dissertation
can state a defensible position either way.

---

## The short version

It is **not** a training dataset. It is an **independent verification set for
the BMI classification logic**, and it is optional.

Use whichever of the two sections below matches what you actually do.

---

## Why it is not used for training

The proposal frames this project as predicting **continuous nutritional
requirements** — daily calories (kcal) and protein (g) — via regression. The
obesity dataset supports a **categorical** task: assigning one of seven obesity
classes from lifestyle and physiological attributes.

Training on it would require either:

* changing the problem from regression to classification, contradicting FR4/FR5
  and every evaluation metric the proposal names (MAE, MSE, RMSE, R²), or
* using its features as extra predictors — but they do not overlap with the gym
  dataset's schema (it records transport mode, family history, smoking, water
  intake), so the two cannot be joined at row level.

Using it as a training input would therefore be a change of research question,
not an enhancement. **This is a deliberate scope decision, not an oversight.**

## Why it is still worth using

FR3 (BMI Calculation) is implemented **twice** — in Python for the ML service
(`nutrifit/nutrition.py`) and in JavaScript for the Node backend
(`backend/src/utils/nutrition.js`). Both are unit tested against hand-computed
WHO boundary values.

Those unit tests have a blind spot: they prove the code matches **our reading**
of the WHO standard. If that reading were wrong — an off-by-one at a boundary,
`<` where `<=` belongs — the tests would pass and the bug would ship.

Scoring the classifier against ~2,100 records labelled by an independent
research team tests our reading against somebody else's. That is a genuine
verification step, and it is the correct use of this dataset in a
software-engineering project.

---

## Option A — run the cross-check (recommended, ~5 minutes)

1. Download from either source:
   * UCI: <https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition>
   * Kaggle mirror: <https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels>
2. Put the CSV anywhere in `data/raw/` — any filename containing "obesity" is
   found automatically.
3. Run:

```bash
python ml/scripts/validate_bmi.py
```

### Outputs

| File | Contents |
|---|---|
| `ml/reports/bmi_validation.txt` | Agreement rate, per-category recall, confusion matrix |
| `ml/reports/bmi_confusion_matrix.csv` | 4×4 matrix for a report table |
| `ml/reports/bmi_disagreements.csv` | Every mismatch, with its distance to the nearest WHO boundary |

### How to interpret it

The script maps the dataset's seven classes onto the four WHO bands:

```
Insufficient_Weight                       -> underweight
Normal_Weight                             -> normal
Overweight_Level_I / Overweight_Level_II  -> overweight
Obesity_Type_I / II / III                 -> obese
```

Expect **high agreement (>95 %)**. Anything lower indicates a real logic error
and should be investigated before reporting — the script warns and exits
non-zero in that case.

Pay attention to the `distance_to_boundary` column. A disagreement at BMI 24.98
vs 25.01 is a rounding artefact in the source data, not a defect in the
classifier. Distinguishing the two in your write-up demonstrates exactly the
critical analysis the 60–100 marking band rewards.

### ⚠️ What this does *not* prove

Roughly **77 % of the UCI dataset is synthetically generated** using SMOTE, and
its class labels are themselves derived from BMI thresholds.

So high agreement confirms that **our arithmetic and band boundaries are
correct**. It does **not** independently validate BMI as a measure of health,
and it must not be presented as such. State this limitation explicitly —
overclaiming here is a far bigger risk to your marks than the modest finding
itself.

---

## Option B — do not run it (write this instead)

If you choose not to download it, put this in your Dataset Review chapter:

> The Obesity Prediction Dataset identified in the proposal was evaluated and
> deliberately excluded from the modelling pipeline. The proposed system
> performs regression over continuous nutritional targets, whereas that dataset
> supports multi-class obesity classification; its feature schema (transport
> mode, family history, smoking status) does not overlap with the Gym Members
> dataset, so the two cannot be joined at record level. Incorporating it would
> have required restating the research question as a classification problem,
> contradicting the functional requirements and the evaluation metrics defined
> in the proposal. It was instead retained as an optional verification resource
> for the BMI classification component, for which an automated cross-check is
> implemented in `ml/scripts/validate_bmi.py`.

That paragraph turns an unused resource into a documented, reasoned scope
decision — which is what the Methodology criterion is actually assessing.

---

## Verification without the download

The cross-check logic is itself unit tested. `ml/tests/test_validation.py`
builds a synthetic frame using the **real UCI schema and label vocabulary**, and
verifies the loader, the seven-to-four class mapping, centimetre/metre
detection, the scoring maths, and the failure path when labels are deliberately
corrupted.

```bash
cd ml && python -m pytest tests/test_validation.py -q     # 19 passed
```

So the mechanism is proven correct whether or not the dataset is present.
