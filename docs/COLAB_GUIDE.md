# Google Colab — step-by-step training guide

Read this once before you start. It tells you exactly which runtime to pick,
when to switch it, and when to delete it.

---

## ⚠️ Read this first: do NOT waste your compute units

You have **60 compute units (CU)** on Colab Pro. Here is the honest position:

**scikit-learn is CPU-only.** `LinearRegression` and `RandomForestRegressor`
physically cannot run on a GPU. Selecting an L4 for this project buys you
**zero** speed-up on training — it only drains CUs.

Measured on this exact codebase:

| Notebook | Runtime | Wall time | CU cost |
|---|---|---|---|
| 01 EDA | CPU (free) | ~30 s | **0** |
| 02 Preprocessing | CPU (free) | ~20 s | **0** |
| 03 Model training (with tuning) | CPU (free) | ~2–4 min | **0** |
| 04 Evaluation (incl. learning curves) | CPU (free) | ~3–6 min | **0** |
| 05 Export | CPU (free) | ~20 s | **0** |
| **Total for the whole project** | **CPU (free)** | **~10 min** | **0 CU** |

> ### Run everything on the **free CPU runtime**. Total cost: 0 CU.

The **only** thing worth spending CUs on is the optional RAPIDS cuML GPU
benchmark in notebook 04 §6 — a bonus paragraph for your report. That costs
roughly **1–2 CU** for a ~15-minute session. Even then it is optional; skip it
if you would rather keep the units.

**Do not leave a GPU runtime idle.** Colab bills for connected time, not
compute time. An idle L4 session drains CUs while you read a PDF.

---

## Step 0 — Prepare your Google Drive (one time, ~5 minutes)

1. Download the two datasets — see [`data/raw/README.md`](../data/raw/README.md)
   for the exact links.
2. Put the CSVs in `NutriFit-AI/data/raw/` **on your laptop** first.
3. Open [drive.google.com](https://drive.google.com).
4. Drag the **entire `NutriFit-AI` folder** into **My Drive** (the root, not a
   subfolder).

Your Drive must end up looking like this:

```
My Drive/
└── NutriFit-AI/
    ├── ml/
    │   ├── nutrifit/          <- the shared Python package
    │   ├── notebooks/         <- the 5 notebooks
    │   └── scripts/
    └── data/
        └── raw/
            ├── gym_members_exercise_tracking.csv
            └── daily_food_nutrition_dataset.csv
```

> **Why upload the whole folder?** The notebooks import the `nutrifit` package —
> the same code the FastAPI service uses. That is what guarantees training and
> serving apply identical preprocessing. The notebooks are deliberately thin
> wrappers around it.

Wait for Drive to finish syncing (the green tick) before moving on.

---

## Step 1 — Open the first notebook

1. In Drive, navigate to `NutriFit-AI/ml/notebooks/`.
2. Double-click `01_EDA.ipynb`.
3. If Colab does not open it: right-click → **Open with** → **Google Colaboratory**.

---

## Step 2 — Set the runtime to CPU

This is the important step.

1. Menu: **Runtime → Change runtime type**
2. Hardware accelerator: select **CPU**
3. Click **Save**

If it was already CPU, nothing happens. If you changed it, **the runtime
restarts and all variables are cleared** — that is normal.

> Colab sometimes defaults new notebooks to a GPU. Check this every time you
> open a notebook. A GPU you never asked for still costs CUs.

---

## Step 3 — Run notebook 01 (EDA)

1. **Runtime → Run all**
2. The first cell asks permission to mount Google Drive:
   - Click the link, choose your Google account, click **Allow**
   - Copy the code back into the box if prompted
3. Check the SETUP cell output. You should see:
   ```
   nutrifit  1.0.0
   project   /content/drive/MyDrive/NutriFit-AI
   in colab  True
   ```
4. In the "Load the gym members dataset" cell, confirm `USE_DEMO = False` and
   that it prints **973 rows**.

**If you see `*** SYNTHETIC DEMO DATA ***`** — your CSVs are not in
`data/raw/`. Fix that before continuing; nothing downstream will be reportable.

**If you get `FileNotFoundError: /content/drive/MyDrive/NutriFit-AI`** — the
folder is not at the root of My Drive. Move it there and re-run the cell.

Figures are written to `ml/reports/figures/`. They appear in your Drive
automatically.

**Do not disconnect yet.**

---

## Step 4 — Run notebooks 02 → 03 → 04 → 05, in order

They must run in sequence: each depends on files the previous one wrote.

For each notebook:

1. Open it from Drive (`ml/notebooks/`).
2. Confirm **Runtime → Change runtime type → CPU**.
3. **Runtime → Run all**.
4. Wait for it to finish, then check the last cell's output.

| Notebook | What it writes | Check before moving on |
|---|---|---|
| `02_preprocessing.ipynb` | `data/processed/gym_users_labelled.csv`, `food_catalogue.csv`, `foods_seed.sql` | "PASS - deficit < maintenance < surplus" |
| `03_model_training.ipynb` | `ml/artifacts/_*_bundle.pkl` | Comparison table prints, 4 rows |
| `04_evaluation.ipynb` | all report figures + CSVs | Learning-curve tables print |
| `05_export.ipynb` | `calorie_model.pkl`, `protein_model.pkl`, `recommender.pkl`, `model_metrics.json`, `model_card.json` | All `[OK]`, no `OUT OF RANGE` |

### Do I delete the runtime between notebooks? — No

**Do not** "Disconnect and delete runtime" after each notebook. It buys you
nothing and costs you time:

* **Nothing carries over anyway.** Each notebook gets its own kernel. Notebook
  03 does not inherit variables from 02 — it reads the files 02 wrote to Drive.
  So there is no state to "clear".
* **Deleting destroys the VM.** The next notebook then has to re-mount Drive,
  re-run `pip install`, and re-import everything: roughly 1–2 minutes wasted
  per notebook.
* **On CPU there is no billing reason.** You are spending 0 CU either way. The
  "delete it immediately" rule applies to *GPU* sessions, which bill for
  connected time — not to CPU sessions.

Just open the next notebook and **Runtime → Run all**. Delete the runtime only
at the very end (Step 5), and immediately after the optional GPU section.

If you have opened several notebooks and want to tidy up, use
**Runtime → Manage sessions** to see what is still active and terminate the
finished ones. Idle sessions self-terminate after ~90 minutes regardless.

### Restart session vs Disconnect and delete runtime

These are not interchangeable — pick the right one:

| Action | What it does | Use it when |
|---|---|---|
| **Restart session** | Clears variables and imports; keeps the VM and installed packages. Takes seconds. | You edited or re-uploaded files under `ml/nutrifit/`; you hit a stale-import `NameError`; memory warnings |
| **Disconnect and delete runtime** | Destroys the VM entirely. Next connect is a fresh machine. | You have finished for the session; you just used a GPU |

> **Important when re-uploading code:** if a runtime is already running and you
> replace files in `ml/nutrifit/` on Drive, Python keeps the **old** module in
> memory and your changes will appear to do nothing. Always **Restart session**
> after re-uploading, then **Run all**.

### Keeping the session alive

Colab disconnects an idle notebook after ~90 minutes. Each notebook here takes
minutes, not hours, so this only bites if you walk away mid-run. If it does
disconnect: **Runtime → Run all** again. Everything is written to Drive, so
nothing is lost — you just re-run.

### If a runtime gets into a bad state

Symptoms: stale imports after you edited `nutrifit/`, weird `NameError`s,
memory warnings.

**Runtime → Restart session**, then **Runtime → Run all**.

Restarting is free and takes seconds. It does *not* delete your Drive files.

> **Note:** if you edit a file in `ml/nutrifit/` while a notebook is running,
> Python will keep using the already-imported old version. You must
> **Restart session** for the change to take effect.

---

## Step 5 — Finish and free your resources

Once `05_export.ipynb` shows all `[OK]`:

1. Confirm the artefacts exist in Drive at `NutriFit-AI/ml/artifacts/`:
   - `calorie_model.pkl`
   - `protein_model.pkl`
   - `recommender.pkl`
   - `model_metrics.json`
   - `model_card.json`
2. **Runtime → Disconnect and delete runtime**
3. Click **Yes** to confirm.

> ### Closing the browser tab does NOT stop the session.
> Always use **Disconnect and delete runtime**. On CPU this costs nothing, but
> build the habit now so you never leave a GPU session running.

---

## Step 6 — Bring the artefacts back to your laptop

Sync the Drive folder down (Google Drive desktop app, or download the
`artifacts` folder), so your local project has:

```
NutriFit-AI/ml/artifacts/
├── calorie_model.pkl
├── protein_model.pkl
├── recommender.pkl
├── model_metrics.json
└── model_card.json
```

Also bring back `data/processed/` (you need `foods_seed.sql` to seed the
database).

Then, from the project root:

```bash
python ml/scripts/export_models.py
```

This copies the artefacts into the FastAPI service **and verifies each one** by
reloading it, running a prediction, and range-checking the result. If anything
is corrupt or version-mismatched, it fails here rather than during your demo.

---

## OPTIONAL — the GPU benchmark (notebook 04 §6)

Only if you want a genuine GPU paragraph in your report. **Budget ~1–2 CU.**

1. Finish notebooks 01–05 on CPU first. Do not do this instead of them.
2. Open `04_evaluation.ipynb`.
3. **Runtime → Change runtime type → L4 GPU → Save.**
   The runtime restarts and clears all variables.
4. Re-run the SETUP cell and the "Loaded training bundles" cell.
5. Run the two cells in §6. The cuML install takes ~5 minutes.
6. **Immediately** when done: **Runtime → Disconnect and delete runtime.**
7. Check your remaining CUs: the Colab resources panel (▾ next to RAM/Disk).

**Expected finding:** on 973 rows the GPU may be *slower* than the CPU, because
kernel-launch overhead outweighs the parallelism at this scale. That is a
legitimate, interesting result — report it honestly. It reinforces the
"dataset size is the binding constraint" argument from §3 of notebook 04.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: /content/drive/MyDrive/NutriFit-AI` | Folder not at Drive root | Move it to **My Drive**, re-run |
| `*** SYNTHETIC DEMO DATA ***` | CSVs missing from `data/raw/` | Download them (see `data/raw/README.md`) |
| `SchemaError: could not resolve required column` | Publisher changed a header | The error lists the headers found — add the real spelling to `ml/nutrifit/schema.py` |
| `ModuleNotFoundError: nutrifit` | SETUP cell not run, or wrong Drive path | Re-run the SETUP cell |
| Edited `nutrifit/` but nothing changed | Python cached the old import | **Runtime → Restart session** |
| Notebook 03 says `USERS_PROCESSED not found` | Notebook 02 not run | Run 02 first |
| Slots with `<20 items` warning | Kaggle catalogue is thin | Add USDA data (`data/raw/README.md` §3) |
| Disconnected mid-run | Idle timeout | **Runtime → Run all** again |

---

## Compute-unit budget summary

| Activity | CU |
|---|---|
| Notebooks 01–05 on CPU | **0** |
| Re-running them after a mistake | **0** |
| Optional GPU benchmark, one ~15 min session | **~1–2** |
| **Realistic total for this project** | **0–2 of your 60** |

You will finish this project with essentially all of your compute units intact.
That is the correct outcome — spending them here would buy you nothing.
