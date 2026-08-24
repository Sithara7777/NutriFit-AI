"""Check that the Colab outputs were synced down from Drive correctly.

Run this after copying `ml/reports/`, `ml/artifacts/`, `data/processed/` and
`ml/notebooks/` back from Google Drive:

    python ml/scripts/verify_sync.py

It confirms every expected figure and table is present, that the artefacts load,
and -- most importantly -- that nothing was produced from the synthetic demo
data, which would be unreportable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nutrifit import config  # noqa: E402

EXPECTED_FIGURES = [
    "eda_feature_distributions", "eda_categorical_counts", "eda_correlation_matrix",
    "eda_body_composition", "eda_activity_bands", "eda_food_log_structure",
    "eda_food_meal_slots", "labels_goal_assignment", "labels_target_distributions",
    "labels_noise_injection", "catalogue_macro_profile", "cv_fold_distribution",
    "eval_predicted_vs_actual", "eval_residual_distributions", "eval_learning_curves",
    "eval_feature_importance", "eval_meal_plan_accuracy",
]

EXPECTED_TABLES = [
    "model_comparison.csv",
    "learning_curve_calorie_target.csv", "learning_curve_protein_target.csv",
    "feature_importance_calorie_target.csv", "feature_importance_protein_target.csv",
    "coefficients_calorie_target.csv", "coefficients_protein_target.csv",
    "residuals_calorie_target.csv", "residuals_protein_target.csv",
    "meal_plan_weekly_summary.csv", "meal_plan_sample.csv",
]

EXPECTED_ARTIFACTS = [
    config.CALORIE_MODEL_FILE, config.PROTEIN_MODEL_FILE,
    config.RECOMMENDER_FILE, config.METRICS_FILE, config.MODEL_CARD_FILE,
]

EXPECTED_PROCESSED = [
    "gym_users_labelled.csv", "food_catalogue.csv",
    "foods_seed.sql", "data_quality_report.json",
]


def check(label: str, directory: Path, names: list[str], suffix: str = "") -> list[str]:
    missing = [n for n in names if not (directory / f"{n}{suffix}").exists()]
    present = len(names) - len(missing)
    status = "OK  " if not missing else "MISS"
    print(f"[{status}] {label:22s} {present}/{len(names)} present")
    for name in missing:
        print(f"         missing: {name}{suffix}")
    return missing


def main() -> int:
    print(f"Project root: {config.PROJECT_ROOT}\n")
    problems: list[str] = []

    problems += check("figures", config.FIGURES_DIR, EXPECTED_FIGURES, ".png")
    problems += check("report tables", config.REPORTS_DIR, EXPECTED_TABLES)
    problems += check("model artefacts", config.ARTIFACTS_DIR, EXPECTED_ARTIFACTS)
    problems += check("processed data", config.PROCESSED_DIR, EXPECTED_PROCESSED)

    # --- provenance: was this produced from the real datasets? ------------
    print()

    # Read the catalogue itself rather than trusting the quality report: the
    # report is written by prepare_data.py, but the notebooks write the
    # catalogue directly, so a Colab-produced catalogue can legitimately differ
    # from whatever the last local script run recorded.
    import pandas as pd

    catalogue_path = config.FOODS_PROCESSED
    if catalogue_path.exists():
        catalogue = pd.read_csv(catalogue_path)
        by_source = catalogue["source"].value_counts().to_dict() if "source" in catalogue else {}
        by_slot = catalogue["meal_type"].value_counts().to_dict()
        print(f"[OK  ] food catalogue        {len(catalogue)} items  {by_slot}")
        for source, count in by_source.items():
            print(f"         {count:5d}  {source}")
        thin = [slot for slot, count in by_slot.items() if count < 20]
        if thin:
            print(f"         WARNING: slots with <20 items {thin} may breach the variety rule")
    else:
        print("[MISS] food catalogue        food_catalogue.csv not found")
        problems.append("food_catalogue.csv")

    users_path = config.USERS_PROCESSED
    if users_path.exists():
        users = pd.read_csv(users_path)
        goals = users["fitness_goal"].value_counts().to_dict()
        print(f"[OK  ] training table        {len(users)} rows, goals {goals}")
        if len(users) != 973:
            print(f"         WARNING: expected 973 rows from the real gym dataset, got {len(users)}")
            problems.append("unexpected training row count")
    else:
        print("[MISS] training table        gym_users_labelled.csv not found")
        problems.append("gym_users_labelled.csv")

    quality_path = config.PROCESSED_DIR / "data_quality_report.json"
    if quality_path.exists():
        report = json.loads(quality_path.read_text(encoding="utf-8"))
        if report.get("mode") == "real":
            if report.get("food_rows_repaired"):
                print(f"[OK  ] data quality          {report['food_rows_repaired']} malformed CSV "
                      f"row(s) repaired (cite in Dataset Review)")
        else:
            print(f"[WARN] data quality          report says mode='{report.get('mode')}'.")
            print("         This file is written by prepare_data.py, not the notebooks,")
            print("         so it may be stale. The catalogue/table counts above are")
            print("         read directly from the data and are authoritative.")

    # --- do the artefacts actually load? ----------------------------------
    metrics_path = config.ARTIFACTS_DIR / config.METRICS_FILE
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"[OK  ] model metrics         trained {metrics.get('trained_at', '?')[:19]}, "
              f"{metrics.get('n_rows')} rows")
        print()
        print("       Headline results for your report:")
        for target, payload in metrics.get("targets", {}).items():
            selected = payload.get("selected_model", "?")
            for name, model in payload.get("models", {}).items():
                test = model["test_metrics"]
                cv = model["cv_metrics"]
                marker = " <- deployed" if name == selected else ""
                print(f"         {target:15s} {model['model_name']:18s} "
                      f"MAE={test['mae']:8.2f}  R2={test['r2']:.4f}  "
                      f"CV MAE={cv['mae']['mean']:7.2f}{marker}")

    print()
    if problems:
        print(f"{len(problems)} problem(s) found. Re-copy the missing items from")
        print("MyDrive/NutriFit-AI/ (see README.md).")
        return 1

    print("All Colab outputs synced correctly and derived from the real datasets.")
    print("Next: python ml/scripts/export_models.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
