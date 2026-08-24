"""Stage 2: train, evaluate and export the calorie and protein models.

Usage
-----
    python ml/scripts/train.py                 # full run with tuning
    python ml/scripts/train.py --no-tune       # fast smoke test
    python ml/scripts/train.py --n-iter 60     # wider hyper-parameter search

Outputs
-------
    ml/artifacts/calorie_model.pkl    fitted Pipeline (preprocessing + model)
    ml/artifacts/protein_model.pkl
    ml/artifacts/recommender.pkl      fitted MealRecommender
    ml/artifacts/model_metrics.json   full metric set for the report
    ml/artifacts/model_card.json      provenance and reproducibility record
    ml/reports/model_comparison.csv   LR vs RF comparison table
    ml/reports/feature_importance_*.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from nutrifit import config, foods, training  # noqa: E402
from nutrifit.preprocessing import FEATURE_COLUMNS, select_features  # noqa: E402
from nutrifit.recommender import MealRecommender, evaluate_recommender  # noqa: E402
from nutrifit.planner import generate_meal_plan  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("train")

TARGETS = ("calorie_target", "protein_target")
MODEL_FILES = {
    "calorie_target": config.CALORIE_MODEL_FILE,
    "protein_target": config.PROTEIN_MODEL_FILE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NutriFit-AI models.")
    parser.add_argument("--no-tune", action="store_true", help="Skip RandomizedSearchCV.")
    parser.add_argument("--n-iter", type=int, default=40, help="RandomizedSearchCV iterations.")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--skip-learning-curve",
        action="store_true",
        help="Skip the LR-vs-RF learning curve (slow; needed for the report).",
    )
    parser.add_argument(
        "--refit-full",
        action="store_true",
        default=True,
        help="Refit the winning model on 100%% of the data before export (default on).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not config.USERS_PROCESSED.exists():
        logger.error(
            "%s not found. Run: python ml/scripts/prepare_data.py",
            config.USERS_PROCESSED,
        )
        return 1

    df = pd.read_csv(config.USERS_PROCESSED)
    logger.info("Loaded %d labelled rows, %d features", len(df), len(FEATURE_COLUMNS))

    metrics: dict = {
        "model_version": config.MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "features": FEATURE_COLUMNS,
        "seed": args.seed,
        "targets": {},
    }
    bundles = []

    # ------------------------------------------------------------- training
    for target in TARGETS:
        logger.info("=" * 68)
        logger.info("Training target: %s", target)
        logger.info("=" * 68)

        bundle = training.train_target(
            df, target, tune=not args.no_tune, n_iter=args.n_iter, seed=args.seed
        )
        bundles.append(bundle)

        baseline = training.formula_baseline_metrics(df, target)
        metrics["targets"][target] = {
            "formula_baseline": baseline,
            "models": {
                name: result.to_dict() for name, result in bundle["results"].items()
            },
        }

        for name, result in bundle["results"].items():
            test = result.test_metrics
            logger.info(
                "  %-18s MAE=%8.3f  RMSE=%8.3f  R2=%.4f  (CV R2 %.4f +/- %.4f)",
                name, test["mae"], test["rmse"], test["r2"],
                result.cv_metrics["r2"]["mean"], result.cv_metrics["r2"]["std"],
            )

        # --- select and export the better model on cross-validated MAE ----
        winner_name = min(
            bundle["results"],
            key=lambda key: bundle["results"][key].cv_metrics["mae"]["mean"],
        )
        winner = bundle["pipelines"][winner_name]
        metrics["targets"][target]["selected_model"] = winner_name
        logger.info("  Selected for export: %s", winner_name)

        if args.refit_full:
            # Refit on 100 % of the data: the split existed to produce an
            # honest estimate, and that estimate is already recorded above.
            # The shipped model should see every available row.
            winner.fit(select_features(df), df[target].astype(float))
            metrics["targets"][target]["refit_on_full_data"] = True

        output = config.ARTIFACTS_DIR / MODEL_FILES[target]
        joblib.dump(winner, output, compress=3)
        logger.info("  Exported %s (%.1f KB)", output, output.stat().st_size / 1024)

        bundle["importances"].to_csv(
            config.REPORTS_DIR / f"feature_importance_{target}.csv", index=False
        )
        bundle["coefficients"].to_csv(
            config.REPORTS_DIR / f"coefficients_{target}.csv", index=False
        )
        training.residual_frame(bundle, "RandomForest").to_csv(
            config.REPORTS_DIR / f"residuals_{target}.csv", index=False
        )

        # Learning curve: evidence for *why* one model family wins here.
        if not args.skip_learning_curve:
            curve = training.learning_curve_comparison(df, target, seed=args.seed)
            curve.to_csv(config.REPORTS_DIR / f"learning_curve_{target}.csv", index=False)
            pivot = curve.pivot(index="n_samples", columns="model", values="cv_mae_mean")
            metrics["targets"][target]["learning_curve"] = curve.to_dict(orient="records")
            logger.info("  Learning curve (CV MAE by training size):\n%s",
                        pivot.round(3).to_string())

    # ------------------------------------------------------ comparison table
    comparison = training.comparison_table(bundles)
    comparison_path = config.REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print("\n" + "=" * 100)
    print("MODEL COMPARISON  (Linear Regression vs Random Forest)")
    print("=" * 100)
    print(comparison.to_string(index=False))
    print()

    # ---------------------------------------------------------- recommender
    if config.FOODS_PROCESSED.exists():
        catalogue = pd.read_csv(config.FOODS_PROCESSED)
        catalogue = foods.add_derived_features(catalogue)
        engine = MealRecommender(catalogue, random_state=args.seed)

        recommender_path = config.ARTIFACTS_DIR / config.RECOMMENDER_FILE
        joblib.dump(engine, recommender_path, compress=3)
        logger.info("Exported %s", recommender_path)

        # Evaluate the recommender the way the report needs it evaluated.
        simulation = evaluate_recommender(
            engine, calorie_target=2400, protein_target=150,
            goal="muscle_gain", n_days=30, seed=args.seed,
        )
        plan = generate_meal_plan(
            engine, calorie_target=2400, protein_target=150,
            goal="muscle_gain", weeks=8, seed=args.seed,
        )
        weekly = plan.weekly_summary()

        metrics["recommender"] = {
            "catalogue_items": int(len(catalogue)),
            "items_per_slot": catalogue["meal_type"].value_counts().to_dict(),
            "simulation_30_days": {
                "mean_calorie_error_pct": round(
                    float(simulation["calorie_error_pct"].abs().mean()), 2
                ),
                "mean_protein_error_pct": round(
                    float(simulation["protein_error_pct"].abs().mean()), 2
                ),
                "unique_meals": int(simulation.attrs["unique_meals"]),
                "variety_ratio": round(float(simulation.attrs["variety_ratio"]), 4),
            },
            "eight_week_plan": {
                "variety": plan.variety_report(),
                "repairs_applied": plan.repairs_applied,
                "max_abs_weekly_calorie_error_pct": round(
                    float(weekly["calorie_error_pct"].abs().max()), 2
                ),
                "max_abs_weekly_protein_error_pct": round(
                    float(weekly["protein_error_pct"].abs().max()), 2
                ),
            },
        }
        weekly.to_csv(config.REPORTS_DIR / "meal_plan_weekly_summary.csv", index=False)
        plan.to_dataframe().to_csv(config.REPORTS_DIR / "meal_plan_sample.csv", index=False)

        print("=" * 100)
        print("EIGHT-WEEK MEAL PLAN -- weekly macro accuracy")
        print("=" * 100)
        print(weekly.to_string(index=False))
        print(f"\nVariety: {plan.variety_report()}")
        print()
    else:
        logger.warning("No food catalogue found; skipping recommender export.")

    # ------------------------------------------------------------- artefacts
    metrics_path = config.ARTIFACTS_DIR / config.METRICS_FILE
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", metrics_path)

    model_card = {
        "name": "NutriFit-AI calorie & protein requirement models",
        "version": config.MODEL_VERSION,
        "trained_at": metrics["trained_at"],
        "algorithms": ["LinearRegression", "RandomForestRegressor"],
        "selected": {
            target: metrics["targets"][target]["selected_model"] for target in TARGETS
        },
        "features": FEATURE_COLUMNS,
        "targets": list(TARGETS),
        "training_rows": int(len(df)),
        "label_construction": (
            "Mifflin-St Jeor BMR -> FAO/WHO PAL multiplier -> goal-specific energy "
            "adjustment and ISSN protein coefficient, sampled within published ranges "
            "with Gaussian residual noise. See ml/nutrifit/labels.py."
        ),
        "intended_use": (
            "Nutritional guidance for healthy adult gym users. Not a medical device; "
            "not for use with clinical populations, pregnancy, or eating disorders."
        ),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }
    card_path = config.ARTIFACTS_DIR / config.MODEL_CARD_FILE
    card_path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    logger.info("Wrote %s", card_path)

    print("Done. Next: python ml/scripts/export_models.py "
          "(copies artefacts into the FastAPI service)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
