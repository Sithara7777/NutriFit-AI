"""Stage 1: load raw datasets, clean, generate labels, build the food catalogue.

Usage
-----
    python ml/scripts/prepare_data.py                 # use data/raw/
    python ml/scripts/prepare_data.py --demo          # synthetic smoke-test data
    python ml/scripts/prepare_data.py --seed 42

Outputs
-------
    data/processed/gym_users_labelled.csv   supervised training table
    data/processed/food_catalogue.csv       recommendation-engine catalogue
    data/processed/foods_seed.sql           Postgres seed for the foods table
    data/processed/data_quality_report.json provenance + cleaning evidence
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make `nutrifit` importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from nutrifit import config, data, foods, labels  # noqa: E402
from nutrifit.labels import theoretical_r2_ceiling  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s"
)
logger = logging.getLogger("prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NutriFit-AI datasets.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic stand-in data instead of data/raw/ (smoke tests only).",
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--min-observations",
        type=int,
        default=None,
        help=(
            "Minimum rows before a (food, slot) pair enters the catalogue. "
            "Default: auto-detected from whether the source is a log or a catalogue."
        ),
    )
    parser.add_argument(
        "--with-usda",
        action="store_true",
        help=(
            "Merge USDA FoodData Central into the catalogue. OFF by default: USDA "
            "values are per 100 g while the Kaggle items are per serving, and mixing "
            "the two measurement bases degrades meal-plan accuracy. Enable only if "
            "the Kaggle catalogue is too thin for the variety constraint."
        ),
    )
    parser.add_argument(
        "--no-local-foods",
        action="store_true",
        help=(
            "Exclude the curated Sri Lankan food set (data/reference/). ON by "
            "default because the public datasets contain almost no South Asian "
            "food, making the recommender culturally biased for the target user. "
            "Use this flag to reproduce that biased baseline for comparison."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {"seed": args.seed, "mode": "demo" if args.demo else "real"}

    # ---------------------------------------------------------------- users
    if args.demo:
        logger.warning("=" * 70)
        logger.warning("RUNNING ON SYNTHETIC DEMO DATA -- results are NOT reportable.")
        logger.warning("=" * 70)
        from nutrifit.demo import make_demo_food_dataset, make_demo_gym_dataset
        from nutrifit.schema import (
            FOOD_ALIASES, FOOD_REQUIRED, GYM_ALIASES, GYM_REQUIRED, normalise_columns,
        )

        raw_gym = normalise_columns(
            make_demo_gym_dataset(seed=args.seed), GYM_ALIASES, GYM_REQUIRED, "demo gym"
        )
        raw_gym["height_cm"] = raw_gym["height_m"] * 100.0
        raw_gym["bmi"] = raw_gym["weight_kg"] / (raw_gym["height_m"] ** 2)
        gym = raw_gym

        food_log = normalise_columns(
            make_demo_food_dataset(seed=args.seed), FOOD_ALIASES, FOOD_REQUIRED, "demo food"
        )
        food_log["meal_type"] = food_log["meal_type"].str.lower()
        usda = None
        local_foods = None
    else:
        gym = data.load_gym_members()
        food_log = data.load_food_dataset()
        usda = data.load_usda_foundation() if args.with_usda else None
        if not args.with_usda:
            logger.info(
                "USDA enrichment disabled (per-100g basis differs from the "
                "per-serving Kaggle items). Enable with --with-usda if needed."
            )
        local_foods = None if args.no_local_foods else data.load_local_foods()
        if args.no_local_foods:
            logger.warning(
                "Curated regional foods DISABLED -- this reproduces the "
                "culturally biased baseline. See data/reference/README.md."
            )

    report["gym_rows"] = int(len(gym))
    report["gym_columns"] = list(gym.columns)
    report["food_rows_repaired"] = int(food_log.attrs.get("rows_repaired", 0))
    report["usda_enrichment"] = bool(args.with_usda)
    report["local_foods_enabled"] = bool(local_foods is not None)
    report["local_foods_count"] = int(len(local_foods)) if local_foods is not None else 0

    # --------------------------------------------------------------- labels
    labelled = labels.generate_labels(gym, seed=args.seed)
    labelled.to_csv(config.USERS_PROCESSED, index=False)

    goal_counts = labelled["fitness_goal"].value_counts().to_dict()
    report["fitness_goal_distribution"] = {k: int(v) for k, v in goal_counts.items()}
    report["activity_level_distribution"] = {
        k: int(v) for k, v in labelled["activity_level"].value_counts().to_dict().items()
    }
    report["bmi_category_distribution"] = {
        k: int(v) for k, v in labelled["bmi_category"].value_counts().to_dict().items()
    }
    report["theoretical_r2_ceiling"] = {
        "calorie_target": round(theoretical_r2_ceiling(labelled, "calorie_target"), 4),
        "protein_target": round(theoretical_r2_ceiling(labelled, "protein_target"), 4),
    }
    report["target_summary"] = {
        target: {
            "mean": round(float(labelled[target].mean()), 2),
            "std": round(float(labelled[target].std()), 2),
            "min": round(float(labelled[target].min()), 2),
            "max": round(float(labelled[target].max()), 2),
        }
        for target in ("calorie_target", "protein_target")
    }

    logger.info("Wrote %s (%d rows)", config.USERS_PROCESSED, len(labelled))
    logger.info("Goal distribution: %s", goal_counts)
    logger.info("Theoretical R2 ceiling: %s", report["theoretical_r2_ceiling"])

    # ------------------------------------------------------------ catalogue
    catalogue = foods.build_catalogue(
        food_log,
        usda=usda,
        local_foods=local_foods,
        min_observations=args.min_observations,
    )
    catalogue.to_csv(config.FOODS_PROCESSED, index=False)

    health = foods.catalogue_health_report(catalogue)
    report["catalogue_items"] = int(len(catalogue))
    report["catalogue_by_slot"] = health.to_dict(orient="records")
    report["catalogue_by_source"] = {
        str(k): int(v) for k, v in catalogue["source"].value_counts().to_dict().items()
    }

    # Cultural-coverage audit -- reproducible evidence for the report.
    coverage = foods.regional_coverage_report(catalogue)
    report["regional_coverage"] = coverage.to_dict(orient="records")

    config.FOODS_SEED_SQL.write_text(foods.to_sql_seed(catalogue), encoding="utf-8")

    logger.info("Wrote %s (%d items)", config.FOODS_PROCESSED, len(catalogue))
    logger.info("Wrote %s", config.FOODS_SEED_SQL)
    print("\nCatalogue coverage by meal slot:")
    print(health.to_string(index=False))
    print("\nCatalogue by source:")
    for source, count in catalogue["source"].value_counts().items():
        print(f"  {count:5d}  {source}")
    print("\nRegional coverage audit (Cardiff Met EDGE - GLOBAL evidence):")
    print(coverage.to_string(index=False))

    # ----------------------------------------------------------- provenance
    report_path = config.PROCESSED_DIR / "data_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", report_path)

    # A minimum viable catalogue is needed for the planner's variety rule.
    thin_slots = health[health["items"] < 10]["meal_type"].tolist()
    if thin_slots:
        logger.warning(
            "Slots with fewer than 10 items %s -- the 5-day variety rule will "
            "be hard to satisfy. Lower --min-observations to widen the pool.",
            thin_slots,
        )

    print("\nDone. Next: python ml/scripts/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
