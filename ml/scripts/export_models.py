"""Stage 3: copy trained artefacts into the FastAPI service and verify them.

Reloading each artefact and running a prediction here -- rather than trusting
that ``joblib.dump`` worked -- is what catches a version mismatch or a missing
custom transformer at build time instead of at 3 a.m. in a live demo.

Usage
-----
    python ml/scripts/export_models.py
    python ml/scripts/export_models.py --source /content/drive/MyDrive/NutriFit-AI/ml/artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from nutrifit import config  # noqa: E402
from nutrifit.preprocessing import FEATURE_COLUMNS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("export")

def _sanity_profile() -> pd.DataFrame:
    """A known-good request used to smoke-test every reloaded artefact.

    Built from ``FEATURE_COLUMNS`` rather than hard-coded, so adding a feature
    to the contract cannot leave a stale fixture silently behind.
    """
    reference = {
        "age": 28,
        "height_cm": 178.0,
        "weight_kg": 82.0,
        "bmi": 82.0 / (1.78**2),
        "body_fat_pct": 18.0,
        "workout_frequency": 4,
        "session_duration_h": 1.25,
        "experience_level": 2,
        "gender": "Male",
        "fitness_goal": "muscle_gain",
        "activity_level": "moderate",
    }
    missing = [column for column in FEATURE_COLUMNS if column not in reference]
    if missing:
        raise RuntimeError(
            f"Sanity profile is missing feature(s) {missing}. "
            f"Update _sanity_profile() in ml/scripts/export_models.py."
        )
    return pd.DataFrame([reference])


SANITY_PROFILE = _sanity_profile()

#: Physiologically plausible output ranges.  A reloaded model that predicts
#: outside these has been corrupted or trained on the wrong target.
EXPECTED_RANGES = {
    config.CALORIE_MODEL_FILE: (1200.0, 5000.0),
    config.PROTEIN_MODEL_FILE: (50.0, 300.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export models to the ML service.")
    parser.add_argument("--source", type=Path, default=config.ARTIFACTS_DIR)
    parser.add_argument("--dest", type=Path, default=config.SERVICE_MODELS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source: Path = args.source
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    required = [config.CALORIE_MODEL_FILE, config.PROTEIN_MODEL_FILE]
    optional = [config.RECOMMENDER_FILE, config.METRICS_FILE, config.MODEL_CARD_FILE]

    missing = [name for name in required if not (source / name).exists()]
    if missing:
        logger.error("Missing artefact(s) in %s: %s", source, missing)
        logger.error("Run: python ml/scripts/train.py")
        return 1

    failures: list[str] = []

    for name in required + optional:
        artefact = source / name
        if not artefact.exists():
            logger.warning("Optional artefact not found, skipping: %s", name)
            continue

        shutil.copy2(artefact, dest / name)
        size_kb = artefact.stat().st_size / 1024
        logger.info("Copied %-22s -> %s (%.1f KB)", name, dest, size_kb)

        # ---------------- verification ----------------------------------
        if name in EXPECTED_RANGES:
            try:
                pipeline = joblib.load(dest / name)
                prediction = float(pipeline.predict(SANITY_PROFILE[FEATURE_COLUMNS])[0])
            except Exception as error:  # noqa: BLE001 - report, do not mask
                failures.append(f"{name}: reload/predict failed -- {error}")
                logger.error("  VERIFY FAILED %s: %s", name, error)
                continue

            low, high = EXPECTED_RANGES[name]
            if not low <= prediction <= high:
                failures.append(
                    f"{name}: prediction {prediction:.1f} outside expected [{low}, {high}]"
                )
                logger.error("  VERIFY FAILED %s: predicted %.1f", name, prediction)
            else:
                logger.info("  Verified: predicts %.1f for the reference profile", prediction)

        elif name == config.RECOMMENDER_FILE:
            try:
                engine = joblib.load(dest / name)
                suggestions = engine.recommend("breakfast", 2400, 150, "muscle_gain", top_n=3)
                if not suggestions:
                    failures.append(f"{name}: returned no suggestions")
                else:
                    logger.info(
                        "  Verified: top breakfast = %s (%.0f kcal, %.0f g protein)",
                        suggestions[0].name, suggestions[0].calories, suggestions[0].protein_g,
                    )
            except Exception as error:  # noqa: BLE001
                failures.append(f"{name}: reload failed -- {error}")
                logger.error("  VERIFY FAILED %s: %s", name, error)

    if failures:
        print("\nEXPORT FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    metrics_file = dest / config.METRICS_FILE
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        print("\nExported model summary:")
        for target, payload in metrics.get("targets", {}).items():
            selected = payload.get("selected_model", "?")
            test = payload["models"][selected]["test_metrics"]
            print(
                f"  {target:16s} {selected:18s} "
                f"MAE={test['mae']:.2f}  RMSE={test['rmse']:.2f}  R2={test['r2']:.4f}"
            )

    print(f"\nAll artefacts verified in {dest}")
    print("Next: cd services/ml-service && uvicorn app.main:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
