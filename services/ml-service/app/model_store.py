"""Artefact loading with graceful degradation.

The Reliability non-functional requirement says the system must not crash when
a model file is unavailable.  This module therefore loads what it can at
start-up and records what it could not, and the prediction path falls back to
the deterministic formulas in :mod:`nutrifit.nutrition` when a pipeline is
missing.  A user still gets a scientifically-defensible answer; the response
simply reports ``source="formula"`` instead of ``source="model"`` so the
degradation is visible rather than silent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from nutrifit import config, foods, nutrition
from nutrifit.preprocessing import FEATURE_COLUMNS
from nutrifit.recommender import MealRecommender

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelStore:
    """Holds the loaded pipelines and the recommendation engine."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.calorie_model: Any | None = None
        self.protein_model: Any | None = None
        self.recommender: MealRecommender | None = None
        self.model_version: str = config.MODEL_VERSION
        self.errors: list[str] = []

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        self.calorie_model = self._load_pickle(config.CALORIE_MODEL_FILE)
        self.protein_model = self._load_pickle(config.PROTEIN_MODEL_FILE)
        self.recommender = self._load_recommender()

        card = self.models_dir / config.MODEL_CARD_FILE
        if card.exists():
            try:
                import json

                self.model_version = json.loads(card.read_text(encoding="utf-8")).get(
                    "version", config.MODEL_VERSION
                )
            except Exception as error:  # noqa: BLE001
                logger.warning("Could not read model card: %s", error)

        if self.errors:
            logger.warning("Service starting in DEGRADED mode: %s", self.errors)
        else:
            logger.info("All artefacts loaded (model version %s)", self.model_version)

    def _load_pickle(self, filename: str) -> Any | None:
        path = self.models_dir / filename
        if not path.exists():
            message = f"{filename} not found in {self.models_dir}"
            self.errors.append(message)
            logger.warning("%s -- falling back to formula predictions", message)
            return None
        try:
            return joblib.load(path)
        except Exception as error:  # noqa: BLE001
            message = f"{filename} failed to load: {error}"
            self.errors.append(message)
            logger.error(message)
            return None

    def _load_recommender(self) -> MealRecommender | None:
        path = self.models_dir / config.RECOMMENDER_FILE
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as error:  # noqa: BLE001
                self.errors.append(f"{config.RECOMMENDER_FILE} failed to load: {error}")
                logger.error("Recommender failed to load: %s", error)

        # Second chance: rebuild from the processed catalogue CSV if present.
        if config.FOODS_PROCESSED.exists():
            try:
                catalogue = foods.add_derived_features(pd.read_csv(config.FOODS_PROCESSED))
                logger.info("Rebuilt recommender from %s", config.FOODS_PROCESSED)
                return MealRecommender(catalogue)
            except Exception as error:  # noqa: BLE001
                self.errors.append(f"catalogue rebuild failed: {error}")
                logger.error("Catalogue rebuild failed: %s", error)

        self.errors.append("no recommendation engine available")
        return None

    # --------------------------------------------------------------- status
    @property
    def healthy(self) -> bool:
        return not self.errors

    def status(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.healthy else "degraded",
            "model_version": self.model_version,
            "models_loaded": {
                "calorie_model": self.calorie_model is not None,
                "protein_model": self.protein_model is not None,
                "recommender": self.recommender is not None,
            },
            "catalogue_items": (
                len(self.recommender.catalogue) if self.recommender is not None else 0
            ),
            "detail": "; ".join(self.errors) if self.errors else None,
        }

    # ------------------------------------------------------------- predict
    def predict(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Predict calorie and protein targets for one user profile.

        Always returns the derived physiology (BMI, BMR, TDEE) computed from the
        formulas, plus the model prediction when a pipeline is available.
        """
        # Deterministic physiology -- also the fallback answer.
        formula = nutrition.formula_targets(
            weight_kg=profile["weight_kg"],
            height_cm=profile["height_cm"],
            age=profile["age"],
            gender=profile["gender"],
            goal=profile["fitness_goal"],
            workout_frequency=profile["workout_frequency"],
            session_duration_h=profile.get("session_duration_h", 1.25),
            experience_level=profile.get("experience_level", 2),
            body_fat_pct=profile.get("body_fat_pct"),
        )

        features = pd.DataFrame([{
            "age": profile["age"],
            "height_cm": profile["height_cm"],
            "weight_kg": profile["weight_kg"],
            "bmi": formula["bmi"],
            "body_fat_pct": formula["body_fat_pct"],
            "workout_frequency": profile["workout_frequency"],
            "session_duration_h": profile.get("session_duration_h", 1.25),
            "experience_level": profile.get("experience_level", 2),
            "gender": profile["gender"],
            "fitness_goal": profile["fitness_goal"],
            "activity_level": formula["activity_level"],
        }])[FEATURE_COLUMNS]

        calorie_target = formula["calorie_target"]
        protein_target = formula["protein_target"]
        source = "formula"

        if self.calorie_model is not None and self.protein_model is not None:
            try:
                calorie_target = float(self.calorie_model.predict(features)[0])
                protein_target = float(self.protein_model.predict(features)[0])
                source = "model"
            except Exception as error:  # noqa: BLE001
                # A prediction failure must degrade, not 500.
                logger.error("Prediction failed, using formula fallback: %s", error)

        # Safety clamp: never return a target outside physiological bounds,
        # whatever the model says.
        floor = max(
            nutrition.ABSOLUTE_MIN_CALORIES,
            formula["bmr"] * nutrition.MIN_CALORIES_AS_BMR_MULTIPLE,
        )
        calorie_target = float(min(max(calorie_target, floor), 6000.0))
        protein_target = float(min(max(protein_target, 40.0), 350.0))

        return {
            "calorie_target": round(calorie_target, 1),
            "protein_target": round(protein_target, 1),
            "bmi": formula["bmi"],
            "bmi_category": formula["bmi_category"],
            "bmr": formula["bmr"],
            "bmr_equation": formula["bmr_equation"],
            "tdee": formula["tdee"],
            "body_fat_pct": formula["body_fat_pct"],
            "body_fat_source": formula["body_fat_source"],
            "activity_level": formula["activity_level"],
            "activity_multiplier": formula["activity_multiplier"],
            "model_version": self.model_version,
            "source": source,
            "formula_reference": {
                "calorie_target": formula["calorie_target"],
                "protein_target": formula["protein_target"],
            },
        }


store = ModelStore()
