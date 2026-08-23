"""Column-name normalisation for the raw public datasets.

Kaggle datasets get re-uploaded with slightly different headers (units in
parentheses, spaces vs underscores, different capitalisation).  Rather than
hard-coding one exact spelling and breaking the moment the publisher edits a
header, every raw file is passed through :func:`normalise_columns`, which maps
a set of known aliases onto the canonical internal names used everywhere else
in the codebase.

If a *required* column cannot be resolved the loader raises a
:class:`SchemaError` naming the missing field and listing what was actually
found -- a loud, actionable failure instead of a silent ``KeyError`` twenty
lines later.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

import pandas as pd


class SchemaError(ValueError):
    """Raised when a raw dataset does not contain a required column."""


def _canon_key(name: str) -> str:
    """Reduce a header to a comparison key: lowercase alphanumerics only.

    ``"Session_Duration (hours)"`` -> ``"sessiondurationhours"``
    """
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


# --------------------------------------------------------------------------
# Gym Members Exercise Dataset
#   https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset
# --------------------------------------------------------------------------
GYM_ALIASES: Mapping[str, tuple[str, ...]] = {
    "age": ("Age",),
    "gender": ("Gender", "Sex"),
    "weight_kg": ("Weight (kg)", "Weight_kg", "Weight"),
    "height_m": ("Height (m)", "Height_m"),
    "max_bpm": ("Max_BPM", "Max BPM"),
    "avg_bpm": ("Avg_BPM", "Avg BPM"),
    "resting_bpm": ("Resting_BPM", "Resting BPM"),
    "session_duration_h": ("Session_Duration (hours)", "Session_Duration", "Session Duration"),
    "calories_burned": ("Calories_Burned", "Calories Burned"),
    "workout_type": ("Workout_Type", "Workout Type"),
    "fat_percentage": ("Fat_Percentage", "Fat Percentage", "Body_Fat_Percentage"),
    "water_intake_l": ("Water_Intake (liters)", "Water_Intake", "Water Intake"),
    "workout_frequency": (
        "Workout_Frequency (days/week)",
        "Workout_Frequency",
        "Workout Frequency",
    ),
    "experience_level": ("Experience_Level", "Experience Level"),
    "bmi": ("BMI",),
}

GYM_REQUIRED = (
    "age",
    "gender",
    "weight_kg",
    "height_m",
    "session_duration_h",
    "workout_frequency",
    "experience_level",
)

# --------------------------------------------------------------------------
# Daily Food & Nutrition Dataset
#   https://www.kaggle.com/datasets/adilshamim8/daily-food-and-nutrition-dataset
# --------------------------------------------------------------------------
FOOD_ALIASES: Mapping[str, tuple[str, ...]] = {
    "food_item": ("Food_Item", "Food Item", "Food", "Description", "name"),
    "category": ("Category", "Food_Category", "Food Category"),
    "meal_type": ("Meal_Type", "Meal Type", "Meal"),
    "calories": ("Calories (kcal)", "Calories", "Energy (kcal)", "Energy"),
    "protein_g": ("Protein (g)", "Protein", "Protein_g"),
    "carbs_g": (
        "Carbohydrates (g)",
        "Carbohydrates",
        "Carbs (g)",
        "Carbs",
        "Carbohydrate (g)",
    ),
    "fat_g": ("Fat (g)", "Fat", "Total Fat (g)", "Fat_g", "Total_Fat"),
    "fiber_g": ("Fiber (g)", "Fiber", "Fibre (g)", "Dietary Fiber (g)"),
    "sugar_g": ("Sugars (g)", "Sugar (g)", "Sugars", "Sugar"),
    "sodium_mg": ("Sodium (mg)", "Sodium"),
    "cholesterol_mg": ("Cholesterol (mg)", "Cholesterol"),
}

FOOD_REQUIRED = ("food_item", "calories", "protein_g", "carbs_g", "fat_g")


def normalise_columns(
    df: pd.DataFrame,
    aliases: Mapping[str, Iterable[str]],
    required: Iterable[str] = (),
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """Rename ``df``'s columns to canonical names using ``aliases``.

    Matching is whitespace/punctuation/case insensitive.  Columns that match no
    alias are kept unchanged (they are simply ignored downstream), so an
    upstream publisher *adding* a column never breaks the pipeline.

    Raises
    ------
    SchemaError
        If any name listed in ``required`` could not be resolved.
    """
    lookup: dict[str, str] = {}
    for canonical, variants in aliases.items():
        # The canonical name itself is always an acceptable spelling, which
        # makes the function idempotent (safe to call twice).
        for variant in (canonical, *variants):
            lookup[_canon_key(variant)] = canonical

    rename: dict[str, str] = {}
    for column in df.columns:
        canonical = lookup.get(_canon_key(column))
        if canonical is not None and canonical not in rename.values():
            rename[column] = canonical

    out = df.rename(columns=rename)

    missing = [name for name in required if name not in out.columns]
    if missing:
        raise SchemaError(
            f"{dataset_name}: could not resolve required column(s) {missing}. "
            f"Columns found in the file: {list(df.columns)}. "
            f"Add the actual header spelling to the alias table in ml/nutrifit/schema.py."
        )
    return out


__all__ = [
    "SchemaError",
    "normalise_columns",
    "GYM_ALIASES",
    "GYM_REQUIRED",
    "FOOD_ALIASES",
    "FOOD_REQUIRED",
]
