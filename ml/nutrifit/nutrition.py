"""Domain layer: established sports-nutrition formulas.

This module is the single source of truth for every physiological calculation
in the project.  It is imported by:

* :mod:`nutrifit.labels`   -- to generate the supervised training targets
* the FastAPI ML service   -- as the deterministic fallback when a model
                              artefact fails to load (Reliability NFR)
* the evaluation notebook  -- as the "formula baseline" the ML models are
                              compared against

References
----------
Mifflin, M. D., St Jeor, S. T., Hill, L. A., Scott, B. J., Daugherty, S. A., &
    Koh, Y. O. (1990). A new predictive equation for resting energy expenditure
    in healthy individuals. *American Journal of Clinical Nutrition*, 51(2),
    241-247.
Katch, F. I., & McArdle, W. D. (1996). *Introduction to Nutrition, Exercise and
    Health* (4th ed.). Williams & Wilkins.  -- lean-mass BMR equation, the
    preferred estimator for athletic populations with a known body-fat figure.
Deurenberg, P., Weststrate, J. A., & Seidell, J. C. (1991). Body mass index as
    a measure of body fatness: age- and sex-specific prediction formulas.
    *British Journal of Nutrition*, 65(2), 105-114.
FAO/WHO/UNU (2004). *Human energy requirements*. FAO Food and Nutrition
    Technical Report Series 1.  -- physical activity level (PAL) multipliers.
Jager, R., Kerksick, C. M., Campbell, B. I., et al. (2017). International
    Society of Sports Nutrition Position Stand: protein and exercise.
    *Journal of the International Society of Sports Nutrition*, 14, 20.
Helms, E. R., Zinn, C., Rowlands, D. S., & Brown, S. R. (2014). A systematic
    review of dietary protein during caloric restriction in resistance-trained
    lean athletes. *Int. J. Sport Nutrition and Exercise Metabolism*, 24(2).
WHO (2010). *A healthy lifestyle - WHO recommendations* (BMI categories).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
GOALS = ("fat_loss", "maintenance", "muscle_gain")

ACTIVITY_LEVELS = ("sedentary", "light", "moderate", "active", "very_active")

#: FAO/WHO/UNU physical-activity-level (PAL) multipliers applied to BMR.
ACTIVITY_MULTIPLIERS: dict[str, float] = {
    "sedentary": 1.20,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.90,
}

#: Goal -> energy adjustment applied to TDEE (see ``goal_calorie_factor``).
GOAL_CALORIE_FACTORS: dict[str, float] = {
    "fat_loss": 0.80,      # ~20 % deficit
    "maintenance": 1.00,
    "muscle_gain": 1.12,   # ~12 % surplus (ISSN suggests 10-15 %)
}

#: Goal -> protein target in grams per kg of bodyweight (ISSN / Helms et al.).
GOAL_PROTEIN_COEFFICIENTS: dict[str, float] = {
    "fat_loss": 2.20,      # 2.0-2.4 g/kg preserves lean mass in a deficit
    "maintenance": 1.60,
    "muscle_gain": 2.00,   # 1.8-2.2 g/kg supports hypertrophy
}

#: Fraction of the daily calorie/protein budget allocated to each meal slot.
MEAL_SLOTS = ("breakfast", "lunch", "dinner", "snack")
MEAL_SLOT_RATIOS: dict[str, float] = {
    "breakfast": 0.25,
    "lunch": 0.35,
    "dinner": 0.30,
    "snack": 0.10,
}

#: A calorie floor expressed as a multiple of BMR.  Prevents the fat-loss
#: adjustment from ever prescribing a dangerously low intake for a small,
#: sedentary user -- a safety constraint, not a modelling convenience.
MIN_CALORIES_AS_BMR_MULTIPLE = 1.10
ABSOLUTE_MIN_CALORIES = 1200.0


# --------------------------------------------------------------------------
# Body composition
# --------------------------------------------------------------------------
def calculate_bmi(weight_kg, height_cm):
    """Body Mass Index in kg/m^2.  Scalar or array-safe."""
    height_m = np.asarray(height_cm, dtype=float) / 100.0
    return np.asarray(weight_kg, dtype=float) / np.square(height_m)


def bmi_category(bmi) -> str | np.ndarray:
    """WHO BMI classification.

    Returns a plain ``str`` for scalar input and an ``ndarray`` of strings for
    array input, so it can be used both per-request in the API and
    column-wise in pandas.
    """
    values = np.asarray(bmi, dtype=float)
    categories = np.select(
        [values < 18.5, values < 25.0, values < 30.0],
        ["underweight", "normal", "overweight"],
        default="obese",
    )
    if np.ndim(bmi) == 0:
        return str(categories)
    return categories


# --------------------------------------------------------------------------
# Energy expenditure
# --------------------------------------------------------------------------
def _is_male(gender) -> np.ndarray:
    gender_arr = np.char.lower(np.asarray(gender, dtype=str).astype("U16"))
    return np.isin(gender_arr, ["male", "m", "man", "1"])


def calculate_bmr(weight_kg, height_cm, age, gender) -> np.ndarray:
    """Basal Metabolic Rate via the Mifflin-St Jeor equation (1990).

    ``Male:   10*kg + 6.25*cm - 5*age + 5``
    ``Female: 10*kg + 6.25*cm - 5*age - 161``

    ``gender`` accepts ``"Male"``/``"Female"``/``"M"``/``"F"`` in any case.
    Unrecognised values fall back to the female constant, which is the
    conservative (lower) estimate.
    """
    weight = np.asarray(weight_kg, dtype=float)
    height = np.asarray(height_cm, dtype=float)
    years = np.asarray(age, dtype=float)

    constant = np.where(_is_male(gender), 5.0, -161.0)
    return 10.0 * weight + 6.25 * height - 5.0 * years + constant


def lean_body_mass(weight_kg, body_fat_pct) -> np.ndarray:
    """Fat-free mass in kg."""
    weight = np.asarray(weight_kg, dtype=float)
    fat = np.clip(np.asarray(body_fat_pct, dtype=float), 3.0, 60.0)
    return weight * (1.0 - fat / 100.0)


def calculate_bmr_katch_mcardle(weight_kg, body_fat_pct) -> np.ndarray:
    """BMR from lean mass: ``370 + 21.6 x LBM`` (Katch & McArdle, 1996).

    Preferred over Mifflin-St Jeor for resistance-trained populations, where
    body composition varies far more than height/weight alone can express:
    two 85 kg men at 12 % and 30 % body fat have materially different resting
    expenditure but identical Mifflin-St Jeor estimates.
    """
    return 370.0 + 21.6 * lean_body_mass(weight_kg, body_fat_pct)


def estimate_body_fat(bmi, age, gender) -> np.ndarray:
    """Deurenberg (1991) body-fat estimate from BMI, age and sex.

    Used only when a measured figure is unavailable, so the system always has
    a body-composition input.  The estimate is deterministic in its inputs and
    therefore adds no information beyond BMI/age/sex -- which is exactly why
    :func:`calculate_bmr_best` falls back to Mifflin-St Jeor in that case
    rather than feeding a derived number into a lean-mass equation.
    """
    bmi_arr = np.asarray(bmi, dtype=float)
    age_arr = np.asarray(age, dtype=float)
    sex = np.where(_is_male(gender), 1.0, 0.0)
    return np.clip(1.20 * bmi_arr + 0.23 * age_arr - 10.8 * sex - 5.4, 3.0, 60.0)


def calculate_bmr_best(weight_kg, height_cm, age, gender, body_fat_pct=None) -> np.ndarray:
    """Best-available BMR estimate.

    Katch-McArdle where body fat is genuinely measured; Mifflin-St Jeor
    otherwise.  Mixing the two by availability is standard practice and is the
    reason ``body_fat_source`` is tracked on the user profile.
    """
    mifflin = calculate_bmr(weight_kg, height_cm, age, gender)
    if body_fat_pct is None:
        return mifflin

    fat = np.asarray(body_fat_pct, dtype=float)
    katch = calculate_bmr_katch_mcardle(weight_kg, fat)
    return np.where(np.isnan(fat), mifflin, katch)


def derive_activity_level(workout_frequency, session_duration_h) -> np.ndarray:
    """Map weekly training volume onto a FAO/WHO activity band.

    The gym dataset does not carry a lifestyle activity field, so the band is
    derived from *weekly training hours* (``frequency x session duration``),
    which is the only defensible activity signal actually present in the data.
    Cut-points are chosen to align the resulting distribution with the standard
    sedentary -> extra-active PAL scale.
    """
    weekly_hours = np.asarray(workout_frequency, dtype=float) * np.asarray(
        session_duration_h, dtype=float
    )
    return np.select(
        [weekly_hours < 1.5, weekly_hours < 3.0, weekly_hours < 5.0, weekly_hours < 7.0],
        ["sedentary", "light", "moderate", "active"],
        default="very_active",
    )


def activity_multiplier(activity_level, experience_level=None) -> np.ndarray:
    """PAL multiplier for an activity band, nudged by training experience.

    Experience applies a bounded +/-0.025 adjustment per level away from
    intermediate (level 2): a level-3 lifter trains at a higher relative
    intensity than a novice at the same session count.  The result is always
    clipped to the [1.20, 1.90] range of the published PAL scale.
    """
    levels = np.asarray(activity_level, dtype=str)
    base = np.vectorize(lambda lv: ACTIVITY_MULTIPLIERS.get(str(lv), 1.375))(levels)
    base = np.asarray(base, dtype=float)

    if experience_level is not None:
        experience = np.asarray(experience_level, dtype=float)
        base = base + 0.025 * (experience - 2.0)

    return np.clip(base, 1.20, 1.90)


def calculate_tdee(bmr, multiplier) -> np.ndarray:
    """Total Daily Energy Expenditure = BMR x PAL multiplier."""
    return np.asarray(bmr, dtype=float) * np.asarray(multiplier, dtype=float)


# --------------------------------------------------------------------------
# Goal-adjusted targets
# --------------------------------------------------------------------------
def _map_goal(goal, table: dict[str, float], default_key: str) -> np.ndarray:
    goals = np.asarray(goal, dtype=str)
    return np.asarray(
        np.vectorize(lambda g: table.get(str(g), table[default_key]))(goals), dtype=float
    )


def goal_calorie_factor(goal) -> np.ndarray:
    """Energy adjustment factor for a fitness goal."""
    return _map_goal(goal, GOAL_CALORIE_FACTORS, "maintenance")


def goal_protein_coefficient(goal) -> np.ndarray:
    """Protein target in g/kg bodyweight for a fitness goal."""
    return _map_goal(goal, GOAL_PROTEIN_COEFFICIENTS, "maintenance")


def calorie_target(tdee, goal, bmr=None, factor=None) -> np.ndarray:
    """Goal-adjusted daily calorie target, with a safety floor applied.

    ``factor`` may be supplied to override the table lookup (used by the label
    generator, which jitters the factor within its published range).
    """
    tdee_arr = np.asarray(tdee, dtype=float)
    factors = goal_calorie_factor(goal) if factor is None else np.asarray(factor, dtype=float)
    target = tdee_arr * factors

    floor = np.full_like(target, ABSOLUTE_MIN_CALORIES)
    if bmr is not None:
        floor = np.maximum(
            floor, np.asarray(bmr, dtype=float) * MIN_CALORIES_AS_BMR_MULTIPLE
        )
    return np.maximum(target, floor)


def protein_target(weight_kg, goal, coefficient=None) -> np.ndarray:
    """Daily protein target in grams."""
    weight = np.asarray(weight_kg, dtype=float)
    coeffs = (
        goal_protein_coefficient(goal)
        if coefficient is None
        else np.asarray(coefficient, dtype=float)
    )
    return weight * coeffs


def formula_targets(
    *,
    weight_kg: float,
    height_cm: float,
    age: float,
    gender: str,
    goal: str,
    workout_frequency: float,
    session_duration_h: float = 1.25,
    experience_level: float = 2,
    body_fat_pct: float | None = None,
) -> dict[str, float]:
    """End-to-end deterministic estimate for a single user.

    Used by the ML service as a graceful-degradation fallback when a model
    artefact is unavailable, and by the evaluation notebook as the non-ML
    baseline the trained models are scored against.
    """
    bmi = float(calculate_bmi(weight_kg, height_cm))

    if body_fat_pct is None:
        fat = float(estimate_body_fat(bmi, age, gender))
        fat_source = "estimated_deurenberg"
        bmr = float(calculate_bmr(weight_kg, height_cm, age, gender))
        equation = "Mifflin-St Jeor"
    else:
        fat = float(body_fat_pct)
        fat_source = "measured"
        bmr = float(calculate_bmr_katch_mcardle(weight_kg, fat))
        equation = "Katch-McArdle"

    level = str(derive_activity_level(workout_frequency, session_duration_h))
    multiplier = float(activity_multiplier(level, experience_level))
    tdee = float(calculate_tdee(bmr, multiplier))

    return {
        "bmi": round(bmi, 2),
        "bmi_category": str(bmi_category(bmi)),
        "body_fat_pct": round(fat, 1),
        "body_fat_source": fat_source,
        "bmr": round(bmr, 1),
        "bmr_equation": equation,
        "activity_level": level,
        "activity_multiplier": round(multiplier, 4),
        "tdee": round(tdee, 1),
        "calorie_target": round(float(calorie_target(tdee, goal, bmr=bmr)), 1),
        "protein_target": round(float(protein_target(weight_kg, goal)), 1),
    }


def split_across_meals(daily_total: float, slots: Iterable[str] = MEAL_SLOTS) -> dict[str, float]:
    """Divide a daily budget across meal slots using the standard ratios."""
    return {slot: daily_total * MEAL_SLOT_RATIOS[slot] for slot in slots}


__all__ = [
    "GOALS",
    "ACTIVITY_LEVELS",
    "ACTIVITY_MULTIPLIERS",
    "GOAL_CALORIE_FACTORS",
    "GOAL_PROTEIN_COEFFICIENTS",
    "MEAL_SLOTS",
    "MEAL_SLOT_RATIOS",
    "calculate_bmi",
    "bmi_category",
    "calculate_bmr",
    "calculate_bmr_katch_mcardle",
    "calculate_bmr_best",
    "lean_body_mass",
    "estimate_body_fat",
    "derive_activity_level",
    "activity_multiplier",
    "calculate_tdee",
    "goal_calorie_factor",
    "goal_protein_coefficient",
    "calorie_target",
    "protein_target",
    "formula_targets",
    "split_across_meals",
]
