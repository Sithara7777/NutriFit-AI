"""Synthetic stand-in datasets for smoke-testing the pipeline.

**Not for the report.**  These generators exist so the full pipeline can be
executed and tested before the real Kaggle files are downloaded, and so CI /
unit tests never depend on a manual download.  Every training run intended for
the dissertation must use the real datasets in ``data/raw/`` -- the scripts
print a loud warning whenever they fall back to these.

The generated distributions are drawn to loosely resemble the real Gym Members
dataset (age 18-59, both sexes, 973 rows) so that a smoke test exercises
realistic value ranges, but no result produced from this data is reportable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .nutrition import MEAL_SLOTS

DEMO_SEED = 7


def make_demo_gym_dataset(n_rows: int = 973, seed: int = DEMO_SEED) -> pd.DataFrame:
    """Synthetic frame matching the *raw* Gym Members schema (pre-normalisation)."""
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n_rows, p=[0.53, 0.47])
    is_male = gender == "Male"

    age = rng.integers(18, 60, size=n_rows)
    height_m = np.where(
        is_male, rng.normal(1.76, 0.07, n_rows), rng.normal(1.63, 0.065, n_rows)
    ).clip(1.45, 2.05)
    weight_kg = np.where(
        is_male, rng.normal(82, 13, n_rows), rng.normal(66, 11, n_rows)
    ).clip(40, 160)

    workout_frequency = rng.integers(2, 6, size=n_rows)
    session_duration = rng.normal(1.25, 0.35, n_rows).clip(0.5, 2.5)
    experience_level = rng.choice([1, 2, 3], size=n_rows, p=[0.38, 0.42, 0.20])

    fat_percentage = np.where(
        is_male, rng.normal(20, 6, n_rows), rng.normal(29, 6, n_rows)
    ).clip(8, 48)

    bmi = weight_kg / height_m**2

    return pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Weight (kg)": np.round(weight_kg, 1),
        "Height (m)": np.round(height_m, 2),
        "Max_BPM": rng.integers(160, 200, n_rows),
        "Avg_BPM": rng.integers(120, 170, n_rows),
        "Resting_BPM": rng.integers(50, 74, n_rows),
        "Session_Duration (hours)": np.round(session_duration, 2),
        "Calories_Burned": np.round(session_duration * rng.normal(520, 90, n_rows), 0),
        "Workout_Type": rng.choice(
            ["Cardio", "Strength", "HIIT", "Yoga"], size=n_rows
        ),
        "Fat_Percentage": np.round(fat_percentage, 1),
        "Water_Intake (liters)": np.round(rng.normal(2.6, 0.6, n_rows).clip(1.0, 4.5), 1),
        "Workout_Frequency (days/week)": workout_frequency,
        "Experience_Level": experience_level,
        "BMI": np.round(bmi, 2),
    })


#: A small hand-checked meal template set, used to synthesise a plausible log.
_MEAL_TEMPLATES: list[tuple[str, str, str, float, float, float, float, float]] = [
    # (name, category, slot, kcal, protein, carbs, fat, fiber)
    ("Oatmeal with Berries", "Grains", "breakfast", 320, 11, 55, 6, 8),
    ("Scrambled Eggs on Toast", "Protein", "breakfast", 400, 24, 30, 20, 3),
    ("Greek Yogurt Parfait", "Dairy", "breakfast", 280, 20, 34, 6, 4),
    ("Protein Pancakes", "Grains", "breakfast", 450, 30, 52, 12, 5),
    ("Banana Peanut Butter Smoothie", "Beverage", "breakfast", 380, 16, 48, 14, 6),
    ("Vegetable Omelette", "Protein", "breakfast", 330, 22, 9, 23, 3),
    ("Muesli with Milk", "Grains", "breakfast", 350, 13, 58, 8, 7),
    ("Avocado Toast", "Grains", "breakfast", 390, 12, 40, 21, 9),
    ("Grilled Chicken Salad", "Protein", "lunch", 480, 42, 22, 24, 7),
    ("Turkey Wrap", "Protein", "lunch", 520, 34, 55, 18, 5),
    ("Tuna Pasta Salad", "Protein", "lunch", 560, 36, 62, 16, 6),
    ("Quinoa Buddha Bowl", "Vegetarian", "lunch", 540, 21, 72, 18, 12),
    ("Chicken Burrito Bowl", "Protein", "lunch", 650, 44, 68, 20, 10),
    ("Lentil Soup with Bread", "Vegetarian", "lunch", 430, 22, 62, 9, 14),
    ("Beef and Rice Bowl", "Protein", "lunch", 700, 45, 74, 24, 4),
    ("Salmon Poke Bowl", "Protein", "lunch", 590, 38, 60, 22, 6),
    ("Grilled Salmon with Vegetables", "Protein", "dinner", 520, 42, 18, 30, 7),
    ("Chicken Breast with Sweet Potato", "Protein", "dinner", 560, 48, 52, 14, 8),
    ("Beef Stir Fry with Rice", "Protein", "dinner", 680, 44, 70, 22, 6),
    ("Baked Cod with Quinoa", "Protein", "dinner", 480, 40, 46, 12, 6),
    ("Vegetable Curry with Rice", "Vegetarian", "dinner", 520, 15, 82, 14, 11),
    ("Turkey Meatballs with Pasta", "Protein", "dinner", 620, 41, 68, 18, 7),
    ("Lamb Roast with Potatoes", "Protein", "dinner", 740, 46, 54, 36, 6),
    ("Tofu Stir Fry", "Vegetarian", "dinner", 450, 26, 42, 20, 8),
    ("Mixed Nuts", "Snack", "snack", 200, 7, 8, 17, 3),
    ("Protein Shake", "Beverage", "snack", 180, 28, 8, 3, 1),
    ("Apple with Peanut Butter", "Fruit", "snack", 240, 8, 28, 13, 6),
    ("Greek Yogurt", "Dairy", "snack", 130, 17, 9, 3, 0),
    ("Protein Bar", "Snack", "snack", 220, 20, 24, 7, 5),
    ("Cottage Cheese with Fruit", "Dairy", "snack", 180, 22, 14, 4, 2),
    ("Hummus with Carrots", "Snack", "snack", 190, 7, 22, 9, 7),
    ("Beef Jerky", "Snack", "snack", 150, 25, 6, 3, 0),
    # --- second tranche -------------------------------------------------
    # The catalogue needs roughly 15 items per slot, spanning a realistic
    # range of protein densities, before the five-day no-repeat rule and the
    # +/-5 % macro tolerance can both be satisfied. With only 8 items per slot
    # the variety constraint forces low-protein choices onto most days and the
    # plan cannot hold tolerance -- an artefact of the fixture, not of the
    # planner. These entries make the synthetic catalogue representative of the
    # real one (~120 items per slot).
    ("Cottage Cheese Pancakes", "Protein", "breakfast", 420, 34, 44, 11, 4),
    ("Smoked Salmon Bagel", "Protein", "breakfast", 470, 28, 52, 16, 3),
    ("Egg White Scramble", "Protein", "breakfast", 260, 30, 8, 11, 2),
    ("Steel Cut Oats with Whey", "Grains", "breakfast", 410, 32, 54, 8, 7),
    ("Breakfast Burrito", "Protein", "breakfast", 540, 29, 48, 26, 5),
    ("Chia Pudding", "Dairy", "breakfast", 300, 14, 32, 13, 10),
    ("Turkey Sausage and Eggs", "Protein", "breakfast", 380, 33, 4, 25, 1),
    ("Chicken Caesar Wrap", "Protein", "lunch", 560, 40, 46, 22, 4),
    ("Prawn Noodle Salad", "Protein", "lunch", 470, 34, 52, 12, 5),
    ("Falafel Bowl", "Vegetarian", "lunch", 580, 22, 68, 24, 13),
    ("Roast Beef Sandwich", "Protein", "lunch", 610, 42, 58, 20, 4),
    ("Chickpea Tuna Salad", "Protein", "lunch", 430, 33, 38, 14, 11),
    ("Teriyaki Chicken Rice", "Protein", "lunch", 680, 46, 78, 16, 3),
    ("Egg and Bacon Salad", "Protein", "lunch", 490, 31, 12, 35, 4),
    ("Grilled Chicken Thigh with Rice", "Protein", "dinner", 640, 45, 62, 20, 4),
    ("Baked White Fish and Potatoes", "Protein", "dinner", 510, 41, 48, 14, 6),
    ("Pork Loin with Vegetables", "Protein", "dinner", 560, 47, 24, 30, 7),
    ("Prawn Stir Fry", "Protein", "dinner", 480, 38, 44, 16, 5),
    ("Chilli Con Carne", "Protein", "dinner", 620, 43, 52, 24, 12),
    ("Chicken Fajitas", "Protein", "dinner", 590, 44, 50, 22, 8),
    ("Seitan Bolognese", "Vegetarian", "dinner", 520, 36, 58, 14, 9),
    ("Whey Protein Yogurt Bowl", "Dairy", "snack", 240, 30, 18, 4, 2),
    ("Boiled Eggs", "Protein", "snack", 155, 13, 1, 11, 0),
    ("Tuna Rice Cakes", "Protein", "snack", 210, 24, 18, 4, 1),
    ("Edamame", "Vegetarian", "snack", 190, 17, 15, 8, 8),
    ("Skyr with Honey", "Dairy", "snack", 170, 20, 18, 1, 0),
    ("Turkey Slices and Cheese", "Protein", "snack", 200, 26, 3, 10, 0),
    ("Peanut Butter Protein Balls", "Snack", "snack", 260, 15, 24, 13, 4),
]


def make_demo_food_dataset(n_rows: int = 4000, seed: int = DEMO_SEED) -> pd.DataFrame:
    """Synthetic frame matching the *raw* Daily Food & Nutrition log schema."""
    rng = np.random.default_rng(seed + 1)
    picks = rng.integers(0, len(_MEAL_TEMPLATES), size=n_rows)

    records = []
    for row_index, template_index in enumerate(picks):
        name, category, slot, kcal, protein, carbs, fat, fiber = _MEAL_TEMPLATES[template_index]
        # Portion jitter, mirroring real logging behaviour.
        scale = float(rng.normal(1.0, 0.12))
        scale = min(max(scale, 0.7), 1.4)
        records.append({
            "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(row_index % 365)),
            "User_ID": int(rng.integers(1, 250)),
            "Food_Item": name,
            "Category": category,
            "Calories (kcal)": round(kcal * scale, 1),
            "Protein (g)": round(protein * scale, 1),
            "Carbohydrates (g)": round(carbs * scale, 1),
            "Fat (g)": round(fat * scale, 1),
            "Fiber (g)": round(fiber * scale, 1),
            "Sugars (g)": round(carbs * scale * 0.25, 1),
            "Sodium (mg)": round(float(rng.normal(450, 150)), 1),
            "Cholesterol (mg)": round(float(rng.normal(60, 30)), 1),
            "Meal_Type": slot.capitalize(),
            "Water_Intake (ml)": int(rng.integers(150, 600)),
        })
    return pd.DataFrame(records)


__all__ = [
    "make_demo_gym_dataset",
    "make_demo_food_dataset",
    "MEAL_SLOTS",
    "DEMO_SEED",
]
