"""Tests for the curated regional (Sri Lankan) food set.

These guard two things: that the committed reference file stays valid, and
that merging it does not break the catalogue or the meal planner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nutrifit import config, data, foods
from nutrifit.nutrition import MEAL_SLOTS
from nutrifit.recommender import MealRecommender

REFERENCE_FILE = config.REFERENCE_DIR / config.LOCAL_FOODS_FILENAME
pytestmark = pytest.mark.skipif(
    not REFERENCE_FILE.exists(),
    reason="curated regional food file not present",
)


@pytest.fixture(scope="module")
def local():
    return data.load_local_foods(verbose=False)


class TestReferenceFile:
    def test_loads(self, local):
        assert local is not None and len(local) >= 40

    def test_covers_every_meal_slot(self, local):
        assert set(local["meal_type"]) == set(MEAL_SLOTS)

    def test_each_slot_has_enough_items(self, local):
        # The five-day variety rule needs a workable pool in every slot.
        counts = local["meal_type"].value_counts()
        for slot in MEAL_SLOTS:
            assert counts[slot] >= 8, f"{slot} has only {counts[slot]} regional items"

    def test_names_are_unique(self, local):
        assert local["food_item"].duplicated().sum() == 0

    def test_macros_are_physiologically_sane(self, local):
        assert local["calories"].between(50, 900).all()
        assert local["protein_g"].between(0, 60).all()
        assert (local["carbs_g"] >= 0).all()
        assert (local["fat_g"] >= 0).all()

    def test_macro_energy_matches_stated_calories(self, local):
        """4/4/9 reconstruction should land within 20% of the stated kcal.

        This catches transcription errors — a decimal point in the wrong place
        shows up immediately as an impossible energy balance.
        """
        implied = local["protein_g"] * 4 + local["carbs_g"] * 4 + local["fat_g"] * 9
        ratio = implied / local["calories"]
        bad = local.loc[~ratio.between(0.80, 1.20), "food_item"].tolist()
        assert not bad, f"macros do not reconstruct calories for: {bad}"

    def test_has_high_protein_options_for_fat_loss(self, local):
        """A fat-loss user needs protein-dense regional choices.

        Without these the planner falls back to Western foods for that goal,
        which is the exact bias this dataset exists to correct.
        """
        density = local["protein_g"] / local["calories"] * 100
        assert (density > 8).sum() >= 5, "too few protein-dense regional items"

    def test_has_vegetarian_options(self, local):
        vegetarian_terms = ("dhal", "vegetable", "gram", "chickpea", "green gram", "kola")
        names = local["food_item"].str.lower()
        assert names.str.contains("|".join(vegetarian_terms)).sum() >= 4

    def test_serving_description_present(self, local):
        raw = pd.read_csv(REFERENCE_FILE)
        assert "serving_description" in raw.columns
        assert raw["serving_description"].notna().all()


class TestCatalogueMerge:
    @pytest.fixture(scope="class")
    def merged(self, catalogue):
        local_foods = data.load_local_foods(verbose=False)
        return foods.build_catalogue(
            _reconstruct_log(catalogue), local_foods=local_foods, verbose=False
        )

    def test_regional_items_survive_the_merge(self, merged):
        assert merged["source"].str.startswith("Curated").sum() >= 40

    def test_regional_slots_are_not_overridden_by_inference(self, merged):
        """`Rice and Chicken Curry` is tagged lunch and must stay lunch.

        Keyword inference would move anything containing "curry" to dinner,
        emptying the regional lunch pool — the bug this asserts against.
        """
        regional = merged[merged["source"].str.startswith("Curated")]
        assert (regional["meal_type"] == "lunch").sum() >= 8
        row = regional[regional["name"] == "Rice and Chicken Curry"]
        assert len(row) == 1 and row.iloc[0]["meal_type"] == "lunch"

    def test_food_ids_still_unique(self, merged):
        assert merged["food_id"].is_unique

    def test_derived_features_computed_for_regional_rows(self, merged):
        regional = merged[merged["source"].str.startswith("Curated")]
        assert regional["protein_density"].notna().all()
        assert (regional["protein_density"] > 0).all()

    def test_energy_fractions_still_sum_to_one(self, merged):
        total = merged[["protein_frac", "carb_frac", "fat_frac"]].sum(axis=1)
        assert np.allclose(total, 1.0, atol=0.02)


class TestRegionalCoverageReport:
    def test_reports_zero_without_the_curated_set(self, catalogue):
        report = foods.regional_coverage_report(catalogue)
        lookup = dict(zip(report["term"], report["items"]))
        # The synthetic demo catalogue contains no Sri Lankan food at all.
        for term in ("hopper", "sambol", "kottu"):
            assert lookup[term] == 0

    def test_detects_regional_items_once_merged(self):
        local_foods = data.load_local_foods(verbose=False)
        merged = foods.build_catalogue(
            _minimal_log(), local_foods=local_foods, verbose=False
        )
        report = foods.regional_coverage_report(merged)
        lookup = dict(zip(report["term"], report["items"]))
        assert lookup["hopper"] >= 3
        assert lookup["sambol"] >= 1
        assert lookup["kottu"] >= 2
        assert lookup["curry"] >= 10

    def test_report_shape(self, catalogue):
        report = foods.regional_coverage_report(catalogue)
        assert list(report.columns) == ["term", "items"]
        assert len(report) == len(foods.REGIONAL_AUDIT_TERMS)


class TestPlannerWithRegionalFoods:
    @pytest.fixture(scope="class")
    def engine(self):
        local_foods = data.load_local_foods(verbose=False)
        merged = foods.build_catalogue(
            _minimal_log(), local_foods=local_foods, verbose=False
        )
        return MealRecommender(merged, random_state=42)

    @pytest.mark.parametrize("goal", ["fat_loss", "maintenance", "muscle_gain"])
    def test_can_compose_a_day_for_every_goal(self, engine, goal):
        day = engine.daily_plan(2400, 150, goal, rng=np.random.default_rng(1))
        assert set(day) == set(MEAL_SLOTS)
        assert all(len(plan.items) > 0 for plan in day.values())

    def test_regional_items_actually_get_recommended(self, engine):
        rng = np.random.default_rng(5)
        seen = set()
        for _ in range(20):
            day = engine.daily_plan(2400, 150, "maintenance", rng=rng)
            for plan in day.values():
                seen.update(item.name for item in plan.items)
        regional_names = set(data.load_local_foods(verbose=False)["food_item"])
        assert seen & regional_names, "no regional food surfaced in 20 simulated days"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _minimal_log() -> pd.DataFrame:
    """A tiny valid food source, so merge tests do not depend on the Kaggle file."""
    return pd.DataFrame({
        "food_item": ["Plain Oats", "Green Salad", "Baked Chicken", "Apple"],
        "category": ["Grains", "Vegetables", "Protein", "Fruit"],
        "meal_type": ["breakfast", "lunch", "dinner", "snack"],
        "calories": [300.0, 200.0, 400.0, 90.0],
        "protein_g": [10.0, 8.0, 40.0, 0.5],
        "carbs_g": [50.0, 20.0, 5.0, 24.0],
        "fat_g": [6.0, 9.0, 22.0, 0.3],
        "fiber_g": [7.0, 5.0, 1.0, 4.0],
        "sugar_g": [1.0, 4.0, 0.0, 18.0],
        "sodium_mg": [5.0, 200.0, 300.0, 1.0],
        "cholesterol_mg": [0.0, 0.0, 110.0, 0.0],
    })


def _reconstruct_log(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Turn an existing catalogue back into build_catalogue's input shape."""
    out = catalogue.rename(columns={"name": "food_item"}).copy()
    keep = [
        "food_item", "category", "meal_type", "calories", "protein_g",
        "carbs_g", "fat_g", "fiber_g", "sugar_g", "sodium_mg", "cholesterol_mg",
    ]
    return out[[column for column in keep if column in out.columns]]
