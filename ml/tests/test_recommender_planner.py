"""Tests for the food catalogue, recommendation engine and meal planner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nutrifit import foods
from nutrifit.nutrition import MEAL_SLOTS
from nutrifit.planner import (
    MACRO_TOLERANCE,
    VARIETY_WINDOW_DAYS,
    generate_meal_plan,
    needs_regeneration,
)
from nutrifit.recommender import (
    MAX_SERVINGS,
    MIN_SERVINGS,
    MealRecommender,
    slot_macro_target,
)

GOALS = ("fat_loss", "maintenance", "muscle_gain")


class TestCatalogue:
    def test_aggregates_log_into_unique_pairs(self, catalogue):
        assert catalogue.duplicated(subset=["name", "meal_type"]).sum() == 0

    def test_covers_every_slot(self, catalogue):
        assert set(catalogue["meal_type"]) == set(MEAL_SLOTS)

    def test_derived_features_present(self, catalogue):
        for column in ("protein_density", "fiber_density",
                       "protein_frac", "carb_frac", "fat_frac"):
            assert column in catalogue.columns

    def test_energy_fractions_sum_to_one(self, catalogue):
        total = catalogue[["protein_frac", "carb_frac", "fat_frac"]].sum(axis=1)
        assert np.allclose(total, 1.0, atol=0.02)

    def test_food_ids_unique(self, catalogue):
        assert catalogue["food_id"].is_unique

    def test_no_impossible_macros(self, catalogue):
        assert (catalogue["calories"] > 0).all()
        assert (catalogue["protein_g"] >= 0).all()

    @pytest.mark.parametrize(
        "name,expected",
        [("Oatmeal with Berries", "breakfast"), ("Grilled Chicken Salad", "lunch"),
         ("Grilled Salmon Fillet", "dinner"), ("Mixed Nuts", "snack")],
    )
    def test_meal_type_inference(self, name, expected):
        assert foods.infer_meal_type(name) == expected

    def test_unknown_food_defaults_to_snack(self):
        assert foods.infer_meal_type("Zzzz Unknown Item") == "snack"

    def test_sql_seed_escapes_quotes(self, catalogue):
        rows = catalogue.head(3).copy()
        rows.loc[rows.index[0], "name"] = "Shepherd's Pie"
        sql = foods.to_sql_seed(rows)
        assert "Shepherd''s Pie" in sql
        assert sql.strip().endswith("commit;")


class TestSlotTarget:
    def test_standalone_queries_reconstruct_the_daily_total(self):
        """A fresh query per slot must use plain shares, not redistribution."""
        total = sum(
            slot_macro_target(2400, 150, "maintenance", slot)["calories"]
            for slot in MEAL_SLOTS
        )
        assert total == pytest.approx(2400, rel=1e-6)

    def test_sequential_walk_reconstructs_the_daily_total(self):
        """Walking the day in order with on-target consumption also sums to 2400."""
        consumed = 0.0
        totals = []
        for position, slot in enumerate(MEAL_SLOTS):
            target = slot_macro_target(
                2400, 150, "maintenance", slot,
                consumed_calories=consumed, redistribute=position > 0,
            )
            totals.append(target["calories"])
            consumed += target["calories"]
        assert sum(totals) == pytest.approx(2400, rel=1e-6)

    def test_redistribution_absorbs_a_shortfall(self):
        """If breakfast under-delivers, lunch must pick up the slack."""
        on_target = slot_macro_target(
            2400, 150, "maintenance", "lunch",
            consumed_calories=600, redistribute=True,
        )
        short = slot_macro_target(
            2400, 150, "maintenance", "lunch",
            consumed_calories=400, redistribute=True,
        )
        assert short["calories"] > on_target["calories"]
        assert on_target["calories"] == pytest.approx(840.0, rel=1e-6)

    def test_consumption_shrinks_later_slots(self):
        fresh = slot_macro_target(2400, 150, "maintenance", "dinner")
        after = slot_macro_target(
            2400, 150, "maintenance", "dinner",
            consumed_calories=1800, consumed_protein=110, redistribute=True,
        )
        assert after["calories"] < fresh["calories"]

    def test_macros_reconstruct_the_calorie_budget(self):
        target = slot_macro_target(2400, 150, "maintenance", "lunch")
        energy = target["protein_g"] * 4 + target["carbs_g"] * 4 + target["fat_g"] * 9
        assert energy == pytest.approx(target["calories"], rel=0.01)

    def test_muscle_gain_allocates_more_carbohydrate(self):
        gain = slot_macro_target(2400, 150, "muscle_gain", "lunch")
        loss = slot_macro_target(2400, 150, "fat_loss", "lunch")
        assert gain["carbs_g"] > loss["carbs_g"]

    def test_rejects_unknown_slot(self):
        with pytest.raises(ValueError, match="Unknown meal slot"):
            slot_macro_target(2400, 150, "maintenance", "elevenses")


class TestRecommender:
    def test_indexes_every_slot(self, recommender):
        assert set(recommender.available_slots()) == set(MEAL_SLOTS)

    def test_rejects_catalogue_missing_columns(self, catalogue):
        with pytest.raises(ValueError, match="missing column"):
            MealRecommender(catalogue.drop(columns=["protein_g"]))

    @pytest.mark.parametrize("goal", GOALS)
    @pytest.mark.parametrize("slot", MEAL_SLOTS)
    def test_returns_suggestions_for_every_combination(self, recommender, slot, goal):
        results = recommender.recommend(slot, 2400, 150, goal, top_n=3)
        assert len(results) > 0
        assert all(r.meal_type == slot for r in results)

    def test_results_are_score_ordered(self, recommender):
        results = recommender.recommend("lunch", 2400, 150, "maintenance", top_n=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_exclusions_are_respected(self, recommender):
        first = recommender.recommend("lunch", 2400, 150, "maintenance", top_n=1)[0]
        second = recommender.recommend(
            "lunch", 2400, 150, "maintenance", top_n=1, exclude={first.food_id}
        )
        assert second[0].food_id != first.food_id

    def test_fat_loss_prefers_denser_protein(self, recommender):
        loss = recommender.recommend("lunch", 2000, 160, "fat_loss", top_n=5)
        gain = recommender.recommend("lunch", 2000, 160, "muscle_gain", top_n=5)
        loss_density = np.mean([r.protein_g / r.calories for r in loss])
        gain_density = np.mean([r.protein_g / r.calories for r in gain])
        assert loss_density >= gain_density

    def test_rejects_unknown_slot(self, recommender):
        with pytest.raises(ValueError, match="No items indexed"):
            recommender.recommend("brunch", 2400, 150, "maintenance")


class TestSlotComposition:
    @pytest.mark.parametrize("slot", MEAL_SLOTS)
    def test_composes_within_item_limit(self, recommender, slot):
        plan = recommender.compose_slot(slot, 2400, 150, "maintenance")
        assert 1 <= len(plan.items) <= 3

    def test_servings_are_quarter_steps_within_bounds(self, recommender):
        plan = recommender.compose_slot("lunch", 2400, 150, "maintenance")
        for item in plan.items:
            assert MIN_SERVINGS <= item.servings <= MAX_SERVINGS
            assert (item.servings * 4) == pytest.approx(round(item.servings * 4))

    def test_scaled_macros_track_servings(self, recommender):
        plan = recommender.compose_slot("dinner", 2400, 150, "muscle_gain")
        item = plan.items[0]
        assert item.calories == pytest.approx(
            item.per_serving["calories"] * item.servings
        )

    @pytest.mark.parametrize("slot", MEAL_SLOTS)
    def test_fills_most_of_the_slot_budget(self, recommender, slot):
        plan = recommender.compose_slot(slot, 2400, 150, "maintenance")
        # Composition + portion scaling should land near the slot target.
        assert plan.calories == pytest.approx(plan.target["calories"], rel=0.30)

    def test_to_dict_exposes_per_serving_breakdown(self, recommender):
        payload = recommender.compose_slot("lunch", 2400, 150, "maintenance").to_dict()
        assert "totals" in payload and "items" in payload
        assert "per_serving" in payload["items"][0]


class TestDailyPlan:
    @pytest.mark.parametrize("goal", GOALS)
    def test_hits_the_daily_calorie_target(self, recommender, goal):
        rng = np.random.default_rng(3)
        plan = recommender.daily_plan(2400, 150, goal, rng=rng)
        total = sum(p.calories for p in plan.values())
        assert total == pytest.approx(2400, rel=0.12)

    def test_covers_every_slot(self, recommender):
        plan = recommender.daily_plan(2400, 150, "maintenance")
        assert set(plan) == set(MEAL_SLOTS)
        assert all(len(p.items) > 0 for p in plan.values())

    def test_is_reproducible_for_a_given_seed(self, recommender):
        first = recommender.daily_plan(2400, 150, "maintenance",
                                       rng=np.random.default_rng(7))
        second = recommender.daily_plan(2400, 150, "maintenance",
                                        rng=np.random.default_rng(7))
        assert [i.food_id for p in first.values() for i in p.items] == \
               [i.food_id for p in second.values() for i in p.items]

    def test_varies_across_seeds(self, recommender):
        first = recommender.daily_plan(2400, 150, "maintenance",
                                       rng=np.random.default_rng(1))
        second = recommender.daily_plan(2400, 150, "maintenance",
                                        rng=np.random.default_rng(2))
        assert [i.food_id for p in first.values() for i in p.items] != \
               [i.food_id for p in second.values() for i in p.items]


@pytest.fixture(scope="module")
def plan(recommender):
    """One eight-week plan shared across the planner tests (generation is slow)."""
    return generate_meal_plan(
        recommender, calorie_target=2400, protein_target=150,
        goal="muscle_gain", weeks=8, seed=5,
    )


class TestMealPlanner:
    def test_generates_eight_full_weeks(self, plan):
        assert plan.weeks == 8
        assert len(plan.days) == 56

    def test_every_day_has_all_slots(self, plan):
        for day in plan.days:
            assert set(day.slots) == set(MEAL_SLOTS)

    def test_weekly_calories_within_tolerance(self, plan):
        weekly = plan.weekly_summary()
        assert len(weekly) == 8
        # Calories are always achievable: the portion multiplier scales them
        # directly, so this holds even on the small synthetic catalogue.
        assert weekly["calorie_error_pct"].abs().max() <= MACRO_TOLERANCE * 100

    def test_weekly_protein_stays_in_a_sane_range(self, plan):
        """Sanity bound only -- NOT the product guarantee.

        The demo catalogue holds 15 meal-sized items per slot, so a day is
        assembled from roughly four choices, and the five-day no-repeat rule
        forces lower-protein items onto most days. The real catalogue holds
        ~120 smaller component foods per slot, giving the composer far more
        freedom to land an exact protein figure.

        The strict +/-5 % guarantee is therefore asserted against the real
        catalogue in :class:`TestRealCatalogue`. Here we only catch gross
        regressions -- a planner that ignored protein entirely would miss by
        far more than this.
        """
        weekly = plan.weekly_summary()
        assert weekly["protein_error_pct"].abs().max() <= 15.0

    def test_feature_weights_are_configurable(self, catalogue):
        """The weighting must be an explicit, overridable parameter.

        Note the weighting is tuned for a real-sized catalogue and is *not*
        guaranteed to help on the 15-item-per-slot synthetic one, so the
        effectiveness assertion lives in :class:`TestRealCatalogue`.
        """
        from nutrifit.recommender import FEATURE_VECTOR, FEATURE_WEIGHTS

        default = MealRecommender(catalogue, random_state=11)
        assert default.feature_weights == FEATURE_WEIGHTS

        flat = MealRecommender(
            catalogue,
            random_state=11,
            feature_weights={feature: 1.0 for feature in FEATURE_VECTOR},
        )
        assert set(flat.weights) == {1.0}

    def test_within_tolerance_helper_matches_the_summary(self, plan):
        weekly = plan.weekly_summary()
        expected = (
            weekly["calorie_error_pct"].abs().max() <= MACRO_TOLERANCE * 100
            and weekly["protein_error_pct"].abs().max() <= MACRO_TOLERANCE * 100
        )
        assert plan.within_tolerance() is bool(expected)

    def test_dataframe_export_shape(self, plan):
        df = plan.to_dataframe()
        assert len(df) > 200
        assert {"week", "day_of_week", "meal_slot", "food_id", "servings"} <= set(df.columns)
        assert df["week"].between(1, 8).all()

    def test_serialises_to_json_safe_dict(self, plan):
        import json

        payload = plan.to_dict()
        json.dumps(payload)  # must not raise
        assert len(payload["days"]) == 56
        assert len(payload["weekly_summary"]) == 8

    def test_is_reproducible(self, recommender):
        first = generate_meal_plan(recommender, 2400, 150, "fat_loss", weeks=2, seed=21)
        second = generate_meal_plan(recommender, 2400, 150, "fat_loss", weeks=2, seed=21)
        assert first.to_dataframe()["food_id"].tolist() == \
               second.to_dataframe()["food_id"].tolist()

    def test_variety_report_is_consistent(self, plan):
        report = plan.variety_report()
        assert report["unique_foods"] <= report["total_items"]
        assert 0 < report["variety_ratio"] <= 1.0

    def test_variety_rule_is_nearly_always_respected(self, plan):
        """Near-zero on the synthetic catalogue; strictly zero on the real one.

        ``MealRecommender._rank`` deliberately drops the exclusion filter when
        it would leave fewer than two candidates -- a plan with a repeated meal
        is better than a plan with a missing meal. With only 15 items per slot
        and a five-day window excluding most of them, that fallback can fire.

        The real 489-item catalogue never triggers it, which
        :meth:`TestRealCatalogue.test_five_day_variety_rule_is_never_violated`
        asserts strictly.
        """
        assert plan.count_variety_violations() <= 2


class TestThinCatalogueDegradation:
    """A catalogue can be too small to satisfy the macro tolerance.

    The planner cannot conjure protein that is not in the catalogue. What it
    must do is fail *honestly*: still produce a complete plan, still respect
    the variety rule, and report `within_tolerance() is False` rather than
    quietly claiming success. Those guarantees are what these tests pin.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def thin_recommender(cls, catalogue):
        # Keep only the lowest-protein items so the target is unreachable.
        thin = (
            catalogue.sort_values("protein_density")
            .groupby("meal_type", group_keys=False)
            .head(6)
            .reset_index(drop=True)
        )
        return MealRecommender(thin, random_state=3)

    def test_still_produces_a_complete_plan(self, thin_recommender):
        plan = generate_meal_plan(thin_recommender, 2400, 190, "muscle_gain", weeks=2, seed=3)
        assert len(plan.days) == 14
        for day in plan.days:
            assert set(day.slots) == set(MEAL_SLOTS)

    def test_reports_tolerance_failure_honestly(self, thin_recommender):
        plan = generate_meal_plan(thin_recommender, 2400, 190, "muscle_gain", weeks=2, seed=3)
        weekly = plan.weekly_summary()
        unreachable = weekly["protein_error_pct"].abs().max() > MACRO_TOLERANCE * 100
        # If the target genuinely could not be met, the helper must say so.
        assert plan.within_tolerance() is not unreachable

    def test_attempts_repairs_before_giving_up(self, thin_recommender):
        plan = generate_meal_plan(thin_recommender, 2400, 190, "muscle_gain", weeks=2, seed=3)
        assert plan.repairs_applied > 0


class TestRealCatalogue:
    """The +/-5 % product guarantee, asserted against the real food catalogue.

    Skipped automatically until ``ml/scripts/prepare_data.py`` has been run on
    the downloaded datasets, so the suite still passes on a clean checkout.
    """

    @staticmethod
    def _load():
        from nutrifit import config

        if not config.FOODS_PROCESSED.exists():
            pytest.skip("Real catalogue not built yet - run ml/scripts/prepare_data.py")

        catalogue = foods.add_derived_features(pd.read_csv(config.FOODS_PROCESSED))
        if len(catalogue) < 100:
            pytest.skip(f"Catalogue too small to assert the guarantee ({len(catalogue)} items)")
        return MealRecommender(catalogue, random_state=42)

    @pytest.mark.parametrize(
        "goal,calories,protein",
        [("fat_loss", 1900, 145), ("maintenance", 2400, 130), ("muscle_gain", 3000, 175)],
    )
    def test_eight_week_plan_holds_tolerance(self, goal, calories, protein):
        engine = self._load()
        plan = generate_meal_plan(engine, calories, protein, goal, weeks=8, seed=42)
        weekly = plan.weekly_summary()

        assert weekly["calorie_error_pct"].abs().max() <= MACRO_TOLERANCE * 100, (
            f"{goal}: worst weekly calorie error "
            f"{weekly['calorie_error_pct'].abs().max():.2f}%"
        )
        assert weekly["protein_error_pct"].abs().max() <= MACRO_TOLERANCE * 100, (
            f"{goal}: worst weekly protein error "
            f"{weekly['protein_error_pct'].abs().max():.2f}%"
        )
        assert plan.within_tolerance() is True

    def test_five_day_variety_rule_is_never_violated(self):
        engine = self._load()
        plan = generate_meal_plan(engine, 2400, 150, "muscle_gain", weeks=8, seed=42)
        assert plan.count_variety_violations() == 0

    def test_plan_is_complete(self):
        engine = self._load()
        plan = generate_meal_plan(engine, 2400, 150, "muscle_gain", weeks=8, seed=42)
        assert len(plan.days) == 56
        for day in plan.days:
            assert set(day.slots) == set(MEAL_SLOTS)
            assert all(len(slot_plan.items) > 0 for slot_plan in day.slots.values())

    @pytest.mark.parametrize(
        "goal,calories,protein",
        [("fat_loss", 1900, 145), ("muscle_gain", 3000, 175)],
    )
    def test_protein_weighting_beats_the_unweighted_baseline(self, goal, calories, protein):
        """Ablation guarding the weighted-distance design decision.

        Both scenarios *fail* the tolerance with an unweighted metric and pass
        with the shipped weights, so this pins the decision to a measured
        outcome rather than an arbitrary constant.
        """
        from nutrifit import config
        from nutrifit.recommender import FEATURE_VECTOR

        engine = self._load()
        catalogue = engine.catalogue

        weighted = generate_meal_plan(engine, calories, protein, goal, weeks=8, seed=42)
        unweighted = generate_meal_plan(
            MealRecommender(
                catalogue,
                random_state=42,
                feature_weights={feature: 1.0 for feature in FEATURE_VECTOR},
            ),
            calories, protein, goal, weeks=8, seed=42,
        )

        weighted_error = weighted.weekly_summary()["protein_error_pct"].abs().max()
        unweighted_error = unweighted.weekly_summary()["protein_error_pct"].abs().max()
        assert weighted_error < unweighted_error, (
            f"{goal}: weighted {weighted_error:.2f}% vs unweighted {unweighted_error:.2f}%"
        )
        assert weighted.repairs_applied <= unweighted.repairs_applied


class TestRegenerationTrigger:
    def test_no_prompt_below_threshold(self):
        needed, detail = needs_regeneration(2400, 2450, 150, 152)
        assert needed is False
        assert detail["calorie_drift_pct"] < 7

    def test_prompts_on_large_calorie_drift(self):
        needed, detail = needs_regeneration(2400, 2700, 150, 150)
        assert needed is True
        assert detail["calorie_drift_pct"] == pytest.approx(12.5)

    def test_prompts_on_large_protein_drift(self):
        needed, _ = needs_regeneration(2400, 2400, 150, 170)
        assert needed is True

    def test_threshold_is_configurable(self):
        assert needs_regeneration(2400, 2500, 150, 150, threshold=0.01)[0] is True
        assert needs_regeneration(2400, 2500, 150, 150, threshold=0.50)[0] is False
