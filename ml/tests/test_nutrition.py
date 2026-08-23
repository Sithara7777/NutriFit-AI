"""Unit tests for the nutrition domain formulas.

Expected values are computed by hand from the published equations, so these
tests fail if anyone silently changes a coefficient.
"""

from __future__ import annotations

import numpy as np
import pytest

from nutrifit import nutrition


class TestBMI:
    def test_known_value(self):
        # 80 kg at 1.80 m -> 80 / 3.24 = 24.69
        assert nutrition.calculate_bmi(80, 180) == pytest.approx(24.691, abs=1e-3)

    def test_vectorised(self):
        result = nutrition.calculate_bmi([80, 60], [180, 165])
        assert result == pytest.approx([24.691, 22.039], abs=1e-3)

    @pytest.mark.parametrize(
        "bmi,expected",
        [(17.0, "underweight"), (22.0, "normal"), (27.5, "overweight"), (33.0, "obese"),
         (18.5, "normal"), (25.0, "overweight"), (30.0, "obese")],
    )
    def test_who_boundaries(self, bmi, expected):
        assert nutrition.bmi_category(bmi) == expected

    def test_category_is_array_for_array_input(self):
        result = nutrition.bmi_category([17.0, 22.0])
        assert isinstance(result, np.ndarray)
        assert list(result) == ["underweight", "normal"]


class TestBMR:
    def test_mifflin_male(self):
        # 10*80 + 6.25*180 - 5*30 + 5 = 800 + 1125 - 150 + 5 = 1780
        assert nutrition.calculate_bmr(80, 180, 30, "Male") == pytest.approx(1780.0)

    def test_mifflin_female(self):
        # 10*65 + 6.25*165 - 5*30 - 161 = 650 + 1031.25 - 150 - 161 = 1370.25
        assert nutrition.calculate_bmr(65, 165, 30, "Female") == pytest.approx(1370.25)

    @pytest.mark.parametrize("token", ["Male", "male", "M", "m", "MALE"])
    def test_gender_tokens_accepted(self, token):
        assert nutrition.calculate_bmr(80, 180, 30, token) == pytest.approx(1780.0)

    def test_unknown_gender_uses_conservative_constant(self):
        assert nutrition.calculate_bmr(80, 180, 30, "unspecified") == pytest.approx(1614.0)

    def test_katch_mcardle(self):
        # LBM = 80 * 0.85 = 68 -> 370 + 21.6*68 = 1838.8
        assert nutrition.calculate_bmr_katch_mcardle(80, 15) == pytest.approx(1838.8)

    def test_lean_body_mass(self):
        assert nutrition.lean_body_mass(80, 15) == pytest.approx(68.0)

    def test_best_prefers_katch_when_fat_known(self):
        with_fat = nutrition.calculate_bmr_best(80, 180, 30, "Male", body_fat_pct=15)
        assert float(with_fat) == pytest.approx(1838.8)

    def test_best_falls_back_to_mifflin(self):
        without = nutrition.calculate_bmr_best(80, 180, 30, "Male", body_fat_pct=None)
        assert float(without) == pytest.approx(1780.0)

    def test_best_handles_nan_fat(self):
        result = nutrition.calculate_bmr_best(
            [80, 80], [180, 180], [30, 30], ["Male", "Male"],
            body_fat_pct=[15.0, np.nan],
        )
        assert result[0] == pytest.approx(1838.8)
        assert result[1] == pytest.approx(1780.0)


class TestBodyFatEstimate:
    def test_deurenberg_male(self):
        # 1.20*25 + 0.23*30 - 10.8*1 - 5.4 = 30 + 6.9 - 10.8 - 5.4 = 20.7
        assert nutrition.estimate_body_fat(25, 30, "Male") == pytest.approx(20.7)

    def test_deurenberg_female_is_higher(self):
        male = nutrition.estimate_body_fat(25, 30, "Male")
        female = nutrition.estimate_body_fat(25, 30, "Female")
        assert female > male

    def test_clipped_to_physiological_range(self):
        assert 3.0 <= float(nutrition.estimate_body_fat(10, 18, "Male")) <= 60.0
        assert 3.0 <= float(nutrition.estimate_body_fat(60, 80, "Female")) <= 60.0


class TestActivity:
    @pytest.mark.parametrize(
        "frequency,duration,expected",
        [(1, 1.0, "sedentary"),     # 1.0 weekly hours
         (2, 1.0, "light"),          # 2.0
         (4, 1.0, "moderate"),       # 4.0
         (5, 1.2, "active"),         # 6.0
         (6, 1.5, "very_active")],   # 9.0
    )
    def test_bands(self, frequency, duration, expected):
        assert nutrition.derive_activity_level(frequency, duration) == expected

    def test_multiplier_within_pal_range(self):
        for level in nutrition.ACTIVITY_LEVELS:
            for experience in (1, 2, 3):
                value = float(nutrition.activity_multiplier(level, experience))
                assert 1.20 <= value <= 1.90

    def test_experience_raises_multiplier(self):
        novice = float(nutrition.activity_multiplier("moderate", 1))
        expert = float(nutrition.activity_multiplier("moderate", 3))
        assert expert > novice

    def test_unknown_level_uses_safe_default(self):
        assert float(nutrition.activity_multiplier("nonsense", 2)) == pytest.approx(1.375)


class TestTargets:
    def test_tdee(self):
        assert nutrition.calculate_tdee(1800, 1.55) == pytest.approx(2790.0)

    def test_goal_factors_ordered(self):
        assert (
            nutrition.GOAL_CALORIE_FACTORS["fat_loss"]
            < nutrition.GOAL_CALORIE_FACTORS["maintenance"]
            < nutrition.GOAL_CALORIE_FACTORS["muscle_gain"]
        )

    def test_calorie_target_applies_deficit(self):
        assert float(nutrition.calorie_target(3000, "fat_loss")) == pytest.approx(2400.0)

    def test_safety_floor_blocks_unsafe_deficit(self):
        # A very low TDEE with a deficit must not fall below 1.1 x BMR.
        result = float(nutrition.calorie_target(1400, "fat_loss", bmr=1300))
        assert result >= 1300 * nutrition.MIN_CALORIES_AS_BMR_MULTIPLE

    def test_absolute_calorie_floor(self):
        assert float(nutrition.calorie_target(900, "fat_loss")) >= 1200.0

    def test_protein_target(self):
        assert float(nutrition.protein_target(80, "muscle_gain")) == pytest.approx(160.0)
        assert float(nutrition.protein_target(80, "fat_loss")) == pytest.approx(176.0)

    def test_protein_highest_during_deficit(self):
        # Lean-mass preservation requires the most protein in a deficit.
        coefficients = nutrition.GOAL_PROTEIN_COEFFICIENTS
        assert coefficients["fat_loss"] > coefficients["muscle_gain"] > coefficients["maintenance"]


class TestFormulaTargets:
    def test_end_to_end_measured_body_fat(self):
        result = nutrition.formula_targets(
            weight_kg=82, height_cm=178, age=28, gender="Male",
            goal="muscle_gain", workout_frequency=4, session_duration_h=1.25,
            experience_level=2, body_fat_pct=18.0,
        )
        assert result["bmr_equation"] == "Katch-McArdle"
        assert result["body_fat_source"] == "measured"
        # LBM = 82 * 0.82 = 67.24 -> BMR = 370 + 21.6*67.24 = 1822.38
        # 4 days x 1.25 h = 5.0 weekly hours -> "active" band -> PAL 1.725
        # TDEE = 3143.61 -> muscle gain x1.12 -> 3520.8
        assert result["activity_level"] == "active"
        assert result["bmr"] == pytest.approx(1822.4, abs=0.1)
        assert result["calorie_target"] == pytest.approx(3520.8, rel=1e-3)
        assert result["protein_target"] == pytest.approx(164.0)

    def test_end_to_end_estimated_body_fat(self):
        result = nutrition.formula_targets(
            weight_kg=82, height_cm=178, age=28, gender="Male",
            goal="maintenance", workout_frequency=3, session_duration_h=1.0,
        )
        assert result["bmr_equation"] == "Mifflin-St Jeor"
        assert result["body_fat_source"] == "estimated_deurenberg"
        assert 1500 < result["calorie_target"] < 4000

    def test_all_goals_produce_sane_output(self):
        for goal in nutrition.GOALS:
            result = nutrition.formula_targets(
                weight_kg=70, height_cm=170, age=35, gender="Female",
                goal=goal, workout_frequency=3, session_duration_h=1.0,
            )
            assert 1200 <= result["calorie_target"] <= 5000
            assert 50 <= result["protein_target"] <= 300
            assert result["bmi_category"] in {"underweight", "normal", "overweight", "obese"}


class TestMealSplit:
    def test_ratios_sum_to_one(self):
        assert sum(nutrition.MEAL_SLOT_RATIOS.values()) == pytest.approx(1.0)

    def test_split_conserves_total(self):
        split = nutrition.split_across_meals(2000)
        assert sum(split.values()) == pytest.approx(2000.0)
        assert set(split) == set(nutrition.MEAL_SLOTS)
