"""Tests for schema normalisation, label generation and the sklearn pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from nutrifit import labels, nutrition
from nutrifit.preprocessing import (
    FEATURE_COLUMNS,
    IQRClipper,
    build_pipeline,
    get_feature_names,
    select_features,
)
from nutrifit.schema import GYM_ALIASES, GYM_REQUIRED, SchemaError, normalise_columns


class TestSchemaNormalisation:
    def test_maps_units_in_parentheses(self):
        df = pd.DataFrame({
            "Age": [30], "Gender": ["Male"], "Weight (kg)": [80.0],
            "Height (m)": [1.8], "Session_Duration (hours)": [1.0],
            "Workout_Frequency (days/week)": [4], "Experience_Level": [2],
        })
        out = normalise_columns(df, GYM_ALIASES, GYM_REQUIRED, "test")
        assert {"age", "gender", "weight_kg", "height_m", "workout_frequency"} <= set(out.columns)

    def test_is_case_and_punctuation_insensitive(self):
        df = pd.DataFrame({
            "AGE": [30], "sex": ["M"], "weight kg": [80.0], "height_m": [1.8],
            "session duration": [1.0], "workout frequency": [4], "experience level": [2],
        })
        out = normalise_columns(df, GYM_ALIASES, GYM_REQUIRED, "test")
        assert "age" in out.columns and "gender" in out.columns

    def test_is_idempotent(self):
        df = pd.DataFrame({
            "Age": [30], "Gender": ["Male"], "Weight (kg)": [80.0], "Height (m)": [1.8],
            "Session_Duration (hours)": [1.0], "Workout_Frequency (days/week)": [4],
            "Experience_Level": [2],
        })
        once = normalise_columns(df, GYM_ALIASES, GYM_REQUIRED, "test")
        twice = normalise_columns(once, GYM_ALIASES, GYM_REQUIRED, "test")
        assert list(once.columns) == list(twice.columns)

    def test_unknown_columns_are_preserved(self):
        df = pd.DataFrame({
            "Age": [30], "Gender": ["Male"], "Weight (kg)": [80.0], "Height (m)": [1.8],
            "Session_Duration (hours)": [1.0], "Workout_Frequency (days/week)": [4],
            "Experience_Level": [2], "Some_New_Column": ["x"],
        })
        out = normalise_columns(df, GYM_ALIASES, GYM_REQUIRED, "test")
        assert "Some_New_Column" in out.columns

    def test_missing_required_column_raises_with_context(self):
        df = pd.DataFrame({"Age": [30], "Gender": ["Male"]})
        with pytest.raises(SchemaError) as error:
            normalise_columns(df, GYM_ALIASES, GYM_REQUIRED, "Gym Members dataset")
        message = str(error.value)
        assert "weight_kg" in message
        assert "Gym Members dataset" in message


class TestLabelGeneration:
    def test_all_targets_present(self, labelled_gym):
        for column in ("fitness_goal", "bmr", "tdee", "calorie_target",
                       "protein_target", "activity_level", "body_fat_pct"):
            assert column in labelled_gym.columns

    def test_goals_are_valid(self, labelled_gym):
        assert set(labelled_gym["fitness_goal"]).issubset(set(nutrition.GOALS))

    def test_all_three_goals_represented(self, labelled_gym):
        assert len(set(labelled_gym["fitness_goal"])) == 3

    def test_targets_physiologically_plausible(self, labelled_gym):
        assert labelled_gym["calorie_target"].between(1200, 6000).all()
        assert labelled_gym["protein_target"].between(40, 350).all()

    def test_reproducible_for_a_given_seed(self, clean_gym):
        first = labels.generate_labels(clean_gym, seed=99)
        second = labels.generate_labels(clean_gym, seed=99)
        pd.testing.assert_series_equal(first["calorie_target"], second["calorie_target"])
        pd.testing.assert_series_equal(first["fitness_goal"], second["fitness_goal"])

    def test_different_seeds_differ(self, clean_gym):
        first = labels.generate_labels(clean_gym, seed=1)
        second = labels.generate_labels(clean_gym, seed=2)
        assert not first["calorie_target"].equals(second["calorie_target"])

    def test_noise_can_be_disabled(self, clean_gym):
        clean = labels.generate_labels(clean_gym, seed=5, add_noise=False)
        # Without noise the label equals the formula output (floor aside).
        assert np.allclose(
            clean["calorie_target"], clean["calorie_target_clean"], atol=0.05
        )

    def test_noise_creates_a_finite_r2_ceiling(self, labelled_gym):
        # The ceiling must be high enough to be useful but strictly below 1.0,
        # otherwise the models could report a suspicious perfect fit.
        for target in ("calorie_target", "protein_target"):
            ceiling = labels.theoretical_r2_ceiling(labelled_gym, target)
            assert 0.90 < ceiling < 0.999

    def test_goal_assignment_correlates_with_bmi(self, labelled_gym):
        # Fat-loss users should have a higher mean BMI than muscle-gain users;
        # this is the whole point of probabilistic goal assignment.
        means = labelled_gym.groupby("fitness_goal")["bmi"].mean()
        assert means["fat_loss"] > means["muscle_gain"]

    def test_fat_loss_targets_below_maintenance(self, labelled_gym):
        # Compare like with like by normalising against each user's own TDEE.
        ratio = labelled_gym["calorie_target"] / labelled_gym["tdee"]
        by_goal = ratio.groupby(labelled_gym["fitness_goal"]).mean()
        assert by_goal["fat_loss"] < by_goal["maintenance"] < by_goal["muscle_gain"]


class TestIQRClipper:
    def test_learns_bounds_from_training_data_only(self):
        train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        clipper = IQRClipper(factor=1.5).fit(train)
        # A wild test-time value is compressed to the learned upper bound.
        result = clipper.transform(np.array([[1000.0]]))
        assert result[0, 0] == pytest.approx(clipper.upper_bounds_[0])

    def test_leaves_in_range_values_untouched(self):
        train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        clipper = IQRClipper().fit(train)
        assert clipper.transform(np.array([[3.0]]))[0, 0] == pytest.approx(3.0)


class TestPreprocessingPipeline:
    def test_select_features_returns_contract_order(self, labelled_gym):
        selected = select_features(labelled_gym)
        assert list(selected.columns) == FEATURE_COLUMNS

    def test_select_features_reports_missing(self, labelled_gym):
        with pytest.raises(KeyError, match="weight_kg"):
            select_features(labelled_gym.drop(columns=["weight_kg"]))

    def test_pipeline_fits_and_predicts(self, labelled_gym):
        pipeline = build_pipeline(LinearRegression())
        X = select_features(labelled_gym)
        y = labelled_gym["calorie_target"]
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape == (len(X),)
        assert np.isfinite(predictions).all()

    def test_pipeline_handles_missing_values(self, labelled_gym):
        pipeline = build_pipeline(LinearRegression())
        X = select_features(labelled_gym)
        y = labelled_gym["calorie_target"]
        pipeline.fit(X, y)

        broken = X.iloc[[0]].copy()
        broken.loc[:, "weight_kg"] = np.nan
        broken.loc[:, "gender"] = None
        assert np.isfinite(pipeline.predict(broken)).all()

    def test_pipeline_handles_unseen_category(self, labelled_gym):
        pipeline = build_pipeline(LinearRegression())
        X = select_features(labelled_gym)
        pipeline.fit(X, labelled_gym["calorie_target"])

        unseen = X.iloc[[0]].copy()
        unseen.loc[:, "activity_level"] = "hyperactive"
        unseen.loc[:, "gender"] = "Other"
        assert np.isfinite(pipeline.predict(unseen)).all()

    def test_feature_names_are_recoverable(self, labelled_gym):
        pipeline = build_pipeline(LinearRegression())
        pipeline.fit(select_features(labelled_gym), labelled_gym["calorie_target"])
        names = get_feature_names(pipeline)
        assert "weight_kg" in names
        assert any(name.startswith("fitness_goal") for name in names)

    def test_column_order_does_not_affect_prediction(self, labelled_gym):
        """Guards against train/serve skew from differently-ordered API payloads."""
        pipeline = build_pipeline(LinearRegression())
        X = select_features(labelled_gym)
        pipeline.fit(X, labelled_gym["calorie_target"])

        shuffled = X.iloc[[0]][list(reversed(FEATURE_COLUMNS))]
        assert pipeline.predict(shuffled)[0] == pytest.approx(
            pipeline.predict(X.iloc[[0]])[0]
        )
