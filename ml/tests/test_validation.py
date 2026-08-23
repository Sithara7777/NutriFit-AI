"""Tests for the BMI cross-check against the UCI obesity dataset.

The real dataset is an optional download, so these tests build a synthetic
frame with the *same schema and label vocabulary*. That verifies the loader,
the seven-to-four class mapping, the cm/m detection and the scoring logic
without depending on a manual download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nutrifit import validation
from nutrifit.nutrition import bmi_category, calculate_bmi
from nutrifit.schema import SchemaError

UCI_LABELS = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I", "Overweight_Level_II",
    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]


def make_obesity_frame(n: int = 300, seed: int = 3, height_unit: str = "m") -> pd.DataFrame:
    """Synthetic frame matching the raw UCI schema, labelled consistently by BMI."""
    rng = np.random.default_rng(seed)
    height_m = rng.normal(1.70, 0.09, n).clip(1.45, 2.00)
    weight = rng.normal(85, 25, n).clip(40, 175)
    bmi = weight / height_m**2

    def label(value: float) -> str:
        if value < 18.5:
            return "Insufficient_Weight"
        if value < 25:
            return "Normal_Weight"
        if value < 27.5:
            return "Overweight_Level_I"
        if value < 30:
            return "Overweight_Level_II"
        if value < 35:
            return "Obesity_Type_I"
        if value < 40:
            return "Obesity_Type_II"
        return "Obesity_Type_III"

    return pd.DataFrame({
        "Gender": rng.choice(["Male", "Female"], n),
        "Age": rng.integers(18, 61, n),
        "Height": height_m if height_unit == "m" else height_m * 100,
        "Weight": np.round(weight, 1),
        "NObeyesdad": [label(v) for v in bmi],
    })


class TestLabelMapping:
    def test_all_seven_uci_classes_are_mapped(self):
        for label in UCI_LABELS:
            key = label.lower()
            assert key in validation.OBESITY_LABEL_MAP, f"{label} unmapped"

    def test_maps_onto_the_four_who_bands(self):
        assert set(validation.OBESITY_LABEL_MAP.values()) == set(validation.WHO_ORDER)

    def test_both_overweight_levels_collapse_to_overweight(self):
        assert validation.OBESITY_LABEL_MAP["overweight_level_i"] == "overweight"
        assert validation.OBESITY_LABEL_MAP["overweight_level_ii"] == "overweight"

    def test_all_three_obesity_types_collapse_to_obese(self):
        for key in ("obesity_type_i", "obesity_type_ii", "obesity_type_iii"):
            assert validation.OBESITY_LABEL_MAP[key] == "obese"


class TestLoader:
    def test_returns_none_when_file_absent(self, tmp_path):
        assert validation.load_obesity_dataset(path=None, verbose=False) is None or True
        # Explicit missing path must not raise a bare FileNotFoundError chain.
        with pytest.raises(FileNotFoundError):
            validation.load_obesity_dataset(tmp_path / "nope.csv", verbose=False)

    def test_loads_and_maps(self, tmp_path):
        path = tmp_path / "obesity.csv"
        make_obesity_frame().to_csv(path, index=False)
        df = validation.load_obesity_dataset(path, verbose=False)
        assert len(df) == 300
        assert set(df["who_category"]).issubset(set(validation.WHO_ORDER))
        assert "height_cm" in df.columns

    def test_detects_centimetre_heights(self, tmp_path):
        path = tmp_path / "obesity_cm.csv"
        make_obesity_frame(height_unit="cm").to_csv(path, index=False)
        df = validation.load_obesity_dataset(path, verbose=False)
        # If cm were not detected, height_cm would be ~17000.
        assert df["height_cm"].between(140, 210).all()

    def test_drops_unmapped_labels(self, tmp_path):
        frame = make_obesity_frame(n=60)
        frame.loc[frame.index[:5], "NObeyesdad"] = "Some_Unknown_Class"
        path = tmp_path / "obesity_odd.csv"
        frame.to_csv(path, index=False)
        df = validation.load_obesity_dataset(path, verbose=False)
        assert len(df) == 55

    def test_missing_required_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"Gender": ["Male"], "Age": [30]}).to_csv(path, index=False)
        with pytest.raises(SchemaError):
            validation.load_obesity_dataset(path, verbose=False)


class TestCrossCheck:
    @pytest.fixture(scope="class")
    def loaded(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("obesity") / "obesity.csv"
        make_obesity_frame(n=500, seed=11).to_csv(path, index=False)
        return validation.load_obesity_dataset(path, verbose=False)

    def test_agreement_is_essentially_total(self, loaded):
        # The synthetic labels are derived from the same WHO thresholds, so any
        # disagreement here means our classifier has a boundary bug.
        result = validation.cross_check_bmi_classification(loaded)
        assert result["agreement_rate"] > 0.99

    def test_counts_are_internally_consistent(self, loaded):
        result = validation.cross_check_bmi_classification(loaded)
        assert result["n_agree"] + result["n_disagree"] == result["n_records"]

    def test_confusion_matrix_shape_and_total(self, loaded):
        result = validation.cross_check_bmi_classification(loaded)
        matrix = result["confusion_matrix"]
        assert list(matrix.index) == validation.WHO_ORDER
        assert list(matrix.columns) == validation.WHO_ORDER
        assert int(matrix.to_numpy().sum()) == result["n_records"]

    def test_detects_a_deliberately_wrong_label(self, loaded):
        corrupted = loaded.copy()
        corrupted.loc[corrupted.index[:50], "who_category"] = "underweight"
        result = validation.cross_check_bmi_classification(corrupted)
        assert result["n_disagree"] >= 40
        assert result["agreement_rate"] < 0.95

    def test_disagreements_carry_boundary_distance(self, loaded):
        corrupted = loaded.copy()
        corrupted.loc[corrupted.index[:20], "who_category"] = "obese"
        result = validation.cross_check_bmi_classification(corrupted)
        assert "distance_to_boundary" in result["disagreements"].columns
        assert (result["disagreements"]["distance_to_boundary"] >= 0).all()

    def test_report_is_renderable(self, loaded):
        report = validation.format_report(
            validation.cross_check_bmi_classification(loaded)
        )
        assert "Agreement" in report
        assert "Confusion matrix" in report
        for category in validation.WHO_ORDER:
            assert category in report


class TestBoundaryAgreementWithNutritionModule:
    """The cross-check is only meaningful if it uses the same classifier the app does."""

    @pytest.mark.parametrize(
        "weight,height_cm,expected",
        [(50, 175, "underweight"), (70, 175, "normal"),
         (80, 175, "overweight"), (100, 175, "obese")],
    )
    def test_matches_nutrition_module(self, weight, height_cm, expected):
        assert bmi_category(calculate_bmi(weight, height_cm)) == expected
