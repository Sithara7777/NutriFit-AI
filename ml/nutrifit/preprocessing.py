"""Feature definition and the scikit-learn preprocessing pipeline.

The whole of preprocessing is expressed as a single ``ColumnTransformer`` that
is fitted *inside* the same ``Pipeline`` as the estimator.  Consequences:

* the exported ``.pkl`` contains preprocessing **and** model, so the FastAPI
  service applies byte-identical transforms at inference time -- there is no
  train/serve skew and no chance of the API forgetting to scale a feature;
* cross-validation folds re-fit the imputer/scaler on each training fold only,
  so no target or test-fold information leaks through the scaler statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .nutrition import ACTIVITY_LEVELS, GOALS

# --------------------------------------------------------------------------
# Feature contract
# --------------------------------------------------------------------------
# Raw user attributes only.  Derived physiology (bmr, activity_multiplier,
# tdee) is deliberately excluded: handing the model the multiplier would give
# away the answer and collapse the Linear-Regression vs Random-Forest
# comparison into a tie.  See nutrifit/labels.py for the full rationale.
NUMERIC_FEATURES: list[str] = [
    "age",
    "height_cm",
    "weight_kg",
    "bmi",
    "body_fat_pct",
    "workout_frequency",
    "session_duration_h",
    "experience_level",
]

NOMINAL_FEATURES: list[str] = ["gender", "fitness_goal"]

ORDINAL_FEATURES: list[str] = ["activity_level"]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES

TARGET_COLUMNS: list[str] = ["calorie_target", "protein_target"]

#: Explicit category orders keep the encoding stable between training runs and
#: between train and serve, regardless of which categories a given split saw.
NOMINAL_CATEGORIES: list[list[str]] = [["Female", "Male"], list(GOALS)]
ORDINAL_CATEGORIES: list[list[str]] = [list(ACTIVITY_LEVELS)]


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns to ``[Q1 - k*IQR, Q3 + k*IQR]``.

    Bounds are learned on the training fold only and re-applied unchanged at
    transform time, which protects the deployed service against absurd input
    (a mistyped 700 kg bodyweight) without letting test-set statistics leak
    into training.

    Clipping rather than dropping is deliberate: with 973 records, discarding
    rows costs more information than compressing the tails does.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(values, 25, axis=0)
        q3 = np.nanpercentile(values, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(clip_outliers: bool = True) -> ColumnTransformer:
    """Assemble the ColumnTransformer described in Implementation Plan 3.2."""
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    if clip_outliers:
        numeric_steps.append(("clip", IQRClipper(factor=1.5)))
    numeric_steps.append(("scale", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    nominal_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(
            categories=NOMINAL_CATEGORIES,
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    ordinal_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
        ("scale", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("nominal", nominal_pipeline, NOMINAL_FEATURES),
            ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, clip_outliers: bool = True) -> Pipeline:
    """Wrap ``estimator`` behind the shared preprocessor."""
    return Pipeline([
        ("preprocess", build_preprocessor(clip_outliers=clip_outliers)),
        ("model", estimator),
    ])


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Post-transformation feature names, for coefficient/importance plots."""
    preprocessor = pipeline.named_steps["preprocess"]
    return [str(name) for name in preprocessor.get_feature_names_out()]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the model input columns, in the contracted order.

    Raising here -- rather than letting the ColumnTransformer fail with an
    opaque message -- is what turns a malformed API request into a clear
    422 response instead of a 500.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature column(s): {missing}")
    return df.loc[:, FEATURE_COLUMNS].copy()


__all__ = [
    "NUMERIC_FEATURES",
    "NOMINAL_FEATURES",
    "ORDINAL_FEATURES",
    "FEATURE_COLUMNS",
    "TARGET_COLUMNS",
    "IQRClipper",
    "build_preprocessor",
    "build_pipeline",
    "get_feature_names",
    "select_features",
]
