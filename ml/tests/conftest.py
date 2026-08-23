"""Shared pytest fixtures.

Every fixture is built from :mod:`nutrifit.demo`, so the test suite never
depends on a manual Kaggle download and can run in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nutrifit import foods, labels  # noqa: E402
from nutrifit.demo import make_demo_food_dataset, make_demo_gym_dataset  # noqa: E402
from nutrifit.recommender import MealRecommender  # noqa: E402
from nutrifit.schema import (  # noqa: E402
    FOOD_ALIASES,
    FOOD_REQUIRED,
    GYM_ALIASES,
    GYM_REQUIRED,
    normalise_columns,
)


@pytest.fixture(scope="session")
def raw_gym():
    return make_demo_gym_dataset(n_rows=400, seed=11)


@pytest.fixture(scope="session")
def clean_gym(raw_gym):
    df = normalise_columns(raw_gym, GYM_ALIASES, GYM_REQUIRED, "demo gym")
    df = df.copy()
    df["height_cm"] = df["height_m"] * 100.0
    df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
    return df


@pytest.fixture(scope="session")
def labelled_gym(clean_gym):
    return labels.generate_labels(clean_gym, seed=11)


@pytest.fixture(scope="session")
def catalogue():
    log = normalise_columns(
        make_demo_food_dataset(n_rows=2500, seed=11),
        FOOD_ALIASES, FOOD_REQUIRED, "demo food",
    ).copy()
    log["meal_type"] = log["meal_type"].str.lower()
    return foods.build_catalogue(log, verbose=False)


@pytest.fixture(scope="session")
def recommender(catalogue):
    return MealRecommender(catalogue, random_state=11)
