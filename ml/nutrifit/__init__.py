"""NutriFit-AI shared machine-learning package.

One package is imported by the Colab notebooks, the local training scripts and
the FastAPI inference service.  That is deliberate: the BMR/TDEE formulas, the
feature contract and the preprocessing pipeline exist in exactly one place, so
training and serving cannot drift apart.

Typical use::

    from nutrifit import data, labels, training, foods, recommender, planner

    df = data.load_gym_members()
    df = labels.generate_labels(df)
    bundle = training.train_target(df, "calorie_target")
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import (  # noqa: F401
    config,
    data,
    foods,
    labels,
    nutrition,
    planner,
    preprocessing,
    recommender,
    schema,
    training,
)

__all__ = [
    "config",
    "data",
    "foods",
    "labels",
    "nutrition",
    "planner",
    "preprocessing",
    "recommender",
    "schema",
    "training",
    "__version__",
]
