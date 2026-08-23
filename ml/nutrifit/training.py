"""Model training, hyper-parameter search and evaluation.

Implements Implementation Plan sections 3.3-3.4:

* Linear Regression (interpretable baseline) and Random Forest Regression
  (main model), exactly as named in the proposal -- no substitutions.
* ``RandomizedSearchCV`` over the Random Forest's ``n_estimators``,
  ``max_depth``, ``min_samples_leaf`` and ``max_features``, scored on
  ``neg_mean_absolute_error`` so tuning optimises the same quantity the
  evaluation reports.
* MAE / MSE / RMSE / R2 on a held-out stratified test split **and** via 5-fold
  cross-validation reported as mean +/- standard deviation.
* A non-ML formula baseline, so the report can state what the ML actually buys
  over the textbook equation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)

from .config import CV_FOLDS, RANDOM_SEED, TEST_SIZE
from .preprocessing import FEATURE_COLUMNS, build_pipeline, get_feature_names, select_features

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """The four metrics named in the proposal, plus MAPE for interpretability."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(mean_squared_error(y_true, y_pred))
    non_zero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)
        if non_zero.any()
        else float("nan")
    )
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": mape,
    }


@dataclass
class ModelResult:
    """Everything the report needs about one (model, target) combination."""

    target: str
    model_name: str
    test_metrics: dict[str, float]
    cv_metrics: dict[str, dict[str, float]]
    train_metrics: dict[str, float]
    best_params: dict[str, Any] = field(default_factory=dict)
    fit_seconds: float = 0.0
    n_train: int = 0
    n_test: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_row(self) -> dict[str, Any]:
        """One row for the Linear-Regression vs Random-Forest comparison table."""
        return {
            "Target": self.target,
            "Model": self.model_name,
            "MAE": round(self.test_metrics["mae"], 3),
            "MSE": round(self.test_metrics["mse"], 2),
            "RMSE": round(self.test_metrics["rmse"], 3),
            "R2": round(self.test_metrics["r2"], 4),
            "MAPE %": round(self.test_metrics["mape_pct"], 2),
            "CV MAE (mean +/- sd)": (
                f"{self.cv_metrics['mae']['mean']:.3f} +/- {self.cv_metrics['mae']['std']:.3f}"
            ),
            "CV R2 (mean +/- sd)": (
                f"{self.cv_metrics['r2']['mean']:.4f} +/- {self.cv_metrics['r2']['std']:.4f}"
            ),
            "Fit time (s)": round(self.fit_seconds, 3),
        }


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def make_split(
    df: pd.DataFrame,
    target: str,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split.

    Stratifying on ``fitness_goal`` matters here: goal drives both the calorie
    factor and the protein coefficient, so an unstratified split could hand the
    test set a materially different goal mix and make the metrics unstable.
    """
    X = select_features(df)
    y = df[target].astype(float)
    stratify = df["fitness_goal"] if "fitness_goal" in df.columns else None
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------
def linear_regression_pipeline():
    return build_pipeline(LinearRegression())


def random_forest_pipeline(seed: int = RANDOM_SEED, **overrides):
    params = dict(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    params.update(overrides)
    return build_pipeline(RandomForestRegressor(**params))


#: Search space for the Random Forest.  Prefixed with ``model__`` because the
#: estimator sits inside a Pipeline.
RF_SEARCH_SPACE: dict[str, list[Any]] = {
    "model__n_estimators": [200, 300, 400, 600, 800],
    "model__max_depth": [None, 6, 8, 10, 14, 20],
    "model__min_samples_leaf": [1, 2, 3, 5, 8],
    "model__min_samples_split": [2, 4, 6, 10],
    "model__max_features": [1.0, "sqrt", "log2", 0.6, 0.8],
}


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 40,
    cv: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
    verbose: int = 0,
):
    """RandomizedSearchCV over the Random Forest, scored on negative MAE."""
    search = RandomizedSearchCV(
        estimator=random_forest_pipeline(seed=seed),
        param_distributions=RF_SEARCH_SPACE,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=cv, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=-1,
        refit=True,
        verbose=verbose,
    )
    search.fit(X_train, y_train)
    logger.info("Best RF params: %s (CV MAE %.3f)", search.best_params_, -search.best_score_)
    return search


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
def cross_validate_pipeline(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> dict[str, dict[str, float]]:
    """5-fold CV returning mean and standard deviation for each metric."""
    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=KFold(n_splits=cv, shuffle=True, random_state=seed),
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary: dict[str, dict[str, float]] = {}
    for name in scoring:
        scores = results[f"test_{name}"]
        # neg_* scorers return negated values; flip them back for reporting.
        if name != "r2":
            scores = -scores
        summary[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "folds": [float(value) for value in scores],
        }
    return summary


# --------------------------------------------------------------------------
# End-to-end training for one target
# --------------------------------------------------------------------------
def train_target(
    df: pd.DataFrame,
    target: str,
    tune: bool = True,
    n_iter: int = 40,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Train and evaluate both models for a single target.

    Returns a dict with the fitted pipelines, :class:`ModelResult` objects,
    the held-out split (for residual plots) and Random-Forest feature
    importances.
    """
    X_train, X_test, y_train, y_test = make_split(df, target, seed=seed)
    X_all, y_all = select_features(df), df[target].astype(float)

    results: dict[str, ModelResult] = {}
    fitted: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}

    # ---------------- Linear Regression (baseline) -----------------------
    linear = linear_regression_pipeline()
    started = time.perf_counter()
    linear.fit(X_train, y_train)
    linear_seconds = time.perf_counter() - started

    y_pred_linear = linear.predict(X_test)
    results["LinearRegression"] = ModelResult(
        target=target,
        model_name="Linear Regression",
        test_metrics=regression_metrics(y_test, y_pred_linear),
        train_metrics=regression_metrics(y_train, linear.predict(X_train)),
        cv_metrics=cross_validate_pipeline(linear_regression_pipeline(), X_all, y_all, seed=seed),
        fit_seconds=linear_seconds,
        n_train=len(X_train),
        n_test=len(X_test),
    )
    fitted["LinearRegression"] = linear
    predictions["LinearRegression"] = y_pred_linear

    # ---------------- Random Forest (main model) -------------------------
    if tune:
        started = time.perf_counter()
        search = tune_random_forest(X_train, y_train, n_iter=n_iter, seed=seed)
        forest = search.best_estimator_
        forest_seconds = time.perf_counter() - started
        best_params = {
            key.replace("model__", ""): value for key, value in search.best_params_.items()
        }
    else:
        forest = random_forest_pipeline(seed=seed)
        started = time.perf_counter()
        forest.fit(X_train, y_train)
        forest_seconds = time.perf_counter() - started
        best_params = {"note": "defaults (tuning disabled)"}

    y_pred_forest = forest.predict(X_test)
    forest_for_cv = build_pipeline(
        RandomForestRegressor(
            **{k: v for k, v in best_params.items() if k != "note"},
            random_state=seed,
            n_jobs=-1,
        )
    ) if tune else random_forest_pipeline(seed=seed)

    results["RandomForest"] = ModelResult(
        target=target,
        model_name="Random Forest",
        test_metrics=regression_metrics(y_test, y_pred_forest),
        train_metrics=regression_metrics(y_train, forest.predict(X_train)),
        cv_metrics=cross_validate_pipeline(forest_for_cv, X_all, y_all, seed=seed),
        best_params=best_params,
        fit_seconds=forest_seconds,
        n_train=len(X_train),
        n_test=len(X_test),
    )
    fitted["RandomForest"] = forest
    predictions["RandomForest"] = y_pred_forest

    # ---------------- Feature importance & coefficients ------------------
    feature_names = get_feature_names(forest)
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": forest.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    coefficients = pd.DataFrame({
        "feature": get_feature_names(linear),
        "coefficient": linear.named_steps["model"].coef_,
    }).sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True)

    return {
        "target": target,
        "results": results,
        "pipelines": fitted,
        "importances": importances,
        "coefficients": coefficients,
        "split": {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
        "predictions": predictions,
    }


# --------------------------------------------------------------------------
# Formula baseline
# --------------------------------------------------------------------------
def formula_baseline_metrics(df: pd.DataFrame, target: str) -> dict[str, float]:
    """Score the noise-free textbook formula against the noisy labels.

    This is the "what would a plain calculator achieve?" reference point.  The
    ML models are only interesting insofar as they approach it from raw
    features *without* being told the activity multiplier -- which is exactly
    the comparison the Analysis chapter should make.
    """
    column = f"{target}_clean"
    if column not in df.columns:
        raise KeyError(f"{column} not found; regenerate labels with nutrifit.labels.")
    return regression_metrics(df[target], df[column])


def comparison_table(all_results: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten several :func:`train_target` outputs into one report table."""
    rows = [
        result.summary_row()
        for bundle in all_results
        for result in bundle["results"].values()
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Learning curves
# --------------------------------------------------------------------------
def learning_curve_comparison(
    df: pd.DataFrame,
    target: str,
    fractions: Sequence[float] = (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0),
    cv: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Cross-validated MAE for both models across increasing training-set sizes.

    Purpose: the headline result on this dataset is that Linear Regression
    *beats* Random Forest, which contradicts the usual expectation.  A learning
    curve tests the explanation.  If the gap narrows as the sample grows, the
    cause is Random Forest's data hunger (a smooth, largely additive target
    approximated by axis-aligned splits needs far more samples than ~1000).  If
    the gap is flat, the cause is model-family mismatch alone.  Either way the
    Analysis chapter gets an evidenced conclusion instead of an assertion.
    """
    X = select_features(df)
    y = df[target].astype(float)
    stratify = df["fitness_goal"] if "fitness_goal" in df.columns else None

    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        if fraction >= 1.0:
            X_subset, y_subset = X, y
        else:
            X_subset, _, y_subset, _ = train_test_split(
                X, y, train_size=fraction, random_state=seed, stratify=stratify
            )

        n_samples = len(X_subset)
        # Guard against a fold count that exceeds the subset size.
        folds = int(min(cv, max(2, n_samples // 20)))

        for name, factory in (
            ("Linear Regression", linear_regression_pipeline),
            ("Random Forest", lambda: random_forest_pipeline(seed=seed)),
        ):
            scores = cross_validate(
                factory(),
                X_subset,
                y_subset,
                cv=KFold(n_splits=folds, shuffle=True, random_state=seed),
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )["test_score"]
            rows.append({
                "fraction": fraction,
                "n_samples": n_samples,
                "model": name,
                "cv_mae_mean": float(-np.mean(scores)),
                "cv_mae_std": float(np.std(scores)),
                "folds": folds,
            })

    return pd.DataFrame(rows)


def residual_frame(bundle: dict[str, Any], model_key: str = "RandomForest") -> pd.DataFrame:
    """Actual/predicted/residual table for the error-analysis section."""
    y_test = bundle["split"]["y_test"].to_numpy(dtype=float)
    predicted = np.asarray(bundle["predictions"][model_key], dtype=float)
    X_test = bundle["split"]["X_test"].reset_index(drop=True)

    return pd.DataFrame({
        "actual": y_test,
        "predicted": predicted,
        "residual": y_test - predicted,
        "abs_error": np.abs(y_test - predicted),
        "fitness_goal": X_test["fitness_goal"],
        "gender": X_test["gender"],
        "activity_level": X_test["activity_level"],
    })


__all__ = [
    "regression_metrics",
    "ModelResult",
    "make_split",
    "linear_regression_pipeline",
    "random_forest_pipeline",
    "RF_SEARCH_SPACE",
    "tune_random_forest",
    "cross_validate_pipeline",
    "train_target",
    "formula_baseline_metrics",
    "comparison_table",
    "learning_curve_comparison",
    "residual_frame",
    "FEATURE_COLUMNS",
]
