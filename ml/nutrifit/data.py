"""Raw dataset loading and cleaning.

Every loader follows the same contract: locate the file (tolerating the various
names Kaggle downloads arrive with), normalise its headers via
:mod:`nutrifit.schema`, coerce dtypes, clean, and return a tidy DataFrame with
canonical column names.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .nutrition import calculate_bmi
from .schema import (
    FOOD_ALIASES,
    FOOD_REQUIRED,
    GYM_ALIASES,
    GYM_REQUIRED,
    SchemaError,
    normalise_columns,
)

logger = logging.getLogger(__name__)

# Plausible physiological ranges.  Values outside these are treated as
# data-entry errors and clipped (not dropped -- dropping rows from a 973-row
# dataset is more costly than winsorising them).
VALID_RANGES: dict[str, tuple[float, float]] = {
    "age": (16, 80),
    "weight_kg": (35, 200),
    "height_cm": (140, 215),
    "session_duration_h": (0.25, 4.0),
    "workout_frequency": (1, 7),
    "experience_level": (1, 3),
    "fat_percentage": (5, 55),
}


def _resolve_file(directory: Path, preferred: str, patterns: tuple[str, ...]) -> Path:
    """Find a dataset file, tolerating Kaggle's varied download filenames."""
    exact = directory / preferred
    if exact.exists():
        return exact

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(directory.glob(pattern)))

    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique = [p for p in candidates if not (p in seen or seen.add(p))]

    if not unique:
        raise FileNotFoundError(
            f"No dataset file found in {directory}.\n"
            f"Expected '{preferred}' (or any of {list(patterns)}).\n"
            f"Download instructions: see data/raw/README.md"
        )
    if len(unique) > 1:
        logger.warning(
            "Multiple candidate files in %s: %s -- using %s",
            directory,
            [p.name for p in unique],
            unique[0].name,
        )
    return unique[0]


def read_csv_resilient(path: Path, verbose: bool = True) -> tuple[pd.DataFrame, int]:
    """Read a CSV, repairing rows broken by unquoted commas in a text field.

    The published Daily Food & Nutrition file contains rows such as::

        Milk (2%, 1 cup),Dairy,122,8.0,...

    where the comma inside the parenthesised food name is **not** quoted, so the
    row tokenises to 13 fields against a 12-field header and pandas aborts the
    whole read.

    Rather than dropping those rows (they are perfectly good foods) this parses
    from the right: the trailing ``n-1`` fields are positional and well-formed,
    so any excess leading fields must belong to the first column and are
    re-joined with the comma that split them. Short rows are padded with nulls.

    Returns ``(dataframe, rows_repaired)`` so the caller can record the repair
    count as data-quality evidence rather than silently hiding it.
    """
    try:
        return pd.read_csv(path), 0
    except pd.errors.ParserError:
        pass  # fall through to the repair pass

    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        width = len(header)

        rows: list[list[str | None]] = []
        repaired = 0
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue
            if len(row) > width:
                excess = len(row) - width + 1
                row = [",".join(row[:excess]).strip()] + row[excess:]
                repaired += 1
            elif len(row) < width:
                row = [*row, *([None] * (width - len(row)))]
                repaired += 1
            rows.append(row)

    if verbose and repaired:
        logger.warning(
            "%s: repaired %d malformed row(s) caused by unquoted commas.",
            path.name, repaired,
        )

    # Columns come back as text; the caller coerces the numeric ones explicitly,
    # so no dtype inference is attempted here.
    return pd.DataFrame(rows, columns=header), repaired


def _clip_to_valid_range(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Winsorise physiologically implausible values.  Returns (df, n_clipped)."""
    out = df.copy()
    clipped: dict[str, int] = {}
    for column, (low, high) in VALID_RANGES.items():
        if column not in out.columns:
            continue
        series = pd.to_numeric(out[column], errors="coerce")
        n = int(((series < low) | (series > high)).sum())
        if n:
            clipped[column] = n
        out[column] = series.clip(low, high)
    return out, clipped


# --------------------------------------------------------------------------
# Gym Members Exercise Dataset
# --------------------------------------------------------------------------
def load_gym_members(path: str | Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """Load and clean the Gym Members Exercise Dataset.

    Returns a DataFrame with canonical columns including a *recomputed*
    ``bmi`` and a derived ``height_cm``.
    """
    file_path = (
        Path(path)
        if path is not None
        else _resolve_file(
            config.RAW_DIR,
            config.GYM_RAW_FILENAME,
            ("*gym*member*.csv", "*gym*.csv"),
        )
    )

    raw = pd.read_csv(file_path)
    df = normalise_columns(raw, GYM_ALIASES, GYM_REQUIRED, dataset_name="Gym Members dataset")

    n_start = len(df)

    # --- units -----------------------------------------------------------
    # The source stores height in metres; the rest of the project works in cm.
    df["height_cm"] = pd.to_numeric(df["height_m"], errors="coerce") * 100.0

    for column in ("age", "weight_kg", "session_duration_h", "workout_frequency",
                   "experience_level", "fat_percentage", "calories_burned"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # --- gender normalisation -------------------------------------------
    df["gender"] = (
        df["gender"].astype(str).str.strip().str.lower()
        .map({"male": "Male", "m": "Male", "female": "Female", "f": "Female"})
    )

    # --- missing values ---------------------------------------------------
    n_missing_before = int(df.isna().sum().sum())
    df = df.dropna(subset=["age", "gender", "weight_kg", "height_cm"])
    n_dropped = n_start - len(df)

    # --- outliers ---------------------------------------------------------
    df, clipped = _clip_to_valid_range(df)

    # --- recompute BMI ----------------------------------------------------
    # The file ships a BMI column; we recompute it from height/weight so the
    # feature is guaranteed internally consistent, and report the discrepancy
    # as a data-quality finding for the EDA chapter.
    recomputed = calculate_bmi(df["weight_kg"], df["height_cm"])
    if "bmi" in df.columns:
        supplied = pd.to_numeric(df["bmi"], errors="coerce")
        max_diff = float(np.nanmax(np.abs(supplied - recomputed))) if len(df) else 0.0
        if verbose:
            logger.info("Max |supplied BMI - recomputed BMI| = %.3f", max_diff)
        df["bmi_supplied"] = supplied
    df["bmi"] = recomputed

    df = df.reset_index(drop=True)

    if verbose:
        logger.info(
            "Gym dataset: %s -> %d rows (%d dropped, %d missing cells, clipped=%s)",
            file_path.name, len(df), n_dropped, n_missing_before, clipped or "none",
        )
    return df


# --------------------------------------------------------------------------
# Daily Food & Nutrition Dataset
# --------------------------------------------------------------------------
def load_food_dataset(path: str | Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """Load and clean the Daily Food & Nutrition Dataset (meal-level macros)."""
    file_path = (
        Path(path)
        if path is not None
        else _resolve_file(
            config.RAW_DIR,
            config.FOOD_RAW_FILENAME,
            ("*daily*food*.csv", "*food*nutrition*.csv", "*nutrition*.csv"),
        )
    )

    raw, repaired = read_csv_resilient(file_path, verbose=verbose)
    df = normalise_columns(raw, FOOD_ALIASES, FOOD_REQUIRED, dataset_name="Food dataset")
    df.attrs["rows_repaired"] = repaired

    numeric_cols = [
        "calories", "protein_g", "carbs_g", "fat_g",
        "fiber_g", "sugar_g", "sodium_mg", "cholesterol_mg",
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan

    if "meal_type" not in df.columns:
        df["meal_type"] = np.nan
    if "category" not in df.columns:
        df["category"] = "Uncategorised"

    df["food_item"] = df["food_item"].astype(str).str.strip()
    df["meal_type"] = df["meal_type"].astype(str).str.strip().str.lower()
    df["category"] = df["category"].astype(str).str.strip()

    n_start = len(df)
    df = df.dropna(subset=["food_item", "calories", "protein_g", "carbs_g", "fat_g"])
    df = df[(df["calories"] > 0) & (df["calories"] < 2000)]
    df = df[(df["protein_g"] >= 0) & (df["protein_g"] < 200)]
    df = df.reset_index(drop=True)

    df.attrs["rows_repaired"] = repaired
    if verbose:
        logger.info(
            "Food dataset: %s -> %d rows (%d removed as invalid, %d repaired)",
            file_path.name, len(df), n_start - len(df), repaired,
        )
    return df


# --------------------------------------------------------------------------
# Curated local (Sri Lankan) foods
# --------------------------------------------------------------------------
def load_local_foods(path: str | Path | None = None,
                     verbose: bool = True) -> pd.DataFrame | None:
    """Load the curated regional food set from ``data/reference/``.

    The two public datasets this project trains on are overwhelmingly Western:
    an audit of the catalogue built from them alone found zero matches for
    hopper, roti, dhal, sambol or kottu.  For the stated target user -- a
    gym-going adult in Sri Lanka -- that is a culturally biased recommender,
    and an adherence failure rather than a cosmetic one.

    Returns ``None`` when the file is absent so the pipeline still runs, but
    unlike the USDA loader this file is committed to the repository, so a
    missing file means something is wrong rather than merely not downloaded.

    See ``data/reference/README.md`` for provenance and accuracy caveats.
    """
    file_path = (
        Path(path) if path is not None
        else config.REFERENCE_DIR / config.LOCAL_FOODS_FILENAME
    )

    if not file_path.exists():
        if verbose:
            logger.warning(
                "Local food reference not found at %s -- catalogue will contain "
                "no regional items.", file_path,
            )
        return None

    df = pd.read_csv(file_path)
    df = normalise_columns(df, FOOD_ALIASES, FOOD_REQUIRED, dataset_name="Local foods")

    numeric_cols = [
        "calories", "protein_g", "carbs_g", "fat_g",
        "fiber_g", "sugar_g", "sodium_mg", "cholesterol_mg",
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan

    df["food_item"] = df["food_item"].astype(str).str.strip()
    df["meal_type"] = df["meal_type"].astype(str).str.strip().str.lower()
    if "category" not in df.columns:
        df["category"] = "Regional"

    n_start = len(df)
    df = df.dropna(subset=["food_item", "calories", "protein_g", "carbs_g", "fat_g"])
    df = df[df["meal_type"].isin(("breakfast", "lunch", "dinner", "snack"))]
    df = df.reset_index(drop=True)

    if len(df) < n_start and verbose:
        logger.warning("Local foods: dropped %d invalid row(s)", n_start - len(df))

    if verbose:
        by_slot = df["meal_type"].value_counts().to_dict()
        logger.info("Local foods: %d items %s", len(df), by_slot)
    return df


# --------------------------------------------------------------------------
# USDA FoodData Central (optional enrichment)
# --------------------------------------------------------------------------
def load_usda_foundation(directory: str | Path | None = None,
                         verbose: bool = True) -> pd.DataFrame | None:
    """Load USDA FoodData Central *Foundation Foods* bulk CSVs, if present.

    Expects the unzipped bulk download in ``data/external/`` containing at
    least ``food.csv`` and ``food_nutrient.csv``.  Returns ``None`` when the
    files are absent, so the pipeline degrades gracefully to the Kaggle
    catalogue alone.

    Nutrient IDs used (FoodData Central ``nutrient.id``):
        1008 Energy (kcal) | 1003 Protein | 1005 Carbohydrate, by difference
        1004 Total lipid (fat) | 1079 Fiber, total dietary | 2000 Total sugars
        1093 Sodium | 1253 Cholesterol
    """
    base = Path(directory) if directory is not None else config.EXTERNAL_DIR
    food_file = base / "food.csv"
    nutrient_file = base / "food_nutrient.csv"

    if not (food_file.exists() and nutrient_file.exists()):
        if verbose:
            logger.info("USDA bulk files not found in %s -- skipping enrichment.", base)
        return None

    nutrient_map = {
        1008: "calories", 1003: "protein_g", 1005: "carbs_g", 1004: "fat_g",
        1079: "fiber_g", 2000: "sugar_g", 1093: "sodium_mg", 1253: "cholesterol_mg",
    }

    foods = pd.read_csv(food_file, usecols=["fdc_id", "description", "data_type"])
    foods = foods[foods["data_type"].isin(["foundation_food", "sr_legacy_food"])]

    nutrients = pd.read_csv(nutrient_file, usecols=["fdc_id", "nutrient_id", "amount"])
    nutrients = nutrients[nutrients["nutrient_id"].isin(nutrient_map)]

    wide = (
        nutrients.pivot_table(
            index="fdc_id", columns="nutrient_id", values="amount", aggfunc="mean"
        )
        .rename(columns=nutrient_map)
        .reset_index()
    )

    merged = foods.merge(wide, on="fdc_id", how="inner")
    merged = merged.rename(columns={"description": "food_item"})
    merged["category"] = "USDA"
    merged["source"] = "USDA FoodData Central"

    # USDA values are per 100 g; keep that basis and record it explicitly.
    merged["serving_basis"] = "per_100g"

    merged = merged.dropna(subset=["calories", "protein_g", "carbs_g", "fat_g"])
    merged = merged[merged["calories"] > 0].reset_index(drop=True)

    if verbose:
        logger.info("USDA Foundation/SR-Legacy: %d usable food records", len(merged))
    return merged


__all__ = [
    "load_gym_members",
    "load_food_dataset",
    "load_local_foods",
    "load_usda_foundation",
    "SchemaError",
    "VALID_RANGES",
]
