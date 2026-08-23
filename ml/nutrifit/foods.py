"""Food catalogue construction for the recommendation engine.

The Daily Food & Nutrition dataset is a *consumption log*, not a catalogue: the
same food appears on many rows, logged by different users on different dates
with different portion sizes.  Feeding those rows straight into a recommender
would (a) duplicate popular foods hundreds of times and let them dominate the
candidate pool, and (b) expose portion noise as if it were menu variety.

So the log is aggregated into one row per ``(food_item, meal_type)`` pair --
which is also exactly the shape of the ``foods`` table in the database schema
-- using the **median** of each macro, since medians are robust to the
occasional absurd portion entry.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from .nutrition import MEAL_SLOTS

logger = logging.getLogger(__name__)

CATALOGUE_COLUMNS = [
    "food_id", "name", "category", "meal_type", "calories",
    "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g",
    "sodium_mg", "cholesterol_mg", "source",
]

#: Minimum log entries before a ``(food, slot)`` pair is trusted, when the
#: source file is genuinely a *log*.  Pass ``min_observations=None`` (the
#: default) to let :func:`build_catalogue` decide -- see the note there.
MIN_OBSERVATIONS = 2

#: Median observations per (food, slot) above which a file is treated as a log
#: rather than an already-aggregated catalogue.
LOG_DETECTION_THRESHOLD = 3

#: Keyword rules used to place foods that carry no meal-type tag.
MEAL_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "breakfast": (
        "oat", "cereal", "porridge", "pancake", "waffle", "egg", "omelet",
        "omelette", "toast", "bagel", "granola", "muesli", "yogurt", "yoghurt",
        "smoothie", "croissant", "bacon", "sausage", "juice", "coffee", "milk",
    ),
    "lunch": (
        "sandwich", "wrap", "salad", "burger", "sushi", "burrito", "taco",
        "soup", "pasta", "noodle", "rice bowl", "quiche", "pizza",
    ),
    "dinner": (
        "steak", "chicken breast", "salmon", "curry", "roast", "stew",
        "casserole", "lasagna", "risotto", "grilled", "baked", "fillet",
        "pork", "lamb", "beef", "fish", "stir fry", "stir-fry",
    ),
    "snack": (
        "nut", "almond", "cashew", "peanut", "bar", "chip", "crisp", "cookie",
        "biscuit", "fruit", "apple", "banana", "orange", "berry", "chocolate",
        "popcorn", "cracker", "hummus", "protein shake", "jerky", "cheese",
    ),
}


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:60]


def infer_meal_type(name: str, category: str = "") -> str:
    """Best-effort meal slot for an untagged food, via keyword rules."""
    haystack = f"{name} {category}".lower()
    scores = {
        slot: sum(1 for keyword in keywords if keyword in haystack)
        for slot, keywords in MEAL_TYPE_KEYWORDS.items()
    }
    best_slot, best_score = max(scores.items(), key=lambda item: item[1])
    # "snack" is the safest default: a snack-sized portion never breaks a plan
    # the way a mis-slotted 900 kcal dinner would.
    return best_slot if best_score > 0 else "snack"


def _prepare_supplement(
    df: pd.DataFrame,
    source_label: str,
    macro_columns: list[str],
    infer_slot: bool,
) -> pd.DataFrame:
    """Shape a supplementary food source to match the aggregated catalogue.

    Shared by the USDA and local-food merges so both arrive with identical
    columns, avoiding the silent column mismatch that an ad-hoc concat invites.
    """
    out = df.copy()
    out["name"] = out["food_item"].astype(str).str.strip()

    if infer_slot or "meal_type" not in out.columns:
        out["meal_type"] = [
            infer_meal_type(name, category)
            for name, category in zip(out["name"], out.get("category", ""))
        ]
    else:
        out["meal_type"] = out["meal_type"].astype(str).str.strip().str.lower()

    for column in macro_columns:
        if column not in out.columns:
            out[column] = np.nan

    if "category" not in out.columns:
        out["category"] = "Uncategorised"

    out["n_observations"] = 1
    out["source"] = source_label
    return out


def build_catalogue(
    food_log: pd.DataFrame,
    usda: pd.DataFrame | None = None,
    local_foods: pd.DataFrame | None = None,
    min_observations: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Aggregate a food source (+ optional USDA rows) into a catalogue.

    ``min_observations=None`` (the default) auto-detects the shape of the input.
    The Daily Food & Nutrition dataset is published in two very different
    forms: some releases are true consumption *logs* (``Date``/``User_ID``
    columns, each food logged dozens of times), while others are already
    one-row-per-food *catalogues*.  Applying a "require at least 2 observations"
    filter to the latter would discard the entire dataset.

    So the threshold is chosen from the data: if the median number of rows per
    ``(food, slot)`` pair suggests a log, duplicates are required to filter out
    one-off entries; otherwise every row is kept.
    """
    df = food_log.copy()

    # --- normalise meal_type ---------------------------------------------
    df["meal_type"] = df["meal_type"].astype(str).str.strip().str.lower()
    df["meal_type"] = df["meal_type"].replace({
        "breakfast": "breakfast", "lunch": "lunch", "dinner": "dinner",
        "snack": "snack", "snacks": "snack", "supper": "dinner",
        "brunch": "breakfast", "nan": np.nan, "": np.nan, "none": np.nan,
    })
    untagged = df["meal_type"].isna()
    if untagged.any():
        df.loc[untagged, "meal_type"] = [
            infer_meal_type(name, category)
            for name, category in zip(
                df.loc[untagged, "food_item"], df.loc[untagged, "category"]
            )
        ]
    df = df[df["meal_type"].isin(MEAL_SLOTS)]

    macro_columns = [
        "calories", "protein_g", "carbs_g", "fat_g",
        "fiber_g", "sugar_g", "sodium_mg", "cholesterol_mg",
    ]

    grouped = (
        df.groupby(["food_item", "meal_type"], dropna=False)
        .agg(
            **{column: (column, "median") for column in macro_columns},
            category=("category", lambda values: values.mode().iat[0] if len(values.mode()) else "Uncategorised"),
            n_observations=("calories", "size"),
        )
        .reset_index()
    )

    median_observations = float(grouped["n_observations"].median()) if len(grouped) else 1.0
    if min_observations is None:
        looks_like_log = median_observations >= LOG_DETECTION_THRESHOLD
        min_observations = MIN_OBSERVATIONS if looks_like_log else 1
        if verbose:
            logger.info(
                "Source detected as %s (median %.1f rows per food/slot) -> min_observations=%d",
                "consumption log" if looks_like_log else "pre-aggregated catalogue",
                median_observations, min_observations,
            )

    before = len(grouped)
    grouped = grouped[grouped["n_observations"] >= min_observations]
    if verbose:
        logger.info(
            "Catalogue: %d (food, slot) pairs -> %d after requiring >=%d observations",
            before, len(grouped), min_observations,
        )

    grouped = grouped.rename(columns={"food_item": "name"})
    grouped["source"] = "Kaggle Daily Food & Nutrition"

    # --- curated regional foods -------------------------------------------
    # Merged *before* USDA so that if a name ever collides, the curated
    # per-serving entry wins the later drop_duplicates(keep="first").
    if local_foods is not None and len(local_foods):
        regional = _prepare_supplement(
            local_foods,
            source_label="Curated regional (Sri Lankan, per serving)",
            macro_columns=macro_columns,
            # These rows carry a hand-assigned meal_type; never override it
            # with keyword inference, which would mis-slot "Rice and Chicken
            # Curry" as dinner and empty the lunch pool.
            infer_slot=False,
        )
        grouped = pd.concat(
            [grouped, regional[grouped.columns.intersection(regional.columns)]],
            ignore_index=True,
        )

    # --- optional USDA enrichment ----------------------------------------
    if usda is not None and len(usda):
        enrichment = _prepare_supplement(
            usda,
            source_label="USDA FoodData Central (per 100 g)",
            macro_columns=macro_columns,
            infer_slot=True,
        )
        grouped = pd.concat(
            [grouped, enrichment[grouped.columns.intersection(enrichment.columns)]],
            ignore_index=True,
        )

    # --- clean up ---------------------------------------------------------
    grouped = grouped.dropna(subset=["calories", "protein_g", "carbs_g", "fat_g"])
    grouped[["fiber_g", "sugar_g", "sodium_mg", "cholesterol_mg"]] = (
        grouped[["fiber_g", "sugar_g", "sodium_mg", "cholesterol_mg"]].fillna(0.0)
    )
    grouped = grouped[grouped["calories"].between(20, 1500)]

    # Drop duplicate (name, slot) pairs that the USDA merge may have created.
    grouped = grouped.drop_duplicates(subset=["name", "meal_type"], keep="first")

    grouped["food_id"] = [
        f"{_slugify(name)}__{slot}"
        for name, slot in zip(grouped["name"], grouped["meal_type"])
    ]

    grouped = add_derived_features(grouped)
    grouped = grouped.sort_values(["meal_type", "name"]).reset_index(drop=True)

    if verbose:
        counts = grouped["meal_type"].value_counts().to_dict()
        logger.info("Final catalogue: %d items %s", len(grouped), counts)
    return grouped


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nutrient-density ratios used by the goal-aware re-ranking stage."""
    out = df.copy()
    calories = out["calories"].replace(0, np.nan)

    # Protein density: grams of protein per 100 kcal.  The single most useful
    # discriminator for a fat-loss goal (high satiety, lean-mass sparing).
    out["protein_density"] = (out["protein_g"] / calories * 100).fillna(0.0)
    out["fiber_density"] = (out["fiber_g"] / calories * 100).fillna(0.0)

    # Macro energy fractions (4/4/9 kcal per gram) -- used for the
    # "balanced ratio" score under a maintenance goal.
    protein_kcal = out["protein_g"] * 4.0
    carb_kcal = out["carbs_g"] * 4.0
    fat_kcal = out["fat_g"] * 9.0
    total_kcal = (protein_kcal + carb_kcal + fat_kcal).replace(0, np.nan)

    out["protein_frac"] = (protein_kcal / total_kcal).fillna(0.0)
    out["carb_frac"] = (carb_kcal / total_kcal).fillna(0.0)
    out["fat_frac"] = (fat_kcal / total_kcal).fillna(0.0)
    return out


#: Terms used to audit regional coverage.  Evidence for the "cultural
#: sensitivity / avoiding bias" strand of the Cardiff Met EDGE GLOBAL attribute,
#: and the design response to proposal §11.4 (limited food database coverage).
REGIONAL_AUDIT_TERMS = (
    "hopper", "roti", "dhal", "dal ", "sambol", "kottu", "kiribath",
    "pittu", "curry", "vadai", "kanda", "thosai", "idli", "lamprais",
    "watalappan", "rice",
)


def regional_coverage_report(df: pd.DataFrame,
                             terms: tuple[str, ...] = REGIONAL_AUDIT_TERMS) -> pd.DataFrame:
    """Count catalogue items matching each regional food term.

    Run this before and after enabling the curated set to quantify the bias and
    the correction -- a defensible, reproducible number for the report rather
    than an assertion.
    """
    names = df["name"].astype(str).str.lower()
    return pd.DataFrame(
        [{"term": term.strip(), "items": int(names.str.contains(term, regex=False).sum())}
         for term in terms]
    ).sort_values("items", ascending=False).reset_index(drop=True)


def catalogue_health_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-slot coverage summary -- evidence for the Dataset Review chapter."""
    return (
        df.groupby("meal_type")
        .agg(
            items=("food_id", "count"),
            median_calories=("calories", "median"),
            min_calories=("calories", "min"),
            max_calories=("calories", "max"),
            median_protein_g=("protein_g", "median"),
            median_protein_density=("protein_density", "median"),
        )
        .round(2)
        .reset_index()
    )


def to_sql_seed(df: pd.DataFrame, table: str = "foods") -> str:
    """Render the catalogue as an idempotent Postgres seed script."""

    def quote(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "NULL"
        return "'" + str(value).replace("'", "''") + "'"

    columns = [
        "food_id", "name", "category", "meal_type", "calories", "protein_g",
        "carbs_g", "fat_g", "fiber_g", "sugar_g", "sodium_mg",
        "cholesterol_mg", "protein_density", "source",
    ]
    lines = [
        "-- Auto-generated by ml/scripts/prepare_data.py -- do not edit by hand.",
        f"-- {len(df)} food items.",
        "begin;",
        f"insert into public.{table} (" + ", ".join(columns) + ") values",
    ]

    values: list[str] = []
    for row in df.itertuples(index=False):
        record = row._asdict()
        rendered = []
        for column in columns:
            value = record.get(column)
            if column in {"food_id", "name", "category", "meal_type", "source"}:
                rendered.append(quote(value))
            else:
                rendered.append("NULL" if pd.isna(value) else f"{float(value):.4f}")
        values.append("  (" + ", ".join(rendered) + ")")

    lines.append(",\n".join(values))
    lines.append("on conflict (food_id) do update set")
    lines.append(
        ",\n".join(
            f"  {column} = excluded.{column}"
            for column in columns
            if column != "food_id"
        )
    )
    lines.append(";")
    lines.append("commit;")
    return "\n".join(lines) + "\n"


__all__ = [
    "CATALOGUE_COLUMNS",
    "infer_meal_type",
    "build_catalogue",
    "add_derived_features",
    "catalogue_health_report",
    "regional_coverage_report",
    "REGIONAL_AUDIT_TERMS",
    "to_sql_seed",
]
