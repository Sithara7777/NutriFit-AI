"""Independent validation of the BMI classification logic.

Proposal §8.3 lists an *Obesity Prediction Dataset* as a data resource. It is
not used for training -- the system predicts continuous nutritional
requirements, not obesity classes -- but it serves a genuine purpose:
**independent verification**.

The BMI category shown to every user comes from
:func:`nutrifit.nutrition.bmi_category`, which the project implements twice
(Python for the ML service, JavaScript for the Node backend). Both are unit
tested against hand-computed WHO boundaries, but those tests only prove the code
matches *our reading* of the standard. Scoring it against ~2,100 independently
labelled records from the UCI obesity dataset tests that reading against a
third party's.

The dataset used is:
    Palechor, F. M., & de la Hoz Manotas, A. (2019). Dataset for estimation of
    obesity levels based on eating habits and physical condition in individuals
    from Colombia, Peru and Mexico. *Data in Brief*, 25, 104344.
    UCI ML Repository ID 544.

Its ``NObeyesdad`` label uses a seven-class scheme; this module maps it onto the
four WHO bands so the two are comparable:

    Insufficient_Weight                       -> underweight
    Normal_Weight                             -> normal
    Overweight_Level_I / Overweight_Level_II  -> overweight
    Obesity_Type_I / II / III                 -> obese

An important honesty note for the report: roughly 77 % of that dataset is
*synthetically balanced* via SMOTE, and its labels derive from BMI thresholds in
the first place. So high agreement confirms our arithmetic and band boundaries
are right -- it is **not** independent evidence that BMI is a good health
measure, and should not be presented as such.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .nutrition import bmi_category, calculate_bmi
from .schema import SchemaError, normalise_columns

logger = logging.getLogger(__name__)

#: Seven-class UCI label -> four-band WHO category.
OBESITY_LABEL_MAP: dict[str, str] = {
    "insufficient_weight": "underweight",
    "normal_weight": "normal",
    "overweight_level_i": "overweight",
    "overweight_level_ii": "overweight",
    "obesity_type_i": "obese",
    "obesity_type_ii": "obese",
    "obesity_type_iii": "obese",
}

OBESITY_ALIASES: dict[str, tuple[str, ...]] = {
    "gender": ("Gender", "Sex"),
    "age": ("Age",),
    "height_m": ("Height", "Height (m)", "Height_m"),
    "weight_kg": ("Weight", "Weight (kg)", "Weight_kg"),
    "obesity_label": ("NObeyesdad", "Obesity_Level", "ObesityCategory", "label", "Class"),
}

OBESITY_REQUIRED = ("height_m", "weight_kg", "obesity_label")

WHO_ORDER = ["underweight", "normal", "overweight", "obese"]


def load_obesity_dataset(path: str | Path | None = None,
                         verbose: bool = True) -> pd.DataFrame | None:
    """Load the UCI obesity dataset if present, else ``None``.

    Returns ``None`` rather than raising when the file is absent: this is an
    optional cross-check, and the pipeline must not depend on an extra manual
    download.
    """
    if path is not None:
        file_path = Path(path)
    else:
        candidates = [config.RAW_DIR / config.OBESITY_RAW_FILENAME]
        candidates += sorted(config.RAW_DIR.glob("*besity*.csv"))
        candidates += sorted(config.RAW_DIR.glob("*ObesityData*.csv"))
        existing = [p for p in candidates if p.exists()]
        if not existing:
            if verbose:
                logger.info(
                    "Obesity dataset not found in %s -- skipping BMI cross-check. "
                    "See docs/BMI_VALIDATION.md for the download link.",
                    config.RAW_DIR,
                )
            return None
        file_path = existing[0]

    raw = pd.read_csv(file_path)
    df = normalise_columns(
        raw, OBESITY_ALIASES, OBESITY_REQUIRED, dataset_name="Obesity dataset"
    )

    df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")

    # Some mirrors publish height in centimetres. Detect and normalise rather
    # than silently producing BMIs around 0.002.
    median_height = float(df["height_m"].median())
    if median_height > 3.0:
        if verbose:
            logger.info("Height appears to be in cm (median %.1f) -- converting", median_height)
        df["height_m"] = df["height_m"] / 100.0

    df["height_cm"] = df["height_m"] * 100.0
    df["obesity_label"] = (
        df["obesity_label"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    )
    df["who_category"] = df["obesity_label"].map(OBESITY_LABEL_MAP)

    unmapped = df["who_category"].isna()
    if unmapped.any():
        labels = sorted(df.loc[unmapped, "obesity_label"].unique())
        if verbose:
            logger.warning("Dropping %d row(s) with unmapped label(s): %s",
                           int(unmapped.sum()), labels)
        df = df[~unmapped]

    df = df.dropna(subset=["height_m", "weight_kg"]).reset_index(drop=True)

    if verbose:
        logger.info("Obesity dataset: %s -> %d usable rows", file_path.name, len(df))
    return df


def cross_check_bmi_classification(df: pd.DataFrame) -> dict:
    """Score :func:`bmi_category` against the dataset's own labels.

    Returns agreement rate, a confusion matrix, and the disagreement rows so
    the report can discuss *where* the two differ rather than only how often.
    """
    computed_bmi = calculate_bmi(df["weight_kg"], df["height_cm"])
    predicted = pd.Series(bmi_category(computed_bmi), index=df.index)
    actual = df["who_category"]

    agree = predicted == actual
    confusion = pd.crosstab(
        actual.rename("dataset_label"),
        predicted.rename("our_classification"),
    ).reindex(index=WHO_ORDER, columns=WHO_ORDER, fill_value=0)

    disagreements = df.loc[~agree, ["height_cm", "weight_kg"]].copy()
    disagreements["bmi"] = computed_bmi[~agree]
    disagreements["dataset_label"] = actual[~agree]
    disagreements["our_classification"] = predicted[~agree]
    # How far is each disagreement from the nearest WHO boundary? Near-boundary
    # disagreements are rounding, not logic errors -- a distinction the report
    # should make explicitly.
    boundaries = np.array([18.5, 25.0, 30.0])
    disagreements["distance_to_boundary"] = [
        float(np.min(np.abs(boundaries - value))) for value in disagreements["bmi"]
    ]

    near_boundary = int((disagreements["distance_to_boundary"] < 0.5).sum())

    return {
        "n_records": int(len(df)),
        "n_agree": int(agree.sum()),
        "agreement_rate": float(agree.mean()),
        "n_disagree": int((~agree).sum()),
        "n_disagree_near_boundary": near_boundary,
        "confusion_matrix": confusion,
        "disagreements": disagreements.sort_values("distance_to_boundary"),
        "per_category_recall": {
            category: float(
                (predicted[actual == category] == category).mean()
            ) if (actual == category).any() else float("nan")
            for category in WHO_ORDER
        },
    }


def format_report(result: dict) -> str:
    """Human-readable summary, ready to paste into the Testing chapter."""
    lines = [
        "BMI classification cross-check (UCI obesity dataset)",
        "=" * 56,
        f"Records evaluated : {result['n_records']}",
        f"Agreement         : {result['n_agree']}/{result['n_records']} "
        f"({result['agreement_rate'] * 100:.2f} %)",
        f"Disagreements     : {result['n_disagree']} "
        f"(of which {result['n_disagree_near_boundary']} lie within 0.5 BMI of a WHO boundary)",
        "",
        "Recall per WHO category:",
    ]
    for category, recall in result["per_category_recall"].items():
        lines.append(f"  {category:12s} {recall * 100:6.2f} %")
    lines += ["", "Confusion matrix (rows = dataset label, columns = our classification):", ""]
    lines.append(result["confusion_matrix"].to_string())
    return "\n".join(lines)


__all__ = [
    "OBESITY_LABEL_MAP",
    "OBESITY_ALIASES",
    "WHO_ORDER",
    "load_obesity_dataset",
    "cross_check_bmi_classification",
    "format_report",
    "SchemaError",
]
