"""Central configuration: paths, constants and the reproducibility seed.

Every module in the project imports its paths from here so that the Colab
notebooks, the local training scripts and the FastAPI service all agree on
where artefacts live.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# nutrifit/config.py -> nutrifit/ -> ml/ -> <project root>
ML_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ML_DIR.parent

# Allow Colab (or any other environment) to relocate the data directory
# without editing code:  export NUTRIFIT_DATA_DIR=/content/drive/MyDrive/...
DATA_DIR = Path(os.environ.get("NUTRIFIT_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
#: Curated, version-controlled data compiled as part of this project
#: (as opposed to third-party downloads in raw/ and external/).
REFERENCE_DIR = DATA_DIR / "reference"

ARTIFACTS_DIR = Path(os.environ.get("NUTRIFIT_ARTIFACTS_DIR", ML_DIR / "artifacts"))
REPORTS_DIR = Path(os.environ.get("NUTRIFIT_REPORTS_DIR", ML_DIR / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"

# Where the FastAPI microservice expects to find the exported models.
SERVICE_MODELS_DIR = PROJECT_ROOT / "services" / "ml-service" / "models"

# --------------------------------------------------------------------------
# Canonical raw file names (what the user drops into data/raw/)
# --------------------------------------------------------------------------
GYM_RAW_FILENAME = "gym_members_exercise_tracking.csv"
FOOD_RAW_FILENAME = "daily_food_nutrition_dataset.csv"
OBESITY_RAW_FILENAME = "obesity_prediction.csv"  # optional, cross-check only

# --------------------------------------------------------------------------
# Curated reference data (committed to the repository)
# --------------------------------------------------------------------------
LOCAL_FOODS_FILENAME = "sri_lankan_foods.csv"

# --------------------------------------------------------------------------
# Processed / artefact file names
# --------------------------------------------------------------------------
USERS_PROCESSED = PROCESSED_DIR / "gym_users_labelled.csv"
FOODS_PROCESSED = PROCESSED_DIR / "food_catalogue.csv"
FOODS_SEED_SQL = PROCESSED_DIR / "foods_seed.sql"

CALORIE_MODEL_FILE = "calorie_model.pkl"
PROTEIN_MODEL_FILE = "protein_model.pkl"
RECOMMENDER_FILE = "recommender.pkl"
METRICS_FILE = "model_metrics.json"
MODEL_CARD_FILE = "model_card.json"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

MODEL_VERSION = "1.0.0"

__all__ = [name for name in dir() if not name.startswith("_")]
