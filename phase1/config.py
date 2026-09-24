"""
config.py
Central configuration: file paths and dataset schemas.
Keeping this separate means Phases 2-5 never hardcode paths/column names again.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TARGET_COL = "Default"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# raw_filename: expected file inside data/raw/
# download_url: only populated when a direct, script-fetchable URL exists
#   (GMSC and Taiwan are Kaggle/UCI-hosted and require manual download —
#   see README instructions printed by phase1_preprocessing.py)
DATASETS = {
    "german": {
        "raw_filename": "german.csv",
        "download_url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv",
    },
    "gmsc": {
        "raw_filename": "cs-training.csv",
        "download_url": None,  # Kaggle: "Give Me Some Credit" competition, requires login
    },
    "taiwan": {
        "raw_filename": "UCI_Credit_Card.csv",
        "download_url": None,  # UCI/Kaggle mirror, requires manual download
    },
}
