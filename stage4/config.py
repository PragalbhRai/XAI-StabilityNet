"""
stage4/config.py
================
STAGE 4A — FAITHFULNESS CONFIGURATION

Extends Stage 2 configuration for faithfulness evaluation.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE2_DIR = ROOT / "stage2"
STAGE4_DIR = ROOT / "stage4"
RESULTS_DIR = ROOT / "results" / "stage4"

# Stage 4A sub-directories
FAITHFULNESS_DIR = RESULTS_DIR / "faithfulness"

for _d in (FAITHFULNESS_DIR,):
    _d.mkdir(parents=True, exist_ok=True)

# Import Stage 2 config
import sys
sys.path.insert(0, str(ROOT))
from stage2.config import (
    TRAINING_SEEDS,
    DATASETS,
    MODELS,
    TARGET,
    N_EXPLAIN,
    EXPLAIN_SEED,
    PROCESSED,
    MODELS_DIR as STAGE2_MODELS_DIR,
    EXPLAIN_DIR as STAGE2_EXPLAIN_DIR,
)

# Faithfulness-specific parameters
TOPK_VALUES = [1, 3, 5, 10]  # Top-k features to ablate
N_RANDOM_BASELINES = 30        # Random ablation baseline repetitions
FAITHFULNESS_SEED = 42         # RNG for random feature selection

