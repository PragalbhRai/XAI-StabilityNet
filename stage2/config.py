"""
stage2/config.py
================
STAGE 2 — MASTER CONFIGURATION
All tunable parameters live here. No script hardcodes seeds, thresholds,
or sample counts.

Version: 1.0  (Stage 2 initial run)
"""

from pathlib import Path
import importlib.metadata

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
PROCESSED    = ROOT / "data" / "processed"
MODELS_V1    = ROOT / "models"                     # existing Phase-2 models (seed=42)
STAGE2_DIR   = ROOT / "stage2"
RESULTS_DIR  = ROOT / "results" / "stage2"

# Stage-2 sub-directories
MODELS_DIR          = RESULTS_DIR / "models"
EXPLAIN_DIR         = RESULTS_DIR / "explanations"
LIME_STOCH_DIR      = RESULTS_DIR / "lime_stochasticity"
ROBUSTNESS_DIR      = RESULTS_DIR / "robustness"
SMOKE_DIR           = RESULTS_DIR / "smoke_test"

for _d in (MODELS_DIR, EXPLAIN_DIR, LIME_STOCH_DIR,
           ROBUSTNESS_DIR, SMOKE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experimental parameters
# ---------------------------------------------------------------------------

# A. Multi-seed training
# Seeds are fixed before any run. The Phase-2 seed (42) is seed-slot 0.
TRAINING_SEEDS = [42, 0, 7, 123, 2024]    # 5 seeds; 42 re-uses existing models
N_SEEDS        = len(TRAINING_SEEDS)

DATASETS = ["german", "taiwan", "gmsc"]
MODELS   = ["xgboost", "lightgbm", "random_forest", "mlp"]
TARGET   = "Default"

# B. Explanation parameters
N_EXPLAIN            = 50     # instances per dataset (same 50 used across all seeds/models)
SHAP_BACKGROUND_SIZE = 100    # KernelSHAP background rows (≥50 for convergence)
KERNEL_SHAP_NSAMPLES = 512    # KernelSHAP Shapley samples per instance (Phase-3 used 100)
LIME_NUM_SAMPLES     = 2000   # LIME neighbourhood samples
EXPLAIN_SEED         = 42     # seed for selecting instances + LIME base seed

# C. LIME stochasticity
LIME_STOCH_SEEDS    = list(range(30))    # 30 independent LIME seeds
LIME_STOCH_N        = 20                 # instances per dataset for stochasticity test
LIME_STOCH_DATASETS = ["german"]         # subset for stochasticity (German only, fast)
LIME_STOCH_MODELS   = ["xgboost"]        # one model per dataset

# D. Perturbation experiment
PERTURBATION_RATES  = [-0.10, -0.05, +0.05, +0.10]  # must match perturbation_schema rates
N_ROBUSTNESS        = 50                  # instances per dataset for robustness experiment
                                          # (same indices as N_EXPLAIN)

# E. Prediction-preserving threshold (defined before the full run)
EPSILON_PREDICTION  = 0.05               # |P(x') - P(x)| < epsilon -> "prediction-preserving"
                                          # Chosen at 5 pp = half the "clinically meaningful"
                                          # 10 pp change threshold.

# F. Perturbation plausibility
# Simple train-distribution-based check:
# For each perturbed feature f, compute z-score against training distribution.
# Record max |z_f| across all perturbed features. If below PLAUSIBILITY_Z,
# mark as "plausible".
PLAUSIBILITY_Z      = 4.0                # |z| < 4 is within 4 SD of training mean

# G. Drift metrics
TOP_K = 5                                # for top-k overlap metric

# ---------------------------------------------------------------------------
# Library versions (recorded for reproducibility)
# ---------------------------------------------------------------------------
def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"

LIBRARY_VERSIONS = {
    "shap":     _pkg_version("shap"),
    "lime":     _pkg_version("lime"),
    "xgboost":  _pkg_version("xgboost"),
    "lightgbm": _pkg_version("lightgbm"),
    "sklearn":  _pkg_version("scikit-learn"),
    "numpy":    _pkg_version("numpy"),
    "pandas":   _pkg_version("pandas"),
}

# ---------------------------------------------------------------------------
# Semantic feature groups (for cross-dataset comparison, RQ3)
# Mapping of canonical semantic category -> per-dataset feature name(s).
# Only numeric-perturbable features that have a clear semantic analog.
# ---------------------------------------------------------------------------
SEMANTIC_GROUPS = {
    "credit_exposure": {
        "german":  ["credit_amount"],
        "taiwan":  ["LIMIT_BAL", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3"],
        "gmsc":    ["RevolvingUtilizationOfUnsecuredLines"],
    },
    "repayment_history": {
        "german":  [],                         # encoded in credit_history (categorical)
        "taiwan":  ["PAY_0", "PAY_2", "PAY_3"],
        "gmsc":    ["NumberOfTime30-59DaysPastDueNotWorse",
                    "NumberOfTimes90DaysLate",
                    "NumberOfTime60-89DaysPastDueNotWorse"],
    },
    "income_debt": {
        "german":  ["installment_rate"],
        "taiwan":  ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"],
        "gmsc":    ["DebtRatio", "MonthlyIncome"],
    },
    "tenure_age": {
        "german":  ["age", "duration"],
        "taiwan":  ["AGE"],
        "gmsc":    ["age"],
    },
    "dependents": {
        "german":  ["num_dependents"],
        "taiwan":  [],
        "gmsc":    ["NumberOfDependents"],
    },
}
