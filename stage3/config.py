# stage3/config.py
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
STAGE2_RESULTS = ROOT / "results" / "stage2"
EXPLANATIONS = STAGE2_RESULTS / "explanations"
ROBUSTNESS = STAGE2_RESULTS / "robustness"
LIME_STOCH = STAGE2_RESULTS / "lime_stochasticity"
STAGE3_RESULTS = ROOT / "results" / "stage3"
PLOTS_DIR = STAGE3_RESULTS / "plots"
for d in (STAGE3_RESULTS, PLOTS_DIR): d.mkdir(parents=True, exist_ok=True)
DATASETS = ["german", "taiwan", "gmsc"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]
TRAINING_SEEDS = [42, 0, 7, 123, 2024]
EXPLAINERS = ["shap", "lime"]
N_EXPLAIN = 50
PERTURBATION_RATES = [-0.10, -0.05, 0.05, 0.10]
EPSILON_PREDICTION = 0.05
PLAUSIBILITY_Z = 4.0
TOP_K = 5
FEATURE_COUNTS = {"german": 48, "taiwan": 23, "gmsc": 10}
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 5000
ALPHA = 0.05
