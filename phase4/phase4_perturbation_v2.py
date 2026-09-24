"""
phase4/phase4_perturbation.py
==============================
XAI-STABILITYNET  -  PHASE 4  -  PERTURBATION / ROBUSTNESS ANALYSIS
Stage 1.5 Corrected Implementation

Changes from the preliminary prototype (pre-stage1.5):
  RF-1 FIX: Perturbable features are now drawn exclusively from the canonical
       schema in phase3/perturbation_schema.py.  The inconsistent hand-coded
       PERTURBABLE_FEATURES dict has been removed.  Features marked
       non_perturbable or categorical_engine_only are never touched.

  I-4 FIX: The signed relative perturbation formula is now
       x' = x * (1 + r)   where r in {-0.10, -0.05, +0.05, +0.10}
       This is correct for both positive AND negative original values.
       The old formula used abs(original) and was sign-inverted for negatives.

  I-8 FIX: The unused SEED constant has been removed.  The script contains
       no stochastic operations; determinism is guaranteed by the fixed
       perturbation magnitudes.

  ordinal_step features (installment_rate, existing_credits,
  NumberOfOpenCreditLinesAndLoans, NumberRealEstateLoansOrLines) now use
  the correct +/-1 step perturbation instead of relative-%.

Purpose:
    Create controlled numeric perturbations of the SAME 100 explanation
    instances used in Phase 3 and measure prediction robustness across:
        3 datasets x 4 models x perturbable-features x 4 magnitudes

Run:
    python phase4/phase4_perturbation.py
"""

from pathlib import Path
import sys
import gc
import warnings
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
PHASE3_DIR  = ROOT / "results" / "phase3"
MODEL_DIR   = ROOT / "models"
RESULT_DIR  = ROOT / "results" / "phase4_v2"   # corrected output; do NOT overwrite old results
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Add repo root to path so we can import phase3/perturbation_schema
sys.path.insert(0, str(ROOT))
from phase3.perturbation_schema import (
    get_perturbable_features, get_non_perturbable_features,
    PERTURBATION_RATES, FeatureSpec,
)

DATASETS = ["german", "taiwan", "gmsc"]
MODELS   = ["xgboost", "lightgbm", "random_forest", "mlp"]
TARGET   = "Default"

# Relative magnitudes applied to perturbable_for_local_robustness features.
# Each magnitude is tried independently (one feature changed at a time).
NUMERIC_PERTURBATIONS = {
    "relative_minus_10": -0.10,
    "relative_minus_5":  -0.05,
    "relative_plus_5":   +0.05,
    "relative_plus_10":  +0.10,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(dataset: str, model_name: str):
    model_path = MODEL_DIR / f"{dataset}_{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")
    model = joblib.load(model_path)
    scaler = None
    if model_name == "mlp":
        scaler_path = MODEL_DIR / f"{dataset}_mlp_scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing MLP scaler: {scaler_path}")
        scaler = joblib.load(scaler_path)
    return model, scaler


def predict_probability(model, scaler, X: pd.DataFrame, model_name: str) -> np.ndarray:
    if model_name == "mlp":
        return model.predict_proba(scaler.transform(X))[:, 1]
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Signed relative perturbation  (I-4 fix)
# ---------------------------------------------------------------------------

def perturb_relative_signed(
    original: np.ndarray,
    rate: float,
    lo: float | None,
    hi: float | None,
) -> np.ndarray:
    """
    Correct signed relative perturbation:  x' = x * (1 + r)

    Works correctly for BOTH positive and negative original values.
    Zero values receive a small absolute nudge based on feature median to
    avoid producing zero output for every perturbation magnitude.

    Bounds [lo, hi] are enforced by clipping after perturbation.
    """
    perturbed = original * (1.0 + rate)

    # Handle near-zero values: use column median as fallback scale
    near_zero = np.abs(original) < 1e-12
    if near_zero.any():
        scale = np.nanmedian(np.abs(original[~near_zero])) if (~near_zero).any() else 1.0
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        perturbed[near_zero] = scale * rate  # small absolute shift

    if lo is not None:
        perturbed = np.clip(perturbed, lo, None)
    if hi is not None:
        perturbed = np.clip(perturbed, None, hi)
    return perturbed


def perturb_ordinal_step(
    original: np.ndarray,
    rate: float,
    lo: float | None,
    hi: float | None,
) -> np.ndarray:
    """
    Ordinal ±1 step perturbation.
    rate > 0 -> +1 step; rate < 0 -> -1 step.
    """
    step = 1 if rate > 0 else -1
    perturbed = original + step
    if lo is not None:
        perturbed = np.clip(perturbed, lo, None)
    if hi is not None:
        perturbed = np.clip(perturbed, None, hi)
    return np.round(perturbed).astype(original.dtype)


# ---------------------------------------------------------------------------
# Run one (dataset, model) combination
# ---------------------------------------------------------------------------

def run_dataset_model(dataset: str, model_name: str) -> pd.DataFrame | None:
    print(f"\n{'-'*70}")
    print(f"{dataset.upper()} -- {model_name.upper()}")
    print(f"{'-'*70}")

    instance_path = PHASE3_DIR / f"{dataset}_explanation_instances.csv"
    if not instance_path.exists():
        raise FileNotFoundError(f"Missing Phase-3 instances: {instance_path}")

    instances = pd.read_csv(instance_path)
    if TARGET in instances.columns:
        y_original = instances[TARGET].copy()
        X = instances.drop(columns=[TARGET]).copy()
    else:
        y_original = None
        X = instances.copy()

    model, scaler = load_model(dataset, model_name)

    # Restore exact training feature order
    feature_path = MODEL_DIR / f"{dataset}_feature_columns.joblib"
    if feature_path.exists():
        feature_columns = joblib.load(feature_path)
        missing = [c for c in feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"{dataset}: missing feature columns: {missing}")
        X = X[feature_columns]

    print(f"  Samples  : {len(X)}")
    print(f"  Features : {X.shape[1]}")

    original_prob  = predict_probability(model, scaler, X, model_name)
    original_pred  = (original_prob >= 0.5).astype(int)

    # Build list of (feature, perturbation_name, rate) from canonical schema
    perturbable = get_perturbable_features(dataset)
    non_perturbable_names = {s.feature_name for s in get_non_perturbable_features(dataset)}

    rows = []
    perturbation_count = 0

    for spec in perturbable:
        feat = spec.feature_name
        if feat not in X.columns:
            warnings.warn(f"{dataset}: perturbable feature '{feat}' not found in "
                          f"explanation instances; skipped.")
            continue
        if not pd.api.types.is_numeric_dtype(X[feat]):
            warnings.warn(f"{dataset}: feature '{feat}' is non-numeric; skipped.")
            continue

        for pert_name, rate in NUMERIC_PERTURBATIONS.items():
            X_new = X.copy()
            original_vals = X_new[feat].to_numpy(dtype=float)

            if spec.perturbation_rule == "relative_signed":
                perturbed_vals = perturb_relative_signed(
                    original_vals.copy(), rate, spec.lower_bound, spec.upper_bound)

            elif spec.perturbation_rule == "ordinal_step":
                perturbed_vals = perturb_ordinal_step(
                    original_vals.copy(), rate, spec.lower_bound, spec.upper_bound)

            elif spec.perturbation_rule == "time_anchor":
                # Time anchor: only advance (non-negative delta), treat + rates as advance
                if rate > 0:
                    delta = original_vals * rate
                    near_zero = np.abs(original_vals) < 1e-12
                    if near_zero.any():
                        delta[near_zero] = rate
                    perturbed_vals = np.clip(original_vals + delta, original_vals, spec.upper_bound)
                else:
                    # Negative rates: age cannot decrease; produce no effective change
                    perturbed_vals = original_vals.copy()
            else:
                # Should not happen for perturbable features
                warnings.warn(f"Unhandled rule '{spec.perturbation_rule}' for {feat}; skipped.")
                continue

            X_new[feat] = perturbed_vals

            pert_prob  = predict_probability(model, scaler, X_new, model_name)
            pert_pred  = (pert_prob >= 0.5).astype(int)
            prob_delta = pert_prob - original_prob
            pred_flip  = (pert_pred != original_pred).astype(int)

            for i in range(len(X)):
                rows.append({
                    "dataset":                    dataset,
                    "model":                      model_name,
                    "instance_id":                i,
                    "feature":                    feat,
                    "perturbation_rule":          spec.perturbation_rule,
                    "perturbation":               pert_name,
                    "magnitude":                  rate,
                    "original_value":             float(original_vals[i]),
                    "perturbed_value":            float(perturbed_vals[i]),
                    "original_probability":       float(original_prob[i]),
                    "perturbed_probability":      float(pert_prob[i]),
                    "probability_change":         float(prob_delta[i]),
                    "absolute_probability_change":float(abs(prob_delta[i])),
                    "original_prediction":        int(original_pred[i]),
                    "perturbed_prediction":       int(pert_pred[i]),
                    "prediction_changed":         int(pred_flip[i]),
                    "true_label":                 int(y_original.iloc[i]) if y_original is not None else -1,
                })
            perturbation_count += 1

    result = pd.DataFrame(rows)

    output_path = RESULT_DIR / f"{dataset}_{model_name}_perturbations.csv"
    result.to_csv(output_path, index=False)

    flip_rate      = result["prediction_changed"].mean() if len(result) else float("nan")
    mean_abs_delta = result["absolute_probability_change"].mean() if len(result) else float("nan")

    summary = pd.DataFrame([{
        "dataset":                         dataset,
        "model":                           model_name,
        "n_instances":                     len(X),
        "n_features_perturbed":            result["feature"].nunique() if len(result) else 0,
        "n_perturbation_types":            result["perturbation"].nunique() if len(result) else 0,
        "total_perturbation_combinations": perturbation_count,
        "mean_absolute_probability_change":mean_abs_delta,
        "prediction_flip_rate":            flip_rate,
    }])
    summary.to_csv(RESULT_DIR / f"{dataset}_{model_name}_robustness_summary.csv", index=False)

    print(f"  Perturbation rows  : {len(result):,}")
    print(f"  Mean |dp|          : {mean_abs_delta:.6f}")
    print(f"  Flip rate          : {flip_rate:.4%}")
    print(f"  Saved              : {output_path.name}")

    del model, scaler, X, instances, result
    gc.collect()
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("XAI-STABILITYNET  -  PHASE 4 v2  -  CORRECTED PERTURBATION ENGINE")
    print("Stage 1.5 corrective pass: RF-1 + I-4 fixes applied")
    print("=" * 70)
    print(f"\nOutput directory: {RESULT_DIR}")
    print("NOTE: Old results in results/phase4/ are PRESERVED and NOT overwritten.\n")

    summaries = []
    for dataset in DATASETS:
        for model_name in MODELS:
            s = run_dataset_model(dataset, model_name)
            if s is not None:
                summaries.append(s)

    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        combined.to_csv(RESULT_DIR / "phase4_v2_robustness_summary.csv", index=False)
        print("\n" + "=" * 70)
        print("PHASE 4 v2 SUMMARY")
        print("=" * 70)
        print(combined.to_string(index=False))

    print("\n" + "=" * 70)
    print("PHASE 4 v2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
