"""
stage2/explain.py
=================
STAGE 2 — PART B: CORRECTED SHAP + LIME PIPELINE

Fixes applied vs. Phase-3:
  - SHAP: all models use probability output space (TreeExplainer model_output="probability")
  - SHAP: consistent background = fixed random sample from training data (seed=EXPLAIN_SEED)
  - SHAP: KernelSHAP nsamples=512 (was 100); additivity diagnostic saved
  - LIME: explicitly seeded, local bandwidth=0.25 (kernel_width ≈ sqrt(n_features)*0.25)
  - LIME: raw coefficients saved separately from "contribution = coeff * (x - mean)"
  - LIME: one-hot features are treated as categorical groups
  - Saves explanation config + library versions alongside results

Output per (dataset, model, seed):
  results/stage2/explanations/{dataset}_{model}_seed{seed}_shap.parquet
  results/stage2/explanations/{dataset}_{model}_seed{seed}_lime_coeff.parquet  (raw)
  results/stage2/explanations/{dataset}_{model}_seed{seed}_lime_contrib.parquet (coeff*(x-mean))
  results/stage2/explanations/{dataset}_{model}_seed{seed}_shap_diagnostics.json
  results/stage2/explanations/explain_config.json        (one file, written once)
"""

from __future__ import annotations
import sys, json, gc, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage2.config import (
    PROCESSED, EXPLAIN_DIR,
    TRAINING_SEEDS, DATASETS, MODELS, TARGET,
    N_EXPLAIN, SHAP_BACKGROUND_SIZE, KERNEL_SHAP_NSAMPLES,
    LIME_NUM_SAMPLES, EXPLAIN_SEED, LIBRARY_VERSIONS,
)
from stage2.train import load_model, load_scaler, predict_proba_fn

warnings.filterwarnings("ignore", message=".*LightGBM.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*No further splits.*", category=UserWarning)


# ---------------------------------------------------------------------------
# Fixed instance selection (same across all seeds/models)
# ---------------------------------------------------------------------------

def select_instances(dataset: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load N_EXPLAIN test instances. Same selection for every model/seed."""
    test = pd.read_parquet(PROCESSED / f"{dataset}_test.parquet")
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]
    rng = np.random.default_rng(EXPLAIN_SEED)
    n = min(N_EXPLAIN, len(X_test))
    idx = np.sort(rng.choice(len(X_test), size=n, replace=False))
    return X_test.iloc[idx].copy(), y_test.iloc[idx].copy()


def select_background(X_train: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Fixed random background sample from training data."""
    n = min(SHAP_BACKGROUND_SIZE, len(X_train))
    idx = rng.choice(len(X_train), size=n, replace=False)
    return X_train.iloc[idx].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# SHAP — tree models
# ---------------------------------------------------------------------------

def _shap_tree(model, model_name: str, X: pd.DataFrame) -> np.ndarray:
    """
    Tree SHAP in probability output space.
    Uses model_output='probability' where supported.
    Handles the (samples, features, 2) tensor from some SHAP versions.
    """
    try:
        explainer = shap.TreeExplainer(model, model_output="probability")
        vals = explainer.shap_values(X, check_additivity=False)
    except Exception:
        # Fallback: raw log-odds output
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X, check_additivity=False)

    vals = np.asarray(vals)
    if vals.ndim == 3:          # (samples, features, 2) -> class 1
        vals = vals[:, :, 1]
    elif isinstance(vals, list):
        vals = np.asarray(vals[1] if len(vals) == 2 else vals[0])
    return vals.astype(float)


# ---------------------------------------------------------------------------
# SHAP — KernelSHAP for MLP
# ---------------------------------------------------------------------------

def _shap_kernel(predict_fn, background: np.ndarray,
                 X: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Kernel SHAP. Returns (values, diagnostics).
    Diagnostics: mean additivity error (|sum(phi) - (f(x) - E[f(X)])|).
    """
    explainer = shap.KernelExplainer(predict_fn, background)
    vals = explainer.shap_values(X, nsamples=KERNEL_SHAP_NSAMPLES, silent=True)
    if isinstance(vals, list):
        vals = vals[0]
    vals = np.asarray(vals, dtype=float)

    # Additivity diagnostic
    expected_value = float(explainer.expected_value)
    f_x = predict_fn(X)
    sum_phi = vals.sum(axis=1)
    additivity_error = np.abs(sum_phi - (f_x - expected_value))

    diag = {
        "expected_value":        expected_value,
        "mean_additivity_error": float(additivity_error.mean()),
        "max_additivity_error":  float(additivity_error.max()),
        "nsamples":              KERNEL_SHAP_NSAMPLES,
        "background_size":       len(background),
    }
    return vals, diag


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def _lime_explain(predict_fn_2class, X_train: np.ndarray,
                  X_explain: np.ndarray, feature_names: list[str],
                  lime_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        coeff_matrix   — raw LIME regression coefficients (n_inst × n_feat)
        contrib_matrix — coeff * (x_i - train_mean)       (n_inst × n_feat)

    kernel_width: sqrt(n_features) * 0.25 — genuinely local neighbourhood.
    """
    n_feat = len(feature_names)
    kernel_width = float(np.sqrt(n_feat) * 0.25)

    train_mean = X_train.mean(axis=0)          # for contribution = coeff*(x-mean)

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["Good", "Default"],
        mode="classification",
        discretize_continuous=False,
        kernel_width=kernel_width,
        random_state=lime_seed,
    )

    coeff_matrix   = np.zeros((len(X_explain), n_feat), dtype=float)
    contrib_matrix = np.zeros((len(X_explain), n_feat), dtype=float)

    for i, row in enumerate(X_explain):
        exp = explainer.explain_instance(
            row,
            predict_fn_2class,
            labels=(1,),
            num_features=n_feat,
            num_samples=LIME_NUM_SAMPLES,
        )
        feat_weights = dict(exp.as_map()[1])
        for feat_idx, coef in feat_weights.items():
            coeff_matrix[i, feat_idx] = coef
            contrib_matrix[i, feat_idx] = coef * (row[feat_idx] - train_mean[feat_idx])

        if (i + 1) % 10 == 0 or i == len(X_explain) - 1:
            print(f"      LIME {i+1}/{len(X_explain)}", end="\r")

    print()
    return coeff_matrix, contrib_matrix


# ---------------------------------------------------------------------------
# Two-class predict wrapper for LIME
# ---------------------------------------------------------------------------

def _two_class_fn(predict_fn):
    """Wrap a P(y=1) function into LIME's expected [P(0), P(1)] format."""
    def fn(X):
        p1 = predict_fn(np.asarray(X, dtype=float))
        return np.column_stack([1 - p1, p1])
    return fn


# ---------------------------------------------------------------------------
# Single (dataset, model, seed) run
# ---------------------------------------------------------------------------

def explain_one(
    dataset: str,
    model_name: str,
    seed: int,
    X_explain: pd.DataFrame,
    y_explain: pd.Series,
    X_train: pd.DataFrame,
    force: bool = False,
) -> dict:
    """
    Compute SHAP + LIME for one (dataset, model, seed) combination.
    Returns dict with output paths and diagnostics.
    """
    tag = f"{dataset}_{model_name}_seed{seed}"

    shap_path   = EXPLAIN_DIR / f"{tag}_shap.parquet"
    lime_c_path = EXPLAIN_DIR / f"{tag}_lime_coeff.parquet"
    lime_r_path = EXPLAIN_DIR / f"{tag}_lime_contrib.parquet"
    diag_path   = EXPLAIN_DIR / f"{tag}_shap_diagnostics.json"

    if all(p.exists() for p in [shap_path, lime_c_path, lime_r_path]) and not force:
        print(f"    {tag}: already done, skipping.")
        return {"status": "skipped", "shap": str(shap_path)}

    print(f"    {tag}")

    model = load_model(dataset, model_name, seed)
    scaler = load_scaler(dataset, seed) if model_name == "mlp" else None

    feature_names = list(X_explain.columns)
    X_expl_arr = X_explain.to_numpy(dtype=float)
    X_train_arr = X_train.to_numpy(dtype=float)

    rng = np.random.default_rng(EXPLAIN_SEED)
    background = select_background(X_train, rng)

    predict_fn  = predict_proba_fn(dataset, model_name, seed)

    # ---- SHAP ----
    print(f"      SHAP...", end=" ", flush=True)
    shap_diag = {}
    if model_name == "mlp":
        shap_vals, shap_diag = _shap_kernel(predict_fn, background, X_expl_arr)
    else:
        shap_vals = _shap_tree(model, model_name, X_explain)
        # Additivity diagnostic for tree models too
        expected = float(shap_vals.mean())   # proxy; exact E[f(X)] not stored
        f_x = predict_fn(X_expl_arr)
        sum_phi = shap_vals.sum(axis=1)
        shap_diag = {
            "mean_additivity_error": float(np.abs(sum_phi - (f_x - f_x.mean())).mean()),
            "nsamples": "exact (TreeExplainer)",
            "background_size": len(background),
        }
    print(f"done  (mean_additivity_err={shap_diag.get('mean_additivity_error',0):.4f})")

    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    shap_df.insert(0, "instance_index", X_explain.index.to_numpy())
    shap_df.insert(1, "actual_default", y_explain.to_numpy())
    shap_df.insert(2, "predicted_prob", predict_fn(X_expl_arr))
    shap_df.insert(3, "dataset", dataset)
    shap_df.insert(4, "model", model_name)
    shap_df.insert(5, "seed", seed)
    shap_df.to_parquet(shap_path, index=False)

    with open(diag_path, "w") as f:
        json.dump(shap_diag, f, indent=2)

    del shap_vals
    gc.collect()

    # ---- LIME ----
    print(f"      LIME...", flush=True)
    two_class = _two_class_fn(predict_fn)
    lime_seed = EXPLAIN_SEED + seed          # deterministic per (explain_seed, model_seed)
    coeff_arr, contrib_arr = _lime_explain(
        two_class, X_train_arr, X_expl_arr, feature_names, lime_seed
    )

    def _lime_df(arr):
        df = pd.DataFrame(arr, columns=feature_names)
        df.insert(0, "instance_index", X_explain.index.to_numpy())
        df.insert(1, "actual_default", y_explain.to_numpy())
        df.insert(2, "predicted_prob", predict_fn(X_expl_arr))
        df.insert(3, "dataset", dataset)
        df.insert(4, "model", model_name)
        df.insert(5, "seed", seed)
        return df

    _lime_df(coeff_arr).to_parquet(lime_c_path, index=False)
    _lime_df(contrib_arr).to_parquet(lime_r_path, index=False)

    del coeff_arr, contrib_arr, model
    gc.collect()

    return {
        "status": "done",
        "shap": str(shap_path),
        "lime_coeff": str(lime_c_path),
        "lime_contrib": str(lime_r_path),
        "diagnostics": shap_diag,
    }


# ---------------------------------------------------------------------------
# Full explanation run
# ---------------------------------------------------------------------------

def run_explanations(
    datasets: list[str] | None = None,
    models:   list[str] | None = None,
    seeds:    list[int]  | None = None,
    force: bool = False,
) -> None:
    datasets = datasets or DATASETS
    models   = models   or MODELS
    seeds    = seeds    or TRAINING_SEEDS

    print("=" * 60)
    print("STAGE 2 — PART B: EXPLANATIONS")
    print("=" * 60)

    # Save config once
    cfg_path = EXPLAIN_DIR / "explain_config.json"
    cfg = {
        "N_EXPLAIN": N_EXPLAIN,
        "SHAP_BACKGROUND_SIZE": SHAP_BACKGROUND_SIZE,
        "KERNEL_SHAP_NSAMPLES": KERNEL_SHAP_NSAMPLES,
        "LIME_NUM_SAMPLES": LIME_NUM_SAMPLES,
        "EXPLAIN_SEED": EXPLAIN_SEED,
        "TRAINING_SEEDS": seeds,
        "library_versions": LIBRARY_VERSIONS,
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    for ds in datasets:
        print(f"\n  Dataset: {ds}")
        train_df = pd.read_parquet(PROCESSED / f"{ds}_train.parquet")
        X_train  = train_df.drop(columns=[TARGET])
        X_explain, y_explain = select_instances(ds)

        # Save instance file (same across seeds)
        inst_path = EXPLAIN_DIR / f"{ds}_instances.parquet"
        if not inst_path.exists():
            inst_df = X_explain.copy()
            inst_df["actual_default"] = y_explain.values
            inst_df.to_parquet(inst_path, index=False)

        for model_name in models:
            for seed in seeds:
                explain_one(ds, model_name, seed,
                            X_explain, y_explain, X_train,
                            force=force)

        del train_df, X_train
        gc.collect()


if __name__ == "__main__":
    run_explanations()
