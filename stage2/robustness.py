"""
stage2/robustness.py
====================
STAGE 2 â€” PARTS C, D, E, F:
  C. LIME stochasticity
  D. Main perturbation robustness experiment
  E. Prediction-preserving analysis
  F. Perturbation plausibility

The key outputs for one (dataset, model, seed, instance, perturbation_rate) are:
  - original prediction P(x)
  - perturbed prediction P(x')
  - prediction_drift = |P(x') - P(x)|
  - prediction_preserving flag: drift < EPSILON_PREDICTION
  - plausibility z-score and flag
  - SHAP before and after perturbation
  - LIME coefficients before and after perturbation
  - rank_similarity, attribution_similarity, sign_consistency, topk_overlap

Output:
  results/stage2/robustness/{dataset}_{model}_seed{seed}_robustness.parquet
  results/stage2/robustness/{dataset}_{model}_seed{seed}_robustness_summary.csv
  results/stage2/lime_stochasticity/{dataset}_{model}_stoch.parquet
"""

from __future__ import annotations
import sys, json, gc, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage2.config import (
    PROCESSED, ROBUSTNESS_DIR, LIME_STOCH_DIR, EXPLAIN_DIR,
    TRAINING_SEEDS, DATASETS, MODELS, TARGET,
    PERTURBATION_RATES, N_ROBUSTNESS, EPSILON_PREDICTION,
    PLAUSIBILITY_Z, TOP_K,
    LIME_STOCH_SEEDS, LIME_STOCH_N, LIME_STOCH_DATASETS, LIME_STOCH_MODELS,
    LIME_NUM_SAMPLES, EXPLAIN_SEED, SHAP_BACKGROUND_SIZE, KERNEL_SHAP_NSAMPLES,
    LIBRARY_VERSIONS,
)
from stage2.train import load_model, load_scaler, predict_proba_fn
from stage2.explain import select_instances, select_background, _two_class_fn
from stage2.explain import _shap_tree, _shap_kernel, _lime_explain

import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore", message=".*LightGBM.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*No further splits.*", category=UserWarning)


# ---------------------------------------------------------------------------
# Perturbation engine (uses canonical schema)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(ROOT))
from phase4.phase4_perturbation_v2 import perturb_relative_signed, perturb_ordinal_step
from phase3.perturbation_schema import get_perturbable_features


def apply_perturbation(x: np.ndarray, feature_names: list[str],
                       dataset: str, rate: float) -> np.ndarray:
    """Apply canonical perturbation to a single instance (1D array).
    Returns the perturbed instance."""
    x_prime = x.copy().astype(float)
    for spec in get_perturbable_features(dataset):
        if spec.feature_name not in feature_names:
            continue
        fi = feature_names.index(spec.feature_name)
        v  = np.array([x_prime[fi]], dtype=float)

        if spec.perturbation_rule == "relative_signed":
            v = perturb_relative_signed(v, rate, spec.lower_bound, spec.upper_bound)
        elif spec.perturbation_rule == "ordinal_step":
            v = perturb_ordinal_step(v, rate, spec.lower_bound, spec.upper_bound)
        elif spec.perturbation_rule == "time_anchor":
            # Only advance if rate > 0
            if rate > 0:
                delta = abs(x_prime[fi]) * rate
                hi = spec.upper_bound
                v = np.array([min(x_prime[fi] + delta, hi if hi is not None else 1e18)])
        # time_follower / none: skip
        x_prime[fi] = v[0]
    return x_prime


# ---------------------------------------------------------------------------
# Plausibility check (F)
# ---------------------------------------------------------------------------

def compute_plausibility(x_orig: np.ndarray, x_prime: np.ndarray,
                         feature_names: list[str], dataset: str,
                         train_mean: np.ndarray, train_std: np.ndarray) -> dict:
    """
    For each perturbed feature compute z-score against training distribution.
    Returns: max_z, plausible (bool), n_perturbed_features.
    """
    pert_names = {s.feature_name for s in get_perturbable_features(dataset)}
    z_scores = []
    for i, fname in enumerate(feature_names):
        if fname not in pert_names:
            continue
        if x_prime[i] == x_orig[i]:
            continue
        std = train_std[i]
        if std < 1e-12:
            continue
        z = abs((x_prime[i] - train_mean[i]) / std)
        z_scores.append(z)

    max_z = float(max(z_scores)) if z_scores else 0.0
    return {
        "max_z": max_z,
        "plausible": bool(max_z < PLAUSIBILITY_Z),
        "n_perturbed": len(z_scores),
    }


# ---------------------------------------------------------------------------
# Drift metrics (D)
# ---------------------------------------------------------------------------

def rank_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of absolute attribution values."""
    abs_a = np.abs(a)
    abs_b = np.abs(b)
    if abs_a.std() < 1e-12 or abs_b.std() < 1e-12:
        return 1.0 if np.allclose(abs_a, abs_b) else 0.0
    r, _ = spearmanr(abs_a, abs_b)
    return float(r)


def attribution_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized dot product (cosine similarity) of raw attribution vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if (na < 1e-12 and nb < 1e-12) else 0.0
    return float(np.dot(a, b) / (na * nb))


def sign_consistency(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of features where sign(a) == sign(b), excluding zeros."""
    nonzero = (np.abs(a) > 1e-12) & (np.abs(b) > 1e-12)
    if not nonzero.any():
        return 1.0
    return float(np.mean(np.sign(a[nonzero]) == np.sign(b[nonzero])))


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int = TOP_K) -> float:
    """Fraction of top-k (by |attribution|) features shared between a and b."""
    top_a = set(np.argsort(np.abs(a))[-k:])
    top_b = set(np.argsort(np.abs(b))[-k:])
    if not top_a:
        return 1.0
    return float(len(top_a & top_b) / k)


def drift_metrics(orig: np.ndarray, pert: np.ndarray) -> dict:
    return {
        "rank_similarity":       rank_similarity(orig, pert),
        "attribution_similarity": attribution_similarity(orig, pert),
        "sign_consistency":      sign_consistency(orig, pert),
        "topk_overlap":          topk_overlap(orig, pert),
    }


# ---------------------------------------------------------------------------
# Single SHAP explanation for one instance (used in robustness loop)
# ---------------------------------------------------------------------------

def _single_shap(model, model_name: str, x: np.ndarray,
                 background: np.ndarray, predict_fn) -> np.ndarray:
    """SHAP for a single instance as a 1D array."""
    X = pd.DataFrame([x], columns=None) if not isinstance(x, pd.DataFrame) else x
    X = np.atleast_2d(np.asarray(x, dtype=float))

    if model_name == "mlp":
        vals, _ = _shap_kernel(predict_fn, background,
                                X.reshape(1, -1))
        return vals[0]
    else:
        X_df = pd.DataFrame(X, columns=None)  # column names don't matter here
        vals = _shap_tree(model, model_name, X_df)
        return vals[0]


def _single_lime_coeff(predict_fn, X_train: np.ndarray,
                       x: np.ndarray, n_feat: int,
                       lime_seed: int) -> np.ndarray:
    """LIME coefficients for a single instance."""
    kernel_width = float(np.sqrt(n_feat) * 0.25)
    two_class = _two_class_fn(predict_fn)
    feat_names = [str(i) for i in range(n_feat)]
    explainer = LimeTabularExplainer(
        X_train, feature_names=feat_names,
        class_names=["Good", "Default"], mode="classification",
        discretize_continuous=False, kernel_width=kernel_width,
        random_state=lime_seed,
    )
    exp = explainer.explain_instance(
        x, two_class, labels=(1,), num_features=n_feat,
        num_samples=LIME_NUM_SAMPLES,
    )
    coeff = np.zeros(n_feat, dtype=float)
    for fi, c in dict(exp.as_map()[1]).items():
        coeff[fi] = c
    return coeff


# ---------------------------------------------------------------------------
# Part C: LIME stochasticity
# ---------------------------------------------------------------------------

def run_lime_stochasticity(force: bool = False) -> None:
    """
    Explain the same (model, seed=42, instance) with 30 different LIME seeds.
    Measure explanation variability: std and range of coefficients per feature.
    """
    print("\n" + "=" * 60)
    print("STAGE 2 â€” PART C: LIME STOCHASTICITY")
    print("=" * 60)

    for ds in LIME_STOCH_DATASETS:
        for model_name in LIME_STOCH_MODELS:
            out_path = LIME_STOCH_DIR / f"{ds}_{model_name}_stoch.parquet"
            if out_path.exists() and not force:
                print(f"  {ds}/{model_name}: already done.")
                continue

            print(f"\n  {ds}/{model_name}")
            seed = TRAINING_SEEDS[0]       # always seed=42 for this test
            predict_fn = predict_proba_fn(ds, model_name, seed)

            train = pd.read_parquet(PROCESSED / f"{ds}_train.parquet")
            X_train = train.drop(columns=[TARGET]).to_numpy(dtype=float)
            feature_names = list(train.drop(columns=[TARGET]).columns)

            X_explain, y_explain = select_instances(ds)
            n = min(LIME_STOCH_N, len(X_explain))
            X_sub = X_explain.iloc[:n].to_numpy(dtype=float)
            n_feat = X_train.shape[1]

            rows = []
            for lime_seed in LIME_STOCH_SEEDS:
                print(f"    lime_seed={lime_seed}", end="\r", flush=True)
                for i, x in enumerate(X_sub):
                    coeff = _single_lime_coeff(predict_fn, X_train, x,
                                               n_feat, lime_seed)
                    r = {"dataset": ds, "model": model_name,
                         "instance_i": i, "lime_seed": lime_seed}
                    for fi, fname in enumerate(feature_names):
                        r[fname] = float(coeff[fi])
                    rows.append(r)

            df = pd.DataFrame(rows)
            df.to_parquet(out_path, index=False)
            print(f"\n    Saved â†’ {out_path.relative_to(ROOT)}")
            del train, X_train
            gc.collect()


# ---------------------------------------------------------------------------
# Part D+E+F: Main robustness experiment
# ---------------------------------------------------------------------------

def run_robustness(
    datasets:  list[str] | None = None,
    models:    list[str] | None = None,
    seeds:     list[int]  | None = None,
    n_inst:    int | None = None,
    force:     bool = False,
) -> None:
    """
    For each (dataset, model, seed, instance, rate):
      - compute original SHAP + LIME
      - apply perturbation
      - compute plausibility
      - compute perturbed SHAP + LIME
      - compute prediction drift + explanation drift metrics
    """
    datasets = datasets or DATASETS
    models   = models   or MODELS
    seeds    = seeds    or TRAINING_SEEDS
    n_inst   = n_inst   or N_ROBUSTNESS

    print("\n" + "=" * 60)
    print("STAGE 2 â€” PARTS D+E+F: ROBUSTNESS EXPERIMENT")
    print(f"Epsilon (prediction-preserving) = {EPSILON_PREDICTION}")
    print(f"Plausibility z-threshold        = {PLAUSIBILITY_Z}")
    print("=" * 60)

    for ds in datasets:
        train_df = pd.read_parquet(PROCESSED / f"{ds}_train.parquet")
        X_train_full  = train_df.drop(columns=[TARGET])
        X_train_arr   = X_train_full.to_numpy(dtype=float)
        feature_names = list(X_train_full.columns)
        n_feat        = len(feature_names)

        train_mean = X_train_arr.mean(axis=0)
        train_std  = X_train_arr.std(axis=0)

        X_explain, y_explain = select_instances(ds)
        n = min(n_inst, len(X_explain))
        X_sub = X_explain.iloc[:n]
        y_sub = y_explain.iloc[:n]

        rng        = np.random.default_rng(EXPLAIN_SEED)
        background = select_background(X_train_full, rng)

        for model_name in models:
            for seed in seeds:
                tag      = f"{ds}_{model_name}_seed{seed}"
                out_path = ROBUSTNESS_DIR / f"{tag}_robustness.parquet"
                if out_path.exists() and not force:
                    print(f"  {tag}: already done.")
                    continue

                print(f"\n  {tag}")
                model      = load_model(ds, model_name, seed)
                predict_fn = predict_proba_fn(ds, model_name, seed)
                lime_seed  = EXPLAIN_SEED + seed

                rows = []

                for inst_i, (idx, x_row) in enumerate(X_sub.iterrows()):
                    x_orig = x_row.to_numpy(dtype=float)
                    p_orig = float(predict_fn(x_orig.reshape(1, -1))[0])

                    # Original SHAP
                    shap_orig = _single_shap(model, model_name, x_orig,
                                             background, predict_fn)
                    # Original LIME
                    lime_orig = _single_lime_coeff(predict_fn, X_train_arr,
                                                   x_orig, n_feat, lime_seed)

                    for rate in PERTURBATION_RATES:
                        x_prime = apply_perturbation(x_orig, feature_names, ds, rate)

                        # Plausibility
                        plaus = compute_plausibility(
                            x_orig, x_prime, feature_names, ds,
                            train_mean, train_std
                        )

                        p_pert = float(predict_fn(x_prime.reshape(1, -1))[0])
                        pred_drift = abs(p_pert - p_orig)
                        pred_preserving = bool(pred_drift < EPSILON_PREDICTION)

                        # Perturbed SHAP
                        shap_pert = _single_shap(model, model_name, x_prime,
                                                 background, predict_fn)
                        # Perturbed LIME
                        lime_pert = _single_lime_coeff(predict_fn, X_train_arr,
                                                       x_prime, n_feat, lime_seed)

                        shap_drift = drift_metrics(shap_orig, shap_pert)
                        lime_drift = drift_metrics(lime_orig, lime_pert)

                        row = {
                            "dataset": ds, "model": model_name, "seed": seed,
                            "instance_index": int(idx),
                            "actual_default": int(y_sub.loc[idx]),
                            "perturbation_rate": rate,
                            "p_orig": p_orig,
                            "p_pert": p_pert,
                            "prediction_drift": pred_drift,
                            "prediction_preserving": pred_preserving,
                            "plausibility_max_z": plaus["max_z"],
                            "plausibility_pass": plaus["plausible"],
                            "n_perturbed_features": plaus["n_perturbed"],
                            # SHAP drift
                            "shap_rank_sim":       shap_drift["rank_similarity"],
                            "shap_attr_sim":       shap_drift["attribution_similarity"],
                            "shap_sign_cons":      shap_drift["sign_consistency"],
                            "shap_topk_overlap":   shap_drift["topk_overlap"],
                            # LIME drift (coefficients)
                            "lime_rank_sim":       lime_drift["rank_similarity"],
                            "lime_attr_sim":       lime_drift["attribution_similarity"],
                            "lime_sign_cons":      lime_drift["sign_consistency"],
                            "lime_topk_overlap":   lime_drift["topk_overlap"],
                        }
                        rows.append(row)

                    if (inst_i + 1) % 5 == 0:
                        print(f"    instance {inst_i+1}/{n}", end="\r")

                print(f"    {n} instances done")
                df = pd.DataFrame(rows)
                df.to_parquet(out_path, index=False)

                # Also write a readable summary CSV
                summary_cols = [
                    "dataset","model","seed","instance_index","perturbation_rate",
                    "p_orig","p_pert","prediction_drift","prediction_preserving",
                    "plausibility_pass","plausibility_max_z",
                    "shap_rank_sim","shap_attr_sim","shap_sign_cons","shap_topk_overlap",
                    "lime_rank_sim","lime_attr_sim","lime_sign_cons","lime_topk_overlap",
                ]
                df[summary_cols].to_csv(
                    ROBUSTNESS_DIR / f"{tag}_robustness_summary.csv", index=False
                )
                print(f"    Saved â†’ {out_path.relative_to(ROOT)}")
                del model
                gc.collect()

        del train_df, X_train_full, X_train_arr
        gc.collect()


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def build_summary_report(datasets=None, models=None, seeds=None) -> pd.DataFrame:
    """Load all robustness parquets and compute aggregate metrics."""
    datasets = datasets or DATASETS
    models   = models   or MODELS
    seeds    = seeds    or TRAINING_SEEDS

    dfs = []
    for ds in datasets:
        for model_name in models:
            for seed in seeds:
                p = ROBUSTNESS_DIR / f"{ds}_{model_name}_seed{seed}_robustness.parquet"
                if p.exists():
                    dfs.append(pd.read_parquet(p))
    if not dfs:
        print("No robustness results found.")
        return pd.DataFrame()

    all_df = pd.concat(dfs, ignore_index=True)

    agg = all_df.groupby(["dataset","model","seed","perturbation_rate"]).agg(
        n_instances=("instance_index","count"),
        mean_pred_drift=("prediction_drift","mean"),
        frac_pred_preserving=("prediction_preserving","mean"),
        frac_plausible=("plausibility_pass","mean"),
        shap_rank_sim_mean=("shap_rank_sim","mean"),
        shap_attr_sim_mean=("shap_attr_sim","mean"),
        shap_sign_cons_mean=("shap_sign_cons","mean"),
        lime_rank_sim_mean=("lime_rank_sim","mean"),
        lime_attr_sim_mean=("lime_attr_sim","mean"),
        lime_sign_cons_mean=("lime_sign_cons","mean"),
    ).reset_index()

    out = ROBUSTNESS_DIR / "aggregate_summary.csv"
    agg.to_csv(out, index=False)
    print(f"Aggregate summary â†’ {out.relative_to(ROOT)}")
    return agg


if __name__ == "__main__":
    run_lime_stochasticity()
    run_robustness()
    df = build_summary_report()
    if not df.empty:
        print(df.to_string(index=False))
