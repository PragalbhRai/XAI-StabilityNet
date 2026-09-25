"""
stage2/run_smoke_test.py
========================
SMOKE TEST: German + XGBoost + seed=42 + 10 instances

Exercises the complete Stage-2 path:
  model → SHAP → LIME → perturbation → plausibility →
  prediction → SHAP/LIME after perturbation → drift metrics →
  prediction-preserving filter

Outputs → results/stage2/smoke_test/

Pass criterion: all steps complete without error and at least one row
in the output parquet with prediction_drift and drift metrics populated.
"""

from __future__ import annotations
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import warnings

from stage2.config import (
    PROCESSED, SMOKE_DIR, TARGET, PERTURBATION_RATES,
    EPSILON_PREDICTION, PLAUSIBILITY_Z, TOP_K,
    LIME_NUM_SAMPLES, EXPLAIN_SEED, SHAP_BACKGROUND_SIZE,
    KERNEL_SHAP_NSAMPLES, LIBRARY_VERSIONS,
)
from stage2.train import run_training, predict_proba_fn, load_model
from stage2.explain import (
    select_background, _shap_tree, _lime_explain,
    _two_class_fn,
)
from stage2.robustness import (
    apply_perturbation, compute_plausibility,
    drift_metrics,
)

warnings.filterwarnings("ignore", message=".*LightGBM.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*No further splits.*", category=UserWarning)

SMOKE_DATASET = "german"
SMOKE_MODEL   = "xgboost"
SMOKE_SEED    = 42
N_SMOKE       = 10


def run_smoke_test() -> bool:
    print("=" * 60)
    print("STAGE 2 SMOKE TEST")
    print(f"  Dataset : {SMOKE_DATASET}")
    print(f"  Model   : {SMOKE_MODEL}")
    print(f"  Seed    : {SMOKE_SEED}")
    print(f"  N inst  : {N_SMOKE}")
    print("=" * 60)

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 1: Train / load model
    # ----------------------------------------------------------------
    print("\n[1/7] Loading model...")
    run_training(datasets=[SMOKE_DATASET], seeds=[SMOKE_SEED])
    predict_fn = predict_proba_fn(SMOKE_DATASET, SMOKE_MODEL, SMOKE_SEED)
    model      = load_model(SMOKE_DATASET, SMOKE_MODEL, SMOKE_SEED)
    print("      OK")

    # ----------------------------------------------------------------
    # Step 2: Load data + select instances
    # ----------------------------------------------------------------
    print("\n[2/7] Loading data and selecting instances...")
    train_df = pd.read_parquet(PROCESSED / f"{SMOKE_DATASET}_train.parquet")
    test_df  = pd.read_parquet(PROCESSED / f"{SMOKE_DATASET}_test.parquet")
    X_train  = train_df.drop(columns=[TARGET])
    X_test   = test_df.drop(columns=[TARGET])
    y_test   = test_df[TARGET]

    feature_names = list(X_train.columns)
    n_feat = len(feature_names)

    rng = np.random.default_rng(EXPLAIN_SEED)
    idx = np.sort(rng.choice(len(X_test), size=N_SMOKE, replace=False))
    X_explain = X_test.iloc[idx].copy()
    y_explain = y_test.iloc[idx].copy()

    X_train_arr  = X_train.to_numpy(dtype=float)
    train_mean   = X_train_arr.mean(axis=0)
    train_std    = X_train_arr.std(axis=0)

    background = select_background(X_train, rng)
    print(f"      train={X_train.shape}, explain={X_explain.shape}, background={background.shape}")

    # Save instances
    inst_df = X_explain.copy()
    inst_df.insert(0, "instance_index", X_explain.index.to_numpy())
    inst_df["actual_default"] = y_explain.to_numpy()
    inst_df.to_csv(SMOKE_DIR / "smoke_instances.csv", index=False)
    print("      Instances saved.")

    # ----------------------------------------------------------------
    # Step 3: SHAP (original)
    # ----------------------------------------------------------------
    print("\n[3/7] SHAP (original instances)...")
    shap_vals_orig = _shap_tree(model, SMOKE_MODEL, X_explain)
    assert shap_vals_orig.shape == (N_SMOKE, n_feat), \
        f"SHAP shape mismatch: {shap_vals_orig.shape}"
    shap_df = pd.DataFrame(shap_vals_orig, columns=feature_names)
    shap_df.insert(0, "instance_index", X_explain.index.to_numpy())
    shap_df["predicted_prob"] = predict_fn(X_explain.to_numpy(dtype=float))
    shap_df.to_csv(SMOKE_DIR / "smoke_shap_original.csv", index=False)
    print(f"      SHAP shape: {shap_vals_orig.shape}  OK")

    # ----------------------------------------------------------------
    # Step 4: LIME (original)
    # ----------------------------------------------------------------
    print("\n[4/7] LIME (original instances)...")
    lime_seed = EXPLAIN_SEED + SMOKE_SEED
    two_class = _two_class_fn(predict_fn)
    lime_coeff_orig, lime_contrib_orig = _lime_explain(
        two_class, X_train_arr, X_explain.to_numpy(dtype=float),
        feature_names, lime_seed,
    )
    assert lime_coeff_orig.shape == (N_SMOKE, n_feat), \
        f"LIME shape mismatch: {lime_coeff_orig.shape}"
    lime_df = pd.DataFrame(lime_coeff_orig, columns=feature_names)
    lime_df.insert(0, "instance_index", X_explain.index.to_numpy())
    lime_df.to_csv(SMOKE_DIR / "smoke_lime_original.csv", index=False)
    print(f"      LIME coeff shape: {lime_coeff_orig.shape}  OK")

    # ----------------------------------------------------------------
    # Step 5: Perturbation + plausibility
    # ----------------------------------------------------------------
    print("\n[5/7] Perturbation + plausibility check...")
    rows = []
    for inst_i, (orig_idx, x_row) in enumerate(X_explain.iterrows()):
        x_orig = x_row.to_numpy(dtype=float)
        p_orig = float(predict_fn(x_orig.reshape(1, -1))[0])
        shap_orig_i = shap_vals_orig[inst_i]
        lime_orig_i = lime_coeff_orig[inst_i]

        for rate in PERTURBATION_RATES:
            x_prime = apply_perturbation(x_orig, feature_names, SMOKE_DATASET, rate)
            plaus   = compute_plausibility(x_orig, x_prime, feature_names,
                                           SMOKE_DATASET, train_mean, train_std)

            # -------------------------------------------------------
            # Step 6: Prediction + SHAP + LIME after perturbation
            # -------------------------------------------------------
            p_pert = float(predict_fn(x_prime.reshape(1, -1))[0])
            pred_drift = abs(p_pert - p_orig)
            pred_preserving = bool(pred_drift < EPSILON_PREDICTION)

            # SHAP after perturbation
            x_prime_df = pd.DataFrame([x_prime], columns=feature_names)
            shap_pert_i = _shap_tree(model, SMOKE_MODEL, x_prime_df)[0]

            # LIME after perturbation
            lime_pert_i_coeff, _ = _lime_explain(
                two_class, X_train_arr, x_prime.reshape(1, -1),
                feature_names, lime_seed,
            )
            lime_pert_i = lime_pert_i_coeff[0]

            # -------------------------------------------------------
            # Step 7: Drift metrics
            # -------------------------------------------------------
            shap_d = drift_metrics(shap_orig_i, shap_pert_i)
            lime_d = drift_metrics(lime_orig_i, lime_pert_i)

            rows.append({
                "instance_index": int(orig_idx),
                "actual_default": int(y_explain.loc[orig_idx]),
                "perturbation_rate": rate,
                "p_orig": p_orig,
                "p_pert": p_pert,
                "prediction_drift": pred_drift,
                "prediction_preserving": pred_preserving,
                "plausibility_max_z": plaus["max_z"],
                "plausibility_pass": plaus["plausible"],
                "n_perturbed_features": plaus["n_perturbed"],
                # SHAP
                "shap_rank_sim":     shap_d["rank_similarity"],
                "shap_attr_sim":     shap_d["attribution_similarity"],
                "shap_sign_cons":    shap_d["sign_consistency"],
                "shap_topk_overlap": shap_d["topk_overlap"],
                # LIME
                "lime_rank_sim":     lime_d["rank_similarity"],
                "lime_attr_sim":     lime_d["attribution_similarity"],
                "lime_sign_cons":    lime_d["sign_consistency"],
                "lime_topk_overlap": lime_d["topk_overlap"],
            })

    smoke_df = pd.DataFrame(rows)
    smoke_df.to_csv(SMOKE_DIR / "smoke_robustness.csv", index=False)
    smoke_df.to_parquet(SMOKE_DIR / "smoke_robustness.parquet", index=False)

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------
    print("\n[6/7] Validating results...")
    assert len(smoke_df) == N_SMOKE * len(PERTURBATION_RATES), \
        f"Expected {N_SMOKE * len(PERTURBATION_RATES)} rows, got {len(smoke_df)}"
    assert smoke_df["prediction_drift"].notna().all(), "NaN in prediction_drift"
    assert smoke_df["shap_rank_sim"].notna().all(), "NaN in shap_rank_sim"
    assert smoke_df["lime_rank_sim"].notna().all(), "NaN in lime_rank_sim"

    n_pp = smoke_df["prediction_preserving"].sum()
    n_pl = smoke_df["plausibility_pass"].sum()
    total = len(smoke_df)

    print(f"      Rows                 : {total}")
    print(f"      Prediction-preserving: {n_pp}/{total} ({100*n_pp/total:.1f}%)")
    print(f"      Plausibility pass    : {n_pl}/{total} ({100*n_pl/total:.1f}%)")
    print(f"      Mean pred drift      : {smoke_df['prediction_drift'].mean():.4f}")
    print(f"      Mean SHAP rank_sim   : {smoke_df['shap_rank_sim'].mean():.4f}")
    print(f"      Mean LIME rank_sim   : {smoke_df['lime_rank_sim'].mean():.4f}")

    # ----------------------------------------------------------------
    # Meta report
    # ----------------------------------------------------------------
    print("\n[7/7] Writing smoke test report...")
    report = {
        "status": "PASS",
        "dataset": SMOKE_DATASET,
        "model": SMOKE_MODEL,
        "seed": SMOKE_SEED,
        "n_instances": N_SMOKE,
        "n_perturbation_rates": len(PERTURBATION_RATES),
        "total_rows": total,
        "prediction_preserving_frac": float(n_pp / total),
        "plausibility_pass_frac": float(n_pl / total),
        "mean_prediction_drift": float(smoke_df["prediction_drift"].mean()),
        "mean_shap_rank_sim": float(smoke_df["shap_rank_sim"].mean()),
        "mean_lime_rank_sim": float(smoke_df["lime_rank_sim"].mean()),
        "epsilon_prediction": EPSILON_PREDICTION,
        "plausibility_z": PLAUSIBILITY_Z,
        "library_versions": LIBRARY_VERSIONS,
        "output_dir": str(SMOKE_DIR.relative_to(ROOT)),
    }

    with open(SMOKE_DIR / "smoke_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("SMOKE TEST: PASS")
    print("=" * 60)
    print(f"Outputs in: {SMOKE_DIR.relative_to(ROOT)}")
    for p in sorted(SMOKE_DIR.iterdir()):
        print(f"  {p.name}")

    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
