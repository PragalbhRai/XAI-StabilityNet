"""
stage2/robustness_fast.py
=========================
STAGE 2 — FAST ROBUSTNESS EXECUTOR

Drop-in replacement for the robustness Parts D+E+F loop.
Identical scientific configuration; optimised for speed.

Key improvements over the original sequential loop:
  A. Resumable at combination level — skips valid 200-row parquets.
  B. Parallel via ProcessPoolExecutor (ROBUSTNESS_WORKERS, default 3).
  C. Deterministic per-call LIME seeds derived via SHA-256 from
     (dataset, model, train_seed, instance_idx, perturbation_rate).
  D. Explainer objects reused within a worker job — model loaded once,
     SHAP TreeExplainer built once, LIME explainer built once per job.
  E. Original explanations loaded from Part-B parquets; only POST-
     perturbation SHAP+LIME is recomputed here.
  F. Atomic writes: temp file → rename.

Nothing in the scientific parameters is changed:
  - datasets / models / seeds / N_ROBUSTNESS / PERTURBATION_RATES
  - EPSILON_PREDICTION / PLAUSIBILITY_Z / TOP_K
  - LIME_NUM_SAMPLES / SHAP configuration
  - output schema (21 columns, identical to existing files)
"""

from __future__ import annotations
import os, sys, gc, hashlib, warnings, tempfile, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", message=".*LightGBM.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*No further splits.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from stage2.config import (
    PROCESSED, ROBUSTNESS_DIR, EXPLAIN_DIR, TARGET,
    TRAINING_SEEDS, DATASETS, MODELS,
    PERTURBATION_RATES, N_ROBUSTNESS, EPSILON_PREDICTION,
    PLAUSIBILITY_Z, TOP_K,
    LIME_NUM_SAMPLES, EXPLAIN_SEED, SHAP_BACKGROUND_SIZE,
)
from stage2.train import load_model, predict_proba_fn
from stage2.explain import select_background, _two_class_fn
from stage2.robustness import (
    apply_perturbation, compute_plausibility, drift_metrics,
    rank_similarity, attribution_similarity, sign_consistency, topk_overlap,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROBUSTNESS_WORKERS = int(os.environ.get("ROBUSTNESS_WORKERS", "3"))
EXPECTED_ROWS = N_ROBUSTNESS * len(PERTURBATION_RATES)   # 50 × 4 = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _combo_parquet(dataset: str, model: str, seed: int) -> Path:
    return ROBUSTNESS_DIR / f"{dataset}_{model}_seed{seed}_robustness.parquet"


def _combo_summary(dataset: str, model: str, seed: int) -> Path:
    return ROBUSTNESS_DIR / f"{dataset}_{model}_seed{seed}_robustness_summary.csv"


def is_valid(dataset: str, model: str, seed: int) -> bool:
    """Return True iff the combo parquet exists with exactly EXPECTED_ROWS clean rows."""
    p = _combo_parquet(dataset, model, seed)
    if not p.exists():
        return False
    try:
        df = pd.read_parquet(p)
        return len(df) == EXPECTED_ROWS and df.isnull().sum().sum() == 0
    except Exception:
        return False


def _lime_seed_for(dataset: str, model: str, train_seed: int,
                   instance_idx: int, rate: float) -> int:
    """
    Deterministic LIME seed derived from the five-tuple via SHA-256.
    Identical tuple always produces identical seed regardless of worker order.
    """
    key = f"{dataset}|{model}|{train_seed}|{instance_idx}|{rate:.6f}"
    digest = hashlib.sha256(key.encode()).digest()
    # take first 8 bytes as little-endian uint64, mod 2^31 for safety
    return int.from_bytes(digest[:8], "little") % (2 ** 31)


# ---------------------------------------------------------------------------
# Per-combination worker (runs in a separate process)
# ---------------------------------------------------------------------------

def _run_one_combo(args: tuple) -> dict:
    """
    Process one (dataset, model, seed) combination.
    Returns a status dict.
    """
    dataset, model_name, seed = args

    import warnings as _w
    _w.filterwarnings("ignore")

    # Re-import inside worker (needed after fork/spawn)
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    import numpy as np
    import pandas as pd
    import shap as _shap
    from lime.lime_tabular import LimeTabularExplainer
    from stage2.train import load_model, predict_proba_fn
    from stage2.explain import select_background, _two_class_fn, _shap_tree, _shap_kernel
    from stage2.robustness import (
        apply_perturbation, compute_plausibility, drift_metrics,
    )
    from stage2.config import (
        PROCESSED, ROBUSTNESS_DIR, EXPLAIN_DIR, TARGET,
        PERTURBATION_RATES, N_ROBUSTNESS, EPSILON_PREDICTION,
        PLAUSIBILITY_Z, TOP_K,
        LIME_NUM_SAMPLES, EXPLAIN_SEED, SHAP_BACKGROUND_SIZE,
    )

    tag = f"{dataset}_{model_name}_seed{seed}"
    out_path = ROBUSTNESS_DIR / f"{tag}_robustness.parquet"
    summary_path = ROBUSTNESS_DIR / f"{tag}_robustness_summary.csv"
    EXPECTED = N_ROBUSTNESS * len(PERTURBATION_RATES)

    # Skip if already valid
    if out_path.exists():
        try:
            df_check = pd.read_parquet(out_path)
            if len(df_check) == EXPECTED and df_check.isnull().sum().sum() == 0:
                return {"combo": tag, "status": "skipped", "rows": len(df_check)}
        except Exception:
            pass

    try:
        # ------------------------------------------------------------------
        # Load data once
        # ------------------------------------------------------------------
        train_df = pd.read_parquet(PROCESSED / f"{dataset}_train.parquet")
        test_df  = pd.read_parquet(PROCESSED / f"{dataset}_test.parquet")
        X_train_full  = train_df.drop(columns=[TARGET])
        X_train_arr   = X_train_full.to_numpy(dtype=float)
        feature_names = list(X_train_full.columns)
        n_feat        = len(feature_names)
        train_mean    = X_train_arr.mean(axis=0)
        train_std     = X_train_arr.std(axis=0)

        # Fixed instance selection (same as Part B)
        rng_inst = np.random.default_rng(EXPLAIN_SEED)
        n_test = len(test_df)
        idx = np.sort(rng_inst.choice(n_test, size=min(N_ROBUSTNESS, n_test), replace=False))
        X_sub  = test_df.drop(columns=[TARGET]).iloc[idx]
        y_sub  = test_df[TARGET].iloc[idx]

        # Background for KernelSHAP
        rng_bg = np.random.default_rng(EXPLAIN_SEED)
        background = select_background(X_train_full, rng_bg)

        # ------------------------------------------------------------------
        # Load model once
        # ------------------------------------------------------------------
        model      = load_model(dataset, model_name, seed)
        predict_fn = predict_proba_fn(dataset, model_name, seed)
        two_class  = _two_class_fn(predict_fn)

        # ------------------------------------------------------------------
        # Load original explanations from Part B (avoid recomputing)
        # ------------------------------------------------------------------
        shap_orig_path  = EXPLAIN_DIR / f"{tag}_shap.parquet"
        lime_orig_path  = EXPLAIN_DIR / f"{tag}_lime_coeff.parquet"

        if shap_orig_path.exists() and lime_orig_path.exists():
            shap_orig_df = pd.read_parquet(shap_orig_path)
            lime_orig_df = pd.read_parquet(lime_orig_path)
            # Build lookup: instance_index -> attribution array
            shap_orig_map = {
                int(row["instance_index"]): row[feature_names].to_numpy(dtype=float)
                for _, row in shap_orig_df.iterrows()
            }
            lime_orig_map = {
                int(row["instance_index"]): row[feature_names].to_numpy(dtype=float)
                for _, row in lime_orig_df.iterrows()
            }
            have_orig = True
        else:
            have_orig = False
            shap_orig_map = {}
            lime_orig_map = {}

        # ------------------------------------------------------------------
        # Build SHAP explainer once (tree models only; MLP uses KernelSHAP per call)
        # ------------------------------------------------------------------
        if model_name != "mlp":
            try:
                _tree_explainer = _shap.TreeExplainer(model, model_output="probability")
            except Exception:
                _tree_explainer = _shap.TreeExplainer(model)

        # ------------------------------------------------------------------
        # Build LIME explainer once per job
        # ------------------------------------------------------------------
        kernel_width = float(np.sqrt(n_feat) * 0.25)

        def _shap_single(x_arr, x_df):
            """SHAP for one instance."""
            if model_name == "mlp":
                vals, _ = _shap_kernel(predict_fn, background, x_arr.reshape(1, -1))
                return vals[0]
            else:
                vals = np.asarray(
                    _tree_explainer.shap_values(x_df, check_additivity=False)
                )
                if vals.ndim == 3:
                    vals = vals[:, :, 1]
                elif isinstance(vals, list):
                    vals = np.asarray(vals[1] if len(vals) == 2 else vals[0])
                return vals.astype(float)[0]

        def _lime_single(x_arr, lime_seed_val):
            """LIME for one instance with a specific seed."""
            expl = LimeTabularExplainer(
                X_train_arr,
                feature_names=feature_names,
                class_names=["Good", "Default"],
                mode="classification",
                discretize_continuous=False,
                kernel_width=kernel_width,
                random_state=lime_seed_val,
            )
            exp = expl.explain_instance(
                x_arr, two_class, labels=(1,),
                num_features=n_feat, num_samples=LIME_NUM_SAMPLES,
            )
            coeff = np.zeros(n_feat, dtype=float)
            for fi, c in dict(exp.as_map()[1]).items():
                coeff[fi] = c
            return coeff

        # ------------------------------------------------------------------
        # Main loop: 50 instances × 4 rates
        # ------------------------------------------------------------------
        rows = []
        for inst_i, (orig_idx, x_row) in enumerate(X_sub.iterrows()):
            x_orig = x_row.to_numpy(dtype=float)
            p_orig = float(predict_fn(x_orig.reshape(1, -1))[0])

            # Original SHAP — reuse from Part B if available
            if have_orig and int(orig_idx) in shap_orig_map:
                shap_orig = shap_orig_map[int(orig_idx)]
            else:
                x_orig_df = pd.DataFrame([x_orig], columns=feature_names)
                shap_orig = _shap_single(x_orig, x_orig_df)

            for rate in PERTURBATION_RATES:
                # Original LIME — reuse from Part B if available
                if have_orig and int(orig_idx) in lime_orig_map:
                    lime_orig = lime_orig_map[int(orig_idx)]
                else:
                    lime_seed_orig = _lime_seed_for(dataset, model_name, seed,
                                                    int(orig_idx), 0.0)
                    lime_orig = _lime_single(x_orig, lime_seed_orig)

                x_prime = apply_perturbation(x_orig, feature_names, dataset, rate)

                plaus = compute_plausibility(
                    x_orig, x_prime, feature_names, dataset, train_mean, train_std
                )

                p_pert = float(predict_fn(x_prime.reshape(1, -1))[0])
                pred_drift = abs(p_pert - p_orig)
                pred_preserving = bool(pred_drift < EPSILON_PREDICTION)

                # Post-perturbation SHAP
                x_prime_df = pd.DataFrame([x_prime], columns=feature_names)
                shap_pert = _shap_single(x_prime, x_prime_df)

                # Post-perturbation LIME — deterministic seed
                lime_seed_pert = _lime_seed_for(dataset, model_name, seed,
                                                int(orig_idx), rate)
                lime_pert = _lime_single(x_prime, lime_seed_pert)

                shap_d = drift_metrics(shap_orig, shap_pert)
                lime_d = drift_metrics(lime_orig, lime_pert)

                rows.append({
                    "dataset": dataset,
                    "model": model_name,
                    "seed": seed,
                    "instance_index": int(orig_idx),
                    "actual_default": int(y_sub.loc[orig_idx]),
                    "perturbation_rate": rate,
                    "p_orig": p_orig,
                    "p_pert": p_pert,
                    "prediction_drift": pred_drift,
                    "prediction_preserving": pred_preserving,
                    "plausibility_max_z": plaus["max_z"],
                    "plausibility_pass": plaus["plausible"],
                    "n_perturbed_features": plaus["n_perturbed"],
                    "shap_rank_sim":       shap_d["rank_similarity"],
                    "shap_attr_sim":       shap_d["attribution_similarity"],
                    "shap_sign_cons":      shap_d["sign_consistency"],
                    "shap_topk_overlap":   shap_d["topk_overlap"],
                    "lime_rank_sim":       lime_d["rank_similarity"],
                    "lime_attr_sim":       lime_d["attribution_similarity"],
                    "lime_sign_cons":      lime_d["sign_consistency"],
                    "lime_topk_overlap":   lime_d["topk_overlap"],
                })

        # ------------------------------------------------------------------
        # Atomic write: temp → rename
        # ------------------------------------------------------------------
        df = pd.DataFrame(rows)
        assert len(df) == EXPECTED, f"Expected {EXPECTED} rows, got {len(df)}"
        assert df.isnull().sum().sum() == 0, "NaNs in output"

        # Write parquet atomically
        tmp_pq = out_path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp_pq, index=False)
        tmp_pq.replace(out_path)

        # Write summary CSV
        summary_cols = [
            "dataset","model","seed","instance_index","perturbation_rate",
            "p_orig","p_pert","prediction_drift","prediction_preserving",
            "plausibility_pass","plausibility_max_z",
            "shap_rank_sim","shap_attr_sim","shap_sign_cons","shap_topk_overlap",
            "lime_rank_sim","lime_attr_sim","lime_sign_cons","lime_topk_overlap",
        ]
        df[summary_cols].to_csv(summary_path, index=False)

        del model, train_df, test_df
        gc.collect()

        return {"combo": tag, "status": "done", "rows": len(df)}

    except Exception as e:
        import traceback
        return {"combo": tag, "status": "error", "error": str(e),
                "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def build_aggregate_summary(datasets=None, models=None, seeds=None) -> pd.DataFrame:
    datasets = datasets or DATASETS
    models   = models   or MODELS
    seeds    = seeds    or TRAINING_SEEDS

    dfs = []
    for ds in datasets:
        for model_name in models:
            for seed in seeds:
                p = _combo_parquet(ds, model_name, seed)
                if p.exists():
                    try:
                        dfs.append(pd.read_parquet(p))
                    except Exception:
                        pass
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
    print(f"Aggregate summary -> {out.relative_to(ROOT)}")
    return agg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_robustness_fast(
    validate_one: bool = False,
    datasets=None, models=None, seeds=None,
    workers: int = ROBUSTNESS_WORKERS,
) -> dict:
    """
    Run all missing (dataset, model, seed) combinations.
    If validate_one=True, run only the first missing combo and return.
    """
    datasets = datasets or DATASETS
    models   = models   or MODELS
    seeds    = seeds    or TRAINING_SEEDS

    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

    # Enumerate all combos in deterministic order
    all_combos = [
        (ds, m, s)
        for ds in datasets
        for m in models
        for s in seeds
    ]

    done_before, to_run = [], []
    for combo in all_combos:
        if is_valid(*combo):
            done_before.append(combo)
        else:
            to_run.append(combo)

    print(f"\nRobustness executor: {len(done_before)} already valid, "
          f"{len(to_run)} to compute, {workers} workers")

    if validate_one and to_run:
        to_run = to_run[:1]
        print(f"VALIDATION MODE: running only {to_run[0]}")

    if not to_run:
        print("All combinations already complete.")
        return {
            "done_before": len(done_before),
            "newly_done": 0,
            "skipped": len(done_before),
            "errors": 0,
            "total_valid": len(done_before),
        }

    t0 = time.time()
    newly_done, errors, skipped = 0, 0, 0
    error_details = []

    # Use ProcessPoolExecutor for parallelism
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one_combo, combo): combo
                   for combo in to_run}

        for future in as_completed(futures):
            combo = futures[future]
            try:
                result = future.result()
                status = result.get("status", "unknown")
                rows   = result.get("rows", "?")
                combo_str = result.get("combo", str(combo))

                if status == "done":
                    newly_done += 1
                    elapsed = time.time() - t0
                    total_done = len(done_before) + newly_done
                    print(f"  [{total_done:02d}/60] {combo_str}: DONE  ({rows} rows)  "
                          f"[{elapsed:.0f}s]")
                elif status == "skipped":
                    skipped += 1
                    print(f"  [skip] {combo_str}: already valid")
                else:
                    errors += 1
                    tb = result.get("traceback", "")
                    print(f"  [ERR]  {combo_str}: {result.get('error','?')}")
                    if tb:
                        print("    " + "\n    ".join(tb.splitlines()[-10:]))
                    error_details.append(result)

            except Exception as e:
                errors += 1
                print(f"  [EXC]  {combo}: {e}")

    total_valid = len(done_before) + newly_done
    elapsed_total = time.time() - t0
    print(f"\nDone. {newly_done} new, {len(done_before)} pre-existing, "
          f"{errors} errors  ({elapsed_total:.1f}s)")

    return {
        "done_before": len(done_before),
        "newly_done": newly_done,
        "skipped": skipped,
        "errors": errors,
        "total_valid": total_valid,
        "error_details": error_details,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-one", action="store_true",
                        help="Run only the first missing combo as a validation step")
    parser.add_argument("--workers", type=int, default=ROBUSTNESS_WORKERS)
    args = parser.parse_args()

    result = run_robustness_fast(
        validate_one=args.validate_one,
        workers=args.workers,
    )

    if not args.validate_one:
        print("\nBuilding aggregate summary...")
        agg = build_aggregate_summary()
        if not agg.empty:
            print(agg.to_string(index=False))

    sys.exit(0 if result["errors"] == 0 else 1)
