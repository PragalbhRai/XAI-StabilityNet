"""
stage2/run_full.py
==================
STAGE 2 — FULL EXPERIMENT RUNNER

Executes the complete Stage-2 pipeline in order:
  A. Multi-seed model training  (train.run_training)
  B. SHAP + LIME explanations   (explain.run_explanations)
  C. LIME stochasticity         (robustness.run_lime_stochasticity)
  D+E+F. Perturbation robustness (robustness.run_robustness)
  G. Aggregate summary report   (robustness.build_summary_report)

Configuration is read entirely from stage2/config.py.
All outputs land under results/stage2/.

Usage:
    python -m stage2.run_full
"""

from __future__ import annotations
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore", message=".*LightGBM.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*No further splits.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)

from stage2.config import (
    DATASETS, MODELS, TRAINING_SEEDS,
    N_EXPLAIN, N_ROBUSTNESS, PERTURBATION_RATES,
    RESULTS_DIR, LIBRARY_VERSIONS,
)
from stage2.train import run_training
from stage2.explain import run_explanations
from stage2.robustness import run_lime_stochasticity, run_robustness, build_summary_report


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_full() -> None:
    t0 = time.time()

    _banner("STAGE 2 — FULL EXPERIMENT")
    print(f"  Datasets          : {DATASETS}")
    print(f"  Models            : {MODELS}")
    print(f"  Seeds             : {TRAINING_SEEDS}")
    print(f"  N explain         : {N_EXPLAIN}")
    print(f"  N robustness      : {N_ROBUSTNESS}")
    print(f"  Perturbation rates: {PERTURBATION_RATES}")
    print(f"  Output root       : results/stage2/")

    # ------------------------------------------------------------------
    # A. Multi-seed training
    # ------------------------------------------------------------------
    _banner("PART A: MULTI-SEED TRAINING")
    metrics_df = run_training(datasets=DATASETS, seeds=TRAINING_SEEDS)
    print(f"\n  Training complete. {len(metrics_df)} model/seed combos evaluated.")

    # ------------------------------------------------------------------
    # B. Explanations (SHAP + LIME, 50 instances per dataset)
    # ------------------------------------------------------------------
    _banner("PART B: SHAP + LIME EXPLANATIONS")
    run_explanations(datasets=DATASETS, models=MODELS, seeds=TRAINING_SEEDS)

    # ------------------------------------------------------------------
    # C. LIME stochasticity (German/XGBoost, 30 seeds, 20 instances)
    # ------------------------------------------------------------------
    _banner("PART C: LIME STOCHASTICITY")
    run_lime_stochasticity()

    # ------------------------------------------------------------------
    # D+E+F. Perturbation robustness
    # ------------------------------------------------------------------
    _banner("PART D+E+F: PERTURBATION ROBUSTNESS")
    run_robustness(datasets=DATASETS, models=MODELS, seeds=TRAINING_SEEDS)

    # ------------------------------------------------------------------
    # G. Aggregate summary
    # ------------------------------------------------------------------
    _banner("PART G: AGGREGATE SUMMARY")
    summary_df = build_summary_report(datasets=DATASETS, models=MODELS, seeds=TRAINING_SEEDS)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    _banner("STAGE 2 COMPLETE")
    print(f"  Elapsed: {elapsed/60:.1f} min")

    # Count output files
    from stage2.config import (
        MODELS_DIR, EXPLAIN_DIR, LIME_STOCH_DIR, ROBUSTNESS_DIR,
    )
    n_models    = len(list(MODELS_DIR.glob("*.joblib")))
    n_shap      = len(list(EXPLAIN_DIR.glob("*_shap.parquet")))
    n_lime_c    = len(list(EXPLAIN_DIR.glob("*_lime_coeff.parquet")))
    n_robust    = len(list(ROBUSTNESS_DIR.glob("*_robustness.parquet")))
    n_stoch     = len(list(LIME_STOCH_DIR.glob("*.parquet")))

    total_explain_rows = N_EXPLAIN * len(DATASETS) * len(MODELS) * len(TRAINING_SEEDS)
    total_robust_rows  = N_ROBUSTNESS * len(PERTURBATION_RATES) * len(DATASETS) * len(MODELS) * len(TRAINING_SEEDS)

    print(f"\n  Models saved      : {n_models}")
    print(f"  SHAP parquets     : {n_shap}")
    print(f"  LIME parquets     : {n_lime_c}")
    print(f"  Robustness pqts   : {n_robust}")
    print(f"  Stochasticity pqt : {n_stoch}")
    print(f"\n  Expected explain rows (per explainer): {total_explain_rows}")
    print(f"  Expected robust  rows               : {total_robust_rows}")

    final = {
        "status": "PASS",
        "elapsed_minutes": round(elapsed / 60, 2),
        "n_model_files": n_models,
        "n_shap_parquets": n_shap,
        "n_lime_coeff_parquets": n_lime_c,
        "n_robustness_parquets": n_robust,
        "n_stochasticity_parquets": n_stoch,
        "expected_explain_rows": total_explain_rows,
        "expected_robust_rows": total_robust_rows,
        "library_versions": LIBRARY_VERSIONS,
        "output_dir": "results/stage2/",
    }
    report_path = RESULTS_DIR / "stage2_run_report.json"
    with open(report_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n  Full report saved → {report_path.relative_to(ROOT)}")
    print("\nSTAGE 2: PASS\n")


if __name__ == "__main__":
    run_full()
