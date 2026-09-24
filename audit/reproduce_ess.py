"""
audit/reproduce_ess.py
======================
Step-by-step reproduction of headline ESS = 0.7515 from saved intermediate
files in results/.

HARD RULES:
  - Read-only.  All file I/O reads from results/; writes only to ./audit/.
  - No model retraining; no regeneration of explanation files.

Run from the repository root:
    python audit/reproduce_ess.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parents[1]
PHASE4_DIR  = ROOT / "results" / "phase4"
PHASE5_DIR  = ROOT / "results" / "phase5"
AUDIT_DIR   = ROOT / "audit"

DATASETS = ["german", "taiwan", "gmsc"]
MODELS   = ["xgboost", "lightgbm", "random_forest", "mlp"]

out = []
out.append("=" * 70)
out.append("ESS REPRODUCTION REPORT")
out.append("Headline figure: overall_ess = 0.751498 (phase5_overall_score.csv)")
out.append("=" * 70)

# STEP 1: Load per-instance explainer-agreement scores
out.append("\n--- STEP 1: Load per-instance explainer agreement ---")
agree = pd.read_csv(PHASE5_DIR / "phase5_instance_explainer_agreement.csv")
out.append(f"  Rows: {len(agree)}  (expected: 12 combos x 100 instances = 1200)")

# STEP 2: Recompute EAS
# EAS_i = 0.40*rank_agreement + 0.30*sign_agreement + 0.30*top10_overlap
out.append("\n--- STEP 2: Recompute EAS per instance ---")
agree["recomp_eas"] = (
    0.40 * agree["rank_agreement"]
    + 0.30 * agree["sign_agreement"]
    + 0.30 * agree["top10_overlap"]
)
delta_eas = (agree["recomp_eas"] - agree["explanation_agreement_score"]).abs().max()
out.append(f"  Max |recomputed - saved| EAS: {delta_eas:.2e}")
eas_summary = (agree.groupby(["dataset", "model"])
               .agg(eas=("recomp_eas", "mean")).reset_index())
out.append(eas_summary.to_string(index=False))

# STEP 3: Recompute CSS
# CSS_i = 0.50*category_rank_agreement + 0.50*category_sign_agreement
out.append("\n--- STEP 3: Recompute CSS per instance ---")
cat = pd.read_csv(PHASE5_DIR / "phase5_instance_category_agreement.csv")
cat["recomp_css"] = 0.50*cat["category_rank_agreement"] + 0.50*cat["category_sign_agreement"]
css_summary = (cat.groupby(["dataset", "model"])
               .agg(css=("recomp_css", "mean")).reset_index())
out.append(css_summary.to_string(index=False))

# STEP 4: Recompute RS from phase-4 raw files
# prob_stability = 1/(1+mean|dp|),  pred_stability = 1-flip_rate
# RS = 0.60*prob_stability + 0.40*pred_stability
out.append("\n--- STEP 4: Recompute RS from phase-4 perturbation files ---")
rob_rows = []
for ds in DATASETS:
    for m in MODELS:
        fpath = PHASE4_DIR / f"{ds}_{m}_perturbations.csv"
        df = pd.read_csv(fpath)
        mean_abs = df["absolute_probability_change"].mean()
        flip     = df["prediction_changed"].mean()
        ps       = 1.0 / (1.0 + mean_abs)
        pe       = 1.0 - flip
        rs       = 0.60 * ps + 0.40 * pe
        rob_rows.append(dict(dataset=ds, model=m, mean_abs=mean_abs, flip=flip, rs=rs))
rob_df = pd.DataFrame(rob_rows)
out.append(rob_df.to_string(index=False))

saved_rob = pd.read_csv(PHASE5_DIR / "phase5_robustness_scores.csv")
check = rob_df.merge(saved_rob[["dataset","model","robustness_score"]], on=["dataset","model"])
check["delta"] = check["rs"] - check["robustness_score"]
out.append(f"  Max |recomputed - saved| RS: {check['delta'].abs().max():.2e}")

# STEP 5: ESS = 0.40*EAS + 0.20*CSS + 0.40*RS
out.append("\n--- STEP 5: Compute per-(dataset,model) ESS ---")
combined = eas_summary.merge(css_summary, on=["dataset","model"])
combined = combined.merge(rob_df[["dataset","model","rs"]], on=["dataset","model"])
combined["recomp_ESS"] = 0.40*combined["eas"] + 0.20*combined["css"] + 0.40*combined["rs"]
out.append(combined[["dataset","model","eas","css","rs","recomp_ESS"]].to_string(index=False))

saved_ms = pd.read_csv(PHASE5_DIR / "phase5_model_dataset_scores.csv")
cmp = combined.merge(saved_ms[["dataset","model","explanation_stability_score"]], on=["dataset","model"])
cmp["ESS_delta"] = cmp["recomp_ESS"] - cmp["explanation_stability_score"]
out.append(f"  Max |recomputed - saved| ESS: {cmp['ESS_delta'].abs().max():.2e}")

# STEP 6: Overall ESS = unweighted mean over 12 rows
out.append("\n--- STEP 6: Overall ESS ---")
overall_recomp = combined["recomp_ESS"].mean()
saved_overall  = pd.read_csv(PHASE5_DIR / "phase5_overall_score.csv")["overall_ess"].iloc[0]
out.append(f"  Recomputed = {overall_recomp:.6f}")
out.append(f"  Saved      = {saved_overall:.6f}")
out.append(f"  Delta      = {overall_recomp - saved_overall:.2e}")
verdict = "REPRODUCED (delta < 1e-6)" if abs(overall_recomp - saved_overall) < 1e-6 else f"DIVERGES by {overall_recomp - saved_overall:.6f}"
out.append(f"  VERDICT: {verdict}")

# STEP 7: Single-row walkthrough -- German / random_forest
out.append("\n--- STEP 7: Single-row walkthrough (German / random_forest) ---")
rEAS = eas_summary.query("dataset=='german' and model=='random_forest'")["eas"].values[0]
rCSS = css_summary.query("dataset=='german' and model=='random_forest'")["css"].values[0]
rRS  = rob_df[(rob_df["dataset"]=="german") & (rob_df["model"]=="random_forest")]["rs"].values[0]
rESS = 0.40*rEAS + 0.20*rCSS + 0.40*rRS
out.append(f"  EAS = {rEAS:.6f}")
out.append(f"  CSS = {rCSS:.6f}")
out.append(f"  RS  = {rRS:.6f}")
out.append(f"  ESS = 0.40x{rEAS:.4f} + 0.20x{rCSS:.4f} + 0.40x{rRS:.4f} = {rESS:.6f}")
saved_ess_val = saved_ms.query("dataset=='german' and model=='random_forest'")["explanation_stability_score"].values[0]
out.append(f"  Saved ESS = {saved_ess_val:.6f}")

report = "\n".join(out)
print(report)

out_path = AUDIT_DIR / "reproduce_ess_output.txt"
out_path.write_text(report, encoding="utf-8")
print(f"\nSaved to {out_path}")
