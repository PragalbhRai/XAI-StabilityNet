# XAI-StabilityNet — Audit Report (Phase 4 & Phase 5)
**Auditor:** read-only research-code auditor  
**Date:** 2026-09-24  
**Scope:** `phase3/phase3_perturbation.py`, `phase3/perturbation_constraints.py`, `phase4/phase4_perturbation.py`, `results/phase4/*`, `phase5/phase5_stability.py`, `results/phase5/*`  
**Context only:** `phase3/phase3_explain.py`  

---

## 1. Verdict Table

| Q | Topic | Verdict | One-sentence justification |
|---|-------|---------|---------------------------|
| P4-a | What each perturbation is / which script produced phase4 results | PASS | Four named relative perturbations (±5 %, ±10 %) applied per-feature; `phase4/phase4_perturbation.py` (L1-643) produced `results/phase4/`. |
| P4-b | Validity per feature type; joint vs per-feature; fixed magnitudes | ISSUE | Phase 4 perturbs features marked **immutable** in the phase-3 constraint spec (all Taiwan PAY/BILL/PAY_AMT, German num_dependents, GMSC delinquency counts); magnitudes are fixed but the sign-handling formula misbehaves for negative-valued features. |
| P4-c | Are SHAP/LIME explanations recomputed after perturbing? | ISSUE | **No**: phase 4 records only model-prediction changes; no explanation recomputation occurs, no explanation-level drift is measured anywhere in phases 4 or 5. |
| P4-d | Is prediction-flip rate used as a proxy for explanation stability? | ISSUE | Yes — `prediction_stability = 1 - flip_rate` (phase5_stability.py L470) feeds directly into the ESS; no justification or validation of this proxy is provided. |
| P4-e | Is the robustness score bounded and normalised consistently? | PASS | `RS = 0.60/(1+mean|Δp|) + 0.40*(1−flip_rate)` is in [0,1] and is evaluated identically for all 12 dataset/model pairs. |
| P4-f | Were perturbation validity checks run for all 3 datasets? | ISSUE | The validation report (`phase3_validation_{dataset}.txt`) exists only for **German** (`results/phase3_validation_german.txt` via `phase3/phase3_validation_german.txt`); no equivalent file exists for GMSC or Taiwan. Additionally, the comment in `perturbation_constraints.py` L197-198 and L258 explicitly states "not yet validated end-to-end" for both GMSC and Taiwan. |
| P5-a | Exact ESS formula / components / aggregation | PASS | See §2 below; formula is internally consistent. |
| P5-b | SHAP-LIME agreement: metric, top-k, signs, ranks, one-hot groups | ISSUE | top_k_overlap with k=10 equals 1.0 trivially for GMSC (exactly 10 features); one-hot dummies for German are treated as independent features in rank/sign metrics rather than being collapsed to their group first; category aggregation sums signed values before computing rank/sign agreement. |
| P5-c | Does perturbation component measure explanation or prediction robustness? | REQUIRED FIX | It measures **prediction** robustness only (probability change and flip rate); no explanation-attribution drift is measured. The label "explanation stability score" is therefore misleading in that component. |
| P5-d | Reproduce ESS = 0.7515 | PASS | Exactly reproduced to floating-point precision (Δ = 1.11 × 10⁻¹⁶); see §3 and `audit/reproduce_ess.py`. |
| P5-e | Cross-model/cross-dataset on comparable scales; NaNs; degenerate values | ISSUE | No NaNs; scales are comparable (all in [0,1]); however GMSC top10_overlap = 1.0 for all 400 instances (degenerate, inflates EAS by ~0.30 × (1.0 − real_overlap)); and GMSC/lightgbm RS=0.885 is a structural outlier (flip_rate 15.4 %) not investigated. |
| P5-f | Were weights chosen by data-dependent process? | ISSUE | The weights (0.40/0.20/0.40 at ESS level; 0.40/0.30/0.30 at EAS level; 0.50/0.50 at CSS level) are hard-coded constants with no ablation, cross-validation, or sensitivity analysis documented anywhere in the codebase. |
| ALSO-1 | Same 100 instances per dataset for all models? | PASS | All 24 explanation CSV files (3 datasets × 4 models × 2 explainers) contain exactly 100 rows and identical `instance_index` sets per dataset (verified by loading all files). |
| ALSO-2 | Hardcoded paths, unseeded randomness, swallowed exceptions | ISSUE | `phase4_perturbation.py` defines `SEED=42` (L54) but never uses it; no `np.random` seed is set; the `perturb_numeric` scale fall-back (`scale=1.0` when median≈0) is effectively a hidden hardcoded magnitude; `warnings.filterwarnings("ignore")` in phase3_explain.py L43 silences all warnings from SHAP and LIME. |
| ALSO-3 | Did phase3_explain.py produce results/phase3/*.csv? | PASS | Source code paths (L54, L354-355, L438-439) match the CSV naming convention; all 27 files have consecutive timestamps from 2026-08-26 18:26:11 to 18:27:23, consistent with a single run; headers match exactly what `save_explanation()` produces. |

---

## 2. ESS Formula (exact mathematical definition)

### 2.1 Inputs

For each **(dataset d, model m)** pair: 100 explanation instances, each with an N-dimensional SHAP vector **s**_i and LIME vector **l**_i (N = 48 for German, 23 for Taiwan, 10 for GMSC).

### 2.2 Per-instance metrics

**Rank agreement** (rank_agreement):

    rank_agreement_i = (ρ(rank(|s_i|), rank(|l_i|)) + 1) / 2

where ρ is Pearson correlation of rank vectors (Spearman-like on absolute values); mapped to [0,1] via (corr+1)/2.  If std=0 for either vector, returns 0.

**Sign agreement** (sign_agreement):

    sign_agreement_i = (1/|M|) Σ_{j∈M} 𝟏[sgn(s_{ij}) = sgn(l_{ij})]

where M = {j : |s_{ij}| > 1e-12 AND |l_{ij}| > 1e-12}. Returns 0 if M is empty.

**Top-k overlap** (top10_overlap, k=10):

    top10_overlap_i = |top10(|s_i|) ∩ top10(|l_i|)| / |top10(|s_i|) ∪ top10(|l_i|)|

Jaccard index; k is clamped to min(10, N). **For GMSC (N=10), this is always 1.0.**

**Explanation Agreement Score** (EAS):

    EAS_i = 0.40 × rank_agreement_i + 0.30 × sign_agreement_i + 0.30 × top10_overlap_i

### 2.3 Category aggregation (CSS)

Feature columns are mapped to semantic groups G (via `CATEGORY_MAP`). For each instance i and group g, compute:

    S_{ig} = Σ_{j∈g} s_{ij}   (signed sum of SHAP values in group g)
    L_{ig} = Σ_{j∈g} l_{ij}   (signed sum of LIME values in group g)

**Category rank agreement**: rank_agreement applied to vector (S_{ig})_{g} vs (L_{ig})_{g}.  
**Category sign agreement**: sign_agreement applied to same vectors.

**Category Stability Score:**

    CSS_i = 0.50 × cat_rank_agreement_i + 0.50 × cat_sign_agreement_i

### 2.4 Robustness score (RS) — prediction-level only

From phase-4 perturbation files (four ±5%/±10% relative perturbations of numeric features):

    prob_stability  = 1 / (1 + mean|Δp|)       over all (instance, feature, perturbation) rows
    pred_stability  = 1 − flip_rate             (fraction of perturbations that changed argmax)

    RS = 0.60 × prob_stability + 0.40 × pred_stability

### 2.5 Aggregation to (dataset, model) level

    EAS_{d,m} = mean_i(EAS_i)     over 100 instances
    CSS_{d,m} = mean_i(CSS_i)
    RS_{d,m}  = computed directly from aggregated perturbation file (not mean of per-instance)

### 2.6 Explanation Stability Score (ESS)

    ESS_{d,m} = 0.40 × EAS_{d,m} + 0.20 × CSS_{d,m} + 0.40 × RS_{d,m}

### 2.7 Overall ESS

    ESS_overall = (1/12) Σ_{d,m} ESS_{d,m}   (unweighted mean over all 12 pairs)

---

## 3. Reproduction Result

Script: `audit/reproduce_ess.py`  
All intermediate values were loaded from `results/phase5/` and `results/phase4/`.

| Step | Value |
|------|-------|
| Max |recomputed - saved| EAS (per instance) | 1.11 × 10⁻¹⁶ |
| Max |recomputed - saved| RS (per row) | 1.11 × 10⁻¹⁶ |
| Max |recomputed - saved| ESS (per row) | 1.11 × 10⁻¹⁶ |
| Recomputed overall ESS | 0.751498 |
| Saved overall ESS | 0.751498 |
| **Delta** | **1.11 × 10⁻¹⁶** |

**VERDICT: REPRODUCED** — the headline ESS = 0.7515 is internally consistent given the saved intermediate files.

Single-row walkthrough — German / random_forest:  
`ESS = 0.40 × 0.6354 + 0.20 × 0.6508 + 0.40 × 0.9982 = 0.7836`  
Saved: 0.783594. ✓

---

## 4. Ranked Defect List

### REQUIRED FIX

**[RF-1] Phase 4 perturbs features marked immutable in phase 3 (Severity: CRITICAL)**  
Evidence: `phase4/phase4_perturbation.py` L81-128 `PERTURBABLE_FEATURES`; `phase3/perturbation_constraints.py` L240-303.  
- Taiwan: all 6 PAY_x, all 6 BILL_AMT_x, all 6 PAY_AMT_x (18 features) are `immutable` in phase-3 constraints but are perturbed in phase 4.  
- GMSC: `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfDependents` — all 4 `immutable` — are perturbed.  
- German: `num_dependents` is `immutable` in phase-3 constraints but appears in `PERTURBABLE_FEATURES["german"]` (L90).  

This means phase 4 is **not** using the same constraint ontology as phase 3. The cross-phase consistency claim in the paper is unverified.

**[RF-2] The "robustness" component of ESS measures prediction stability, not explanation stability (Severity: CRITICAL)**  
Evidence: `phase5/phase5_stability.py` L446-490; no SHAP or LIME recomputation occurs in phase 4.  
The ESS is presented as an *explanation* stability score, but 40 % of its weight comes from how much a model's prediction probability changes under numeric perturbations — an entirely different quantity. Explanation attributions may change dramatically even when the prediction is stable (or vice versa).

### ISSUE

**[I-1] GMSC top10_overlap is trivially 1.0 for all instances (Severity: HIGH)**  
Evidence: GMSC has exactly 10 features (`gmsc/xgboost: shap_feat=10, lime_feat=10`); `top_k_overlap` uses k=10 (`phase5_stability.py` L307); Jaccard(10/10, 10/10) = 1.0 always.  
This inflates GMSC EAS by up to `0.30 × (1.0 − true_overlap)` relative to German and Taiwan.

**[I-2] Sign-agreement anomaly — 64 % of instances have negative SHAP-LIME correlation yet EAS > 0.5 (Severity: HIGH)**  
Evidence: 773 of 1200 rows have `shap_lime_correlation < 0`; mean sign_agreement for those rows = 0.348 (positive-corr rows: 0.489). The `sign_agreement` function (L211-234) and `rank_agreement` (L188-208) can produce moderate positive values even when the linear correlation is negative, masking real disagreement.

**[I-3] Perturbation validity checks performed only for German (Severity: HIGH)**  
Evidence: Only `phase3/phase3_validation_german.txt` exists (773 bytes); no GMSC or Taiwan validation file exists anywhere in the results tree. `perturbation_constraints.py` L197-198 reads "not yet validated end-to-end" for GMSC; L258 says the same for Taiwan.

**[I-4] perturb_numeric inverts sign of perturbation for negative-valued features (Severity: MEDIUM)**  
Evidence: `phase4/phase4_perturbation.py` L282-291.  
```python
delta = np.where(np.abs(original) > 1e-12, np.abs(original)*magnitude, scale*magnitude)
if magnitude > 0: perturbed = original + delta     # delta always >= 0
else:             perturbed = original - delta
```
For a negative original (e.g., BILL_AMT = -1000), magnitude=+0.05 gives `perturbed = -1000 + 50 = -950` (less negative = effectively a decrease in |value|), while magnitude=-0.05 gives `perturbed = -1000 - 50 = -1050` (more negative). The named perturbations ("relative_plus_5" should increase, "relative_minus_5" should decrease) are **inverted** for negative values. BILL_AMT columns in Taiwan can be negative (confirmed: min = -1000, 1-3 negative values per column in the 100-instance sample).

**[I-5] ESS weights are hardcoded without ablation or justification (Severity: MEDIUM)**  
Evidence: `phase5/phase5_stability.py` L687-696 (0.50/0.50 CSS), L692-696 (0.40/0.20/0.40 ESS), L314-318 (0.40/0.30/0.30 EAS). No sensitivity analysis, grid search, or theoretical derivation appears anywhere in the codebase.

**[I-6] GMSC/lightgbm is an unexplained structural outlier (Severity: MEDIUM)**  
Evidence: GMSC/lightgbm flip_rate = 15.35 % vs < 2.5 % for all other 11 pairs; mean|Δp| = 0.0986 vs < 0.015 everywhere else. This depresses GMSC/lightgbm RS to 0.885 while others are ≥ 0.884 (min) to 0.998. No investigation or note appears in the code.

**[I-7] phase3_explain.py silences all warnings (Severity: MEDIUM)**  
Evidence: `phase3/phase3_explain.py` L43: `warnings.filterwarnings("ignore")`. Any numerical warnings from SHAP (e.g., convergence in KernelExplainer) or LIME are suppressed globally, and their content is not logged.

**[I-8] phase4 SEED variable declared but never used (Severity: LOW)**  
Evidence: `phase4/phase4_perturbation.py` L54: `SEED = 42`; no call to `np.random.seed()`, `np.random.default_rng()`, or any other RNG initialisation. The script uses no randomness itself, so this is harmless — but the dangling constant implies intended future use that was never implemented.

**[I-9] One-hot dummies treated as independent features in EAS rank/sign metrics (Severity: LOW)**  
Evidence: `phase5/phase5_stability.py` L265-333. German has 48 SHAP feature columns including one-hot dummies (e.g., `checking_status_A12`, `checking_status_A13`, `checking_status_A14`). Rank and sign agreement treat each dummy as an independent dimension rather than collapsing the group first (as `aggregate_categories` does). This is done consistently for both SHAP and LIME so the bias partially cancels, but it inflates the dimensionality for German (48) relative to GMSC (10) and Taiwan (23).

---

## 5. UNVERIFIED Items

| Item | Reason |
|------|--------|
| Whether phase-4 perturbed results for GMSC and Taiwan honour phase-3 constraint validity (bounds, monotonicity, one-hot coherence) | No validation function is called in phase4; phase-3 validation only exists for German |
| The specific SHAP version used and whether `shap_values[1]` vs `shap_values[:,:,1]` branch in phase3_explain.py L158-166 was exercised correctly for all models | No log of SHAP library version is saved; version cannot be inferred from parquet/CSV headers |
| Whether `compute_kernel_shap` nsamples=100 (phase3_explain.py L209) produces sufficiently converged SHAP values for MLP | No convergence check or variance estimate is saved |
| Whether the test/train split used by phase3_explain.py exactly matches the split from Phase 1 | No split metadata file is present; only the processed parquet files exist |
| Whether GMSC/lightgbm's outlier flip_rate of 15.35 % is caused by a data issue, a model pathology, or the sign-inversion bug (I-4) | Would require rerunning phase 4 with corrected perturbations |

---

*End of audit report.*
