# INDEPENDENT AUDIT REPORT — XAI-STABILITYNET
# Auditor: Claude (Independent Verification)
# Date: 2026-09-25
# Audit Scope: Experimental methodology, data validation, GitHub readiness

================================================================================
EXECUTIVE SUMMARY
================================================================================

**AUDIT RESULT: EXPERIMENTAL WORKFLOW COMPLETE — GITHUB READY WITH MINOR CLEANUP**

**Overall Assessment:**
- ✅ All experiments completed and validated
- ✅ No critical methodological defects detected
- ✅ No accidental use of obsolete Phase 5 results in current analyses
- ✅ Bootstrap implementations correct (instance-level)
- ✅ No test-set leakage detected
- ⚠️ Minor organizational issues (obsolete files present but correctly identified)
- ⚠️ Minor inconsistency in n_instances reporting (cosine metric)

**Recommendation:** Repository is ready for GitHub publication after minor cleanup.

================================================================================
DETAILED FINDINGS BY COMPONENT
================================================================================

## 1. MULTI-SEED TRAINING VERIFICATION

**CLAIM:** 60 model configurations (3 datasets × 4 models × 5 seeds)

**VERIFIED:**
✅ all_metrics.csv contains exactly 60 rows
✅ Seeds: [0, 7, 42, 123, 2024] — matches config.py TRAINING_SEEDS
✅ Datasets: german, gmsc, taiwan — matches config
✅ Models: lightgbm, mlp, random_forest, xgboost — matches config
✅ All 60 unique dataset-model-seed combinations present

**ISSUE CLASSIFICATION:** NO ISSUE

---

## 2. CROSS-SEED STABILITY VERIFICATION

**CLAIM:** 12,000 pairwise rows (24 combos × 50 instances × 10 seed-pairs)

**VERIFIED:**
✅ cross_seed_pairwise.csv: 12,000 rows
✅ 24 unique dataset-model-explainer combinations
✅ 10 unique seed pairs (C(5,2) = 10)
✅ 50 instances per combination
✅ Metrics: spearman, topk_jaccard, cosine, sign_agreement
✅ Aggregate file: 96 rows (24 combos × 4 metrics)
✅ Bootstrap: 5000 reps, seed=42, all CI status='OK'

**COSINE NaN VERIFICATION:**
✅ Exactly 132 NaN values in cosine column
✅ Affecting 5 unique instances: [68, 134, 560, 2685, 2691]
✅ German: 12 NaN (instances 68, 134)
✅ Taiwan: 120 NaN (instances 560, 2685, 2691)
✅ COSINE_NAN_DIAGNOSIS.md correctly identifies instances
✅ Other metrics (spearman, jaccard, sign_agreement): 0 NaN

**MINOR INCONSISTENCY DETECTED:**
⚠️ cross_seed_stability.csv reports n_instances=[50, 47]
⚠️ Taiwan random_forest LIME cosine shows n_instances=47 (not 50)
⚠️ This is because 3 Taiwan instances (560, 2685, 2691) have all-NaN cosine for LIME
⚠️ Bootstrap correctly excludes NaN instances for cosine metric only
⚠️ Other metrics still use all 50 instances

**CLASSIFICATION:** MINOR — n_instances reporting is technically correct (47 non-NaN for cosine) but could be clearer. Not a methodological error.

**VALIDATION:** The audit claim "50 instances per combination" is correct for most metrics but slightly misleading for cosine. More precisely: "50 instances attempted, 47-50 usable depending on metric and combination."

---

## 3. SEMANTIC STABILITY VERIFICATION

**CLAIM:** 24 rows, instance-level bootstrap, 50 instances, 10 seed-pairs

**VERIFIED:**
✅ semantic_stability.csv: 24 rows (3 datasets × 4 models × 2 explainers)
✅ All CI status='OK'
✅ n_instances: all show 50
✅ n_seed_pairs: all show 10
✅ Bootstrap: 5000 reps, seed=42
✅ n_categories: German=4, Taiwan=4, GMSC=5 (matches SEMANTIC_GROUPS in config.py)

**BOOTSTRAP IMPLEMENTATION VERIFICATION:**
✅ stage3_semantic_stability.py line 208: instance_means = pairwise_df.groupby('instance_index')['stability'].mean()
✅ Correctly computes 50 instance-level means from 500 pairwise observations
✅ Then bootstraps from those 50 means WITH REPLACEMENT
✅ NO isin() bug present in active code

**OBSOLETE FILES CONFIRMED:**
✅ semantic_stability_ISIN_BUG.csv exists (buggy version)
✅ semantic_stability_OLD_PAIRWISE_BOOTSTRAP.csv exists (old version)
✅ These are correctly identified as obsolete in audit
✅ Current semantic_stability.csv uses corrected methodology

**ISSUE CLASSIFICATION:** NO ISSUE — Bootstrap implementation is correct. Obsolete files present but not used.

---

## 4. PERTURBATION ROBUSTNESS VERIFICATION

**CLAIM:** 240 perturbation evaluations (60 seed-combos × 4 rates)

**VERIFIED:**
✅ aggregate_summary.csv: 240 rows
✅ Expected: 3 datasets × 4 models × 5 seeds × 4 rates = 240
✅ Perturbation rates: [-0.10, -0.05, +0.05, +0.10]
✅ Seeds: [0, 7, 42, 123, 2024]
✅ frac_pred_preserving mean: 0.473 (range 0.0-0.94)

**PREDICTION-PRESERVING ANALYSIS:**
✅ prediction_preserving_bootstrap.csv: 96 rows
✅ Expected: 12 dataset-model combos × 8 metrics (4 SHAP + 4 LIME) = 96
✅ Metrics: rank_sim, attr_sim, sign_cons, topk_overlap for each explainer

**METRIC INTERPRETATION VERIFICATION:**
✅ All metrics are already stability/similarity metrics (higher = more stable)
✅ NO (1-metric) inversion present in code or results
✅ Audit correctly states metrics are already in correct form

**ISSUE CLASSIFICATION:** NO ISSUE

---

## 5. FAITHFULNESS VERIFICATION

**CLAIM:** 6,000 instance evaluations (24 combos × 5 seeds × 50 instances)

**VERIFIED:**
✅ faithfulness_instance.csv: 6,000 rows
✅ 120 unique dataset-model-explainer-seed combinations (24 × 5)
✅ Seeds: [0, 7, 42, 123, 2024]
✅ GMSC MLP SHAP mean faithfulness_spearman: -0.0762
✅ Negative result preserved (not hidden or "corrected")

**ISSUE CLASSIFICATION:** NO ISSUE — Negative faithfulness correctly preserved as legitimate finding.

---

## 6. ESS VERIFICATION

**CLAIM:** Formula ESS = α × cross_seed_spearman + (1-α) × semantic_stability, α ∈ {0.3, 0.5, 0.7}

**VERIFIED:**
✅ ess_sensitivity.csv: 24 rows
✅ Columns include: cross_seed_spearman, semantic_stability, ess_30_70, ess_50_50, ess_70_30
✅ cross_seed_spearman range: [0.4230, 0.9929] ⊂ [0,1]
✅ semantic_stability range: [0.7261, 0.9890] ⊂ [0,1]
✅ Component correlation: Pearson r=0.9574, Spearman ρ=0.9722 (matches audit claim)

**FORMULA VERIFICATION:**
✅ Manual check: ess_50_50 = 0.5 × cross_seed_spearman + 0.5 × semantic_stability
✅ Example: german/lightgbm/lime: (0.8656+0.9155)/2 = 0.8906 ✓ matches ess_50_50=0.8906

**PHASE 5 ESS CONTAMINATION CHECK:**
✅ NO phase5 files imported in current ESS calculation
✅ audit/reproduce_ess.py references phase5 but is for audit purposes only
✅ Top-level results/stage3/ess_sensitivity.csv is a PLACEHOLDER ("PENDING_STAGE_4")
✅ Actual ESS results are in results/stage3/ess/ subdirectory
✅ No accidental use of old Phase 5 ESS formula

**ISSUE CLASSIFICATION:** NO ISSUE — ESS correctly computed from current Stage 3 components only.

---

## 7. OBSOLETE FILES VERIFICATION

**AUDIT CLAIM:** Multiple obsolete files exist and are correctly identified

**VERIFIED:**
✅ phase1/, phase2/, phase3/, phase4/, phase5/ directories exist (old methodology)
✅ results/phase3/, results/phase4/, results/phase4_v2/, results/phase5/ exist
✅ results/stage3/semantic_stability_ISIN_BUG.csv exists
✅ results/stage3/semantic_stability_OLD_PAIRWISE_BOOTSTRAP.csv exists
✅ results/stage3/seed_stability_pairwise.csv exists (superseded by cross_seed/)
✅ results/stage3/seed_stability_summary.csv exists (superseded)
✅ Top-level results/stage3/ess_sensitivity.csv is placeholder (superseded by ess/)
✅ Top-level results/stage3/ess_ablation.csv is placeholder

**OLD VS NEW FILE COMPARISON:**
✅ seed_stability_pairwise.csv (12,000 rows) has different column names than cross_seed_pairwise.csv
   - Old: rank_sim, attr_sim, sign_cons, topk
   - New: spearman, topk_jaccard, cosine, sign_agreement
✅ These are different methodologies, not duplicates

**ISSUE CLASSIFICATION:** NO ISSUE — Obsolete files correctly identified but not removed. Appropriate for research provenance.

---

## 8. GITHUB ORGANIZATION VERIFICATION

**AUDIT CLAIM:** Repository structure clean, .gitignore incomplete, secrets check passed

**VERIFIED:**
✅ NO .env files detected
✅ NO API keys, passwords, tokens, credentials detected
✅ .gitignore exists but excludes only: __pycache__, .pyc, .venv, .ipynb_checkpoints, data/raw/, models/, outputs/, *.pkl, *.joblib
✅ Many large/obsolete files NOT in .gitignore (correctly identified for addition)

**TEMP FILES PRESENT:**
✅ temp_bootstrap.py, test.txt, validation.txt, verify.txt (correctly identified)
✅ stage3_runner*.py files (temp, correctly identified)
✅ *.log files (correctly identified)

**ISSUE CLASSIFICATION:** MINOR — .gitignore needs updates, temp files need cleanup. No secrets to remove.

================================================================================
CRITICAL ISSUES: 0
================================================================================

NONE DETECTED.

================================================================================
IMPORTANT ISSUES: 0
================================================================================

NONE DETECTED.

================================================================================
MINOR ISSUES: 3
================================================================================

**MINOR-1: n_instances Inconsistency in Cross-Seed Cosine**
- cross_seed_stability.csv reports n_instances=[50, 47]
- Taiwan RF LIME cosine has n_instances=47 due to 3 all-NaN instances
- Other combinations and metrics correctly report n_instances=50
- IMPACT: Technically correct but could confuse readers
- FIX: Add note in documentation explaining variable n_instances for cosine
- BLOCKER: No

**MINOR-2: Top-Level Placeholder Files Misleading**
- results/stage3/ess_sensitivity.csv and ess_ablation.csv are placeholders
- Actual ESS results in results/stage3/ess/ subdirectory
- IMPACT: Could confuse users looking for ESS results
- FIX: Delete placeholders or add README pointing to ess/ directory
- BLOCKER: No

**MINOR-3: Incomplete .gitignore**
- Current .gitignore excludes only 11 patterns
- Audit identifies ~50 additional patterns needed
- IMPACT: Risk of accidentally committing large/temp files
- FIX: Update .gitignore before first git add
- BLOCKER: No (can fix before commit)

================================================================================
ADDITIONAL OBSERVATIONS
================================================================================

**POSITIVE FINDINGS:**

1. **Bootstrap Implementation Correct**
   - stage3_semantic_stability.py uses proper instance-level resampling
   - No isin() bug in active code
   - Duplicate retention verified

2. **No Test-Set Leakage**
   - Separate train/test splits maintained throughout
   - No evidence of information leakage across experiments

3. **Seed Configuration Consistent**
   - Seeds [0, 7, 42, 123, 2024] used consistently across all experiments
   - No accidental mixing of old/new seeds

4. **Negative Results Preserved**
   - GMSC MLP SHAP negative faithfulness (-0.076) correctly retained
   - No evidence of p-hacking or result suppression

5. **Provenance Maintained**
   - Obsolete files retained (not deleted)
   - Audit trails preserved (COSINE_NAN_DIAGNOSIS.md, etc.)

**POTENTIAL IMPROVEMENTS (NOT BLOCKERS):**

1. Add README.md explaining dataset-model-explainer combinations
2. Add note about variable n_instances for cosine metric
3. Delete or clearly label placeholder files
4. Move obsolete files to archive/ directory
5. Update .gitignore before GitHub push

================================================================================
FINAL VERDICT
================================================================================

**EXPERIMENTAL WORKFLOW STATUS:**
✅ COMPLETE WITH DOCUMENTED LIMITATIONS

All experiments finished:
- ✅ 60 multi-seed models trained
- ✅ 12,000 cross-seed comparisons computed
- ✅ 24 semantic stability analyses completed
- ✅ 240 perturbation evaluations performed
- ✅ 6,000 faithfulness instances evaluated
- ✅ 24 ESS sensitivity analyses computed

All validations passed:
- ✅ No methodological defects
- ✅ No test-set leakage
- ✅ No Phase 5 contamination
- ✅ Bootstrap implementations correct
- ✅ Negative results preserved

Known limitations documented:
- ✅ Cosine NaN explained (5 instances)
- ✅ ESS component correlation acknowledged (r=0.96)
- ✅ Negative faithfulness explained
- ✅ Variable prediction-preserving coverage noted

**GITHUB PUBLICATION READINESS:**
✅ READY WITH MINOR CLEANUP

Safe to publish:
- ✅ No secrets detected
- ✅ No sensitive data
- ✅ Clean code structure
- ✅ Lightweight aggregate results (~500KB)

Required before push:
- ⚠️ Update .gitignore (add large/temp files)
- ⚠️ Delete or archive temp files (temp_bootstrap.py, test.txt, etc.)
- ⚠️ Delete or clarify placeholder files (top-level ess_*.csv)
- ⚠️ Add README.md (recommended but not blocking)

**RECOMMENDATION:**
Proceed with GitHub publication after:
1. Update .gitignore
2. Remove temp files (test.txt, temp_bootstrap.py, *.log)
3. Add README.md (optional but recommended)

Repository contains valid, complete, reproducible research.

================================================================================
AUDIT COMPLETE
================================================================================

Auditor: Claude (Independent)
Date: 2026-09-25
Audit Duration: Comprehensive systematic verification
Result: APPROVED FOR PUBLICATION
