================================================================================
XAI-STABILITYNET MULTI-SEED TRAINING STATUS REPORT
================================================================================
Report Date: 2026-09-25
Auditor: Kiro AI
Scope: Determine existing multi-seed training status (AUDIT ONLY)

================================================================================
A. EXISTING MULTI-SEED STATUS
================================================================================

| Dataset | Model         | Training Seeds Found      | Independently Retrained? | Evidence                          |
|---------|---------------|---------------------------|--------------------------|-----------------------------------|
| german  | xgboost       | [0, 7, 42, 123, 2024]     | YES                      | 5 .joblib files + AUC variance    |
| german  | lightgbm      | [0, 7, 42, 123, 2024]     | YES                      | 5 .joblib files + AUC variance    |
| german  | random_forest | [0, 7, 42, 123, 2024]     | YES                      | 5 .joblib files + AUC variance    |
| german  | mlp           | [0, 7, 42, 123, 2024]     | YES                      | 5 .joblib + 5 scaler files + AUC variance |
| taiwan  | xgboost       | [0, 7, 42, 123, 2024]     | YES                      | 5 .joblib files + AUC variance    |
| taiwan  | lightgbm      | [0, 7, 42, 123, 2024]     | YES                      | 5 .joblib files + AUC variance    |
| taiwan  | random_forest | [0, 7, 42, 123, 2024]     | YES                      | 5 .joblib files + AUC variance    |
| taiwan  | mlp           | [0, 7, 42, 123, 2024]     | YES                      | 5 .joblib + 5 scaler files + AUC variance |
| gmsc    | xgboost       | [0, 7, 42, 123, 2024]     | YES                      | 5 .joblib files + AUC variance    |
| gmsc    | lightgbm      | [0, 7, 42, 123, 2024]     | YES                      | 5 .joblib files + AUC variance    |
| gmsc    | random_forest | [0, 7, 42, 123, 2024]     | YES                      | 5 .joblib files + AUC variance    |
| gmsc    | mlp           | [0, 7, 42, 123, 2024]     | YES                      | 5 .joblib + 5 scaler files + AUC variance |

**SUMMARY:**
- Total combinations: 12 (3 datasets × 4 models)
- All combinations have 5 independent training seeds
- Total model checkpoints: 60 models + 15 MLP scalers = 75 files
- All files exist in results/stage2/models/

**EVIDENCE OF INDEPENDENT TRAINING:**

1. **Model Checkpoints:** 
   - 60 distinct .joblib files with naming pattern {dataset}_{model}_seed{seed}.joblib
   - 15 scaler files for MLP with pattern {dataset}_mlp_scaler_seed{seed}.joblib

2. **AUC Variance Across Seeds (confirming independent initialization):**
   
   GERMAN:
   - xgboost:       mean=0.7920, std=0.0043, range=0.0101
   - lightgbm:      mean=0.7706, std=0.0056, range=0.0143
   - random_forest: mean=0.7961, std=0.0042, range=0.0104
   - mlp:           mean=0.7382, std=0.0458, range=0.1126 ✓ HIGH VARIANCE
   
   TAIWAN:
   - xgboost:       mean=0.7748, std=0.0012, range=0.0032
   - lightgbm:      mean=0.7732, std=0.0007, range=0.0017
   - random_forest: mean=0.7683, std=0.0004, range=0.0012
   - mlp:           mean=0.7597, std=0.0057, range=0.0124 ✓ HIGHER VARIANCE
   
   GMSC:
   - xgboost:       mean=0.8643, std=0.0002, range=0.0006
   - lightgbm:      mean=0.8644, std=0.0002, range=0.0004
   - random_forest: mean=0.8601, std=0.0001, range=0.0003
   - mlp:           mean=0.8335, std=0.0018, range=0.0048 ✓ HIGHER VARIANCE

   **INTERPRETATION:** 
   - Tree models show low variance (stable algorithms)
   - MLP shows higher variance (sensitive to initialization)
   - All models show measurable variance → confirming independent training

3. **Seed Implementation in Code:**
   - Location: stage2/train.py::_build_models()
   - All 4 model classes initialized with random_state=seed parameter
   - XGBoost: XGBClassifier(random_state=seed, ...)
   - LightGBM: LGBMClassifier(random_state=seed, ...)
   - RandomForest: RandomForestClassifier(random_state=seed, ...)
   - MLP: MLPClassifier(random_state=seed, ...)

4. **Preprocessing Consistency:**
   - MLP scaling: StandardScaler fitted on training data per seed
   - Scalers saved separately: {dataset}_mlp_scaler_seed{seed}.joblib
   - Train/test split: FIXED (not seed-dependent)
   - Test data: UNTOUCHED during training

5. **Performance Metrics:**
   - results/stage2/models/all_metrics.csv contains 60 rows
   - Each row: (dataset, model, seed, auc, accuracy, f1)
   - All metrics are seed-specific and independently computed

================================================================================
B. WHAT THE EXISTING SEEDS ACTUALLY REPRESENT
================================================================================

**SEEDS IN THE PROJECT:**

1. **MODEL TRAINING SEEDS (TRAINING_SEEDS = [42, 0, 7, 123, 2024])**
   - Purpose: Initialize model random state (tree splits, MLP weights, etc.)
   - Location: stage2/config.py::TRAINING_SEEDS
   - Usage: stage2/train.py::_build_models(seed)
   - Effect: Independent model training runs with different initializations
   - Artifacts: 60 model checkpoints in results/stage2/models/
   - **STATUS: GENUINE MULTI-SEED TRAINING ✓**

2. **EXPLANATION INSTANCE SELECTION SEED (EXPLAIN_SEED = 42)**
   - Purpose: Select SAME 50 test instances across all models/seeds
   - Location: stage2/config.py::EXPLAIN_SEED
   - Usage: stage2/explain.py::select_instances()
   - Effect: Fixed instance selection for reproducibility
   - Artifacts: {dataset}_instances.parquet (one per dataset)
   - **STATUS: Fixed seed for controlled comparison**

3. **SHAP BACKGROUND SELECTION SEED (EXPLAIN_SEED = 42)**
   - Purpose: Select SAME background sample for KernelSHAP across all seeds
   - Location: stage2/explain.py::select_background()
   - Effect: Consistent background for SHAP approximation
   - **STATUS: Fixed seed for controlled comparison**

4. **LIME BASE SEED (EXPLAIN_SEED = 42 + perturbations)**
   - Purpose: Seed LIME neighbourhood generation
   - Location: stage2/explain.py
   - Effect: Reproducible LIME explanations per (model_seed, instance)
   - **STATUS: Derived from EXPLAIN_SEED**

5. **FAITHFULNESS RANDOM BASELINE SEED (FAITHFULNESS_SEED = 42 + instance_index)**
   - Purpose: Generate random feature ablations for baseline comparison
   - Location: stage4/config.py::FAITHFULNESS_SEED
   - Usage: stage4/faithfulness.py::topk_deletion_analysis_batched()
   - Effect: Reproducible random baselines (30 repetitions per instance)
   - **STATUS: Per-instance RNG for random feature selection**

6. **BOOTSTRAP SEED (seed = 42)**
   - Purpose: Resample instances for bootstrap confidence intervals
   - Location: stage4/compute_faithfulness_summary.py::bootstrap_instance_ci()
   - Effect: Reproducible 95% CIs (5000 replicates)
   - **STATUS: Fixed seed for statistical reproducibility**

**CRITICAL DISTINCTION:**

✓ **Model training seeds [0, 7, 42, 123, 2024]** → VARY across runs → produce DIFFERENT models
✓ **Explanation/evaluation seeds** → FIXED → enable CONTROLLED COMPARISON of different models

This design is CORRECT for multi-seed stability analysis:
- Models trained with different random initializations
- Evaluated on SAME instances with SAME explanation protocols
- Differences in results attributable to MODEL VARIABILITY, not evaluation artifacts

================================================================================
C. GAPS
================================================================================

**STATUS: NO GAPS DETECTED**

✓ All 3 datasets: german, taiwan, gmsc
✓ All 4 models: xgboost, lightgbm, random_forest, mlp
✓ All 5 training seeds: [0, 7, 42, 123, 2024]
✓ All 60 model checkpoints exist
✓ All 60 combinations have explanations (SHAP + LIME)
✓ All 60 combinations have faithfulness evaluations
✓ All scalers for MLP exist (15 files)
✓ Seed-specific performance metrics recorded

**COMPLETENESS:**
- Model training: 60/60 combinations complete
- Explanations: 60 × 2 explainers = 120/120 complete
- Faithfulness: 24 combos × 5 seeds × 50 instances = 6,000/6,000 instance rows
- Performance metrics: 60/60 rows in all_metrics.csv

**NO ADDITIONAL TRAINING REQUIRED**

================================================================================
D. METHODOLOGY IMPACT
================================================================================

**QUESTION:** Is the current published/planned methodology already satisfied?

**ANSWER:** YES — FULLY SATISFIED

**METHODOLOGY REQUIREMENTS:**

1. ✓ **Multi-seed model training:** 5 independent training seeds per model/dataset
2. ✓ **Fixed evaluation instances:** Same 50 test instances across all seeds
3. ✓ **Seed-specific explanations:** SHAP/LIME computed per (model_seed, instance)
4. ✓ **Seed-specific faithfulness:** Ablation-based evaluation per model_seed
5. ✓ **Statistical aggregation:** Bootstrap CIs over instance clusters
6. ✓ **Performance variance:** AUC/accuracy/F1 per seed recorded

**EXPERIMENTAL DESIGN CONFIRMED:**

The existing implementation follows best practices for XAI stability research:

- **Stability Measurement:** Variation in explanations/faithfulness ACROSS training seeds
- **Controlled Comparison:** SAME instances + SAME evaluation protocol
- **Statistical Rigor:** Bootstrap CIs + 50 instances × 5 seeds = 250 observations per combo
- **Reproducibility:** All seeds fixed and documented

**PAPER CLAIMS SUPPORTED:**

✓ "We trained each model with 5 different random seeds..." — VERIFIED
✓ "...to assess explanation stability under model initialization variance..." — VERIFIED
✓ "Evaluation on the same 50 test instances..." — VERIFIED
✓ "Faithfulness measured via feature ablation..." — VERIFIED
✓ "95% confidence intervals via bootstrap resampling..." — VERIFIED

================================================================================
E. FINAL VERDICT
================================================================================

**MULTI-SEED TRAINING ALREADY COMPLETE ✓**

**SUMMARY:**

- ✅ All 60 model combinations trained with 5 independent seeds
- ✅ All model checkpoints exist and verified
- ✅ Seed implementation correct (random_state parameter passed)
- ✅ AUC variance confirms independent initialization
- ✅ Preprocessing (MLP scaling) refit per seed
- ✅ Test data untouched
- ✅ Explanations generated per model_seed
- ✅ Faithfulness evaluated per model_seed
- ✅ Performance metrics seed-specific

**NO NEW MULTI-SEED TRAINING EXPERIMENT IS REQUIRED.**

**PROCEED TO:**

1. Analysis of existing Stage 2 + Stage 4 results
2. Cross-seed stability quantification
3. Faithfulness variation analysis
4. Publication-ready figure generation
5. Paper writing with existing data

**PROJECT STATUS:**

✓ Stage 1: Data preprocessing (LOCKED)
✓ Stage 2A: Multi-seed model training (COMPLETE — 60 models)
✓ Stage 2B: Multi-seed explanation generation (COMPLETE — 120 explainer runs)
✓ Stage 2C: LIME stochasticity analysis (COMPLETE — German/XGBoost)
✓ Stage 2D: Perturbation robustness (COMPLETE — prediction-preserving drift)
✓ Stage 4A: Faithfulness evaluation (COMPLETE + AUDITED — 6,000 instances)
→ Stage 5: Cross-analysis and publication (READY TO BEGIN)

**EXPERIMENTAL PIPELINE: 100% COMPLETE**

================================================================================
