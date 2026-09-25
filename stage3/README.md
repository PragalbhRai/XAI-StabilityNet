# Stage 3: Statistical Analysis

## Overview

Stage 3 performs statistical analysis on the **locked outputs** from Stage 2. It quantifies:

1. **Training-seed stability**: How consistent are explanations across different random initializations?
2. **SHAP-LIME consistency**: How well do SHAP and LIME agree?
3. **LIME stochasticity**: How stable is LIME across repeated invocations?
4. **Prediction-preserving explanation drift**: How much do explanations change when predictions stay constant?
5. **Perturbation rate effects**: Does perturbation magnitude affect drift?
6. **Prediction-explanation association**: Are prediction changes correlated with explanation changes?
7. **Plausibility distributions**: How many instances meet plausibility criteria?

**All analyses use locked Stage 2 data. Stage 2 outputs are never modified.**

## Stage 2 Design (Verified)

Stage 3 relies on the following **verified and locked** Stage 2 outputs:

### Experimental Grid
- **Datasets**: german (n=50), taiwan (n=50), gmsc (n=50)
- **Models**: xgboost, lightgbm, random_forest, mlp
- **Training seeds**: [42, 0, 7, 123, 2024] (5 seeds per model)
- **Total combinations**: 3 datasets × 4 models × 5 seeds = 60 combinations

### Data Sources
1. **Robustness data** (`results/stage2/robustness/{dataset}_{model}_seed{seed}_robustness.parquet`):
   - 60 files (one per combination)
   - 12,000 total rows (200 per file)
   - Schema: 21 columns including dataset, model, training_seed, instance_index, perturbation_rate, original_pred, perturbed_pred, drift metrics (prediction, SHAP, LIME)
   - 50 instances × 4 perturbation rates (-0.10, -0.05, 0.05, 0.10) per combination

2. **Explanation data** (`results/stage2/explanations/{dataset}_{model}_seed{seed}_shap.parquet`, `*_lime_contrib.parquet`):
   - 120 files (60 SHAP + 60 LIME)
   - Schema: instance_index + feature columns
   - Used for SHAP-vs-LIME comparisons

3. **LIME stochasticity** (`results/stage2/lime_stochasticity/german_xgboost_stoch.parquet`):
   - 1 file (German + XGBoost + seed 42)
   - 600 rows (20 instances × 30 LIME seeds)
   - Schema: dataset, model, instance_i, lime_seed, feature_columns
   - All pairwise LIME seed comparisons: 20 instances × C(30,2) = 8,700 comparisons

### Metrics (Reused from Stage 2)
All metrics use the exact implementations from `stage2/robustness.py` lines 127-170:
- **rank_similarity**: Spearman rank correlation between feature importance rankings
- **attribution_similarity**: Cosine similarity between raw attribution vectors
- **sign_consistency**: Fraction of features with matching signs (positive/negative)
- **topk_overlap**: Jaccard overlap of top-k most important features (k=5)

## Statistical Units and Hierarchy

Stage 3 respects the hierarchical structure of the experiment:

```
Dataset (german, taiwan, gmsc)
└── Model (xgboost, lightgbm, random_forest, mlp)
    └── Training Seed (42, 0, 7, 123, 2024)
        └── Instance (50 instances per combination)
            └── Perturbation Rate (-0.10, -0.05, 0.05, 0.10)
```

### Key Concepts
- **Instance-level analysis**: Metrics computed per instance, then aggregated
- **Seed-level comparisons**: Pairwise comparisons between training seeds (10 pairs per dataset×model×explainer)
- **Dataset/Model grouping**: Summary statistics grouped by dataset, model, explainer where appropriate

## Bootstrap Confidence Intervals

Many Stage 3 analyses report **95% bootstrap confidence intervals** to account for dependency across observations:

### Why Bootstrap?
- Instances appear across multiple training seeds, perturbation rates, and explainer comparisons
- Standard parametric CIs assume independence, which is violated
- Bootstrap resampling respects the hierarchical structure

### Method
- **Resampling unit**: Observation (row-level, not instance-level clustering)
- **Replicates**: 5,000 bootstrap samples
- **Random seed**: 42 (for reproducibility)
- **CI method**: Percentile (2.5th and 97.5th percentiles)

### Confidence Interval Status
All summary tables include a **ci_status** column indicating confidence interval estimability:

- **OK**: Bootstrap CI successfully estimated (n ≥ 2)
- **NOT_ESTIMABLE_N_LT_2**: CI cannot be estimated due to insufficient observations (n < 2)

When ci_status = NOT_ESTIMABLE_N_LT_2, ci_low and ci_high are NaN. This is a legitimate methodological constraint, not a data quality issue.

### When Bootstrap CIs Are Used
- Seed stability summary statistics (mean, median across seeds)
- SHAP-LIME consistency summary statistics
- Prediction-preserving drift summary statistics (all subsets)

### Edge Cases
- Bootstrap CIs are NOT_ESTIMABLE_N_LT_2 when n < 2
- Example: gmsc/mlp/pert_rate=0.05 in prediction-preserving subset has only 1 row (14 such cases)

## Analyses

### 1. Seed Stability
**Files**: `seed_stability_pairwise.csv` (12,000 rows), `seed_stability_summary.csv` (96 rows)

**Question**: Are explanations consistent across different training seeds?

**Method**:
- For each dataset×model×explainer, compute pairwise metric similarities between all seed pairs
- 5 seeds → C(5,2) = 10 pairwise comparisons per group
- Metrics: rank_sim, attr_sim, sign_cons, topk
- Pairwise file: 3 datasets × 4 models × 2 explainers × 10 pairs × 50 instances = 12,000 rows
- Summary file: mean ± std, median, IQR, 95% CI (with ci_status) for each dataset×model×explainer×metric

**Interpretation**:
- High values (close to 1.0) → stable explanations across seeds
- Low values → seed choice significantly affects explanations

**Verified Structure**:
- SHAP: 6,000 rows (3 × 4 × 50 × 10)
- LIME: 6,000 rows (3 × 4 × 50 × 10)

### 2. SHAP-LIME Consistency
**Files**: `shap_lime_consistency.csv` (3,000 rows), `shap_lime_consistency_summary.csv` (48 rows)

**Question**: Do SHAP and LIME produce similar explanations?

**Method**:
- For each dataset×model×training_seed, compare SHAP and LIME explanations for all 50 instances
- Metrics: rank_sim, attr_sim, sign_cons, topk
- Instance file: 3 datasets × 4 models × 5 seeds × 50 instances = 3,000 rows
- Summary file: mean ± std, median, IQR, 95% CI (with ci_status) grouped by dataset×model×metric

**Interpretation**:
- High consistency → SHAP and LIME converge on similar explanations
- Low consistency → method choice matters significantly

**Note**: Uses `lime_contrib.parquet` (not `lime_coeff.parquet`) per user specification

### 3. LIME Stochasticity
**File**: `lime_stochasticity_analysis.csv` (8,700 rows)

**Question**: Is LIME stable across repeated invocations?

**Method**:
- Reads `lime_stochasticity/german_xgboost_stoch.parquet` (20 instances × 30 LIME seeds = 600 rows)
- Computes ALL pairwise LIME seed comparisons: 20 instances × C(30,2) = 20 × 435 = 8,700 comparisons
- Metrics: rank_sim, attr_sim, sign_cons, topk
- No arbitrary sampling or subsampling

**Interpretation**:
- High similarity metrics → LIME is stable across repeated invocations
- Low similarity → LIME results are stochastic and unreliable

**Verified Structure**:
- Stage 2 source: 600 rows (20 instances, 30 seeds each)
- Stage 3 output: 8,700 pairwise comparisons

**Limitation**: Only German + XGBoost analyzed

### 4. Prediction-Preserving Explanation Drift
**File**: `prediction_preserving_analysis.csv` (1,344 rows)

**Question**: How much do explanations change when predictions stay constant?

**Method**:
- Filter robustness data into 4 subsets:
  1. **all**: All 12,000 rows (no filtering)
  2. **pred_pres**: Prediction-preserving only (original_pred == perturbed_pred)
  3. **plausible**: Plausible instances only
  4. **primary**: Prediction-preserving AND plausible (THE PRIMARY ANALYSIS)
- Report metrics grouped by subset, dataset, model, perturbation_rate, metric
- Include: prediction_drift, shap_rank_sim, shap_attr_sim, shap_sign_cons, lime_rank_sim, lime_attr_sim, lime_sign_cons
- Bootstrap CIs with ci_status column

**Interpretation**:
- In prediction-preserving cases, drift quantifies explanation instability independent of prediction changes
- Primary subset is most reliable (plausibility ensures instance is well-behaved)
- 14 rows have ci_status=NOT_ESTIMABLE_N_LT_2 (single-instance strata, no CI possible)

### 5. Perturbation Rate Effects
**File**: `perturbation_rate_analysis.csv` (36 rows)

**Question**: Does perturbation magnitude affect drift metrics?

**Method**:
- Group by dataset, model, metric
- Run Friedman test (non-parametric repeated measures ANOVA) across 4 perturbation rates
- Metrics: prediction_drift, shap_rank_sim, lime_rank_sim
- Report χ² statistic, p-value, n_per_rate

**Interpretation**:
- Significant p-value (p < 0.05) → perturbation rate affects drift
- Non-significant → drift is independent of perturbation magnitude

### 6. Prediction-Explanation Association
**File**: `prediction_explanation_association.csv` (48 rows)

**Question**: Are prediction changes correlated with explanation changes?

**Method**:
- For each dataset×model×explainer×metric, compute Spearman correlation between:
  - prediction_drift vs. (1 - explanation_similarity)
- Explainers: SHAP, LIME
- Metrics: rank_sim, attr_sim
- Report rho, p-value, n

**Interpretation**:
- Positive correlation → explanations track predictions (desirable)
- No correlation → explanations drift independently of predictions (concerning)

### 7. Plausibility Distributions
**File**: `plausibility_analysis.csv` (3 rows)

**Question**: How many instances meet plausibility criteria?

**Method**:
- For each dataset, compute plausibility statistics based on plausibility_max_z scores
- Report: n, mean, median, std, quantiles, pass rates, z-score thresholds

**Interpretation**:
- Plausibility pass rate indicates fraction of well-behaved instances
- Higher z-scores indicate greater sensitivity to perturbations

**Caveat**: Plausibility is a dataset-level property defined by Stage 2, not a ground truth

## Output Files

All outputs are in `results/stage3/`:

| File | Rows | Description |
|------|------|-------------|
| `seed_stability_pairwise.csv` | 12,000 | All pairwise seed comparisons (6000 SHAP + 6000 LIME) |
| `seed_stability_summary.csv` | 96 | Summary statistics by dataset×model×explainer×metric |
| `shap_lime_consistency.csv` | 3,000 | Instance-level SHAP-LIME comparisons |
| `shap_lime_consistency_summary.csv` | 48 | Summary statistics by dataset×model×metric |
| `lime_stochasticity_analysis.csv` | 8,700 | All pairwise LIME seed comparisons (20 inst × C(30,2)) |
| `prediction_preserving_analysis.csv` | 1,344 | Drift metrics for all 4 subsets with ci_status |
| `perturbation_rate_analysis.csv` | 36 | Friedman tests for perturbation rate effects |
| `prediction_explanation_association.csv` | 48 | Spearman correlations (prediction vs explanation drift) |
| `plausibility_analysis.csv` | 3 | Plausibility distributions by dataset |
| `ess_sensitivity.csv` | 1 | ESS placeholder (status=PENDING_STAGE_4) |
| `ess_ablation.csv` | 1 | ESS ablation placeholder (status=PENDING_STAGE_4) |
| `stage3_summary.json` | - | Completion status |

## Components Marked PENDING_STAGE_4

Stage 3 does NOT implement:
- **ESS (Explanation Shift Score)**: Marked as SECONDARY, requires semantic category validation
- **CSS (Counterfactual Stability Score)**: Marked as PENDING_STAGE_4, requires counterfactual generation

Placeholder files (ess_sensitivity.csv, ess_ablation.csv) explicitly document:
- status = PENDING_STAGE_4
- reason = semantic category validation required

These will be addressed in Stage 4 (counterfactual generation and semantic drift analysis).

## Validation

All output files have been validated:
- **Row counts** match theoretical predictions
- **NaN values** only in ci_low/ci_high when ci_status=NOT_ESTIMABLE_N_LT_2 (expected)
- **No infinite values**
- **Required columns** present with ci_status column documenting estimability

## Single-Instance Limitations

When n < 2, confidence intervals cannot be estimated. This is explicitly documented:

- **ci_status = NOT_ESTIMABLE_N_LT_2**: Appears in 14 rows of prediction_preserving_analysis.csv
- **Not a data quality issue**: Legitimate statistical constraint for strata with insufficient observations
- **Example**: gmsc/mlp/pert_rate=0.05 in pred_pres and primary subsets has only 1 instance

Do not interpret ci_status=NOT_ESTIMABLE_N_LT_2 as evidence or a defect. It is a methodological marker.

## Reproducibility

To regenerate Stage 3 outputs:
```bash
python stage3_runner_fixed.py
```

**Warning**: This will overwrite existing files in `results/stage3/`.

## Dependencies

- pandas
- numpy
- scipy (for Spearman, Friedman tests)
- Stage 2 outputs (must exist and be locked)

## Feature Counts (Context)
- German: 48 features
- Taiwan: 23 features
- GMSC: 10 features

## References

- Stage 2 design: `results/stage2/robustness/`, `results/stage2/explanations/`, `results/stage2/lime_stochasticity/`
- Metric implementations: `stage2/robustness.py` lines 127-170
- Locked data verified: 60 robustness files, 12,000 rows, 0 modifications
