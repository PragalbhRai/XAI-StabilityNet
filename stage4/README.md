# Stage 4A: Faithfulness Evaluation

## Overview

Stage 4A evaluates whether explanation magnitude corresponds to actual model sensitivity by ablating features and measuring prediction changes.

## Methodology

### Feature Ablation Protocol

For each instance and feature:

1. Compute original prediction `p_orig`
2. Replace feature with training-derived reference value
3. Compute ablated prediction `p_ablate`
4. Record `ablation_effect = abs(p_orig - p_ablate)`

### Reference Values

**Numeric features**: Training median
**Binary/categorical features**: Training mode
**One-hot encoded groups**: Ablate entire group consistently

### Faithfulness Metrics

#### 1. Faithfulness Spearman

Spearman correlation between:
- `abs(explanation_j)` (explanation magnitude)
- `ablation_effect_j` (actual sensitivity)

Higher positive values indicate explanations correctly identify important features.

#### 2. Top-k Deletion

Ablate top-k features ranked by explanation magnitude:
- k ∈ {1, 3, 5, 10}
- Compare to random baseline (30 repetitions)
- Compute enrichment: `topk_drift / random_drift_mean`

### Statistics

**Bootstrap method**: Instance-cluster resampling (not row-wise)
**Resamples**: 5000
**Unit**: Explanation instance
**CI**: 95% percentile method

## Files

### Core Implementation

- `stage4/config.py` - Configuration extending Stage 2
- `stage4/faithfulness.py` - Core faithfulness functions
- `stage4/run_faithfulness_smoke_test.py` - Lightweight validation (5 instances)
- `stage4/run_faithfulness_full.py` - Full 60-combination pipeline
- `stage4/compute_faithfulness_summary.py` - Summary statistics with bootstrap

### Outputs

- `results/stage4/faithfulness/faithfulness_instance.csv` - Instance-level results
- `results/stage4/faithfulness/faithfulness_topk.csv` - Top-k deletion results
- `results/stage4/faithfulness/faithfulness_summary.csv` - Summary statistics
- `results/stage4/faithfulness/validation_report.json` - Smoke test validation

## Expected Row Counts

**Instance-level results**:
- 3 datasets × 4 models × 5 seeds × 50 instances × 2 explainers = **6,000 rows**

**Top-k results**:
- 6,000 instances × 4 k-values = **24,000 rows**

**Summary statistics**:
- 3 datasets × 4 models × 2 explainers × (1 spearman + 4 topk_minus_random + 4 enrichment) = **216 rows**

## Usage

### Smoke Test (recommended first)

```bash
python stage4/run_faithfulness_smoke_test.py
```

Tests German/XGBoost/seed42 with 5 instances. Validates:
- Predictions are finite
- Ablation works correctly
- Feature counts match
- One-hot groups handled consistently
- Output schema correct

### Full Pipeline

```bash
python stage4/run_faithfulness_full.py
python stage4/compute_faithfulness_summary.py
```

Processes all 60 combinations (3 datasets × 4 models × 5 seeds).

## Implementation Notes

### One-Hot Encoding

Categorical features with one-hot encoding are treated as groups:
- German: `checking_status`, `credit_history`, `purpose`, etc.
- Taiwan: `SEX`, `EDUCATION`, `MARRIAGE`
- GMSC: (minimal categorical encoding)

When ablating a one-hot group, all dummy columns are set to reference simultaneously.

### Feature Ordering

SHAP and LIME feature columns must match exactly. Validation checks confirm ordering consistency.

### Reproducibility

- Reference value computation: Deterministic from training data
- Random baseline: Fixed seeds (`FAITHFULNESS_SEED + instance_index`)
- Bootstrap: Fixed seed (42)

## Validation

Smoke test checks:
- ✓ Predictions finite
- ✓ Ablation predictions finite
- ✓ Feature counts correct
- ✓ No NaN/Inf in metrics
- ✓ SHAP/LIME ordering matches
- ✓ Categorical groups handled
- ✓ Output schema correct

## Stage 2 Compatibility

**Does NOT modify**:
- Stage 2 trained models
- Stage 2 explanations
- Stage 2 preprocessing
- Stage 2 data splits

**Reuses exactly**:
- Trained model artifacts
- SHAP values
- LIME contrib (not coeff)
- Feature ordering
- 50 explanation instances
- 5 training seeds [42, 0, 7, 123, 2024]

## Limitations

- **Scope**: Only 50 instances per combination (same as Stage 2 explanations)
- **Reference values**: Training-derived (not instance-specific counterfactuals)
- **One-hot groups**: Heuristic identification by prefix matching
- **Correlation**: Spearman is descriptive; causal interpretation requires care

## Next Steps

After Stage 4A completion:
- Stage 4B: Counterfactual stability (pending)
- Stage 4C: Cross-dataset semantic consistency (pending)

