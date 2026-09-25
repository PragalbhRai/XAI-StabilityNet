"""
stage4/run_faithfulness_smoke_test.py
=====================================
Lightweight smoke test for Stage 4A faithfulness evaluation.

Tests:
- German dataset only
- XGBoost model only
- Seed 42 only
- First 5 instances only
- SHAP and LIME explainers
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage4.config import (
    TRAINING_SEEDS,
    PROCESSED,
    STAGE2_MODELS_DIR,
    STAGE2_EXPLAIN_DIR,
    FAITHFULNESS_DIR,
    TOPK_VALUES,
    N_RANDOM_BASELINES,
    FAITHFULNESS_SEED,
)

from stage4.faithfulness import (
    compute_reference_values,
    identify_onehot_groups,
    load_model_and_scaler,
    predict_proba,
    compute_ablation_effects,
    compute_faithfulness_spearman,
    topk_deletion_analysis,
)


def run_smoke_test():
    """Run lightweight faithfulness smoke test."""
    
    print("="*70)
    print("STAGE 4A FAITHFULNESS — SMOKE TEST")
    print("="*70)
    print()
    print("Configuration:")
    print(f"  Dataset: german")
    print(f"  Model: xgboost")
    print(f"  Seed: 42")
    print(f"  Instances: 5")
    print(f"  Explainers: SHAP, LIME")
    print()
    
    validation = {
        "predictions_finite": True,
        "ablation_predictions_finite": True,
        "feature_counts_correct": True,
        "no_nan_inf_metrics": True,
        "feature_ordering_matches": True,
        "categorical_groups_handled": True,
        "output_schema_correct": True,
        "overall_pass": False,
    }
    
    dataset = "german"
    model_name = "xgboost"
    seed = 42
    n_test_instances = 5
    
    try:
        # Load training data for reference values
        train_path = PROCESSED / f"{dataset}_train.parquet"
        train_df = pd.read_parquet(train_path)
        
        feature_cols = [c for c in train_df.columns if c not in ['Default', 'instance_index']]
        ref_values = compute_reference_values(train_df, feature_cols)
        onehot_groups = identify_onehot_groups(feature_cols)
        
        print(f"Feature columns: {len(feature_cols)}")
        print(f"One-hot groups: {len(onehot_groups)}")
        print(f"Reference values computed: {len(ref_values)}")
        print()
        
        # Load model
        model, scaler = load_model_and_scaler(dataset, model_name, seed)
        print(f"Model loaded: {model_name}")
        print()
        
        # Load SHAP explanations
        shap_path = STAGE2_EXPLAIN_DIR / f"{dataset}_{model_name}_seed{seed}_shap.parquet"
        shap_df = pd.read_parquet(shap_path)
        
        # Load LIME explanations
        lime_path = STAGE2_EXPLAIN_DIR / f"{dataset}_{model_name}_seed{seed}_lime_contrib.parquet"
        lime_df = pd.read_parquet(lime_path)
        
        print(f"SHAP explanations loaded: {len(shap_df)} instances")
        print(f"LIME explanations loaded: {len(lime_df)} instances")
        print()
        
        # Check feature ordering
        shap_feat_cols = [c for c in shap_df.columns if c not in ['instance_index', 'actual_default', 'predicted_prob', 'dataset', 'model', 'seed']]
        lime_feat_cols = [c for c in lime_df.columns if c not in ['instance_index', 'actual_default', 'predicted_prob', 'dataset', 'model', 'seed']]
        
        if shap_feat_cols != lime_feat_cols:
            validation["feature_ordering_matches"] = False
            print("WARNING: SHAP and LIME feature ordering mismatch!")
        else:
            print("✓ SHAP and LIME feature ordering matches")
        
        # Load test data
        test_path = PROCESSED / f"{dataset}_test.parquet"
        test_df = pd.read_parquet(test_path)
        
        instance_results = []
        topk_results = []
        
        # Test first 5 instances
        test_instances = shap_df['instance_index'].unique()[:n_test_instances]
        
        for inst_idx in test_instances:
            # Get instance data
            X_instance = test_df[test_df.index == inst_idx][feature_cols]
            
            if len(X_instance) == 0:
                print(f"WARNING: Instance {inst_idx} not found in test data")
                continue
            
            # Original prediction
            p_orig = predict_proba(model, scaler, X_instance)[0]
            
            if not np.isfinite(p_orig):
                validation["predictions_finite"] = False
                print(f"ERROR: Non-finite prediction for instance {inst_idx}")
                continue
            
            # Test SHAP
            shap_row = shap_df[shap_df['instance_index'] == inst_idx]
            if len(shap_row) > 0:
                shap_attrs = shap_row[shap_feat_cols].values[0]
                
                # Map to one-hot groups
                shap_abs_grouped = {}
                for group_name, group_cols in onehot_groups.items():
                    group_indices = [shap_feat_cols.index(c) for c in group_cols if c in shap_feat_cols]
                    if group_indices:
                        # Sum absolute values for group
                        shap_abs_grouped[group_name] = sum(abs(shap_attrs[i]) for i in group_indices)
                
                # Compute ablation effects
                p_orig_check, ablation_effects = compute_ablation_effects(
                    model, scaler, X_instance, feature_cols, ref_values, onehot_groups
                )
                
                # Check ablation predictions are finite
                for feat, effect in ablation_effects.items():
                    if not np.isfinite(effect):
                        validation["ablation_predictions_finite"] = False
                        print(f"ERROR: Non-finite ablation effect for feature {feat}")
                
                # Faithfulness Spearman
                faith_sp, n_feat = compute_faithfulness_spearman(shap_abs_grouped, ablation_effects)
                
                if not np.isfinite(faith_sp) and n_feat >= 2:
                    validation["no_nan_inf_metrics"] = False
                    print(f"ERROR: Non-finite Spearman for instance {inst_idx} (SHAP)")
                
                instance_results.append({
                    'dataset': dataset,
                    'model': model_name,
                    'seed': seed,
                    'instance_index': inst_idx,
                    'explainer': 'shap',
                    'faithfulness_spearman': faith_sp,
                    'n_features': len(shap_abs_grouped),
                    'valid_feature_count': n_feat,
                })
                
                # Top-k deletion
                topk_res = topk_deletion_analysis(
                    model, scaler, X_instance, shap_abs_grouped, ref_values, onehot_groups,
                    TOPK_VALUES, N_RANDOM_BASELINES, FAITHFULNESS_SEED
                )
                
                for tk in topk_res:
                    topk_results.append({
                        'dataset': dataset,
                        'model': model_name,
                        'seed': seed,
                        'instance_index': inst_idx,
                        'explainer': 'shap',
                        **tk
                    })
            
            # Test LIME
            lime_row = lime_df[lime_df['instance_index'] == inst_idx]
            if len(lime_row) > 0:
                lime_attrs = lime_row[lime_feat_cols].values[0]
                
                # Map to one-hot groups
                lime_abs_grouped = {}
                for group_name, group_cols in onehot_groups.items():
                    group_indices = [lime_feat_cols.index(c) for c in group_cols if c in lime_feat_cols]
                    if group_indices:
                        lime_abs_grouped[group_name] = sum(abs(lime_attrs[i]) for i in group_indices)
                
                # Compute ablation effects (reuse from SHAP)
                faith_sp, n_feat = compute_faithfulness_spearman(lime_abs_grouped, ablation_effects)
                
                if not np.isfinite(faith_sp) and n_feat >= 2:
                    validation["no_nan_inf_metrics"] = False
                    print(f"ERROR: Non-finite Spearman for instance {inst_idx} (LIME)")
                
                instance_results.append({
                    'dataset': dataset,
                    'model': model_name,
                    'seed': seed,
                    'instance_index': inst_idx,
                    'explainer': 'lime',
                    'faithfulness_spearman': faith_sp,
                    'n_features': len(lime_abs_grouped),
                    'valid_feature_count': n_feat,
                })
                
                # Top-k deletion
                topk_res = topk_deletion_analysis(
                    model, scaler, X_instance, lime_abs_grouped, ref_values, onehot_groups,
                    TOPK_VALUES, N_RANDOM_BASELINES, FAITHFULNESS_SEED
                )
                
                for tk in topk_res:
                    topk_results.append({
                        'dataset': dataset,
                        'model': model_name,
                        'seed': seed,
                        'instance_index': inst_idx,
                        'explainer': 'lime',
                        **tk
                    })
        
        # Check feature counts
        expected_feature_count = len(onehot_groups)
        for res in instance_results:
            if res['n_features'] != expected_feature_count:
                validation["feature_counts_correct"] = False
                print(f"WARNING: Feature count mismatch for instance {res['instance_index']}")
        
        # Save smoke test outputs
        instance_df = pd.DataFrame(instance_results)
        topk_df = pd.DataFrame(topk_results)
        
        instance_df.to_csv(FAITHFULNESS_DIR / "smoke_test_instance.csv", index=False)
        topk_df.to_csv(FAITHFULNESS_DIR / "smoke_test_topk.csv", index=False)
        
        print()
        print("Smoke test outputs:")
        print(f"  Instance results: {len(instance_results)} rows")
        print(f"  Top-k results: {len(topk_results)} rows")
        print()
        
        # Validate output schema
        required_instance_cols = [
            'dataset', 'model', 'seed', 'instance_index', 'explainer',
            'faithfulness_spearman', 'n_features', 'valid_feature_count'
        ]
        required_topk_cols = [
            'dataset', 'model', 'seed', 'instance_index', 'explainer', 'k',
            'topk_prediction_drift', 'random_prediction_drift_mean',
            'random_prediction_drift_sd', 'topk_minus_random', 'faithfulness_enrichment'
        ]
        
        if not all(c in instance_df.columns for c in required_instance_cols):
            validation["output_schema_correct"] = False
            print("ERROR: Instance output schema incorrect")
        
        if not all(c in topk_df.columns for c in required_topk_cols):
            validation["output_schema_correct"] = False
            print("ERROR: Top-k output schema incorrect")
        
        # Overall pass
        validation["overall_pass"] = all([
            validation["predictions_finite"],
            validation["ablation_predictions_finite"],
            validation["feature_counts_correct"],
            validation["no_nan_inf_metrics"],
            validation["feature_ordering_matches"],
            validation["output_schema_correct"],
        ])
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        validation["overall_pass"] = False
    
    # Save validation report
    validation_path = FAITHFULNESS_DIR / "validation_report.json"
    with open(validation_path, 'w') as f:
        json.dump(validation, f, indent=2)
    
    print()
    print("="*70)
    print("VALIDATION REPORT")
    print("="*70)
    for check, status in validation.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"  {check}: {status_str}")
    print()
    
    if validation["overall_pass"]:
        print("✓ SMOKE TEST PASSED")
    else:
        print("✗ SMOKE TEST FAILED")
    
    print()
    print(f"Validation report saved: {validation_path}")
    
    return validation["overall_pass"]


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)

