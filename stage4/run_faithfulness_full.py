"""
stage4/run_faithfulness_full.py
================================
Full faithfulness evaluation across all 60 combinations:
- 3 datasets × 4 models × 5 seeds = 60 combinations
- 50 instances per combination
- 2 explainers (SHAP, LIME)
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage4.config import (
    TRAINING_SEEDS,
    DATASETS,
    MODELS,
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


def run_full_faithfulness():
    """Run full faithfulness evaluation."""
    
    print("="*70)
    print("STAGE 4A FAITHFULNESS — FULL EVALUATION")
    print("="*70)
    print(f"Datasets: {DATASETS}")
    print(f"Models: {MODELS}")
    print(f"Seeds: {TRAINING_SEEDS}")
    print(f"Instances: 50 per combination")
    print(f"Explainers: SHAP, LIME")
    print()
    
    all_instance_results = []
    all_topk_results = []
    
    total_combinations = len(DATASETS) * len(MODELS) * len(TRAINING_SEEDS)
    current = 0
    
    for dataset in DATASETS:
        # Load training data for reference values
        train_path = PROCESSED / f"{dataset}_train.parquet"
        train_df = pd.read_parquet(train_path)
        
        feature_cols = [c for c in train_df.columns if c not in ['Default', 'instance_index']]
        ref_values = compute_reference_values(train_df, feature_cols)
        onehot_groups = identify_onehot_groups(feature_cols)
        
        # Load test data
        test_path = PROCESSED / f"{dataset}_test.parquet"
        test_df = pd.read_parquet(test_path)
        
        for model_name in MODELS:
            for seed in TRAINING_SEEDS:
                current += 1
                print(f"[{current}/{total_combinations}] Processing {dataset}/{model_name}/seed{seed}...")
                
                try:
                    # Load model
                    model, scaler = load_model_and_scaler(dataset, model_name, seed)
                    
                    # Load SHAP explanations
                    shap_path = STAGE2_EXPLAIN_DIR / f"{dataset}_{model_name}_seed{seed}_shap.parquet"
                    if not shap_path.exists():
                        print(f"  WARNING: SHAP file not found, skipping")
                        continue
                    shap_df = pd.read_parquet(shap_path)
                    
                    # Load LIME explanations
                    lime_path = STAGE2_EXPLAIN_DIR / f"{dataset}_{model_name}_seed{seed}_lime_contrib.parquet"
                    if not lime_path.exists():
                        print(f"  WARNING: LIME file not found, skipping")
                        continue
                    lime_df = pd.read_parquet(lime_path)
                    
                    # Feature columns from explanations
                    shap_feat_cols = [c for c in shap_df.columns if c not in ['instance_index', 'actual_default', 'predicted_prob', 'dataset', 'model', 'seed']]
                    lime_feat_cols = [c for c in lime_df.columns if c not in ['instance_index', 'actual_default', 'predicted_prob', 'dataset', 'model', 'seed']]
                    
                    # Process each instance
                    test_instances = shap_df['instance_index'].unique()
                    
                    for inst_idx in test_instances:
                        # Get instance data
                        X_instance = test_df[test_df.index == inst_idx][feature_cols]
                        
                        if len(X_instance) == 0:
                            continue
                        
                        # Compute ablation effects once (reused for both explainers)
                        p_orig, ablation_effects = compute_ablation_effects(
                            model, scaler, X_instance, feature_cols, ref_values, onehot_groups
                        )
                        
                        # Process SHAP
                        shap_row = shap_df[shap_df['instance_index'] == inst_idx]
                        if len(shap_row) > 0:
                            shap_attrs = shap_row[shap_feat_cols].values[0]
                            
                            # Map to one-hot groups
                            shap_abs_grouped = {}
                            for group_name, group_cols in onehot_groups.items():
                                group_indices = [shap_feat_cols.index(c) for c in group_cols if c in shap_feat_cols]
                                if group_indices:
                                    shap_abs_grouped[group_name] = sum(abs(shap_attrs[i]) for i in group_indices)
                            
                            # Faithfulness Spearman
                            faith_sp, n_feat = compute_faithfulness_spearman(shap_abs_grouped, ablation_effects)
                            
                            all_instance_results.append({
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
                                TOPK_VALUES, N_RANDOM_BASELINES, FAITHFULNESS_SEED + inst_idx
                            )
                            
                            for tk in topk_res:
                                all_topk_results.append({
                                    'dataset': dataset,
                                    'model': model_name,
                                    'seed': seed,
                                    'instance_index': inst_idx,
                                    'explainer': 'shap',
                                    **tk
                                })
                        
                        # Process LIME
                        lime_row = lime_df[lime_df['instance_index'] == inst_idx]
                        if len(lime_row) > 0:
                            lime_attrs = lime_row[lime_feat_cols].values[0]
                            
                            # Map to one-hot groups
                            lime_abs_grouped = {}
                            for group_name, group_cols in onehot_groups.items():
                                group_indices = [lime_feat_cols.index(c) for c in group_cols if c in lime_feat_cols]
                                if group_indices:
                                    lime_abs_grouped[group_name] = sum(abs(lime_attrs[i]) for i in group_indices)
                            
                            # Faithfulness Spearman
                            faith_sp, n_feat = compute_faithfulness_spearman(lime_abs_grouped, ablation_effects)
                            
                            all_instance_results.append({
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
                                TOPK_VALUES, N_RANDOM_BASELINES, FAITHFULNESS_SEED + inst_idx + 10000
                            )
                            
                            for tk in topk_res:
                                all_topk_results.append({
                                    'dataset': dataset,
                                    'model': model_name,
                                    'seed': seed,
                                    'instance_index': inst_idx,
                                    'explainer': 'lime',
                                    **tk
                                })
                    
                    print(f"  Completed: {len(test_instances)} instances")
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Save results
    instance_df = pd.DataFrame(all_instance_results)
    topk_df = pd.DataFrame(all_topk_results)
    
    instance_df.to_csv(FAITHFULNESS_DIR / "faithfulness_instance.csv", index=False)
    topk_df.to_csv(FAITHFULNESS_DIR / "faithfulness_topk.csv", index=False)
    
    print()
    print("="*70)
    print("FULL EVALUATION COMPLETE")
    print("="*70)
    print(f"Instance results: {len(instance_df)} rows")
    print(f"Top-k results: {len(topk_df)} rows")
    print()
    print(f"Outputs saved to: {FAITHFULNESS_DIR}")
    print(f"  - faithfulness_instance.csv")
    print(f"  - faithfulness_topk.csv")
    print()


if __name__ == "__main__":
    run_full_faithfulness()

