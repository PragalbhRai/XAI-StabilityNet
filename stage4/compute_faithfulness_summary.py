"""
stage4/compute_faithfulness_summary.py
======================================
Compute summary statistics with instance-cluster bootstrap.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage4.config import FAITHFULNESS_DIR


def bootstrap_instance_ci(df: pd.DataFrame, value_col: str, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42) -> tuple:
    """
    Bootstrap CI using instance-level resampling.
    
    Args:
        df: DataFrame with 'instance_index' and value column
        value_col: Column name to compute statistics on
        n_boot: Number of bootstrap resamples
        alpha: Significance level
        seed: Random seed
        
    Returns:
        (ci_low, ci_high, ci_status, n_instances)
    """
    instances = df['instance_index'].unique()
    n_inst = len(instances)
    
    if n_inst < 2:
        return (np.nan, np.nan, "NOT_ESTIMABLE_N_LT_2", n_inst)
    
    rng = np.random.default_rng(seed)
    boot_means = []
    
    for _ in range(n_boot):
        sampled_inst = rng.choice(instances, size=n_inst, replace=True)
        sampled_rows = df[df['instance_index'].isin(sampled_inst)]
        boot_means.append(sampled_rows[value_col].mean())
    
    ci_low, ci_high = np.percentile(boot_means, [alpha/2*100, (1-alpha/2)*100])
    
    return (ci_low, ci_high, "OK", n_inst)


def compute_faithfulness_summary():
    """Compute summary statistics for faithfulness evaluation."""
    
    print("="*70)
    print("COMPUTING FAITHFULNESS SUMMARY STATISTICS")
    print("="*70)
    print()
    
    # Load instance-level results
    instance_path = FAITHFULNESS_DIR / "faithfulness_instance.csv"
    if not instance_path.exists():
        print("ERROR: faithfulness_instance.csv not found")
        print("Please run run_faithfulness_full.py first")
        return
    
    instance_df = pd.read_csv(instance_path)
    
    # Load top-k results
    topk_path = FAITHFULNESS_DIR / "faithfulness_topk.csv"
    if not topk_path.exists():
        print("ERROR: faithfulness_topk.csv not found")
        return
    
    topk_df = pd.read_csv(topk_path)
    
    print(f"Instance results: {len(instance_df)} rows")
    print(f"Top-k results: {len(topk_df)} rows")
    print()
    
    # Summary statistics
    summary_rows = []
    
    # 1. Faithfulness Spearman
    print("Computing faithfulness_spearman summaries...")
    for (dataset, model, explainer), grp in instance_df.groupby(['dataset', 'model', 'explainer']):
        vals = grp['faithfulness_spearman'].dropna().values
        
        if len(vals) > 0:
            ci_low, ci_high, ci_status, n_inst = bootstrap_instance_ci(
                grp.dropna(subset=['faithfulness_spearman']), 
                'faithfulness_spearman'
            )
            
            summary_rows.append({
                'dataset': dataset,
                'model': model,
                'explainer': explainer,
                'metric': 'faithfulness_spearman',
                'n_instances': n_inst,
                'mean': vals.mean(),
                'median': np.median(vals),
                'std': vals.std(),
                'q25': np.percentile(vals, 25),
                'q75': np.percentile(vals, 75),
                'ci_low': ci_low,
                'ci_high': ci_high,
                'ci_status': ci_status,
            })
    
    # 2. Top-k metrics (for each k)
    print("Computing top-k summaries...")
    for k_val in topk_df['k'].unique():
        topk_k = topk_df[topk_df['k'] == k_val]
        
        # topk_minus_random
        for (dataset, model, explainer), grp in topk_k.groupby(['dataset', 'model', 'explainer']):
            vals = grp['topk_minus_random'].dropna().values
            
            if len(vals) > 0:
                ci_low, ci_high, ci_status, n_inst = bootstrap_instance_ci(
                    grp.dropna(subset=['topk_minus_random']), 
                    'topk_minus_random'
                )
                
                summary_rows.append({
                    'dataset': dataset,
                    'model': model,
                    'explainer': explainer,
                    'metric': f'topk_minus_random_k{k_val}',
                    'n_instances': n_inst,
                    'mean': vals.mean(),
                    'median': np.median(vals),
                    'std': vals.std(),
                    'q25': np.percentile(vals, 25),
                    'q75': np.percentile(vals, 75),
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'ci_status': ci_status,
                })
        
        # faithfulness_enrichment
        for (dataset, model, explainer), grp in topk_k.groupby(['dataset', 'model', 'explainer']):
            vals = grp['faithfulness_enrichment'].dropna().values
            
            if len(vals) > 0:
                ci_low, ci_high, ci_status, n_inst = bootstrap_instance_ci(
                    grp.dropna(subset=['faithfulness_enrichment']), 
                    'faithfulness_enrichment'
                )
                
                summary_rows.append({
                    'dataset': dataset,
                    'model': model,
                    'explainer': explainer,
                    'metric': f'faithfulness_enrichment_k{k_val}',
                    'n_instances': n_inst,
                    'mean': vals.mean(),
                    'median': np.median(vals),
                    'std': vals.std(),
                    'q25': np.percentile(vals, 25),
                    'q75': np.percentile(vals, 75),
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'ci_status': ci_status,
                })
    
    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = FAITHFULNESS_DIR / "faithfulness_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print()
    print("="*70)
    print("SUMMARY STATISTICS COMPLETE")
    print("="*70)
    print(f"Summary rows: {len(summary_df)}")
    print(f"Saved to: {summary_path}")
    print()
    
    # Display sample
    print("Sample summary statistics:")
    print(summary_df.head(10))
    print()


if __name__ == "__main__":
    compute_faithfulness_summary()

