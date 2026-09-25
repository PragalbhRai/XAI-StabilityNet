import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
from typing import Dict, List, Tuple

# Suppress constant input warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add stage2 to path for config access
sys.path.insert(0, str(Path(__file__).parent / "stage2"))
from config import (
    SEMANTIC_GROUPS, DATASETS, MODELS, TRAINING_SEEDS,
    RESULTS_DIR, ROOT
)

# Constants
EXPLAINERS = ["shap", "lime"]
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPS = 5000
OUTPUT_DIR = ROOT / "results" / "stage3"
OUTPUT_FILE = OUTPUT_DIR / "semantic_stability.csv"

def load_explanation(dataset: str, model: str, seed: int, explainer: str) -> pd.DataFrame:
    """Load explanation file for a specific combination."""
    if explainer == "lime":
        fname = f"{dataset}_{model}_seed{seed}_lime_contrib.parquet"
    else:
        fname = f"{dataset}_{model}_seed{seed}_shap.parquet"
    
    path = RESULTS_DIR / "explanations" / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    
    df = pd.read_parquet(path)
    return df

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature columns (exclude metadata)."""
    metadata_cols = {'instance_index', 'actual_default', 'predicted_prob', 
                     'dataset', 'model', 'seed'}
    return [c for c in df.columns if c not in metadata_cols]

def aggregate_by_semantic_category(
    attributions: pd.Series,
    dataset: str,
    feature_cols: List[str]
) -> Dict[str, float]:
    """
    Aggregate signed feature attributions by semantic category.
    Returns dict: {category: sum_of_attributions}
    Only includes non-empty categories for this dataset.
    """
    category_sums = {}
    
    for category, feature_map in SEMANTIC_GROUPS.items():
        features_for_dataset = feature_map.get(dataset, [])
        if not features_for_dataset:
            continue  # Skip empty categories
        
        # Filter to features that exist in the explanation
        valid_features = [f for f in features_for_dataset if f in feature_cols]
        if not valid_features:
            continue
        
        # Sum signed attributions
        category_sum = attributions[valid_features].sum()
        category_sums[category] = category_sum
    
    return category_sums

def compute_semantic_stability(
    cat_vec1: Dict[str, float],
    cat_vec2: Dict[str, float]
) -> float:
    """
    Compute semantic stability between two category-attribution vectors.
    
    Semantic Stability = 0.5 * rank_agreement + 0.5 * sign_agreement
    
    Returns:
        float: Stability score in [0, 1], or np.nan if categories empty/mismatch
    """
    # Ensure same categories
    if set(cat_vec1.keys()) != set(cat_vec2.keys()):
        return np.nan
    
    if len(cat_vec1) == 0:
        return np.nan
    
    categories = sorted(cat_vec1.keys())
    vals1 = np.array([cat_vec1[c] for c in categories])
    vals2 = np.array([cat_vec2[c] for c in categories])
    
    # Rank agreement: Spearman correlation
    if len(vals1) < 2:
        # Need at least 2 categories for Spearman
        rank_agreement = 1.0 if vals1[0] == vals2[0] else 0.0
    else:
        rho, _ = spearmanr(vals1, vals2)
        if np.isnan(rho):
            rho = 0.0
        # Convert from [-1, 1] to [0, 1]
        rank_agreement = (rho + 1.0) / 2.0
    
    # Sign agreement: fraction of categories with same sign
    signs1 = np.sign(vals1)
    signs2 = np.sign(vals2)
    sign_agreement = (signs1 == signs2).mean()
    
    # Combined metric
    stability = 0.5 * rank_agreement + 0.5 * sign_agreement
    
    return stability

def compute_pairwise_semantic_stability(
    dataset: str,
    model: str,
    explainer: str
) -> List[Dict]:
    """
    Compute pairwise semantic stability across all seed pairs for one dataset/model/explainer.
    
    Returns:
        List of dicts with keys: seed1, seed2, instance_index, stability
    """
    results = []
    
    # Load explanations for all seeds
    seed_dfs = {}
    for seed in TRAINING_SEEDS:
        try:
            df = load_explanation(dataset, model, seed, explainer)
            seed_dfs[seed] = df
        except FileNotFoundError:
            print(f"  WARNING: Missing {dataset}/{model}/seed{seed}/{explainer}")
            return []
    
    # Get feature columns (should be same across all seeds)
    feature_cols = get_feature_columns(seed_dfs[TRAINING_SEEDS[0]])
    
    # For each seed pair
    for seed1, seed2 in combinations(TRAINING_SEEDS, 2):
        df1 = seed_dfs[seed1]
        df2 = seed_dfs[seed2]
        
        # For each instance
        for instance_idx in df1['instance_index'].unique():
            row1 = df1[df1['instance_index'] == instance_idx].iloc[0]
            row2 = df2[df2['instance_index'] == instance_idx].iloc[0]
            
            # Extract attribution vectors
            attrs1 = row1[feature_cols]
            attrs2 = row2[feature_cols]
            
            # Aggregate by category
            cat1 = aggregate_by_semantic_category(attrs1, dataset, feature_cols)
            cat2 = aggregate_by_semantic_category(attrs2, dataset, feature_cols)
            
            # Compute stability
            stability = compute_semantic_stability(cat1, cat2)
            
            results.append({
                'dataset': dataset,
                'model': model,
                'explainer': explainer,
                'seed1': seed1,
                'seed2': seed2,
                'instance_index': instance_idx,
                'stability': stability,
                'n_categories': len(cat1)
            })
    
    return results

def bootstrap_instance_level_ci(
    pairwise_df: pd.DataFrame,
    n_boot: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05
) -> Dict:
    """
    Bootstrap CI using INSTANCE-level resampling (CORRECTED).
    
    CRITICAL FIX: Properly implements bootstrap WITH REPLACEMENT.
    
    Method:
    1. Compute instance-level mean for each of 50 instances (avg of 10 seed-pairs)
    2. Bootstrap resample from these 50 instance-level means
    3. When instance i is sampled k times, its mean contributes k times
    
    This ensures duplicate draws are properly retained.
    
    Args:
        pairwise_df: DataFrame with columns [instance_index, stability]
                     Contains 50 instances × 10 seed-pairs = 500 rows
        n_boot: Number of bootstrap replicates
        seed: Random seed
        alpha: Significance level (0.05 for 95% CI)
    
    Returns:
        Dict with mean, median, std, q25, q75, min, max, ci_low, ci_high, ci_status
    """
    # Step 1: Compute instance-level means (average 10 seed-pairs per instance)
    instance_means = pairwise_df.groupby('instance_index')['stability'].mean()
    instances = instance_means.index.values
    instance_mean_values = instance_means.values
    
    n_inst = len(instances)
    
    if n_inst < 2:
        return {
            'mean': np.nan, 'median': np.nan, 'std': np.nan,
            'q25': np.nan, 'q75': np.nan, 'min': np.nan, 'max': np.nan,
            'ci_low': np.nan, 'ci_high': np.nan, 'ci_status': 'NOT_ESTIMABLE_N_LT_2'
        }
    
    # Point estimates (computed on all 500 pairwise observations, not instance means)
    all_stabilities = pairwise_df['stability'].values
    point_stats = {
        'mean': np.nanmean(all_stabilities),
        'median': np.nanmedian(all_stabilities),
        'std': np.nanstd(all_stabilities, ddof=1),
        'q25': np.nanpercentile(all_stabilities, 25),
        'q75': np.nanpercentile(all_stabilities, 75),
        'min': np.nanmin(all_stabilities),
        'max': np.nanmax(all_stabilities)
    }
    
    # Step 2: Bootstrap CI with CORRECT instance-level resampling
    rng = np.random.default_rng(seed)
    boot_means = []
    
    for _ in range(n_boot):
        # Sample n instances WITH REPLACEMENT
        # If instance i appears k times, its mean contributes k times
        sampled_indices = rng.choice(np.arange(n_inst), size=n_inst, replace=True)
        sampled_instance_means = instance_mean_values[sampled_indices]
        
        # Compute mean of sampled instance means
        boot_means.append(np.nanmean(sampled_instance_means))
    
    # Step 3: Compute percentile CI
    ci_low = np.percentile(boot_means, alpha/2 * 100)
    ci_high = np.percentile(boot_means, (1 - alpha/2) * 100)
    ci_status = 'OK' if not np.isnan(ci_low) and not np.isnan(ci_high) else 'FAIL'
    
    return {
        **point_stats,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'ci_status': ci_status
    }

def main():
    """Main execution: compute semantic stability for all combinations."""
    print("="*80)
    print("SEMANTIC STABILITY ANALYSIS (CORRECTED BOOTSTRAP WITH REPLACEMENT)")
    print("="*80)
    print(f"Datasets: {DATASETS}")
    print(f"Models: {MODELS}")
    print(f"Explainers: {EXPLAINERS}")
    print(f"Seeds: {TRAINING_SEEDS} ({len(TRAINING_SEEDS)} seeds -> {len(list(combinations(TRAINING_SEEDS, 2)))} pairs)")
    print(f"Bootstrap: {BOOTSTRAP_REPS} replicates, seed={BOOTSTRAP_SEED}")
    print(f"Bootstrap unit: INSTANCE-LEVEL MEANS (50 per combination)")
    print(f"Bootstrap method: Sample 50 instance means WITH REPLACEMENT")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    # Collect all pairwise results
    all_pairwise = []
    
    total_combos = len(DATASETS) * len(MODELS) * len(EXPLAINERS)
    current = 0
    
    for dataset in DATASETS:
        for model in MODELS:
            for explainer in EXPLAINERS:
                current += 1
                print(f"[{current}/{total_combos}] Processing {dataset}/{model}/{explainer}...", end=' ', flush=True)
                
                pairwise = compute_pairwise_semantic_stability(dataset, model, explainer)
                all_pairwise.extend(pairwise)
                
                print(f"OK ({len(pairwise)} comparisons)")
    
    # Convert to DataFrame
    pairwise_df = pd.DataFrame(all_pairwise)
    
    print()
    print(f"Total pairwise comparisons: {len(pairwise_df)}")
    print(f"Expected: {total_combos} combos × 50 instances × 10 seed-pairs = {total_combos * 50 * 10}")
    
    # Validation checks
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    # Check coverage
    coverage_ok = True
    for dataset in DATASETS:
        for model in MODELS:
            for explainer in EXPLAINERS:
                subset = pairwise_df[
                    (pairwise_df['dataset'] == dataset) &
                    (pairwise_df['model'] == model) &
                    (pairwise_df['explainer'] == explainer)
                ]
                expected = 50 * 10  # 50 instances × 10 seed pairs
                actual = len(subset)
                if actual != expected:
                    coverage_ok = False
                    print(f"X {dataset}/{model}/{explainer}: {actual}/{expected} comparisons")
    
    if coverage_ok:
        print("OK All 24 combinations have 500 comparisons (50 instances × 10 seed pairs)")
    
    # Check for NaN stabilities
    n_nan = pairwise_df['stability'].isna().sum()
    if n_nan > 0:
        print(f"\nWARNING: {n_nan} NaN stability values detected")
        for dataset in DATASETS:
            for model in MODELS:
                for explainer in EXPLAINERS:
                    subset = pairwise_df[
                        (pairwise_df['dataset'] == dataset) &
                        (pairwise_df['model'] == model) &
                        (pairwise_df['explainer'] == explainer)
                    ]
                    n_nan_subset = subset['stability'].isna().sum()
                    if n_nan_subset > 0:
                        print(f"  {dataset}/{model}/{explainer}: {n_nan_subset} NaNs")
    else:
        print("OK No NaN stability values")
    
    # Show category counts by dataset
    print("\n" + "="*80)
    print("SEMANTIC CATEGORIES BY DATASET")
    print("="*80)
    for dataset in DATASETS:
        dataset_categories = [cat for cat, feats in SEMANTIC_GROUPS.items() if feats.get(dataset, [])]
        print(f"  {dataset}: {len(dataset_categories)} categories - {', '.join(dataset_categories)}")
    
    # Aggregate by dataset/model/explainer with CORRECTED bootstrap
    print("\n" + "="*80)
    print("BOOTSTRAPPING AGGREGATES (INSTANCE-LEVEL MEANS WITH REPLACEMENT)")
    print("="*80)
    
    summary_rows = []
    
    for dataset in DATASETS:
        for model in MODELS:
            for explainer in EXPLAINERS:
                subset = pairwise_df[
                    (pairwise_df['dataset'] == dataset) &
                    (pairwise_df['model'] == model) &
                    (pairwise_df['explainer'] == explainer)
                ]
                
                if len(subset) == 0:
                    continue
                
                # CORRECTED: Use instance-level means bootstrap
                boot_stats = bootstrap_instance_level_ci(subset)
                
                n_instances = subset['instance_index'].nunique()
                n_seed_pairs = len(subset) // n_instances if n_instances > 0 else 0
                n_categories = subset['n_categories'].iloc[0] if len(subset) > 0 else 0
                
                summary_rows.append({
                    'dataset': dataset,
                    'model': model,
                    'explainer': explainer,
                    'metric': 'semantic_stability',
                    'mean': boot_stats['mean'],
                    'median': boot_stats['median'],
                    'std': boot_stats['std'],
                    'q25': boot_stats['q25'],
                    'q75': boot_stats['q75'],
                    'min': boot_stats['min'],
                    'max': boot_stats['max'],
                    'ci_low': boot_stats['ci_low'],
                    'ci_high': boot_stats['ci_high'],
                    'ci_status': boot_stats['ci_status'],
                    'n_instances': n_instances,
                    'n_seed_pairs': n_seed_pairs,
                    'n_categories': n_categories,
                    'bootstrap_seed': BOOTSTRAP_SEED,
                    'bootstrap_reps': BOOTSTRAP_REPS
                })
                
                print(f"  {dataset}/{model}/{explainer}: mean={boot_stats['mean']:.4f}, CI=[{boot_stats['ci_low']:.4f}, {boot_stats['ci_high']:.4f}]")
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"OK Created: {OUTPUT_FILE}")
    print(f"  Rows: {len(summary_df)}")
    print(f"  Columns: {list(summary_df.columns)}")
    print()
    print("Semantic stability ranges by explainer:")
    for explainer in EXPLAINERS:
        subset = summary_df[summary_df['explainer'] == explainer]
        if len(subset) > 0:
            print(f"  {explainer.upper()}: [{subset['mean'].min():.4f}, {subset['mean'].max():.4f}]")
    print()
    print("OK Semantic stability analysis complete (CORRECTED bootstrap with replacement)!")

if __name__ == "__main__":
    main()
