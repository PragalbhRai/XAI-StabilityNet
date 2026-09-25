"""
stage4/faithfulness.py
======================
STAGE 4A — FAITHFULNESS EVALUATION (OPTIMIZED)

Evaluates whether explanation magnitude corresponds to actual model sensitivity
by ablating features and measuring prediction changes.

OPTIMIZATION: Batched predictions to reduce individual model calls from ~432K to ~9K
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage4.config import (
    TRAINING_SEEDS,
    DATASETS,
    MODELS,
    TARGET,
    N_EXPLAIN,
    EXPLAIN_SEED,
    PROCESSED,
    STAGE2_MODELS_DIR,
    STAGE2_EXPLAIN_DIR,
    TOPK_VALUES,
    N_RANDOM_BASELINES,
    FAITHFULNESS_SEED,
)


# ============================================================================
# REFERENCE VALUE COMPUTATION
# ============================================================================

def compute_reference_values(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    """
    Compute training-derived reference values for ablation.
    
    For numeric features: use training median
    For binary/categorical: use training mode
    For one-hot encoded groups: identify and handle consistently
    """
    ref_values = {}
    
    for col in feature_cols:
        if col == TARGET or col == 'instance_index':
            continue
            
        values = train_df[col].dropna()
        
        # Numeric feature: use median
        if pd.api.types.is_numeric_dtype(values):
            # Check if binary (0/1 only)
            unique_vals = values.unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                # Binary: use mode
                ref_values[col] = values.mode()[0] if len(values.mode()) > 0 else 0.0
            else:
                # Continuous: use median
                ref_values[col] = values.median()
        else:
            # Categorical: use mode
            ref_values[col] = values.mode()[0] if len(values.mode()) > 0 else values.iloc[0]
    
    return ref_values


def identify_onehot_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    """
    Identify one-hot encoded feature groups by common prefixes.
    Returns dict mapping group_name -> list of one-hot columns.
    """
    # Common one-hot prefixes in the datasets
    prefixes_to_check = [
        'checking_status_', 'credit_history_', 'purpose_', 'savings_status_',
        'employment_', 'personal_status_', 'other_parties_', 'property_magnitude_',
        'other_payment_plans_', 'housing_', 'job_',
        'SEX_', 'EDUCATION_', 'MARRIAGE_',
    ]
    
    groups = {}
    grouped_cols = set()
    
    for prefix in prefixes_to_check:
        matching = [c for c in feature_cols if c.startswith(prefix)]
        if len(matching) > 1:
            groups[prefix.rstrip('_')] = matching
            grouped_cols.update(matching)
    
    # Ungrouped columns are their own group
    for col in feature_cols:
        if col not in grouped_cols and col != TARGET and col != 'instance_index':
            groups[col] = [col]
    
    return groups


# ============================================================================
# MODEL LOADING AND PREDICTION
# ============================================================================

def load_model_and_scaler(dataset: str, model_name: str, seed: int) -> Tuple[object, Optional[StandardScaler]]:
    """Load trained model and scaler (for MLP only)."""
    model_path = STAGE2_MODELS_DIR / f"{dataset}_{model_name}_seed{seed}.joblib"
    model = joblib.load(model_path)
    
    scaler = None
    if model_name == "mlp":
        scaler_path = STAGE2_MODELS_DIR / f"{dataset}_mlp_scaler_seed{seed}.joblib"
        scaler = joblib.load(scaler_path)
    
    return model, scaler


def predict_proba(model, scaler: Optional[StandardScaler], X: pd.DataFrame) -> np.ndarray:
    """Get prediction probabilities, handling MLP scaling."""
    if scaler is not None:
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        probs = model.predict_proba(X)[:, 1]
    return probs


# ============================================================================
# FEATURE ABLATION
# ============================================================================

def ablate_feature(
    X_orig: pd.DataFrame,
    feature_cols: List[str],
    ref_values: Dict[str, float],
    onehot_groups: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Ablate a single feature or one-hot group by replacing with reference values.
    
    Args:
        X_orig: Original feature matrix (single instance as DataFrame)
        feature_cols: List of column names to ablate
        ref_values: Dictionary of reference values
        onehot_groups: One-hot group mapping
        
    Returns:
        Ablated feature matrix
    """
    X_ablated = X_orig.copy()
    
    for col in feature_cols:
        if col in ref_values:
            X_ablated[col] = ref_values[col]
    
    return X_ablated


# ============================================================================
# FAITHFULNESS METRICS (OPTIMIZED)
# ============================================================================

def compute_ablation_effects_batched(
    model,
    scaler: Optional[StandardScaler],
    X_instance: pd.DataFrame,
    feature_cols: List[str],
    ref_values: Dict[str, float],
    onehot_groups: Dict[str, List[str]]
) -> Tuple[float, Dict[str, float]]:
    """
    Compute ablation effect for each feature group using BATCHED predictions.
    
    OPTIMIZATION: Single batch prediction instead of one per feature group.
    
    Returns:
        p_orig: Original prediction probability
        ablation_effects: Dict mapping feature/group -> ablation effect
    """
    # Original prediction
    p_orig = predict_proba(model, scaler, X_instance)[0]
    
    # Create all ablations at once
    ablation_dfs = []
    group_names = []
    
    for group_name, group_cols in onehot_groups.items():
        X_ablated = ablate_feature(X_instance, group_cols, ref_values, onehot_groups)
        ablation_dfs.append(X_ablated)
        group_names.append(group_name)
    
    # Batch prediction
    if len(ablation_dfs) > 0:
        X_batch = pd.concat(ablation_dfs, ignore_index=True)
        p_batch = predict_proba(model, scaler, X_batch)
        
        # Map predictions back to feature groups
        ablation_effects = {}
        for idx, group_name in enumerate(group_names):
            ablation_effects[group_name] = abs(p_orig - p_batch[idx])
    else:
        ablation_effects = {}
    
    return p_orig, ablation_effects


def compute_faithfulness_spearman(
    explanation_abs: Dict[str, float],
    ablation_effects: Dict[str, float]
) -> Tuple[float, int]:
    """
    Compute Spearman correlation between explanation magnitudes and ablation effects.
    
    Returns:
        spearman_rho: Correlation coefficient
        n_features: Number of features used
    """
    # Align features
    common_features = set(explanation_abs.keys()) & set(ablation_effects.keys())
    
    if len(common_features) < 2:
        return np.nan, len(common_features)
    
    exp_vals = [explanation_abs[f] for f in common_features]
    abl_vals = [ablation_effects[f] for f in common_features]
    
    rho, _ = spearmanr(exp_vals, abl_vals)
    
    return float(rho), len(common_features)


# ============================================================================
# TOP-K DELETION ANALYSIS (OPTIMIZED)
# ============================================================================

def topk_deletion_analysis_batched(
    model,
    scaler: Optional[StandardScaler],
    X_instance: pd.DataFrame,
    explanation_abs: Dict[str, float],
    ref_values: Dict[str, float],
    onehot_groups: Dict[str, List[str]],
    k_values: List[int],
    n_random: int,
    random_seed: int
) -> List[Dict]:
    """
    Perform top-k deletion and compare to random baseline using BATCHED predictions.
    
    OPTIMIZATION: Single batch prediction for all top-k + all random ablations.
    
    Returns list of dicts with k, topk_drift, random_drift_mean, etc.
    """
    results = []
    p_orig = predict_proba(model, scaler, X_instance)[0]
    
    # Rank features by explanation magnitude
    ranked_features = sorted(explanation_abs.items(), key=lambda x: x[1], reverse=True)
    feature_names = [f for f, _ in ranked_features]
    
    n_features_available = len(feature_names)
    
    rng = np.random.default_rng(random_seed)
    
    # Prepare all ablations at once
    ablation_dfs = []
    ablation_metadata = []  # (type, k_idx, rep_idx)
    
    for k_idx, k in enumerate(k_values):
        if k > n_features_available:
            effective_k = n_features_available
        else:
            effective_k = k
        
        # Top-k ablation
        topk_features = feature_names[:effective_k]
        X_topk = X_instance.copy()
        for feat in topk_features:
            if feat in onehot_groups:
                for col in onehot_groups[feat]:
                    if col in ref_values:
                        X_topk[col] = ref_values[col]
        
        ablation_dfs.append(X_topk)
        ablation_metadata.append(('topk', k_idx, 0))
        
        # Random baseline ablations (n_random repetitions)
        for rep in range(n_random):
            random_features = rng.choice(feature_names, size=effective_k, replace=False)
            X_rand = X_instance.copy()
            for feat in random_features:
                if feat in onehot_groups:
                    for col in onehot_groups[feat]:
                        if col in ref_values:
                            X_rand[col] = ref_values[col]
            
            ablation_dfs.append(X_rand)
            ablation_metadata.append(('random', k_idx, rep))
    
    # Batch prediction for ALL ablations
    if len(ablation_dfs) > 0:
        X_batch = pd.concat(ablation_dfs, ignore_index=True)
        p_batch = predict_proba(model, scaler, X_batch)
        
        # Group predictions by k-value
        topk_predictions = {}
        random_predictions = {k_idx: [] for k_idx in range(len(k_values))}
        
        for idx, (abl_type, k_idx, rep_idx) in enumerate(ablation_metadata):
            if abl_type == 'topk':
                topk_predictions[k_idx] = p_batch[idx]
            else:  # random
                random_predictions[k_idx].append(p_batch[idx])
        
        # Compute metrics for each k
        for k_idx, k in enumerate(k_values):
            if k > n_features_available:
                effective_k = n_features_available
            else:
                effective_k = k
            
            topk_drift = abs(p_orig - topk_predictions[k_idx])
            
            random_drifts = [abs(p_orig - p) for p in random_predictions[k_idx]]
            random_drift_mean = np.mean(random_drifts)
            random_drift_sd = np.std(random_drifts)
            
            enrichment = topk_drift / random_drift_mean if random_drift_mean > 0 else np.nan
            topk_minus_random = topk_drift - random_drift_mean
            
            results.append({
                'k': effective_k,
                'topk_prediction_drift': topk_drift,
                'random_prediction_drift_mean': random_drift_mean,
                'random_prediction_drift_sd': random_drift_sd,
                'topk_minus_random': topk_minus_random,
                'faithfulness_enrichment': enrichment,
            })
    
    return results


# ============================================================================
# BACKWARD COMPATIBILITY (Keep original non-batched versions)
# ============================================================================

def compute_ablation_effects(
    model,
    scaler: Optional[StandardScaler],
    X_instance: pd.DataFrame,
    feature_cols: List[str],
    ref_values: Dict[str, float],
    onehot_groups: Dict[str, List[str]]
) -> Tuple[float, Dict[str, float]]:
    """Original non-batched version (for backward compatibility)."""
    return compute_ablation_effects_batched(model, scaler, X_instance, feature_cols, ref_values, onehot_groups)


def topk_deletion_analysis(
    model,
    scaler: Optional[StandardScaler],
    X_instance: pd.DataFrame,
    explanation_abs: Dict[str, float],
    ref_values: Dict[str, float],
    onehot_groups: Dict[str, List[str]],
    k_values: List[int],
    n_random: int,
    random_seed: int
) -> List[Dict]:
    """Original non-batched version (for backward compatibility)."""
    return topk_deletion_analysis_batched(model, scaler, X_instance, explanation_abs, ref_values, onehot_groups, k_values, n_random, random_seed)

