"""
stage2/train.py
===============
STAGE 2 â€” PART A: MULTI-SEED MODEL TRAINING

Trains all 4 model families on all 3 datasets using each of the 5
fixed seeds in config.TRAINING_SEEDS.

For seed=42 (the Phase-2 seed), existing models are loaded directly
rather than retrained to preserve reproducibility of the pre-Stage-2 baseline.

Outputs:
  results/stage2/models/{dataset}_{model}_seed{seed}.joblib
  results/stage2/models/{dataset}_mlp_scaler_seed{seed}.joblib   (MLP only)
  results/stage2/models/all_metrics.csv
"""

from __future__ import annotations
import sys, json, gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage2.config import (
    PROCESSED, MODELS_V1, MODELS_DIR,
    TRAINING_SEEDS, DATASETS, MODELS, TARGET,
)

import warnings
warnings.filterwarnings("ignore", message=".*LightGBM.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*No further splits.*", category=UserWarning)


# ---------------------------------------------------------------------------
# Model factory  â€” same hyperparameters as Phase 2, only seed changes
# ---------------------------------------------------------------------------

def _build_models(y_train: pd.Series, seed: int) -> dict:
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = neg / max(pos, 1)

    return {
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            objective="binary:logistic", eval_metric="auc",
            random_state=seed, n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400, max_depth=4, num_leaves=15,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced",
            random_state=seed, n_jobs=-1, verbosity=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver="adam", alpha=0.0001,
            batch_size=64, learning_rate_init=0.001,
            max_iter=500, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20,
            random_state=seed,
        ),
    }


# ---------------------------------------------------------------------------
# Artifact path helpers
# ---------------------------------------------------------------------------

def model_path(dataset: str, model: str, seed: int) -> Path:
    return MODELS_DIR / f"{dataset}_{model}_seed{seed}.joblib"

def scaler_path(dataset: str, seed: int) -> Path:
    return MODELS_DIR / f"{dataset}_mlp_scaler_seed{seed}.joblib"


def _evaluate(dataset, model_name, seed, y_test, probs) -> dict:
    preds = (probs >= 0.5).astype(int)
    return {
        "dataset": dataset, "model": model_name, "seed": seed,
        "auc":      float(roc_auc_score(y_test, probs)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1":       float(f1_score(y_test, preds, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Single dataset Ã— single seed
# ---------------------------------------------------------------------------

def train_one(dataset: str, seed: int, force_retrain: bool = False) -> list[dict]:
    """Train all 4 models for one dataset/seed.
    Returns list of metric dicts (one per model).
    """
    print(f"\n  {dataset.upper()} / seed={seed}")

    train_df = pd.read_parquet(PROCESSED / f"{dataset}_train.parquet")
    test_df  = pd.read_parquet(PROCESSED / f"{dataset}_test.parquet")

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]

    # Phase-2 seed=42 models already exist at MODELS_V1. Reuse them unless
    # force_retrain=True, to guarantee identical baseline.
    use_existing = (seed == 42 and not force_retrain)

    models  = _build_models(y_train, seed)
    records = []

    for model_name, model in models.items():
        mpath = model_path(dataset, model_name, seed)
        spath = scaler_path(dataset, seed)

        # ------------------------------------------------------------------
        # Check if already done
        # ------------------------------------------------------------------
        if mpath.exists() and not force_retrain:
            print(f"    {model_name}: already exists, skipping.")
            loaded = joblib.load(mpath)
            scaler = joblib.load(spath) if (model_name == "mlp" and spath.exists()) else None
            if model_name == "mlp" and scaler is not None:
                probs = loaded.predict_proba(scaler.transform(X_test.to_numpy(dtype=float)))[:, 1]
            else:
                probs = loaded.predict_proba(X_test)[:, 1]
            records.append(_evaluate(dataset, model_name, seed, y_test, probs))
            continue

        # ------------------------------------------------------------------
        # Load Phase-2 model (seed=42, not re-trained)
        # ------------------------------------------------------------------
        if use_existing:
            v1_path = MODELS_V1 / f"{dataset}_{model_name}.joblib"
            if v1_path.exists():
                print(f"    {model_name}: copying Phase-2 model (seed=42).")
                loaded = joblib.load(v1_path)
                joblib.dump(loaded, mpath)
                if model_name == "mlp":
                    v1_scaler = MODELS_V1 / f"{dataset}_mlp_scaler.joblib"
                    scaler = joblib.load(v1_scaler)
                    joblib.dump(scaler, spath)
                    probs = loaded.predict_proba(scaler.transform(X_test.to_numpy(dtype=float)))[:, 1]
                else:
                    scaler = None
                    probs = loaded.predict_proba(X_test)[:, 1]
                records.append(_evaluate(dataset, model_name, seed, y_test, probs))
                continue

        # ------------------------------------------------------------------
        # Train fresh
        # ------------------------------------------------------------------
        print(f"    {model_name}: training...", end=" ", flush=True)
        if model_name == "mlp":
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train.to_numpy(dtype=float))
            X_te_sc = scaler.transform(X_test.to_numpy(dtype=float))
            model.fit(X_tr_sc, y_train)
            probs = model.predict_proba(X_te_sc)[:, 1]
            joblib.dump(scaler, spath)
        else:
            scaler = None
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]

        joblib.dump(model, mpath)
        m = _evaluate(dataset, model_name, seed, y_test, probs)
        print(f"AUC={m['auc']:.4f}")
        records.append(m)
        del model
        gc.collect()

    del train_df, test_df, X_train, y_train, X_test, y_test
    gc.collect()
    return records


# ---------------------------------------------------------------------------
# Full multi-seed training run
# ---------------------------------------------------------------------------

def run_training(
    datasets: list[str] | None = None,
    seeds:    list[int]  | None = None,
    force_retrain: bool = False,
) -> pd.DataFrame:
    """Train all combinations. Returns merged metrics DataFrame."""
    datasets = datasets or DATASETS
    seeds    = seeds    or TRAINING_SEEDS
    all_rows = []

    print("=" * 60)
    print("STAGE 2 â€” PART A: MULTI-SEED TRAINING")
    print(f"Datasets : {datasets}")
    print(f"Seeds    : {seeds}")
    print("=" * 60)

    for ds in datasets:
        for seed in seeds:
            rows = train_one(ds, seed, force_retrain=force_retrain)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out = MODELS_DIR / "all_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nMetrics saved â†’ {out.relative_to(ROOT)}")
    return df


# ---------------------------------------------------------------------------
# Convenience: load any trained model
# ---------------------------------------------------------------------------

def load_model(dataset: str, model_name: str, seed: int):
    """Load a Stage-2 trained model."""
    p = model_path(dataset, model_name, seed)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}. Run stage2/train.py first.")
    return joblib.load(p)

def load_scaler(dataset: str, seed: int):
    """Load MLP scaler for a given seed."""
    p = scaler_path(dataset, seed)
    if not p.exists():
        raise FileNotFoundError(f"Scaler not found: {p}")
    return joblib.load(p)

def predict_proba_fn(dataset: str, model_name: str, seed: int):
    """Return a callable (X_array) -> P(Default=1) for any model."""
    model = load_model(dataset, model_name, seed)
    if model_name == "mlp":
        scaler = load_scaler(dataset, seed)
        def fn(X):
            X = np.asarray(X, dtype=float)
            return model.predict_proba(scaler.transform(X))[:, 1]
    else:
        def fn(X):
            X = np.asarray(X, dtype=float)
            return model.predict_proba(X)[:, 1]
    return fn


if __name__ == "__main__":
    df = run_training()
    print("\n" + df.to_string(index=False))
