"""
phase2_train_models.py

Model Training — XAI-StabilityNet

Models:
    1. XGBoost
    2. LightGBM
    3. Random Forest
    4. MLP

The held-out test set is NEVER used for training or early stopping.
All four models use the same train/test split so their predictions and
explanations can be compared on identical applicants.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import joblib
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import (
    DATASETS,
    PROCESSED_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    TARGET_COL,
)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def available_datasets():
    keys = []

    for key in DATASETS:
        train_path = PROCESSED_DIR / f"{key}_train.parquet"
        test_path = PROCESSED_DIR / f"{key}_test.parquet"

        if train_path.exists() and test_path.exists():
            keys.append(key)

    return keys


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_xy(dataset_key):

    train_path = PROCESSED_DIR / f"{dataset_key}_train.parquet"
    test_path = PROCESSED_DIR / f"{dataset_key}_test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    del train_df, test_df
    gc.collect()

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_models(y_train):

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)

    scale_pos_weight = n_neg / max(n_pos, 1)

    models = {

        "xgboost": xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            early_stopping_rounds=30,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),

        "lightgbm": lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),

        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=RANDOM_SEED,
        ),
    }

    return models


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y_test, probabilities),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
    }


# ---------------------------------------------------------------------------
# Train one dataset
# ---------------------------------------------------------------------------

def train_dataset(dataset_key, metrics_rows):

    print(f"\n{'=' * 60}")
    print(f"PHASE 2 — {dataset_key.upper()}")
    print(f"{'=' * 60}")

    X_train, y_train, X_test, y_test = load_xy(dataset_key)

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Features         : {X_train.shape[1]}")
    print(f"Default rate     : {y_train.mean():.3f}")

    # ---------------------------------------------------------------
    # Internal validation split
    # ---------------------------------------------------------------

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=y_train,
    )

    models = make_models(y_tr)

    # ---------------------------------------------------------------
    # Train models
    # ---------------------------------------------------------------

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")
        start = time.time()

        # -----------------------------------------------------------
        # XGBoost
        # -----------------------------------------------------------

        if model_name == "xgboost":

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        # -----------------------------------------------------------
        # LightGBM
        # -----------------------------------------------------------

        elif model_name == "lightgbm":

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=30,
                        verbose=False,
                    )
                ],
            )

        # -----------------------------------------------------------
        # Random Forest
        # -----------------------------------------------------------

        elif model_name == "random_forest":

            # RF does not use early stopping.
            # Train on the complete training partition.
            model.fit(X_train, y_train)

        # -----------------------------------------------------------
        # MLP
        # -----------------------------------------------------------

        elif model_name == "mlp":

            # Scaling is fitted ONLY on training data.
            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(
                X_train_scaled,
                y_train,
            )

            metrics = evaluate_model(
                model,
                X_test_scaled,
                y_test,
            )

            scaler_path = (
                MODELS_DIR
                / f"{dataset_key}_mlp_scaler.joblib"
            )

            joblib.dump(scaler, scaler_path)

            print(
                f"  Scaler saved -> {scaler_path.name}"
            )

            elapsed = time.time() - start

            row = {
                "dataset": dataset_key,
                "model": model_name,
                **metrics,
                "train_seconds": round(elapsed, 2),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": X_train.shape[1],
            }

            metrics_rows.append(row)

            print(
                f"  AUC={metrics['auc']:.4f} "
                f"Acc={metrics['accuracy']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"({elapsed:.1f}s)"
            )

            model_path = (
                MODELS_DIR
                / f"{dataset_key}_{model_name}.joblib"
            )

            joblib.dump(model, model_path)

            print(f"  Model saved -> {model_path.name}")

            del X_train_scaled, X_test_scaled

            gc.collect()

            continue

        # -----------------------------------------------------------
        # Tree model evaluation
        # -----------------------------------------------------------

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        elapsed = time.time() - start

        row = {
            "dataset": dataset_key,
            "model": model_name,
            **metrics,
            "train_seconds": round(elapsed, 2),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X_train.shape[1],
        }

        metrics_rows.append(row)

        print(
            f"  AUC={metrics['auc']:.4f} "
            f"Acc={metrics['accuracy']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"({elapsed:.1f}s)"
        )

        # -----------------------------------------------------------
        # Save model
        # -----------------------------------------------------------

        model_path = (
            MODELS_DIR
            / f"{dataset_key}_{model_name}.joblib"
        )

        joblib.dump(model, model_path)

        print(f"  Model saved -> {model_path.name}")

        del model
        gc.collect()

    # ---------------------------------------------------------------
    # Save feature ordering
    # ---------------------------------------------------------------

    feature_order_path = (
        MODELS_DIR
        / f"{dataset_key}_feature_columns.joblib"
    )

    joblib.dump(
        list(X_train.columns),
        feature_order_path,
    )

    print(
        f"\nFeature order saved -> "
        f"{feature_order_path.name}"
    )

    del X_train, y_train, X_test, y_test
    del X_tr, X_val, y_tr, y_val, models

    gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default=None,
    )

    args = parser.parse_args()

    if args.dataset:

        datasets = [args.dataset]

    else:

        datasets = available_datasets()

    if not datasets:

        print(
            "No processed datasets found. "
            "Run Phase 1 first."
        )

        return

    metrics_rows = []

    for dataset_key in datasets:

        train_dataset(
            dataset_key,
            metrics_rows,
        )

    # ---------------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------------

    metrics_df = pd.DataFrame(metrics_rows)

    metrics_path = (
        RESULTS_DIR
        / "phase2_baseline_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print(f"\n{'=' * 60}")
    print("PHASE 2 COMPLETE")
    print(f"{'=' * 60}")

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print(
        f"\nMetrics saved -> {metrics_path}"
    )


if __name__ == "__main__":
    main()