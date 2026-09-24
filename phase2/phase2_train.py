from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
METADATA_DIR = ROOT / "data" / "metadata"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "Default"

DATASETS = [
    "german",
    "taiwan",
    "gmsc",
]


# ============================================================
# MODEL FACTORIES
# ============================================================

def create_models(y_train):
    """
    Create the same four model families for every dataset.

    Class imbalance is handled independently for each dataset.
    """

    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)

    scale_pos_weight = negative / max(positive, 1)

    models = {

        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=SEED,
            n_jobs=-1,
        ),

        # ----------------------------------------------------
        # LightGBM
        # ----------------------------------------------------
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        ),

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=SEED,
        ),
    }

    return models


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    dataset,
    model_name,
    y_test,
    probabilities,
):
    """
    Evaluate using probabilities and a fixed 0.5 classification threshold.
    """

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "dataset": dataset,
        "model": model_name,

        "auc": roc_auc_score(
            y_test,
            probabilities
        ),

        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
    }


# ============================================================
# TRAIN ONE DATASET
# ============================================================

def train_dataset(dataset):
    """

    Train all four models on one dataset.

    Every dataset has:
        train.parquet
        test.parquet

    The test set is never used for training.
    """

    print("\n")
    print("=" * 70)
    print(f"XAI-STABILITYNET — PHASE 2 — {dataset.upper()}")
    print("=" * 70)

    train_path = PROCESSED_DIR / f"{dataset}_train.parquet"
    test_path = PROCESSED_DIR / f"{dataset}_test.parquet"

    if not train_path.exists() or not test_path.exists():
        print(f"Processed files missing for {dataset}.")
        print("Run Phase 1 first.")
        return []

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("\nLoading data...")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Features         : {X_train.shape[1]}")
    print(f"Default rate     : {y_train.mean():.4f}")

    # --------------------------------------------------------
    # VERIFY FEATURE ALIGNMENT
    # --------------------------------------------------------

    if list(X_train.columns) != list(X_test.columns):

        raise ValueError(
            f"Feature mismatch between train and test for {dataset}."
        )

    feature_columns = list(X_train.columns)

    # --------------------------------------------------------
    # SAVE FEATURE METADATA
    # --------------------------------------------------------

    metadata = {
        "dataset": dataset,
        "target": TARGET,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "feature_columns": feature_columns,
        "random_seed": SEED,
    }

    metadata_path = METADATA_DIR / f"{dataset}_model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    joblib.dump(
        feature_columns,
        MODEL_DIR / f"{dataset}_feature_columns.joblib"
    )

    print("\nFeature metadata saved.")

    # --------------------------------------------------------
    # CLASS BALANCE
    # --------------------------------------------------------

    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)

    class_weight_ratio = negative / max(positive, 1)

    print(f"Positive samples : {positive}")
    print(f"Negative samples : {negative}")
    print(f"Class ratio      : {class_weight_ratio:.3f}")

    # --------------------------------------------------------
    # CREATE MODELS
    # --------------------------------------------------------

    models = create_models(y_train)

    results = []

    probabilities = {
        "actual": y_test.to_numpy()
    }

    # ========================================================
    # TRAIN EACH MODEL
    # ========================================================

    for index, (model_name, model) in enumerate(models.items(), start=1):

        print(
            f"\n[{index}/4] Training "
            f"{model_name.replace('_', ' ').title()}..."
        )

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        if model_name == "mlp":

            print("  Scaling features using training data only...")

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(
                X_train
            )

            X_test_scaled = scaler.transform(
                X_test
            )

            model.fit(
                X_train_scaled,
                y_train
            )

            probabilities_model = model.predict_proba(
                X_test_scaled
            )[:, 1]

            joblib.dump(
                scaler,
                MODEL_DIR / f"{dataset}_mlp_scaler.joblib"
            )

        # ----------------------------------------------------
        # TREE MODELS
        # ----------------------------------------------------

        else:

            model.fit(
                X_train,
                y_train
            )

            probabilities_model = model.predict_proba(
                X_test
            )[:, 1]

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        model_path = MODEL_DIR / f"{dataset}_{model_name}.joblib"

        joblib.dump(
            model,
            model_path
        )

        # ----------------------------------------------------
        # SAVE PROBABILITIES
        # ----------------------------------------------------

        probabilities[
            f"{model_name}_probability"
        ] = probabilities_model

        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        metrics = evaluate_model(
            dataset,
            model_name,
            y_test,
            probabilities_model
        )

        results.append(metrics)

        print(
            f"  AUC       : {metrics['auc']:.4f}\n"
            f"  Accuracy  : {metrics['accuracy']:.4f}\n"
            f"  Precision : {metrics['precision']:.4f}\n"
            f"  Recall    : {metrics['recall']:.4f}\n"
            f"  F1        : {metrics['f1']:.4f}"
        )

        print(
            f"  Saved     : {model_path.name}"
        )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions_df = pd.DataFrame(probabilities)

    predictions_path = (
        RESULT_DIR /
        f"{dataset}_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_df = pd.DataFrame(results)

    metrics_path = (
        RESULT_DIR /
        f"{dataset}_model_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "-" * 70)
    print(f"{dataset.upper()} MODEL PERFORMANCE")
    print("-" * 70)

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(f"  {predictions_path}")
    print(f"  {metrics_path}")

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    del train_df
    del test_df
    del X_train
    del X_test
    del y_train
    del y_test
    del models

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("XAI-STABILITYNET")
    print("PHASE 2 — CROSS-DATASET MODEL TRAINING")
    print("=" * 70)

    all_results = []

    # --------------------------------------------------------
    # Train the SAME model families independently
    # on every dataset.
    # --------------------------------------------------------

    for dataset in DATASETS:

        dataset_results = train_dataset(
            dataset
        )

        all_results.extend(
            dataset_results
        )

    # ========================================================
    # COMBINED RESULTS
    # ========================================================

    if not all_results:

        print("\nNo processed datasets found.")
        print("Run Phase 1 first.")
        return

    combined_df = pd.DataFrame(
        all_results
    )

    combined_path = (
        RESULT_DIR /
        "phase2_all_model_metrics.csv"
    )

    combined_df.to_csv(
        combined_path,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("PHASE 2 — FINAL CROSS-DATASET RESULTS")
    print("=" * 70)

    print(
        combined_df.to_string(
            index=False
        )
    )

    print("\n")
    print(f"Combined results saved to:")
    print(f"  {combined_path}")

    print("\n")
    print("=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)

    print(
        "\nNext phase:"
        "\n  SHAP + LIME explanations"
        "\n  perturbation / robustness testing"
        "\n  explanation stability scoring"
        "\n  cross-model comparison"
        "\n  cross-dataset comparison"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()