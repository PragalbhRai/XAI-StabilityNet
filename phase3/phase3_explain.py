"""
XAI-StabilityNet
PHASE 3 — SHAP + LIME EXPLANATIONS

For each dataset:
    German / Taiwan / GMSC

For each model:
    XGBoost / LightGBM / Random Forest / MLP

Uses the EXACT held-out test set from Phase 1.
No model is retrained.

Outputs:
    results/phase3/
        <dataset>_<model>_shap.csv
        <dataset>_<model>_lime.csv

SHAP:
    TreeExplainer for tree models
    KernelExplainer for MLP

LIME:
    LimeTabularExplainer for all models

A fixed number of test samples is explained so that Phase 3
remains computationally practical while using identical instances
across all models within each dataset.
"""

from pathlib import Path
import gc
import warnings

import joblib
import numpy as np
import pandas as pd
import shap

from lime.lime_tabular import LimeTabularExplainer


# Stage 1.5 fix I-7: Replaced broad warnings.filterwarnings("ignore") with
# targeted suppressions. Only suppress specific known non-actionable messages
# from third-party libraries. All other warnings are visible.
warnings.filterwarnings(
    "ignore",
    message=".*LightGBM.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*No further splits with positive gain.*",
    category=UserWarning,
)
# Note: SHAP KernelExplainer nsamples=100 convergence warnings are NOT
# suppressed so they remain visible during future experiments.


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results" / "phase3"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

DATASETS = [
    "german",
    "taiwan",
    "gmsc",
]

MODELS = [
    "xgboost",
    "lightgbm",
    "random_forest",
    "mlp",
]

TARGET = "Default"

SEED = 42

# Number of test instances explained per dataset.
# Same instances are used for every model.
N_EXPLANATIONS = 100

# LIME perturbation samples.
LIME_NUM_SAMPLES = 1000

# Kernel SHAP background size.
SHAP_BACKGROUND_SIZE = 50


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(dataset, model_name):
    path = MODEL_DIR / f"{dataset}_{model_name}.joblib"

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{path}\n"
            f"Run Phase 2 first."
        )

    return joblib.load(path)


def load_scaler(dataset):
    path = MODEL_DIR / f"{dataset}_mlp_scaler.joblib"

    if not path.exists():
        raise FileNotFoundError(
            f"MLP scaler not found:\n{path}"
        )

    return joblib.load(path)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def make_probability_function(model, model_name, scaler=None):
    """
    Returns a function that accepts a numpy matrix and returns
    probability of Default=1.
    """

    if model_name == "mlp":

        def predict_fn(x):
            x = np.asarray(x, dtype=np.float32)
            x_scaled = scaler.transform(x)
            return model.predict_proba(x_scaled)[:, 1]

    else:

        def predict_fn(x):
            x = np.asarray(x, dtype=np.float32)
            return model.predict_proba(x)[:, 1]

    return predict_fn


# ============================================================
# SHAP
# ============================================================

def compute_tree_shap(model, X):
    """
    SHAP values for tree models.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # Binary classification output differs between SHAP versions.
    if isinstance(shap_values, list):
        values = shap_values[1]
    else:
        values = np.asarray(shap_values)

        # Some SHAP versions return:
        # (samples, features, classes)
        if values.ndim == 3:
            values = values[:, :, 1]

    return np.asarray(values)


def compute_kernel_shap(model, model_name, X_train, X_explain, scaler):
    """
    Kernel SHAP for MLP.

    Uses a small training background because KernelExplainer
    is computationally expensive.
    """

    predict_fn = make_probability_function(
        model,
        model_name,
        scaler,
    )

    rng = np.random.default_rng(SEED)

    background_size = min(
        SHAP_BACKGROUND_SIZE,
        len(X_train),
    )

    background_indices = rng.choice(
        len(X_train),
        size=background_size,
        replace=False,
    )

    background = X_train.iloc[background_indices].to_numpy(
        dtype=np.float32
    )

    explainer = shap.KernelExplainer(
        predict_fn,
        background,
    )

    values = explainer.shap_values(
        X_explain.to_numpy(dtype=np.float32),
        nsamples=100,
    )

    if isinstance(values, list):
        values = values[0]

    return np.asarray(values)


# ============================================================
# LIME
# ============================================================

def compute_lime(
    model,
    model_name,
    X_train,
    X_explain,
    feature_names,
    scaler=None,
):
    """
    Generate LIME explanations for the selected test instances.

    We store a dense feature-by-feature explanation matrix so
    Phase 4 can compare explanations directly.
    """

    if model_name == "mlp":

        predict_fn = lambda x: np.column_stack([
            1 - model.predict_proba(
                scaler.transform(
                    np.asarray(x, dtype=np.float32)
                )
            )[:, 1],
            model.predict_proba(
                scaler.transform(
                    np.asarray(x, dtype=np.float32)
                )
            )[:, 1],
        ])

    else:

        predict_fn = lambda x: np.column_stack([
            1 - model.predict_proba(
                np.asarray(x, dtype=np.float32)
            )[:, 1],
            model.predict_proba(
                np.asarray(x, dtype=np.float32)
            )[:, 1],
        ])

    explainer = LimeTabularExplainer(
        X_train.to_numpy(dtype=np.float32),
        feature_names=feature_names,
        class_names=["Good", "Default"],
        mode="classification",
        discretize_continuous=False,
        random_state=SEED,
    )

    lime_matrix = np.zeros(
        (len(X_explain), len(feature_names)),
        dtype=np.float32,
    )

    for i, row in enumerate(
        X_explain.to_numpy(dtype=np.float32)
    ):

        explanation = explainer.explain_instance(
            row,
            predict_fn,
            labels=(1,),
            num_features=len(feature_names),
            num_samples=LIME_NUM_SAMPLES,
        )

        feature_weights = dict(
            explanation.as_map()[1]
        )

        for feature_index, weight in feature_weights.items():
            lime_matrix[i, feature_index] = weight

        if (i + 1) % 10 == 0 or i == len(X_explain) - 1:
            print(
                f"      LIME: {i + 1}/{len(X_explain)}"
            )

    return lime_matrix


# ============================================================
# SAVE EXPLANATIONS
# ============================================================

def save_explanation(
    values,
    X_explain,
    y_explain,
    feature_names,
    dataset,
    model,
    explainer_name,
):

    explanation_df = pd.DataFrame(
        values,
        columns=feature_names,
    )

    explanation_df.insert(
        0,
        "instance_index",
        X_explain.index.to_numpy(),
    )

    explanation_df.insert(
        1,
        "actual_default",
        y_explain.to_numpy(),
    )

    explanation_df.insert(
        2,
        "dataset",
        dataset,
    )

    explanation_df.insert(
        3,
        "model",
        model,
    )

    explanation_df.insert(
        4,
        "explainer",
        explainer_name,
    )

    path = (
        RESULT_DIR
        / f"{dataset}_{model}_{explainer_name}.csv"
    )

    explanation_df.to_csv(
        path,
        index=False,
    )

    return path


# ============================================================
# ONE DATASET
# ============================================================

def process_dataset(dataset):
    print("\n" + "=" * 70)
    print(f"XAI-STABILITYNET — PHASE 3 — {dataset.upper()}")
    print("=" * 70)

    train_path = (
        PROCESSED_DIR
        / f"{dataset}_train.parquet"
    )

    test_path = (
        PROCESSED_DIR
        / f"{dataset}_test.parquet"
    )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    feature_names = list(X_train.columns)

    # --------------------------------------------------------
    # FIXED TEST INSTANCES
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)

    n = min(
        N_EXPLANATIONS,
        len(X_test),
    )

    selected_indices = rng.choice(
        len(X_test),
        size=n,
        replace=False,
    )

    # Sort so every model receives exactly the same ordering.
    selected_indices = np.sort(selected_indices)

    X_explain = X_test.iloc[selected_indices].copy()
    y_explain = y_test.iloc[selected_indices].copy()

    print(f"\nTraining samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Features         : {len(feature_names)}")
    print(f"Explained samples: {len(X_explain)}")

    # Save selected instances once.
    instance_df = X_explain.copy()

    instance_df.insert(
        0,
        "instance_index",
        X_explain.index.to_numpy(),
    )

    instance_df["actual_default"] = (
        y_explain.to_numpy()
    )

    instance_path = (
        RESULT_DIR
        / f"{dataset}_explanation_instances.csv"
    )

    instance_df.to_csv(
        instance_path,
        index=False,
    )

    print(
        f"\nSaved common explanation instances:"
        f"\n  {instance_path.name}"
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    for model_name in MODELS:

        print("\n" + "-" * 60)
        print(
            f"{dataset.upper()} — "
            f"{model_name.upper()}"
        )
        print("-" * 60)

        model = load_model(
            dataset,
            model_name,
        )

        scaler = None

        if model_name == "mlp":
            scaler = load_scaler(dataset)

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        print("\n  [1/2] SHAP...")

        if model_name in [
            "xgboost",
            "lightgbm",
            "random_forest",
        ]:

            shap_values = compute_tree_shap(
                model,
                X_explain,
            )

        else:

            shap_values = compute_kernel_shap(
                model,
                model_name,
                X_train,
                X_explain,
                scaler,
            )

        shap_path = save_explanation(
            shap_values,
            X_explain,
            y_explain,
            feature_names,
            dataset,
            model_name,
            "shap",
        )

        print(
            f"      Saved: {shap_path.name}"
        )

        del shap_values
        gc.collect()

        # ----------------------------------------------------
        # LIME
        # ----------------------------------------------------

        print("\n  [2/2] LIME...")

        lime_values = compute_lime(
            model,
            model_name,
            X_train,
            X_explain,
            feature_names,
            scaler,
        )

        lime_path = save_explanation(
            lime_values,
            X_explain,
            y_explain,
            feature_names,
            dataset,
            model_name,
            "lime",
        )

        print(
            f"      Saved: {lime_path.name}"
        )

        del lime_values
        del model
        gc.collect()

    del train_df
    del test_df
    del X_train
    del y_train
    del X_test
    del y_test
    del X_explain
    del y_explain

    gc.collect()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("XAI-STABILITYNET")
    print("PHASE 3 — SHAP + LIME EXPLANATIONS")
    print("=" * 70)

    print(
        f"\nDatasets       : {len(DATASETS)}"
        f"\nModels         : {len(MODELS)}"
        f"\nExplainers     : 2"
        f"\nSamples/model  : {N_EXPLANATIONS}"
    )

    print(
        "\nTotal explanation runs:"
        f" {len(DATASETS) * len(MODELS) * 2}"
    )

    for dataset in DATASETS:
        process_dataset(dataset)

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)

    print("\nExplanation files saved in:")
    print(f"  {RESULT_DIR}")

    print("\nExpected:")
    print("  12 SHAP explanation files")
    print("  12 LIME explanation files")
    print("  3 common-instance files")

    print("\nNext:")
    print("  Phase 4 — Perturbation / Robustness")
    print("  Phase 5 — Explanation Stability Scoring")


if __name__ == "__main__":
    main()