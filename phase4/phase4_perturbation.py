"""
XAI-STABILITYNET
PHASE 4 — PERTURBATION / ROBUSTNESS ANALYSIS

Purpose:
    Create controlled perturbations of the SAME 100 explanation instances
    used in Phase 3 and measure prediction robustness across:

        3 datasets × 4 models × multiple perturbation types

Important:
    - Original test instances are never modified.
    - Perturbations are generated from the saved Phase 3 instances.
    - No retraining occurs.
    - Models are loaded from Phase 2.
    - MLP uses its saved scaler.
    - Original and perturbed probabilities are preserved.
    - Outputs are saved for Phase 5 stability scoring.

Run:
    python phase4_perturbation.py
"""

from pathlib import Path
import gc
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
PHASE3_DIR = ROOT / "results" / "phase3"
RESULT_DIR = ROOT / "results" / "phase4"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["german", "taiwan", "gmsc"]

MODELS = [
    "xgboost",
    "lightgbm",
    "random_forest",
    "mlp",
]

TARGET = "Default"
SEED = 42


# ============================================================
# PERTURBATION CONFIGURATION
# ============================================================

# Continuous variables are perturbed by small realistic relative changes.
# Categorical / one-hot variables are handled separately.

NUMERIC_PERTURBATIONS = {
    "relative_minus_5": -0.05,
    "relative_plus_5": 0.05,
    "relative_minus_10": -0.10,
    "relative_plus_10": 0.10,
}


# ============================================================
# DATASET-SPECIFIC FEATURE GROUPS
# ============================================================

# These groups identify continuous variables that are meaningful to
# perturb. Features not listed here are left unchanged.
#
# The names correspond to the Phase 1 encoded datasets.

PERTURBABLE_FEATURES = {

    "german": [
        "duration",
        "credit_amount",
        "installment_rate",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents",
    ],

    "taiwan": [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ],

    "gmsc": [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ],
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(dataset, model_name):

    model_path = MODEL_DIR / f"{dataset}_{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing trained model:\n{model_path}"
        )

    model = joblib.load(model_path)

    scaler = None

    if model_name == "mlp":

        scaler_path = MODEL_DIR / f"{dataset}_mlp_scaler.joblib"

        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Missing MLP scaler:\n{scaler_path}"
            )

        scaler = joblib.load(scaler_path)

    return model, scaler


# ============================================================
# PREDICTION
# ============================================================

def predict_probability(model, scaler, X, model_name):

    if model_name == "mlp":

        X_input = scaler.transform(X)

    else:

        X_input = X

    return model.predict_proba(X_input)[:, 1]


# ============================================================
# CLIP VALUES
# ============================================================

def clip_feature(dataset, feature, values, original):

    """
    Keep perturbed values within sensible dataset-specific bounds.

    This prevents artificial perturbations such as negative age,
    negative income, etc.
    """

    values = values.copy()

    if feature == "age":
        values = np.clip(values, 18, 100)

    elif feature == "AGE":
        values = np.clip(values, 18, 100)

    elif feature == "duration":
        values = np.clip(values, 1, None)

    elif feature == "credit_amount":
        values = np.clip(values, 1, None)

    elif feature == "LIMIT_BAL":
        values = np.clip(values, 1, None)

    elif feature.startswith("BILL_AMT"):
        values = np.clip(values, 0, None)

    elif feature.startswith("PAY_AMT"):
        values = np.clip(values, 0, None)

    elif feature == "MonthlyIncome":
        values = np.clip(values, 0, None)

    elif feature == "NumberOfDependents":
        values = np.clip(values, 0, None)

    elif feature == "num_dependents":
        values = np.clip(values, 0, None)

    elif feature == "existing_credits":
        values = np.clip(values, 0, None)

    elif feature == "NumberOfOpenCreditLinesAndLoans":
        values = np.clip(values, 0, None)

    elif feature == "NumberRealEstateLoansOrLines":
        values = np.clip(values, 0, None)

    elif feature in [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]:
        values = np.clip(values, 0, None)

    elif feature in [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]:
        values = np.clip(values, -2, 8)

    return values


# ============================================================
# CREATE PERTURBATION
# ============================================================

def perturb_numeric(
    X,
    dataset,
    feature,
    perturbation_name,
    magnitude,
):

    X_new = X.copy()

    if feature not in X_new.columns:
        return None

    original = X_new[feature].to_numpy(dtype=float)

    # Relative perturbation.
    #
    # For values near zero, use a small absolute amount based on
    # the feature's training-set scale instead of producing no change.

    scale = np.nanmedian(np.abs(original))

    if not np.isfinite(scale) or scale == 0:
        scale = 1.0

    delta = np.where(
        np.abs(original) > 1e-12,
        np.abs(original) * magnitude,
        scale * magnitude,
    )

    if magnitude > 0:
        perturbed = original + delta
    else:
        perturbed = original - delta

    perturbed = clip_feature(
        dataset,
        feature,
        perturbed,
        original,
    )

    X_new[feature] = perturbed

    return X_new


# ============================================================
# RUN ONE DATASET / MODEL
# ============================================================

def run_dataset_model(dataset, model_name):

    print("\n" + "-" * 70)
    print(f"{dataset.upper()} — {model_name.upper()}")
    print("-" * 70)

    instance_path = PHASE3_DIR / f"{dataset}_explanation_instances.csv"

    if not instance_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 3 explanation instances:\n{instance_path}"
        )

    instances = pd.read_csv(instance_path)

    # Remove target if it happens to be present.
    if TARGET in instances.columns:
        y_original = instances[TARGET].copy()
        X = instances.drop(columns=[TARGET]).copy()
    else:
        y_original = None
        X = instances.copy()

    model, scaler = load_model(dataset, model_name)

    # Ensure exact feature order used during training.
    feature_path = MODEL_DIR / f"{dataset}_feature_columns.joblib"

    if feature_path.exists():

        feature_columns = joblib.load(feature_path)

        missing = [
            c for c in feature_columns
            if c not in X.columns
        ]

        if missing:
            raise ValueError(
                f"{dataset}: missing feature columns: {missing}"
            )

        X = X[feature_columns]

    print(f"Samples   : {len(X)}")
    print(f"Features  : {X.shape[1]}")

    # --------------------------------------------------------
    # ORIGINAL PREDICTIONS
    # --------------------------------------------------------

    original_probability = predict_probability(
        model,
        scaler,
        X,
        model_name,
    )

    original_prediction = (
        original_probability >= 0.5
    ).astype(int)

    rows = []

    perturbation_count = 0

    # --------------------------------------------------------
    # PERTURB EACH FEATURE
    # --------------------------------------------------------

    for feature in PERTURBABLE_FEATURES[dataset]:

        if feature not in X.columns:
            continue

        # Only perturb numerical columns.
        if not pd.api.types.is_numeric_dtype(X[feature]):
            continue

        for perturbation_name, magnitude in NUMERIC_PERTURBATIONS.items():

            X_perturbed = perturb_numeric(
                X,
                dataset,
                feature,
                perturbation_name,
                magnitude,
            )

            if X_perturbed is None:
                continue

            perturbed_probability = predict_probability(
                model,
                scaler,
                X_perturbed,
                model_name,
            )

            perturbed_prediction = (
                perturbed_probability >= 0.5
            ).astype(int)

            probability_change = (
                perturbed_probability -
                original_probability
            )

            prediction_changed = (
                perturbed_prediction !=
                original_prediction
            )

            for i in range(len(X)):

                rows.append({
                    "dataset": dataset,
                    "model": model_name,
                    "instance_id": i,
                    "feature": feature,
                    "perturbation": perturbation_name,
                    "magnitude": magnitude,

                    "original_value": float(
                        X.iloc[i][feature]
                    ),

                    "perturbed_value": float(
                        X_perturbed.iloc[i][feature]
                    ),

                    "original_probability": float(
                        original_probability[i]
                    ),

                    "perturbed_probability": float(
                        perturbed_probability[i]
                    ),

                    "probability_change": float(
                        probability_change[i]
                    ),

                    "absolute_probability_change": float(
                        abs(probability_change[i])
                    ),

                    "original_prediction": int(
                        original_prediction[i]
                    ),

                    "perturbed_prediction": int(
                        perturbed_prediction[i]
                    ),

                    "prediction_changed": int(
                        prediction_changed[i]
                    ),

                    "true_label": (
                        int(y_original.iloc[i])
                        if y_original is not None
                        else -1
                    ),
                })

            perturbation_count += 1

    result = pd.DataFrame(rows)

    output_path = (
        RESULT_DIR /
        f"{dataset}_{model_name}_perturbations.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    prediction_flip_rate = (
        result["prediction_changed"].mean()
        if len(result)
        else np.nan
    )

    mean_probability_change = (
        result["absolute_probability_change"].mean()
        if len(result)
        else np.nan
    )

    summary = pd.DataFrame([{
        "dataset": dataset,
        "model": model_name,
        "n_instances": len(X),
        "n_features_perturbed": result["feature"].nunique(),
        "n_perturbation_types": result["perturbation"].nunique(),
        "total_perturbations": perturbation_count,
        "mean_absolute_probability_change":
            mean_probability_change,
        "prediction_flip_rate":
            prediction_flip_rate,
    }])

    summary_path = (
        RESULT_DIR /
        f"{dataset}_{model_name}_robustness_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"Perturbation rows : {len(result):,}"
    )

    print(
        f"Mean |Δ probability| : "
        f"{mean_probability_change:.6f}"
    )

    print(
        f"Prediction flip rate : "
        f"{prediction_flip_rate:.4%}"
    )

    print(f"Saved: {output_path.name}")

    del model, scaler, X, instances, result
    gc.collect()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("XAI-STABILITYNET")
    print("PHASE 4 — PERTURBATION / ROBUSTNESS ANALYSIS")
    print("=" * 70)

    print("\nDatasets       :", len(DATASETS))
    print("Models         :", len(MODELS))
    print(
        "Perturbations  :",
        len(NUMERIC_PERTURBATIONS)
    )

    print(
        "\nExpected model runs:",
        len(DATASETS) * len(MODELS)
    )

    summaries = []

    for dataset in DATASETS:

        for model_name in MODELS:

            run_dataset_model(
                dataset,
                model_name,
            )

            summary_path = (
                RESULT_DIR /
                f"{dataset}_{model_name}_robustness_summary.csv"
            )

            if summary_path.exists():

                summary = pd.read_csv(
                    summary_path
                )

                summaries.append(summary)

    # --------------------------------------------------------
    # COMBINED SUMMARY
    # --------------------------------------------------------

    if summaries:

        combined = pd.concat(
            summaries,
            ignore_index=True,
        )

        combined_path = (
            RESULT_DIR /
            "phase4_robustness_summary.csv"
        )

        combined.to_csv(
            combined_path,
            index=False,
        )

        print("\n" + "=" * 70)
        print("PHASE 4 — ROBUSTNESS SUMMARY")
        print("=" * 70)

        print(
            combined.to_string(index=False)
        )

        print(
            f"\nSaved combined summary:"
            f"\n  {combined_path}"
        )

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)

    print(
        "\nNext:"
        "\n  Phase 5 — Explanation Stability Scoring"
        "\n  SHAP/LIME stability"
        "\n  Cross-model comparison"
        "\n  Cross-dataset comparison"
    )


if __name__ == "__main__":
    main()