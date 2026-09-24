from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# XAI-STABILITYNET
# PHASE 5 — EXPLANATION STABILITY SCORING
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PHASE3_DIR = ROOT / "results" / "phase3"
PHASE4_DIR = ROOT / "results" / "phase4"
RESULT_DIR = ROOT / "results" / "phase5"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["german", "taiwan", "gmsc"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]
EXPLAINERS = ["shap", "lime"]

SEED = 42


# ============================================================
# FEATURE / CATEGORY MAPPINGS
#
# These are semantic groups, NOT claims that raw features are
# identical across datasets.
# ============================================================

CATEGORY_MAP = {

    "german": {
        "duration": "loan_structure",
        "credit_amount": "loan_structure",
        "installment_rate": "loan_structure",

        "residence_since": "demographics",
        "age": "demographics",
        "num_dependents": "demographics",

        "existing_credits": "credit_history",

        "checking_status_A12": "financial_resources",
        "checking_status_A13": "financial_resources",
        "checking_status_A14": "financial_resources",

        "savings_status_A62": "financial_resources",
        "savings_status_A63": "financial_resources",
        "savings_status_A64": "financial_resources",
        "savings_status_A65": "financial_resources",

        "credit_history_A31": "credit_history",
        "credit_history_A32": "credit_history",
        "credit_history_A33": "credit_history",
        "credit_history_A34": "credit_history",

        "purpose_A41": "loan_purpose",
        "purpose_A410": "loan_purpose",
        "purpose_A42": "loan_purpose",
        "purpose_A43": "loan_purpose",
        "purpose_A44": "loan_purpose",
        "purpose_A45": "loan_purpose",
        "purpose_A46": "loan_purpose",
        "purpose_A48": "loan_purpose",
        "purpose_A49": "loan_purpose",

        "employment_A72": "employment",
        "employment_A73": "employment",
        "employment_A74": "employment",
        "employment_A75": "employment",

        "personal_status_A92": "demographics",
        "personal_status_A93": "demographics",
        "personal_status_A94": "demographics",

        "other_parties_A102": "household_support",
        "other_parties_A103": "household_support",

        "property_magnitude_A122": "assets",
        "property_magnitude_A123": "assets",
        "property_magnitude_A124": "assets",

        "other_payment_plans_A142": "financial_resources",
        "other_payment_plans_A143": "financial_resources",

        "housing_A152": "housing",
        "housing_A153": "housing",

        "job_A172": "employment",
        "job_A173": "employment",
        "job_A174": "employment",

        "own_telephone_A192": "contact_information",
        "foreign_worker_A202": "demographics",
    },

    "taiwan": {
        "LIMIT_BAL": "financial_resources",
        "SEX": "demographics",
        "EDUCATION": "demographics",
        "MARRIAGE": "demographics",
        "AGE": "demographics",

        "PAY_0": "payment_history",
        "PAY_2": "payment_history",
        "PAY_3": "payment_history",
        "PAY_4": "payment_history",
        "PAY_5": "payment_history",
        "PAY_6": "payment_history",

        "BILL_AMT1": "bill_history",
        "BILL_AMT2": "bill_history",
        "BILL_AMT3": "bill_history",
        "BILL_AMT4": "bill_history",
        "BILL_AMT5": "bill_history",
        "BILL_AMT6": "bill_history",

        "PAY_AMT1": "payment_amount",
        "PAY_AMT2": "payment_amount",
        "PAY_AMT3": "payment_amount",
        "PAY_AMT4": "payment_amount",
        "PAY_AMT5": "payment_amount",
        "PAY_AMT6": "payment_amount",
    },

    "gmsc": {
        "RevolvingUtilizationOfUnsecuredLines": "credit_utilization",
        "age": "demographics",
        "NumberOfTime30-59DaysPastDueNotWorse": "payment_history",
        "DebtRatio": "financial_resources",
        "MonthlyIncome": "financial_resources",
        "NumberOfOpenCreditLinesAndLoans": "credit_history",
        "NumberOfTimes90DaysLate": "payment_history",
        "NumberRealEstateLoansOrLines": "assets",
        "NumberOfTime60-89DaysPastDueNotWorse": "payment_history",
        "NumberOfDependents": "demographics",
    },
}


# ============================================================
# HELPERS
# ============================================================

def load_explanations(dataset, model, explainer):
    path = PHASE3_DIR / f"{dataset}_{model}_{explainer}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing explanation file: {path}")

    return pd.read_csv(path)


def get_feature_columns(df):
    metadata_cols = {
        "instance_index",
        "actual_default",
        "dataset",
        "model",
        "explainer",
    }

    return [
        c for c in df.columns
        if c not in metadata_cols
    ]


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    value = np.corrcoef(a, b)[0, 1]

    if np.isnan(value):
        return 0.0

    return float(value)


def rank_agreement(shap_values, lime_values):
    """
    Spearman-like rank agreement calculated from absolute
    explanation magnitudes.

    Measures whether SHAP and LIME identify similar important
    features, regardless of their raw explanation scale.
    """

    shap_rank = pd.Series(
        np.abs(shap_values)
    ).rank(method="average")

    lime_rank = pd.Series(
        np.abs(lime_values)
    ).rank(method="average")

    corr = safe_corr(shap_rank, lime_rank)

    # Convert [-1,1] -> [0,1]
    return (corr + 1.0) / 2.0


def sign_agreement(shap_values, lime_values):
    """
    Fraction of features for which SHAP and LIME agree on
    direction of influence.

    Features with zero contribution are ignored.
    """

    shap_values = np.asarray(shap_values, dtype=float)
    lime_values = np.asarray(lime_values, dtype=float)

    mask = (np.abs(shap_values) > 1e-12) & (
        np.abs(lime_values) > 1e-12
    )

    if mask.sum() == 0:
        return 0.0

    same_sign = (
        np.sign(shap_values[mask])
        == np.sign(lime_values[mask])
    )

    return float(np.mean(same_sign))


def top_k_overlap(shap_values, lime_values, k=10):
    """
    Jaccard overlap of the top-k most important features.
    """

    n = len(shap_values)
    k = min(k, n)

    shap_top = set(
        np.argsort(np.abs(shap_values))[-k:]
    )

    lime_top = set(
        np.argsort(np.abs(lime_values))[-k:]
    )

    union = shap_top | lime_top

    if not union:
        return 0.0

    return len(shap_top & lime_top) / len(union)


# ============================================================
# EXPLANER AGREEMENT
# ============================================================

def calculate_explainer_agreement(dataset, model):

    shap_df = load_explanations(dataset, model, "shap")
    lime_df = load_explanations(dataset, model, "lime")

    feature_columns = [
        c for c in get_feature_columns(shap_df)
        if c in lime_df.columns
    ]

    # Ensure identical instance ordering
    shap_df = shap_df.sort_values("instance_index").reset_index(drop=True)
    lime_df = lime_df.sort_values("instance_index").reset_index(drop=True)

    if not np.array_equal(
        shap_df["instance_index"].to_numpy(),
        lime_df["instance_index"].to_numpy()
    ):
        raise ValueError(
            f"Instance mismatch between SHAP and LIME: "
            f"{dataset} / {model}"
        )

    rows = []

    for i in range(len(shap_df)):

        shap_values = shap_df.loc[i, feature_columns].astype(float).to_numpy()
        lime_values = lime_df.loc[i, feature_columns].astype(float).to_numpy()

        correlation = safe_corr(shap_values, lime_values)

        rank_score = rank_agreement(
            shap_values,
            lime_values
        )

        sign_score = sign_agreement(
            shap_values,
            lime_values
        )

        top10_score = top_k_overlap(
            shap_values,
            lime_values,
            k=10
        )

        # Explanation Agreement Score
        agreement_score = (
            0.40 * rank_score
            + 0.30 * sign_score
            + 0.30 * top10_score
        )

        rows.append({
            "dataset": dataset,
            "model": model,
            "instance_index": int(
                shap_df.loc[i, "instance_index"]
            ),
            "shap_lime_correlation": correlation,
            "rank_agreement": rank_score,
            "sign_agreement": sign_score,
            "top10_overlap": top10_score,
            "explanation_agreement_score": agreement_score,
        })

    return pd.DataFrame(rows)


# ============================================================
# CATEGORY-LEVEL AGREEMENT
# ============================================================

def aggregate_categories(df, dataset):

    mapping = CATEGORY_MAP[dataset]

    available_features = [
        c for c in df.columns
        if c in mapping
    ]

    categories = sorted(
        set(mapping[c] for c in available_features)
    )

    result = pd.DataFrame(
        index=df.index,
        columns=categories,
        dtype=float
    )

    for category in categories:

        features = [
            c for c in available_features
            if mapping[c] == category
        ]

        # Sum signed contributions within the semantic category.
        result[category] = df[features].sum(axis=1)

    return result


def calculate_category_agreement(dataset, model):

    shap_df = load_explanations(dataset, model, "shap")
    lime_df = load_explanations(dataset, model, "lime")

    shap_cat = aggregate_categories(
        shap_df,
        dataset
    )

    lime_cat = aggregate_categories(
        lime_df,
        dataset
    )

    rows = []

    for i in range(len(shap_cat)):

        shap_values = shap_cat.iloc[i].to_numpy()
        lime_values = lime_cat.iloc[i].to_numpy()

        correlation = safe_corr(
            shap_values,
            lime_values
        )

        rank_score = rank_agreement(
            shap_values,
            lime_values
        )

        sign_score = sign_agreement(
            shap_values,
            lime_values
        )

        rows.append({
            "dataset": dataset,
            "model": model,
            "instance_index": int(
                shap_df.iloc[i]["instance_index"]
            ),
            "category_correlation": correlation,
            "category_rank_agreement": rank_score,
            "category_sign_agreement": sign_score,
        })

    return pd.DataFrame(rows)


# ============================================================
# ROBUSTNESS COMPONENT
# ============================================================

def load_robustness(dataset, model):

    path = (
        PHASE4_DIR
        / f"{dataset}_{model}_perturbations.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing perturbation file: {path}"
        )

    return pd.read_csv(path)


def calculate_robustness_score(dataset, model):

    df = load_robustness(dataset, model)

    mean_probability_change = (
        df["absolute_probability_change"]
        .mean()
    )

    flip_rate = (
        df["prediction_changed"]
        .mean()
    )

    # Bounded scores:
    #
    # probability stability:
    # 0 change -> 1.0
    # larger changes progressively reduce score
    #
    # prediction stability:
    # 0 flips -> 1.0
    # 100% flips -> 0.0

    probability_stability = 1.0 / (
        1.0 + mean_probability_change
    )

    prediction_stability = 1.0 - flip_rate

    robustness_score = (
        0.60 * probability_stability
        + 0.40 * prediction_stability
    )

    return {
        "dataset": dataset,
        "model": model,
        "mean_absolute_probability_change":
            mean_probability_change,
        "prediction_flip_rate":
            flip_rate,
        "probability_stability":
            probability_stability,
        "prediction_stability":
            prediction_stability,
        "robustness_score":
            robustness_score,
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("XAI-STABILITYNET")
print("PHASE 5 — EXPLANATION STABILITY SCORING")
print("=" * 70)

print("\nDatasets :", len(DATASETS))
print("Models   :", len(MODELS))
print("Explainers: SHAP + LIME")

agreement_results = []
category_results = []
robustness_results = []


# ------------------------------------------------------------
# Process all 12 model/dataset combinations
# ------------------------------------------------------------

for dataset in DATASETS:

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset.upper()}")
    print("=" * 70)

    for model in MODELS:

        print(f"\n{dataset.upper()} — {model.upper()}")

        # ----------------------------------------------------
        # SHAP / LIME agreement
        # ----------------------------------------------------

        agreement_df = calculate_explainer_agreement(
            dataset,
            model
        )

        agreement_results.append(
            agreement_df
        )

        print(
            f"  Explainer agreement : "
            f"{agreement_df['explanation_agreement_score'].mean():.4f}"
        )

        # ----------------------------------------------------
        # Category-level agreement
        # ----------------------------------------------------

        category_df = calculate_category_agreement(
            dataset,
            model
        )

        category_results.append(
            category_df
        )

        print(
            f"  Category agreement  : "
            f"{category_df['category_rank_agreement'].mean():.4f}"
        )

        # ----------------------------------------------------
        # Robustness
        # ----------------------------------------------------

        robustness = calculate_robustness_score(
            dataset,
            model
        )

        robustness_results.append(
            robustness
        )

        print(
            f"  Robustness score    : "
            f"{robustness['robustness_score']:.4f}"
        )


# ============================================================
# COMBINE RESULTS
# ============================================================

agreement_all = pd.concat(
    agreement_results,
    ignore_index=True
)

category_all = pd.concat(
    category_results,
    ignore_index=True
)

robustness_all = pd.DataFrame(
    robustness_results
)


# ============================================================
# AGGREGATE MODEL/DATASET SCORES
# ============================================================

agreement_summary = (
    agreement_all
    .groupby(["dataset", "model"])
    .agg(
        explanation_agreement_score=(
            "explanation_agreement_score",
            "mean"
        ),
        shap_lime_correlation=(
            "shap_lime_correlation",
            "mean"
        ),
        rank_agreement=(
            "rank_agreement",
            "mean"
        ),
        sign_agreement=(
            "sign_agreement",
            "mean"
        ),
        top10_overlap=(
            "top10_overlap",
            "mean"
        ),
    )
    .reset_index()
)


category_summary = (
    category_all
    .groupby(["dataset", "model"])
    .agg(
        category_correlation=(
            "category_correlation",
            "mean"
        ),
        category_rank_agreement=(
            "category_rank_agreement",
            "mean"
        ),
        category_sign_agreement=(
            "category_sign_agreement",
            "mean"
        ),
    )
    .reset_index()
)


summary = (
    agreement_summary
    .merge(
        category_summary,
        on=["dataset", "model"],
        how="left"
    )
    .merge(
        robustness_all,
        on=["dataset", "model"],
        how="left"
    )
)


# ============================================================
# FINAL EXPLANATION STABILITY SCORE
# ============================================================

# ESS is deliberately transparent:
#
# 40% explanation agreement
# 20% semantic/category agreement
# 40% perturbation robustness
#
# Explanation agreement itself combines:
#   40% ranking agreement
#   30% sign agreement
#   30% top-feature overlap
#
# Category agreement uses:
#   50% category rank agreement
#   50% category sign agreement

summary["category_stability_score"] = (
    0.50 * summary["category_rank_agreement"]
    + 0.50 * summary["category_sign_agreement"]
)

summary["explanation_stability_score"] = (
    0.40 * summary["explanation_agreement_score"]
    + 0.20 * summary["category_stability_score"]
    + 0.40 * summary["robustness_score"]
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

agreement_path = (
    RESULT_DIR
    / "phase5_instance_explainer_agreement.csv"
)

category_path = (
    RESULT_DIR
    / "phase5_instance_category_agreement.csv"
)

robustness_path = (
    RESULT_DIR
    / "phase5_robustness_scores.csv"
)

summary_path = (
    RESULT_DIR
    / "phase5_model_dataset_scores.csv"
)

agreement_all.to_csv(
    agreement_path,
    index=False
)

category_all.to_csv(
    category_path,
    index=False
)

robustness_all.to_csv(
    robustness_path,
    index=False
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# CROSS-MODEL ANALYSIS
# ============================================================

cross_model = (
    summary
    .groupby("model")
    .agg(
        mean_ess=(
            "explanation_stability_score",
            "mean"
        ),
        mean_explanation_agreement=(
            "explanation_agreement_score",
            "mean"
        ),
        mean_category_stability=(
            "category_stability_score",
            "mean"
        ),
        mean_robustness=(
            "robustness_score",
            "mean"
        ),
    )
    .reset_index()
    .sort_values(
        "mean_ess",
        ascending=False
    )
)

cross_model_path = (
    RESULT_DIR
    / "phase5_cross_model_comparison.csv"
)

cross_model.to_csv(
    cross_model_path,
    index=False
)


# ============================================================
# CROSS-DATASET ANALYSIS
# ============================================================

cross_dataset = (
    summary
    .groupby("dataset")
    .agg(
        mean_ess=(
            "explanation_stability_score",
            "mean"
        ),
        mean_explanation_agreement=(
            "explanation_agreement_score",
            "mean"
        ),
        mean_category_stability=(
            "category_stability_score",
            "mean"
        ),
        mean_robustness=(
            "robustness_score",
            "mean"
        ),
    )
    .reset_index()
    .sort_values(
        "mean_ess",
        ascending=False
    )
)

cross_dataset_path = (
    RESULT_DIR
    / "phase5_cross_dataset_comparison.csv"
)

cross_dataset.to_csv(
    cross_dataset_path,
    index=False
)


# ============================================================
# OVERALL RESULT
# ============================================================

overall = pd.DataFrame([{
    "overall_ess":
        summary["explanation_stability_score"].mean(),

    "overall_explanation_agreement":
        summary["explanation_agreement_score"].mean(),

    "overall_category_stability":
        summary["category_stability_score"].mean(),

    "overall_robustness":
        summary["robustness_score"].mean(),
}])

overall_path = (
    RESULT_DIR
    / "phase5_overall_score.csv"
)

overall.to_csv(
    overall_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5 — MODEL/DATASET STABILITY RESULTS")
print("=" * 70)

display_columns = [
    "dataset",
    "model",
    "explanation_agreement_score",
    "category_stability_score",
    "robustness_score",
    "explanation_stability_score",
]

print(
    summary[display_columns]
    .sort_values(
        "explanation_stability_score",
        ascending=False
    )
    .to_string(index=False)
)


print("\n" + "-" * 70)
print("CROSS-MODEL COMPARISON")
print("-" * 70)

print(
    cross_model.to_string(index=False)
)


print("\n" + "-" * 70)
print("CROSS-DATASET COMPARISON")
print("-" * 70)

print(
    cross_dataset.to_string(index=False)
)


print("\n" + "-" * 70)
print("OVERALL XAI-STABILITYNET SCORE")
print("-" * 70)

print(
    overall.to_string(index=False)
)


print("\n" + "=" * 70)
print("PHASE 5 COMPLETE")
print("=" * 70)

print("\nSaved:")
print(f"  {agreement_path}")
print(f"  {category_path}")
print(f"  {robustness_path}")
print(f"  {summary_path}")
print(f"  {cross_model_path}")
print(f"  {cross_dataset_path}")
print(f"  {overall_path}")