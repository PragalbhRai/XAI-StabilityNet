from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
METADATA_DIR = ROOT / "data" / "metadata"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "Default"


# ============================================================
# DATASET PATHS
# ============================================================

DATASETS = {
    "german": ROOT / "phase1" / "german.csv",
    "taiwan": ROOT / "UCI_credit_data" / "UCI_Credit_Card.csv",
    "gmsc": ROOT / "GiveMeSomeCredit" / "cs-training.csv",
}


# ============================================================
# LOADERS
# ============================================================

def load_german(path):

    columns = [
        "checking_status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_status",
        "employment",
        "installment_rate",
        "personal_status",
        "other_parties",
        "residence_since",
        "property_magnitude",
        "age",
        "other_payment_plans",
        "housing",
        "existing_credits",
        "job",
        "num_dependents",
        "own_telephone",
        "foreign_worker",
        "target",
    ]

    df = pd.read_csv(
        path,
        header=None,
        names=columns
    )

    # Original German dataset:
    # 1 = good credit
    # 2 = bad credit
    df[TARGET] = (
        df["target"] == 2
    ).astype(np.int8)

    df.drop(
        columns=["target"],
        inplace=True
    )

    return df


def load_taiwan(path):

    df = pd.read_csv(path)

    # ID is not a predictive feature.
    df.drop(
        columns=["ID"],
        inplace=True
    )

    df.rename(
        columns={
            "default.payment.next.month": TARGET
        },
        inplace=True
    )

    df[TARGET] = df[TARGET].astype(np.int8)

    return df


def load_gmsc(path):

    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df.drop(
            columns=["Unnamed: 0"],
            inplace=True
        )

    df.rename(
        columns={
            "SeriousDlqin2yrs": TARGET
        },
        inplace=True
    )

    df[TARGET] = df[TARGET].astype(np.int8)

    return df


LOADERS = {
    "german": load_german,
    "taiwan": load_taiwan,
    "gmsc": load_gmsc,
}


# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

def split_dataset(df):

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df[TARGET],
        random_state=SEED
    )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


# ============================================================
# MISSING VALUE HANDLING
# ============================================================

def fit_missing_value_rules(train_df):

    rules = {}

    for column in train_df.columns:

        if column == TARGET:
            continue

        if pd.api.types.is_numeric_dtype(
            train_df[column]
        ):

            rules[column] = {
                "type": "numeric",
                "value": float(
                    train_df[column].median()
                )
            }

        else:

            mode = train_df[column].mode()

            value = (
                mode.iloc[0]
                if len(mode) > 0
                else "Unknown"
            )

            rules[column] = {
                "type": "categorical",
                "value": str(value)
            }

    return rules


def apply_missing_value_rules(
    df,
    rules
):

    df = df.copy()

    for column, rule in rules.items():

        if rule["type"] == "numeric":

            df[column] = df[column].fillna(
                rule["value"]
            )

        else:

            df[column] = df[column].fillna(
                rule["value"]
            )

    return df


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

def fit_encoding_rules(train_df):

    categorical_columns = []

    category_values = {}

    for column in train_df.columns:

        if column == TARGET:
            continue

        if not pd.api.types.is_numeric_dtype(
            train_df[column]
        ):

            categorical_columns.append(column)

            values = sorted(
                train_df[column]
                .astype(str)
                .unique()
                .tolist()
            )

            category_values[column] = values

    return (
        categorical_columns,
        category_values
    )


def apply_encoding(
    df,
    categorical_columns,
    category_values
):

    df = df.copy()

    for column in categorical_columns:

        # Fixed categories learned from TRAIN only.
        dtype = pd.CategoricalDtype(
            categories=category_values[column]
        )

        df[column] = (
            df[column]
            .astype(str)
            .astype(dtype)
        )

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
        dtype=np.float32
    )

    return df


# ============================================================
# FEATURE MAPPING
# ============================================================

def build_feature_mapping(
    original_columns,
    encoded_columns,
    categorical_columns
):

    mapping = {}

    for original in original_columns:

        if original == TARGET:
            continue

        if original in categorical_columns:

            prefix = original + "_"

            encoded = [
                column
                for column in encoded_columns
                if column.startswith(prefix)
            ]

            mapping[original] = encoded

        else:

            mapping[original] = [original]

    return mapping


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(name):

    print("\n" + "=" * 70)
    print(f"XAI-STABILITYNET — PHASE 1 — {name.upper()}")
    print("=" * 70)

    path = DATASETS[name]

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = LOADERS[name](path)

    print(
        f"Raw dataset shape : {df.shape}"
    )

    print(
        f"Raw default rate  : {df[TARGET].mean():.4f}"
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    duplicates = df.duplicated().sum()

    if duplicates > 0:

        print(
            f"Duplicate rows removed: {duplicates}"
        )

        df = df.drop_duplicates(
            ignore_index=True
        )

    # --------------------------------------------------------
    # SPLIT FIRST
    #
    # IMPORTANT:
    # Everything learned from data happens AFTER
    # the train/test split.
    # --------------------------------------------------------

    train_df, test_df = split_dataset(df)

    print(
        f"Train rows          : {len(train_df)}"
    )

    print(
        f"Test rows           : {len(test_df)}"
    )

    # --------------------------------------------------------
    # ORIGINAL FEATURE INFORMATION
    # --------------------------------------------------------

    original_features = [
        column
        for column in train_df.columns
        if column != TARGET
    ]

    # --------------------------------------------------------
    # FIT MISSING-VALUE RULES ON TRAIN ONLY
    # --------------------------------------------------------

    missing_rules = fit_missing_value_rules(
        train_df
    )

    train_df = apply_missing_value_rules(
        train_df,
        missing_rules
    )

    test_df = apply_missing_value_rules(
        test_df,
        missing_rules
    )

    # --------------------------------------------------------
    # FIT ENCODING ON TRAIN ONLY
    # --------------------------------------------------------

    (
        categorical_columns,
        category_values
    ) = fit_encoding_rules(train_df)

    X_train = apply_encoding(
        train_df.drop(columns=[TARGET]),
        categorical_columns,
        category_values
    )

    X_test = apply_encoding(
        test_df.drop(columns=[TARGET]),
        categorical_columns,
        category_values
    )

    # --------------------------------------------------------
    # FORCE IDENTICAL FEATURE ORDER
    # --------------------------------------------------------

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # --------------------------------------------------------
    # NUMERIC DATA FOR ALL MODELS
    # --------------------------------------------------------

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_train = train_df[TARGET].astype(np.int8)
    y_test = test_df[TARGET].astype(np.int8)

    # --------------------------------------------------------
    # SANITY CHECKS
    # --------------------------------------------------------

    assert X_train.shape[1] == X_test.shape[1]

    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()

    assert y_train.isin([0, 1]).all()
    assert y_test.isin([0, 1]).all()

    # --------------------------------------------------------
    # SAVE PROCESSED DATA
    # --------------------------------------------------------

    train_processed = X_train.copy()
    train_processed[TARGET] = y_train.to_numpy()

    test_processed = X_test.copy()
    test_processed[TARGET] = y_test.to_numpy()

    train_path = (
        PROCESSED_DIR /
        f"{name}_train.parquet"
    )

    test_path = (
        PROCESSED_DIR /
        f"{name}_test.parquet"
    )

    train_processed.to_parquet(
        train_path,
        index=False
    )

    test_processed.to_parquet(
        test_path,
        index=False
    )

    # --------------------------------------------------------
    # FEATURE → ENCODED FEATURE MAPPING
    # --------------------------------------------------------

    feature_mapping = build_feature_mapping(
        original_features,
        list(X_train.columns),
        categorical_columns
    )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata = {

        "dataset": name,

        "target": TARGET,

        "seed": SEED,

        "original_features": original_features,

        "encoded_features": list(
            X_train.columns
        ),

        "categorical_features": (
            categorical_columns
        ),

        "feature_mapping": feature_mapping,

        "missing_value_rules": missing_rules,

        "category_values": category_values,

        "train_rows": len(X_train),

        "test_rows": len(X_test),

        "n_encoded_features": X_train.shape[1],

        "train_default_rate": float(
            y_train.mean()
        ),

        "test_default_rate": float(
            y_test.mean()
        )
    }

    metadata_path = (
        METADATA_DIR /
        f"{name}_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # SAVE TRAIN/TEST INDICES
    #
    # These let us prove that every model in Phase 2
    # sees exactly the same instances.
    # --------------------------------------------------------

    split_metadata = {

        "dataset": name,

        "seed": SEED,

        "train_original_indices": (
            train_df.index.tolist()
        ),

        "test_original_indices": (
            test_df.index.tolist()
        )
    }

    with open(
        METADATA_DIR /
        f"{name}_split.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            split_metadata,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print(
        f"\nProcessed train shape : {X_train.shape}"
    )

    print(
        f"Processed test shape  : {X_test.shape}"
    )

    print(
        f"Train default rate    : {y_train.mean():.4f}"
    )

    print(
        f"Test default rate     : {y_test.mean():.4f}"
    )

    print(
        f"Categorical features  : {len(categorical_columns)}"
    )

    print(
        f"Encoded features      : {X_train.shape[1]}"
    )

    print(
        f"\nSaved:"
    )

    print(
        f"  {train_path}"
    )

    print(
        f"  {test_path}"
    )

    print(
        f"  {metadata_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("XAI-STABILITYNET")
    print("PHASE 1 — LEAKAGE-SAFE PREPROCESSING")
    print("=" * 70)

    for dataset in [
        "german",
        "taiwan",
        "gmsc"
    ]:

        process_dataset(dataset)

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE — ALL THREE DATASETS")
    print("=" * 70)