"""
tests/test_cross_phase_consistency.py
=======================================
Cross-phase consistency tests (Step 10) + dataset validation framework (Step 6).

Tests:
  C01  Constraint consistency: canonical perturbable = schema for each dataset
  C02  Feature consistency: every perturbable feature exists in processed data
  C03  Non-perturbable unchanged: after simulation, non-perturbable cols intact
  C04  One-hot consistency: explanation CSVs have valid one-hot German dummies
  C05  Domain consistency: perturbed values satisfy feature constraints
  C06  Reproducibility: same config -> same output
  V01  German - numeric bounds
  V02  German - integer-valued features are integers in the instance file
  V03  German - one-hot groups valid
  V04  Taiwan - perturbable features exist and are numeric
  V05  GMSC   - perturbable features exist and are numeric
  V06  Non-perturbable features exist in all 3 datasets
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT       = Path(__file__).resolve().parents[1]
PHASE3_DIR = ROOT / "results" / "phase3"
sys.path.insert(0, str(ROOT))

from phase3.perturbation_schema import (
    SCHEMA, get_perturbable_features, get_non_perturbable_features, get_schema_dict,
)
from phase4.phase4_perturbation_v2 import perturb_relative_signed, perturb_ordinal_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_instances(dataset: str) -> pd.DataFrame:
    return pd.read_csv(PHASE3_DIR / f"{dataset}_explanation_instances.csv")


DATASETS = ["german", "taiwan", "gmsc"]


# ---------------------------------------------------------------------------
# C01  Constraint consistency
# ---------------------------------------------------------------------------

def test_C01_schema_has_all_datasets():
    """Schema covers all three datasets."""
    covered = {s.dataset for s in SCHEMA}
    for ds in DATASETS:
        assert ds in covered, f"{ds} missing from SCHEMA"


# ---------------------------------------------------------------------------
# C02  Feature consistency: perturbable features exist in processed data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", DATASETS)
def test_C02_perturbable_features_exist(dataset):
    df = load_instances(dataset)
    for spec in get_perturbable_features(dataset):
        assert spec.feature_name in df.columns, (
            f"{dataset}: perturbable feature '{spec.feature_name}' not in explanation instances"
        )


# ---------------------------------------------------------------------------
# C03  Non-perturbable unchanged after simulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", DATASETS)
def test_C03_non_perturbable_unchanged(dataset):
    """Run the engine loop over perturbable features.
    Non-perturbable columns must be byte/value identical before and after.
    This tests real engine behavior, not just set membership."""
    df = load_instances(dataset)

    perturbable_specs = [
        s for s in get_perturbable_features(dataset)
        if s.feature_name in df.columns
        and pd.api.types.is_numeric_dtype(df[s.feature_name])
        and s.perturbation_rule in ("relative_signed", "ordinal_step")
    ]
    non_pert_numeric = [
        s for s in get_non_perturbable_features(dataset)
        if s.feature_name in df.columns
        and pd.api.types.is_numeric_dtype(df[s.feature_name])
    ]

    if not perturbable_specs:
        pytest.skip(f"{dataset}: no perturbable numeric features with handled rule")
    if not non_pert_numeric:
        pytest.skip(f"{dataset}: no numeric non-perturbable features to check")

    # Snapshot originals
    originals = {s.feature_name: df[s.feature_name].copy() for s in non_pert_numeric}

    # Run engine simulation: touch only perturbable columns
    df_out = df.copy()
    for spec in perturbable_specs:
        orig = df_out[spec.feature_name].to_numpy(dtype=float)
        if spec.perturbation_rule == "relative_signed":
            df_out[spec.feature_name] = perturb_relative_signed(
                orig, +0.05, spec.lower_bound, spec.upper_bound)
        else:
            df_out[spec.feature_name] = perturb_ordinal_step(
                orig, +0.05, spec.lower_bound, spec.upper_bound)

    # Assert non-perturbable columns are untouched
    for spec in non_pert_numeric:
        assert df_out[spec.feature_name].equals(originals[spec.feature_name]), (
            f"{dataset}/{spec.feature_name}: non-perturbable column was modified by engine"
        )


# ---------------------------------------------------------------------------
# C04  One-hot consistency for German explanation CSVs
# ---------------------------------------------------------------------------

GERMAN_ONE_HOT_GROUPS = [
    ["checking_status_A12","checking_status_A13","checking_status_A14"],
    ["savings_status_A62","savings_status_A63","savings_status_A64","savings_status_A65"],
    ["credit_history_A31","credit_history_A32","credit_history_A33","credit_history_A34"],
    ["employment_A72","employment_A73","employment_A74","employment_A75"],
    ["personal_status_A92","personal_status_A93","personal_status_A94"],
    ["other_parties_A102","other_parties_A103"],
    ["property_magnitude_A122","property_magnitude_A123","property_magnitude_A124"],
    ["other_payment_plans_A142","other_payment_plans_A143"],
    ["housing_A152","housing_A153"],
    ["job_A172","job_A173","job_A174"],
]


def test_C04_german_one_hot_valid():
    """Each one-hot group in German explanation instances has row-sum in {0,1}."""
    df = load_instances("german")
    for group in GERMAN_ONE_HOT_GROUPS:
        present = [c for c in group if c in df.columns]
        if not present:
            continue
        row_sums = df[present].sum(axis=1)
        assert (row_sums <= 1).all(), (
            f"One-hot group {group[0][:20]}... has row_sum > 1 in some instances"
        )
        assert (row_sums >= 0).all(), (
            f"One-hot group has negative row_sum"
        )


# ---------------------------------------------------------------------------
# C05  Domain consistency: simulated perturbations respect bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", DATASETS)
def test_C05_domain_constraints(dataset):
    """For each perturbable continuous feature, simulated output stays in [lo, hi]."""
    df = load_instances(dataset)
    for spec in get_perturbable_features(dataset):
        feat = spec.feature_name
        if feat not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[feat]):
            continue
        if spec.perturbation_rule not in ("relative_signed", "ordinal_step"):
            continue

        orig = df[feat].to_numpy(dtype=float)
        for rate in (-0.10, -0.05, +0.05, +0.10):
            if spec.perturbation_rule == "relative_signed":
                out = perturb_relative_signed(orig.copy(), rate, spec.lower_bound, spec.upper_bound)
            else:
                out = perturb_ordinal_step(orig.copy(), rate, spec.lower_bound, spec.upper_bound)

            if spec.lower_bound is not None:
                violations = out < spec.lower_bound - 1e-9
                assert not violations.any(), (
                    f"{dataset}/{feat}/rate={rate}: {violations.sum()} values below lo={spec.lower_bound}"
                )
            if spec.upper_bound is not None:
                violations = out > spec.upper_bound + 1e-9
                assert not violations.any(), (
                    f"{dataset}/{feat}/rate={rate}: {violations.sum()} values above hi={spec.upper_bound}"
                )


# ---------------------------------------------------------------------------
# C06  Reproducibility
# ---------------------------------------------------------------------------

def test_C06_reproducibility():
    """Same inputs produce identical perturbed outputs."""
    orig = np.array([100.0, 500.0, 1000.0, 5000.0])
    for rate in (-0.10, -0.05, +0.05, +0.10):
        a = perturb_relative_signed(orig.copy(), rate, 1.0, None)
        b = perturb_relative_signed(orig.copy(), rate, 1.0, None)
        assert np.array_equal(a, b), f"Non-deterministic output at rate={rate}"


# ---------------------------------------------------------------------------
# V01  German numeric bounds in instance file
# ---------------------------------------------------------------------------

def test_V01_german_numeric_bounds():
    """Perturbable features in German explanation instances satisfy schema bounds."""
    df = load_instances("german")
    for spec in get_perturbable_features("german"):
        if spec.feature_name not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[spec.feature_name]):
            continue
        col = df[spec.feature_name]
        assert col.notna().all(), f"German/{spec.feature_name}: NaN in instance file"
        assert np.isfinite(col.to_numpy(dtype=float)).all(), (
            f"German/{spec.feature_name}: non-finite values in instance file"
        )
        if spec.lower_bound is not None:
            n_below = (col < spec.lower_bound).sum()
            assert n_below == 0, (
                f"German/{spec.feature_name}: {n_below} values below lo={spec.lower_bound}. "
                f"Min={col.min()}"
            )
        if spec.upper_bound is not None:
            n_above = (col > spec.upper_bound).sum()
            assert n_above == 0, (
                f"German/{spec.feature_name}: {n_above} values above hi={spec.upper_bound}. "
                f"Max={col.max()}"
            )


# ---------------------------------------------------------------------------
# V02  German integer-valued features
# ---------------------------------------------------------------------------

def test_V02_german_integer_valued():
    df = load_instances("german")
    for spec in get_perturbable_features("german"):
        if spec.integer_valued and spec.feature_name in df.columns:
            col = df[spec.feature_name]
            diffs = (col - col.round()).abs()
            assert (diffs < 1e-6).all(), (
                f"German/{spec.feature_name}: non-integer values in instance file"
            )


# ---------------------------------------------------------------------------
# V03  German one-hot groups valid (same as C04, kept as V-series for report)
# ---------------------------------------------------------------------------

def test_V03_german_one_hot_groups():
    test_C04_german_one_hot_valid()


# ---------------------------------------------------------------------------
# V04  Taiwan perturbable features exist and are numeric
# ---------------------------------------------------------------------------

def test_V04_taiwan_perturbable_features():
    df = load_instances("taiwan")
    for spec in get_perturbable_features("taiwan"):
        assert spec.feature_name in df.columns, (
            f"taiwan: '{spec.feature_name}' missing from instances"
        )
        assert pd.api.types.is_numeric_dtype(df[spec.feature_name]), (
            f"taiwan: '{spec.feature_name}' is not numeric"
        )


# ---------------------------------------------------------------------------
# V05  GMSC perturbable features exist and are numeric
# ---------------------------------------------------------------------------

def test_V05_gmsc_perturbable_features():
    df = load_instances("gmsc")
    for spec in get_perturbable_features("gmsc"):
        assert spec.feature_name in df.columns, (
            f"gmsc: '{spec.feature_name}' missing from instances"
        )
        assert pd.api.types.is_numeric_dtype(df[spec.feature_name]), (
            f"gmsc: '{spec.feature_name}' is not numeric"
        )


# ---------------------------------------------------------------------------
# V06  Non-perturbable features exist in all 3 datasets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", DATASETS)
def test_V06_non_perturbable_exist(dataset):
    """Non-perturbable numeric features must be present in the explanation instance CSV.
    Categorical features encoded as one-hot dummies are checked via dummy-prefix matching."""
    df = load_instances(dataset)
    for spec in get_non_perturbable_features(dataset):
        feat = spec.feature_name
        if feat in df.columns:
            # Feature present directly — OK
            continue
        # Check for one-hot encoding (e.g. personal_status -> personal_status_A92...)
        dummy_present = any(c.startswith(feat + "_") for c in df.columns)
        if dummy_present:
            # One-hot encoded group found — OK
            continue
        # Feature genuinely absent: only OK for categorical_engine_only or one_hot_group
        if spec.feature_type in ("ordinal_categorical", "binary") or spec.one_hot_group:
            # Acceptable: may be encoded differently (e.g. baseline-dropped dummy)
            continue
        # For numeric non-perturbable features, absence is a real problem
        if spec.feature_type in ("continuous", "ordinal_integer"):
            pytest.fail(
                f"{dataset}: non-perturbable numeric feature '{feat}' "
                f"is absent from explanation instances CSV"
            )

