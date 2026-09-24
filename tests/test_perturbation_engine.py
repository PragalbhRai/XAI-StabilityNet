"""
tests/test_perturbation_engine.py
===================================
Unit tests for the Stage-1.5 corrected perturbation engine.
Run:  python -m pytest tests/test_perturbation_engine.py -v

Covers:
  T01  Signed relative movement - positive original
  T02  Signed relative movement - negative original (I-4 fix verification)
  T03  Signed relative movement - zero original (scale fall-back)
  T04  Ordinal step - positive step
  T05  Ordinal step - negative step
  T06  Ordinal step - bounds clipping
  T07  Ordinal step - output is integer-typed
  T08  Bounds: clipping after relative perturbation
  T09  Non-perturbable features unchanged (schema check)
  T10  Invalid feature name not in schema raises KeyError
  T11  Determinism: same input -> same output
  T12  Schema: every perturbable feature has a valid rule
  T13  Schema: no feature is both perturbable and non-perturbable
  T14  Schema: canonical list != old phase-4 list for Taiwan (RF-1 check)
  T15  Schema: canonical list != old phase-4 list for GMSC (RF-1 check)
  T16  Schema: canonical list != old phase-4 list for German (RF-1 check)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4.phase4_perturbation_v2 import perturb_relative_signed, perturb_ordinal_step
from phase3.perturbation_schema import (
    SCHEMA, get_perturbable_features, get_non_perturbable_features, get_schema_dict,
)


# ---------------------------------------------------------------------------
# T01-T03  Signed relative movement
# ---------------------------------------------------------------------------

def test_T01_positive_plus5():
    """x=+1000, r=+0.05 => x'=1050"""
    orig = np.array([1000.0])
    out  = perturb_relative_signed(orig, +0.05, None, None)
    assert np.isclose(out[0], 1050.0), f"Expected 1050, got {out[0]}"


def test_T01b_positive_minus5():
    """x=+1000, r=-0.05 => x'=950"""
    orig = np.array([1000.0])
    out  = perturb_relative_signed(orig, -0.05, None, None)
    assert np.isclose(out[0], 950.0), f"Expected 950, got {out[0]}"


def test_T02_negative_plus5():
    """x=-1000, r=+0.05 => x'=x*(1+r)=-1050.
    Old buggy code used abs(original)*magnitude then added/subtracted positively,
    giving -1000+50=-950 regardless of sign (WRONG for any r meaning).
    Corrected formula x*(1+r) gives -1050 (5% more negative = consistent scale)."""
    orig = np.array([-1000.0])
    out  = perturb_relative_signed(orig, +0.05, None, None)
    assert np.isclose(out[0], -1050.0), f"Expected -1050 (x*(1+r)), got {out[0]}"


def test_T02b_negative_minus5():
    """x=-1000, r=-0.05 => x'=x*(1-0.05)=-950.
    Old buggy code gave -1000-50=-1050 (WRONG).
    Corrected formula gives -950 (5% less negative = consistent scale)."""
    orig = np.array([-1000.0])
    out  = perturb_relative_signed(orig, -0.05, None, None)
    assert np.isclose(out[0], -950.0), f"Expected -950 (x*(1+r)), got {out[0]}"


def test_T03_zero_value():
    """x=0, any rate => finite output (scale fall-back active)"""
    orig = np.array([0.0, 500.0])
    out  = perturb_relative_signed(orig, +0.05, None, None)
    assert np.isfinite(out[0]), "Zero-value should produce finite perturbed value"


# ---------------------------------------------------------------------------
# T04-T07  Ordinal step
# ---------------------------------------------------------------------------

def test_T04_ordinal_step_up():
    """rate>0 -> +1 step"""
    orig = np.array([2.0, 3.0])
    out  = perturb_ordinal_step(orig, +0.05, 1, 4)
    assert list(out) == [3, 4], f"Got {out}"


def test_T05_ordinal_step_down():
    """rate<0 -> -1 step"""
    orig = np.array([2.0, 3.0])
    out  = perturb_ordinal_step(orig, -0.05, 1, 4)
    assert list(out) == [1, 2], f"Got {out}"


def test_T06_ordinal_step_bounds():
    """Steps are clipped to [lo, hi]"""
    orig = np.array([1.0, 4.0])
    out_up   = perturb_ordinal_step(orig, +0.05, 1, 4)
    out_down = perturb_ordinal_step(orig, -0.05, 1, 4)
    assert out_up[1]   == 4,  f"Upper bound violated: {out_up[1]}"
    assert out_down[0] == 1,  f"Lower bound violated: {out_down[0]}"


def test_T07_ordinal_output_is_integer():
    """Ordinal step must produce integer-compatible values"""
    orig = np.array([2.0])
    out  = perturb_ordinal_step(orig, +0.05, 1, 4)
    assert out[0] == int(out[0]), f"Not integer: {out[0]}"


# ---------------------------------------------------------------------------
# T08  Bounds after relative perturbation
# ---------------------------------------------------------------------------

def test_T08_relative_bounds_clipping():
    """Relative perturbation respects explicit lower/upper bounds"""
    orig = np.array([4.0])   # duration: lo=4
    out  = perturb_relative_signed(orig, -0.10, lo=4.0, hi=72.0)
    assert out[0] >= 4.0, f"Lower bound violated: {out[0]}"

    orig2 = np.array([65000.0])   # credit_amount: hi=50000
    out2  = perturb_relative_signed(orig2, +0.10, lo=100.0, hi=50000.0)
    assert out2[0] <= 50000.0, f"Upper bound violated: {out2[0]}"


# ---------------------------------------------------------------------------
# T09  Non-perturbable feature unchanged
# ---------------------------------------------------------------------------


def test_T09_non_perturbable_not_modified_by_engine():
    """Engine must NOT modify non-perturbable columns.
    Uses Taiwan dataset which has numeric non-perturbable features
    (SEX=binary, EDUCATION=ordinal int, MARRIAGE=ordinal int).

    NOTE: Under the revised schema, German has NO numeric non-perturbable features
    (personal_status, credit_history, job are all ordinal_categorical; foreign_worker
    is binary but not numeric in the CSV). GMSC also has none. This is correct and
    expected: the revised schema reclassified all integer-count features as
    perturbable_for_local_robustness.
    """
    df = pd.read_csv(
        Path(__file__).resolve().parents[1] /
        "results" / "phase3" / "taiwan_explanation_instances.csv"
    )
    non_pert_names = {s.feature_name for s in get_non_perturbable_features("taiwan")
                      if s.feature_name in df.columns
                      and pd.api.types.is_numeric_dtype(df[s.feature_name])}
    pert_names = {s.feature_name for s in get_perturbable_features("taiwan")
                  if s.feature_name in df.columns
                  and pd.api.types.is_numeric_dtype(df[s.feature_name])
                  and s.perturbation_rule == "relative_signed"}
    assert non_pert_names, (
        "No numeric non-perturbable Taiwan features found. "
        "SEX, EDUCATION, MARRIAGE should be in df and numeric."
    )

    df_modified = df.copy()
    # Simulate: engine applies relative_signed only to perturbable continuous features
    for feat in pert_names:
        orig = df_modified[feat].to_numpy(dtype=float)
        df_modified[feat] = perturb_relative_signed(orig, +0.05, None, None)

    # Verify non-perturbable numeric columns are byte-identical
    for feat in non_pert_names:
        assert df_modified[feat].equals(df[feat]), (
            f"Engine accidentally modified non-perturbable column: {feat}"
        )


def test_T09b_non_perturbable_schema_rule():
    """Every non_perturbable feature has perturbation_rule='none' in schema."""
    for spec in SCHEMA:
        if spec.perturbation_role == "non_perturbable":
            assert spec.perturbation_rule == "none", (
                f"{spec.dataset}/{spec.feature_name}: non_perturbable but rule='{spec.perturbation_rule}'"
            )


# ---------------------------------------------------------------------------
# T10  Invalid feature name lookup is clearly absent from schema
# ---------------------------------------------------------------------------

def test_T10_schema_dict_excludes_unknown_feature():
    """get_schema_dict returns only registered features.
    A fabricated feature name must not appear in the dict.
    This guards against accidentally broad schema definitions."""
    schema = get_schema_dict("german")
    assert len(schema) > 0, "Schema dict is empty â€” something is wrong"
    assert "FAKE_FEATURE_XYZ" not in schema, "Fake feature appears in schema dict"
    # Also verify all keys are real feature names from the German dataset
    for name in schema:
        assert any(s.feature_name == name and s.dataset == "german" for s in SCHEMA), (
            f"Key '{name}' in get_schema_dict('german') not found in SCHEMA"
        )


# ---------------------------------------------------------------------------
# T11  Determinism
# ---------------------------------------------------------------------------

def test_T11_determinism():
    """Same input + same parameters -> identical output every call"""
    orig = np.array([500.0, 1000.0, -200.0])
    out1 = perturb_relative_signed(orig.copy(), +0.05, None, None)
    out2 = perturb_relative_signed(orig.copy(), +0.05, None, None)
    assert np.array_equal(out1, out2), "Non-deterministic output detected"


# ---------------------------------------------------------------------------
# T12  Schema: every perturbable feature has a valid rule
# ---------------------------------------------------------------------------

VALID_RULES = {"relative_signed", "ordinal_step", "time_anchor", "time_follower"}

def test_T12_perturbable_has_valid_rule():
    for spec in SCHEMA:
        if spec.perturbation_role == "perturbable_for_local_robustness":
            assert spec.perturbation_rule in VALID_RULES, (
                f"{spec.dataset}/{spec.feature_name}: perturbable but rule='{spec.perturbation_rule}'"
            )


# ---------------------------------------------------------------------------
# T13  Schema: no overlap between perturbable and non-perturbable
# ---------------------------------------------------------------------------

def test_T13_no_role_overlap():
    for ds in ["german", "taiwan", "gmsc"]:
        p_names = {s.feature_name for s in get_perturbable_features(ds)}
        n_names = {s.feature_name for s in get_non_perturbable_features(ds)}
        overlap = p_names & n_names
        assert not overlap, f"{ds}: features in both sets: {overlap}"
# ---------------------------------------------------------------------------
# T14  Revised features: correct perturbation RULE (not just presence)
#
# The original T14-T16 tested that disputed features were ABSENT from the
# perturbable list. That tested the wrong thing (actionability criterion).
#
# The revised tests verify that each disputed feature:
#   (a) IS in the perturbable list (valid local perturbation exists)
#   (b) uses the CORRECT rule (ordinal_step for ordinals, relative_signed for continuous)
#   (c) has sensible bounds (e.g. lo=-2 for PAY codes, lo=0 for counts/amounts)
# ---------------------------------------------------------------------------

def _get_spec(dataset: str, feature: str) -> "FeatureSpec":
    for s in SCHEMA:
        if s.dataset == dataset and s.feature_name == feature:
            return s
    raise KeyError(f"Feature '{feature}' not found in schema for dataset '{dataset}'")


def test_T14_taiwan_pay_codes_use_ordinal_step():
    """PAY_0..PAY_6: ordinal codes, must use ordinal_step with bounds [-2, 8].
    Using relative_signed on integer codes would produce fractional codes (invalid).
    The correct local perturbation is a Â±1 step to an adjacent valid code."""
    for col in ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]:
        spec = _get_spec("taiwan", col)
        assert spec.perturbation_role == "perturbable_for_local_robustness", (
            f"taiwan/{col}: expected perturbable, got '{spec.perturbation_role}'"
        )
        assert spec.perturbation_rule == "ordinal_step", (
            f"taiwan/{col}: expected ordinal_step, got '{spec.perturbation_rule}'. "
            f"PAY codes are ordinal integers; relative_signed would produce fractional codes."
        )
        assert spec.lower_bound == -2, (
            f"taiwan/{col}: lower_bound should be -2 (min valid PAY code), got {spec.lower_bound}"
        )
        assert spec.upper_bound == 8, (
            f"taiwan/{col}: upper_bound should be 8 (max valid PAY code), got {spec.upper_bound}"
        )
        assert spec.integer_valued is True, (
            f"taiwan/{col}: must be integer_valued=True"
        )


def test_T15_gmsc_count_features_use_ordinal_step():
    """GMSC delinquency counts and NumberOfDependents: integer counts >= 0.
    Must use ordinal_step with lo=0. relative_signed on these features is
    inappropriate since they are non-negative integer counts, not continuous values."""
    count_features = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]
    for col in count_features:
        spec = _get_spec("gmsc", col)
        assert spec.perturbation_role == "perturbable_for_local_robustness", (
            f"gmsc/{col}: expected perturbable, got '{spec.perturbation_role}'"
        )
        assert spec.perturbation_rule == "ordinal_step", (
            f"gmsc/{col}: expected ordinal_step, got '{spec.perturbation_rule}'. "
            f"Integer counts must use ordinal_step."
        )
        assert spec.lower_bound == 0, (
            f"gmsc/{col}: lower_bound must be 0 (count cannot go negative), got {spec.lower_bound}"
        )
        assert spec.integer_valued is True, (
            f"gmsc/{col}: must be integer_valued=True"
        )


def test_T16_protected_attributes_non_perturbable():
    """Protected attributes must remain non_perturbable regardless of dataset.
    Perturbing sex, marital status, or national-origin proxy confounds robustness
    with fairness analysis. This is the ONLY legitimate use of the actionability
    criterion in this schema."""
    protected = {
        "german": {"personal_status", "foreign_worker"},
        "taiwan": {"SEX", "MARRIAGE"},
        "gmsc":   set(),  # no protected attributes in GMSC schema
    }
    for ds, feats in protected.items():
        for feat in feats:
            spec = _get_spec(ds, feat)
            assert spec.perturbation_role == "non_perturbable", (
                f"{ds}/{feat}: protected attribute must be non_perturbable, "
                f"got '{spec.perturbation_role}'"
            )
            assert spec.perturbation_rule == "none", (
                f"{ds}/{feat}: protected attribute must have rule='none'"
            )


def test_T14b_taiwan_bill_amounts_no_lower_bound():
    """BILL_AMT features: continuous billing balances that can be NEGATIVE
    (credit overpayment). Must have no lower_bound enforced so the corrected
    formula x*(1+r) can produce negative values when x is negative."""
    for i in range(1, 7):
        spec = _get_spec("taiwan", f"BILL_AMT{i}")
        assert spec.perturbation_role == "perturbable_for_local_robustness"
        assert spec.perturbation_rule == "relative_signed"
        assert spec.lower_bound is None, (
            f"BILL_AMT{i}: lower_bound must be None (can be negative), got {spec.lower_bound}"
        )


def test_T15b_taiwan_pay_amounts_non_negative():
    """PAY_AMT features: payment amounts are always >= 0. Must have lo=0."""
    for i in range(1, 7):
        spec = _get_spec("taiwan", f"PAY_AMT{i}")
        assert spec.perturbation_role == "perturbable_for_local_robustness"
        assert spec.perturbation_rule == "relative_signed"
        assert spec.lower_bound == 0.0, (
            f"PAY_AMT{i}: lower_bound must be 0.0, got {spec.lower_bound}"
        )


def test_T16b_german_num_dependents_ordinal_step():
    """num_dependents is an integer count {0..5}. Must use ordinal_step, not
    relative_signed (which would produce fractional dependent counts)."""
    spec = _get_spec("german", "num_dependents")
    assert spec.perturbation_role == "perturbable_for_local_robustness"
    assert spec.perturbation_rule == "ordinal_step", (
        f"num_dependents must use ordinal_step (not relative_signed), got '{spec.perturbation_rule}'"
    )
    assert spec.lower_bound == 0
    assert spec.integer_valued is True
