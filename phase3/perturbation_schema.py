"""
phase3/perturbation_schema.py
==============================
CANONICAL PERTURBATION ONTOLOGY  -  Stage 1.5 (revised)
Single source of truth for every feature across all three datasets.

Replaces the two inconsistent prior sources:
  * phase3/perturbation_constraints.py   (phase-3 engine, German-only validated)
  * phase4/phase4_perturbation.py        (PERTURBABLE_FEATURES dict, phase-4 engine)

Usage
-----
    from phase3.perturbation_schema import SCHEMA, get_perturbable_features

===============================================================================
CRITICAL METHODOLOGICAL DISTINCTION
===============================================================================

This experiment measures LOCAL EXPLANATION ROBUSTNESS: how sensitive are
model predictions and SHAP/LIME attributions to small changes in the input?

A feature is classified as PERTURBABLE FOR LOCAL ROBUSTNESS if a domain-valid
local perturbation can be constructed. That is:
  (a) the perturbed value x' is a legitimate value for that feature
  (b) x' is "nearby" x in the feature's natural metric
  (c) the perturbation is mathematically well-defined

A feature is PERTURBABLE irrespective of whether the perturbation represents
a real-world intervention a borrower could actually make.

These are DIFFERENT questions:
  "Can the borrower change this value?"      -> ACTIONABILITY (not our criterion)
  "Can we construct a valid nearby value?"   -> PERTURBABILITY (our criterion)

This distinction is standard in robustness/sensitivity analysis of ML models.
A model that drastically changes its explanation when PAY_0 shifts from status
2 to status 3 is fragile regardless of whether a borrower could cause that
shift.

NON-PERTURBABLE features in this schema are non-perturbable because:
  (1) Protected attributes: perturbing sex, marital status, or national origin
      confounds robustness with fairness experiments and changes the semantic
      meaning of the experiment.
  (2) No valid neighboring state: e.g. a binary indicator that is either 0 or 1
      where neither neighbor is interpretable in a local sense.
  (3) Identifiers or target variable: structurally excluded.

===============================================================================
Classification vocabulary
===============================================================================
perturbation_role:
  "perturbable_for_local_robustness"
      Feature is deliberately varied to test model sensitivity.
      Rule determines how the variation is constructed.

  "non_perturbable"
      Feature is NEVER modified in this experiment.
      Only two legitimate reasons in this schema:
        (A) Protected attribute (sex, marital status, national origin proxy)
        (B) No valid local perturbation can be constructed (identifiers, target)

  "categorical_engine_only"
      Perturbable in principle, but ONLY via the categorical one-hot engine
      in phase3_perturbation.py. The phase-4 numeric engine must NOT touch these.

perturbation_rule:
  "relative_signed"    x' = x*(1+r),  r in {-0.10,-0.05,+0.05,+0.10}
                       Works for positive AND negative x. Zero-value fallback
                       uses column-median scale.
  "ordinal_step"       x' = clip(x +/- 1, lo, hi).  Integer output.
                       Direction: r>0 -> +1 step, r<0 -> -1 step.
  "time_anchor"        x' = clip(x + |x|*r, x, hi).  Can only advance.
                       Negative r -> no change (x cannot decrease).
  "time_follower"      Categorical engine only. Phase-4 numeric engine ignores.
  "none"               Feature is non_perturbable.

===============================================================================
Disputed feature decisions (Stage 1.5 revision)
===============================================================================

Previous version classified the following as non_perturbable based on the
reasoning that they are "historical records" and therefore not actionable.
That reasoning applied the wrong criterion (actionability).

All disputed features have been re-evaluated on the correct criterion:
can a domain-valid local perturbation be constructed?

German  num_dependents:
  Previous: non_perturbable (fractional perturbation invalid)
  Revised:  perturbable_for_local_robustness, ordinal_step, [0,5]
  Reason:   ordinal_step +/-1 produces {0,1,2,3} which are all valid integer
            states for this feature. The fractional-invalid argument applies
            to relative_signed but NOT to ordinal_step. The robustness
            question "does the model explanation change if num_dependents
            shifts from 1 to 2?" is scientifically legitimate.

Taiwan  PAY_0, PAY_2..PAY_6 (monthly repayment status codes, ordinal -2..8):
  Previous: non_perturbable (historical records)
  Revised:  perturbable_for_local_robustness, ordinal_step, [-2, 8]
  Reason:   Adjacent status codes are valid ordinal states. The code -2=no
            consumption, -1=paid-in-full, 0=revolving-credit-used,
            1=one-month-delay, ..., 8=eight-month-delay. A step from code 1
            to code 2 is a well-defined neighboring state. Historical status
            is irrelevant to whether a PERTURBATION is valid.

Taiwan  BILL_AMT1..BILL_AMT6 (monthly billing statement balance, continuous):
  Previous: non_perturbable (historical; sign bug)
  Revised:  perturbable_for_local_robustness, relative_signed, lo=None hi=None
  Reason:   x*(1+r) produces a valid billing balance regardless of sign.
            The sign-inversion bug (I-4) applied to the OLD formula; the
            corrected formula has no sign issue. Empirically, values in
            the 100 instances range from -1000 to +520651; a +/-5%
            multiplicative perturbation remains in a plausible range.
            No lower or upper bound enforced (billing balances can be
            negative; outliers exist above 50000 NTD).

Taiwan  PAY_AMT1..PAY_AMT6 (monthly payment amount, continuous >= 0):
  Previous: non_perturbable (historical)
  Revised:  perturbable_for_local_robustness, relative_signed, lo=0
  Reason:   Payment amounts are non-negative continuous values. x*(1+r) with
            lo=0 clipping produces a valid payment amount. The robustness
            question "does the explanation change if this payment was 5% higher?"
            is scientifically well-posed.

GMSC  NumberOfTime30-59DaysPastDueNotWorse (integer count >= 0):
GMSC  NumberOfTimes90DaysLate (integer count >= 0):
GMSC  NumberOfTime60-89DaysPastDueNotWorse (integer count >= 0):
  Previous: non_perturbable (historical delinquency counts)
  Revised:  perturbable_for_local_robustness, ordinal_step, lo=0
  Reason:   These are non-negative integer counts. ordinal_step +/-1 with lo=0
            produces {0,1,2,...} which are all valid states for the feature.
            In the 100 instances, the observed range is small (0-3), making
            a +/-1 step a genuinely local perturbation.

GMSC  NumberOfDependents (integer count >= 0):
  Previous: non_perturbable (integer; fractional invalid)
  Revised:  perturbable_for_local_robustness, ordinal_step, lo=0
  Reason:   Same as num_dependents above. ordinal_step +/-1 with lo=0
            produces valid integer states {0,1,2,3,4,...}.
            Observed range in instances: 0-4.

Features that REMAIN non_perturbable (with justification):
  SEX, personal_status, foreign_worker, MARRIAGE  ->  protected attributes
  EDUCATION                                       ->  sensitive proxy attribute
  credit_history                                  ->  categorical with no valid
                                                      numeric neighbor (handled
                                                      by categorical engine only)
  job                                             ->  categorical engine only;
                                                      not numeric-perturbable
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

PERTURBATION_RATES = (-0.10, -0.05, +0.05, +0.10)


@dataclass(frozen=True)
class FeatureSpec:
    dataset: str
    feature_name: str
    perturbation_role: str
    feature_type: str
    perturbation_rule: str
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    integer_valued: bool
    one_hot_group: Optional[str]
    rationale: str


# ---------------------------------------------------------------------------
# German Credit (Statlog)
# ---------------------------------------------------------------------------
_GERMAN = [
    FeatureSpec("german","age","perturbable_for_local_robustness","continuous",
                "time_anchor",18,100,False,None,
                "Age: physical elapsed time; only advances; time_anchor."),
    FeatureSpec("german","duration","perturbable_for_local_robustness","continuous",
                "relative_signed",4,72,False,None,
                "Loan duration in months; continuous; relative_signed."),
    FeatureSpec("german","credit_amount","perturbable_for_local_robustness","continuous",
                "relative_signed",100,50000,False,None,
                "Credit amount; strictly positive; relative_signed."),
    FeatureSpec("german","installment_rate","perturbable_for_local_robustness","ordinal_integer",
                "ordinal_step",1,4,True,None,
                "Installment rate bucket 1-4; ordinal_step +/-1."),
    FeatureSpec("german","existing_credits","perturbable_for_local_robustness","ordinal_integer",
                "ordinal_step",1,4,True,None,
                "Existing credits at bank 1-4; ordinal_step +/-1."),
    FeatureSpec("german","num_dependents","perturbable_for_local_robustness","ordinal_integer",
                "ordinal_step",0,5,True,None,
                "REVISED: ordinal_step +/-1 produces valid integer states {0,1,2,3,4,5}. "
                "Previous version incorrectly marked non_perturbable using actionability "
                "as criterion. Observed values in instances: {1,2}. A step to {0,2,3} "
                "is a valid local perturbation for robustness testing."),

    # Categorical: Phase-4 numeric engine must not touch these
    FeatureSpec("german","residence_since","categorical_engine_only","ordinal_integer",
                "time_follower",1,4,True,None,
                "Ordinal residence bucket; categorical engine only."),
    FeatureSpec("german","employment","categorical_engine_only","ordinal_categorical",
                "time_follower",None,None,False,"employment",
                "One-hot employment duration; categorical engine only."),
    FeatureSpec("german","checking_status","categorical_engine_only","ordinal_categorical",
                "none",None,None,False,"checking_status","Categorical."),
    FeatureSpec("german","purpose","categorical_engine_only","ordinal_categorical",
                "none",None,None,False,"purpose","Categorical."),
    FeatureSpec("german","savings_status","categorical_engine_only","ordinal_categorical",
                "none",None,None,False,"savings_status","Categorical."),
    FeatureSpec("german","other_parties","categorical_engine_only","ordinal_categorical",
                "none",None,None,False,"other_parties","Categorical."),
    FeatureSpec("german","property_magnitude","categorical_engine_only","ordinal_categorical",
                "none",None,None,False,"property_magnitude","Categorical."),
    FeatureSpec("german","other_payment_plans","categorical_engine_only","ordinal_categorical",
                "none",None,None,False,"other_payment_plans","Categorical."),
    FeatureSpec("german","housing","categorical_engine_only","ordinal_categorical",
                "none",None,None,False,"housing","Categorical."),
    FeatureSpec("german","own_telephone","categorical_engine_only","ordinal_categorical",
                "none",None,None,False,"own_telephone","Categorical."),

    # Non-perturbable: PROTECTED ATTRIBUTES ONLY
    FeatureSpec("german","personal_status","non_perturbable","ordinal_categorical",
                "none",None,None,False,"personal_status",
                "Non-perturbable: protected attribute (sex + marital status combined). "
                "Perturbing sex/marital-status confounds robustness with fairness."),
    FeatureSpec("german","foreign_worker","non_perturbable","binary",
                "none",None,None,False,"foreign_worker",
                "Non-perturbable: protected attribute (national-origin proxy)."),
    FeatureSpec("german","credit_history","non_perturbable","ordinal_categorical",
                "none",None,None,False,"credit_history",
                "Non-perturbable: handled by categorical engine only; "
                "no valid numeric neighbor. Included here for completeness."),
    FeatureSpec("german","job","non_perturbable","ordinal_categorical",
                "none",None,None,False,"job",
                "Non-perturbable: categorical; no valid numeric neighbor."),
]

# ---------------------------------------------------------------------------
# Taiwan Default of Credit Card Clients
# ---------------------------------------------------------------------------
_TAIWAN_STATIC = [
    FeatureSpec("taiwan","LIMIT_BAL","perturbable_for_local_robustness","continuous",
                "relative_signed",1000,2_000_000,False,None,
                "Credit limit; strictly positive; relative_signed."),
    FeatureSpec("taiwan","AGE","perturbable_for_local_robustness","continuous",
                "time_anchor",18,100,False,None,
                "Age; physical elapsed time; time_anchor."),
    # Protected attributes
    FeatureSpec("taiwan","SEX","non_perturbable","binary",
                "none",None,None,False,None,
                "Non-perturbable: protected attribute. Perturbing sex confounds "
                "robustness with fairness analysis."),
    FeatureSpec("taiwan","EDUCATION","non_perturbable","ordinal_integer",
                "none",None,None,False,None,
                "Non-perturbable: sensitive socioeconomic proxy attribute. "
                "Adjacent ordinal states are not well-defined (codes 5,6='unknown')."),
    FeatureSpec("taiwan","MARRIAGE","non_perturbable","ordinal_integer",
                "none",None,None,False,None,
                "Non-perturbable: protected attribute (marital status)."),
]

# PAY_0, PAY_2..PAY_6: ordinal repayment status codes -2..8
# REVISED: perturbable via ordinal_step. Adjacent status is a valid state.
_TAIWAN_PAY = [
    FeatureSpec("taiwan",f"PAY_{i}","perturbable_for_local_robustness","ordinal_integer",
                "ordinal_step",-2,8,True,None,
                f"REVISED: ordinal_step +/-1 in [-2,8]. Status codes: "
                "-2=no consumption, -1=paid-in-full, 0=revolving used, "
                "1..8=months-delayed. Adjacent code is a valid neighboring state. "
                "Previous version marked non_perturbable using actionability criterion: INCORRECT.")
    for i in [0,2,3,4,5,6]
]

# BILL_AMT1..6: continuous billing balances (can be negative, e.g. credit overpayment)
# REVISED: perturbable via relative_signed. x*(1+r) is valid for any real value.
_TAIWAN_BILL = [
    FeatureSpec("taiwan",f"BILL_AMT{i}","perturbable_for_local_robustness","continuous",
                "relative_signed",None,None,False,None,
                f"REVISED: relative_signed with no enforced bounds. Billing balances "
                "are continuous and can be negative (credit overpayment). x*(1+r) "
                "produces a valid neighboring balance. Observed range in instances: "
                "approx -1000 to +520651. Previous version: non_perturbable "
                "using actionability criterion: INCORRECT.")
    for i in range(1,7)
]

# PAY_AMT1..6: past payment amounts, always >= 0
# REVISED: perturbable via relative_signed with lo=0
_TAIWAN_PAYAMT = [
    FeatureSpec("taiwan",f"PAY_AMT{i}","perturbable_for_local_robustness","continuous",
                "relative_signed",0.0,None,False,None,
                f"REVISED: relative_signed, lo=0. Payment amounts are non-negative "
                "continuous values. x*(1+r) with lo=0 produces a valid amount. "
                "Previous version: non_perturbable using actionability criterion: INCORRECT.")
    for i in range(1,7)
]

_TAIWAN = _TAIWAN_STATIC + _TAIWAN_PAY + _TAIWAN_BILL + _TAIWAN_PAYAMT

# ---------------------------------------------------------------------------
# Give Me Some Credit (GMSC / Kaggle)
# ---------------------------------------------------------------------------
_GMSC = [
    FeatureSpec("gmsc","age","perturbable_for_local_robustness","continuous",
                "time_anchor",18,100,False,None,"Age; physical elapsed time; time_anchor."),
    FeatureSpec("gmsc","RevolvingUtilizationOfUnsecuredLines","perturbable_for_local_robustness",
                "continuous","relative_signed",0.0,None,False,None,
                "Utilization ratio; non-negative; no upper bound (GMSC has extreme outliers). "
                "relative_signed, lo=0."),
    FeatureSpec("gmsc","DebtRatio","perturbable_for_local_robustness","continuous",
                "relative_signed",0.0,None,False,None,
                "Debt-to-income ratio; non-negative; no upper bound enforced "
                "(instances contain values up to ~3000). relative_signed, lo=0."),
    FeatureSpec("gmsc","MonthlyIncome","perturbable_for_local_robustness","continuous",
                "relative_signed",0.0,None,False,None,
                "Monthly income; non-negative; relative_signed, lo=0."),
    FeatureSpec("gmsc","NumberOfOpenCreditLinesAndLoans","perturbable_for_local_robustness",
                "ordinal_integer","ordinal_step",0,60,True,None,
                "Count of open credit lines; integer; ordinal_step +/-1."),
    FeatureSpec("gmsc","NumberRealEstateLoansOrLines","perturbable_for_local_robustness",
                "ordinal_integer","ordinal_step",0,20,True,None,
                "Count of real-estate loans/lines; integer; ordinal_step +/-1."),

    # Delinquency counts: REVISED to perturbable
    FeatureSpec("gmsc","NumberOfTime30-59DaysPastDueNotWorse","perturbable_for_local_robustness",
                "ordinal_integer","ordinal_step",0,None,True,None,
                "REVISED: ordinal_step +/-1, lo=0. Non-negative integer count. "
                "Observed in instances: {0,1,2,3}. A +/-1 step is a well-defined "
                "local perturbation. Previous: non_perturbable (actionability criterion): INCORRECT."),
    FeatureSpec("gmsc","NumberOfTimes90DaysLate","perturbable_for_local_robustness",
                "ordinal_integer","ordinal_step",0,None,True,None,
                "REVISED: ordinal_step +/-1, lo=0. Observed in instances: {0,1}. "
                "Previous: non_perturbable (actionability criterion): INCORRECT."),
    FeatureSpec("gmsc","NumberOfTime60-89DaysPastDueNotWorse","perturbable_for_local_robustness",
                "ordinal_integer","ordinal_step",0,None,True,None,
                "REVISED: ordinal_step +/-1, lo=0. Observed in instances: {0,1,2}. "
                "Previous: non_perturbable (actionability criterion): INCORRECT."),
    FeatureSpec("gmsc","NumberOfDependents","perturbable_for_local_robustness",
                "ordinal_integer","ordinal_step",0,None,True,None,
                "REVISED: ordinal_step +/-1, lo=0. Non-negative integer count. "
                "Observed in instances: {0,1,2,3,4}. Previous: non_perturbable "
                "(actionability criterion): INCORRECT."),
]

# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------
SCHEMA: list[FeatureSpec] = _GERMAN + _TAIWAN + _GMSC


def get_perturbable_features(dataset: str) -> list[FeatureSpec]:
    """Return features with role 'perturbable_for_local_robustness'."""
    return [s for s in SCHEMA
            if s.dataset == dataset
            and s.perturbation_role == "perturbable_for_local_robustness"]


def get_non_perturbable_features(dataset: str) -> list[FeatureSpec]:
    """Return features with role 'non_perturbable'."""
    return [s for s in SCHEMA
            if s.dataset == dataset
            and s.perturbation_role == "non_perturbable"]


def get_schema_dict(dataset: str) -> dict[str, FeatureSpec]:
    return {s.feature_name: s for s in SCHEMA if s.dataset == dataset}


if __name__ == "__main__":
    import pandas as pd
    rows = [
        {"dataset": s.dataset, "feature": s.feature_name,
         "role": s.perturbation_role, "type": s.feature_type,
         "rule": s.perturbation_rule, "lo": s.lower_bound,
         "hi": s.upper_bound, "int": s.integer_valued}
        for s in SCHEMA
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nTotal: {len(SCHEMA)}")
    for ds in ["german", "taiwan", "gmsc"]:
        p = get_perturbable_features(ds)
        n = get_non_perturbable_features(ds)
        print(f"  {ds}: {len(p)} perturbable, {len(n)} non-perturbable")
