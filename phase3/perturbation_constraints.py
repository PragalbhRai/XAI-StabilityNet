"""
perturbation_constraints.py
The domain-constraint design table — the methodological core of the
Prediction-Invariant Evaluation Framework's perturbation step.

Every feature in every dataset is assigned exactly one constraint "kind".
This module is deliberately written as an explicit, documented data
structure (not buried in perturbation logic) so it can be exported
directly into a paper appendix / methods table, and so a reviewer can
audit every constraint decision in one place.

Constraint kinds implemented by the engine (phase3_perturbation.py):

  immutable
      Feature is never changed by any perturbation. Used for:
      (a) historical/backward-looking facts that cannot be un-happened
          (e.g. past credit history, past delinquency counts), and
      (b) legally protected attributes under fair-lending regimes
          (sex, marital status, foreign-worker/national-origin proxy)
          which should not be treated as actionable "recourse levers"
          in an adverse-action-reasons context.

  time_anchor_numeric
      A continuous feature representing elapsed time (age). Sampled once
      per perturbed instance as a non-negative delta; every other
      time-linked feature in the same instance reuses this delta so the
      whole row stays causally coherent (age cannot increase while a
      derived time-linked feature decreases).

  time_follower_ordinal_numeric / time_follower_ordinal_categorical
      A feature that mechanically correlates with elapsed time (years at
      current residence, employment-duration bucket) but is stored as a
      small ordinal (numeric or one-hot-categorical). Moves up 0 or 1
      ordinal step, with probability increasing in the shared time delta,
      and is capped at the top category — it never moves backward.

  bounded_numeric
      Free-form numeric feature perturbed with additive jitter, clipped
      to an explicit [min, max] domain range. Represents an applicant
      choice variable (e.g. requested loan duration).

  bounded_numeric_positive_multiplicative
      Like bounded_numeric, but perturbed multiplicatively and floored
      strictly above zero — for quantities that are causally impossible
      when negative (credit amount, income).

  bounded_ordinal_numeric
      A small-range integer/ordinal feature perturbed by +/-1 step,
      clipped to its valid range (e.g. installment rate bucket 1-4).

  free_categorical
      A one-hot-encoded nominal feature that may be reassigned to any
      other valid category for that feature (uniformly), representing a
      plausible applicant state change (e.g. adding a co-applicant,
      changing housing arrangement).

Every constraint entry also carries a `p_perturb` (probability this
feature is touched at all in a given perturbed instance) so a single
perturbed row is a *plausible partial change*, not a rewrite of every
feature at once — closer to how a real re-assessment or recourse action
would look, and it also gives us variation in *how much* changed per
instance for later stability-vs-magnitude-of-change analysis.
"""

# ---------------------------------------------------------------------------
# German Credit (Statlog)
# ---------------------------------------------------------------------------
GERMAN_CONSTRAINTS = {
    # --- Time-linked group (shared delta => causal coherence) ---------------
    "age": {
        "kind": "time_anchor_numeric",
        "params": {"max_value": 100},
        "rationale": "Age is a physical quantity that only advances with "
                     "time; a valid future/alternate snapshot of the same "
                     "applicant can never show a lower age.",
    },
    "residence_since": {
        "kind": "time_follower_ordinal_numeric",
        "params": {"min_value": 1, "max_value": 4},
        "rationale": "Years-at-residence bucket mechanically accumulates "
                     "with elapsed time; must not decrease when age (same "
                     "instance) increases.",
    },
    "employment": {
        "kind": "time_follower_ordinal_categorical",
        "params": {},
        "rationale": "Employment-duration bucket (A71..A75) accumulates "
                     "with elapsed time under continued employment; kept "
                     "consistent with the shared time delta.",
    },

    # --- Applicant-choice numeric variables ---------------------------------
    "duration": {
        "kind": "bounded_numeric",
        "params": {"min_value": 4, "max_value": 72, "jitter_abs": 6, "p_perturb": 0.5},
        "rationale": "Requested loan term is an applicant choice; bounded "
                     "to the realistic range observed for this product.",
    },
    "credit_amount": {
        "kind": "bounded_numeric_positive_multiplicative",
        "params": {"min_value": 100, "max_value": 50000, "jitter_frac": 0.20, "p_perturb": 0.5},
        "rationale": "Requested credit amount must stay strictly positive "
                     "(negative/zero credit amounts are causally "
                     "impossible); applicant can request more or less.",
    },
    "installment_rate": {
        "kind": "bounded_ordinal_numeric",
        "params": {"min_value": 1, "max_value": 4, "p_perturb": 0.4},
        "rationale": "Installment-as-%-of-income bucket is a financing "
                     "structure choice, bounded to the coded 1-4 range.",
    },
    "existing_credits": {
        "kind": "bounded_ordinal_numeric",
        "params": {"min_value": 1, "max_value": 4, "p_perturb": 0.3},
        "rationale": "Number of existing credits at this bank can "
                     "plausibly go up (new credit) or down (one paid off) "
                     "by a small amount between snapshots.",
    },

    # --- Free nominal categoricals (recourse-relevant) ----------------------
    "checking_status": {
        "kind": "free_categorical", "params": {"p_perturb": 0.4},
        "rationale": "Checking-account balance bracket is directly "
                     "actionable by the applicant (deposit funds).",
    },
    "purpose": {
        "kind": "free_categorical", "params": {"p_perturb": 0.2},
        "rationale": "Loan purpose could plausibly differ on reapplication.",
    },
    "savings_status": {
        "kind": "free_categorical", "params": {"p_perturb": 0.3},
        "rationale": "Savings bracket is actionable (build up savings) "
                     "and not strictly time-ordered enough to enforce a "
                     "one-directional ordinal constraint.",
    },
    "other_parties": {
        "kind": "free_categorical", "params": {"p_perturb": 0.15},
        "rationale": "Adding a co-applicant/guarantor is a common, "
                     "concrete recourse action.",
    },
    "property_magnitude": {
        "kind": "free_categorical", "params": {"p_perturb": 0.1},
        "rationale": "Property ownership status changes slowly but is not "
                     "impossible between snapshots (e.g. between loan "
                     "applications).",
    },
    "other_payment_plans": {
        "kind": "free_categorical", "params": {"p_perturb": 0.15},
        "rationale": "Whether the applicant holds other installment plans "
                     "elsewhere is actionable and time-varying.",
    },
    "housing": {
        "kind": "free_categorical", "params": {"p_perturb": 0.15},
        "rationale": "Housing arrangement (rent/own/free) can change "
                     "between snapshots.",
    },
    "own_telephone": {
        "kind": "free_categorical", "params": {"p_perturb": 0.1},
        "rationale": "Binary nominal, low-impact administrative field.",
    },

    # --- Immutable: historical facts or protected attributes ----------------
    "credit_history": {
        "kind": "immutable", "params": {},
        "rationale": "Represents the applicant's PAST repayment record; "
                     "cannot be retroactively rewritten by a perturbation "
                     "meant to represent a plausible alternate snapshot.",
    },
    "personal_status": {
        "kind": "immutable", "params": {},
        "rationale": "Encodes sex and marital status jointly — a "
                     "protected attribute under fair-lending law; not an "
                     "actionable recourse lever and should never be "
                     "perturbed for an adverse-action-reasons analysis.",
    },
    "job": {
        "kind": "immutable", "params": {},
        "rationale": "Occupational skill category is not a short-term "
                     "recourse lever (unlike, say, account balance).",
    },
    "num_dependents": {
        "kind": "immutable", "params": {},
        "rationale": "Number of dependents changes rarely and is not "
                     "something an applicant can act on for recourse "
                     "purposes; held fixed conservatively.",
    },
    "foreign_worker": {
        "kind": "immutable", "params": {},
        "rationale": "National-origin proxy — protected attribute, never "
                     "perturbed.",
    },
}

# ---------------------------------------------------------------------------
# Give Me Some Credit (GMSC) — all-numeric feature set, no categoricals.
# NOTE: not yet validated end-to-end (raw data pending manual download);
# structure mirrors the German spec and will be sanity-checked once
# data/raw/cs-training.csv is available.
# ---------------------------------------------------------------------------
GMSC_CONSTRAINTS = {
    "age": {
        "kind": "time_anchor_numeric",
        "params": {"max_value": 100},
        "rationale": "Same physical-time argument as German Credit.",
    },
    "RevolvingUtilizationOfUnsecuredLines": {
        "kind": "bounded_numeric_positive_multiplicative",
        "params": {"min_value": 0.0, "max_value": 3.0, "jitter_frac": 0.20, "p_perturb": 0.5},
        "rationale": "A utilization ratio; must stay non-negative, "
                     "conventionally capped well above 1.0 to allow for "
                     "over-limit balances seen in this dataset.",
    },
    "DebtRatio": {
        "kind": "bounded_numeric_positive_multiplicative",
        "params": {"min_value": 0.0, "max_value": 5.0, "jitter_frac": 0.20, "p_perturb": 0.5},
        "rationale": "Debt-to-income ratio must stay non-negative.",
    },
    "MonthlyIncome": {
        "kind": "bounded_numeric_positive_multiplicative",
        "params": {"min_value": 0.0, "max_value": 1_000_000, "jitter_frac": 0.20, "p_perturb": 0.5},
        "rationale": "Income must stay positive — the canonical example "
                     "from the project brief.",
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "kind": "bounded_ordinal_numeric",
        "params": {"min_value": 0, "max_value": 60, "p_perturb": 0.3},
        "rationale": "Count of open lines can plausibly move by a small "
                     "amount between snapshots.",
    },
    "NumberRealEstateLoansOrLines": {
        "kind": "bounded_ordinal_numeric",
        "params": {"min_value": 0, "max_value": 20, "p_perturb": 0.2},
        "rationale": "Count of real-estate-secured lines, small plausible "
                     "movement.",
    },
    "NumberOfDependents": {
        "kind": "immutable", "params": {},
        "rationale": "Held fixed conservatively, as in German Credit.",
    },
    "NumberOfTime30-59DaysPastDueNotWorse": {
        "kind": "immutable", "params": {},
        "rationale": "Historical delinquency count — cannot be "
                     "retroactively changed.",
    },
    "NumberOfTimes90DaysLate": {
        "kind": "immutable", "params": {},
        "rationale": "Historical delinquency count — cannot be "
                     "retroactively changed.",
    },
    "NumberOfTime60-89DaysPastDueNotWorse": {
        "kind": "immutable", "params": {},
        "rationale": "Historical delinquency count — cannot be "
                     "retroactively changed.",
    },
}

# ---------------------------------------------------------------------------
# Taiwan Credit Default (UCI)
# NOTE: not yet validated end-to-end (raw data pending manual download).
# ---------------------------------------------------------------------------
TAIWAN_CONSTRAINTS = {
    "AGE": {
        "kind": "time_anchor_numeric",
        "params": {"max_value": 100},
        "rationale": "Same physical-time argument as German Credit.",
    },
    "LIMIT_BAL": {
        "kind": "bounded_numeric_positive_multiplicative",
        "params": {"min_value": 1000, "max_value": 2_000_000, "jitter_frac": 0.20, "p_perturb": 0.5},
        "rationale": "Credit limit must stay strictly positive.",
    },
    # PAY_0..PAY_6: past monthly repayment status codes -> historical, immutable
    **{f"PAY_{i}": {
        "kind": "immutable", "params": {},
        "rationale": "Past monthly repayment status is a historical "
                     "record and cannot be retroactively changed.",
    } for i in [0, 2, 3, 4, 5, 6]},
    # BILL_AMT1..6: past statement balances -> historical, immutable
    **{f"BILL_AMT{i}": {
        "kind": "immutable", "params": {},
        "rationale": "Past billing-cycle statement balance is a "
                     "historical record.",
    } for i in range(1, 7)},
    # PAY_AMT1..6: past payments made -> historical, immutable
    **{f"PAY_AMT{i}": {
        "kind": "immutable", "params": {},
        "rationale": "Past payment amounts are a historical record.",
    } for i in range(1, 7)},
    "SEX": {
        "kind": "immutable", "params": {},
        "rationale": "Protected attribute — never perturbed.",
    },
    "MARRIAGE": {
        "kind": "immutable", "params": {},
        "rationale": "Protected attribute (marital status) — never "
                     "perturbed.",
    },
    "EDUCATION": {
        "kind": "immutable", "params": {},
        "rationale": "Treated conservatively as not a short-term "
                     "recourse lever.",
    },
}

CONSTRAINTS_BY_DATASET = {
    "german": GERMAN_CONSTRAINTS,
    "gmsc": GMSC_CONSTRAINTS,
    "taiwan": TAIWAN_CONSTRAINTS,
}
