# Stage 1.5 Corrective Pass — XAI-StabilityNet

**Date:** 2026-09-24  
**Branch:** `stage1.5-corrective-pass`  
**Pre-Stage-1.5 checkpoint commit:** `f29adc9`

---

## A. Why Stage 1.5 Was Required

The Stage-1 audit of the preliminary prototype identified five defects that must
be corrected before the project can produce final experimental evidence.

### RF-1 — Inconsistent perturbation ontology between Phase 3 and Phase 4

`phase3/perturbation_constraints.py` marked the following features as **immutable**:

| Dataset | Features |
|---------|----------|
| German  | `num_dependents` |
| Taiwan  | `PAY_0..PAY_6`, `BILL_AMT1..6`, `PAY_AMT1..6` (18 features) |
| GMSC    | `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfDependents` |

`phase4/phase4_perturbation.py` `PERTURBABLE_FEATURES` perturbed **all of them anyway**.

This created contradictory constraints in the same repository and contaminated
the Phase-4/Phase-5 preliminary results.

**Evidence:** `phase4/phase4_perturbation.py` L81-128 vs `phase3/perturbation_constraints.py` L240-303.

### I-3 — Perturbation validity checks performed only for German

`phase3/phase3_validation_german.txt` existed. No equivalent for Taiwan or GMSC.
`perturbation_constraints.py` explicitly marked both as "not yet validated end-to-end".

### I-4 — Sign-inversion bug for negative-valued features

The old perturbation formula used `abs(original) * abs(magnitude)`, then
conditionally added or subtracted the result. For negative originals, the
named perturbation directions were inverted:
- `relative_plus_5` on `x=-1000` → `−1000+50 = −950` (should be `−1050`)
- `relative_minus_5` on `x=-1000` → `−1000−50 = −1050` (should be `−950`)

**Evidence:** `phase4/phase4_perturbation.py` L282-291.

### I-7 — Global warning suppression in phase3_explain.py

`warnings.filterwarnings("ignore")` at L43 silenced all SHAP, LIME, LightGBM
and NumPy warnings globally. This hid potential convergence issues in
KernelSHAP and LIME, making future debugging harder.

### I-8 — Dangling unused SEED in phase4_perturbation.py

`SEED = 42` was declared at L54 of the old Phase-4 script but never passed
to any random-number generator. The script contained no stochastic operations,
so this was harmless but misleading.

---

## B. Canonical Perturbation Ontology

**File:** `phase3/perturbation_schema.py`

This module is the **single source of truth** for all perturbation decisions.
It replaces both prior sources.

### Feature roles

| Role | Meaning |
|------|---------|
| `perturbable_for_local_robustness` | Feature is explicitly varied ±5%/±10% by the Phase-4 numeric engine to test model sensitivity. |
| `non_perturbable` | Feature must never be modified. Reasons include: historical record, protected attribute, integer-count with no valid fractional state. |
| `categorical_engine_only` | Feature is in principle changeable but only through the Phase-3 categorical perturbation engine (one-hot group switching). The Phase-4 numeric engine must ignore it. |

### Perturbation rules

| Rule | Formula / Behavior |
|------|--------------------|
| `relative_signed` | `x' = x(1+r)`, where `r ∈ {−0.10, −0.05, +0.05, +0.10}`. See §D. |
| `ordinal_step` | `x' = clip(x ± 1, lo, hi)`. Direction follows sign of `r`. |
| `time_anchor` | Age-type: `x' = clip(x + |x|·r, x, hi)`. Can only advance. |
| `time_follower` | Categorical engine only. Phase-4 engine ignores these. |
| `none` | Feature is non_perturbable; never modified. |

---

## C. Disputed Feature Decisions

### German — `num_dependents`

- **Old Phase-3:** immutable  
- **Old Phase-4:** perturbable (relative ±%)  
- **Stage-1.5 decision:** `non_perturbable`  
- **Rationale:** `num_dependents` is an integer count (observed values: 1 or 2).
  Applying relative ±5% produces fractional dependents (e.g., 1.05), which is
  not a valid state. Additionally, number of dependents is not a short-term
  actionable lever in a local-robustness experiment.

### Taiwan — `PAY_0..PAY_6`

- **Old Phase-3:** immutable  
- **Old Phase-4:** perturbable (relative ±%)  
- **Stage-1.5 decision:** `non_perturbable`  
- **Rationale:** These are monthly repayment **status codes** (ordinal −2..8),
  recording past billing history. Past events cannot be retroactively changed.
  Perturbing historical status codes does not constitute a valid local-robustness
  experiment; it modifies a factual record.

### Taiwan — `BILL_AMT1..BILL_AMT6`

- **Old Phase-3:** immutable  
- **Old Phase-4:** perturbable (relative ±%)  
- **Stage-1.5 decision:** `non_perturbable`  
- **Rationale:** Past statement balances are historical facts. Additionally,
  BILL_AMT can be negative (credit overpayments). The old formula was
  sign-inverted for negative values (audit I-4), so these rows were doubly
  incorrect.

### Taiwan — `PAY_AMT1..PAY_AMT6`

- **Old Phase-3:** immutable  
- **Old Phase-4:** perturbable (relative ±%)  
- **Stage-1.5 decision:** `non_perturbable`  
- **Rationale:** Past payment amounts are historical records. Cannot be changed.

### GMSC — Delinquency counts (3 features)

- **Old Phase-3:** immutable  
- **Old Phase-4:** perturbable  
- **Stage-1.5 decision:** `non_perturbable`  
- **Rationale:** `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`,
  and `NumberOfTime60-89DaysPastDueNotWorse` record the count of past delinquency
  events. These are historical facts that cannot be retroactively altered.

### GMSC — `NumberOfDependents`

- **Old Phase-3:** immutable  
- **Old Phase-4:** perturbable  
- **Stage-1.5 decision:** `non_perturbable`  
- **Rationale:** Same as German `num_dependents`: integer count, fractional
  perturbation is invalid, not an actionable short-term lever.

---

## D. Perturbation Mathematics

### Relative signed perturbation (main formula)

For features with `perturbation_rule = "relative_signed"`:

```
x' = x * (1 + r)
```

where `r ∈ {−0.10, −0.05, +0.05, +0.10}`.

This formula is correct for **both positive and negative originals**:

| x | r | x' | Meaning |
|---|---|----|---------|
| +1000 | +0.05 | +1050 | 5% increase |
| +1000 | −0.05 | +950  | 5% decrease |
| −1000 | +0.05 | −1050 | 5% scale-up (more negative) |
| −1000 | −0.05 | −950  | 5% scale-down (less negative) |

This is **multiplicative scaling**: `r > 0` scales the magnitude up, `r < 0`
scales it down. The old formula used `abs(original) * abs(magnitude)` and
then conditionally added or subtracted, which produced the opposite direction
for negative values.

**Zero-value handling:** If `|x| < 1e-12`, the formula degenerates to `x' = 0`.
A small absolute fallback nudge is applied using the column median as scale.

**Bounds:** After perturbation, values are clipped to `[lo, hi]` per the schema.

### Ordinal step perturbation

For features with `perturbation_rule = "ordinal_step"`:

```
x' = clip(x + step, lo, hi)
```

where `step = +1` if `r > 0`, `step = -1` if `r < 0`. Output is rounded to integer.

### Time-anchor perturbation

For age-type features (`perturbation_rule = "time_anchor"`, only positive rates):

```
x' = clip(x + |x|·r, x, hi)
```

Negative rates produce no change (age cannot decrease).

---

## E. Baseline Contamination

The preliminary Phase-4 results (`results/phase4/`) were produced using the
inconsistent ontology. The quantitative impact is documented in:

```
audit/phase4_baseline_contamination.csv
audit/phase4_baseline_contamination.md
```

| Dataset | Affected features | Affected rows | Total rows | Affected % |
|---------|------------------:|--------------:|-----------:|-----------:|
| German  | 1 | 1,600 | 11,200 | 14.3% |
| Taiwan  | 18 | 28,800 | 32,000 | 90.0% |
| GMSC    | 4 | 6,400 | 16,000 | 40.0% |

For Taiwan, 90% of all Phase-4 rows correspond to features that should have
been non-perturbable. Only `LIMIT_BAL` and `AGE` were legitimately perturbable
in Taiwan under the corrected schema.

---

## F. Validation

Three dataset validation reports were generated by inspecting the Phase-3
explanation instances against the canonical schema and simulating perturbation
output for each perturbable feature and each of the four rate values.

Results: **All three datasets PASS.**

| Dataset | Schema | Bounds | Integer | One-hot | Domain sim | Overall |
|---------|--------|--------|---------|---------|------------|---------|
| German  | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Taiwan  | PASS | PASS | N/A  | N/A  | PASS | **PASS** |
| GMSC    | PASS | PASS | PASS | N/A  | PASS | **PASS** |

Reports in:
```
results/validation/german_perturbation_validation.{txt,json}
results/validation/taiwan_perturbation_validation.{txt,json}
results/validation/gmsc_perturbation_validation.{txt,json}
results/validation/SUPERSEDED_phase3_validation_german.txt
```

---

## G. Test Methodology

Two test modules cover the corrective pass:

### `tests/test_perturbation_engine.py` (20 tests)

Tests the perturbation math functions in isolation:

| Test | What it proves |
|------|---------------|
| T01/T01b | Positive originals perturbed correctly |
| T02/T02b | Negative originals: `x*(1+r)` formula correct (I-4 fix) |
| T03 | Zero-value fallback produces finite output |
| T04–T07 | Ordinal step: direction, bounds, integer output |
| T08 | Relative perturbation: schema bounds enforced |
| T09 | Engine loop does not touch non-perturbable columns (runs engine, checks bytes) |
| T09b | Non-perturbable schema entries all have `rule="none"` |
| T10 | Schema dict excludes unknown/fabricated feature names |
| T11 | Determinism: same input → same output |
| T12 | Every perturbable feature has a valid perturbation rule |
| T13 | No feature in both perturbable and non-perturbable sets |
| T14–T16 | RF-1 regression: disputed features absent from corrected canonical list |

### `tests/test_cross_phase_consistency.py` (19 tests)

Tests against real Phase-3 explanation instances:

| Test | What it proves |
|------|---------------|
| C01 | Schema covers all three datasets |
| C02 | Every perturbable feature exists in the data |
| C03 | Engine simulation: non-perturbable cols unchanged when engine runs |
| C04 | German one-hot groups: row-sum ≤ 1 in all instances |
| C05 | Simulated perturbations respect schema bounds for all 4 rates |
| C06 | Same input → identical output |
| V01 | German perturbable features: finite, no-NaN, within bounds |
| V02 | German integer features: integer-valued in instance file |
| V03 | German one-hot groups (alias of C04) |
| V04 | Taiwan perturbable features exist and are numeric |
| V05 | GMSC perturbable features exist and are numeric |
| V06 | Non-perturbable numeric features present in instance CSVs |

**Final result: 39/39 tests pass.**

---

## H. Historical Baseline Results

> The original Phase-4/Phase-5 results are retained as **preliminary historical
> baseline results**. They are not treated as final evidence because the
> Stage-1.5 audit identified methodological defects in the perturbation protocol,
> most critically the inconsistent ontology (RF-1) and the sign-inversion bug
> (I-4). The files in `results/phase4/` and `results/phase5/` have not been
> modified and remain available for historical reference.

The preliminary overall ESS = 0.7515 was reproduced exactly (Δ = 1.11×10⁻¹⁶)
from the saved intermediates by `audit/reproduce_ess.py`. It is an internally
consistent result **given those intermediate files**, but those intermediates
were produced by the defective prototype.

---

## I. Deferred Stage-2 Changes

The following items are NOT yet implemented and belong to Stage 2 or later:

- SHAP output-space correction (TreeExplainer output interpretation)
- SHAP background sample standardisation
- KernelSHAP convergence / `nsamples` validation
- LIME categorical feature handling
- LIME neighborhood bandwidth correction
- LIME contribution conversion (coefficient vs absolute value)
- LIME stochasticity control (fixed-seed evaluation across seeds)
- Multi-seed model training (RQ1)
- Explanation recomputation after perturbation (RQ4)
- Prediction-preserving perturbation analysis (RQ4)
- Explanation drift metrics (RQ4)
- ESS formula redesign or freeze with ablation
- ESS sensitivity / weight analysis (Stage 3)
- Bootstrap confidence intervals (Stage 3)
- Faithfulness evaluation (Stage 4)
- Semantic mapping validation (Stage 4)
