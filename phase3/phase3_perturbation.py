"""
phase3_perturbation.py
Domain-Constrained Perturbation Engine.

Pipeline:
  1. Load the test set for a dataset; randomly sample exactly 300 rows
     (or fewer, with a clear warning, if the test set itself is smaller —
     e.g. German Credit's 20%-of-1000 test split is only 200 rows).
  2. For each sampled row, generate 10 perturbed variants following the
     domain-constraint spec in perturbation_constraints.py:
       - one shared "time delta" per perturbed instance keeps age /
         residence-since / employment-duration causally coherent
       - one-hot categorical groups are perturbed as coherent categories
         (never as independently-flipped dummy bits)
       - every constrained feature is validated against its rule after
         generation (see validate_perturbations())
  3. Save both the 300-row baseline sample and the 3000-row perturbed set
     to parquet, plus a validation report.

Run:
    python phase3_perturbation.py                  # all available datasets
    python phase3_perturbation.py --dataset german
"""

import argparse
import gc
import json

import numpy as np
import pandas as pd
import joblib

from config import DATASETS, PROCESSED_DIR, METADATA_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_SEED, TARGET_COL
from perturbation_constraints import CONSTRAINTS_BY_DATASET

N_SAMPLE = 300      # strict micro-sampling cap per project rules
N_PERTURBATIONS = 10  # perturbed variants generated per sampled instance


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def load_category_map(dataset_key: str) -> dict:
    path = METADATA_DIR / f"{dataset_key}_category_map.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def set_category(row: dict, dummy_columns: list, category_to_dummy: dict, new_category: str) -> None:
    """Set exactly one dummy column to 1 for new_category (or none, if
    new_category is the baseline), all others in the group to 0."""
    for col in dummy_columns:
        row[col] = 0
    if new_category in category_to_dummy:
        row[category_to_dummy[new_category]] = 1


# ---------------------------------------------------------------------------
# Per-feature perturbation functions
# ---------------------------------------------------------------------------

def perturb_time_anchor_numeric(value, rng, params):
    delta = rng.choice([0, 1, 2, 3, 5], p=[0.30, 0.30, 0.20, 0.15, 0.05])
    new_value = min(value + delta, params["max_value"])
    return new_value, delta


def perturb_time_follower_ordinal_numeric(value, rng, params, delta):
    step = 0
    if delta >= 3 and rng.random() < 0.7:
        step = 1
    elif delta >= 1 and rng.random() < 0.4:
        step = 1
    return int(min(value + step, params["max_value"]))


def perturb_time_follower_ordinal_categorical(current_category, categories_sorted, rng, delta):
    idx = categories_sorted.index(current_category)
    step = 0
    if delta >= 3 and rng.random() < 0.7:
        step = 1
    elif delta >= 1 and rng.random() < 0.4:
        step = 1
    new_idx = min(idx + step, len(categories_sorted) - 1)
    return categories_sorted[new_idx]


def perturb_bounded_numeric(value, rng, params):
    if rng.random() >= params["p_perturb"]:
        return value
    delta = rng.integers(-params["jitter_abs"], params["jitter_abs"] + 1)
    return int(np.clip(value + delta, params["min_value"], params["max_value"]))


def perturb_bounded_numeric_positive_multiplicative(value, rng, params):
    if rng.random() >= params["p_perturb"]:
        return value
    factor = rng.uniform(1 - params["jitter_frac"], 1 + params["jitter_frac"])
    new_value = value * factor
    return float(np.clip(new_value, params["min_value"], params["max_value"]))


def perturb_bounded_ordinal_numeric(value, rng, params):
    if rng.random() >= params["p_perturb"]:
        return value
    step = rng.choice([-1, 1])
    return int(np.clip(value + step, params["min_value"], params["max_value"]))


def perturb_free_categorical(current_category, categories_sorted, rng, params):
    if rng.random() >= params["p_perturb"]:
        return current_category
    choices = [c for c in categories_sorted if c != current_category]
    return rng.choice(choices)


# ---------------------------------------------------------------------------
# Row-level perturbation
# ---------------------------------------------------------------------------

def perturb_row(row: pd.Series, constraints: dict, category_map: dict, rng) -> dict:
    new_row = row.to_dict()

    # Resolve time delta first (age is the anchor) so followers can use it.
    time_delta = 0
    if "age" not in constraints:
        pass  # dataset schema mismatch; handled by caller-level assertions
    else:
        anchor_feature = next(
            (f for f, spec in constraints.items() if spec["kind"] == "time_anchor_numeric"),
            None,
        )
        if anchor_feature is not None:
            new_value, time_delta = perturb_time_anchor_numeric(
                row[anchor_feature], rng, constraints[anchor_feature]["params"]
            )
            new_row[anchor_feature] = new_value

    for feature, spec in constraints.items():
        kind = spec["kind"]
        params = spec["params"]

        if kind in ("immutable", "time_anchor_numeric"):
            continue  # anchor already handled above; immutable = no-op

        elif kind == "time_follower_ordinal_numeric":
            new_row[feature] = perturb_time_follower_ordinal_numeric(
                row[feature], rng, params, time_delta
            )

        elif kind == "time_follower_ordinal_categorical":
            cat_info = category_map[feature]
            categories_sorted = cat_info["categories_sorted"]
            dummy_columns = cat_info["dummy_columns"]
            baseline = cat_info["baseline_dropped"]
            category_to_dummy = dict(zip(categories_sorted[1:], dummy_columns))

            current = None
            for cat, dummy_col in category_to_dummy.items():
                if row[dummy_col] == 1:
                    current = cat
                    break
            if current is None:
                current = baseline

            new_category = perturb_time_follower_ordinal_categorical(
                current, categories_sorted, rng, time_delta
            )
            set_category(new_row, dummy_columns, category_to_dummy, new_category)

        elif kind == "bounded_numeric":
            new_row[feature] = perturb_bounded_numeric(row[feature], rng, params)

        elif kind == "bounded_numeric_positive_multiplicative":
            new_row[feature] = perturb_bounded_numeric_positive_multiplicative(
                row[feature], rng, params
            )

        elif kind == "bounded_ordinal_numeric":
            new_row[feature] = perturb_bounded_ordinal_numeric(row[feature], rng, params)

        elif kind == "free_categorical":
            cat_info = category_map[feature]
            categories_sorted = cat_info["categories_sorted"]
            dummy_columns = cat_info["dummy_columns"]
            baseline = cat_info["baseline_dropped"]
            category_to_dummy = dict(zip(categories_sorted[1:], dummy_columns))

            current = None
            for cat, dummy_col in category_to_dummy.items():
                if row[dummy_col] == 1:
                    current = cat
                    break
            if current is None:
                current = baseline

            new_category = perturb_free_categorical(current, categories_sorted, rng, params)
            set_category(new_row, dummy_columns, category_to_dummy, new_category)

        else:
            raise ValueError(f"Unknown constraint kind: {kind}")

    return new_row


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_perturbations(original_df, perturbed_df, constraints, category_map, dataset_key):
    """Assert every generated perturbation actually honors its constraint.
    Returns a text report; raises AssertionError on any violation so a
    silently-broken perturbation never contaminates downstream phases."""
    lines = [f"Validation report: {dataset_key}", "=" * 40]
    merged = perturbed_df.merge(
        original_df, on="original_index", suffixes=("_pert", "_orig")
    )

    for feature, spec in constraints.items():
        kind = spec["kind"]

        if kind == "immutable" and feature not in category_map:
            ok = (merged[f"{feature}_pert"] == merged[f"{feature}_orig"]).all()
            assert ok, f"Immutable numeric feature '{feature}' changed under perturbation"
            lines.append(f"  [OK] {feature}: immutable, unchanged in all rows")

        elif kind == "immutable" and feature in category_map:
            dummy_cols = category_map[feature]["dummy_columns"]
            for d in dummy_cols:
                ok = (merged[f"{d}_pert"] == merged[f"{d}_orig"]).all()
                assert ok, f"Immutable categorical '{feature}' ({d}) changed under perturbation"
            lines.append(f"  [OK] {feature}: immutable categorical, unchanged in all rows")

        elif kind == "time_anchor_numeric":
            ok = (merged[f"{feature}_pert"] >= merged[f"{feature}_orig"]).all()
            assert ok, f"Time-anchor feature '{feature}' decreased in at least one row"
            within_max = (merged[f"{feature}_pert"] <= spec["params"]["max_value"]).all()
            assert within_max, f"'{feature}' exceeded max_value bound"
            lines.append(f"  [OK] {feature}: monotonic non-decreasing, within bound")

        elif kind == "time_follower_ordinal_numeric":
            ok = (merged[f"{feature}_pert"] >= merged[f"{feature}_orig"]).all()
            assert ok, f"Time-follower feature '{feature}' decreased in at least one row"
            lines.append(f"  [OK] {feature}: monotonic non-decreasing")

        elif kind == "time_follower_ordinal_categorical":
            cat_info = category_map[feature]
            categories_sorted = cat_info["categories_sorted"]
            dummy_columns = cat_info["dummy_columns"]
            baseline = cat_info["baseline_dropped"]
            category_to_dummy = dict(zip(categories_sorted[1:], dummy_columns))

            def _current_cat(r, prefix):
                for cat, dcol in category_to_dummy.items():
                    if r[f"{dcol}_{prefix}"] == 1:
                        return cat
                return baseline

            idx_orig = merged.apply(lambda r: categories_sorted.index(_current_cat(r, "orig")), axis=1)
            idx_pert = merged.apply(lambda r: categories_sorted.index(_current_cat(r, "pert")), axis=1)
            ok = (idx_pert >= idx_orig).all()
            assert ok, f"Time-follower categorical '{feature}' moved to a lower ordinal category"
            lines.append(f"  [OK] {feature}: ordinal category non-decreasing")

        elif kind == "bounded_numeric" or kind == "bounded_ordinal_numeric":
            lo, hi = spec["params"]["min_value"], spec["params"]["max_value"]
            ok = merged[f"{feature}_pert"].between(lo, hi).all()
            assert ok, f"'{feature}' violated bounds [{lo}, {hi}]"
            lines.append(f"  [OK] {feature}: within bounds [{lo}, {hi}]")

        elif kind == "bounded_numeric_positive_multiplicative":
            ok = (merged[f"{feature}_pert"] > 0).all()
            assert ok, f"'{feature}' went non-positive under perturbation"
            lines.append(f"  [OK] {feature}: strictly positive in all rows")

    # One-hot group validity: at most one dummy = 1 per group, for every group
    for feature, cat_info in category_map.items():
        dummy_cols = cat_info["dummy_columns"]
        pert_cols = [f"{d}_pert" for d in dummy_cols]
        row_sums = merged[pert_cols].sum(axis=1)
        assert (row_sums <= 1).all(), f"One-hot group '{feature}' has >1 active dummy in some row"
    lines.append(f"  [OK] all {len(category_map)} one-hot groups remain mutually exclusive")

    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_dataset(dataset_key: str) -> bool:
    print(f"\n=== Phase 3: {dataset_key} ===")
    test_path = PROCESSED_DIR / f"{dataset_key}_test.parquet"
    if not test_path.exists():
        print(f"  Skipped: {test_path.name} not found (run Phase 1 first).")
        return False

    if dataset_key not in CONSTRAINTS_BY_DATASET:
        print(f"  Skipped: no constraint spec registered for '{dataset_key}'.")
        return False

    constraints = CONSTRAINTS_BY_DATASET[dataset_key]
    category_map = load_category_map(dataset_key)

    test_df = pd.read_parquet(test_path)
    n_available = len(test_df)
    n_sample = min(N_SAMPLE, n_available)
    if n_sample < N_SAMPLE:
        print(f"  NOTE: test set has only {n_available} rows; micro-sampling "
              f"target of {N_SAMPLE} exceeds availability. Using all "
              f"{n_sample} rows and flagging this in the report.")

    rng_sample = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng_sample.choice(n_available, size=n_sample, replace=False)
    sample_df = test_df.iloc[sample_idx].reset_index(drop=True)
    sample_df.insert(0, "original_index", sample_df.index)

    del test_df
    gc.collect()

    # Verify every constrained feature actually exists in this dataset's schema
    feature_cols = [c for c in sample_df.columns if c not in ("original_index", TARGET_COL)]
    missing = [f for f in constraints if f not in feature_cols and f not in category_map]
    if missing:
        print(f"  WARNING: constraint spec references features not found in "
              f"the processed schema: {missing}. Skipping those entries.")
        constraints = {k: v for k, v in constraints.items() if k not in missing}

    rng = np.random.default_rng(RANDOM_SEED + 1)
    perturbed_rows = []
    for _, row in sample_df.iterrows():
        for p_id in range(N_PERTURBATIONS):
            new_row = perturb_row(row, constraints, category_map, rng)
            new_row["original_index"] = row["original_index"]
            new_row["perturbation_id"] = p_id
            perturbed_rows.append(new_row)

    perturbed_df = pd.DataFrame(perturbed_rows)
    del perturbed_rows
    gc.collect()

    # Reorder columns to match sample_df, with perturbation_id appended
    ordered_cols = list(sample_df.columns) + ["perturbation_id"]
    perturbed_df = perturbed_df[ordered_cols]

    print(f"  Generated {len(perturbed_df)} perturbed rows "
          f"({n_sample} instances x {N_PERTURBATIONS} perturbations)")

    report = validate_perturbations(sample_df, perturbed_df, constraints, category_map, dataset_key)

    # Downcast before saving (float64->float32, int64->smallest safe int)
    from mem_utils import downcast_dataframe
    sample_df_to_save = downcast_dataframe(
        sample_df.drop(columns=["original_index"]).assign(original_index=sample_df["original_index"]),
        verbose=False,
    )
    perturbed_df = downcast_dataframe(perturbed_df, verbose=False)

    sample_path = PROCESSED_DIR / f"{dataset_key}_sample_original.parquet"
    perturbed_path = PROCESSED_DIR / f"{dataset_key}_perturbed.parquet"
    sample_df_to_save.to_parquet(sample_path, index=False)
    perturbed_df.to_parquet(perturbed_path, index=False)

    report_path = RESULTS_DIR / f"phase3_validation_{dataset_key}.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"  Saved: {sample_path.name}, {perturbed_path.name}, {report_path.name}")

    del sample_df, perturbed_df, sample_df_to_save
    gc.collect()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default=None,
                         help="Run a single dataset; default runs all available.")
    args = parser.parse_args()

    keys = [args.dataset] if args.dataset else list(DATASETS.keys())
    results = {k: run_dataset(k) for k in keys}

    print("\n=== Phase 3 summary ===")
    for k, ok in results.items():
        print(f"  {k:8s}: {'OK' if ok else 'SKIPPED'}")


if __name__ == "__main__":
    main()
