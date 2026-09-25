from __future__ import annotations

from pathlib import Path
from datetime import datetime
import ast
import hashlib
import inspect
import itertools
import json
import re
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, friedmanchisquare, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()
STAGE2 = ROOT / "results" / "stage2"
STAGE3 = ROOT / "results" / "stage3"
PLOTS = STAGE3 / "plots"

DATASETS = ["german", "taiwan", "gmsc"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]
SEEDS = [42, 0, 7, 123, 2024]
RATES = [-0.10, -0.05, 0.05, 0.10]

ROBUST_METRICS = [
    "prediction_drift",
    "shap_rank_sim",
    "shap_attr_sim",
    "shap_sign_cons",
    "lime_rank_sim",
    "lime_attr_sim",
    "lime_sign_cons",
]

EXPL_METRICS = ["rank_sim", "attr_sim", "sign_cons", "topk"]

META_EXPL = {
    "instance_index", "actual_default", "predicted_prob",
    "dataset", "model", "seed", "duration"
}

BOOTSTRAPS = 5000
PERMUTATIONS = 10000
BOOT_SEED = 4262026
PERM_SEED = 4262027
PLAUSIBILITY_Z = 4.0


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, str]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan, np.nan, "NOT_ESTIMABLE_N_LT_2"

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    boot_means = values[idx].mean(axis=1)

    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi), "OK"


def summarize_instance_values(
    values: pd.Series | np.ndarray,
    seed_parts: tuple[object, ...],
) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_status": "NO_VALID_VALUES",
        }

    ci_low, ci_high, ci_status = bootstrap_mean_ci(
        arr,
        stable_seed(*seed_parts),
    )

    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_status": ci_status,
    }


def load_exact_metric_helpers():
    """
    Extract the existing metric helper function definitions from
    stage3_runner_fixed.py without executing its analysis code.
    This preserves the existing Stage 2/Stage 3 metric definitions.
    """
    source_path = ROOT / "stage3_runner_fixed.py"
    if not source_path.exists():
        raise FileNotFoundError(
            "stage3_runner_fixed.py not found. "
            "The existing metric definitions are required."
        )

    tree = ast.parse(
        source_path.read_text(encoding="utf-8-sig", errors="ignore")
    )

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def choose(kind: str):
        candidates = []
        for name in functions:
            low = name.lower()
            if kind == "rank":
                ok = "rank" in low and "boot" not in low and "top" not in low
            elif kind == "attr":
                ok = "attrib" in low
            elif kind == "sign":
                ok = "sign" in low
            elif kind == "topk":
                ok = "topk" in low or "top_k" in low
            else:
                ok = False

            if ok:
                candidates.append(name)

        if not candidates:
            raise RuntimeError(
                f"Could not locate existing {kind} metric helper in "
                f"stage3_runner_fixed.py. Found functions: {sorted(functions)}"
            )

        return candidates[0]

    wanted_names = {
        "rank": choose("rank"),
        "attr": choose("attr"),
        "sign": choose("sign"),
        "topk": choose("topk"),
    }

    namespace = {
        "np": np,
        "pd": pd,
        "spearmanr": spearmanr,
    }

    for key, name in wanted_names.items():
        node = functions[name]
        code = compile(
            ast.Module(body=[node], type_ignores=[]),
            filename=str(source_path),
            mode="exec",
        )
        exec(code, namespace)

    print("Metric helpers loaded from:", source_path)
    print("  rank :", wanted_names["rank"])
    print("  attr :", wanted_names["attr"])
    print("  sign :", wanted_names["sign"])
    print("  topk :", wanted_names["topk"])

    return (
        namespace[wanted_names["rank"]],
        namespace[wanted_names["attr"]],
        namespace[wanted_names["sign"]],
        namespace[wanted_names["topk"]],
    )


RANK_SIM, ATTR_SIM, SIGN_CONS, TOPK = load_exact_metric_helpers()


def metric_values(a: np.ndarray, b: np.ndarray) -> dict:
    out = {
        "rank_sim": float(RANK_SIM(a, b)),
        "attr_sim": float(ATTR_SIM(a, b)),
        "sign_cons": float(SIGN_CONS(a, b)),
    }

    try:
        out["topk"] = float(TOPK(a, b))
    except TypeError:
        out["topk"] = float(TOPK(a, b, 10))

    return out


def parse_explanation_files():
    pattern = re.compile(
        r"^(german|taiwan|gmsc)_"
        r"(xgboost|lightgbm|random_forest|mlp)_"
        r"seed(\d+)_"
        r"(shap|lime_contrib)\.parquet$"
    )

    store = {}
    feature_orders = {}

    files = sorted((STAGE2 / "explanations").glob("*.parquet"))

    for f in files:
        m = pattern.match(f.name)
        if not m:
            continue

        ds, model, seed_text, expl = m.groups()
        seed = int(seed_text)

        df = pd.read_parquet(f)

        expected_meta = META_EXPL
        features = [c for c in df.columns if c not in expected_meta]

        if "instance_index" not in df.columns:
            raise ValueError(f"{f.name}: missing instance_index")

        if len(df) != 50:
            raise ValueError(f"{f.name}: expected 50 rows, got {len(df)}")

        key = (ds, model, expl)

        if key not in feature_orders:
            feature_orders[key] = features
        elif feature_orders[key] != features:
            raise ValueError(
                f"Feature-order mismatch for {key}: {f.name}"
            )

        store[(ds, model, expl, seed)] = df.sort_values(
            "instance_index"
        ).reset_index(drop=True)

    expected_count = 3 * 4 * 2 * 5
    if len(store) != expected_count:
        raise ValueError(
            f"Expected {expected_count} explanation files, found {len(store)}"
        )

    for ds in DATASETS:
        for model in MODELS:
            for expl in ["shap", "lime_contrib"]:
                expected_ids = None

                for seed in SEEDS:
                    df = store[(ds, model, expl, seed)]
                    ids = tuple(df["instance_index"].tolist())

                    if expected_ids is None:
                        expected_ids = ids
                    elif ids != expected_ids:
                        raise ValueError(
                            f"Instance mismatch for {ds}/{model}/{expl}/seed={seed}"
                        )

    return store, feature_orders


def build_seed_stability(store, feature_orders):
    rows = []

    for ds in DATASETS:
        for model in MODELS:
            for expl in ["shap", "lime_contrib"]:
                features = feature_orders[(ds, model, expl)]

                for seed1, seed2 in itertools.combinations(SEEDS, 2):
                    df1 = store[(ds, model, expl, seed1)]
                    df2 = store[(ds, model, expl, seed2)]

                    for i in range(50):
                        a = df1.loc[i, features].to_numpy(float)
                        b = df2.loc[i, features].to_numpy(float)
                        metrics = metric_values(a, b)

                        rows.append({
                            "dataset": ds,
                            "model": model,
                            "explainer": "lime" if expl == "lime_contrib" else "shap",
                            "seed1": seed1,
                            "seed2": seed2,
                            "instance_index": int(df1.loc[i, "instance_index"]),
                            "rank_sim": metrics["rank_sim"],
                            "attr_sim": metrics["attr_sim"],
                            "sign_cons": metrics["sign_cons"],
                            "topk": metrics["topk"],
                        })

    pairwise = pd.DataFrame(rows)

    if len(pairwise) != 12000:
        raise ValueError(
            f"Seed stability expected 12000 rows, got {len(pairwise)}"
        )

    summary_rows = []

    for (ds, model, expl), grp in pairwise.groupby(
        ["dataset", "model", "explainer"],
        sort=True,
    ):
        for metric in EXPL_METRICS:
            instance_values = grp.groupby(
                "instance_index",
                sort=True,
            )[metric].mean()

            stats = summarize_instance_values(
                instance_values,
                ("seed_stability", ds, model, expl, metric),
            )

            summary_rows.append({
                "dataset": ds,
                "model": model,
                "explainer": expl,
                "metric": metric,
                "n": stats["n"],
                "n_rows": int(len(grp)),
                "mean": stats["mean"],
                "median": stats["median"],
                "std": stats["std"],
                "q25": stats["q25"],
                "q75": stats["q75"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "ci_status": stats["ci_status"],
                "bootstrap_n": BOOTSTRAPS,
                "bootstrap_unit": "instance",
            })

    summary = pd.DataFrame(summary_rows)

    if len(summary) != 96:
        raise ValueError(
            f"Seed stability summary expected 96 rows, got {len(summary)}"
        )

    return pairwise, summary


def build_shap_lime(store, feature_orders):
    rows = []

    for ds in DATASETS:
        for model in MODELS:
            features_shap = feature_orders[(ds, model, "shap")]
            features_lime = feature_orders[(ds, model, "lime_contrib")]

            if features_shap != features_lime:
                raise ValueError(
                    f"SHAP/LIME feature mismatch for {ds}/{model}"
                )

            for seed in SEEDS:
                shap_df = store[(ds, model, "shap", seed)]
                lime_df = store[(ds, model, "lime_contrib", seed)]

                if not np.array_equal(
                    shap_df["instance_index"].to_numpy(),
                    lime_df["instance_index"].to_numpy(),
                ):
                    raise ValueError(
                        f"SHAP/LIME instance mismatch for {ds}/{model}/seed={seed}"
                    )

                for i in range(50):
                    a = shap_df.loc[i, features_shap].to_numpy(float)
                    b = lime_df.loc[i, features_lime].to_numpy(float)

                    metrics = metric_values(a, b)

                    rows.append({
                        "dataset": ds,
                        "model": model,
                        "seed": seed,
                        "instance_index": int(
                            shap_df.loc[i, "instance_index"]
                        ),
                        "rank_sim": metrics["rank_sim"],
                        "attr_sim": metrics["attr_sim"],
                        "sign_cons": metrics["sign_cons"],
                        "topk": metrics["topk"],
                    })

    detail = pd.DataFrame(rows)

    if len(detail) != 3000:
        raise ValueError(
            f"SHAP/LIME consistency expected 3000 rows, got {len(detail)}"
        )

    summary_rows = []

    for (ds, model), grp in detail.groupby(
        ["dataset", "model"],
        sort=True,
    ):
        for metric in EXPL_METRICS:
            instance_values = grp.groupby(
                "instance_index",
                sort=True,
            )[metric].mean()

            stats = summarize_instance_values(
                instance_values,
                ("shap_lime", ds, model, metric),
            )

            summary_rows.append({
                "dataset": ds,
                "model": model,
                "metric": metric,
                "n": stats["n"],
                "n_rows": int(len(grp)),
                "mean": stats["mean"],
                "median": stats["median"],
                "std": stats["std"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "ci_status": stats["ci_status"],
                "bootstrap_n": BOOTSTRAPS,
                "bootstrap_unit": "instance",
            })

    return detail, pd.DataFrame(summary_rows)


def build_lime_stochasticity():
    src = (
        STAGE2
        / "lime_stochasticity"
        / "german_xgboost_stoch.parquet"
    )

    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_parquet(src)

    required = {"instance_i", "lime_seed"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"LIME stochasticity missing required columns: {required - set(df.columns)}"
        )

    features = [
        c for c in df.columns
        if c not in {
            "dataset", "model", "instance_i", "lime_seed", "duration"
        }
    ]

    if df["instance_i"].nunique() != 20:
        raise ValueError("Expected 20 stochasticity instances")

    if df["lime_seed"].nunique() != 30:
        raise ValueError("Expected 30 LIME stochasticity seeds")

    rows = []

    for instance, grp in df.groupby("instance_i", sort=True):
        grp = grp.sort_values("lime_seed").reset_index(drop=True)

        for i, j in itertools.combinations(range(len(grp)), 2):
            a = grp.loc[i, features].to_numpy(float)
            b = grp.loc[j, features].to_numpy(float)

            metrics = metric_values(a, b)

            rows.append({
                "instance": int(instance),
                "lime_seed1": int(grp.loc[i, "lime_seed"]),
                "lime_seed2": int(grp.loc[j, "lime_seed"]),
                "rank_sim": metrics["rank_sim"],
                "attr_sim": metrics["attr_sim"],
                "sign_cons": metrics["sign_cons"],
                "topk": metrics["topk"],
            })

    out = pd.DataFrame(rows)

    expected = 20 * 30 * 29 // 2

    if len(out) != expected:
        raise ValueError(
            f"LIME stochasticity expected {expected} rows, got {len(out)}"
        )

    return out


def load_robustness():
    files = sorted(
        (STAGE2 / "robustness").glob("*_robustness.parquet")
    )

    if len(files) != 60:
        raise ValueError(
            f"Expected 60 robustness files, found {len(files)}"
        )

    frames = [pd.read_parquet(f) for f in files]
    rob = pd.concat(frames, ignore_index=True)

    if len(rob) != 12000:
        raise ValueError(
            f"Expected 12000 robustness rows, got {len(rob)}"
        )

    return rob


def build_prediction_preserving(rob):
    subsets = {
        "all": pd.Series(True, index=rob.index),
        "plausible": rob["plausibility_pass"].astype(bool),
        "prediction_preserving": rob["prediction_preserving"].astype(bool),
        "prediction_preserving_plausible": (
            rob["prediction_preserving"].astype(bool)
            & rob["plausibility_pass"].astype(bool)
        ),
    }

    rows = []

    for subset_name, mask in subsets.items():
        sdf = rob.loc[mask].copy()

        for (ds, model, rate), grp in sdf.groupby(
            ["dataset", "model", "perturbation_rate"],
            sort=True,
        ):
            for metric in ROBUST_METRICS:
                instance_values = (
                    grp.groupby("instance_index", sort=True)[metric]
                    .mean()
                )

                stats = summarize_instance_values(
                    instance_values,
                    (
                        "robustness",
                        subset_name,
                        ds,
                        model,
                        rate,
                        metric,
                    ),
                )

                rows.append({
                    "subset": subset_name,
                    "dataset": ds,
                    "model": model,
                    "pert_rate": float(rate),
                    "metric": metric,
                    "n": stats["n"],
                    "n_rows": int(len(grp)),
                    "n_inst": stats["n"],
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "std": stats["std"],
                    "q25": stats["q25"],
                    "q75": stats["q75"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "ci_status": stats["ci_status"],
                    "bootstrap_n": BOOTSTRAPS,
                    "bootstrap_unit": "instance",
                })

    return pd.DataFrame(rows)


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    if m == 0:
        return np.array([])

    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)

    running = 0.0

    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)

    return adjusted


def safe_wilcoxon(a, b):
    diff = np.asarray(a) - np.asarray(b)

    if np.allclose(diff, 0.0):
        return 0.0, 1.0

    try:
        result = wilcoxon(
            a,
            b,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, 1.0


def build_rate_analysis(rob):
    rate_rows = []

    metrics = ROBUST_METRICS

    for ds in DATASETS:
        for model in MODELS:
            subset = rob[
                (rob["dataset"] == ds)
                & (rob["model"] == model)
            ].copy()

            # Aggregate all five training seeds to one matched value
            # per INSTANCE x RATE.
            inst_rate = (
                subset.groupby(
                    ["instance_index", "perturbation_rate"],
                    sort=True,
                )[metrics]
                .mean()
                .reset_index()
            )

            pivot_data = {
                metric: (
                    inst_rate.pivot(
                        index="instance_index",
                        columns="perturbation_rate",
                        values=metric,
                    )
                    .reindex(columns=RATES)
                    .dropna()
                )
                for metric in metrics
            }

            for metric in metrics:
                table = pivot_data[metric]

                if len(table) != 50:
                    raise ValueError(
                        f"Rate analysis {ds}/{model}/{metric}: "
                        f"expected 50 matched instances, got {len(table)}"
                    )

                try:
                    stat, p = friedmanchisquare(
                        table[RATES[0]].to_numpy(),
                        table[RATES[1]].to_numpy(),
                        table[RATES[2]].to_numpy(),
                        table[RATES[3]].to_numpy(),
                    )
                    stat = float(stat)
                    p = float(p)
                except ValueError:
                    stat = np.nan
                    p = np.nan

                rate_rows.append({
                    "dataset": ds,
                    "model": model,
                    "metric": metric,
                    "test": "Friedman",
                    "rate_1": np.nan,
                    "rate_2": np.nan,
                    "statistic": stat,
                    "raw_p": p,
                    "holm_p": np.nan,
                    "n_instances": int(len(table)),
                    "correction": "",
                })

                post_rows = []

                for r1, r2 in itertools.combinations(RATES, 2):
                    a = table[r1].to_numpy()
                    b = table[r2].to_numpy()

                    wstat, wp = safe_wilcoxon(a, b)

                    post_rows.append({
                        "dataset": ds,
                        "model": model,
                        "metric": metric,
                        "test": "Wilcoxon",
                        "rate_1": float(r1),
                        "rate_2": float(r2),
                        "statistic": wstat,
                        "raw_p": wp,
                        "holm_p": np.nan,
                        "n_instances": int(len(table)),
                        "correction": "Holm",
                    })

                adjusted = holm_adjust(
                    [r["raw_p"] for r in post_rows]
                )

                for r, adj in zip(post_rows, adjusted):
                    r["holm_p"] = float(adj)
                    rate_rows.append(r)

    return pd.DataFrame(rate_rows)


def cluster_permutation_pvalue(x, y, seed):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if len(x) < 3:
        return np.nan, "NOT_ESTIMABLE_N_LT_3"

    observed = spearmanr(x, y).statistic

    if not np.isfinite(observed):
        return np.nan, "NOT_ESTIMABLE_CONSTANT_VECTOR"

    rng = np.random.default_rng(seed)
    extreme = 0

    for _ in range(PERMUTATIONS):
        yp = rng.permutation(y)
        null_rho = spearmanr(x, yp).statistic

        if np.isfinite(null_rho) and abs(null_rho) >= abs(observed):
            extreme += 1

    p = (extreme + 1) / (PERMUTATIONS + 1)
    return float(p), "OK"


def build_association(rob):
    rows = []

    subset_defs = {
        "all": rob,
        "prediction_preserving_plausible": rob[
            rob["prediction_preserving"].astype(bool)
            & rob["plausibility_pass"].astype(bool)
        ],
    }

    for subset_name, sdf in subset_defs.items():
        for ds in DATASETS:
            for model in MODELS:
                base = sdf[
                    (sdf["dataset"] == ds)
                    & (sdf["model"] == model)
                ].copy()

                for explainer, sim_cols in [
                    ("shap", ("shap_rank_sim", "shap_attr_sim")),
                    ("lime", ("lime_rank_sim", "lime_attr_sim")),
                ]:
                    for sim_col in sim_cols:
                        if base.empty:
                            continue

                        tmp = base[[
                            "instance_index",
                            "prediction_drift",
                            sim_col,
                        ]].copy()

                        tmp["explanation_drift"] = 1.0 - tmp[sim_col]

                        instance = (
                            tmp.groupby("instance_index", sort=True)
                            [["prediction_drift", "explanation_drift"]]
                            .mean()
                            .dropna()
                        )

                        if len(instance) < 3:
                            rho = np.nan
                            p = np.nan
                            status = "NOT_ESTIMABLE_N_LT_3"
                        else:
                            rho = float(
                                spearmanr(
                                    instance["prediction_drift"],
                                    instance["explanation_drift"],
                                ).statistic
                            )

                            p, status = cluster_permutation_pvalue(
                                instance["prediction_drift"].to_numpy(),
                                instance["explanation_drift"].to_numpy(),
                                stable_seed(
                                    "association",
                                    subset_name,
                                    ds,
                                    model,
                                    explainer,
                                    sim_col,
                                    PERM_SEED,
                                ),
                            )

                        rows.append({
                            "subset": subset_name,
                            "dataset": ds,
                            "model": model,
                            "explainer": explainer,
                            "metric": (
                                "rank_sim"
                                if sim_col.endswith("rank_sim")
                                else "attr_sim"
                            ),
                            "n_instances": int(len(instance)),
                            "n_rows": int(len(base)),
                            "rho": rho,
                            "p_value_cluster_permutation": p,
                            "p_method": "instance_cluster_permutation",
                            "n_permutations": PERMUTATIONS,
                            "p_status": status,
                        })

    return pd.DataFrame(rows)


def build_plausibility(rob):
    feature_counts = {
        "german": 48,
        "taiwan": 23,
        "gmsc": 10,
    }

    rows = []

    for ds in DATASETS:
        z = rob.loc[
            rob["dataset"] == ds,
            "plausibility_max_z",
        ].astype(float).to_numpy()

        rows.append({
            "dataset": ds,
            "feature_count": feature_counts[ds],
            "n": int(len(z)),
            "mean": float(np.mean(z)),
            "median": float(np.median(z)),
            "std": float(np.std(z, ddof=1)),
            "q25": float(np.percentile(z, 25)),
            "q75": float(np.percentile(z, 75)),
            "p90": float(np.percentile(z, 90)),
            "p95": float(np.percentile(z, 95)),
            "p99": float(np.percentile(z, 99)),
            "max": float(np.max(z)),
            "pass_rate": float(np.mean(z < PLAUSIBILITY_Z)),
            "z_gt_2": float(np.mean(z > 2)),
            "z_gt_3": float(np.mean(z > 3)),
            "z_gt_3_5": float(np.mean(z > 3.5)),
            "z_gte_4": float(np.mean(z >= 4)),
            "threshold": PLAUSIBILITY_Z,
        })

    return pd.DataFrame(rows)


def write_ess_placeholders():
    pd.DataFrame([{
        "status": "PENDING_STAGE_4",
        "reason": "semantic category validation required",
        "note": "Full ESS/CSS is deferred until Stage 4 semantic mapping and validation."
    }]).to_csv(
        STAGE3 / "ess_sensitivity.csv",
        index=False,
    )

    pd.DataFrame([{
        "status": "PENDING_STAGE_4",
        "reason": "semantic category validation required",
        "note": "ESS ablation is deferred until defensible semantic stability components exist."
    }]).to_csv(
        STAGE3 / "ess_ablation.csv",
        index=False,
    )


def make_plots(
    seed_pairwise,
    shap_lime_detail,
    pred_analysis,
    rob,
):
    PLOTS.mkdir(parents=True, exist_ok=True)

    # 1. Seed stability
    instance_seed = (
        seed_pairwise
        .groupby(["explainer", "instance_index"])["rank_sim"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [
        instance_seed.loc[
            instance_seed["explainer"] == "shap", "rank_sim"
        ].to_numpy(),
        instance_seed.loc[
            instance_seed["explainer"] == "lime", "rank_sim"
        ].to_numpy(),
    ]
    ax.boxplot(data, labels=["SHAP", "LIME"])
    ax.set_ylabel("Rank similarity")
    ax.set_title("Training-seed explanation stability")
    fig.tight_layout()
    fig.savefig(PLOTS / "seed_stability.png", dpi=200)
    plt.close(fig)

    # 2. SHAP vs LIME
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, metric in zip(axes, ["rank_sim", "attr_sim"]):
        vals = shap_lime_detail[metric].to_numpy(float)
        ax.hist(vals, bins=30)
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.set_title(f"SHAP vs LIME: {metric}")
    fig.tight_layout()
    fig.savefig(PLOTS / "shap_vs_lime_consistency.png", dpi=200)
    plt.close(fig)

    # 3. Prediction vs explanation drift
    primary = rob[
        rob["prediction_preserving"].astype(bool)
        & rob["plausibility_pass"].astype(bool)
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, col, title in [
        (axes[0], "shap_rank_sim", "SHAP rank drift"),
        (axes[1], "lime_rank_sim", "LIME rank drift"),
    ]:
        x = primary["prediction_drift"].to_numpy(float)
        y = 1.0 - primary[col].to_numpy(float)
        ax.scatter(x, y, s=8, alpha=0.25)
        ax.set_xlabel("Prediction drift")
        ax.set_ylabel("Explanation drift = 1 - rank similarity")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(PLOTS / "prediction_vs_explanation_drift.png", dpi=200)
    plt.close(fig)

    # 4. Primary plausible + prediction preserving drift
    p = primary.copy()
    p["dataset_model"] = p["dataset"] + "/" + p["model"]
    grouped = (
        p.groupby(["dataset", "model"])[
            ["shap_rank_sim", "lime_rank_sim"]
        ]
        .mean()
        .reset_index()
    )

    labels = grouped["dataset"] + "/" + grouped["model"]
    x = np.arange(len(grouped))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        x - width / 2,
        1 - grouped["shap_rank_sim"],
        width,
        label="SHAP",
    )
    ax.bar(
        x + width / 2,
        1 - grouped["lime_rank_sim"],
        width,
        label="LIME",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean explanation drift")
    ax.set_title("Prediction-preserving + plausible explanation drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        PLOTS / "prediction_preserving_plausible_drift.png",
        dpi=200,
    )
    plt.close(fig)

    # 5. Perturbation rate
    rate = (
        rob.groupby("perturbation_rate")[
            ["prediction_drift", "shap_rank_sim", "lime_rank_sim"]
        ]
        .mean()
        .reindex(RATES)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [str(r) for r in RATES],
        rate["prediction_drift"],
        marker="o",
        label="Prediction drift",
    )
    ax.plot(
        [str(r) for r in RATES],
        1 - rate["shap_rank_sim"],
        marker="o",
        label="SHAP explanation drift",
    )
    ax.plot(
        [str(r) for r in RATES],
        1 - rate["lime_rank_sim"],
        marker="o",
        label="LIME explanation drift",
    )
    ax.set_xlabel("Perturbation rate")
    ax.set_ylabel("Mean drift")
    ax.set_title("Perturbation-rate effects")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "perturbation_rate.png", dpi=200)
    plt.close(fig)

    # 6. Plausibility distribution
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, ds in zip(axes, DATASETS):
        z = rob.loc[
            rob["dataset"] == ds,
            "plausibility_max_z",
        ].to_numpy(float)

        ax.hist(z, bins=30)
        ax.axvline(PLAUSIBILITY_Z, linestyle="--")
        ax.set_title(ds.upper())
        ax.set_xlabel("plausibility_max_z")
        ax.set_ylabel("Count")

    fig.suptitle("Plausibility distributions with fixed z=4 threshold")
    fig.tight_layout()
    fig.savefig(PLOTS / "plausibility_distribution.png", dpi=200)
    plt.close(fig)


def write_readme(plausibility):
    text = f"""# Stage 3 Statistical Analysis

Stage 3 reanalyzes the locked Stage 2 outputs. No model training, SHAP,
LIME, or perturbation generation is performed here.

## Locked Stage 2 design

- Datasets: German, Taiwan, GMSC
- Models: XGBoost, LightGBM, Random Forest, MLP
- Training seeds: {SEEDS}
- Matched explanation instances: 50 per dataset
- Perturbation rates: {RATES}
- Robustness rows: 12,000
- Plausibility threshold: z < {PLAUSIBILITY_Z}

## Statistical unit

The experimental hierarchy is:

dataset -> model -> training seed -> instance -> perturbation rate

Rows sharing an original test instance are therefore repeated observations.
Confidence intervals are generated by resampling the original test instances,
not individual seed/perturbation rows.

Seed-stability bootstrap uses the 50 matched instances.

SHAP-vs-LIME bootstrap uses the 50 matched instances.

Robustness summaries first aggregate repeated seed observations within an
instance and then bootstrap the resulting instance-level values.

LIME stochasticity is scoped to the existing German/XGBoost experiment:
20 instances and 30 LIME seeds.

## Perturbation-rate inference

For each dataset/model/metric, the five training-seed observations are first
averaged within each instance and perturbation rate. This produces 50 matched
instances at each of the four perturbation rates.

Friedman tests are therefore performed on four matched vectors of length 50.

All six pairwise rate comparisons are performed with paired Wilcoxon tests.
Holm correction is applied within each dataset/model/metric family.

## Prediction vs explanation association

Spearman rho is calculated after aggregating repeated observations to one
value per original test instance. Significance is obtained with an
instance-level permutation test using {PERMUTATIONS} permutations.

This treats the original applicant instances as the independent clusters.

Association is not interpreted causally.

## Prediction-preserving explanation drift

Four views are kept separate:

1. all perturbations;
2. plausible perturbations;
3. prediction-preserving perturbations;
4. prediction-preserving AND plausible perturbations.

The fourth subset is the primary analysis.

Prediction robustness and explanation robustness are not collapsed into one
headline metric.

## Plausibility

The plausibility check is a train-data-based local statistical screen using a
fixed maximum standardized-distance threshold of z < {PLAUSIBILITY_Z}.
Passing this screen does not establish that a perturbed instance lies on the
full joint data manifold.

Observed pass rates:

{plausibility.to_string(index=False)}

Feature counts are descriptive:
- German = 48
- Taiwan = 23
- GMSC = 10

These feature counts are not interpreted as proven causal explanations of
plausibility differences.

## ESS

Full ESS/CSS is explicitly deferred:

PENDING_STAGE_4

Semantic category validation is required before a defensible final CSS/ESS
calculation.

The historical preliminary ESS value is not used as a final Stage 3 result.
"""
    (STAGE3 / "README.md").write_text(text, encoding="utf-8")


def main():
    print("=" * 78)
    print("XAI-StabilityNet — FINAL STAGE 3 STATISTICAL REPAIR")
    print("=" * 78)

    STAGE3.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    # Backup current Stage 3 outputs.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = STAGE3 / f"_pre_statistical_repair_backup_{timestamp}"

    if any(STAGE3.iterdir()):
        backup.mkdir(parents=True, exist_ok=True)

        for item in STAGE3.iterdir():
            if item.name == backup.name:
                continue

            target = backup / item.name

            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

        print("Backup:", backup)

    # Remove global warning suppression from the legacy runner only.
    legacy = ROOT / "stage3_runner.py"
    if legacy.exists():
        text = legacy.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        text = text.replace(
            "import warnings\nwarnings.filterwarnings(\"ignore\")\n",
            "",
        )
        legacy.write_text(text, encoding="utf-8")
        print("Removed global warning suppression from legacy stage3_runner.py")

    # Load locked Stage 2 data.
    store, feature_orders = parse_explanation_files()
    rob = load_robustness()

    # Build corrected analyses.
    print("\n[1/7] Training-seed stability")
    seed_pairwise, seed_summary = build_seed_stability(
        store,
        feature_orders,
    )
    print(
        "  pairwise rows:",
        len(seed_pairwise),
        "| summary:",
        len(seed_summary),
    )

    print("\n[2/7] SHAP vs LIME")
    shap_lime_detail, shap_lime_summary = build_shap_lime(
        store,
        feature_orders,
    )
    print(
        "  detail rows:",
        len(shap_lime_detail),
        "| summary:",
        len(shap_lime_summary),
    )

    print("\n[3/7] LIME stochasticity")
    lime_stoch = build_lime_stochasticity()
    print("  pairwise rows:", len(lime_stoch))

    print("\n[4/7] Prediction-preserving analysis")
    pred_analysis = build_prediction_preserving(rob)
    print("  rows:", len(pred_analysis))

    print("\n[5/7] Perturbation-rate statistics")
    rate_analysis = build_rate_analysis(rob)
    print("  rows:", len(rate_analysis))

    print("\n[6/7] Prediction-explanation association")
    association = build_association(rob)
    print("  rows:", len(association))

    print("\n[7/7] Plausibility")
    plausibility = build_plausibility(rob)
    print(plausibility.to_string(index=False))

    # ESS remains explicitly pending.
    write_ess_placeholders()

    # Validate before replacing output files.
    required_values = [
        len(seed_pairwise) == 12000,
        len(seed_summary) == 96,
        len(shap_lime_detail) == 3000,
        len(shap_lime_summary) == 48,
        len(lime_stoch) == 8700,
        len(pred_analysis) == 4 * 3 * 4 * 4 * 7,
        len(rate_analysis) == 3 * 4 * 7 * (1 + 6),
        len(association) == 96,
        len(plausibility) == 3,
        len(rob) == 12000,
    ]

    if not all(required_values):
        raise RuntimeError(
            "One or more Stage 3 row-count validations failed."
        )

    for name, df in [
        ("seed_stability_pairwise", seed_pairwise),
        ("seed_stability_summary", seed_summary),
        ("shap_lime_consistency", shap_lime_detail),
        ("shap_lime_consistency_summary", shap_lime_summary),
        ("lime_stochasticity_analysis", lime_stoch),
        ("prediction_preserving_analysis", pred_analysis),
        ("perturbation_rate_analysis", rate_analysis),
        ("prediction_explanation_association", association),
        ("plausibility_analysis", plausibility),
    ]:
        numeric = df.select_dtypes(include=[np.number])
        if np.isinf(numeric.to_numpy()).any():
            raise RuntimeError(f"{name}: Inf detected")

    # Save final corrected CSVs.
    seed_pairwise.to_csv(
        STAGE3 / "seed_stability_pairwise.csv",
        index=False,
    )
    seed_summary.to_csv(
        STAGE3 / "seed_stability_summary.csv",
        index=False,
    )
    shap_lime_detail.to_csv(
        STAGE3 / "shap_lime_consistency.csv",
        index=False,
    )
    shap_lime_summary.to_csv(
        STAGE3 / "shap_lime_consistency_summary.csv",
        index=False,
    )
    lime_stoch.to_csv(
        STAGE3 / "lime_stochasticity_analysis.csv",
        index=False,
    )
    pred_analysis.to_csv(
        STAGE3 / "prediction_preserving_analysis.csv",
        index=False,
    )
    rate_analysis.to_csv(
        STAGE3 / "perturbation_rate_analysis.csv",
        index=False,
    )
    association.to_csv(
        STAGE3 / "prediction_explanation_association.csv",
        index=False,
    )
    plausibility.to_csv(
        STAGE3 / "plausibility_analysis.csv",
        index=False,
    )

    # Plots and documentation.
    make_plots(
        seed_pairwise,
        shap_lime_detail,
        pred_analysis,
        rob,
    )
    write_readme(plausibility)

    summary = {
        "status": "CORRECTED",
        "stage2_locked": True,
        "stage2_rows_processed": int(len(rob)),
        "seed_stability_pairwise_rows": int(len(seed_pairwise)),
        "seed_stability_summary_rows": int(len(seed_summary)),
        "shap_lime_rows": int(len(shap_lime_detail)),
        "shap_lime_summary_rows": int(len(shap_lime_summary)),
        "lime_stochasticity_pairwise_rows": int(len(lime_stoch)),
        "prediction_preserving_rows": int(len(pred_analysis)),
        "perturbation_rate_rows": int(len(rate_analysis)),
        "association_rows": int(len(association)),
        "plausibility_rows": int(len(plausibility)),
        "training_seeds": SEEDS,
        "n_explain_instances": 50,
        "plausibility_threshold": PLAUSIBILITY_Z,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "instance",
        "association_permutations": PERMUTATIONS,
        "association_p_value_method": "instance_cluster_permutation",
        "rate_test": "Friedman on 50 matched instance means",
        "posthoc_test": "paired Wilcoxon",
        "posthoc_correction": "Holm",
        "ess_status": "PENDING_STAGE_4",
        "css_status": "PENDING_STAGE_4",
        "plots_created": [
            "seed_stability.png",
            "shap_vs_lime_consistency.png",
            "prediction_vs_explanation_drift.png",
            "prediction_preserving_plausible_drift.png",
            "perturbation_rate.png",
            "plausibility_distribution.png",
        ],
        "backup": str(backup),
    }

    (STAGE3 / "stage3_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("STAGE 3 REPAIR COMPLETE")
    print("=" * 78)
    print("Seed stability             :", len(seed_pairwise), "rows")
    print("SHAP vs LIME               :", len(shap_lime_detail), "rows")
    print("LIME stochasticity         :", len(lime_stoch), "rows")
    print("Prediction-preserving      :", len(pred_analysis), "rows")
    print("Rate tests                 :", len(rate_analysis), "rows")
    print("Association                :", len(association), "rows")
    print("Plausibility               :", len(plausibility), "datasets")
    print("ESS                        : PENDING_STAGE_4")
    print("Plots                      : 6")
    print("Backup                     :", backup)
    print("=" * 78)


if __name__ == "__main__":
    main()

