#!/usr/bin/env python3
"""
statistics.py

Statistical tests comparing industry-tied vs independent authors/papers.

Tests performed:
  1. Chi-square / Fisher exact: industry involvement vs outcome direction
  2. Odds ratio with 95% CI: industry → positive outcome
  3. Mann-Whitney U: centrality of industry vs independent authors
  4. Permutation test: is the industry-positive association stronger than chance?
  5. Proportion z-test: % positive outcomes by group

Outputs (to --output_dir):
  - full_statistics.json
  - outcome_comparison.csv
  - centrality_comparison.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def odds_ratio_ci(a: int, b: int, c: int, d: int) -> Dict[str, float]:
    """
    2x2 table:
                 Positive  Not-Positive
    Industry        a          b
    Independent     c          d

    Returns OR, 95% CI, and log-OR SE.
    """
    # Continuity correction if any cell is 0
    if any(x == 0 for x in [a, b, c, d]):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    or_val = (a * d) / (b * c)
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_lo = math.exp(math.log(or_val) - 1.96 * se)
    ci_hi = math.exp(math.log(or_val) + 1.96 * se)

    return {
        "odds_ratio": round(or_val, 4),
        "ci_95_lower": round(ci_lo, 4),
        "ci_95_upper": round(ci_hi, 4),
        "log_or_se": round(se, 4),
    }


def proportion_ztest(n1_success: int, n1_total: int, n2_success: int, n2_total: int) -> Dict[str, float]:
    """Two-proportion z-test."""
    if n1_total == 0 or n2_total == 0:
        return {"z": None, "p_value": None, "p1": None, "p2": None}

    p1 = n1_success / n1_total
    p2 = n2_success / n2_total
    p_pool = (n1_success + n2_success) / (n1_total + n2_total)

    if p_pool == 0 or p_pool == 1:
        return {"z": 0.0, "p_value": 1.0, "p1": p1, "p2": p2}

    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1_total + 1/n2_total))
    z = (p1 - p2) / se if se > 0 else 0.0
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))

    return {
        "z": round(z, 4),
        "p_value": round(p_value, 6),
        "p1_industry": round(p1, 4),
        "p2_independent": round(p2, 4),
        "diff": round(p1 - p2, 4),
    }


def permutation_test(
    outcomes: pd.Series,
    groups: pd.Series,
    n_permutations: int = 10000,
    target_outcome: str = "Positive",
    target_group: str = "Yes",
) -> Dict[str, Any]:
    """
    Permutation test: is the observed proportion of target_outcome in target_group
    significantly different from what we'd expect by chance?
    """
    mask_group = groups == target_group
    mask_outcome = outcomes == target_outcome

    observed = mask_outcome[mask_group].mean() if mask_group.sum() > 0 else 0.0

    np.random.seed(42)
    n_target = mask_group.sum()
    perm_props = []
    outcome_arr = mask_outcome.values.copy()

    for _ in range(n_permutations):
        np.random.shuffle(outcome_arr)
        perm_prop = outcome_arr[:n_target].mean()
        perm_props.append(perm_prop)

    perm_props = np.array(perm_props)
    # Two-sided p-value: proportion of permuted differences at least as extreme
    # as observed in either direction
    mean_perm = perm_props.mean()
    observed_diff = abs(observed - mean_perm)
    p_value = (np.sum(np.abs(perm_props - mean_perm) >= observed_diff) + 1) / (n_permutations + 1)

    return {
        "observed_proportion": round(float(observed), 4),
        "mean_permuted": round(float(perm_props.mean()), 4),
        "p_value": round(float(p_value), 6),
        "n_permutations": n_permutations,
        "n_target_group": int(n_target),
    }


CATEGORY_ORDER = ["Tobacco Company", "COI Declared", "Independent"]


def run_all_tests(papers_df: pd.DataFrame, centrality_df: pd.DataFrame) -> Dict[str, Any]:
    """Run all statistical tests and return results dict.

    Uses three-category classification: Tobacco Company, COI Declared, Independent.
    """
    results: Dict[str, Any] = {}

    outcome_order = ["Positive", "Negative", "Neutral", "Mixed", "Not coded"]

    # --- 1. Contingency table: industry_category vs Outcome ---
    ctab = pd.crosstab(papers_df["industry_category"], papers_df["outcome"])
    for col in outcome_order:
        if col not in ctab.columns:
            ctab[col] = 0
    ctab = ctab[outcome_order]
    # Ensure all categories present
    for cat in CATEGORY_ORDER:
        if cat not in ctab.index:
            ctab.loc[cat] = 0
    ctab = ctab.loc[[c for c in CATEGORY_ORDER if c in ctab.index]]

    results["contingency_table"] = ctab.to_dict()

    # Chi-square — exclude "Not coded" papers
    coded_outcomes = ["Positive", "Negative", "Neutral", "Mixed"]
    ctab_coded = ctab[coded_outcomes]
    ctab_coded = ctab_coded.loc[:, (ctab_coded.sum(axis=0) > 0)]
    # Drop rows with all zeros
    ctab_coded = ctab_coded.loc[(ctab_coded.sum(axis=1) > 0)]

    try:
        chi2, p, dof, expected = sp_stats.chi2_contingency(ctab_coded.values)
        results["chi_square"] = {
            "chi2": round(chi2, 4),
            "p_value": round(p, 6),
            "dof": int(dof),
            "note": "3-group test (Tobacco/COI/Independent), excludes 'Not coded'",
        }
    except Exception as e:
        results["chi_square"] = {"error": str(e)}

    # --- 2. Pairwise Odds Ratios: each group vs Independent ---
    indep = papers_df[papers_df["industry_category"] == "Independent"]
    c_indep = (indep["outcome"] == "Positive").sum()
    d_indep = len(indep) - c_indep

    for group_name in ["Tobacco Company", "COI Declared"]:
        grp = papers_df[papers_df["industry_category"] == group_name]
        a_grp = (grp["outcome"] == "Positive").sum()
        b_grp = len(grp) - a_grp

        key = f"odds_ratio_{group_name.lower().replace(' ', '_')}_vs_independent"
        results[key] = odds_ratio_ci(a_grp, b_grp, c_indep, d_indep)
        results[key]["table"] = {"a": a_grp, "b": b_grp, "c": c_indep, "d": d_indep}
        results[key]["comparison"] = f"{group_name} vs Independent"

        # Fisher exact
        try:
            _, fisher_p = sp_stats.fisher_exact([[a_grp, b_grp], [c_indep, d_indep]])
            results[key]["fisher_exact_p"] = round(fisher_p, 6)
        except Exception:
            results[key]["fisher_exact_p"] = None

    # --- 3. Proportion z-tests (pairwise vs Independent) ---
    for group_name in ["Tobacco Company", "COI Declared"]:
        grp = papers_df[papers_df["industry_category"] == group_name]
        key = f"proportion_ztest_{group_name.lower().replace(' ', '_')}_vs_independent"
        results[key] = proportion_ztest(
            n1_success=(grp["outcome"] == "Positive").sum(),
            n1_total=len(grp),
            n2_success=(indep["outcome"] == "Positive").sum(),
            n2_total=len(indep),
        )
        results[key]["comparison"] = f"{group_name} vs Independent"

    # --- 4. Permutation tests (pairwise vs Independent) ---
    for group_name in ["Tobacco Company", "COI Declared"]:
        # Create a binary grouping for this comparison
        mask = papers_df["industry_category"].isin([group_name, "Independent"])
        subset = papers_df[mask]
        groups_binary = (subset["industry_category"] == group_name).map({True: "target", False: "ref"})

        key = f"permutation_test_{group_name.lower().replace(' ', '_')}_vs_independent"
        results[key] = permutation_test(
            outcomes=subset["outcome"],
            groups=groups_binary,
            n_permutations=10000,
            target_group="target",
        )
        results[key]["comparison"] = f"{group_name} vs Independent"

    # --- 5. Centrality comparison (Mann-Whitney U, three groups) ---
    if not centrality_df.empty and "author_category" in centrality_df.columns:
        centrality_results = {}
        for metric in ["degree_centrality", "betweenness_centrality", "closeness_centrality", "eigenvector_centrality"]:
            if metric not in centrality_df.columns:
                continue

            metric_results = {}
            indep_vals = centrality_df[centrality_df["author_category"] == "Independent"][metric].dropna()

            for group_name in ["Tobacco Company", "COI Declared"]:
                grp_vals = centrality_df[centrality_df["author_category"] == group_name][metric].dropna()
                if len(grp_vals) > 0 and len(indep_vals) > 0:
                    u_stat, u_p = sp_stats.mannwhitneyu(grp_vals, indep_vals, alternative="two-sided")
                    metric_results[group_name] = {
                        "mean": round(grp_vals.mean(), 6),
                        "median": round(grp_vals.median(), 6),
                        "vs_independent_U": round(u_stat, 2),
                        "vs_independent_p": round(u_p, 6),
                    }

            metric_results["Independent"] = {
                "mean": round(indep_vals.mean(), 6),
                "median": round(indep_vals.median(), 6),
            }
            centrality_results[metric] = metric_results

        # Kruskal-Wallis across all three groups
        kw_results = {}
        for metric in ["degree_centrality", "betweenness_centrality", "closeness_centrality", "eigenvector_centrality"]:
            if metric not in centrality_df.columns:
                continue
            groups_data = [
                centrality_df[centrality_df["author_category"] == cat][metric].dropna()
                for cat in CATEGORY_ORDER
            ]
            groups_data = [g for g in groups_data if len(g) > 0]
            if len(groups_data) >= 2:
                h_stat, h_p = sp_stats.kruskal(*groups_data)
                kw_results[metric] = {"H": round(h_stat, 4), "p_value": round(h_p, 6)}

        results["centrality_comparison"] = centrality_results
        results["centrality_kruskal_wallis"] = kw_results

    # --- Summary counts ---
    results["counts"] = {
        "total_papers": len(papers_df),
        "outcome_distribution": papers_df["outcome"].value_counts().to_dict(),
    }
    for cat in CATEGORY_ORDER:
        subset = papers_df[papers_df["industry_category"] == cat]
        results["counts"][f"n_{cat.lower().replace(' ', '_')}"] = len(subset)
        results["counts"][f"{cat.lower().replace(' ', '_')}_outcomes"] = (
            subset["outcome"].value_counts().to_dict() if len(subset) > 0 else {}
        )

    return results


def main():
    ap = argparse.ArgumentParser(description="Statistical analysis of industry ties vs outcomes")
    ap.add_argument("--input_dir", required=True, help="Dir with papers.csv from load_and_identify.py")
    ap.add_argument("--network_dir", required=True, help="Dir with top_authors_centrality.csv from network.py")
    ap.add_argument("--output_dir", required=True, help="Where to write statistics output")
    args = ap.parse_args()

    papers_df = pd.read_csv(os.path.join(args.input_dir, "papers.csv"))
    centrality_path = os.path.join(args.network_dir, "top_authors_centrality.csv")
    centrality_df = pd.read_csv(centrality_path) if os.path.exists(centrality_path) else pd.DataFrame()

    results = run_all_tests(papers_df, centrality_df)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "full_statistics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Outcome comparison table
    outcome_rows = []
    for cat in CATEGORY_ORDER:
        subset = papers_df[papers_df["industry_category"] == cat]
        total = len(subset)
        for outcome in ["Positive", "Negative", "Neutral", "Mixed", "Not coded"]:
            count = (subset["outcome"] == outcome).sum()
            pct = (count / total * 100) if total > 0 else 0
            outcome_rows.append({
                "group": cat,
                "outcome": outcome,
                "count": count,
                "total": total,
                "percentage": round(pct, 1),
            })
    outcome_df = pd.DataFrame(outcome_rows)
    outcome_df.to_csv(os.path.join(args.output_dir, "outcome_comparison.csv"), index=False)

    print("Statistical analysis complete.")
    print(f"  Chi-square p-value: {results.get('chi_square', {}).get('p_value', 'N/A')}")
    for grp in ["tobacco_company", "coi_declared"]:
        key = f"odds_ratio_{grp}_vs_independent"
        or_data = results.get(key, {})
        print(f"  OR ({grp} vs independent): {or_data.get('odds_ratio', 'N/A')} "
              f"(Fisher p={or_data.get('fisher_exact_p', 'N/A')})")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
