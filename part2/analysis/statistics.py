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
    p_value = (np.sum(perm_props >= observed) + 1) / (n_permutations + 1)

    return {
        "observed_proportion": round(float(observed), 4),
        "mean_permuted": round(float(perm_props.mean()), 4),
        "p_value": round(float(p_value), 6),
        "n_permutations": n_permutations,
        "n_target_group": int(n_target),
    }


def run_all_tests(papers_df: pd.DataFrame, centrality_df: pd.DataFrame) -> Dict[str, Any]:
    """Run all statistical tests and return results dict."""
    results: Dict[str, Any] = {}

    # --- 1. Contingency table: Industry_Involved vs Outcome ---
    outcome_order = ["Positive", "Negative", "Neutral", "Mixed", "Not coded"]
    ctab = pd.crosstab(papers_df["industry_involved"], papers_df["outcome"])
    for col in outcome_order:
        if col not in ctab.columns:
            ctab[col] = 0
    ctab = ctab[outcome_order]

    results["contingency_table"] = ctab.to_dict()

    # Chi-square
    try:
        chi2, p, dof, expected = sp_stats.chi2_contingency(ctab.values)
        results["chi_square"] = {
            "chi2": round(chi2, 4),
            "p_value": round(p, 6),
            "dof": int(dof),
        }
    except Exception as e:
        results["chi_square"] = {"error": str(e)}

    # --- 2. 2x2 Odds Ratio: Positive vs Not-Positive by Industry ---
    a = int(ctab.loc["Yes", "Positive"]) if "Yes" in ctab.index else 0
    c = int(ctab.loc["No", "Positive"]) if "No" in ctab.index else 0
    b = int(ctab.loc["Yes", :].sum() - a) if "Yes" in ctab.index else 0
    d = int(ctab.loc["No", :].sum() - c) if "No" in ctab.index else 0

    results["odds_ratio_positive"] = odds_ratio_ci(a, b, c, d)
    results["odds_ratio_positive"]["table"] = {"a": a, "b": b, "c": c, "d": d}

    # Fisher exact
    try:
        _, fisher_p = sp_stats.fisher_exact([[a, b], [c, d]])
        results["fisher_exact_p"] = round(fisher_p, 6)
    except Exception:
        results["fisher_exact_p"] = None

    # --- 3. Proportion z-test ---
    ind_yes = papers_df[papers_df["industry_involved"] == "Yes"]
    ind_no = papers_df[papers_df["industry_involved"] == "No"]
    results["proportion_ztest_positive"] = proportion_ztest(
        n1_success=(ind_yes["outcome"] == "Positive").sum(),
        n1_total=len(ind_yes),
        n2_success=(ind_no["outcome"] == "Positive").sum(),
        n2_total=len(ind_no),
    )

    # --- 4. Permutation test ---
    results["permutation_test"] = permutation_test(
        outcomes=papers_df["outcome"],
        groups=papers_df["industry_involved"],
        n_permutations=10000,
    )

    # --- 5. Centrality comparison (Mann-Whitney U) ---
    if not centrality_df.empty and "is_industry" in centrality_df.columns:
        ind_cent = centrality_df[centrality_df["is_industry"] == True]
        noind_cent = centrality_df[centrality_df["is_industry"] == False]

        centrality_results = {}
        for metric in ["degree_centrality", "betweenness_centrality", "closeness_centrality", "eigenvector_centrality"]:
            if metric in centrality_df.columns:
                vals_ind = ind_cent[metric].dropna()
                vals_noind = noind_cent[metric].dropna()

                if len(vals_ind) > 0 and len(vals_noind) > 0:
                    u_stat, u_p = sp_stats.mannwhitneyu(vals_ind, vals_noind, alternative="two-sided")
                    centrality_results[metric] = {
                        "industry_mean": round(vals_ind.mean(), 6),
                        "industry_median": round(vals_ind.median(), 6),
                        "independent_mean": round(vals_noind.mean(), 6),
                        "independent_median": round(vals_noind.median(), 6),
                        "mann_whitney_U": round(u_stat, 2),
                        "p_value": round(u_p, 6),
                    }

        results["centrality_comparison"] = centrality_results

    # --- Summary counts ---
    results["counts"] = {
        "total_papers": len(papers_df),
        "industry_involved_yes": len(ind_yes),
        "industry_involved_no": len(ind_no),
        "outcome_distribution": papers_df["outcome"].value_counts().to_dict(),
        "industry_outcome_distribution": ind_yes["outcome"].value_counts().to_dict() if len(ind_yes) > 0 else {},
        "independent_outcome_distribution": ind_no["outcome"].value_counts().to_dict() if len(ind_no) > 0 else {},
    }

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
    for group_name, group_label in [("Yes", "Industry-Involved"), ("No", "Independent")]:
        subset = papers_df[papers_df["industry_involved"] == group_name]
        total = len(subset)
        for outcome in ["Positive", "Negative", "Neutral", "Mixed", "Not coded"]:
            count = (subset["outcome"] == outcome).sum()
            pct = (count / total * 100) if total > 0 else 0
            outcome_rows.append({
                "group": group_label,
                "outcome": outcome,
                "count": count,
                "total": total,
                "percentage": round(pct, 1),
            })
    outcome_df = pd.DataFrame(outcome_rows)
    outcome_df.to_csv(os.path.join(args.output_dir, "outcome_comparison.csv"), index=False)

    print("Statistical analysis complete.")
    print(f"  Chi-square p-value: {results.get('chi_square', {}).get('p_value', 'N/A')}")
    print(f"  Odds ratio (Positive): {results.get('odds_ratio_positive', {}).get('odds_ratio', 'N/A')}")
    print(f"  Fisher exact p-value: {results.get('fisher_exact_p', 'N/A')}")
    print(f"  Proportion z-test p-value: {results.get('proportion_ztest_positive', {}).get('p_value', 'N/A')}")
    print(f"  Permutation test p-value: {results.get('permutation_test', {}).get('p_value', 'N/A')}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
