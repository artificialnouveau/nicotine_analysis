#!/usr/bin/env python3
"""
plots.py

Generate static and interactive visualizations for the tobacco/nicotine
industry influence analysis.

Outputs (to --output_dir/figures/):
  Static (matplotlib/seaborn):
    - outcome_by_industry.png
    - outcome_proportions.png
    - centrality_distribution.png
    - community_industry_scatter.png
    - timeline_industry_papers.png
    - odds_ratio_forest.png

  Interactive (plotly HTML):
    - sankey_funding_outcome.html
    - outcome_heatmap.html
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Color scheme
COLORS = {
    "industry": "#e74c3c",
    "independent": "#3498db",
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral": "#95a5a6",
    "Mixed": "#f39c12",
    "Not coded": "#bdc3c7",
}

OUTCOME_ORDER = ["Positive", "Negative", "Neutral", "Mixed", "Not coded"]


def plot_outcome_by_industry(papers_df: pd.DataFrame, output_dir: str):
    """Grouped bar chart: outcome counts by industry involvement."""
    ctab = pd.crosstab(papers_df["industry_involved"], papers_df["outcome"])
    for col in OUTCOME_ORDER:
        if col not in ctab.columns:
            ctab[col] = 0
    ctab = ctab[OUTCOME_ORDER]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(OUTCOME_ORDER))
    width = 0.35

    ind_yes = ctab.loc["Yes"].values if "Yes" in ctab.index else np.zeros(len(OUTCOME_ORDER))
    ind_no = ctab.loc["No"].values if "No" in ctab.index else np.zeros(len(OUTCOME_ORDER))

    bars1 = ax.bar(x - width/2, ind_yes, width, label="Industry-Involved",
                   color=COLORS["industry"], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, ind_no, width, label="Independent",
                   color=COLORS["independent"], alpha=0.85, edgecolor="white")

    ax.set_xlabel("Outcome Direction", fontsize=12)
    ax.set_ylabel("Number of Papers", fontsize=12)
    ax.set_title("Paper Outcomes: Industry-Involved vs Independent", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(OUTCOME_ORDER, fontsize=11)
    ax.legend(fontsize=11)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outcome_by_industry.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_outcome_proportions(papers_df: pd.DataFrame, output_dir: str):
    """Stacked bar chart showing proportions instead of counts."""
    ctab = pd.crosstab(papers_df["industry_involved"], papers_df["outcome"], normalize="index") * 100
    for col in OUTCOME_ORDER:
        if col not in ctab.columns:
            ctab[col] = 0
    ctab = ctab[OUTCOME_ORDER]

    fig, ax = plt.subplots(figsize=(10, 6))
    ctab.plot(kind="barh", stacked=True, ax=ax,
              color=[COLORS.get(o, "#999") for o in OUTCOME_ORDER], edgecolor="white")

    ax.set_xlabel("Percentage of Papers (%)", fontsize=12)
    ax.set_ylabel("")
    ax.set_yticklabels(["Industry-Involved" if v == "Yes" else "Independent" for v in ctab.index], fontsize=12)
    ax.set_title("Outcome Distribution by Industry Involvement", fontsize=14, fontweight="bold")
    ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outcome_proportions.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_centrality_distribution(centrality_df: pd.DataFrame, output_dir: str):
    """Box plot comparing centrality metrics between industry and independent authors."""
    if centrality_df.empty:
        return

    metrics = ["degree_centrality", "betweenness_centrality", "eigenvector_centrality"]
    available = [m for m in metrics if m in centrality_df.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 6))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        data_ind = centrality_df[centrality_df["is_industry"] == True][metric].dropna()
        data_noind = centrality_df[centrality_df["is_industry"] == False][metric].dropna()

        bp = ax.boxplot(
            [data_ind, data_noind],
            labels=["Industry", "Independent"],
            patch_artist=True,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor(COLORS["industry"])
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(COLORS["independent"])
        bp["boxes"][1].set_alpha(0.7)

        label = metric.replace("_", " ").title()
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel(label, fontsize=10)

    fig.suptitle("Network Centrality: Industry vs Independent Authors", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "centrality_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_community_scatter(comm_df: pd.DataFrame, output_dir: str):
    """Scatter plot: community % industry vs % positive outcomes."""
    if comm_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    sizes = comm_df["n_authors"] * 5
    sizes = sizes.clip(lower=20)

    scatter = ax.scatter(
        comm_df["pct_industry"],
        comm_df["pct_positive"],
        s=sizes,
        c=comm_df["pct_industry"],
        cmap="RdYlBu_r",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add labels for larger communities
    for _, row in comm_df.iterrows():
        if row["n_authors"] >= 5:
            ax.annotate(
                f"C{row['community_id']} (n={row['n_authors']})",
                (row["pct_industry"], row["pct_positive"]),
                fontsize=8, ha="center", va="bottom",
            )

    ax.set_xlabel("% Industry-Affiliated Authors in Community", fontsize=12)
    ax.set_ylabel("% Positive Outcomes in Community", fontsize=12)
    fig.suptitle("Community-Level: Industry Concentration vs Positive Outcomes", fontsize=14, fontweight="bold", y=0.98)
    ax.set_title(
        "Each bubble is a Louvain community of co-authors. Size = number of authors.\n"
        "X-axis shows how concentrated industry affiliations are; Y-axis shows the share of positive findings.",
        fontsize=9.5, color="#555", fontstyle="italic", pad=12,
    )

    plt.colorbar(scatter, ax=ax, label="% Industry", shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "community_industry_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_timeline(papers_df: pd.DataFrame, output_dir: str):
    """Timeline: number of industry vs independent papers per year."""
    if "year" not in papers_df.columns:
        return

    df = papers_df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 1960) & (df["year"] <= 2026)]

    yearly = df.groupby(["year", "industry_involved"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 6))

    if "Yes" in yearly.columns:
        ax.fill_between(yearly.index, yearly["Yes"], alpha=0.3, color=COLORS["industry"])
        ax.plot(yearly.index, yearly["Yes"], color=COLORS["industry"], linewidth=2, label="Industry-Involved")
    if "No" in yearly.columns:
        ax.fill_between(yearly.index, yearly["No"], alpha=0.3, color=COLORS["independent"])
        ax.plot(yearly.index, yearly["No"], color=COLORS["independent"], linewidth=2, label="Independent")

    ax.set_xlabel("Publication Year", fontsize=12)
    ax.set_ylabel("Number of Papers", fontsize=12)
    ax.set_title("Publication Timeline: Industry vs Independent Papers", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timeline_industry_papers.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_odds_ratio_forest(stats_path: str, output_dir: str):
    """Forest plot showing odds ratio with 95% CI."""
    if not os.path.exists(stats_path):
        return

    with open(stats_path) as f:
        stats = json.load(f)

    or_data = stats.get("odds_ratio_positive", {})
    or_val = or_data.get("odds_ratio")
    ci_lo = or_data.get("ci_95_lower")
    ci_hi = or_data.get("ci_95_upper")

    if or_val is None:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.errorbar(
        or_val, 0, xerr=[[or_val - ci_lo], [ci_hi - or_val]],
        fmt="o", color=COLORS["industry"], markersize=12, capsize=6, linewidth=2.5,
    )

    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_yticks([0])
    ax.set_yticklabels(["Industry vs\nIndependent"], fontsize=12)
    ax.set_xlabel("Odds Ratio (Positive Outcome)", fontsize=12)
    ax.set_title(f"Odds Ratio: {or_val:.2f} (95% CI: {ci_lo:.2f} - {ci_hi:.2f})",
                 fontsize=13, fontweight="bold")

    # Reference labels on either side of the null line (below the axis)
    ax.text(0.82, -0.45, "Favors independent", ha="center", fontsize=10,
            color=COLORS["independent"], fontstyle="italic")
    ax.text(1.18, -0.45, "Favors industry", ha="center", fontsize=10,
            color=COLORS["industry"], fontstyle="italic")

    # Arrow indicators
    ax.annotate("", xy=(0.72, -0.3), xytext=(0.95, -0.3),
                arrowprops=dict(arrowstyle="->", color=COLORS["independent"], lw=1.5))
    ax.annotate("", xy=(1.28, -0.3), xytext=(1.05, -0.3),
                arrowprops=dict(arrowstyle="->", color=COLORS["industry"], lw=1.5))

    ax.set_ylim(-0.7, 0.5)

    # Color legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["industry"], edgecolor="black", label="Industry-Involved (OR point & 95% CI)"),
        mpatches.Patch(facecolor=COLORS["independent"], edgecolor="black", label="Independent (reference group)"),
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="Null (OR = 1.0, no difference)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "odds_ratio_forest.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_sankey_interactive(papers_df: pd.DataFrame, output_dir: str):
    """Interactive Sankey diagram: Industry Status -> Outcome."""
    if not HAS_PLOTLY:
        print("[WARN] plotly not installed, skipping Sankey diagram")
        return

    # Nodes: Industry-Involved, Independent, Positive, Negative, Neutral, Mixed, Not coded
    node_labels = ["Industry-Involved", "Independent"] + OUTCOME_ORDER
    node_colors = [
        COLORS["industry"], COLORS["independent"],
        COLORS["Positive"], COLORS["Negative"], COLORS["Neutral"],
        COLORS["Mixed"], COLORS["Not coded"],
    ]

    # Links
    sources = []
    targets = []
    values = []
    link_colors = []

    for i, group in enumerate(["Yes", "No"]):
        subset = papers_df[papers_df["industry_involved"] == group]
        for j, outcome in enumerate(OUTCOME_ORDER):
            count = (subset["outcome"] == outcome).sum()
            if count > 0:
                sources.append(i)
                targets.append(2 + j)
                values.append(count)
                link_colors.append(node_colors[i].replace(")", ", 0.4)").replace("rgb", "rgba")
                                   if "rgb" in node_colors[i] else node_colors[i])

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        ),
    ))

    fig.update_layout(
        title_text="Flow: Industry Involvement to Paper Outcomes",
        font_size=12,
        width=900,
        height=500,
    )

    fig.write_html(os.path.join(output_dir, "sankey_funding_outcome.html"))


def plot_outcome_heatmap_interactive(papers_df: pd.DataFrame, output_dir: str):
    """Interactive heatmap: outcome by year and industry status."""
    if not HAS_PLOTLY:
        return

    df = papers_df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 1990) & (df["year"] <= 2026)]

    # Compute % positive per year per group
    rows = []
    for year in sorted(df["year"].unique()):
        for group in ["Yes", "No"]:
            subset = df[(df["year"] == year) & (df["industry_involved"] == group)]
            total = len(subset)
            n_pos = (subset["outcome"] == "Positive").sum()
            pct = (n_pos / total * 100) if total > 0 else 0
            label = "Industry" if group == "Yes" else "Independent"
            rows.append({"year": year, "group": label, "pct_positive": round(pct, 1), "n": total})

    plot_df = pd.DataFrame(rows)
    pivot = plot_df.pivot(index="group", columns="year", values="pct_positive").fillna(0)

    fig = px.imshow(
        pivot.values,
        labels=dict(x="Year", y="Group", color="% Positive"),
        x=list(pivot.columns),
        y=list(pivot.index),
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig.update_layout(
        title="% Positive Outcomes Over Time: Industry vs Independent",
        width=1200,
        height=300,
    )
    fig.write_html(os.path.join(output_dir, "outcome_heatmap.html"))


def main():
    ap = argparse.ArgumentParser(description="Generate visualizations")
    ap.add_argument("--data_dir", required=True, help="Dir with papers.csv, authors.csv")
    ap.add_argument("--network_dir", required=True, help="Dir with centrality/community CSVs")
    ap.add_argument("--stats_dir", required=True, help="Dir with full_statistics.json")
    ap.add_argument("--output_dir", required=True, help="Where to write figures/")
    args = ap.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    papers_df = pd.read_csv(os.path.join(args.data_dir, "papers.csv"))

    centrality_path = os.path.join(args.network_dir, "top_authors_centrality.csv")
    centrality_df = pd.read_csv(centrality_path) if os.path.exists(centrality_path) else pd.DataFrame()

    comm_path = os.path.join(args.network_dir, "community_summary.csv")
    comm_df = pd.read_csv(comm_path) if os.path.exists(comm_path) else pd.DataFrame()

    stats_path = os.path.join(args.stats_dir, "full_statistics.json")

    print("Generating static plots...")
    plot_outcome_by_industry(papers_df, fig_dir)
    plot_outcome_proportions(papers_df, fig_dir)
    plot_centrality_distribution(centrality_df, fig_dir)
    plot_community_scatter(comm_df, fig_dir)
    plot_timeline(papers_df, fig_dir)
    plot_odds_ratio_forest(stats_path, fig_dir)

    print("Generating interactive plots...")
    plot_sankey_interactive(papers_df, fig_dir)
    plot_outcome_heatmap_interactive(papers_df, fig_dir)

    print(f"All figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
